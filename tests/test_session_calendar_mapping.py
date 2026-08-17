"""
Regression tests — session calendar-day and month mapping (Bug #2).

The refactored code had two related errors in BWEmbedder:
  a) _add_day_encoding used session index (0-44 for CARD) as the sinusoidal
     PE lookup position instead of calendar days (0-611), so sessions spaced
     months apart produced nearly identical encodings.
  b) session_to_month fell back to session_idx // 4 rather than
     calendar_day // 30, incorrectly splitting same-month sessions across
     different month buckets and merging different-month sessions together.

The fix: sessions_to_calendar_days() converts acquisition-date strings to
days-since-first-session, SESSIONS_BY_DATASET maps dataset name → session list,
and BrainWhisperer now threads both through to each BWEmbedder.
"""
import torch
import pytest

from brainwhisperer.config.schema import BWEmbedderConfig
from brainwhisperer.models.embedder import BWEmbedder
from brainwhisperer.utils.sessions import (
    CARD_SESSIONS,
    KUNZ_T12_SESSIONS,
    KUNZ_T15_SESSIONS,
    SESSIONS_BY_DATASET,
    WILLETT_SESSIONS,
    sessions_to_calendar_days,
)


# ─── sessions_to_calendar_days ────────────────────────────────────────────────


def test_card_first_session_is_day_zero():
    days = sessions_to_calendar_days(CARD_SESSIONS)
    assert days[0] == 0


def test_card_known_calendar_days():
    """Spot-check a handful of CARD sessions against hand-computed day counts."""
    days = sessions_to_calendar_days(CARD_SESSIONS)
    # t15.2023.08.11 → t15.2023.08.13: 2 days
    assert days[1] == 2
    # t15.2023.08.11 → t15.2023.09.03: 23 days
    assert days[7] == 23
    # t15.2023.08.11 → t15.2023.12.29: 140 days
    assert days[26] == 140
    # t15.2023.08.11 → t15.2024.02.25: 198 days
    assert days[27] == 198
    # t15.2023.08.11 → t15.2025.04.13: 611 days
    assert days[44] == 611


def test_kunz_t12_same_date_gives_zero_days():
    """All three Kunz T12 sessions are on the same date → all day 0."""
    days = sessions_to_calendar_days(KUNZ_T12_SESSIONS)
    assert days == [0, 0, 0]


def test_kunz_t15_three_distinct_dates():
    """Kunz T15 has three distinct dates; spot-check the two later ones."""
    days = sessions_to_calendar_days(KUNZ_T15_SESSIONS)
    assert days[0] == days[1] == days[2] == 0    # 2024.03.03 — first date
    assert days[3] == days[4] == 287             # 2024.12.15 — 287 days later
    assert days[5] == days[6] == 308             # 2025.01.05 — 308 days later


def test_sessions_by_dataset_covers_all_four_datasets():
    assert set(SESSIONS_BY_DATASET.keys()) == {"card", "willet", "kunz_t12", "kunz_t15"}


def test_sessions_by_dataset_lengths():
    """SESSIONS_BY_DATASET lengths must match dataset NUM_DAYS constants."""
    assert len(SESSIONS_BY_DATASET["card"]) == 45
    assert len(SESSIONS_BY_DATASET["willet"]) == 24
    assert len(SESSIONS_BY_DATASET["kunz_t12"]) == 3
    assert len(SESSIONS_BY_DATASET["kunz_t15"]) == 7


# ─── BWEmbedder sinusoidal lookup ─────────────────────────────────────────────


def _sin_cfg(day_projections: bool = False) -> BWEmbedderConfig:
    return BWEmbedderConfig(
        num_features=16,
        d_model=64,
        embed_type="conv",
        kernel_size_1=3,
        stride_2=2,
        patch_size=5,
        encodings="sin",
        day_projections=day_projections,
        day_proj_rank=4 if day_projections else 0,
        dropout=0.0,
        day_layer_dropout=0.0,
        session_dropout=0.0,
    )


def test_session_to_calendar_day_buffer_registered():
    """When session_to_calendar_day is provided the buffer exists and is a tensor."""
    cal_days = sessions_to_calendar_days(CARD_SESSIONS)
    emb = BWEmbedder(_sin_cfg(), num_days=45, session_to_calendar_day=cal_days)
    assert isinstance(emb.session_to_calendar_day, torch.Tensor)
    assert emb.session_to_calendar_day.shape == (45,)
    assert emb.session_to_calendar_day[26].item() == 140
    assert emb.session_to_calendar_day[27].item() == 198
    assert emb.session_to_calendar_day[44].item() == 611


def test_no_calendar_buffer_gives_none():
    """Without session_to_calendar_day the buffer is None (fallback: session index)."""
    emb = BWEmbedder(_sin_cfg(), num_days=45)
    assert emb.session_to_calendar_day is None


def test_sinusoidal_looks_up_calendar_day_not_session_index():
    """Encoding for session 26 must equal direct lookup at calendar day 140."""
    cal_days = sessions_to_calendar_days(CARD_SESSIONS)
    emb = BWEmbedder(_sin_cfg(), num_days=45, session_to_calendar_day=cal_days)

    # Buffer lookup
    session_idx = torch.tensor([26])
    via_buffer = emb.day_encoding(emb.session_to_calendar_day[session_idx])

    # Direct lookup at calendar position
    via_direct = emb.day_encoding(torch.tensor([140]))

    assert torch.allclose(via_buffer, via_direct)


def test_sinusoidal_calendar_day_differs_from_session_index():
    """Session 26 maps to calendar day 140; encoding at 140 ≠ encoding at 26."""
    cal_days = sessions_to_calendar_days(CARD_SESSIONS)
    emb = BWEmbedder(_sin_cfg(), num_days=45, session_to_calendar_day=cal_days)

    enc_at_26 = emb.day_encoding(torch.tensor([26]))
    enc_at_140 = emb.day_encoding(torch.tensor([140]))
    assert not torch.allclose(enc_at_26, enc_at_140), (
        "Sinusoidal encoding at session index 26 must differ from calendar day 140"
    )


def test_add_day_encoding_with_buffer_uses_calendar_position():
    """Full _add_day_encoding call routes through calendar_day buffer."""
    cal_days = sessions_to_calendar_days(CARD_SESSIONS)
    emb_fixed = BWEmbedder(_sin_cfg(), num_days=45, session_to_calendar_day=cal_days)
    emb_broken = BWEmbedder(_sin_cfg(), num_days=45)  # no calendar info

    # Copy all shared weights so the only difference is the lookup position.
    with torch.no_grad():
        emb_broken.conv1.weight.copy_(emb_fixed.conv1.weight)
        emb_broken.conv1.bias.copy_(emb_fixed.conv1.bias)
        emb_broken.conv2.weight.copy_(emb_fixed.conv2.weight)
        emb_broken.conv2.bias.copy_(emb_fixed.conv2.bias)
        # PositionalEncoding has no learned params (it's a buffer), so pe is identical.

    # Session 26 has calendar day 140; if broken, looks up position 26 instead.
    day_idx = torch.tensor([26, 27])  # calendar days: [140, 198]
    signals = torch.randn(2, 16, 3000)

    emb_fixed.eval()
    emb_broken.eval()
    with torch.no_grad():
        out_fixed = emb_fixed(signals, day_idx)
        out_broken = emb_broken(signals, day_idx)

    # The outputs differ because the sinusoidal encoding differs.
    assert not torch.allclose(out_fixed, out_broken), (
        "Fixed (calendar-day) and broken (session-index) embeddings must differ"
    )


# ─── Session-to-month grouping ────────────────────────────────────────────────


def test_calendar_months_group_aug_sessions_together():
    """CARD sessions 0-7 are all in Aug 2023 → calendar month 0 for all."""
    cal_days = sessions_to_calendar_days(CARD_SESSIONS)
    months = [d // 30 for d in cal_days]

    cfg = _sin_cfg(day_projections=True)
    emb = BWEmbedder(cfg, num_days=45,
                     session_to_calendar_day=cal_days,
                     session_to_month=months)

    month_buckets = emb.session_to_month[:8].tolist()
    assert all(m == 0 for m in month_buckets), (
        f"First 8 CARD sessions (all Aug 2023) must all be in month 0, got {month_buckets}"
    )


def test_heuristic_incorrectly_splits_aug_sessions():
    """The 4-per-month heuristic puts sessions 0-3 in month 0 and 4-7 in month 1."""
    cfg = _sin_cfg(day_projections=True)
    emb_broken = BWEmbedder(cfg, num_days=45)  # no calendar info → 4-per-month

    assert emb_broken.session_to_month[0].item() == 0
    assert emb_broken.session_to_month[7].item() == 1  # wrong: should be 0


def test_calendar_months_sessions_26_27_in_distinct_buckets():
    """Session 26 (day 140 → month 4) and session 27 (day 198 → month 6) differ."""
    cal_days = sessions_to_calendar_days(CARD_SESSIONS)
    months = [d // 30 for d in cal_days]

    cfg = _sin_cfg(day_projections=True)
    emb = BWEmbedder(cfg, num_days=45, session_to_calendar_day=cal_days,
                     session_to_month=months)

    assert emb.session_to_month[26].item() == 4  # day 140 ÷ 30 = 4
    assert emb.session_to_month[27].item() == 6  # day 198 ÷ 30 = 6


def test_kunz_t15_calendar_month_grouping():
    """Sessions 3-4 (day 287 → month 9) must not share month bucket with sessions 0-2 (day 0)."""
    cal_days = sessions_to_calendar_days(KUNZ_T15_SESSIONS)
    months = [d // 30 for d in cal_days]

    cfg = _sin_cfg(day_projections=True)
    emb = BWEmbedder(cfg, num_days=7, session_to_calendar_day=cal_days,
                     session_to_month=months)

    assert emb.session_to_month[0].item() == 0   # day 0 → month 0
    assert emb.session_to_month[3].item() == 9   # day 287 → month 9
    assert emb.session_to_month[5].item() == 10  # day 308 → month 10


def test_kunz_t15_heuristic_incorrectly_merges_months():
    """The 4-per-month heuristic puts sessions 0-3 all in month 0, merging day-0 with day-287."""
    cfg = _sin_cfg(day_projections=True)
    emb_broken = BWEmbedder(cfg, num_days=7)  # no calendar info

    # Sessions 0 and 3 are 287 days apart but the heuristic puts them both in month 0.
    assert emb_broken.session_to_month[0].item() == 0
    assert emb_broken.session_to_month[3].item() == 0  # wrong: should be 9
