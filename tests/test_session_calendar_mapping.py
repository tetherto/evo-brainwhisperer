"""
Regression tests: session calendar-day and month mapping in BWEmbedder.

Two related bugs once lived in BWEmbedder:
  a) _add_day_encoding used the session index (0-44 for CARD) as the sinusoidal
     PE lookup position instead of calendar days (0-611), so sessions spaced
     months apart produced nearly identical encodings.
  b) session_to_month fell back to session_idx // 4 rather than
     calendar_day // 30, splitting same-month sessions across month buckets
     and merging different-month sessions together.

The fix: each recipe's calendar_days() gives days-since-first-session from the
neuralshed session rules, and BrainWhisperer threads them through to each
BWEmbedder. Tests that read a recipe skip when its dataset metadata is not
available on this machine.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from brainwhisperer.config.schema import BWEmbedderConfig
from brainwhisperer.data.recipes import RECIPES
from brainwhisperer.models.embedder import BWEmbedder

#: Day-axis size of each dataset's embedder in the published model.
PUBLISHED_NUM_DAYS = {
    "card": 45,
    "willet": 24,
    "kunz_t12": 1,
    "kunz_t15": 3,
    "wairagkar": 10,
    "jude": 5,
}


def _recipe_value(name: str, read):
    try:
        return read(RECIPES[name])
    except Exception as exc:  # dataset metadata not available here
        pytest.skip(f"{name} day rule unavailable: {exc}")


@pytest.fixture(scope="module")
def card_days() -> list[int]:
    return _recipe_value("card", lambda r: r.calendar_days())


# ─── recipe calendar days ────────────────────────────────────────────────────


def test_card_first_session_is_day_zero(card_days):
    assert card_days[0] == 0


def test_card_known_calendar_days(card_days):
    """Spot-check a handful of CARD sessions against hand-computed day counts."""
    # t15.2023.08.11 → t15.2023.08.13: 2 days
    assert card_days[1] == 2
    # t15.2023.08.11 → t15.2023.09.03: 23 days
    assert card_days[7] == 23
    # t15.2023.08.11 → t15.2023.12.29: 140 days
    assert card_days[26] == 140
    # t15.2023.08.11 → t15.2024.02.25: 198 days
    assert card_days[27] == 198
    # t15.2023.08.11 → t15.2025.04.13: 611 days
    assert card_days[44] == 611


def test_kunz_t12_single_date_gives_day_zero():
    assert _recipe_value("kunz_t12", lambda r: r.calendar_days()) == [0]


def test_recipes_cover_all_datasets():
    assert set(RECIPES) == set(PUBLISHED_NUM_DAYS)


@pytest.mark.parametrize("name", sorted(PUBLISHED_NUM_DAYS))
def test_recipe_num_days_match_published_model(name):
    """NUM_DAYS sizes the per-day parameters, so it must match the model."""
    assert _recipe_value(name, lambda r: r.NUM_DAYS) == PUBLISHED_NUM_DAYS[name]


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


def test_session_to_calendar_day_buffer_registered(card_days):
    """When session_to_calendar_day is provided the buffer exists and is a tensor."""
    emb = BWEmbedder(_sin_cfg(), num_days=45, session_to_calendar_day=card_days)
    assert isinstance(emb.session_to_calendar_day, torch.Tensor)
    assert emb.session_to_calendar_day.shape == (45,)
    assert emb.session_to_calendar_day[26].item() == 140
    assert emb.session_to_calendar_day[27].item() == 198
    assert emb.session_to_calendar_day[44].item() == 611


def test_no_calendar_buffer_gives_none():
    """Without session_to_calendar_day the buffer is None (fallback: session index)."""
    emb = BWEmbedder(_sin_cfg(), num_days=45)
    assert emb.session_to_calendar_day is None


def test_sinusoidal_looks_up_calendar_day_not_session_index(card_days):
    """Encoding for session 26 must equal direct lookup at calendar day 140."""
    emb = BWEmbedder(_sin_cfg(), num_days=45, session_to_calendar_day=card_days)

    # Buffer lookup
    session_idx = torch.tensor([26])
    via_buffer = emb.day_encoding(emb.session_to_calendar_day[session_idx])

    # Direct lookup at calendar position
    via_direct = emb.day_encoding(torch.tensor([140]))

    assert torch.allclose(via_buffer, via_direct)


def test_sinusoidal_calendar_day_differs_from_session_index(card_days):
    """Session 26 maps to calendar day 140; encoding at 140 ≠ encoding at 26."""
    emb = BWEmbedder(_sin_cfg(), num_days=45, session_to_calendar_day=card_days)

    enc_at_26 = emb.day_encoding(torch.tensor([26]))
    enc_at_140 = emb.day_encoding(torch.tensor([140]))
    assert not torch.allclose(enc_at_26, enc_at_140), (
        "Sinusoidal encoding at session index 26 must differ from calendar day 140"
    )


def test_add_day_encoding_with_buffer_uses_calendar_position(card_days):
    """Full _add_day_encoding call routes through calendar_day buffer."""
    emb_fixed = BWEmbedder(_sin_cfg(), num_days=45, session_to_calendar_day=card_days)
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


def test_calendar_months_group_aug_sessions_together(card_days):
    """CARD sessions 0-7 are all in Aug 2023 → calendar month 0 for all."""
    months = [d // 30 for d in card_days]

    cfg = _sin_cfg(day_projections=True)
    emb = BWEmbedder(
        cfg, num_days=45, session_to_calendar_day=card_days, session_to_month=months
    )

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


def test_calendar_months_sessions_26_27_in_distinct_buckets(card_days):
    """Session 26 (day 140 → month 4) and session 27 (day 198 → month 6) differ."""
    months = [d // 30 for d in card_days]

    cfg = _sin_cfg(day_projections=True)
    emb = BWEmbedder(
        cfg, num_days=45, session_to_calendar_day=card_days, session_to_month=months
    )

    assert emb.session_to_month[26].item() == 4  # day 140 ÷ 30 = 4
    assert emb.session_to_month[27].item() == 6  # day 198 ÷ 30 = 6


# ─── Kunz T15 calendar days ──────────────────────────────────────────────────
# Kunz T15's three sessions are recorded on 2024-03-03, 2024-12-15 and
# 2025-01-05. The published T15 embedder treated them as one calendar day
# ([0, 0, 0]). The shipped T15 config never reads calendar days (see the
# config test below), so the true dates change no parameter.

T15_CALENDAR_DAYS = [0, 287, 308]
T15_MONTHS = [d // 30 for d in T15_CALENDAR_DAYS]


def test_kunz_t15_recipe_derives_true_days():
    assert _recipe_value("kunz_t15", lambda r: r.calendar_days()) == T15_CALENDAR_DAYS


def test_kunz_t15_shipped_config_ignores_calendar_days():
    """T15 ships `encodings: learnable` + `day_proj_rank: 0`.

    BrainWhisperer passes session_to_calendar_day only when encodings ==
    "sin", and BWEmbedder reads session_to_month only when day_proj_rank != 0.
    So the true calendar days give the same parameters and shapes as the
    published [0, 0, 0], and T15 numbers stay comparable with the paper.
    """
    root = Path(__file__).resolve().parent.parent
    base = OmegaConf.load(root / "brainwhisperer/config/base.yaml")
    exp = OmegaConf.load(
        root / "brainwhisperer/config/experiment/cross_dataset_full.yaml"
    )
    override = exp.data.embedder_overrides["kunz_t15"]
    emb_cfg = OmegaConf.merge(base.model.embedder, override)

    # If either fact changes, calendar days start shaping T15's parameters.
    assert emb_cfg.encodings == "learnable"
    assert emb_cfg.day_proj_rank == 0

    def build(cal):
        return BWEmbedder(
            emb_cfg,
            num_days=len(cal),
            session_to_month=[d // 30 for d in cal],
            session_to_calendar_day=(
                cal if emb_cfg.encodings == "sin" else None
            ),
        )

    published_m, derived_m = build([0, 0, 0]), build(T15_CALENDAR_DAYS)
    published, derived = published_m.state_dict(), derived_m.state_dict()

    assert set(published) == set(derived)
    assert all(published[k].shape == derived[k].shape for k in published)
    assert sum(p.numel() for p in published_m.parameters()) == sum(
        p.numel() for p in derived_m.parameters()
    )
    # learnable -> no calendar-day buffer at all (a None buffer is absent from
    # state_dict), so calendar days never reach the day embedding, which has
    # only num_days rows.
    assert derived_m.session_to_calendar_day is None
    # rank 0 -> month projections are never created; only the unread
    # session_to_month buffer's VALUES differ.
    assert not [k for k in derived if k.startswith("proj_") and "_month." in k]
    assert published["session_to_month"].tolist() == [0, 0, 0]
    assert derived["session_to_month"].tolist() == T15_MONTHS


def test_kunz_t15_month_sizing_with_day_projections():
    """With day projections on, the true dates would size 11 month slots.

    Documents the coupling so that raising T15's day_proj_rank is a deliberate,
    visible decision.
    """
    assert T15_MONTHS == [0, 9, 10]
    emb = BWEmbedder(
        _sin_cfg(day_projections=True),  # rank 4 + "sin": the hypothetical
        num_days=len(T15_CALENDAR_DAYS),
        session_to_calendar_day=T15_CALENDAR_DAYS,
        session_to_month=T15_MONTHS,
    )
    assert emb.session_to_month.tolist() == [0, 9, 10]
    # num_months = max(month) + 1 sizes these ParameterLists.
    assert len(emb.proj_W_month) == 11
    assert len(emb.proj_b_month) == 11
    # Per-day params follow NUM_DAYS, which the dates do NOT change.
    assert len(emb.proj_A) == len(T15_CALENDAR_DAYS) == 3
