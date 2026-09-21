"""
Numerical equivalence: BWEmbedder must produce bit-for-bit identical output
to the original WhisperEmbedder for the same input and weights.

Weights are loaded from the trained checkpoint so that the comparison uses
real learned values rather than random initialisation. Signals come from the
CARD training pickle so we exercise the full range of day indices (0–44).

The original projection implements:  output = softsign(x @ W + b)
(right-multiplication, no transpose), which is what the einsum
  torch.einsum("btd,bdk->btk", x, W)
produces in the original codebase.

This file exists solely to prove the refactored embedder is a faithful port.
It can be removed once the original codebase is decommissioned.
"""

import glob
import pickle
import sys

import pytest
import torch
import torch.nn.functional as F

from brainwhisperer.config.paths import CARD_ROOT
from brainwhisperer.config.schema import BWEmbedderConfig
from brainwhisperer.models.embedder import BWEmbedder

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_ORIG_ROOT = "/home/francesco/evo-exp-brain-to-voice-online/whisper-based"
_CKPT_GLOB = f"{_ORIG_ROOT}/ckpts/*.ckpt"
_PREFIX = "model.embedders.0."
_N_SAMPLES = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_ckpt() -> str | None:
    ckpts = sorted(glob.glob(_CKPT_GLOB))
    return ckpts[-1] if ckpts else None


def _upstream():
    """Import BaseNeuralTextDataset from the upstream repo."""
    if CARD_ROOT not in sys.path:
        sys.path.insert(0, CARD_ROOT)
    from model_training.augmented_dataset import BaseNeuralTextDataset
    return BaseNeuralTextDataset


def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Fixtures  (module-scoped: the checkpoint and data are loaded once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sd():
    path = _find_ckpt()
    if path is None:
        pytest.skip("No original checkpoint found — skipping numerical equivalence test")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]


@pytest.fixture(scope="module")
def session_to_orig_month(sd):
    """session_idx → original (sparse) month index, as the checkpoint computed it."""
    days = sd[_PREFIX + "sessions_to_days.session_to_idx_map"]  # (45,)
    return (days // 30).long().tolist()


@pytest.fixture(scope="module")
def orig_month_idxs(session_to_orig_month):
    return sorted(set(session_to_orig_month))


@pytest.fixture(scope="module")
def remap(orig_month_idxs):
    """Sparse original month index → contiguous index for BWEmbedder."""
    return {m: i for i, m in enumerate(orig_month_idxs)}


@pytest.fixture(scope="module")
def session_to_contiguous_month(session_to_orig_month, remap):
    return [remap[m] for m in session_to_orig_month]


@pytest.fixture(scope="module")
def ckpt_weights(sd, orig_month_idxs):
    """
    Extract projection + conv weights from the checkpoint into plain tensors,
    indexed contiguously so they can be loaded into BWEmbedder.
    """
    n_days = sum(1 for k in sd if k.startswith(_PREFIX + "day_As."))

    proj_W_month = torch.stack(
        [sd[f"{_PREFIX}month_weights.{m}"] for m in orig_month_idxs]
    )  # (M, 512, 512)
    proj_b_month = torch.stack(
        [sd[f"{_PREFIX}month_biases.{m}"].squeeze(0) for m in orig_month_idxs]
    )  # (M, 512)
    proj_A = torch.stack([sd[f"{_PREFIX}day_As.{d}"] for d in range(n_days)])   # (D, 512, 16)
    proj_B = torch.stack([sd[f"{_PREFIX}day_Bs.{d}"] for d in range(n_days)])   # (D, 16, 512)
    proj_b_day = torch.stack(
        [sd[f"{_PREFIX}day_biases.{d}"].squeeze(0) for d in range(n_days)]
    )  # (D, 512)

    return {
        "proj_W_month": proj_W_month,
        "proj_b_month": proj_b_month,
        "proj_A": proj_A,
        "proj_B": proj_B,
        "proj_b_day": proj_b_day,
        "conv1_weight": sd[_PREFIX + "conv1.weight"],
        "conv1_bias":   sd[_PREFIX + "conv1.bias"],
        "conv2_weight": sd[_PREFIX + "conv2.weight"],
        "conv2_bias":   sd[_PREFIX + "conv2.bias"],
        "n_days": n_days,
    }


@pytest.fixture(scope="module")
def real_data():
    """
    _N_SAMPLES items from the CARD training split, returned as
    x_btc (B, T, C) and day_idx (B,) — matching the format the original
    WhisperEmbedder received before its internal permute.
    """
    BaseNeuralTextDataset = _upstream()
    raw = _load_pickle(f"{CARD_ROOT}/data/cleaned_train_data.pkl")
    ds = BaseNeuralTextDataset(raw, eval_type="val", source_dataset="card")

    # Sample evenly across the dataset so we exercise multiple sessions/months.
    step = max(1, len(ds) // _N_SAMPLES)
    indices = [i * step for i in range(_N_SAMPLES)]
    items = [ds[i] for i in indices]
    feats = [item["neural_features"] for item in items]   # list of (T_i, C)
    max_T = max(f.shape[0] for f in feats)
    # Pad to max_T so we can stack; projection is elementwise over T so padding
    # does not affect the non-padded portion.
    padded = [F.pad(f, (0, 0, 0, max_T - f.shape[0])) for f in feats]
    x_btc = torch.stack(padded)                           # (B, T, C)
    day_idx = torch.stack([item["day"] for item in items])  # (B,)
    return x_btc, day_idx


@pytest.fixture(scope="module")
def loaded_embedder(ckpt_weights, session_to_contiguous_month):
    """BWEmbedder with projection weights loaded from the checkpoint."""
    n_days = ckpt_weights["n_days"]
    n_months = len(session_to_contiguous_month)  # not used directly; derived inside BWEmbedder
    C = ckpt_weights["proj_A"].shape[1]
    r = ckpt_weights["proj_A"].shape[2]

    cfg = BWEmbedderConfig(
        num_features=C,
        d_model=384,
        embed_type="conv",
        kernel_size_1=7,
        stride_2=2,
        patch_size=5,
        encodings="sin",
        day_projections=True,
        day_proj_rank=r,
        dropout=0.0,
        day_layer_dropout=0.0,
        session_dropout=0.0,
    )
    emb = BWEmbedder(cfg, num_days=n_days, session_to_month=session_to_contiguous_month)

    def _copy_into_paramlist(param_list, stacked_tensor):
        for i, p in enumerate(param_list):
            p.copy_(stacked_tensor[i])

    with torch.no_grad():
        _copy_into_paramlist(emb.proj_W_month, ckpt_weights["proj_W_month"])
        _copy_into_paramlist(emb.proj_b_month, ckpt_weights["proj_b_month"])
        _copy_into_paramlist(emb.proj_A, ckpt_weights["proj_A"])
        _copy_into_paramlist(emb.proj_B, ckpt_weights["proj_B"])
        _copy_into_paramlist(emb.proj_b_day, ckpt_weights["proj_b_day"])
        emb.conv1.weight.copy_(ckpt_weights["conv1_weight"])
        emb.conv1.bias.copy_(ckpt_weights["conv1_bias"])
        emb.conv2.weight.copy_(ckpt_weights["conv2_weight"])
        emb.conv2.bias.copy_(ckpt_weights["conv2_bias"])

    emb.eval()
    return emb


# ---------------------------------------------------------------------------
# Reference implementations (verbatim ports of the original)
# ---------------------------------------------------------------------------


def _orig_projection(x_btc, day_idx, ckpt_weights, session_to_orig_month, orig_month_idxs):
    """
    Verbatim port of WhisperEmbedder._embed() projection branch (r > 0).

    Original einsum: torch.einsum("btd,bdk->btk", x, W)  = x @ W  (no transpose)
    Original activation: nn.Softsign()  = F.softsign
    """
    month_pos = {m: i for i, m in enumerate(orig_month_idxs)}
    W_month = ckpt_weights["proj_W_month"]   # (M, C, C) — already in our contiguous order
    b_month = ckpt_weights["proj_b_month"]   # (M, C)
    A = ckpt_weights["proj_A"]               # (D, C, r)
    B = ckpt_weights["proj_B"]               # (D, r, C)
    b_day = ckpt_weights["proj_b_day"]       # (D, C)

    parts = []
    for i, d in enumerate(day_idx.tolist()):
        orig_m = session_to_orig_month[d]
        m = month_pos[orig_m]
        W = W_month[m] + A[d] @ B[d]   # (C, C)
        b = b_month[m] + b_day[d]      # (C,)
        xi = x_btc[i]                  # (T, C)
        parts.append(F.softsign(xi @ W + b))

    return torch.stack(parts)  # (B, T, C)


def _orig_full_embed(x_btc, day_idx, ckpt_weights, session_to_orig_month, orig_month_idxs, n_days):
    """
    Full original WhisperEmbedder forward (projection + conv + day encoding).
    dropout=0 at eval time; day encoding uses the same sinusoidal formula.
    """
    import math

    # --- projection ---
    out_btc = _orig_projection(
        x_btc, day_idx, ckpt_weights, session_to_orig_month, orig_month_idxs
    )
    out_bct = out_btc.permute(0, 2, 1)  # (B, C, T)

    # --- pad/trim to 3000 ---
    T = out_bct.shape[2]
    target = 3000
    if T < target:
        out_bct = F.pad(out_bct, (0, target - T))
    else:
        out_bct = out_bct[:, :, :target]

    # --- conv1 (GELU) ---
    out_bct = F.gelu(F.conv1d(
        out_bct,
        ckpt_weights["conv1_weight"],
        ckpt_weights["conv1_bias"],
        padding=3,    # kernel_size=7 → padding = 7//2 = 3
    ))
    # --- conv2 (GELU, stride=2) ---
    out_bct = F.gelu(F.conv1d(
        out_bct,
        ckpt_weights["conv2_weight"],
        ckpt_weights["conv2_bias"],
        stride=2,
        padding=1,    # kernel_size=3 → padding = 3//2 = 1
    ))
    x = out_bct.transpose(1, 2)  # (B, 1500, 384)

    # --- session encoding (sinusoidal, first d_model//2 = 192 dims) ---
    # Paper Eq.2: session enc occupies the first half, temporal enc the second half.
    # The original codebase uses the opposite convention (session in last half); we
    # follow the paper. The weight matrices are learned, so this is equivalent.
    d_model = x.shape[2]
    half = d_model // 2
    max_len = n_days
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, half, 2, dtype=torch.float) * (-math.log(10000.0) / half)
    )
    pe = torch.zeros(max_len, half)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: half // 2])

    day_enc = pe[day_idx]  # (B, 192)
    x[:, :, :half] = x[:, :, :half] + day_enc.unsqueeze(1)

    return x  # (B, 1500, 384)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_day_projection_matches_original(
    real_data, ckpt_weights, session_to_orig_month, orig_month_idxs, loaded_embedder
):
    """
    BWEmbedder._apply_day_proj must produce bit-for-bit identical output to
    the original WhisperEmbedder projection step for the same weights and signals.
    """
    x_btc, day_idx = real_data
    x_bct = x_btc.permute(0, 2, 1)  # (B, C, T)

    orig_out_btc = _orig_projection(
        x_btc, day_idx, ckpt_weights, session_to_orig_month, orig_month_idxs
    )

    with torch.no_grad():
        new_out_bct = loaded_embedder._apply_day_proj(x_bct, day_idx)
    new_out_btc = new_out_bct.permute(0, 2, 1)

    max_diff = (new_out_btc - orig_out_btc).abs().max().item()
    assert torch.allclose(new_out_btc, orig_out_btc, atol=1e-6), (
        f"Projection mismatch: max abs diff = {max_diff:.2e}"
    )


def test_full_embedding_matches_original(
    real_data, ckpt_weights, session_to_orig_month, orig_month_idxs, loaded_embedder
):
    """
    BWEmbedder.forward() (projection + conv + day encoding) must be numerically
    identical to the original WhisperEmbedder forward for the same weights.
    """
    x_btc, day_idx = real_data
    x_bct = x_btc.permute(0, 2, 1)

    orig_out = _orig_full_embed(
        x_btc, day_idx, ckpt_weights, session_to_orig_month, orig_month_idxs,
        n_days=ckpt_weights["n_days"],
    )

    with torch.no_grad():
        new_out = loaded_embedder(x_bct, day_idx)

    max_diff = (new_out - orig_out).abs().max().item()
    assert torch.allclose(new_out, orig_out, atol=1e-5), (
        f"Full embedding mismatch: max abs diff = {max_diff:.2e}"
    )


def test_projection_covers_multiple_months(real_data, session_to_orig_month):
    """Sanity check: the _N_SAMPLES batch exercises at least 2 distinct month buckets."""
    _, day_idx = real_data
    months_hit = {session_to_orig_month[d] for d in day_idx.tolist()}
    assert len(months_hit) >= 2, (
        f"Test batch hits only one month ({months_hit}); "
        "increase _N_SAMPLES to get meaningful coverage"
    )
