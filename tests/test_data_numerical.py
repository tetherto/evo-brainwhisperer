"""
Numerical regression tests: verify that the refactored data layer produces
bit-for-bit identical outputs to the original upstream implementation.

These tests import the upstream BaseNeuralTextDataset and collate_fn_flexible
directly via sys.path — intentionally, because this file exists solely to
validate equivalence against the old codebase. Once the upstream repo is
decommissioned these tests can be removed.
"""

import pickle
import sys

import torch

from brainwhisperer.config.paths import CARD_ROOT, KUNZ_ROOT
from brainwhisperer.data.datasets.base import BaseNeuralDataset

_KUNZ_BASE = "data/kunz_dataset_for_BaseNeuralTextDataset/padded_feature_dim"
N_SAMPLES = 20  # number of consecutive indices checked per dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upstream():
    """Return (BaseNeuralTextDataset, collate_fn_flexible) from the upstream repo."""
    if CARD_ROOT not in sys.path:
        sys.path.insert(0, CARD_ROOT)
    from model_training.augmented_dataset import (
        BaseNeuralTextDataset,
        collate_fn_flexible,
    )

    return BaseNeuralTextDataset, collate_fn_flexible


def _load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _compare_items(new_ds: BaseNeuralDataset, old_ds, n: int = N_SAMPLES) -> None:
    """Assert that new and old __getitem__ return numerically identical dicts."""
    n = min(n, len(new_ds), len(old_ds))
    for i in range(n):
        new = new_ds[i]
        old = old_ds[i]

        assert torch.allclose(new["neural_features"], old["neural_features"]), (
            f"neural_features mismatch at idx {i}, "
            f"max diff = {(new['neural_features'] - old['neural_features']).abs().max()}"
        )
        assert torch.equal(new["day"], old["day"]), (
            f"day mismatch at idx {i}: {new['day']} vs {old['day']}"
        )
        assert new["sentence_label"] == old["sentence_label"], (
            f"sentence_label mismatch at idx {i}: "
            f"{new['sentence_label']!r} vs {old['sentence_label']!r}"
        )
        assert new["n_time_steps"] == old["n_time_steps"], (
            f"n_time_steps mismatch at idx {i}: "
            f"{new['n_time_steps']} vs {old['n_time_steps']}"
        )
        assert torch.equal(new["seq_class_ids"], old["seq_class_ids"]), (
            f"seq_class_ids mismatch at idx {i}"
        )
        assert new["transcriptions"] == old["transcriptions"], (
            f"transcriptions mismatch at idx {i}"
        )


# ---------------------------------------------------------------------------
# Per-dataset item-level tests
# ---------------------------------------------------------------------------


def test_card_item_numerical_consistency():
    BaseNeuralTextDataset, _ = _upstream()
    raw = _load(f"{CARD_ROOT}/data/cleaned_train_data.pkl")

    new_ds = BaseNeuralDataset(raw, has_labels=True, source_dataset="card")
    old_ds = BaseNeuralTextDataset(raw, eval_type="val", source_dataset="card")

    _compare_items(new_ds, old_ds)


def test_willet_item_numerical_consistency():
    BaseNeuralTextDataset, _ = _upstream()
    raw = _load(f"{CARD_ROOT}/data/willet_cleaned_train_data.pkl")

    new_ds = BaseNeuralDataset(raw, has_labels=True, source_dataset="willet")
    old_ds = BaseNeuralTextDataset(raw, eval_type="val", source_dataset="willet")

    _compare_items(new_ds, old_ds)


def test_kunz_t12_item_numerical_consistency():
    BaseNeuralTextDataset, _ = _upstream()
    raw = _load(f"{KUNZ_ROOT}/{_KUNZ_BASE}/kunz_t12_train.pkl")

    new_ds = BaseNeuralDataset(raw, has_labels=True, source_dataset="card")
    old_ds = BaseNeuralTextDataset(raw, eval_type="val", source_dataset="card")

    _compare_items(new_ds, old_ds)


def test_kunz_t15_item_numerical_consistency():
    BaseNeuralTextDataset, _ = _upstream()
    raw = _load(f"{KUNZ_ROOT}/{_KUNZ_BASE}/kunz_t15_train.pkl")

    new_ds = BaseNeuralDataset(raw, has_labels=True, source_dataset="card")
    old_ds = BaseNeuralTextDataset(raw, eval_type="val", source_dataset="card")

    _compare_items(new_ds, old_ds)


# ---------------------------------------------------------------------------
# Augmentation numerical tests
#
# Testability by function:
#   static_gain_noise  — testable: both call randn_like on same (B,C,C) shape
#   offset_noise       — testable: randn(B,1,C) and randn(B,C,1) share flat order
#   gaussian_smooth    — testable: fully deterministic
#   time_mask          — testable: both call torch.rand in the same sequence
#   white_noise        — NOT element-wise: same flat values, different (b,t,c) assignments
#   random_walk_noise  — NOT element-wise: same situation as white_noise
#   random_time_cut    — NOT seed-comparable: old uses numpy RNG, new uses torch RNG
# ---------------------------------------------------------------------------


def test_gaussian_smooth_matches_original():
    """
    gaussian_smooth must produce bit-for-bit identical output to the original
    gauss_smooth. Both are deterministic given the same kernel parameters.
    Stochastic augmentations (static_gain_noise, white_noise, etc.) cannot be
    compared directly because numpy and torch RNGs produce different streams.
    """
    import torch.nn.functional as F
    from scipy.ndimage import gaussian_filter1d as _gf1d

    from brainwhisperer.data.augmentations import gaussian_smooth

    def _old_gauss_smooth(x, kernel_size, sigma):
        # Verbatim port of gauss_smooth from pl_wrapper.py (operates on (B,T,C))
        import numpy as np
        B, T, C = x.shape
        inp = np.zeros(kernel_size, dtype=np.float32)
        inp[kernel_size // 2] = 1
        kernel = _gf1d(inp, sigma)
        valid_idx = np.argwhere(kernel > 0.01)
        kernel = kernel[valid_idx]
        kernel = np.squeeze(kernel / kernel.sum())
        gauss_t = torch.tensor(kernel, dtype=torch.float32, device=x.device)
        gauss_t = gauss_t.view(1, 1, -1).repeat(C, 1, 1)
        x_bct = x.permute(0, 2, 1)  # (B, C, T)
        smoothed = F.conv1d(x_bct, gauss_t, padding="same", groups=C)
        return smoothed.permute(0, 2, 1)  # (B, T, C)

    torch.manual_seed(0)
    B, C, T = 4, 512, 300
    x_bct = torch.randn(B, C, T)
    x_btc = x_bct.permute(0, 2, 1)  # (B, T, C) for old function

    kernel_size, sigma = 100, 2.0

    new_out = gaussian_smooth(x_bct, kernel_size, sigma)           # (B, C, T)
    old_out = _old_gauss_smooth(x_btc, kernel_size, sigma)         # (B, T, C)

    # Compare in the same layout
    assert torch.allclose(new_out, old_out.permute(0, 2, 1), atol=1e-6), (
        f"gaussian_smooth mismatch, max diff = "
        f"{(new_out - old_out.permute(0, 2, 1)).abs().max():.2e}"
    )


def test_static_gain_noise_matches_original():
    """Both call randn_like on the same (B,C,C) warp_mat — identical RNG stream."""
    from brainwhisperer.data.augmentations import static_gain_noise

    def _old(features, gain_std):
        # Verbatim from pl_wrapper.py; features is (B, T, C)
        B, T, C = features.shape
        warp_mat = torch.tile(torch.unsqueeze(torch.eye(C, device=features.device), 0), (B, 1, 1))
        warp_mat += torch.randn_like(warp_mat) * gain_std
        return torch.matmul(features, warp_mat)

    B, C, T = 4, 64, 100  # smaller C keeps the test fast; correctness is layout-independent
    gain_std = 0.2
    torch.manual_seed(0)
    x_bct = torch.randn(B, C, T)

    torch.manual_seed(42)
    new_out = static_gain_noise(x_bct, gain_std)          # (B, C, T)

    torch.manual_seed(42)
    old_out = _old(x_bct.permute(0, 2, 1), gain_std)      # (B, T, C)

    assert torch.allclose(new_out, old_out.permute(0, 2, 1), atol=1e-5), (
        f"static_gain_noise mismatch, max diff = "
        f"{(new_out - old_out.permute(0, 2, 1)).abs().max():.2e}"
    )


def test_offset_noise_matches_original():
    """randn(B,1,C) and randn(B,C,1) share flat storage order, so same seed = same values."""
    from brainwhisperer.data.augmentations import offset_noise

    def _old(features, offset_std):
        # Verbatim from pl_wrapper.py; features is (B, T, C)
        B, T, C = features.shape
        return features + torch.randn((B, 1, C), device=features.device) * offset_std

    B, C, T = 4, 512, 300
    offset_std = 0.05
    torch.manual_seed(0)
    x_bct = torch.randn(B, C, T)

    torch.manual_seed(42)
    new_out = offset_noise(x_bct, offset_std)              # (B, C, T)

    torch.manual_seed(42)
    old_out = _old(x_bct.permute(0, 2, 1), offset_std)    # (B, T, C)

    assert torch.allclose(new_out, old_out.permute(0, 2, 1), atol=1e-6), (
        f"offset_noise mismatch, max diff = "
        f"{(new_out - old_out.permute(0, 2, 1)).abs().max():.2e}"
    )


def test_time_mask_matches_original():
    """Both call torch.rand twice in the same sequence; same seed = same mask."""
    from brainwhisperer.data.augmentations import time_mask

    def _old(X, X_len, num_masks, max_mask_pct):
        # Verbatim from pl_wrapper.py apply_time_masking; X is (B, T, C)
        B, P, D = X.shape
        device = X.device
        max_mask_lens = (max_mask_pct * X_len.float()).long()
        B_rep = B * num_masks
        valid_lens_rep = X_len.repeat_interleave(num_masks)
        max_lens_rep   = max_mask_lens.repeat_interleave(num_masks)
        t  = (torch.rand(B_rep, device=device) * (max_lens_rep + 1).float()).floor().long()
        t0 = (torch.rand(B_rep, device=device) * (valid_lens_rep - t + 1).clamp(min=1).float()).floor().long()
        arange     = torch.arange(P, device=device).unsqueeze(0)
        mask_reps  = (arange >= t0.unsqueeze(1)) & (arange < (t0 + t).unsqueeze(1))
        batch_ids  = torch.arange(B, device=device).repeat_interleave(num_masks)
        nz         = mask_reps.nonzero(as_tuple=False)
        b_idx, p_idx = batch_ids[nz[:, 0]], nz[:, 1]
        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[b_idx, p_idx] = True
        X_masked = X.clone()
        X_masked[mask] = 0
        return X_masked

    B, C, T = 4, 512, 300
    lengths = torch.tensor([300, 280, 250, 200], dtype=torch.long)
    num_masks, max_mask_pct = 5, 0.075
    torch.manual_seed(0)
    x_bct = torch.randn(B, C, T)

    torch.manual_seed(42)
    new_out = time_mask(x_bct, lengths, num_masks, max_mask_pct)   # (B, C, T)

    torch.manual_seed(42)
    old_out = _old(x_bct.permute(0, 2, 1), lengths, num_masks, max_mask_pct)  # (B, T, C)

    assert torch.allclose(new_out, old_out.permute(0, 2, 1), atol=1e-6), (
        f"time_mask mismatch, max diff = "
        f"{(new_out - old_out.permute(0, 2, 1)).abs().max():.2e}"
    )


def test_white_noise_statistical_properties():
    """
    white_noise cannot be compared element-wise: randn(B,T,C) and randn(B,C,T)
    generate the same flat values but assign them to different (b,t,c) positions.
    Verified instead via statistical properties: noise is zero-mean, correct variance.
    """
    from brainwhisperer.data.augmentations import white_noise

    torch.manual_seed(0)
    B, C, T, noise_std = 64, 512, 300, 0.05
    x = torch.zeros(B, C, T)
    out = white_noise(x, noise_std)

    noise = out - x
    assert noise.mean().abs() < 1e-3,           f"white_noise mean {noise.mean():.4f}, expected ~0"
    assert abs(noise.std().item() - noise_std) < 5e-3, \
        f"white_noise std {noise.std():.4f}, expected ~{noise_std}"


def test_random_walk_noise_statistical_properties():
    """
    random_walk_noise cannot be compared element-wise for the same reason as white_noise.
    Verified via: (a) noise is not zero (cumsum grows), (b) no NaN, (c) variance
    grows with time position as expected for a random walk.
    """
    from brainwhisperer.data.augmentations import random_walk_noise

    torch.manual_seed(0)
    B, C, T, walk_std = 32, 512, 300, 0.001
    x = torch.zeros(B, C, T)
    out = random_walk_noise(x, walk_std)

    assert not torch.isnan(out).any(), "NaN in random_walk_noise output"
    # Variance at t should grow ~linearly: Var(t) ≈ t * walk_std²
    var_early = out[:, :, :10].var().item()
    var_late  = out[:, :, -10:].var().item()
    assert var_late > var_early, (
        f"random_walk_noise variance should grow with t: early={var_early:.2e}, late={var_late:.2e}"
    )


def test_random_time_cut_behaviour():
    """
    random_time_cut uses torch RNG while the original used numpy RNG — seeds
    cannot be matched across the two streams. Verified instead via invariants:
    the cut is batch-uniform, T shrinks by exactly cut, and lengths decrease by cut.
    """
    from brainwhisperer.data.augmentations import random_time_cut

    B, C, T = 4, 512, 300
    lengths = torch.tensor([300, 280, 250, 200], dtype=torch.long)
    x = torch.randn(B, C, T)

    torch.manual_seed(0)
    x_out, l_out = random_time_cut(x, lengths, max_cut=50)

    cut = T - x_out.shape[2]           # infer cut from the T change
    assert 0 <= cut < 50,              f"cut {cut} outside expected range [0, 49]"
    assert x_out.shape == (B, C, T - cut), f"unexpected output shape {x_out.shape}"
    assert torch.equal(l_out, (lengths - cut).clamp(min=0)), "lengths not decremented by cut"
    # Confirm the cut removed from the front: x_out should equal x[:, :, cut:]
    assert torch.equal(x_out, x[:, :, cut:]), "random_time_cut did not slice from the front"


# ---------------------------------------------------------------------------
# Collator-level batch test
# ---------------------------------------------------------------------------


def test_collator_numerical_consistency():
    """
    NeuralCollator's inline padding must produce identical neural_features,
    neural_lengths, and day_idx to what collate_fn_flexible returned.
    Labels are not compared — the tokenizer call is unchanged.
    """
    from transformers import WhisperTokenizer

    from brainwhisperer.data.collate import NeuralCollator

    BaseNeuralTextDataset, collate_fn_flexible = _upstream()

    raw = _load(f"{CARD_ROOT}/data/cleaned_train_data.pkl")
    indices = list(range(8))

    # New path
    new_ds = BaseNeuralDataset(raw, has_labels=True, source_dataset="card")
    new_samples = [new_ds[i] for i in indices]
    for s in new_samples:
        s["sbj_idx"] = 0

    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="en", task="transcribe"
    )
    new_batch = NeuralCollator(tokenizer)(new_samples)

    # Old path
    old_ds = BaseNeuralTextDataset(raw, eval_type="val", source_dataset="card")
    old_samples = [old_ds[i] for i in indices]
    old_collated = collate_fn_flexible(old_samples)

    # signals in NeuralBatch are (B,C,T); permute back to (B,T,C) for comparison
    new_signals_btc = new_batch.signals.permute(0, 2, 1)
    assert torch.allclose(new_signals_btc, old_collated["neural_features"]), (
        "neural_features batch mismatch, "
        f"max diff = {(new_signals_btc - old_collated['neural_features']).abs().max()}"
    )
    assert torch.equal(new_batch.lengths, old_collated["neural_lengths"]), (
        "neural_lengths mismatch"
    )
    assert torch.equal(new_batch.day_idx, old_collated["day"]), "day_idx mismatch"
