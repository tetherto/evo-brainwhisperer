"""Numerical regression tests for the augmentation functions.

Pure functions on synthetic tensors. Each test compares the package function
against a copy of the original implementation (the code behind the published
runs, written with (B, T, C) inputs), so no data is needed.
``test_augmentations_shapes`` checks ``transform_data`` on synthetic input.
"""

import torch

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

    new_out = gaussian_smooth(x_bct, kernel_size, sigma)  # (B, C, T)
    old_out = _old_gauss_smooth(x_btc, kernel_size, sigma)  # (B, T, C)

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
        warp_mat = torch.tile(
            torch.unsqueeze(torch.eye(C, device=features.device), 0), (B, 1, 1)
        )
        warp_mat += torch.randn_like(warp_mat) * gain_std
        return torch.matmul(features, warp_mat)

    B, C, T = (
        4,
        64,
        100,
    )  # smaller C keeps the test fast; correctness is layout-independent
    gain_std = 0.2
    torch.manual_seed(0)
    x_bct = torch.randn(B, C, T)

    torch.manual_seed(42)
    new_out = static_gain_noise(x_bct, gain_std)  # (B, C, T)

    torch.manual_seed(42)
    old_out = _old(x_bct.permute(0, 2, 1), gain_std)  # (B, T, C)

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
    new_out = offset_noise(x_bct, offset_std)  # (B, C, T)

    torch.manual_seed(42)
    old_out = _old(x_bct.permute(0, 2, 1), offset_std)  # (B, T, C)

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
        max_lens_rep = max_mask_lens.repeat_interleave(num_masks)
        t = (
            (torch.rand(B_rep, device=device) * (max_lens_rep + 1).float())
            .floor()
            .long()
        )
        t0 = (
            (
                torch.rand(B_rep, device=device)
                * (valid_lens_rep - t + 1).clamp(min=1).float()
            )
            .floor()
            .long()
        )
        arange = torch.arange(P, device=device).unsqueeze(0)
        mask_reps = (arange >= t0.unsqueeze(1)) & (arange < (t0 + t).unsqueeze(1))
        batch_ids = torch.arange(B, device=device).repeat_interleave(num_masks)
        nz = mask_reps.nonzero(as_tuple=False)
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
    new_out = time_mask(x_bct, lengths, num_masks, max_mask_pct)  # (B, C, T)

    torch.manual_seed(42)
    old_out = _old(
        x_bct.permute(0, 2, 1), lengths, num_masks, max_mask_pct
    )  # (B, T, C)

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
    assert noise.mean().abs() < 1e-3, (
        f"white_noise mean {noise.mean():.4f}, expected ~0"
    )
    assert abs(noise.std().item() - noise_std) < 5e-3, (
        f"white_noise std {noise.std():.4f}, expected ~{noise_std}"
    )


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
    var_late = out[:, :, -10:].var().item()
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

    cut = T - x_out.shape[2]  # infer cut from the T change
    assert 0 <= cut < 50, f"cut {cut} outside expected range [0, 49]"
    assert x_out.shape == (B, C, T - cut), f"unexpected output shape {x_out.shape}"
    assert torch.equal(l_out, (lengths - cut).clamp(min=0)), (
        "lengths not decremented by cut"
    )
    # Confirm the cut removed from the front: x_out should equal x[:, :, cut:]
    assert torch.equal(x_out, x[:, :, cut:]), (
        "random_time_cut did not slice from the front"
    )


def test_augmentations_shapes():
    """transform_data preserves (B, C, T) shape and does not produce NaN."""
    from brainwhisperer.data.augmentations import transform_data

    B, C, T = 4, 512, 300
    x = torch.randn(B, C, T)
    lengths = torch.full((B,), T, dtype=torch.long)

    for mode in ("train", "val"):
        x_out, l_out = transform_data(x, lengths, mode=mode)
        assert x_out.shape == (B, C, x_out.shape[2]), f"shape mismatch in {mode}"
        assert not torch.isnan(x_out).any(), f"NaN in {mode} output"
        assert l_out.shape == (B,)
        assert (l_out >= 0).all()
