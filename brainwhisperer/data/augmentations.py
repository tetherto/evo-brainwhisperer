"""
Pure augmentation functions operating on (B, C, T) float tensors.
All functions are stateless and device-agnostic.
transform_data() is the composition used in lightning_module.training_step().
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from torch import Tensor


# Cross-channel warp: constructs a (B, C, C) matrix (identity + noise) and
# mixes every input channel into every output channel via matmul. Each output
# channel is a weighted sum of all input channels, not just a scaled copy of one.
def static_gain_noise(x: Tensor, gain_std: float) -> Tensor:
    B, C, T = x.shape
    warp_mat = torch.tile(
        torch.unsqueeze(torch.eye(C, device=x.device, dtype=x.dtype), 0), (B, 1, 1)
    )  # initialise an eye tensor with (B, C, C) shape
    warp_mat += gain_std * torch.randn_like(
        warp_mat
    )  # produce Gaussian samples with shape like `warp_mat`
    x_btc = x.permute(
        0, 2, 1
    )  # prepare input data format for matrix multiplication (B, T, C)
    x_btc = torch.bmm(
        x_btc, warp_mat
    )  # `bmm`: batched matrix multiply > (B, T, C)x @ (B, C, C) = (B, T, C)
    return x_btc.permute(0, 2, 1)  # (B, C, T)


# Adds independent Gaussian noise to every element.
def white_noise(x: Tensor, noise_std: float) -> Tensor:
    return x + noise_std * torch.randn_like(x)


# Adds a per-channel constant offset (same value for all time steps within a channel).
# Simulates slow DC baseline drift between trials.
def offset_noise(x: Tensor, offset_std: float) -> Tensor:
    B, C, _ = x.shape
    offset = offset_std * torch.randn(B, C, 1, device=x.device, dtype=x.dtype)
    return x + offset


# Adds noise whose cumulative sum grows along the time axis.
# Simulates low-frequency drift that accumulates over a trial.
def random_walk_noise(x: Tensor, walk_std: float) -> Tensor:
    noise = walk_std * torch.randn_like(x)
    return x + noise.cumsum(dim=-1)


# Single cut applied uniformly to all samples in the batch: the entire batch
# is shifted forward by the same number of frames and the tensor is physically
# truncated (T shrinks).
def random_time_cut(x: Tensor, lengths: Tensor, max_cut: int) -> tuple[Tensor, Tensor]:
    cut = int(torch.randint(0, max_cut, (1,)).item())  # [0, max_cut-1], draws an `int`
    if cut == 0:
        return x, lengths
    return x[:, :, cut:], (lengths - cut).clamp(min=0)


# Applies a depthwise 1D Gaussian convolution along the time axis independently per channel.
# Pre-processing step that's always applied (both train and val).
# Low-pass filter suppressing high-frequency noise irrelevant to speech decoding.
def gaussian_smooth(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    B, C, T = x.shape
    imp = np.zeros(kernel_size, dtype=np.float32)
    imp[kernel_size // 2] = 1
    kernel = gaussian_filter1d(imp, sigma)
    valid_idx = np.argwhere(kernel > 0.01)
    kernel = kernel[valid_idx]
    kernel = np.squeeze(kernel / kernel.sum()).astype(
        np.float32
    )  # np.squeeze drops the trailing 1-size dimension to (k,) shape
    weight = torch.tensor(kernel, dtype=torch.float32, device=x.device)
    weight = weight.view(1, 1, -1).repeat(C, 1, 1)  # (C, 1, k)
    return F.conv1d(x, weight, padding="same", groups=C)


# Vectorized over all B * num_masks (sample, slot) pairs to avoid a Python loop.
# The function is highly efficient but poorly readable; a toy example is included
# with the following mock values: B=2, num_masks=2, T=10
#   lengths       = [10, 8]          (sample 0 has 10 valid frames, sample 1 has 8)
#   max_mask_pct  = 0.3
#   max_widths  = [3, 2]           (sample 0 can be masked up to 3 frames, sample 1 up to 2)
def time_mask(
    x: Tensor, lengths: Tensor, num_masks: int, max_mask_pct: float
) -> Tensor:
    B, C, T = x.shape
    device = x.device

    max_widths = (max_mask_pct * lengths.float()).long()

    # To avoid looping over num_masks, one `n_slots = B * num_masks entries`
    # is created per (sample, mask slot) pair, and all of them are processed in parallel.
    # `repeat_interleave` repeats each element `num_masks` times consecutively:
    #   lengths    = [10,  8]
    #   → slot_valid_len = [10, 10,  8,  8]   (2 slots per sample)
    #   → slot_max_width = [ 3,  3,  2,  2]
    n_slots = B * num_masks
    slot_valid_len = lengths.repeat_interleave(num_masks)  # (n_slots,)
    slot_max_width = max_widths.repeat_interleave(num_masks)  # (n_slots,)

    # `rand()` gives a float in [0, 1). Multiplying by (max+1) and flooring gives
    # a discrete uniform draw from {0, 1, ..., max}.
    # In the example it could yield `widths = [2, 1, 0, 2]`  (one draw per slot)
    widths = (
        (torch.rand(n_slots, device=device) * (slot_max_width + 1).float())
        .floor()
        .long()
    )

    # Start position is constrained so the window [start, start+width) stays within
    # the valid signal, clamping to 1 prevents an empty range when width happens to equal valid_len.
    # `slot_valid_len - widths + 1` = [10-2+1, 10-1+1, 8-0+1, 8-2+1] = [9, 10, 9, 7]
    # Assuming `starts `= [3, 7, 0, 5], then  gives windows: [3,5), [7,8), [0,0), [5,7)
    starts = (
        (
            torch.rand(n_slots, device=device)
            * (slot_valid_len - widths + 1).clamp(min=1).float()
        )
        .floor()
        .long()
    )

    # arange is [0,1,...,T-1] shaped (1,T); comparing with starts/ends shaped (n_slots,1)
    # broadcasts to (n_slots, T). Elements are `True` wherever the time index falls inside the window.
    # slot 0 → [F,F,F,T,T,F,...]
    # slot 1 → [F,...,F,T,F,...],
    # slot 2 → all F
    # slot 3 → [F,...,F,T,T,F,...]
    arange = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
    slot_masks = (arange >= starts.unsqueeze(1)) & (
        arange < (starts + widths).unsqueeze(1)
    )  # (n_slots, T)

    # `slot_to_sample` maps each slot index back to its original sample:
    # `arange(B).repeat_interleave(num_masks)` gives [0, 0, 1, 1], meaning that the first
    # two slots are for B=0 and the second for B=1.
    slot_to_sample = torch.arange(B, device=device).repeat_interleave(
        num_masks
    )  # (n_slots,)
    # nonzero() returns the (slot_idx, time_idx) of every True entry in slot_masks.
    masked_pos = slot_masks.nonzero(
        as_tuple=False
    )  # (N, 2): each row is [slot_idx, time_idx]
    # Indexing slot_to_sample with slot_idx translates to the real sample index.
    sample_idx = slot_to_sample[masked_pos[:, 0]]  # translate slot → sample
    time_idx = masked_pos[:, 1]
    # Setting mask[sample_idx, time_idx] = True unions all windows for the same sample.
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[sample_idx, time_idx] = True
    # In the running example, we have:
    #  mask[0, 3]=True
    #  mask[0,4]=True
    #  mask[0,7]=True
    #  mask[1,5]=True
    #  mask[1,6]=True

    # mask=True marks positions to silence (exclusion mask, SpecAugment convention).
    # ~mask is therefore True where the signal is kept and False where it is zeroed out.
    # unsqueeze(1) gives (B,1,T) which broadcasts across all C channels.
    return x * (~mask).unsqueeze(1)


def transform_data(
    x: Tensor,
    lengths: Tensor,
    mode: str,
    gain_std: float = 0.2,
    noise_std: float = 0.05,
    offset_std: float = 0.05,
    walk_std: float = 0.001,
    max_cut: int = 50,
    smooth_kernel_size: int = 100,
    smooth_sigma: float = 2.0,
    num_masks: int = 10,
    max_mask_pct: float = 0.075,
) -> tuple[Tensor, Tensor]:
    """
    Full augmentation pipeline.
    mode="train": all seven augmentations applied.
    mode="val":   Gaussian smoothing only.
    """
    assert mode in ("train", "val"), f"mode must be 'train' or 'val', got {mode!r}"
    if mode == "train":
        x = static_gain_noise(x, gain_std)
        x = white_noise(x, noise_std)
        x = offset_noise(x, offset_std)
        x = random_walk_noise(x, walk_std)
        x, lengths = random_time_cut(x, lengths, max_cut)
    x = gaussian_smooth(x, smooth_kernel_size, smooth_sigma)
    if mode == "train":
        x = time_mask(x, lengths, num_masks, max_mask_pct)
    return x, lengths
