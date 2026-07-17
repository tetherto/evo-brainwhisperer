"""
Pure augmentation functions operating on (B, C, T) float tensors.
All functions are stateless and device-agnostic.
transform_data() is the composition used in lightning_module.training_step().
"""

import torch
import torch.nn.functional as F
from torch import Tensor


# Simulates electrode sensitivity drift: each channel gets its own
# scaling coefficient drawn near 1. Equivalent to electrodes being randomly
# more or less sensitive on a trial.
def static_gain_noise(x: Tensor, gain_std: float) -> Tensor:
    B, C, T = x.shape
    scale = 1.0 + gain_std * torch.randn(B, C, 1, device=x.device, dtype=x.dtype)
    return x * scale


# Simplest augmentation, adds independent Gaussian noise to every element.
def white_noise(x: Tensor, noise_std: float) -> Tensor:
    return x + noise_std * torch.randn_like(x)


# Adds a per-channel constant offset: same value for all time steps within a channel).
# Simulates slow baseline drift where an electrode's DC level shifts between trials.
def offset_noise(x: Tensor, offset_std: float) -> Tensor:
    B, C, _ = x.shape
    offset = offset_std * torch.randn(B, C, 1, device=x.device, dtype=x.dtype)
    return x + offset


# Adds noise whose cumulative sum grows along the time axis.
# Simulates low-frequency drift that accumulates over the course of a trial.
def random_walk_noise(x: Tensor, walk_std: float) -> Tensor:
    noise = walk_std * torch.randn_like(x)
    return x + noise.cumsum(dim=-1)


# Removes up to max_cut frames from the start of the signal,
# shifts everything left, and zero-pads the end. Simulates variability in the neural response onset
# so that the model learns not to rely on the precise alignment of the signal to the trial start.
def random_time_cut(x: Tensor, lengths: Tensor, max_cut: int) -> tuple[Tensor, Tensor]:
    """Remove up to max_cut frames from the start of each signal, shift left, zero-pad right."""
    B, C, T = x.shape
    cuts = torch.randint(0, max_cut + 1, (B,))
    out = torch.zeros_like(x)
    new_lengths = lengths.clone()
    for i in range(B):
        c = int(cuts[i].item())
        out[i, :, : T - c] = x[i, :, c:]
        new_lengths[i] = max(0, int(lengths[i].item()) - c)
    return out, new_lengths


# Applies a depthwise 1D Gaussian convolution along the time axis independently per channel.
# Pre-processing step that's always applied (both train and val).
# Low-pass filter suppressing high-frequency noise irrelevant to speech decoding.
def gaussian_smooth(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    """Depthwise 1D Gaussian smoothing along the time axis."""
    B, C, T = x.shape
    k = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - kernel_size // 2
    kernel = torch.exp(-0.5 * (k / sigma) ** 2)
    kernel = kernel / kernel.sum()
    # (C, 1, kernel_size) for depthwise conv
    weight = kernel.view(1, 1, -1).expand(C, 1, -1).contiguous()
    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad), mode="reflect")
    return F.conv1d(x_padded, weight, groups=C)


# Sets random rectangular windows of time steps to zero, independently per sample.
# Forces the model to decode from partial signal, regularising against overfitting
# to specific temporal patterns.
def time_mask(x: Tensor, num_masks: int, max_mask_pct: float) -> Tensor:
    """SpecAugment-style time masking: set random rectangular windows to zero."""
    B, C, T = x.shape
    out = x.clone()
    max_width = max(1, int(max_mask_pct * T))
    for _ in range(num_masks):
        widths = torch.randint(1, max_width + 1, (B,))
        starts = torch.randint(0, max(1, T - max_width), (B,))
        for i in range(B):
            s = int(starts[i].item())
            w = int(widths[i].item())
            out[i, :, s : s + w] = 0.0
    return out


def transform_data(
    x: Tensor,
    lengths: Tensor,
    mode: str,
    gain_std: float = 0.2,
    noise_std: float = 0.05,
    offset_std: float = 0.05,
    walk_std: float = 0.001,
    max_cut: int = 50,
    smooth_kernel_size: int = 5,
    smooth_sigma: float = 1.0,
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
        x = time_mask(x, num_masks, max_mask_pct)
    return x, lengths
