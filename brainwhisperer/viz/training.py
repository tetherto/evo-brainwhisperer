from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_loss_curves(
    steps: Sequence[int],
    ce_losses: Sequence[float],
    ctc_losses: Sequence[float],
) -> Figure:
    """CE and CTC phone loss per epoch on a shared axis.

    Args:
        steps:      Epoch indices (x-axis).
        ce_losses:  Cross-entropy loss values, one per epoch.
        ctc_losses: CTC phoneme loss values, one per epoch.

    Returns:
        A Figure with both curves, legend, and no top/right spines.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(steps, ce_losses, label="CE loss", linewidth=1.5, color="#5B8DB8")
    ax.plot(steps, ctc_losses, label="CTC phone loss", linewidth=1.5, color="#E285A0")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss curves")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_wer_curve(
    epochs: Sequence[int],
    wer_values: Sequence[float],
) -> Figure:
    """Validation WER per epoch.

    Args:
        epochs:     Epoch indices at which WER was computed (x-axis).
        wer_values: WER values in [0, 1], one per validation epoch.

    Returns:
        A Figure with the WER curve and no top/right spines.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        epochs, wer_values, linewidth=1.5, color="#6DAE81", marker="o", markersize=3
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("WER")
    ax.set_title("Validation word error rate")
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
