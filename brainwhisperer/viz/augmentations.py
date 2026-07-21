import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from torch import Tensor

# Raw signal: purple → off-white → blue (same as the heatmap in data_pipeline)
_BW_CMAP = LinearSegmentedColormap.from_list(
    "bw_signal", ["#5237B7", "#F4F0FF", "#39ADFF"], N=256
)

# Augmented signal: pink → off-white → light-blue
_ZOOM_CMAP = LinearSegmentedColormap.from_list(
    "bw_zoom", ["#39ADFF", "#F4F0FF", "#B3DEFA"], N=256
)

_BLUE = "#39ADFF"
_PINK = "#DE5698"
_DARK = "#333333"
_GRAY = "#888888"
_PURPLE = "#5237B7"


def _masked_spans(sig_zoom: "np.ndarray", t0: int) -> "list[tuple[int,int]]":
    """
    Return (start, end) frame pairs (in original signal coordinates) where every
    channel in the zoom window is zero — i.e. time-masked frames.
    """
    zeroed = np.abs(sig_zoom).max(axis=0) < 1e-6  # (t1-t0,)
    if not zeroed.any():
        return []
    changes = np.diff(zeroed.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0] + t0
    ends = np.where(changes == -1)[0] + t0
    return list(zip(starts.tolist(), ends.tolist()))


def _draw_panel(
    ax_heat: "plt.Axes",
    ax_tr: "plt.Axes",
    sig: "np.ndarray",  # (C, T) full signal
    cmap,
    accent: str,
    title: str,
    ch0: int,
    ch1: int,
    t0: int,
    t1: int,
    show_masks: bool = False,  # shade time-masked frames in the traces panel
) -> None:
    """Draw one heatmap + traces pair into the given axes."""
    C, T = sig.shape

    # ── heatmap ───────────────────────────────────────────────────
    vmax = float(np.percentile(np.abs(sig), 98)) or 1.0
    ax_heat.imshow(
        sig,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        origin="upper",
    )
    ax_heat.set_title(title, fontsize=10, color=_DARK, pad=6)
    ax_heat.set_xlabel("Time (frames)", fontsize=9, color=_DARK, labelpad=6)
    ax_heat.set_ylabel("Channel", fontsize=9, color=_DARK, labelpad=6)
    ax_heat.tick_params(labelsize=7, colors=_DARK)
    ax_heat.spines[:].set_color(_GRAY)

    # zoom box: fill + border drawn separately so alpha doesn't wash the edge
    ax_heat.add_patch(
        mpatches.Rectangle(
            (t0 - 0.5, ch0 - 0.5),
            t1 - t0,
            ch1 - ch0,
            facecolor=accent,
            edgecolor="none",
            alpha=0.20,
            linewidth=0,
            zorder=2,
        )
    )
    ax_heat.add_patch(
        mpatches.Rectangle(
            (t0 - 0.5, ch0 - 0.5),
            t1 - t0,
            ch1 - ch0,
            fill=False,
            edgecolor=accent,
            linewidth=1.8,
            zorder=3,
        )
    )

    # ── traces ────────────────────────────────────────────────────
    n_ch = ch1 - ch0
    traces = sig[ch0:ch1, t0:t1]
    t_axis = np.arange(t0, t1)
    spacing = 2.8

    # shade masked regions behind the traces
    if show_masks:
        for span_s, span_e in _masked_spans(traces, t0):
            ax_tr.axvspan(span_s, span_e, color=accent, alpha=0.18, linewidth=0)

    for i in range(n_ch):
        tr = traces[i]
        amp = float(np.percentile(np.abs(tr), 95)) + 1e-8
        ax_tr.plot(
            t_axis,
            tr / amp * 0.9 + (n_ch - 1 - i) * spacing,
            linewidth=0.85,
            color=accent,
            alpha=0.92,
        )

    ax_tr.set_xlim(t0, t1)
    ax_tr.set_ylim(-spacing * 0.5, n_ch * spacing)
    ax_tr.set_xlabel("Time (frames)", fontsize=9, color=_DARK, labelpad=6)

    ytick_pos = [(n_ch - 1 - i) * spacing for i in range(n_ch)]
    ax_tr.set_yticks(ytick_pos)
    ax_tr.set_yticklabels([str(ch0 + i) for i in range(n_ch)], fontsize=7)
    ax_tr.tick_params(axis="y", which="both", length=0, colors=_DARK)
    ax_tr.tick_params(axis="x", labelsize=7, colors=_DARK)
    ax_tr.yaxis.grid(False)
    ax_tr.spines[["top", "right"]].set_visible(False)
    ax_tr.spines[["bottom", "left"]].set_color(accent)


def plot_augmentation(
    raw: Tensor,
    augmented: Tensor,
    lengths: Tensor,
    sample_idx: int = 0,
    zoom_channels: tuple[int, int] = (0, 16),
    zoom_time: tuple[int, int] = (0, 150),
) -> Figure:
    """
    Four-panel figure comparing a raw and augmented EEG signal.

    Left pair  — raw signal:       purple/blue heatmap  +  blue accent traces
    Right pair — augmented signal: pink/lightblue heatmap  +  pink accent traces

    Each pair uses the same layout as the data-pipeline zoom: a full heatmap
    with a coloured zoom box, and a traces panel for the zoomed region.
    """
    raw_sig = raw[sample_idx].detach().cpu().float().numpy()  # (C, T)
    aug_sig = augmented[sample_idx].detach().cpu().float().numpy()  # (C, T)
    valid = int(lengths[sample_idx].item())

    # clip to valid length
    raw_sig = raw_sig[:, :valid]
    aug_sig = aug_sig[:, :valid]
    C, T = raw_sig.shape

    # clamp zoom bounds
    ch0 = max(0, min(zoom_channels[0], C - 2))
    ch1 = max(ch0 + 1, min(zoom_channels[1], C))
    t0 = max(0, min(zoom_time[0], T - 2))
    t1 = max(t0 + 1, min(zoom_time[1], T))

    # layout: [raw_heat | raw_tr | spacer | aug_heat | aug_tr]
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(
        1, 5, figure=fig, width_ratios=[1.3, 0.7, 0.10, 1.3, 0.7], wspace=0.30
    )

    ax_raw_heat = fig.add_subplot(gs[0, 0])
    ax_raw_tr = fig.add_subplot(gs[0, 1])
    ax_aug_heat = fig.add_subplot(gs[0, 3])
    ax_aug_tr = fig.add_subplot(gs[0, 4])

    _draw_panel(
        ax_raw_heat, ax_raw_tr, raw_sig, _BW_CMAP, _PURPLE, "Raw", ch0, ch1, t0, t1
    )
    _draw_panel(
        ax_aug_heat,
        ax_aug_tr,
        aug_sig,
        _ZOOM_CMAP,
        _BLUE,
        "Augmented",
        ch0,
        ch1,
        t0,
        t1,
        show_masks=True,
    )

    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12)
    return fig
