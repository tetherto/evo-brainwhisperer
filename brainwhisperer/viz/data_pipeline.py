"""
Explanatory diagrams for the data pipeline.

Two figures that walk through how a recipe dataset becomes a NeuralBatch.

  plot_dataset_structure  —  one recording day's trials from a recipe dataset
  plot_batch_anatomy      —  a collated NeuralBatch with real-vs-padding split

Both functions draw every element explicitly with no shared sub-routines,
so each can be read top-to-bottom as a flat drawing script.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.data.collection import day_indices
from brainwhisperer.data.recipes import RECIPES

# ── brand palette ──────────────────────────────────────────────────────────────
_PURPLE = "#5237B7"  # real-signal fill
_PINK = "#DE5698"  # boundary marker
_OFFWHITE = "#F4F0FF"  # padding fill
_GRAY = "#888888"  # secondary / muted
_DARK = "#333333"  # all annotation text — single colour avoids contrast issues

_BLUE = "#39ADFF"  # accent — zoom box and traces

_SIGNAL_CMAP = LinearSegmentedColormap.from_list(
    "bw_signal", ["#5237B7", "#F4F0FF", "#39ADFF"], N=256
)


# ── diagram 1: dataset structure ───────────────────────────────────────────────


def plot_dataset_structure(
    ds,
    name: str,
    day: int = 0,
    n_sentences: int = 6,
    zoom_sentence: int = 0,
    zoom_channels: tuple[int, int] = (0, 16),
    zoom_time: tuple[int, int] = (0, 150),
) -> Figure:
    """
    Four-panel figure showing one day's raw data at three levels of detail.

    ``ds`` is a prepared recipe dataset (``RECIPES[name].build_val_dataset``
    or ``build_train_dataset``) and ``name`` its registered dataset name.
    Every trial of the chosen day is loaded, so pick a val set for speed.

    Col 0  — title + subtitle (top), horizontal bar chart of every sentence's
              duration in appearance order (sentence 0 at top).
    Col 1  — isometric card stack: one card per sentence.  The zoom_sentence
              card has a blue border to mark it as the one expanded on the right.
    Col 2  — full EEG heatmap for the selected sentence with a blue zoom box.
    Col 3  — stacked waveform traces for the zoom region, drawn in accent blue.
    """
    from matplotlib.gridspec import GridSpec

    # ── load every trial of this day as a (C, T) array ─────────────
    neural = RECIPES[name].NEURAL
    all_day_indices = [i for i, d in enumerate(day_indices(ds, name)) if d == day]
    if not all_day_indices:
        raise ValueError(f"{name}: no trials on day {day}")
    signals = [
        np.asarray(ds[i].data[neural], dtype=np.float32) for i in all_day_indices
    ]
    total_in_day = len(all_day_indices)
    n_shown = min(n_sentences, total_in_day)

    all_t_values = [x.shape[1] for x in signals]
    t_values = all_t_values[:n_shown]
    C = signals[0].shape[0]
    T_ref = max(t_values)

    # ── resolve zoom bounds ────────────────────────────────────────
    zoom_sentence = max(0, min(zoom_sentence, n_shown - 1))
    sig_zoom = signals[zoom_sentence]  # (C, T)
    T_zoom = sig_zoom.shape[1]

    ch0 = max(0, min(zoom_channels[0], C - 2))
    ch1 = max(ch0 + 1, min(zoom_channels[1], C))
    t0 = max(0, min(zoom_time[0], T_zoom - 2))
    t1 = max(t0 + 1, min(zoom_time[1], T_zoom))

    # ── figure: 4 columns, 2 rows ──────────────────────────────────
    fig = plt.figure(figsize=(26, 7))
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[1.0, 1.5, 1.3, 0.7],
        height_ratios=[0.11, 0.89],
        wspace=0.12,
        hspace=0.04,
    )

    ax_title = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_stack = fig.add_subplot(gs[:, 1])
    ax_heat = fig.add_subplot(gs[:, 2])
    ax_tr = fig.add_subplot(gs[:, 3])

    ax_title.axis("off")
    ax_stack.axis("off")

    # ── title panel ────────────────────────────────────────────────
    ax_title.text(
        0.0,
        1.0,
        f"{name} dataset",
        fontsize=15,
        fontweight="medium",
        color=_DARK,
        va="top",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.0,
        0.35,
        f"Day {day}: {total_in_day} sentences",
        fontsize=11,
        color=_GRAY,
        va="top",
        transform=ax_title.transAxes,
    )

    # ── histogram: horizontal bars, sentence 0 at top ─────────────
    ax_hist.barh(
        range(n_shown),
        all_t_values[:n_shown],
        color=_PURPLE,
        alpha=0.80,
        height=0.8,
        linewidth=0,
    )
    if total_in_day > n_shown:
        ax_hist.barh(
            range(n_shown, total_in_day),
            all_t_values[n_shown:],
            color=_GRAY,
            alpha=0.35,
            height=0.8,
            linewidth=0,
        )

    ax_hist.invert_yaxis()
    ax_hist.set_xlabel("Duration (frames)", fontsize=9, color=_DARK)
    ax_hist.set_ylabel("Sentences", fontsize=9, color=_DARK)
    ax_hist.tick_params(length=0, labelbottom=False, labelleft=False)
    ax_hist.minorticks_off()
    ax_hist.grid(False)
    ax_hist.spines[:].set_visible(False)

    # ── card stack ─────────────────────────────────────────────────
    CARD_H = 1.0
    CARD_MAX = 3.0
    ISO = np.radians(30)
    STEP = CARD_MAX * 0.30
    ey = np.array([np.cos(ISO), np.sin(ISO)]) * STEP

    card_widths = [CARD_MAX * T_i / T_ref for T_i in t_values]
    right_edges = [i * ey[0] + w for i, w in enumerate(card_widths)]
    label_col_x = max(right_edges) + 0.25

    y_max = CARD_H + (n_shown - 1) * ey[1]
    ARR_OFFSET = 0.22
    ax_stack.set_xlim(-0.40, label_col_x + 2.0)
    ax_stack.set_ylim(-0.38, y_max + ARR_OFFSET + 0.45)

    LABEL_ZO = n_shown + 10
    N_CH_DRAW = 80
    N_T_DRAW = 60

    for sentence_pos in reversed(range(n_shown)):
        T_i = t_values[sentence_pos]
        card_w = card_widths[sentence_pos]
        zo = n_shown - sentence_pos
        x0 = sentence_pos * ey[0]
        y0 = sentence_pos * ey[1]

        sig_raw = signals[sentence_pos]
        ch_idx = np.round(np.linspace(0, C - 1, N_CH_DRAW)).astype(int)
        t_idx = np.round(np.linspace(0, T_i - 1, N_T_DRAW)).astype(int)
        sig_sub = sig_raw[np.ix_(ch_idx, t_idx)]
        vmax = float(np.percentile(np.abs(sig_sub), 95)) + 1e-8
        sig_n = (np.clip(sig_sub / vmax, -1.0, 1.0) + 1.0) / 2.0

        xs = np.linspace(x0, x0 + card_w, N_T_DRAW + 1)
        ys = np.linspace(y0, y0 + CARD_H, N_CH_DRAW + 1)
        XX, YY = np.meshgrid(xs, ys)
        ax_stack.pcolormesh(
            XX,
            YY,
            sig_n,
            cmap=_SIGNAL_CMAP,
            vmin=0,
            vmax=1,
            shading="flat",
            alpha=0.75,
            zorder=zo,
            rasterized=True,
        )

        # Purple border on all cards
        border_color = _PURPLE if sentence_pos == zoom_sentence else _PURPLE
        border_lw = 1.8 if sentence_pos == zoom_sentence else 0.9
        ax_stack.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                card_w,
                CARD_H,
                fill=False,
                edgecolor=border_color,
                linewidth=border_lw,
                zorder=zo + 0.5,
            )
        )

        ax_stack.text(
            x0 + card_w,
            y0,
            f"T = {T_i}",
            ha="right",
            va="bottom",
            fontsize=8,
            color=_DARK,
            zorder=LABEL_ZO,
        )
        ax_stack.plot(
            [x0 + card_w + 0.02, label_col_x - 0.05],
            [y0, y0],
            color=_DARK,
            linewidth=0.5,
            linestyle=":",
            alpha=0.5,
            zorder=LABEL_ZO,
        )
        ax_stack.text(
            label_col_x,
            y0,
            f"Sentence [{sentence_pos}]",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=_DARK,
            zorder=LABEL_ZO,
        )

    front_w = card_widths[0]
    ax_stack.text(
        front_w / 2, -0.25, "Time →", ha="center", va="top", fontsize=9, color=_DARK
    )
    ax_stack.text(
        -0.28,
        CARD_H / 2,
        "Channels",
        ha="right",
        va="center",
        fontsize=9,
        color=_DARK,
        rotation=90,
    )

    ax_stack.annotate(
        "",
        xytext=(0, CARD_H + ARR_OFFSET),
        xy=((n_shown - 1) * ey[0], (n_shown - 1) * ey[1] + CARD_H + ARR_OFFSET),
        arrowprops=dict(arrowstyle="->", color=_DARK, lw=0.9),
        zorder=LABEL_ZO,
    )
    ax_stack.text(
        (n_shown - 1) * ey[0] / 2 - 0.08,
        CARD_H + ARR_OFFSET + (n_shown - 1) * ey[1] / 2 + 0.08,
        "Sentences",
        ha="right",
        va="bottom",
        fontsize=9,
        color=_DARK,
        rotation=np.degrees(ISO),
        zorder=LABEL_ZO,
    )

    # ── heatmap panel: full signal for zoom_sentence ───────────────
    vmax_h = float(np.percentile(np.abs(sig_zoom), 98)) or 1.0
    ax_heat.imshow(
        sig_zoom,
        aspect="auto",
        interpolation="nearest",
        cmap=_SIGNAL_CMAP,
        vmin=-vmax_h,
        vmax=vmax_h,
        origin="upper",
    )
    ax_heat.set_title(f"Sentence [{zoom_sentence}]", fontsize=9, color=_DARK, pad=6)
    ax_heat.set_xlabel("Time (frames)", labelpad=6, fontsize=9, color=_DARK)
    ax_heat.set_ylabel("Channel", labelpad=6, fontsize=9, color=_DARK)
    ax_heat.tick_params(labelsize=7, colors=_DARK)
    ax_heat.spines[:].set_color(_GRAY)

    # Blue zoom box — fill + border drawn separately so alpha doesn't wash the edge
    ax_heat.add_patch(
        mpatches.Rectangle(
            (t0 - 0.5, ch0 - 0.5),
            t1 - t0,
            ch1 - ch0,
            facecolor=_PURPLE,
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
            edgecolor=_PURPLE,
            linewidth=1.8,
            zorder=3,
        )
    )

    # ── traces panel: waveforms for the zoom region ────────────────
    n_ch = ch1 - ch0
    traces = sig_zoom[ch0:ch1, t0:t1]
    t_axis = np.arange(t0, t1)
    spacing = 2.8

    for i in range(n_ch):
        tr = traces[i]
        amp = float(np.percentile(np.abs(tr), 95)) + 1e-8
        ax_tr.plot(
            t_axis,
            tr / amp * 0.9 + (n_ch - 1 - i) * spacing,
            linewidth=0.85,
            color=_PURPLE,
            alpha=0.92,
        )

    ax_tr.set_xlim(t0, t1)
    ax_tr.set_ylim(-spacing * 0.5, n_ch * spacing)
    ax_tr.set_xlabel("Time (frames)", labelpad=6, fontsize=9, color=_DARK)
    ytick_pos = [(n_ch - 1 - i) * spacing for i in range(n_ch)]
    ax_tr.set_yticks(ytick_pos)
    ax_tr.set_yticklabels([str(ch0 + i) for i in range(n_ch)], fontsize=7)
    ax_tr.tick_params(axis="y", which="both", length=0, colors=_DARK)
    ax_tr.tick_params(axis="x", labelsize=7, colors=_DARK)
    ax_tr.yaxis.grid(False)
    ax_tr.spines[["top", "right"]].set_visible(False)
    ax_tr.spines[["bottom", "left"]].set_color(_PURPLE)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.09)
    return fig


# ── diagram 2: batch anatomy ───────────────────────────────────────────────────


def plot_batch_anatomy(
    batch: NeuralBatch,
    max_rows: int = 6,
) -> Figure:
    """
    Schematic of a collated NeuralBatch, shape (B, C, T_max).

    Each slab represents one sentence.
    Left region  — real neural signal ( T_i frames )  — solid purple
    Right region — zero padding       ( T_max - T_i ) — near-white

    Label placement rules:
      - T_i label: anchored at the TOP-LEFT corner of the real-signal region
      - pad label: anchored at the TOP-LEFT corner of the padding region
      - Both at zorder > all slabs so front slabs never occlude back labels
      - Day labels: fixed right column connected by dotted lines (same style
        as plot_dataset_structure)
      - T_max arrow: below the front slab, spanning its exact x extent (0→SLAB_W)
      - B arrow: vertical on the far left, outside the slab cluster
    """

    B = min(int(batch.signals.shape[0]), max_rows)
    T_max = int(batch.signals.shape[2])

    lengths = [int(batch.lengths[i]) for i in range(B)]
    day_idxs = [int(batch.day_idx[i]) for i in range(B)]

    # ── isometric geometry ────────────────────────────────────────
    SLAB_W = 5.0
    SLAB_H = 1.0
    ISO = np.radians(30)
    STEP = SLAB_H * 0.7  # ey[1] ≈ 0.35 — enough gap between slab tops
    ey = np.array([np.cos(ISO), np.sin(ISO)]) * STEP

    x_max = SLAB_W + (B - 1) * ey[0]
    y_max = SLAB_H + (B - 1) * ey[1]

    day_col_x = x_max + 0.30
    LABEL_ZO = B + 10  # above every slab layer

    fig, ax = plt.subplots(figsize=(13, max(5.5, y_max + 2.2)))
    ax.axis("off")
    ax.set_xlim(-0.90, day_col_x + 1.6)
    ax.set_ylim(-0.65, y_max + 0.70)

    # ── draw slabs back-to-front ──────────────────────────────────
    for i in reversed(range(B)):
        T_i = lengths[i]
        day_i = day_idxs[i]
        real_frac = T_i / T_max
        real_w = SLAB_W * real_frac
        pad_w = SLAB_W - real_w

        zo = B - i
        x0 = i * ey[0]
        y0 = i * ey[1]

        # Real signal region
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                real_w,
                SLAB_H,
                facecolor=_PURPLE,
                edgecolor="none",
                alpha=0.45,
                zorder=zo,
            )
        )

        # Zero-padding region
        ax.add_patch(
            mpatches.Rectangle(
                (x0 + real_w, y0),
                pad_w,
                SLAB_H,
                facecolor=_OFFWHITE,
                edgecolor="none",
                alpha=0.90,
                zorder=zo,
            )
        )

        # Outer slab border
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y0),
                SLAB_W,
                SLAB_H,
                fill=False,
                edgecolor=_PURPLE,
                linewidth=0.9,
                zorder=zo + 0.5,
            )
        )

        # Pink dashed boundary between real and padding
        ax.plot(
            [x0 + real_w, x0 + real_w],
            [y0, y0 + SLAB_H],
            color=_PINK,
            linewidth=1.3,
            linestyle="--",
            zorder=zo + 0.6,
        )

        # ── T_i label: TOP-LEFT corner of the real region ─────────
        top_y = y0 + SLAB_H + 0.06
        ax.text(
            x0,
            top_y,
            r"$T_{}$ = {}".format(i, T_i),
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=_DARK,
            zorder=LABEL_ZO,
        )

        # ── pad label: TOP-LEFT corner of the padding region ──────
        if pad_w / SLAB_W > 0.06:
            ax.text(
                x0 + real_w + pad_w,
                top_y,
                f"pad = {T_max - T_i}",
                ha="right",
                va="bottom",
                fontsize=8.5,
                color=_DARK,
                zorder=LABEL_ZO,
            )

        # ── day label: fixed column, dotted connector from slab ────
        mid_y = y0 + SLAB_H / 2
        ax.plot(
            [x0 + SLAB_W + 0.03, day_col_x - 0.05],
            [mid_y, mid_y],
            color=_DARK,
            linewidth=0.5,
            linestyle=":",
            alpha=0.5,
            zorder=LABEL_ZO,
        )
        ax.text(
            day_col_x,
            mid_y,
            f"Day {day_i}",
            ha="left",
            va="center",
            fontsize=9.5,
            color=_DARK,
            zorder=LABEL_ZO,
        )

    # ── T_max arrow: below the front slab, x = 0 → SLAB_W ────────
    # Aligns exactly with the front slab's time extent.
    ax.annotate(
        "",
        xy=(SLAB_W, -0.14),
        xytext=(0.0, -0.14),
        arrowprops=dict(arrowstyle="<->", color=_DARK, lw=1.0),
    )
    ax.text(
        SLAB_W / 2,
        -0.24,
        r"$T_{max}$" + " = {}".format(T_max),
        ha="center",
        va="top",
        fontsize=9,
        color=_DARK,
    )

    # "time ->" sits below the T_max annotation
    ax.text(
        SLAB_W / 2, -0.50, "Time  →", ha="center", va="top", fontsize=10, color=_DARK
    )

    # ── C channels label ──────────────────────────────────────────
    ax.text(
        ey[0] - 0.7,
        SLAB_H / 2,
        "Channels",
        ha="right",
        va="center",
        fontsize=10,
        color=_DARK,
        rotation=90,
    )

    # ── B arrow: diagonal along the isometric depth direction ────────
    # Starts above the front slab top so it clears the T_i label rows,
    # then runs diagonally to the back slab — matching the stack's depth axis.
    depth_x = (B - 1) * ey[0]
    depth_y = (B - 1) * ey[1]
    B_orig_y = SLAB_H + 0.32  # above the highest T_i label row
    ax.annotate(
        "",
        xytext=(0, B_orig_y),
        xy=(depth_x, B_orig_y + depth_y),
        arrowprops=dict(arrowstyle="<->", color=_DARK, lw=1.0),
        zorder=LABEL_ZO,
    )
    ax.text(
        depth_x / 2 - 0.10,
        B_orig_y + depth_y / 2 + 0.10,
        f"B = {B}",
        ha="right",
        va="bottom",
        fontsize=9,
        color=_DARK,
        rotation=np.degrees(ISO),
        zorder=LABEL_ZO,
    )

    # ── legend ────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=_PURPLE, alpha=0.45, label="Real signal"),
        mpatches.Patch(
            facecolor=_OFFWHITE, edgecolor=_GRAY, linewidth=0.5, label="Zero padding"
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=9,
        framealpha=0.85,
        edgecolor=_GRAY,
    )

    fig.tight_layout()
    return fig
