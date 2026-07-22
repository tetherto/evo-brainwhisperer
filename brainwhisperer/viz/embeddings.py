import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from torch import Tensor

_SIGNAL_CMAP = LinearSegmentedColormap.from_list(
    "bw_signal", ["#DE5698", "#F0E4FA", "#5237B7"], N=256
)
_PURPLE = "#5237B7"
_BLUE = "#39ADFF"
_DARK = "#333333"
_GRAY = "#888888"


def plot_conv_pipeline(
    signal: "np.ndarray",  # (C, T_padded)        raw neural signal (T_padded = 3000)
    act_conv1: "np.ndarray",  # (d_model, T_padded)  activations after conv1 + GELU
    act_conv2: "np.ndarray",  # (d_model, 1500)       activations after conv2 + GELU
    kernel_size_1: int = 7,
    stride_2: int = 2,
    n_lines: int = 60,
) -> Figure:
    """
    Three-card pipeline showing the actual Conv1d mechanics.

    Cards display a short zoomed-in time window (T_SHOW steps) so individual
    time positions are wide enough to annotate:
      - A kernel window of width k on the input/conv1 cards (highlighted box)
      - Fan-in lines from every position inside the kernel to the single output
        column it produces — showing how k × C cells collapse to d_model values
      - Two stride-2 windows on the Conv1 card showing how consecutive windows
        are offset by stride steps, explaining T → T/2

    Card heights are proportional to channel count (C vs d_model) and widths
    are proportional to displayed time (T_SHOW vs T_SHOW/stride_2).
    """
    C, T_in = signal.shape
    Dm, T_c1 = act_conv1.shape
    _, T_c2 = act_conv2.shape
    K2 = 3  # conv2 kernel size is always 3 in the architecture

    # ── pick an active time window to zoom into ───────────────────
    T_SHOW = 20
    T_SHOW2 = T_SHOW // stride_2  # = 10 for stride=2
    energy = np.abs(signal).mean(axis=0)
    active = np.where(energy > energy.max() * 0.05)[0]
    t0 = int(active[len(active) // 4]) if len(active) > T_SHOW else 0
    t0 = max(kernel_size_1, min(t0, T_c1 - T_SHOW - kernel_size_1))

    sig_win = signal[:, t0 : t0 + T_SHOW]
    conv1_win = act_conv1[:, t0 : t0 + T_SHOW]
    t0_c2 = t0 // stride_2
    conv2_win = act_conv2[:, t0_c2 : t0_c2 + T_SHOW2]

    # ── display subsampling (channel axis only; keep all T_SHOW cols) ─
    N_CH_IN = min(C, 96)
    N_CH_DM = min(Dm, round(N_CH_IN * Dm / C))

    def _prep(arr: "np.ndarray", n_ch: int) -> "np.ndarray":
        C_, T_ = arr.shape
        ci = np.round(np.linspace(0, C_ - 1, n_ch)).astype(int)
        sub = arr[ci, :].astype(np.float32)
        v = float(np.percentile(np.abs(sub), 98)) or 1.0
        return np.clip(sub / v, -1.0, 1.0)

    sig_d = _prep(sig_win, N_CH_IN)
    conv1_d = _prep(conv1_win, N_CH_DM)
    conv2_d = _prep(conv2_win, N_CH_DM)

    # ── geometry: heights ∝ channels, widths ∝ displayed T ────────
    C_H = 4.0
    Dm_H = C_H * Dm / C  # e.g. 4.0 × 384/512 = 3.0
    T_W = 6.0  # input + conv1 cards (T_SHOW columns)
    T2_W = T_W * T_SHOW2 / T_SHOW  # conv2 card (T_SHOW/2 columns) = 3.0
    GAP = 2.4

    x0s, x0e = 0.0, T_W
    x1s, x1e = x0e + GAP, x0e + GAP + T_W
    x2s, x2e = x1e + GAP, x1e + GAP + T2_W

    fig, ax = plt.subplots(figsize=(22, 7.5))
    ax.set_xlim(-0.6, x2e + 0.7)
    ax.set_ylim(-2.0, C_H + 1.8)
    ax.axis("off")

    # ── helpers ────────────────────────────────────────────────────
    def _card(
        x0: float, x1: float, card_h: float, data: "np.ndarray", title: str
    ) -> None:
        xs = np.linspace(x0, x1, data.shape[1] + 1)
        ys = np.linspace(0, card_h, data.shape[0] + 1)
        XX, YY = np.meshgrid(xs, ys)
        ax.pcolormesh(
            XX,
            YY,
            data,
            cmap=_SIGNAL_CMAP,
            vmin=-1,
            vmax=1,
            shading="flat",
            alpha=0.88,
            rasterized=True,
            zorder=2,
        )
        ax.add_patch(
            mpatches.Rectangle(
                (x0, 0),
                x1 - x0,
                card_h,
                fill=False,
                edgecolor=_PURPLE,
                linewidth=1.2,
                zorder=3,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            card_h + 0.18,
            title,
            ha="center",
            va="bottom",
            fontsize=10,
            color=_DARK,
        )

    # x-position of time index t within a card of width W spanning T steps
    def _tx(card_x0: float, card_w: float, t: float, T: int) -> float:
        return card_x0 + (t / T) * card_w

    # ── draw the three cards ───────────────────────────────────────
    _card(x0s, x0e, C_H, sig_d, "Input signal")
    _card(x1s, x1e, Dm_H, conv1_d, "After  Conv1 + GELU")
    _card(x2s, x2e, Dm_H, conv2_d, "After  Conv2 + GELU")

    # shape labels below each card, coloured to flag what changed
    for (x0, x1), (ch_txt, t_txt), (ch_col, t_col) in zip(
        [(x0s, x0e), (x1s, x1e), (x2s, x2e)],
        [
            (f"C={C}", f"T={T_SHOW}"),
            (f"d={Dm}", f"T={T_SHOW}"),
            (f"d={Dm}", f"T={T_SHOW2}"),
        ],
        [(_GRAY, _GRAY), (_PURPLE, _GRAY), (_GRAY, _BLUE)],
    ):
        xm = (x0 + x1) / 2
        ax.text(
            xm - 0.05,
            -0.32,
            f"({ch_txt},",
            ha="right",
            va="top",
            fontsize=9,
            color=ch_col,
            fontfamily="monospace",
        )
        ax.text(
            xm + 0.05,
            -0.32,
            f"{t_txt})",
            ha="left",
            va="top",
            fontsize=9,
            color=t_col,
            fontfamily="monospace",
        )

    # ─────────────────────────────────────────────────────────────
    # CONV1 KERNEL WINDOW — annotate on the INPUT card
    # A window of width k1, spanning all C channels at one time position,
    # collapses to a single output column of d_model features.
    # ─────────────────────────────────────────────────────────────
    K1_T = T_SHOW // 2  # center time-step for the demo window
    K1_L = _tx(x0s, T_W, K1_T - kernel_size_1 // 2, T_SHOW)
    K1_R = _tx(x0s, T_W, K1_T + kernel_size_1 // 2 + 1, T_SHOW)

    # highlight the kernel window on the input card (full channel height)
    ax.add_patch(
        mpatches.Rectangle(
            (K1_L, 0), K1_R - K1_L, C_H, fc=_BLUE, ec="none", alpha=0.20, zorder=4
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (K1_L, 0),
            K1_R - K1_L,
            C_H,
            fc="none",
            ec=_BLUE,
            lw=1.8,
            alpha=0.90,
            zorder=5,
        )
    )

    # bracket below showing kernel width
    ax.annotate(
        "",
        xy=(K1_R, -0.70),
        xytext=(K1_L, -0.70),
        arrowprops=dict(arrowstyle="<->", color=_BLUE, lw=1.0),
    )
    ax.text(
        (K1_L + K1_R) / 2,
        -0.88,
        f"k = {kernel_size_1}",
        ha="center",
        va="top",
        fontsize=8,
        color=_BLUE,
    )

    # highlight the corresponding output column on the conv1 card
    OUT1_CX = _tx(x1s, T_W, K1_T, T_SHOW)
    OUT1_W = T_W / T_SHOW
    ax.add_patch(
        mpatches.Rectangle(
            (OUT1_CX - OUT1_W / 2, 0),
            OUT1_W,
            Dm_H,
            fc=_BLUE,
            ec="none",
            alpha=0.28,
            zorder=4,
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (OUT1_CX - OUT1_W / 2, 0),
            OUT1_W,
            Dm_H,
            fc="none",
            ec=_BLUE,
            lw=1.8,
            alpha=0.90,
            zorder=5,
        )
    )

    # fan-in lines: start at the RIGHT EDGE of the kernel window (K1_R),
    # end at the left edge of the output column — gap zone only, no card overlap.
    # Multiple channel positions fan into the narrower d_model height.
    out1_left = OUT1_CX - OUT1_W / 2
    t_ = np.linspace(0, 1, 18)
    B0f, B1f, B2f, B3f = (
        (1 - t_) ** 3,
        3 * (1 - t_) ** 2 * t_,
        3 * (1 - t_) * t_**2,
        t_**3,
    )
    cx_fan = (K1_R + out1_left) / 2
    for ch_frac in np.linspace(0.06, 0.94, 14):
        y_src = ch_frac * C_H
        y_dst = ch_frac * Dm_H  # converge toward d_model height
        dy = (y_dst - y_src) * 0.4
        xs_f = B0f * K1_R + (B1f + B2f) * cx_fan + B3f * out1_left
        ys_f = B0f * y_src + B1f * (y_src + dy) + B2f * (y_dst - dy) + B3f * y_dst
        ax.plot(xs_f, ys_f, color=_BLUE, lw=0.6, alpha=0.35, zorder=3)

    # ── Conv1 operation label ──────────────────────────────────────
    gap1_x = (x0e + x1s) / 2
    ax.text(
        gap1_x,
        C_H + 1.4,
        f"Conv1d  (k={kernel_size_1}, pad=same)  +  GELU",
        ha="center",
        va="bottom",
        fontsize=9,
        color=_DARK,
    )
    ax.text(
        gap1_x,
        C_H + 1.00,
        f"k×C = {kernel_size_1}×{C} input values  →  {Dm} output features",
        ha="center",
        va="bottom",
        fontsize=8,
        color=_GRAY,
    )

    # ─────────────────────────────────────────────────────────────
    # CONV2 STRIDE WINDOWS — annotate on the CONV1 card
    # Two kernel windows of width K2=3, offset by stride_2=2, showing how
    # adjacent output positions draw from overlapping but shifted input regions.
    # ─────────────────────────────────────────────────────────────
    # Three consecutive stride windows — the repeating offset makes stride=2 obvious
    _WIN_COLORS = [_PURPLE, "#E07A3A", "#2A9D8F"]  # purple · orange · teal

    K2_T1 = T_SHOW // 5  # first window centre (leaving room for all three)
    _win_centers = [K2_T1 + w * stride_2 for w in range(3)]

    for w_i, (k2_ct, wcol) in enumerate(zip(_win_centers, _WIN_COLORS)):
        k2_xl = _tx(x1s, T_W, k2_ct - K2 // 2, T_SHOW)
        k2_xr = _tx(x1s, T_W, k2_ct + K2 // 2 + 1, T_SHOW)

        # highlight window on conv1 card
        ax.add_patch(
            mpatches.Rectangle(
                (k2_xl, 0),
                k2_xr - k2_xl,
                Dm_H,
                fc=wcol,
                ec="none",
                alpha=0.22,
                zorder=4,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (k2_xl, 0),
                k2_xr - k2_xl,
                Dm_H,
                fc="none",
                ec=wcol,
                lw=1.8,
                alpha=0.88,
                zorder=5,
            )
        )

        # corresponding output column on conv2 card
        # input time k2_ct → output time ≈ k2_ct // stride_2
        out2_t = k2_ct // stride_2
        out2_cx = _tx(x2s, T2_W, out2_t, T_SHOW2)
        out2_w = T2_W / T_SHOW2
        ax.add_patch(
            mpatches.Rectangle(
                (out2_cx - out2_w / 2, 0),
                out2_w,
                Dm_H,
                fc=wcol,
                ec="none",
                alpha=0.30,
                zorder=4,
            )
        )
        ax.add_patch(
            mpatches.Rectangle(
                (out2_cx - out2_w / 2, 0),
                out2_w,
                Dm_H,
                fc="none",
                ec=wcol,
                lw=1.8,
                alpha=0.88,
                zorder=5,
            )
        )

        # Single bold arrow from window centre to output column centre.
        # Offset y so the two arrows sit at different heights and are readable.
        out2_left = out2_cx - out2_w / 2
        y_arr = Dm_H * (0.20 + w_i * 0.30)  # spread: 0.20, 0.50, 0.80 · Dm_H
        ax.annotate(
            "",
            xy=(out2_cx, y_arr),
            xytext=(k2_xr + 0.05, y_arr),
            arrowprops=dict(arrowstyle="-|>", color=wcol, lw=1.5, mutation_scale=11),
            zorder=6,
        )
        # label below the output column
        ax.text(
            out2_cx,
            -0.65,
            f"t′={w_i}",
            ha="center",
            va="top",
            fontsize=8,
            color=wcol,
            fontfamily="monospace",
        )

    # stride bracket between first and second window starts
    stride_xl = _tx(x1s, T_W, _win_centers[0] - K2 // 2, T_SHOW)
    stride_xr = _tx(x1s, T_W, _win_centers[1] - K2 // 2, T_SHOW)
    ax.annotate(
        "",
        xy=(stride_xr, -0.70),
        xytext=(stride_xl, -0.70),
        arrowprops=dict(arrowstyle="->", color=_PURPLE, lw=1.0),
    )
    ax.text(
        (stride_xl + stride_xr) / 2,
        -0.88,
        f"stride = {stride_2}",
        ha="center",
        va="top",
        fontsize=8,
        color=_PURPLE,
    )

    # ── Conv2 operation label ──────────────────────────────────────
    gap2_x = (x1e + x2s) / 2
    ax.text(
        gap2_x,
        C_H + 1.4,
        f"Conv1d  (k={K2}, stride={stride_2})  +  GELU",
        ha="center",
        va="bottom",
        fontsize=9,
        color=_DARK,
    )
    ax.text(
        gap2_x,
        C_H + 1.00,
        f"T = {T_SHOW}  →  T = {T_SHOW2}   (÷{stride_2})",
        ha="center",
        va="bottom",
        fontsize=8,
        color=_GRAY,
    )

    # ── light Bezier background flow (zone 1 only, subtle) ────────
    rng = np.random.default_rng(0)
    yas = np.linspace(0.0, C_H, n_lines)
    ybs = np.linspace(0.0, Dm_H, n_lines)
    cx_z1 = (x0e + x1s) / 2
    t_ = np.linspace(0, 1, 20)
    B0_z, B1_z, B2_z, B3_z = (
        (1 - t_) ** 3,
        3 * (1 - t_) ** 2 * t_,
        3 * (1 - t_) * t_**2,
        t_**3,
    )
    all_x_z, all_y_z = [], []
    for ya, yb in zip(yas, ybs):
        d = float(rng.uniform(-0.25, 0.25))
        all_x_z.extend((B0_z * x0e + (B1_z + B2_z) * cx_z1 + B3_z * x1s).tolist())
        all_x_z.append(np.nan)
        all_y_z.extend(
            (B0_z * ya + B1_z * (ya + d) + B2_z * (yb + d) + B3_z * yb).tolist()
        )
        all_y_z.append(np.nan)
    ax.plot(
        all_x_z, all_y_z, color=_PURPLE, lw=0.5, alpha=0.09, rasterized=True, zorder=1
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    return fig


def plot_patch_embeddings(embeddings: Tensor, sample_idx: int = 0) -> Figure:
    """
    Heatmap of (n_patches, d_model) embeddings for one sample.
    Works for both conv (1500 patches) and patch-mode outputs.
    """
    emb = embeddings[sample_idx].detach().cpu().float()  # (n_patches, d_model)
    n_patches, d_model = emb.shape

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        emb.T.numpy(), aspect="auto", interpolation="nearest", origin="lower"
    )
    ax.set_xlabel(f"Patch index (n={n_patches})")
    ax.set_ylabel(f"Feature dim (d={d_model})")
    ax.set_title(f"Patch embeddings — sample {sample_idx}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="activation")
    fig.tight_layout()
    return fig


_TEMPORAL_CMAP = LinearSegmentedColormap.from_list(
    "bw_temporal", [_PURPLE, "#DE5698", _BLUE], N=256
)


def plot_day_projection_pca(
    frames_before: "np.ndarray",  # (N, C) raw frames
    frames_after: "np.ndarray",  # (N, C) projected frames
    day_labels: "np.ndarray",  # (N,)  integer day indices
    title_prefix: str = "",
    n_components: int = 2,  # 2 or 3
    color_mode: str = "temporal",  # "temporal" | "discrete"
) -> Figure:
    """
    Side-by-side PCA of raw vs projected frames coloured by day.
    Fits PCA on the raw frames and reuses the same axes for the projected panel
    so cluster movement is directly comparable.

    color_mode="temporal": continuous colormap by day index — early sessions in
        purple, late in blue, gradient reveals temporal drift.
    color_mode="discrete": one colour per day (tab10), useful for small day counts.
    """
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    assert n_components in (2, 3), "n_components must be 2 or 3"
    assert color_mode in ("temporal", "discrete"), (
        "color_mode must be 'temporal' or 'discrete'"
    )

    pca = PCA(n_components=n_components).fit(frames_before)
    pc_before = pca.transform(frames_before)
    pc_after = pca.transform(frames_after)

    n_days = int(day_labels.max()) + 1
    norm = plt.Normalize(vmin=0, vmax=n_days - 1)
    cmap = (
        _TEMPORAL_CMAP
        if color_mode == "temporal"
        else plt.colormaps["tab10"].resampled(n_days)
    )

    def _inter_day_var(pcs: "np.ndarray") -> float:
        global_c = pcs.mean(axis=0)
        return float(
            np.mean(
                [
                    np.sum((pcs[day_labels == d].mean(axis=0) - global_c) ** 2)
                    for d in np.unique(day_labels)
                ]
            )
        )

    var_before = _inter_day_var(pc_before)
    var_after = _inter_day_var(pc_after)

    subplot_kw = {"projection": "3d"} if n_components == 3 else {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), subplot_kw=subplot_kw)

    for ax, pcs, subtitle in [
        (axes[0], pc_before, f"Before  (inter-day var = {var_before:.2f})"),
        (axes[1], pc_after, f"After   (inter-day var = {var_after:.2f})"),
    ]:
        if color_mode == "temporal":
            colors = cmap(norm(day_labels))
            if n_components == 3:
                ax.scatter(
                    pcs[:, 0],
                    pcs[:, 1],
                    pcs[:, 2],
                    c=day_labels,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.35,
                    s=12,
                )
            else:
                ax.scatter(
                    pcs[:, 0],
                    pcs[:, 1],
                    c=day_labels,
                    cmap=cmap,
                    norm=norm,
                    alpha=0.35,
                    s=16,
                )
            # centroids coloured by day, outlined in white so they pop
            for d in np.unique(day_labels):
                mask = day_labels == d
                c = pcs[mask].mean(axis=0)
                col = cmap(norm(d))
                if n_components == 3:
                    ax.scatter(
                        [c[0]],
                        [c[1]],
                        [c[2]],
                        color=col,
                        s=80,
                        marker="X",
                        edgecolors="white",
                        linewidths=0.8,
                        zorder=5,
                    )
                else:
                    ax.scatter(
                        c[0],
                        c[1],
                        color=col,
                        s=100,
                        marker="X",
                        edgecolors="white",
                        linewidths=0.8,
                        zorder=5,
                    )
        else:
            disc_cmap = plt.colormaps["tab10"].resampled(n_days)
            for d in range(n_days):
                mask = day_labels == d
                pts = pcs[mask]
                col = disc_cmap(d)
                if n_components == 3:
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        color=col,
                        alpha=0.35,
                        s=12,
                        label=f"Day {d}",
                    )
                    c = pts.mean(axis=0)
                    ax.scatter(
                        [c[0]],
                        [c[1]],
                        [c[2]],
                        color=col,
                        s=80,
                        marker="X",
                        edgecolors="white",
                        linewidths=0.6,
                        zorder=5,
                    )
                else:
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        color=col,
                        alpha=0.35,
                        s=16,
                        label=f"Day {d}",
                    )
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    ax.scatter(
                        cx,
                        cy,
                        color=col,
                        s=100,
                        marker="X",
                        edgecolors="white",
                        linewidths=0.7,
                        zorder=5,
                    )
            ax.legend(
                loc="upper right", fontsize=8, markerscale=1.2, framealpha=0.7, ncol=2
            )

        ax.set_title(subtitle)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        if n_components == 3:
            ax.set_zlabel("PC 3")

    fig.tight_layout()

    if color_mode == "temporal":
        # reserve a fixed strip on the right for the colorbar after tight_layout
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax, label="Recording day")

    prefix = f"{title_prefix} — " if title_prefix else ""
    fig.suptitle(f"{prefix}month/day projection: inter-session variability")
    return fig


def plot_session_centroids_pca(
    frames_before: "np.ndarray",  # (N, C) raw frames
    frames_after: "np.ndarray",  # (N, C) projected frames
    day_labels: "np.ndarray",  # (N,)  integer day indices
    title_prefix: str = "",
) -> Figure:
    """
    One point per recording session (centroid of all sampled frames), coloured
    by day index. PCA is fit on the full frame set so the coordinate system is
    shared with plot_day_projection_pca for direct comparison.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2).fit(frames_before)
    days = np.unique(day_labels)
    n_days = int(day_labels.max()) + 1
    norm = plt.Normalize(vmin=0, vmax=n_days - 1)

    def _centroids(frames):
        return np.stack([frames[day_labels == d].mean(axis=0) for d in days])

    cents_before = pca.transform(_centroids(frames_before))  # (D, 2)
    cents_after = pca.transform(_centroids(frames_after))  # (D, 2)

    def _inter_centroid_spread(cents):
        return float(np.mean(np.sum((cents - cents.mean(axis=0)) ** 2, axis=1)))

    spread_before = _inter_centroid_spread(cents_before)
    spread_after = _inter_centroid_spread(cents_after)

    # Compute shared axis bounds from the union of both panels so the visual
    # compression from projection is directly readable (before fills the space,
    # after clusters in the center of the same coordinate system).
    all_pts = np.vstack([cents_before, cents_after])
    pad = 0.12
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
    xrng = (xmax - xmin) * pad
    yrng = (ymax - ymin) * pad
    shared_xlim = (xmin - xrng, xmax + xrng)
    shared_ylim = (ymin - yrng, ymax + yrng)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, cents, subtitle in [
        (axes[0], cents_before, f"Before  (centroid spread = {spread_before:.2f})"),
        (axes[1], cents_after, f"After   (centroid spread = {spread_after:.2f})"),
    ]:
        ax.scatter(
            cents[:, 0],
            cents[:, 1],
            c=days,
            cmap=_TEMPORAL_CMAP,
            norm=norm,
            s=90,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.set_xlim(shared_xlim)
        ax.set_ylim(shared_ylim)
        ax.set_title(subtitle)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")

    fig.tight_layout()
    fig.subplots_adjust(right=0.88, top=0.88)
    cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=_TEMPORAL_CMAP, norm=norm),
        cax=cax,
        label="Recording day",
    )

    prefix = f"{title_prefix} — " if title_prefix else ""
    fig.suptitle(f"{prefix}session centroids: before/after projection", y=1.02)
    return fig


def plot_day_encoding(embedder, day_indices: Tensor) -> Figure:
    """
    Visualise day encoding vectors for a set of day indices.

    embedder: a BWEmbedder instance (used to extract day_encoding weights).
    day_indices: 1-D integer tensor of unique day indices to inspect.
    """
    with torch.no_grad():
        vecs = (
            embedder.day_encoding(day_indices).detach().cpu().float()
        )  # (n_days, d//2)

    n_days, dim = vecs.shape
    fig, ax = plt.subplots(figsize=(12, max(3, n_days * 0.4)))
    im = ax.imshow(
        vecs.numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=_SIGNAL_CMAP,
        vmin=-1,
        vmax=1,
    )
    ax.set_xlabel(f"Encoding dim (d={dim})")
    ax.set_ylabel("Day index")
    ax.set_yticks(range(n_days))
    ax.set_yticklabels([str(int(d)) for d in day_indices.tolist()], fontsize=7)
    ax.set_title("Day encoding vectors")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_positional_encoding(n_tokens: int = 1500, d_model: int = 384) -> Figure:
    """
    Heatmap of the sinusoidal positional encoding table used by BWEncoder.
    Shows the (d_model//2, n_tokens) matrix before it is added to token embeddings.
    """
    import math

    half = d_model // 2
    position = torch.arange(n_tokens, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, half, 2, dtype=torch.float) * (-math.log(10000.0) / half)
    )
    pe = torch.zeros(n_tokens, half)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: half // 2])

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        pe.numpy().T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=_SIGNAL_CMAP,
        vmin=-1,
        vmax=1,
    )
    ax.set_xlabel(f"Token position (n={n_tokens})")
    ax.set_ylabel(f"Encoding dim (d={half})")
    ax.set_title("Sinusoidal positional encoding (BWEncoder)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig
