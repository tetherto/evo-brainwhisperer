import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from transformers.modeling_outputs import BaseModelOutput


def plot_encoder_architecture(
    n_layers: int = 6,
    last_phoneme_layer: int = 4,
    d_model: int = 384,
    n_heads: int = 6,
    d_ff: int = 1536,
    attn_window_size: int = 29,
) -> Figure:
    """
    Static architecture diagram of BWEncoder.

    Layer stack colored purple→blue (paper Fig.1 convention: purple = phonemic,
    blue = word-level), with windowed/full attention split and CTC marker.
    Default dims are whisper-tiny: d_model=384, n_heads=6, d_ff=1536.
    """
    # Paper colors: phonemic purple → word-level blue
    _purple = np.array([0.420, 0.373, 0.745])
    _blue = np.array([0.718, 0.820, 0.949])

    BOX_W = 4.0
    BOX_H = 0.62
    GAP = 0.28
    BOX_X = 3.0  # wider left margin for the Phonemes arrow

    def _y(slot):
        """Bottom y-coordinate of slot; slot 0 = PE block, 1..n_layers = encoder layers."""
        return slot * (BOX_H + GAP)

    def _lerp(t):
        return tuple((_purple + (_blue - _purple) * t).tolist())

    def _text_color(color):
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        return "white" if lum < 0.55 else "#1A1A3E"

    top_y = _y(n_layers) + BOX_H
    mid_x = BOX_X + BOX_W / 2

    fig, ax = plt.subplots(figsize=(8.5, (top_y + 2.4) * 1.05))
    ax.set_xlim(0, 8.5)
    ax.set_ylim(-1.3, top_y + 1.3)
    ax.axis("off")

    # ── Input arrow ───────────────────────────────────────────────────────────
    ax.annotate(
        "",
        xy=(mid_x, _y(0)),
        xytext=(mid_x, _y(0) - 0.60),
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2),
    )
    ax.text(
        mid_x,
        _y(0) - 0.72,
        f"Embedder output, (B, 1500, {d_model})",
        ha="center",
        va="top",
        fontsize=12,
        color="#555555",
    )

    # ── Temporal PE block ─────────────────────────────────────────────────────
    ax.add_patch(
        FancyBboxPatch(
            (BOX_X, _y(0)),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.04",
            facecolor="#000000",
            edgecolor="none",
            zorder=2,
        )
    )
    ax.text(
        mid_x,
        _y(0) + BOX_H / 2,
        "Temporal positional encoding  \n (sinusoidal, dims [D/2 : D))",
        ha="center",
        va="center",
        fontsize=10,
        color="#ffffff",
        fontweight="medium",
    )

    # ── Encoder layer blocks ──────────────────────────────────────────────────
    radius = attn_window_size // 2
    for i in range(n_layers):
        t = i / max(n_layers - 1, 1)
        color = _lerp(t)
        label = (
            f"Layer {i}  \n   Windowed attention (±{radius} tokens)"
            if i < last_phoneme_layer
            else f"Layer {i}  \n  Full attention"
        )

        ax.add_patch(
            Rectangle(
                (BOX_X, _y(i + 1)),
                BOX_W,
                BOX_H,
                # boxstyle="round,pad=0.04",
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            mid_x,
            _y(i + 1) + BOX_H / 2,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=_text_color(color),
            fontweight="medium",
        )

    # ── Windowed / full separator ─────────────────────────────────────────────
    sep_y = _y(last_phoneme_layer) + BOX_H + GAP / 2
    ax.plot(
        [BOX_X - 0.05, BOX_X + BOX_W + 0.05],
        [sep_y, sep_y],
        color="#999999",
        linestyle="--",
        linewidth=0.9,
        zorder=1,
    )

    # ── Right-side bracket labels ─────────────────────────────────────────────
    w_mid = (_y(1) + _y(last_phoneme_layer) + BOX_H) / 2
    f_mid = (_y(last_phoneme_layer + 1) + top_y) / 2
    ax.text(
        BOX_X + BOX_W + 0.15,
        w_mid,
        f"{last_phoneme_layer} ×\nWindowed attn. layers",
        ha="left",
        va="center",
        fontsize=10,
        color="#555555",
        multialignment="center",
    )
    ax.text(
        BOX_X + BOX_W + 0.15,
        f_mid,
        f"{n_layers - last_phoneme_layer} ×\nFull attn. layers",
        ha="left",
        va="center",
        fontsize=10,
        color="#555555",
        multialignment="center",
    )

    # ── Phonemes arrow (left side, matching reference study design) ────────────────────
    # Three thick stroked lines with round caps:
    #   shaft (horizontal) + two angled wings converging at the tip.
    # Rounded corners come entirely from solid_capstyle='round'.
    ctc_y = _y(last_phoneme_layer + 1) + BOX_H / 2
    x_junc = BOX_X - 1.0  # where wings branch off the shaft
    x_tip = BOX_X - 1.25  # tip of the arrowhead (compact head: 0.35 units)
    wing_h = 0.2  # half-height of arrowhead at junction
    arrow_lw = 9  # linewidth (pts) — same for shaft and both wings
    arrow_color = "#E285A0"

    # Shaft below encoder boxes (zorder=1 < box zorder=2) so the round cap
    # at BOX_X is hidden behind the box edge rather than painted on top.
    ax.plot(
        [BOX_X, x_tip],
        [ctc_y, ctc_y],
        color=arrow_color,
        lw=arrow_lw,
        solid_capstyle="round",
        zorder=1,
    )
    for (x0, y0), (x1, y1) in [
        ((x_junc, ctc_y + wing_h), (x_tip, ctc_y)),  # top wing
        ((x_junc, ctc_y - wing_h), (x_tip, ctc_y)),  # bottom wing
    ]:
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=arrow_color,
            lw=arrow_lw,
            solid_capstyle="round",
            zorder=5,
        )

    ax.text(
        x_tip - 0.2,
        ctc_y,
        "Phonemes",
        ha="right",
        va="center",
        fontsize=11,
        color="#333333",
    )
    ax.text(
        (BOX_X + x_tip) / 2,
        ctc_y - wing_h - 0.12,
        f"Output of \n Layer {last_phoneme_layer}",
        ha="center",
        va="top",
        fontsize=10,
        color="#888888",
    )

    # ── Output arrow ──────────────────────────────────────────────────────────
    ax.annotate(
        "",
        xy=(mid_x, top_y + 0.60),
        xytext=(mid_x, top_y),
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.2),
    )
    ax.text(
        mid_x,
        top_y + 0.72,
        "To Whisper decoder",
        ha="center",
        va="bottom",
        fontsize=12,
        color="#555555",
    )

    fig.tight_layout()
    return fig


def plot_encoder_block(
    d_model: int = 384,
    n_heads: int = 6,
    d_ff: int = 1536,
    windowed: bool = True,
    attn_window_size: int = 29,
) -> Figure:
    """
    Pre-norm WhisperEncoderLayer diagram.

    Shows the two sub-layers (self-attention + FFN), LayerNorms, residual skip
    connections, and whisper-tiny dimensions.  Pass windowed=False for the
    full-attention variant (layers 4-5).
    """
    head_dim = d_model // n_heads
    radius = attn_window_size // 2

    BOX_W = 4.2
    BOX_X = 1.5
    MID_X = BOX_X + BOX_W / 2  # 3.6
    ADD_R = 0.21
    RES_X = BOX_X + BOX_W + 0.5  # vertical skip line x (right of all boxes)
    GAP = 0.28

    _dark = np.array([0.067, 0.067, 0.243])
    _purple = np.array([0.55, 0.47, 0.82])
    _blue = np.array([0.56, 0.72, 0.88])

    def _tc(c):
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        return "white" if lum < 0.55 else "#1A1A3E"

    H_NORM = 0.52
    H_ATTN = 0.78 if windowed else 0.62
    H_FFN = 0.68

    # y-coordinates (bottom-up)
    y_ln1 = 0.55
    y_attn = y_ln1 + H_NORM + GAP
    y_add1 = y_attn + H_ATTN + GAP + ADD_R   # center of ⊕1
    y_ln2 = y_add1 + ADD_R + GAP
    y_ffn = y_ln2 + H_NORM + GAP
    y_add2 = y_ffn + H_FFN + GAP + ADD_R     # center of ⊕2
    y_out = y_add2 + ADD_R + 0.55

    fig_h = y_out + 0.5
    fig, ax = plt.subplots(figsize=(9.5, fig_h * 1.3))
    ax.set_xlim(0, 7.5)
    ax.set_ylim(-0.35, fig_h)
    ax.axis("off")

    def _box(y_bot, h, label, color):
        ax.add_patch(FancyBboxPatch(
            (BOX_X, y_bot), BOX_W, h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="none", zorder=2,
        ))
        ax.text(MID_X, y_bot + h / 2, label,
                ha="center", va="center", fontsize=10,
                color=_tc(color), multialignment="center", zorder=3)

    def _add(cy):
        ax.add_patch(Circle(
            (MID_X, cy), ADD_R,
            facecolor="#999999", edgecolor="none", zorder=3,
        ))
        # Draw + with lines — avoids missing-glyph warning for ⊕ (U+2295)
        arm = ADD_R * 0.55
        ax.plot([MID_X - arm, MID_X + arm], [cy, cy],
                color="white", lw=1.8, solid_capstyle="round", zorder=4)
        ax.plot([MID_X, MID_X], [cy - arm, cy + arm],
                color="white", lw=1.8, solid_capstyle="round", zorder=4)

    def _up(y1, y2, x=MID_X):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

    def _skip(y_branch, y_dest):
        # horizontal stub → vertical run → arrowhead into right side of ⊕
        ax.plot([MID_X, RES_X], [y_branch, y_branch],
                color="#888888", lw=1.2, zorder=1)
        ax.plot([RES_X, RES_X], [y_branch, y_dest],
                color="#888888", lw=1.2, zorder=1)
        ax.annotate("", xy=(MID_X + ADD_R + 0.03, y_dest), xytext=(RES_X, y_dest),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.2))

    # ── Input ─────────────────────────────────────────────────────────────────
    ax.text(MID_X, 0.0, f"x   (B, T, {d_model})",
            ha="center", va="bottom", fontsize=11, color="#444444")
    _up(0.22, y_ln1)

    # ── Sub-layer 1: self-attention ───────────────────────────────────────────
    _box(y_ln1, H_NORM, "LayerNorm", _dark)
    _up(y_ln1 + H_NORM, y_attn)
    attn_label = f"Multi-Head Self-Attention\n{n_heads} heads × {head_dim} = {d_model}"
    if windowed:
        attn_label += f"\n[windowed ±{radius} tokens]"
    _box(y_attn, H_ATTN, attn_label, _purple)
    _up(y_attn + H_ATTN, y_add1 - ADD_R)
    _add(y_add1)
    _skip(y_ln1 - GAP / 2, y_add1)   # branch in gap between input arrow and LN1

    # ── Sub-layer 2: FFN ──────────────────────────────────────────────────────
    _up(y_add1 + ADD_R, y_ln2)
    _box(y_ln2, H_NORM, "LayerNorm", _dark)
    _up(y_ln2 + H_NORM, y_ffn)
    _box(y_ffn, H_FFN, f"Feed-Forward\n{d_model} → {d_ff} → GELU → {d_model}", _blue)
    _up(y_ffn + H_FFN, y_add2 - ADD_R)
    _add(y_add2)
    _skip(y_add1 + ADD_R + GAP / 2, y_add2)   # branch in gap between ⊕1 and LN2

    # ── Output ────────────────────────────────────────────────────────────────
    _up(y_add2 + ADD_R, y_out - 0.1)
    ax.text(MID_X, y_out, f"x'   (B, T, {d_model})",
            ha="center", va="center", fontsize=11, color="#444444")

    fig.tight_layout()
    return fig


def plot_attention_block(
    d_model: int = 384,
    n_heads: int = 6,
    windowed: bool = True,
    attn_window_size: int = 29,
) -> Figure:
    """
    Zoom into the Multi-Head Self-Attention sub-layer.
    Shows Q/K/V projections, scaled dot-product attention, and output projection.
    """
    head_dim = d_model // n_heads
    radius = attn_window_size // 2

    FIG_W = 8.5
    MID_X = FIG_W / 2

    _qkv  = np.array([0.45, 0.38, 0.72])
    _attn = np.array([0.28, 0.22, 0.52])
    _out  = np.array([0.55, 0.47, 0.82])

    def _tc(c):
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        return "white" if lum < 0.55 else "#1A1A3E"

    QKV_W   = 1.8
    QKV_H   = 0.55
    QKV_GAP = 0.28
    GROUP_W = 3 * QKV_W + 2 * QKV_GAP   # 5.96 — spans Q/K/V group and main boxes
    GROUP_X = MID_X - GROUP_W / 2

    q_cx = GROUP_X + QKV_W / 2
    k_cx = GROUP_X + QKV_W + QKV_GAP + QKV_W / 2   # == MID_X
    v_cx = GROUP_X + 2 * (QKV_W + QKV_GAP) + QKV_W / 2

    H_ATTN = 1.00 if windowed else 0.75
    H_OUT  = 0.55
    GAP    = 0.30

    y_qkv  = 0.55
    y_attn = y_qkv + QKV_H + GAP * 2.0
    y_out  = y_attn + H_ATTN + GAP
    y_top  = y_out + H_OUT + 0.6

    fig_h = y_top + 0.4
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h * 1.1))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(-0.35, fig_h)
    ax.axis("off")

    def _box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="none", zorder=2,
        ))
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=9.5,
                color=_tc(color), multialignment="center", zorder=3)

    # ── Input ─────────────────────────────────────────────────────────────────
    ax.text(MID_X, 0.0, f"x   (B, T, {d_model})", ha="center", va="bottom",
            fontsize=11, color="#444444")

    # ── Fork: stem → horizontal bus → three arrows up to Q / K / V ────────────
    bus_y = y_qkv - 0.20
    ax.plot([MID_X, MID_X], [0.22, bus_y], color="#555555", lw=1.0, zorder=1)
    ax.plot([q_cx, v_cx],   [bus_y, bus_y], color="#555555", lw=1.0, zorder=1)
    for cx in (q_cx, k_cx, v_cx):
        ax.annotate("", xy=(cx, y_qkv), xytext=(cx, bus_y),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

    # ── Q / K / V projection boxes ────────────────────────────────────────────
    for cx, name in [(q_cx, "Q"), (k_cx, "K"), (v_cx, "V")]:
        _box(cx - QKV_W / 2, y_qkv, QKV_W, QKV_H,
             f"{name}\n{d_model} -> {n_heads}x{head_dim}", _qkv)

    # ── Fan: short stub up → diagonal converge → arrow into attention ──────────
    stub_y = y_qkv + QKV_H + 0.08
    meet_y = y_attn - 0.08
    for cx in (q_cx, k_cx, v_cx):
        ax.plot([cx, cx],      [y_qkv + QKV_H, stub_y], color="#555555", lw=1.0, zorder=1)
        ax.plot([cx, MID_X],   [stub_y, meet_y],         color="#555555", lw=1.0, zorder=1)
    ax.annotate("", xy=(MID_X, y_attn), xytext=(MID_X, meet_y),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

    # ── Attention box ──────────────────────────────────────────────────────────
    attn_label = f"Scaled Dot-Product Attention  ({n_heads} heads x {head_dim})\n"
    attn_label += f"QK^T / sqrt({head_dim})  ->  softmax  ->  . V"
    if windowed:
        attn_label += f"\n[windowed +/-{radius} tokens]"
    _box(GROUP_X, y_attn, GROUP_W, H_ATTN, attn_label, _attn)

    # ── Output projection ──────────────────────────────────────────────────────
    ax.annotate("", xy=(MID_X, y_out), xytext=(MID_X, y_attn + H_ATTN),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
    _box(GROUP_X, y_out, GROUP_W, H_OUT,
         f"Output projection   ({d_model} -> {d_model})", _out)

    # ── Output ─────────────────────────────────────────────────────────────────
    ax.annotate("", xy=(MID_X, y_top - 0.12), xytext=(MID_X, y_out + H_OUT),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
    ax.text(MID_X, y_top, f"x'   (B, T, {d_model})", ha="center", va="center",
            fontsize=11, color="#444444")

    fig.tight_layout()
    return fig


def plot_attention_mechanism(
    d_model: int = 384,
    n_heads: int = 6,
    windowed: bool = True,
    attn_window_size: int = 29,
) -> Figure:
    """
    Attention diagram with proportional tensor shapes.

    Rectangle widths reflect actual d_k : d_model ratio (d_k = d_model / n_heads).
    Shows the full single-head computation: Q/K/V projections → score matrix →
    head output → concat (n_heads stripes) → W_O → (T, d_model) ready for FFN.
    """
    head_dim = d_model // n_heads

    # T = width, d_model/d_k = height; W_T/H_DM ≈ 3 approximates T/d_model (1500/384)
    W_T   = 1.20                         # width  = T (sequence dimension)
    H_DM  = 0.65                         # height = d_model
    H_DK  = H_DM * head_dim / d_model   # height = d_k  (= H_DM / n_heads)
    W_TT  = W_T                          # score matrix T×T is square
    PH    = 0.40                         # projection-box height (indicator only)
    PW    = 0.22                         # projection-box width
    GAP_H = 0.38
    GAP_V = 0.65                         # vertical gap between Q/K/V rows

    _in  = np.array([0.78, 0.78, 0.80])
    _q   = np.array([0.42, 0.58, 0.92])
    _k   = np.array([0.60, 0.82, 0.42])
    _v   = np.array([0.75, 0.55, 0.88])
    _out = np.array([0.90, 0.62, 0.72])
    _dm  = np.array([0.56, 0.72, 0.88])

    def _tc(c):
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        return "white" if lum < 0.58 else "#1A1A3E"

    # Y centers: Q at top, K middle, V at bottom
    y_v = H_DK / 2 + 0.80
    y_k = y_v + H_DK + GAP_V
    y_q = y_k + H_DK + GAP_V

    # X positions left to right
    x_in    = 0.50
    x_proj  = x_in    + W_T   + GAP_H
    x_qkv   = x_proj  + PW    + 0.18
    x_score = x_qkv   + W_T   + GAP_H
    x_head  = x_score + W_TT  + GAP_H
    x_cat   = x_head  + W_T   + GAP_H
    x_wo    = x_cat   + W_T   + 0.25
    x_out   = x_wo    + PW    + 0.18

    FIG_W = x_out + W_T + 0.50
    FIG_H = y_q + W_TT / 2 + 0.80

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    def _rect(x, y_ctr, w, h, color):
        ax.add_patch(FancyBboxPatch(
            (x, y_ctr - h / 2), w, h,
            boxstyle="round,pad=0.03",
            facecolor=tuple(color), edgecolor="none", zorder=2,
        ))

    def _lbl(x, w, y_bot, text, **kw):
        kw.setdefault("color", "#666666")
        ax.text(x + w / 2, y_bot - 0.11, text,
                ha="center", va="top", fontsize=7.5, zorder=3, **kw)

    def _proj_box(x, y_ctr, label, color):
        ax.add_patch(FancyBboxPatch(
            (x, y_ctr - PH / 2), PW, PH,
            boxstyle="round,pad=0.03",
            facecolor=tuple(color), edgecolor="none", zorder=3,
        ))
        ax.text(x + PW / 2, y_ctr, label,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=_tc(color), zorder=4)

    def _arr(x1, y1, x2, y2, color=(0.45, 0.45, 0.45), lw=1.35, rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    zorder=5,
                    arrowprops=dict(
                        arrowstyle="->", color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}",
                    ))

    def _score_heatmap(x0, y_ctr, n=7):
        cw = W_TT / n
        ch = W_TT / n
        rng = np.random.default_rng(99)
        y0 = y_ctr - W_TT / 2
        for r in range(n):
            for c in range(n):
                val = float(rng.random() * 0.80 + 0.10)
                face = plt.colormaps["RdYlBu_r"](val)
                ax.add_patch(Rectangle(
                    (x0 + c * cw + 0.007, y0 + r * ch + 0.007),
                    cw - 0.014, ch - 0.014,
                    facecolor=face, edgecolor="none", zorder=2,
                ))

    # ── Input block (landscape, centered at middle row = y_k) ────────────────
    _rect(x_in, y_k, W_T, H_DM, _in)
    ax.text(x_in + W_T / 2, y_k, "x",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="#333333", zorder=3)
    _lbl(x_in, W_T, y_k - H_DM / 2, f"(T, {d_model})")

    # Fan out: right edge of input → three projection boxes
    for y_row, c in [(y_q, _q), (y_k, _k), (y_v, _v)]:
        _arr(x_in + W_T, y_k, x_proj, y_row, color=tuple(c * 0.70), lw=1.20)

    # ── Projection boxes and Q/K/V flat bars ─────────────────────────────────
    for y_row, wlabel, c in [(y_q, "W_Q", _q), (y_k, "W_K", _k), (y_v, "W_V", _v)]:
        _proj_box(x_proj, y_row, wlabel, c * 0.82)
        _arr(x_proj + PW, y_row, x_qkv, y_row, color=tuple(c * 0.70), lw=1.20)
        _rect(x_qkv, y_row, W_T, H_DK, c)

    _lbl(x_qkv, W_T, y_v - H_DK / 2, f"(T, {head_dim})")

    # ── Score matrix Q·K^T ────────────────────────────────────────────────────
    _score_heatmap(x_score, y_q, n=7)
    ax.text(x_score + W_TT / 2, y_q + W_TT / 2 + 0.10,
            f"softmax( Q·K^T / sqrt({head_dim}) )",
            ha="center", va="bottom", fontsize=8, color="#444444")

    # Q → score (horizontal into left edge)
    _arr(x_qkv + W_T, y_q, x_score - 0.02, y_q,
         color=tuple(_q * 0.70), lw=1.20)

    # K → score: L-shape — right from K bottom-centre, then up into score bottom
    k_bot_y   = y_k - H_DK / 2
    sc_feed_x = x_score + W_TT * 0.38
    ax.plot([x_qkv + W_T * 0.5, sc_feed_x], [k_bot_y, k_bot_y],
            color=tuple(_k * 0.70), lw=1.20, solid_capstyle="round", zorder=1)
    _arr(sc_feed_x, k_bot_y, sc_feed_x, y_q - W_TT / 2 + 0.04,
         color=tuple(_k * 0.70), lw=1.20)

    # ── Head output ────────────────────────────────────────────────────────────
    _rect(x_head, y_q, W_T, H_DK, _out)
    _arr(x_score + W_TT + 0.02, y_q, x_head, y_q,
         color=(0.55, 0.35, 0.55), lw=1.35)
    _lbl(x_head, W_T, y_q - H_DK / 2, f"(T, {head_dim})")

    # V → head output: L-shape — right from V top-centre, then up into head bottom
    v_top_y   = y_v + H_DK / 2
    ho_feed_x = x_head + W_T * 0.50
    ax.plot([x_qkv + W_T * 0.5, ho_feed_x], [v_top_y, v_top_y],
            color=tuple(_v * 0.70), lw=1.20, solid_capstyle="round", zorder=1)
    _arr(ho_feed_x, v_top_y, ho_feed_x, y_q - H_DK / 2,
         color=tuple(_v * 0.70), lw=1.20)

    # ── Concat: n_heads stripes = d_model ─────────────────────────────────────
    stripe_w = W_T / n_heads
    for hi in range(n_heads):
        alpha = 0.45 + 0.55 * hi / max(n_heads - 1, 1)
        ax.add_patch(Rectangle(
            (x_cat + hi * stripe_w, y_q - H_DM / 2), stripe_w, H_DM,
            facecolor=(*_out, alpha), edgecolor="none", zorder=2,
        ))
    ax.add_patch(Rectangle(
        (x_cat, y_q - H_DM / 2), W_T, H_DM,
        facecolor="none", edgecolor="#999999", linewidth=0.8, zorder=4,
    ))
    ax.text(x_cat + W_T / 2, y_q, f"concat\n({n_heads}x)",
            ha="center", va="center", fontsize=8.5, color="#333333", zorder=5)
    _arr(x_head + W_T, y_q, x_cat, y_q, color=tuple(_out * 0.70), lw=1.35)
    _lbl(x_cat, W_T, y_q - H_DM / 2, f"(T, {n_heads}x{head_dim}={d_model})")

    # ── W_O projection ─────────────────────────────────────────────────────────
    _proj_box(x_wo, y_q, "W_O", _dm * 0.85)
    _arr(x_cat + W_T, y_q, x_wo, y_q, color=tuple(_dm * 0.70), lw=1.20)

    # ── Output: (T, d_model) ready for FFN ───────────────────────────────────
    _rect(x_out, y_q, W_T, H_DM, _dm)
    ax.text(x_out + W_T / 2, y_q, "-> FFN",
            ha="center", va="center", fontsize=9, color=_tc(_dm), zorder=3)
    _arr(x_wo + PW, y_q, x_out, y_q, color=tuple(_dm * 0.70), lw=1.20)
    _lbl(x_out, W_T, y_q - H_DM / 2, f"(T, {d_model})")

    fig.tight_layout()
    return fig


def plot_ffn_block(
    d_model: int = 384,
    d_ff: int = 1536,
) -> Figure:
    """
    Zoom into the Feed-Forward sub-layer.
    Proportional-width dimension bars at each junction visualise the 4x expansion.
    """
    BOX_W = 4.2
    BOX_X = 1.5
    MID_X = BOX_X + BOX_W / 2
    BAR_H = 0.14
    ANN_X = BOX_X + BOX_W + 0.18   # fixed x for right-side dimension labels
    GAP   = 0.22
    H_LIN = 0.58
    H_GEL = 0.52

    _lin1 = np.array([0.40, 0.60, 0.82])
    _gelu = np.array([0.25, 0.45, 0.68])
    _lin2 = np.array([0.56, 0.72, 0.88])
    _bar  = np.array([0.70, 0.83, 0.95])

    def _tc(c):
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        return "white" if lum < 0.55 else "#1A1A3E"

    # y-coordinates (bottom-up): dim_bar and op_box alternating
    y_bar0 = 0.55
    y_lin1 = y_bar0 + BAR_H + GAP
    y_bar1 = y_lin1 + H_LIN + GAP
    y_gelu = y_bar1 + BAR_H + GAP
    y_bar2 = y_gelu + H_GEL + GAP
    y_lin2 = y_bar2 + BAR_H + GAP
    y_bar3 = y_lin2 + H_LIN + GAP
    y_top  = y_bar3 + BAR_H + 0.55

    fig_h = y_top + 0.4
    fig, ax = plt.subplots(figsize=(9.5, fig_h * 1.3))
    ax.set_xlim(0, 7.5)
    ax.set_ylim(-0.35, fig_h)
    ax.axis("off")

    def _box(y, h, label, color):
        ax.add_patch(FancyBboxPatch(
            (BOX_X, y), BOX_W, h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="none", zorder=2,
        ))
        ax.text(MID_X, y + h / 2, label,
                ha="center", va="center", fontsize=10,
                color=_tc(color), multialignment="center", zorder=3)

    def _dim_bar(y, dim):
        """Width-proportional bar; right-side label shows the dimension."""
        w = dim / max(d_model, d_ff) * BOX_W
        ax.add_patch(FancyBboxPatch(
            (MID_X - w / 2, y), w, BAR_H,
            boxstyle="round,pad=0.02",
            facecolor=_bar, edgecolor="none", zorder=2,
        ))
        ax.text(ANN_X, y + BAR_H / 2, f"{dim}",
                ha="left", va="center", fontsize=8.5, color="#4A6080", zorder=3)

    def _up(y1, y2):
        ax.annotate("", xy=(MID_X, y2), xytext=(MID_X, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

    # ── Input ─────────────────────────────────────────────────────────────────
    ax.text(MID_X, 0.0, f"x   (B, T, {d_model})", ha="center", va="bottom",
            fontsize=11, color="#444444")
    _up(0.22, y_bar0)
    _dim_bar(y_bar0, d_model)

    # ── Linear 384 → 1536 ─────────────────────────────────────────────────────
    _up(y_bar0 + BAR_H, y_lin1)
    _box(y_lin1, H_LIN, f"Linear\n{d_model} -> {d_ff}", _lin1)
    _up(y_lin1 + H_LIN, y_bar1)
    _dim_bar(y_bar1, d_ff)

    # ── GELU ──────────────────────────────────────────────────────────────────
    _up(y_bar1 + BAR_H, y_gelu)
    _box(y_gelu, H_GEL, "GELU", _gelu)
    _up(y_gelu + H_GEL, y_bar2)
    _dim_bar(y_bar2, d_ff)

    # ── Linear 1536 → 384 ─────────────────────────────────────────────────────
    _up(y_bar2 + BAR_H, y_lin2)
    _box(y_lin2, H_LIN, f"Linear\n{d_ff} -> {d_model}", _lin2)
    _up(y_lin2 + H_LIN, y_bar3)
    _dim_bar(y_bar3, d_model)

    # ── Output ────────────────────────────────────────────────────────────────
    _up(y_bar3 + BAR_H, y_top - 0.12)
    ax.text(MID_X, y_top, f"x'   (B, T, {d_model})", ha="center", va="center",
            fontsize=11, color="#444444")

    fig.tight_layout()
    return fig


def plot_block_section(
    d_model: int = 384,
    n_heads: int = 6,
    d_ff: int = 1536,
    attn_window_size: int = 29,
    dpi: int = 120,
) -> Figure:
    """
    Three-panel side-by-side overview of one encoder block.
    Left: block structure. Centre: attention mechanism. Right: FFN.
    Each panel is rendered at `dpi` and stitched horizontally with bottom-aligned
    white padding so all panels share the same height.
    """
    import io
    import matplotlib.image as _mpimg

    panels = [
        plot_encoder_block(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            windowed=False, attn_window_size=attn_window_size,
        ),
        plot_attention_mechanism(
            d_model=d_model, n_heads=n_heads,
            windowed=False, attn_window_size=attn_window_size,
        ),
        plot_ffn_block(d_model=d_model, d_ff=d_ff),
    ]

    arrays = []
    for f in panels:
        buf = io.BytesIO()
        f.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                  facecolor="white", edgecolor="none")
        buf.seek(0)
        arrays.append(_mpimg.imread(buf))
        plt.close(f)

    # Pad shorter panels to the tallest height with white rows at the bottom
    max_h = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        h, w, c = a.shape
        if h < max_h:
            pad = np.ones((max_h - h, w, c), dtype=a.dtype)
            a = np.concatenate([a, pad], axis=0)
        padded.append(a)

    combined = np.concatenate(padded, axis=1)

    fig, ax = plt.subplots(figsize=(combined.shape[1] / dpi, combined.shape[0] / dpi))
    ax.imshow(combined, interpolation="bilinear")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def plot_hidden_state_norms(
    encoder_output: BaseModelOutput, sample_idx: int = 0
) -> Figure:
    """
    L2 norm per position for every layer in the encoder output.

    encoder_output.hidden_states is a tuple of (B, T, D) tensors:
      index 0  = encoder input (after pos encoding)
      index k  = output of layer k-1
    """
    if encoder_output.hidden_states is None:
        raise ValueError(
            "encoder_output.hidden_states is None; "
            "BWEncoder must populate all_hidden in its forward() return value"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    for layer_idx, h in enumerate(encoder_output.hidden_states):
        norms = h[sample_idx].detach().cpu().float().norm(dim=-1).numpy()
        label = "input" if layer_idx == 0 else f"layer {layer_idx}"
        ax.plot(norms, alpha=0.7, linewidth=0.8, label=label)

    ax.set_xlabel("Position")
    ax.set_ylabel("L2 norm")
    ax.set_title(f"Hidden state norms per layer — sample {sample_idx}")
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_windowed_attention_mask(bwencoder) -> Figure:
    """
    Render the windowed attention mask for a compact sequence so the structure is legible.
    Uses 4× the window radius, capped at 256 positions.
    """
    seq_len = min(bwencoder.attn_window_size * 4, 256)
    mask = bwencoder._windowed_mask(seq_len, device=torch.device("cpu"))
    mask_2d = mask.squeeze().numpy()  # (seq_len, seq_len)

    # Map -1e9 → -1, 0 → 0 for a clean two-value colourmap
    display = (mask_2d < -1).astype(float) * -1.0 + (mask_2d >= -1).astype(float) * 0.0
    display = (mask_2d < -0.5).astype(float)  # 1 = blocked, 0 = allowed

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        display, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1
    )
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(
        f"Windowed attention mask  (window={bwencoder.attn_window_size}, seq={seq_len})\n"
        "black = blocked, white = attended"
    )
    fig.tight_layout()
    return fig
