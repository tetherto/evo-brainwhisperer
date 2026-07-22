"""
Manim animations for the Conv1d pipeline in BWEmbedder.

Two animations:
  - render_conv1_mechanics_gif: per-position channel-reduction operation
    (C=512, k=7) patch × (d=384 filters of shape C×k) → one output column
  - render_conv1_sliding_gif: kernel sweeps across T time steps
    full input heatmap → kernel window slides → output heatmap fills in
"""

import glob
import os
import shutil
import tempfile

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage

# ── Palette (matches embeddings.py + theme.json) ─────────────────────────────
_PURPLE = "#5237B7"
_BLUE = "#39ADFF"
_PINK = "#DE5698"  # brand accent (slot 2 in theme.json palette)
_DARK = "#333333"
_GRAY = "#888888"
_OFFWHITE = "#F4F0FF"
_WHITE = "#FFFFFF"

_SIGNAL_CMAP = LinearSegmentedColormap.from_list(
    "bw_signal", [_PURPLE, _OFFWHITE, _BLUE], N=256
)
_WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "bw_weight", [_PINK, _OFFWHITE, _PURPLE], N=256
)
_OUT_CMAP = LinearSegmentedColormap.from_list(
    "bw_out", [_BLUE, _OFFWHITE, _PURPLE], N=256
)

_FONT = "Sharp Grotesk"  # registered via manimpango; used by all manim Text() calls

_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts", "sharp_grotesk")
_FONT_REGISTERED = False  # guard against duplicate registration calls


def _register_font() -> None:
    """Register Sharp Grotesk with Pango (manimpango) for the current process."""
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    try:
        from manimpango import register_font

        for weight in ("Book20", "Medium20", "SemiBold20"):
            path = os.path.join(_FONT_DIR, f"SharpGrotesk-{weight}.otf")
            if os.path.exists(path):
                register_font(path)
        _FONT_REGISTERED = True
    except Exception:
        pass  # falls back gracefully to system sans-serif


# ── Image utilities ───────────────────────────────────────────────────────────


def _arr_to_pil(arr: np.ndarray, cmap, vmin=None, vmax=None) -> PILImage.Image:
    norm = mcolors.Normalize(
        vmin=arr.min() if vmin is None else vmin,
        vmax=arr.max() if vmax is None else vmax,
    )
    rgba = cmap(norm(arr.astype(np.float32)))
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return PILImage.fromarray(rgb)


def _save_img(arr: np.ndarray, cmap, pix_h: int, pix_w: int, **kw) -> str:
    """Save array as a PNG resized to (pix_h × pix_w) pixels; return temp path."""
    img = _arr_to_pil(arr, cmap, **kw)
    img = img.resize((pix_w, pix_h), PILImage.LANCZOS)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    return path


# ── Scene: per-position mechanics ─────────────────────────────────────────────


class _ConvChannelScene:
    """
    Not a subclass of Scene — instantiated fresh inside render_conv1_mechanics_gif
    so that manim config is applied before Scene.__init__ runs.
    """

    def __init__(self, patch_arr, C, Dm, K1):
        self._patch = patch_arr  # (C, K1)
        self._C = C
        self._Dm = Dm
        self._K1 = K1
        self._tmp: list[str] = []

    # called by render_conv1_mechanics_gif after importing and configuring manim
    def build_and_render(self):
        _register_font()  # must happen before any Text() is created

        from manim import (
            DOWN,
            LEFT,
            ORIGIN,
            RIGHT,
            UP,
            WHITE,
            Arrow,
            Brace,
            Create,
            FadeIn,
            FadeOut,
            ImageMobject,
            Rectangle,
            Scene,
            Text,
            VGroup,
            Write,
            config,
        )

        scene_self = self
        tmp = self._tmp
        patch_arr = self._patch
        C, Dm, K1 = self._C, self._Dm, self._K1

        # ── Geometry ───────────────────────────────────────────────────────
        BLOCK_H = 3.2  # scene-unit height of a (C, k) block
        BLOCK_W = 1.2  # scene-unit width  (schematic; labeled as k=K1)
        COL_H = BLOCK_H * Dm / C  # output column height ∝ d_model  (≈2.4)
        COL_W = 0.35

        # pixel dimensions that match the scene-unit aspect ratios
        PIX_H = 256
        PIX_W = round(PIX_H * BLOCK_W / BLOCK_H)  # ≈96
        CPIX_W = round(PIX_H * COL_W / COL_H)  # ≈37 → 32

        # x positions (scene centre = 0, frame ≈ ±7.1)
        PX = -5.0  # input patch centre
        FX = -2.0  # filter block centre
        SX = 2.1  # single scalar centre
        OX = 4.8  # output column centre

        # ── Temp images ────────────────────────────────────────────────────
        patch_file = _save_img(patch_arr[:, :K1], _SIGNAL_CMAP, PIX_H, PIX_W)
        tmp.append(patch_file)

        rng = np.random.default_rng(42)
        filter_arr = rng.standard_normal((C, K1)).astype(np.float32)
        filter_file = _save_img(filter_arr, _WEIGHT_CMAP, PIX_H, PIX_W)
        tmp.append(filter_file)

        W_flat = rng.standard_normal((Dm, C * K1)).astype(np.float32)
        v_flat = patch_arr.flatten().astype(np.float32)
        out_col = (W_flat @ v_flat).reshape(Dm, 1).astype(np.float32)
        col_file = _save_img(out_col, _SIGNAL_CMAP, PIX_H, max(CPIX_W, 8))
        tmp.append(col_file)

        class _Scene(Scene):
            def construct(inner):
                inner.camera.background_color = _WHITE

                # ── Input patch mobjects ───────────────────────────────────
                p_img = ImageMobject(patch_file)
                p_img.set_height(BLOCK_H)
                p_img.move_to([PX, 0, 0])
                p_bord = Rectangle(
                    width=BLOCK_W,
                    height=BLOCK_H,
                    stroke_color=_PURPLE,
                    stroke_width=2.5,
                    fill_opacity=0,
                ).move_to([PX, 0, 0])
                p_title = Text("Input patch", color=_DARK, font=_FONT, font_size=22)
                p_title.next_to(p_bord, UP, buff=0.25)

                p_brace = Brace(p_bord, LEFT, color=_DARK, buff=0.05)
                p_brace_lbl = Text(f"C={C}", color=_PURPLE, font=_FONT, font_size=18)
                p_brace_lbl.next_to(p_brace, LEFT, buff=0.1)

                p_k_lbl = Text(f"k={K1}", color=_DARK, font=_FONT, font_size=18)
                p_k_lbl.next_to(p_bord, DOWN, buff=0.2)

                # ── Multiply symbol ────────────────────────────────────────
                mul = Text("⊙", color=_DARK, font=_FONT, font_size=54)
                mul.move_to([(PX + FX) / 2, 0, 0])

                # ── Filter mobjects ────────────────────────────────────────
                f_img = ImageMobject(filter_file)
                f_img.set_height(BLOCK_H)
                f_img.move_to([FX, 0, 0])
                f_bord = Rectangle(
                    width=BLOCK_W,
                    height=BLOCK_H,
                    stroke_color=_PINK,
                    stroke_width=2.5,
                    fill_opacity=0,
                ).move_to([FX, 0, 0])
                f_title = Text(
                    f"Filter  (one of  d={Dm})", color=_DARK, font=_FONT, font_size=22
                )
                f_title.next_to(f_bord, UP, buff=0.25)
                f_shape = Text(
                    f"same shape: (C={C}, k={K1})",
                    color=_GRAY,
                    font=_FONT,
                    font_size=16,
                )
                f_shape.next_to(f_bord, DOWN, buff=0.2)

                # ── Sum arrow + scalar ─────────────────────────────────────
                arr_start = np.array([FX + BLOCK_W / 2 + 0.12, 0, 0])
                arr_end = np.array([SX - 0.22, 0, 0])
                s_arrow = Arrow(
                    arr_start,
                    arr_end,
                    color=_DARK,
                    stroke_width=3,
                    max_tip_length_to_length_ratio=0.18,
                )
                s_tex = Text("Σ(c,k)", color=_DARK, font=_FONT, font_size=22)
                s_tex.next_to(s_arrow, UP, buff=0.1)

                scalar = Rectangle(
                    width=0.32,
                    height=0.32,
                    fill_color=_BLUE,
                    fill_opacity=0.9,
                    stroke_width=0,
                ).move_to([SX, 0, 0])
                # label above scalar so the rightward arrow to the column doesn't bisect it
                s_lbl = Text("1 scalar", color=_DARK, font=_FONT, font_size=18)
                s_lbl.next_to(scalar, UP, buff=0.15)

                # ── "× d=384" text and arrow to output column ──────────────
                x_text = Text(
                    f"×  d={Dm}  filters", color=_DARK, font=_FONT, font_size=22
                )
                x_text.next_to(scalar, DOWN, buff=0.55)

                o_arr_start = np.array([SX + 0.22, 0, 0])
                o_arr_end = np.array([OX - COL_W / 2 - 0.18, 0, 0])
                o_arrow = Arrow(
                    o_arr_start,
                    o_arr_end,
                    color=_DARK,
                    stroke_width=3,
                    max_tip_length_to_length_ratio=0.18,
                )

                # ── Output column ──────────────────────────────────────────
                c_img = ImageMobject(col_file)
                c_img.set_height(COL_H)
                c_img.move_to([OX, 0, 0])
                c_bord = Rectangle(
                    width=COL_W,
                    height=COL_H,
                    stroke_color=_BLUE,
                    stroke_width=2.5,
                    fill_opacity=0,
                ).move_to([OX, 0, 0])
                c_title = Text(
                    "Output column\n(at time t)", color=_DARK, font=_FONT, font_size=20
                )
                c_title.next_to(c_bord, UP, buff=0.25)

                c_brace = Brace(c_bord, RIGHT, color=_DARK, buff=0.05)
                c_brace_lbl = Text(f"d={Dm}", color=_BLUE, font=_FONT, font_size=18)
                c_brace_lbl.next_to(c_brace, RIGHT, buff=0.1)

                # ── Animation sequence ─────────────────────────────────────
                # 1. Input patch
                inner.play(FadeIn(p_img), Create(p_bord), run_time=0.55)
                inner.play(Write(p_title), run_time=0.35)
                inner.play(
                    Create(p_brace), Write(p_brace_lbl), Write(p_k_lbl), run_time=0.45
                )
                inner.wait(0.3)

                # 2. Filter (same shape)
                inner.play(Write(mul), run_time=0.25)
                inner.play(FadeIn(f_img), Create(f_bord), run_time=0.55)
                inner.play(Write(f_title), Write(f_shape), run_time=0.4)
                inner.wait(0.25)

                # 3. Element-wise multiply flash
                fl_a = Rectangle(
                    width=BLOCK_W,
                    height=BLOCK_H,
                    fill_color=_BLUE,
                    fill_opacity=0.28,
                    stroke_width=0,
                ).move_to([PX, 0, 0])
                fl_b = Rectangle(
                    width=BLOCK_W,
                    height=BLOCK_H,
                    fill_color=_BLUE,
                    fill_opacity=0.28,
                    stroke_width=0,
                ).move_to([FX, 0, 0])
                inner.play(FadeIn(fl_a), FadeIn(fl_b), run_time=0.2)
                inner.play(FadeOut(fl_a), FadeOut(fl_b), run_time=0.25)

                # 4. Sum → scalar
                inner.play(Create(s_arrow), Write(s_tex), run_time=0.45)
                inner.play(FadeIn(scalar), Write(s_lbl), run_time=0.35)
                inner.wait(0.4)

                # 5. ×384 → output column
                inner.play(Write(x_text), run_time=0.35)
                inner.play(Create(o_arrow), run_time=0.35)
                inner.play(FadeIn(c_img), Create(c_bord), Write(c_title), run_time=0.5)
                inner.play(Create(c_brace), Write(c_brace_lbl), run_time=0.35)
                inner.wait(1.8)

        scene = _Scene()
        try:
            scene.render()
        finally:
            for f in tmp:
                try:
                    os.unlink(f)
                except OSError:
                    pass


# ── Scene: sliding kernel across time ─────────────────────────────────────────


class _ConvSlidingScene:
    def __init__(self, sig_win, conv1_win, C, Dm, K1, T_SHOW):
        self._sig = sig_win  # (C, T_SHOW)
        self._out = conv1_win  # (Dm, T_SHOW)
        self._C = C
        self._Dm = Dm
        self._K1 = K1
        self._T = T_SHOW
        self._tmp: list[str] = []

    def build_and_render(self):
        _register_font()  # must happen before any Text() is created

        from manim import (
            DOWN,
            LEFT,
            ORIGIN,
            RIGHT,
            UP,
            Brace,
            Create,
            FadeIn,
            FadeOut,
            ImageMobject,
            Rectangle,
            Scene,
            Text,
            Write,
            config,
            linear,
        )

        tmp = self._tmp
        sig_win, conv1_win = self._sig, self._out
        C, Dm, K1, T = self._C, self._Dm, self._K1, self._T

        # ── Geometry ───────────────────────────────────────────────────────
        MAT_W = 4.8  # full matrix width in scene units
        IN_H = 3.2  # input height ∝ C
        OUT_H = IN_H * Dm / C  # output height ∝ d_model  (≈2.4)
        KW = MAT_W * K1 / T  # kernel window width (T=20, K1=7 → 1.68)

        IN_CX = -2.8  # input matrix x-centre (shifted right for label room)
        OUT_CX = 3.5  # output matrix x-centre

        PIX_H, PIX_W = 256, 512

        sig_file = _save_img(sig_win, _SIGNAL_CMAP, PIX_H, PIX_W)
        out_file = _save_img(conv1_win, _OUT_CMAP, PIX_H, PIX_W)
        tmp.extend([sig_file, out_file])

        class _Scene(Scene):
            def construct(inner):
                inner.camera.background_color = _WHITE

                # ── Input matrix ────────────────────────────────────────────
                in_img = ImageMobject(sig_file)
                in_img.set_height(IN_H)
                in_img.stretch_to_fit_width(MAT_W)
                in_img.move_to([IN_CX, 0, 0])
                in_bord = Rectangle(
                    width=MAT_W,
                    height=IN_H,
                    stroke_color=_PURPLE,
                    stroke_width=2.5,
                    fill_opacity=0,
                ).move_to([IN_CX, 0, 0])
                in_title = Text(
                    f"Input  (C={C})", color=_DARK, font=_FONT, font_size=21
                )
                in_title.next_to(in_bord, UP, buff=0.22)
                in_t_lbl = Text(
                    f"T={T} time steps  (zoomed)", color=_GRAY, font=_FONT, font_size=16
                )
                in_t_lbl.next_to(in_bord, DOWN, buff=0.18)

                in_c_brace = Brace(in_bord, LEFT, color=_DARK, buff=0.05)
                in_c_label = Text(f"C={C}", color=_PURPLE, font=_FONT, font_size=17)
                in_c_label.next_to(in_c_brace, LEFT, buff=0.1)

                # ── Output matrix ────────────────────────────────────────────
                out_img = ImageMobject(out_file)
                out_img.set_height(OUT_H)
                out_img.stretch_to_fit_width(MAT_W)
                out_img.move_to([OUT_CX, 0, 0])
                out_bord = Rectangle(
                    width=MAT_W,
                    height=OUT_H,
                    stroke_color=_BLUE,
                    stroke_width=2.5,
                    fill_opacity=0,
                ).move_to([OUT_CX, 0, 0])
                out_title = Text(
                    f"After Conv1+GELU  (d={Dm})", color=_DARK, font=_FONT, font_size=21
                )
                out_title.next_to(out_bord, UP, buff=0.22)

                out_d_brace = Brace(out_bord, LEFT, color=_DARK, buff=0.05)
                out_d_label = Text(f"d={Dm}", color=_BLUE, font=_FONT, font_size=17)
                out_d_label.next_to(out_d_brace, LEFT, buff=0.1)

                # ── Kernel window (slides on input) ──────────────────────────
                kwin = Rectangle(
                    width=KW,
                    height=IN_H,
                    fill_color=_PINK,
                    fill_opacity=0.22,
                    stroke_color=_PINK,
                    stroke_width=2.5,
                )
                # start: kernel left-aligned at input left edge
                kernel_start_x = IN_CX - MAT_W / 2 + KW / 2
                kernel_end_x = IN_CX + MAT_W / 2 - KW / 2
                kwin.move_to([kernel_start_x, 0, 0])

                k_lbl = Text(f"k={K1}", color=_PINK, font=_FONT, font_size=16)
                # buff=0.55 keeps it clear of the "T=20 time steps" label below the matrix
                k_lbl.add_updater(lambda m: m.next_to(kwin, DOWN, buff=0.55))

                # ── Output reveal mask ────────────────────────────────────────
                # Covers the output image; shrinks leftward as kernel advances.
                mask = Rectangle(
                    width=MAT_W,
                    height=OUT_H + 0.05,
                    fill_color=_WHITE,
                    fill_opacity=1.0,
                    stroke_width=0,
                )
                mask.move_to([OUT_CX, 0, 0])

                def _update_mask(m):
                    # fraction of input scanned = kernel-centre progress
                    progress = (kwin.get_center()[0] - kernel_start_x) / (
                        kernel_end_x - kernel_start_x
                    )
                    progress = max(0.0, min(1.0, progress))
                    unrevealed = 1.0 - progress
                    w = MAT_W * unrevealed
                    if w > 0.02:
                        m.stretch_to_fit_width(w)
                        m.move_to([OUT_CX + MAT_W / 2 - w / 2, 0, 0])
                    else:
                        m.set_opacity(0)

                mask.add_updater(_update_mask)

                # ── Animate ──────────────────────────────────────────────────
                # 1. Fade in both matrices
                inner.play(
                    FadeIn(in_img),
                    Create(in_bord),
                    Write(in_title),
                    run_time=0.6,
                )
                inner.play(
                    Create(in_c_brace),
                    Write(in_c_label),
                    Write(in_t_lbl),
                    run_time=0.4,
                )

                # 2. Fade in output (hidden behind mask)
                inner.play(
                    FadeIn(out_img),
                    Create(out_bord),
                    Write(out_title),
                    Create(out_d_brace),
                    Write(out_d_label),
                    run_time=0.6,
                )
                inner.add(mask)  # mask goes on top of output image
                inner.wait(0.2)

                # 3. Kernel appears at left edge
                inner.play(Create(kwin), run_time=0.4)
                inner.add(k_lbl)
                inner.wait(0.3)

                # 4. Kernel slides to the right; mask reveals output
                inner.play(
                    kwin.animate.move_to([kernel_end_x, 0, 0]),
                    run_time=3.2,
                    rate_func=linear,
                )
                inner.wait(1.2)

        scene = _Scene()
        try:
            scene.render()
        finally:
            for f in tmp:
                try:
                    os.unlink(f)
                except OSError:
                    pass


# ── Public render functions ────────────────────────────────────────────────────


def _mp4_to_gif(mp4_path: str, gif_path: str, fps: int) -> None:
    """
    Two-pass ffmpeg GIF conversion: palettegen builds an optimal 256-color
    palette from the video content, paletteuse applies it.  This avoids PIL's
    quantisation which maps dark grays to random hues when a white background
    is present alongside colourful heatmaps.
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        palette = os.path.join(tmp, "palette.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                mp4_path,
                "-vf",
                f"fps={fps},palettegen=max_colors=256:reserve_transparent=0:stats_mode=full",
                palette,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                mp4_path,
                "-i",
                palette,
                "-filter_complex",
                f"[0:v]fps={fps}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3",
                gif_path,
            ],
            check=True,
            capture_output=True,
        )


def _render_scene_to_gif(
    scene_build_fn,
    output_path: str,
    fps: int,
    pixel_h: int,
    pixel_w: int,
) -> str:
    """Render a manim scene (via build_fn) to MP4, then convert to GIF with ffmpeg."""
    from manim import tempconfig

    out_name = os.path.splitext(os.path.basename(output_path))[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with tempconfig(
            {
                "media_dir": tmp_dir,
                "output_file": out_name,
                "format": "mp4",
                "pixel_height": pixel_h,
                "pixel_width": pixel_w,
                "frame_rate": fps,
                "background_color": _WHITE,
                "verbosity": "WARNING",
                "progress_bar": "none",
            }
        ):
            scene_build_fn()

        mp4s = glob.glob(os.path.join(tmp_dir, "**", "*.mp4"), recursive=True)
        if not mp4s:
            raise RuntimeError(f"manim produced no MP4 in {tmp_dir}")

        _mp4_to_gif(mp4s[0], output_path, fps)

    return output_path


def render_conv1_mechanics_gif(
    signal: np.ndarray,
    output_path: str,
    C: int = 512,
    Dm: int = 384,
    K1: int = 7,
    fps: int = 15,
) -> str:
    """
    Render the per-position Conv1d mechanics to a GIF.

    Shows: one (C, k) input patch × one (C, k) filter → 1 scalar,
    then ×d filters → one output column of length d.

    Args:
        signal:      (C, T_padded) raw neural signal array.
        output_path: destination .gif path.
        C:           input channel count (default 512).
        Dm:          output channel count / number of filters (default 384).
        K1:          Conv1 kernel size (default 7).
        fps:         frames per second for the GIF (default 15).

    Returns:
        output_path after the file has been written.
    """
    energy = np.abs(signal).mean(axis=0)
    active = np.where(energy > energy.max() * 0.05)[0]
    t0 = int(active[len(active) // 4]) if len(active) > K1 else 0
    t0 = min(t0, signal.shape[1] - K1 - 1)
    patch = signal[:, t0 : t0 + K1]

    scene = _ConvChannelScene(patch, C, Dm, K1)
    return _render_scene_to_gif(
        scene.build_and_render, output_path, fps, pixel_h=480, pixel_w=854
    )


def render_conv1_sliding_gif(
    signal: np.ndarray,
    act_conv1: np.ndarray,
    output_path: str,
    C: int = 512,
    Dm: int = 384,
    K1: int = 7,
    T_SHOW: int = 20,
    fps: int = 15,
) -> str:
    """
    Render the Conv1 sliding-kernel animation to a GIF.

    Shows: the full input heatmap, a kernel window (width k, full channel height)
    that slides from left to right, and the output heatmap revealing in sync.

    Args:
        signal:      (C, T_padded) raw neural signal array.
        act_conv1:   (Dm, T_padded) activations after Conv1 + GELU.
        output_path: destination .gif path.
        C:           input channel count (default 512).
        Dm:          output channel count (default 384).
        K1:          Conv1 kernel size (default 7).
        T_SHOW:      number of time steps to display (default 20).
        fps:         frames per second (default 15).

    Returns:
        output_path after the file has been written.
    """
    energy = np.abs(signal).mean(axis=0)
    active = np.where(energy > energy.max() * 0.05)[0]
    t0 = int(active[len(active) // 4]) if len(active) > T_SHOW else 0
    t0 = min(t0, signal.shape[1] - T_SHOW - 1)

    sig_win = signal[:, t0 : t0 + T_SHOW]
    conv1_win = act_conv1[:, t0 : t0 + T_SHOW]

    scene = _ConvSlidingScene(sig_win, conv1_win, C, Dm, K1, T_SHOW)
    return _render_scene_to_gif(
        scene.build_and_render, output_path, fps, pixel_h=480, pixel_w=854
    )
