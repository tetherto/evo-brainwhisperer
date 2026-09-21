from pathlib import Path

from aquarel import Theme

_THEME_PATH = Path(__file__).parent / "theme.json"
_FONTS_DIR = Path(__file__).parent / "fonts"


def load_theme() -> Theme:
    """
    Load the Tether.evo theme.

    If Sharp Grotesk OTF/TTF files are present in the same directory as this script, they are registered
    with matplotlib automatically. Falls back to Helvetica Neue / Arial otherwise.

    Usage:
        with load_theme():
            fig, ax = plt.subplots()
            ...

        # or apply globally (e.g. in a notebook):
        load_theme().apply()
    """
    _register_fonts()
    return Theme.from_file(str(_THEME_PATH))


def _register_fonts() -> None:
    """
    Register OTF/TTF files from `fonts/` with matplotlib's font manager.

    Also silences the per-glyph 'findfont: Generic family not found' warnings that
    matplotlib emits at WARNING level when a requested font isn't installed. Real font
    errors (corrupted files, bad paths) still propagate as exceptions.
    """
    import logging

    import matplotlib.font_manager as fm

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    if not _FONTS_DIR.exists():
        return
    # Register only the optical-size-20 cut so matplotlib consistently
    # resolves "Sharp Grotesk" to that variant across all weights.
    for path in sorted(_FONTS_DIR.rglob("*20.otf")):
        fm.fontManager.addfont(str(path))
    for path in sorted(_FONTS_DIR.rglob("*20.ttf")):
        fm.fontManager.addfont(str(path))
