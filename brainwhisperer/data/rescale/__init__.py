"""Per-dataset rescale transforms, applied inside the recipe chains.

Each published dataset was normalized with a statistic that the library
``RescaleNeural`` cannot fit, because it lives on other rows or on the
*paired* array the extractor builds later. Each transform computes that
statistic at build time and writes it into the stream rows' loader JSON
(see `bake.py`), so reading a row applies a plain affine rescale. One module
per dataset; the recipe chains pick them by name.
"""

from brainwhisperer.data.rescale.jude import JudeLastBlockRescale
from brainwhisperer.data.rescale.kunz import KunzPairedBlockRescale
from brainwhisperer.data.rescale.wairagkar import WairagkarBlockRescale

__all__ = [
    "JudeLastBlockRescale",
    "KunzPairedBlockRescale",
    "WairagkarBlockRescale",
]
