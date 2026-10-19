"""Per-dataset rescale transforms, applied inside the recipe chains.

Every dataset's published builder normalized the neural data with a
statistic the library ``RescaleNeural`` cannot fit (it lives on other rows,
or on the post-extraction *paired* array). Each transform here reproduces
its builder's statistic by hand at build time and bakes it into the stream
rows' loader JSONs (see `bake.py`), so the read is the plain affine the
parity audits pinned. One module per dataset; the recipe chains pick them
by name.
"""

from brainwhisperer.data.rescale.jude import JudeLastBlockRescale
from brainwhisperer.data.rescale.kunz import KunzPairedBlockRescale
from brainwhisperer.data.rescale.wairagkar import WairagkarBlockRescale

__all__ = [
    "JudeLastBlockRescale",
    "KunzPairedBlockRescale",
    "WairagkarBlockRescale",
]
