"""Loader-rewrite plumbing shared by the rescale transforms.

"Bake" is this package's verb for computing a published builder's
normalization statistic at build time and writing it into a stream row's
loader JSON via ``scaling.rescaled_loader_json``: the read then applies
``(x - mean) / divisor`` and fits nothing (``stat_rows=[]``).

The transforms (``jude``, ``kunz``, ``wairagkar``) differ in *how* they fit
— those asymmetries are audit-pinned and stay in each module — but they are
identical in how they read a stored stream, insist a timeline carries
exactly one row per event type, and rewrite loader rows. That shared
plumbing lives here; the audit-pinned error wordings are kept verbatim.
"""

import json
import typing as tp

import numpy as np
import pandas as pd
from neuralset.events.utils import SpecialLoader

from tether_scratch import scaling


def _stream_error(owner: str, raw: str) -> ValueError:
    return ValueError(
        f"{owner}: stores its array at {raw!r}; "
        "rescaling needs a SpecialLoader-backed stream"
    )


def load_stream(owner: str, raw: str) -> np.ndarray:
    """The stored float64 array behind a loader-JSON ``filepath``."""
    try:
        return np.asarray(SpecialLoader.from_json(raw).load(), dtype=np.float64)
    except ValueError as error:  # ujson's JSONDecodeError is a ValueError
        raise _stream_error(owner, raw) from error


def bake_loader(
    owner: str,
    raw: str,
    *,
    mean: tp.Sequence[float],
    divisor: tp.Sequence[float],
    guard: str = "eps",
) -> str:
    """The loader JSON that reads ``raw``'s array as ``(x - mean) / divisor``."""
    try:
        json.loads(raw)
    except json.JSONDecodeError as error:
        raise _stream_error(owner, raw) from error
    return scaling.rescaled_loader_json(
        raw,
        stat_rows=[],
        guard=guard,
        divisor=[float(v) for v in divisor],
        mean=[float(v) for v in mean],
    )


def sole_stream_row(
    owner: str, frame: pd.DataFrame, etype: str, timeline: tp.Any, expect: str
) -> tp.Any:
    """The index label of the single ``etype`` row of one timeline."""
    rows = frame.index[frame["type"] == etype]
    if len(rows) != 1:
        raise ValueError(
            f"{owner}: timeline {timeline!r} has {len(rows)} {etype} rows, {expect}"
        )
    (index,) = rows
    return index


__all__ = ["load_stream", "bake_loader", "sole_stream_row"]
