"""Wairagkar normalization: per-block z-score pooled over the block's trials.

The published Wairagkar data was built by joining each block's 24 trials
along time, fitting one per-channel ``(mean, std)`` over that pooled array,
and dividing by ``sd`` with zeros replaced by ``1.0``. ``RescaleNeural`` fits
per timeline (one timeline is one trial here), so it cannot pool across
trials; this transform pools by hand and bakes ``(mean, divisor)`` into every
stream row of the block via ``scaling.rescaled_loader_json``.

The recipe extracts at ``frequency="native"`` (the release ships 100 Hz bins
and nothing rescales or pairs them), so a statistic pooled over stored bins
is exactly the published one, with no pairing correction as in Kunz. The
zero guard is applied in the divisor itself (``where(sd == 0, 1.0, sd)``), so
the loader's ``guard`` never fires on these rows; it is still written as
``"zero_to_one"`` so the JSON names the formula.
"""

import typing as tp

import neuralset as ns
import numpy as np
import pandas as pd

from brainwhisperer.data.rescale.bake import bake_loader, load_stream


class WairagkarBlockRescale(ns.EventsTransform):
    """Rewrite Wairagkar control-experiment rows to the block-pooled z-score.

    Group the frame by ``group_field`` (the study's ``block``, 0-9: one
    session of ten blocks of 24 trials), pool every stream row's whole
    stored payload per event type (the builder concatenated whole trials,
    so pooling covers exactly the same bins), fit ``(mean, std)`` per
    channel in float64, and bake ``mean`` and ``where(std == 0, 1.0,
    std)`` into each row's loader with ``stat_rows=[]`` — the read then
    applies the baked affine and fits nothing.

    A stream row whose ``filepath`` is not a loader JSON, or a frame
    without any row of ``event_types``, raises; a group with no rows of
    one event type simply fits nothing for that type (the study always
    emits both streams per trial, so silence would otherwise be invisible
    only if the frame itself were partial — the rewritten-nothing guard
    below catches that).
    """

    event_types: tuple[str, ...] = ("SpikeCounts", "BandPower")
    group_field: str = "block"

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.event_types:
            raise ValueError(f"{type(self).__name__}.event_types must be non-empty")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        missing = {"timeline", "type", "filepath"} - set(events.columns)
        if missing:
            raise KeyError(
                f"{type(self).__name__}: events are missing columns: {sorted(missing)}"
            )
        if self.group_field not in events.columns:
            raise KeyError(
                f"{type(self).__name__}: events are missing the pool key "
                f"column {self.group_field!r}"
            )
        out = events.copy()
        rewritten = 0
        for _, frame in events.groupby(self.group_field, sort=True):
            fits: dict[str, tuple[list[float], list[float]]] = {}
            for etype in self.event_types:
                rows = frame.index[frame["type"] == etype]
                if rows.empty:
                    continue
                pooled = self._pool(events, rows)
                mu = pooled.mean(axis=1)
                sd = pooled.std(axis=1)
                fits[etype] = (
                    [float(v) for v in mu],
                    # the builder's own zero treatment, baked verbatim
                    [float(v) for v in np.where(sd == 0, 1.0, sd)],
                )
            for etype, (mean, divisor) in fits.items():
                for index in frame.index[frame["type"] == etype]:
                    raw = str(out.at[index, "filepath"])
                    out.at[index, "filepath"] = bake_loader(
                        type(self).__name__,
                        raw,
                        mean=mean,
                        divisor=divisor,
                        guard="zero_to_one",
                    )
                    rewritten += 1
        if not rewritten:
            raise ValueError(
                f"{type(self).__name__}: no rows of types {list(self.event_types)} "
                "to rescale in this frame"
            )
        return out

    @classmethod
    def _pool(cls, events: pd.DataFrame, rows: pd.Index) -> np.ndarray:
        """Whole stored payloads of ``rows`` joined along their time axis."""
        arrays = [
            load_stream(cls.__name__, str(events.at[index, "filepath"]))
            for index in rows
        ]
        return np.concatenate(arrays, axis=1)


__all__ = ["WairagkarBlockRescale"]
