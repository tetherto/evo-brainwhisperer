"""Kunz T15 normalization: per-block z-score fitted on the paired array.

The published T15 data was built by pairing each trial's bins two by two
from the trial's *delay* start (spike pairs summed, power pairs averaged),
joining the paired trials of one block, and dividing every value by the
block's per-channel std fitted on that paired array (``sd + 1e-8``). The
statistic lives on an array the stored payload is not: pairing is the
extractor's job (``pair_trigger_field="pair_start"``) and happens *after* the
loader read, so no ``stat_row_query`` can reach it. This transform pairs by
hand, fits per block, and bakes ``(mean, divisor)`` into the stream rows
through ``scaling.rescaled_loader_json``. The read rescales the *unpaired*
bins, and the extractor's linear pairing then yields the z-scored pairs.

The bake differs between the two streams. A rescaled stored bin is
``(a - B) / D``, so a summed spike pair reads back as ``(a1 + a2 - 2B) / D``.
To land on ``(S - mu) / (sd + eps)`` for pair sum ``S``, the transform bakes
``B = mu / 2``. The extractor then multiplies each summed spike pair by the
native-rate factor 2 *after* pairing
(``neuralshed/extractors.py::_get_paired_timed_array``), so the spike
divisor absorbs it too: ``D = 2 * (sd + eps)``, giving
``2 * (S - 2B) / D = (S - mu) / (sd + eps)``. An averaged power pair is
``(M - B) / D`` at factor 1, so the BandPower stream bakes ``B = mu``,
``D = sd + eps`` unchanged. Each trial's delay and go epochs abut, so pairing
the region ``[pair_start, sentence end)`` covers the same bins the published
data used.
"""

import typing as tp

import neuralset as ns
import numpy as np
import pandas as pd
from neuralshed import scaling

from brainwhisperer.data.rescale.bake import bake_loader, load_stream, sole_stream_row


class KunzPairedBlockRescale(ns.EventsTransform):
    """Rewrite Kunz's dense rows to the builder's paired-block z-score.

    For every timeline (one Kunz block) the transform pairs each cue row's
    region from ``pair_field`` to the row's end (trailing partial pair
    dropped), concatenates the paired rows in frame order, and fits
    ``(mean, std)`` per channel in float64. SpikeCounts pairs are summed,
    BandPower pairs averaged — the builder's resampling ops. SpikeCounts
    rows bake ``mean / 2`` (a stored shift removes *two* bins per summed
    pair), BandPower rows bake ``mean`` as is; both bake ``std + 1e-8`` as
    the divisor, and the read fits nothing (``stat_rows=[]``).

    A timeline with no cue rows, a second row of one stream type (the fit
    has no unambiguous stored array to pair against), a non-loader
    ``filepath``, or a region without a whole pair raise instead of
    guessing.

    SpikeCounts rows bake ``(mu / 2, 2 * (sd + eps))`` — the mean halving
    because a stored shift double-subtracts per summed pair, the divisor
    doubling because the extractor's after-pairing native-rate factor lands
    on summed spikes only. BandPower rows bake ``(mu, sd + eps)``.
    """

    event_types: tuple[str, ...] = ("SpikeCounts", "BandPower")
    cue_type: str = "Sentence"
    pair_field: str = "pair_start"
    rate_hz: float = 100.0

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.event_types:
            raise ValueError(f"{type(self).__name__}.event_types must be non-empty")
        if self.rate_hz <= 0:
            raise ValueError(f"{type(self).__name__}.rate_hz must be positive")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        missing = {"timeline", "type", "filepath", "start", "duration"}
        missing -= set(events.columns)
        if missing:
            raise KeyError(
                f"{type(self).__name__}: events are missing columns: {sorted(missing)}"
            )
        if self.pair_field not in events.columns:
            raise KeyError(
                f"{type(self).__name__}: events are missing the pairing anchor "
                f"column {self.pair_field!r}"
            )
        out = events.copy()
        rewritten = 0
        for timeline, frame in events.groupby("timeline", sort=False):
            baked = self._fit_timeline(timeline, frame)
            for etype in self.event_types:
                mean, divisor = baked[etype]
                index = sole_stream_row(
                    type(self).__name__,
                    frame,
                    etype,
                    timeline,
                    "expected exactly one stored stream per block",
                )
                raw = str(out.at[index, "filepath"])
                out.at[index, "filepath"] = bake_loader(
                    type(self).__name__, raw, mean=mean, divisor=divisor
                )
                rewritten += 1
        if not rewritten:
            raise ValueError(
                f"{type(self).__name__}: no rows of types {list(self.event_types)} "
                "to rescale in this frame"
            )
        return out

    def _fit_timeline(
        self, timeline: tp.Any, frame: pd.DataFrame
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``{stream type: (baked mean, baked divisor)}`` for one block."""
        cues = frame[frame["type"] == self.cue_type]
        if cues.empty:
            raise ValueError(
                f"{type(self).__name__}: timeline {timeline!r} has no "
                f"{self.cue_type!r} rows; the builder fitted its statistics "
                "over the paired cue regions"
            )
        fits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for etype in self.event_types:
            index = sole_stream_row(
                type(self).__name__, frame, etype, timeline, "expected exactly one"
            )
            raw = str(frame.at[index, "filepath"])
            x = load_stream(type(self).__name__, raw)
            t0 = float(frame.at[index, "start"])
            spike = etype == "SpikeCounts"
            paired = [self._pair_row(x, t0, cue, spike) for cue in cues.itertuples()]
            array = np.concatenate(paired, axis=1).astype(np.float64)
            mu = array.mean(axis=1)
            divisor = array.std(axis=1) + scaling.EPS
            # A summed spike pair of shifted bins double-subtracts the shift,
            # so the stored mean is half the paired one; and the extractor's
            # after-pairing native-rate factor 2 hits summed spikes only, so
            # the divisor doubles to absorb it (module docstring's algebra).
            # Averaged power needs neither.
            if spike:
                mu = mu / 2.0
                divisor = 2.0 * divisor
            fits[etype] = (mu, divisor)
        return fits

    def _pair_row(
        self, x: np.ndarray, t0: float, cue: tp.Any, spike: bool
    ) -> np.ndarray:
        """The builder's pairing of one cue region: ``(channels, n_pairs)``."""
        a_bin = int(round((float(getattr(cue, self.pair_field)) - t0) * self.rate_hz))
        s_bin = int(round((float(cue.start) - t0) * self.rate_hz))
        # Bin start and duration are rounded separately, the extractor's own
        # convention, so the region is the same bins the extractor will pair.
        e_bin = s_bin + int(round(float(cue.duration) * self.rate_hz))
        n_pairs = (e_bin - a_bin) // 2
        if n_pairs <= 0:
            raise ValueError(
                f"{type(self).__name__}: cue region [{a_bin}, {e_bin}) has no "
                "whole pair to fit on"
            )
        region = x[:, a_bin : a_bin + 2 * n_pairs]
        pairs = region.reshape(x.shape[0], n_pairs, 2)
        # spikes summed, power averaged - the builder's resampling ops
        # (oldsteps.py ``sum`` / ``average``).
        return pairs.sum(axis=2) if spike else pairs.mean(axis=2)


__all__ = ["KunzPairedBlockRescale"]
