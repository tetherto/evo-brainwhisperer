"""Jude's normalization: paired block mean, divided by the last block's std.

The published Jude data was built by pairing each trial's bins two by two
from the go cue (threshold crossings summed, spike band power averaged after
capping values at 50,000), joining the paired trials of each block,
subtracting that block's per-channel mean, and dividing by the std of the
**session's last published block**. That divisor lives on other rows, and the
mean lives on the paired array rather than the stored one, so no
``stat_row_query`` can reach either. This transform pairs the same windows
per block (from the study's own ``Sentence`` rows), fits both statistics,
and bakes ``(mean, divisor)`` into the stream rows via
``scaling.rescaled_loader_json``.

"Last block" follows the published block lists
(``splits/jude_published_scope.json``), not the study's catalogue: the study
also holds blocks the published data never used, and on 2024.04.17, 05.06
and 05.07 its highest block is not the published last one. Blocks outside
the published scope are left unscaled.

The bake differs between the two streams, as in ``kunz.py``. The recipe
extracts at ``target_unit="native"`` with ``pair_trigger_field="start"``, so a
summed spike pair subtracts the stored shift **twice**: the transform bakes
the paired spike mean divided by 2. The spike divisor doubles too, because
the extractor multiplies each summed spike pair by the native-rate factor 2
*after* pairing (``neuralshed/extractors.py::_get_paired_timed_array``).
The read pair then equals ``(S - mean) / (std + eps)``. Averaged power bakes
mean and divisor unchanged. The released values are already capped at
50,000; the cap here spells out the original formula.
"""

import typing as tp

import neuralset as ns
import numpy as np
import pandas as pd
from neuralshed import scaling

from brainwhisperer.data.rescale.bake import bake_loader, load_stream, sole_stream_row
from brainwhisperer.data.splits import published_scope


class JudeLastBlockRescale(ns.EventsTransform):
    """Rewrite Jude dense rows to subtract the block mean, divide by the last block.

    One timeline is one block. Per timeline inside ``scope`` the transform
    pairs every ``cue_type`` window (spikes summed, power averaged after
    capping at ``power_clip``, trailing partial pair dropped), concatenates
    the paired windows in frame order, and fits the block's per-channel
    mean. Per session it fits the std the same way on the paired windows of
    the scope's **highest** block and bakes ``std + 1e-8`` as the divisor
    the paired windows of the scope's **highest** block and bakes
    ``std + 1e-8`` as the divisor for every block of that session.
    SpikeCounts rows bake ``block mean / 2`` and double that divisor
    (the extractor's pair sums double-subtract a stored shift, and its
    after-pairing native-rate factor lands on summed spikes only); BandPower
    rows bake the block mean and ``std + 1e-8`` as is; the loader fits
    nothing (``stat_rows=[]``).

    Timelines outside ``scope`` are left raw on purpose: their triggers
    never survive the recipe's held-out filter, and reading them would
    just cost the pass. A session missing from ``scope``, a scope's last
    block missing from the frame, a cue-less in-scope block, or a
    non-loader ``filepath`` raise.
    """

    event_types: tuple[str, ...] = ("SpikeCounts", "BandPower")
    cue_type: str = "Sentence"
    session_column: str = "session"
    block_column: str = "block"
    rate_hz: float = 100.0
    power_clip: float = 50000.0
    #: session -> the published run's block list; defaults to the pinned
    #: ``jude_published_scope.json`` so the recipe states no dataset value.
    scope: tp.Optional[dict[str, tp.Any]] = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.event_types:
            raise ValueError(f"{type(self).__name__}.event_types must be non-empty")
        if self.scope is None:
            self.scope = published_scope("jude")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        missing = {"timeline", "type", "filepath", "start", "duration"}
        missing -= set(events.columns)
        if self.block_column not in events.columns:
            missing.add(self.block_column)
        if missing:
            raise KeyError(
                f"{type(self).__name__}: events are missing columns: {sorted(missing)}"
            )
        if self.session_column not in events.columns:
            raise KeyError(
                f"{type(self).__name__}: events are missing the session column "
                f"{self.session_column!r}"
            )
        scope = {
            session: set(map(int, blocks)) for session, blocks in self.scope.items()
        }
        last_block = {session: max(blocks) for session, blocks in scope.items()}

        out = events.copy()
        bmeans: dict[tp.Any, dict[str, np.ndarray]] = {}  # timeline -> type -> mean
        pooled_last: dict[str, dict[str, list[np.ndarray]]] = {
            session: {etype: [] for etype in self.event_types} for session in scope
        }
        seen_last: set[tuple[str, str]] = set()
        fitted_sessions: set[str] = set()
        for timeline, frame in events.groupby("timeline", sort=False):
            session, block = self._session_block(frame)
            if session not in scope:
                raise ValueError(
                    f"{type(self).__name__}: session {session!r} is not in the "
                    f"published scope (have {sorted(scope)})"
                )
            if block not in scope[session]:
                continue  # outside the published scope: left unscaled
            fitted_sessions.add(session)
            fits = {}
            for etype in self.event_types:
                paired = self._pair_block(timeline, frame, etype)
                fits[etype] = paired.mean(axis=1)
                if block == last_block[session]:
                    pooled_last[session][etype].append(paired)
                    seen_last.add((session, etype))
            bmeans[timeline] = fits
        for session in fitted_sessions:
            for etype in self.event_types:
                if (session, etype) not in seen_last:
                    raise ValueError(
                        f"{type(self).__name__}: the published last block "
                        f"({last_block[session]}) of session {session!r} is not "
                        "in this frame; its std is this session's divisor"
                    )

        rewritten = 0
        gstds = {
            (session, etype): np.std(
                np.concatenate(pooled_last[session][etype], axis=1), axis=1
            )
            for session in fitted_sessions
            for etype in self.event_types
        }
        for timeline, frame in events.groupby("timeline", sort=False):
            if timeline not in bmeans:
                continue
            session, _ = self._session_block(frame)
            for etype in self.event_types:
                if etype == "SpikeCounts":
                    mean = bmeans[timeline][etype] / 2.0
                    # the extractor's after-pairing native-rate factor 2 hits
                    # summed spikes only; the divisor absorbs it (module
                    # docstring's algebra)
                    divisor = 2.0 * (gstds[(session, etype)] + scaling.EPS)
                else:
                    mean = bmeans[timeline][etype]
                    divisor = gstds[(session, etype)] + scaling.EPS
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
                f"{type(self).__name__}: no in-scope rows of types "
                f"{list(self.event_types)} to rescale in this frame"
            )
        return out

    def _session_block(self, frame: pd.DataFrame) -> tuple[str, int]:
        sessions = set(frame[self.session_column].dropna().astype(str))
        if len(sessions) != 1:
            raise ValueError(
                f"{type(self).__name__}: a timeline spans sessions "
                f"{sorted(sessions)}; the block mean is per session-block"
            )
        blocks = set(map(int, frame[self.block_column].dropna()))
        if len(blocks) != 1:
            raise ValueError(
                f"{type(self).__name__}: a timeline of session {sessions.pop()!r} "
                f"spans blocks {sorted(blocks)}; one Jude timeline is one block"
            )
        return sessions.pop(), blocks.pop()

    def _pair_block(
        self, timeline: tp.Any, frame: pd.DataFrame, etype: str
    ) -> np.ndarray:
        """The builder's paired block array for one stream: ``(channels, pairs)``."""
        index = sole_stream_row(
            type(self).__name__, frame, etype, timeline, "expected exactly one"
        )
        raw = str(frame.at[index, "filepath"])
        stream = load_stream(type(self).__name__, raw)
        t0 = float(frame.at[index, "start"])
        n_bins = stream.shape[-1]
        cues = frame[frame["type"] == self.cue_type]
        if cues.empty:
            raise ValueError(
                f"{type(self).__name__}: timeline {timeline!r} has no "
                f"{self.cue_type!r} rows to pair (the builder's block mean "
                "is fitted over every cued trial of the block)"
            )
        spike = etype == "SpikeCounts"
        paired = []
        for cue in cues.itertuples():
            s_bin = int(round((float(cue.start) - t0) * self.rate_hz))
            e_bin = min(s_bin + int(round(float(cue.duration) * self.rate_hz)), n_bins)
            n_pairs = (e_bin - s_bin) // 2  # trailing odd bin dropped, as compress()
            if n_pairs <= 0:
                raise ValueError(
                    f"{type(self).__name__}: cue window [{s_bin}, {e_bin}) of "
                    f"timeline {timeline!r} has no whole pair to fit on"
                )
            region = stream[:, s_bin : s_bin + 2 * n_pairs]
            if not spike:
                region = np.minimum(region, self.power_clip)  # the builder's cap,
                # before pairing: kept channels measured zero clips (jude_repro.log)
            pairs = region.reshape(stream.shape[0], n_pairs, 2)
            paired.append(pairs.sum(axis=2) if spike else pairs.mean(axis=2))
        return np.concatenate(paired, axis=1)


__all__ = ["JudeLastBlockRescale"]
