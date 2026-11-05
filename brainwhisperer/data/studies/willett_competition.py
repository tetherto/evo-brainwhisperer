"""Willett 2023 competitionData as a neuralset Study.

neuralshed ships the Willett ``sentences`` recordings only; the Brain-to-Text
'24 benchmark files (``competitionData/``) are read here, through neuralshed's
public Study base, so this module is also the template for bringing other
data into the pipeline.

Each timeline emits:
- ``SpikeCounts`` (tx1) and ``BandPower`` (spikePow): the block's trials joined
  end to end, ``(256, n_bins)`` at 50 Hz, channels in neuralshed's array order
  (the layout ``Willett2023Sentences`` emits);
- one ``Stimulus`` per trial over its go period, the trial marker the
  segmenter triggers on and ``RescaleNeural`` pools statistics over;
- one ``Sentence`` per trial when the file carries the text (``test``; the
  hold-out reads ``"HELD OUT"``).

The files already drop the first bin of every go period, so recipes reading
this study must not trim it again.
"""

import functools
import typing as tp

import numpy as np
import pandas as pd
import scipy.io as sio
from neuralset.events import study
from neuralshed.datasets import participants
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.datasets.willet_2023 import utils as willet_utils
from neuralshed.events import (  # noqa: F401, registers the neural-feature types
    BandPower,
    SpikeCounts,
)

_RATE_HZ = 50.0
_BIN_S = 1 / _RATE_HZ
_HELD_OUT_TEXT = "HELD OUT"


@functools.lru_cache(maxsize=4)
def _load_mat(path: str) -> dict:
    """One session file; cached because every block of a session reads it."""
    return sio.loadmat(path, simplify_cells=True)


class WillettCompetitionData(NeuralshedStudy):
    """Willett 2023 competitionData, one partition per instance (T12)."""

    # Same Dryad deposit and participant as Willett2023Sentences, so the
    # participant mapping comes from neuralshed's willet_2023 metadata.
    dataset_id: tp.ClassVar[str] = "willet_2023"
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.x69p8czpq"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Willett et al. 2023 competitionData: go-period snippets of T12's "
        "attempted speech, partitioned for the Brain-to-Text '24 benchmark."
    )

    partition: tp.Literal["competitionHoldOut", "test"] = "competitionHoldOut"

    def _cls_kwargs(self) -> dict[str, tp.Any]:
        # The partition picks which files are read, so it is part of the
        # timeline identity.
        return {"partition": self.partition}

    def _file(self, session: str) -> str:
        return str(self.path / self.partition / f"{session}.mat")

    def _trial_indices(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """Rows of the session file that belong to this timeline's block."""
        d = _load_mat(self._file(timeline["session"]))
        return np.flatnonzero(np.atleast_1d(d["blockIdx"]) == timeline["block"])

    # -- Timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        files = sorted(
            f
            for f in (self.path / self.partition).glob("t12.*.mat")
            if not f.name.startswith("._")
        )
        for f in files:
            session = f.stem
            date = willet_utils.session_date(session)
            blocks = np.atleast_1d(_load_mat(str(f))["blockIdx"])
            for block in sorted(set(blocks.tolist())):
                yield dict(
                    subject="T12",
                    session=session,
                    date=date,
                    post_implant_day=participants.post_implant_day(
                        willet_utils.PARTICIPANT_ID, date
                    ),
                    block=int(block),  # numbered within the file, not the recording
                    original_split=self.partition,
                )

    def _load_stream(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: one block's trials joined, ``(256, n_bins)``."""
        d = _load_mat(self._file(timeline["session"]))
        x = np.concatenate([d[key][i] for i in self._trial_indices(timeline)])
        return np.ascontiguousarray(willet_utils.to_array_order(x).T, dtype=np.float32)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        d = _load_mat(self._file(timeline["session"]))
        trials = self._trial_indices(timeline)
        n_bins = [d["tx1"][i].shape[0] for i in trials]
        duration_s = sum(n_bins) * _BIN_S

        def _neural(key: str, **fields) -> dict[str, tp.Any]:
            return dict(
                start=0.0,
                duration=duration_s,
                frequency=_RATE_HZ,
                n_features=256,
                normalization="raw",
                filepath=study.SpecialLoader(
                    method=self._load_stream, timeline=timeline, key=key
                ).to_json(),
                **fields,
            )

        rows: list[dict[str, tp.Any]] = [
            _neural(
                "tx1",
                type="SpikeCounts",
                feature="threshold_crossings",
                method="threshold_crossing",
                threshold="-3.5 x RMS",
                unit="counts/bin",
            ),
            _neural(
                "spikePow",
                type="BandPower",
                feature="spike_band_power",
                unit="uV^2",
            ),
        ]

        offset = 0
        for i, n in zip(trials, n_bins):
            start = offset * _BIN_S
            rows.append(
                dict(
                    type="Stimulus",
                    start=start,
                    duration=n * _BIN_S,
                    modality="visual",
                    description="go period",
                    stimulus_kind="speech_go_period",
                    trial_id=int(i),  # row in the session file
                )
            )
            text = str(d["sentenceText"][i]).strip()
            if text and text != _HELD_OUT_TEXT:
                rows.append(
                    dict(
                        type="Sentence",
                        start=start,
                        duration=n * _BIN_S,
                        text=text,
                        language="en",
                        modality="read",
                        trial_id=int(i),
                    )
                )
            offset += n
        return pd.DataFrame(rows)
