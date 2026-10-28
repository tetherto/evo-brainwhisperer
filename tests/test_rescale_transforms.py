"""The Kunz T15 and Wairagkar rescale transforms, on synthetic arrays only.

Both fit a statistic on an array other than the stored payload:

- ``KunzPairedBlockRescale`` (T15): pairs delay+go per trial and fits the
  block's ``(mean, std)`` on the *paired* array, joined over the block's
  trials.
- ``WairagkarBlockRescale``: fits ``(mean, sd)`` pooled over all 24 trials of
  a block file, dividing by ``sd`` with ``sd == 0`` replaced by ``1.0``.

These tests compute each formula by hand and check the values the transform
bakes into the loader JSON, including the spike correction the extractor
needs: ``mean / 2`` for pair sums and ``divisor x 2`` for its after-pairing
native-rate factor.
"""

import json
import typing as tp
import zlib

import numpy as np
import pandas as pd
import pytest
from neuralset.events import study as ns_study
from neuralset.events.utils import SpecialLoader
from neuralshed import scaling
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import BandPower, SpikeCounts  # noqa: F401, register types

from brainwhisperer.data.rescale import KunzPairedBlockRescale
from brainwhisperer.data.rescale import WairagkarBlockRescale

N_CH, N_BINS, RATE_HZ = 4, 80, 100.0
# Kunz's two cued trials: (sentence start, duration, pair_start) in seconds.
# Region [pair_start, sentence end) is the delay+go span (the two epochs
# abut); the first region is odd-length so the trailing-bin drop is exercised.
KUNZ_CUES = ((0.2, 0.2, 0.1), (0.5, 0.3, 0.4))


def _array(name: str) -> np.ndarray:
    # crc32, not hash(): Python's string hash is salted per process, which
    # would re-roll the fixture data on every pytest run
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return rng.standard_normal((N_CH, N_BINS))


def _kunz_pair_array(x: np.ndarray, spike: bool, clip: float = np.inf) -> np.ndarray:
    """Hand copy of the builder's pairing over KUNZ_CUES: ``(channels, pairs)``."""
    cols = []
    for start, duration, pair_start in KUNZ_CUES:
        a = int(round(pair_start * RATE_HZ))
        e = int(round(start * RATE_HZ)) + int(round(duration * RATE_HZ))
        n_pairs = (e - a) // 2
        region = np.minimum(x[:, a : a + 2 * n_pairs], clip)
        pairs = region.reshape(N_CH, n_pairs, 2)
        cols.append(pairs.sum(axis=2) if spike else pairs.mean(axis=2))
    return np.concatenate(cols, axis=1)


class KunzFixtureStudy(NeuralshedStudy):
    """One block timeline: two cued trials, two 100 Hz streams, pair_start stamps."""

    dataset_id: tp.ClassVar[str] = "kunz_2025"

    def iter_timelines(self):
        yield {"subject": "T12", "session": "b1"}

    def _load_stream(self, timeline: dict[str, tp.Any], *, which: str) -> np.ndarray:
        return _array(which)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        rows = [
            {
                "type": etype,
                "start": 0.0,
                "duration": N_BINS / RATE_HZ,
                "frequency": RATE_HZ,
                "n_features": N_CH,
                "filepath": ns_study.SpecialLoader(
                    method=self._load_stream, timeline=timeline, which=which
                ).to_json(),
            }
            for etype, which in (("SpikeCounts", "sc"), ("BandPower", "bp"))
        ]
        for start, duration, pair_start in KUNZ_CUES:
            rows.append(
                {
                    "type": "Sentence",
                    "start": start,
                    "duration": duration,
                    "pair_start": pair_start,
                    "text": "the cued sentence",
                }
            )
        return pd.DataFrame(rows)

    def _download(self) -> None:
        pass


def _kunz_events(tmp_path) -> pd.DataFrame:
    study_obj = KunzFixtureStudy(path=tmp_path)
    return study_obj._load_one({"subject": "T12", "session": "b1"})


def _baked(out: pd.DataFrame, etype: str) -> dict[str, tp.Any]:
    rows = out[out["type"] == etype]
    assert len(rows) == 1
    return json.loads(rows.iloc[0]["filepath"])


def test_kunz_bakes_paired_stats_with_the_spike_halving(tmp_path):
    out = KunzPairedBlockRescale()(_kunz_events(tmp_path))
    for etype, spike in (("SpikeCounts", True), ("BandPower", False)):
        payload = _baked(out, etype)
        assert payload["method"] == scaling.METHOD
        assert payload["kwargs"]["stat_rows"] == []
        assert payload["kwargs"]["guard"] == "eps"
        paired = _kunz_pair_array(_array("sc" if spike else "bp"), spike)
        mu = paired.mean(axis=1)
        expected_mean = mu / 2.0 if spike else mu
        assert payload["kwargs"]["mean"] == [float(v) for v in expected_mean]
        expected_divisor = paired.std(axis=1) + scaling.EPS
        if spike:
            # the extractor multiplies summed spike pairs by 2 after pairing
            # (native 100 Hz -> 50 Hz), and the baked divisor absorbs it
            expected_divisor = 2.0 * expected_divisor
        assert payload["kwargs"]["divisor"] == [float(v) for v in expected_divisor]


def test_kunz_paired_read_reproduces_the_builder_zscore(tmp_path):
    """Pairing the rescaled stored bins (plus the extractor's after-pairing
    factor 2 on summed spikes) lands on the builder's (S - mu) / (sd + eps)."""
    out = KunzPairedBlockRescale()(_kunz_events(tmp_path))
    for etype, spike in (("SpikeCounts", True), ("BandPower", False)):
        payload = _baked(out, etype)
        y = SpecialLoader.from_json(json.dumps(payload)).load()
        which = "sc" if spike else "bp"
        x = _array(which)
        paired_x = _kunz_pair_array(x, spike)
        mu = paired_x.mean(axis=1).reshape(-1, 1)
        divisor = (paired_x.std(axis=1) + scaling.EPS).reshape(-1, 1)
        # the first cued region, paired from its pair_start
        start, duration, pair_start = KUNZ_CUES[0]
        a = int(round(pair_start * RATE_HZ))
        e = int(round(start * RATE_HZ)) + int(round(duration * RATE_HZ))
        n_pairs = (e - a) // 2
        pairs = y[:, a : a + 2 * n_pairs].reshape(N_CH, n_pairs, 2)
        # model the extractor: summed spikes get the native-rate factor 2
        # after pairing, averaged power gets none
        joined = pairs.sum(axis=2) * 2.0 if spike else pairs.mean(axis=2)
        np.testing.assert_allclose(
            joined, (paired_x[:, :n_pairs] - mu) / divisor, rtol=1e-5
        )


def test_kunz_plain_filepath_is_refused(tmp_path):
    events = _kunz_events(tmp_path).copy()
    index = events.index[events["type"] == "SpikeCounts"][0]
    events.at[index, "filepath"] = "some/raw/path.npy"
    with pytest.raises(ValueError, match="SpecialLoader-backed"):
        KunzPairedBlockRescale()(events)


def test_kunz_timeline_without_cues_is_refused(tmp_path):
    events = _kunz_events(tmp_path)
    with pytest.raises(ValueError, match=r"has no 'Sentence' rows"):
        KunzPairedBlockRescale()(events[events["type"] != "Sentence"])


def test_kunz_double_stream_row_is_refused(tmp_path):
    events = _kunz_events(tmp_path)
    doubled = pd.concat([events, events], ignore_index=True)
    with pytest.raises(ValueError, match="expected exactly one"):
        KunzPairedBlockRescale()(doubled)


# ---------------------------------------------------------------- Wairagkar


WAIR_BINS = 40
#: (block, trial) pairs of the fixture; two trials share block 0.
WAIR_TIMELINES = ((0, 0), (0, 1), (1, 0))
#: one flat channel per stream, exercising the builder's sd[sd == 0] = 1.0 line
WAIR_FLAT = {"sc": 1, "bp": 2}


def _served(block: int, trial: int, which: str) -> np.ndarray:
    """One fixture stream: fixed random array with the flat channel zeroed."""
    x = _array(f"{block}-{trial}-{which}").copy()
    x[WAIR_FLAT[which]] = 0.0
    return x


class WairagkarFixtureStudy(NeuralshedStudy):
    """Two block files of pooled trials; one constant channel per stream."""

    dataset_id: tp.ClassVar[str] = "wairagkar_2025"

    def iter_timelines(self):
        for block, trial in WAIR_TIMELINES:
            yield {
                "subject": "T15",
                "session": "day482",
                "block": block,
                "trial_id": trial,
            }

    def _load_stream(self, timeline: dict[str, tp.Any], *, which: str) -> np.ndarray:
        return _served(int(timeline["block"]), int(timeline["trial_id"]), which)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        rows = [
            {
                "type": etype,
                "start": 0.0,
                "duration": WAIR_BINS / RATE_HZ,
                "frequency": RATE_HZ,
                "n_features": N_CH,
                "filepath": ns_study.SpecialLoader(
                    method=self._load_stream, timeline=timeline, which=which
                ).to_json(),
            }
            for etype, which in (("SpikeCounts", "sc"), ("BandPower", "bp"))
        ]
        rows.append(
            {
                "type": "Sentence",
                "start": 0.0,
                "duration": 0.5,
                "text": "the cued sentence",
            }
        )
        return pd.DataFrame(rows)

    def _download(self) -> None:
        pass


def _wair_events(tmp_path) -> pd.DataFrame:
    study_obj = WairagkarFixtureStudy(path=tmp_path)
    frames = [
        study_obj._load_one(
            {"subject": "T15", "session": "day482", "block": block, "trial_id": trial}
        )
        for block, trial in WAIR_TIMELINES
    ]
    return pd.concat(frames, ignore_index=True)


def _wair_pool(events: pd.DataFrame, block: int, etype: str) -> np.ndarray:
    """Hand copy of the builder's pool: whole stored payloads of the block."""
    which = "sc" if etype == "SpikeCounts" else "bp"
    arrays = [_served(b, t, which) for b, t in WAIR_TIMELINES if b == block]
    return np.concatenate(arrays, axis=1)


def test_wairagkar_bakes_the_block_pooled_affine(tmp_path):
    events = _wair_events(tmp_path)
    out = WairagkarBlockRescale()(events)
    for block in (0, 1):
        for etype in ("SpikeCounts", "BandPower"):
            pooled = _wair_pool(events, block, etype)
            mu = pooled.mean(axis=1)
            sd = pooled.std(axis=1)
            divisor = np.where(sd == 0, 1.0, sd)
            rows = out[(out["block"] == block) & (out["type"] == etype)]
            assert len(rows) == (2 if block == 0 else 1)
            for _, row in rows.iterrows():
                payload = json.loads(row["filepath"])
                assert payload["method"] == scaling.METHOD
                assert payload["kwargs"]["stat_rows"] == []
                assert payload["kwargs"]["guard"] == "zero_to_one"
                assert payload["kwargs"]["mean"] == [float(v) for v in mu]
                assert payload["kwargs"]["divisor"] == [float(v) for v in divisor]


def test_wairagkar_read_lands_on_the_builders_zscore(tmp_path):
    events = _wair_events(tmp_path)
    out = WairagkarBlockRescale()(events)
    for block, trial in WAIR_TIMELINES:
        for etype in ("SpikeCounts", "BandPower"):
            which = "sc" if etype == "SpikeCounts" else "bp"
            row = out[(out["block"] == block) & (out["trial_id"] == trial)]
            row = row[row["type"] == etype].iloc[0]
            got = SpecialLoader.from_json(row["filepath"]).load()
            pooled = _wair_pool(events, block, etype)
            mu = pooled.mean(axis=1).reshape(-1, 1)
            sd = pooled.std(axis=1).reshape(-1, 1)
            div = np.where(sd == 0, 1.0, sd)
            expected = (_served(block, trial, which) - mu) / div
            # the flat channel divides by 1.0 verbatim, not by eps
            assert div[WAIR_FLAT[which], 0] == 1.0
            np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_wairagkar_plain_filepath_is_refused(tmp_path):
    events = _wair_events(tmp_path).copy()
    index = events.index[events["type"] == "BandPower"][0]
    events.at[index, "filepath"] = "some/raw/path.mat"
    with pytest.raises(ValueError, match="SpecialLoader-backed"):
        WairagkarBlockRescale()(events)


def test_wairagkar_frame_without_dense_rows_is_refused(tmp_path):
    events = _wair_events(tmp_path)
    dense = events["type"].isin(["SpikeCounts", "BandPower"])
    with pytest.raises(ValueError, match="no rows of types"):
        WairagkarBlockRescale()(events[~dense])
