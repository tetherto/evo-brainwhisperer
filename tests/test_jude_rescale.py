"""The Jude rescale transform: paired block mean, last-published-block divisor.

Synthetic arrays only. The transform fits the block mean over the *paired*
cue windows (spikes summed, power averaged after the 50,000 cap) and the
divisor as the std of the session's **last published** block, not the
study's highest block. These tests check the baked values against a
hand-computed copy of the formula, on the same bins the extractor pairs. The
spike copy also models the extractor's after-pairing native-rate factor 2:
baked mean / 2, divisor x 2.
"""

import json
import typing as tp

import numpy as np
import pandas as pd
import pytest
from neuralset.events import study as ns_study
from neuralset.events.utils import SpecialLoader
from neuralshed import scaling
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import BandPower, SpikeCounts  # noqa: F401, register types

from brainwhisperer.data.rescale import JudeLastBlockRescale

N_CH, N_BINS, RATE_HZ = 4, 40, 100.0
#: BandPower cap, applied before pairing
CLIP = 50000.0
#: (start, duration) of the two cued windows of a block, in seconds
CUES = ((0.0, 0.2), (0.2, 0.2))
#: the published blocks for s1; block 3 exists in the study but is out of scope
SCOPE = {"s1": [1, 2]}


def _array(block: int, which: str) -> np.ndarray:
    """One stored ``(channels, bins)`` stream: fixed per (block, which)."""
    rng = np.random.default_rng(100 * block + (0 if which == "sc" else 1))
    x = rng.standard_normal((N_CH, N_BINS))
    if which == "sc":
        x = np.abs(x) * 3.0  # spike counts are non-negative
    else:
        x = np.abs(x) * 1000.0 + 500.0  # band power is positive
    x[3] = 2.5  # constant channel: paired std is 0, the eps guard earns its keep
    return x


def _paired(block: int, which: str, clip: float = CLIP) -> np.ndarray:
    """The builder's paired block array, hand-computed: ``(channels, pairs)``.

    Same bins and ops as the extractor's pairing: window from
    its own start, trailing odd bin dropped, spikes summed, power capped
    *before* the pair average.
    """
    return _pair_array(_array(block, which), which, clip)


def _pair_array(x: np.ndarray, which: str, clip: float = CLIP) -> np.ndarray:
    """``_paired`` over an explicit array (the clip test feeds a loud one)."""
    cols = []
    for start, duration in CUES:
        a = int(round(start * RATE_HZ))
        e = min(a + int(round(duration * RATE_HZ)), x.shape[1])
        n_pairs = (e - a) // 2
        region = x[:, a : a + 2 * n_pairs]
        if which == "bp":
            region = np.minimum(region, clip)
        pairs = region.reshape(x.shape[0], n_pairs, 2)
        cols.append(pairs.sum(axis=2) if which == "sc" else pairs.mean(axis=2))
    return np.concatenate(cols, axis=1)


class JudeFixtureStudy(NeuralshedStudy):
    """s1 blocks 1-2 (published) + block 3 (out of scope), 100 Hz bins."""

    dataset_id: tp.ClassVar[str] = "jude_2026_typing"

    def iter_timelines(self):
        for block in (1, 2, 3):
            yield {"subject": "T17", "session": "s1", "block": block}

    def _load_stream(self, timeline: dict[str, tp.Any], *, which: str) -> np.ndarray:
        return _array(int(timeline["block"]), which)

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
        for start, duration in CUES:
            rows.append(
                {
                    "type": "Sentence",
                    "start": start,
                    "duration": duration,
                    "text": "the cued sentence",
                }
            )
        return pd.DataFrame(rows)

    def _download(self) -> None:
        pass


def _events(tmp_path, blocks: tp.Sequence[int] = (1, 2, 3)) -> pd.DataFrame:
    study_obj = JudeFixtureStudy(path=tmp_path)
    frames = [
        study_obj._load_one({"subject": "T17", "session": "s1", "block": block})
        for block in blocks
    ]
    return pd.concat(frames, ignore_index=True)


def _baked(out: pd.DataFrame, block: int, etype: str) -> dict[str, tp.Any]:
    rows = out[(out["block"] == block) & (out["type"] == etype)]
    assert len(rows) == 1
    return json.loads(rows.iloc[0]["filepath"])


def test_bakes_paired_mean_and_last_block_std_exactly(tmp_path):
    out = JudeLastBlockRescale(scope=SCOPE)(_events(tmp_path))
    for etype, which in (("SpikeCounts", "sc"), ("BandPower", "bp")):
        # a summed spike pair double-subtracts the shift -> mean / 2;
        # an averaged power pair needs no halving
        for block in (1, 2):
            payload = _baked(out, block, etype)
            assert payload["method"] == scaling.METHOD
            assert payload["kwargs"]["stat_rows"] == []
            assert payload["kwargs"]["guard"] == "eps"
            paired = _paired(block, which)
            if etype == "SpikeCounts":
                assert payload["kwargs"]["mean"] == [
                    float(v) for v in paired.mean(axis=1) / 2.0
                ]
            else:
                assert payload["kwargs"]["mean"] == [
                    float(v) for v in paired.mean(axis=1)
                ]
            # divisor: the std of block 2 (published last), not this block's;
            # spikes double it to absorb the extractor's after-pairing
            # native-rate factor 2 (rescale/jude.py's module docstring)
            divisor = _paired(2, which).std(axis=1) + scaling.EPS
            if etype == "SpikeCounts":
                divisor = 2.0 * divisor
            assert payload["kwargs"]["divisor"] == [float(v) for v in divisor]


def test_read_applies_the_baked_affine(tmp_path):
    events = _events(tmp_path)
    out = JudeLastBlockRescale(scope=SCOPE)(events)
    for etype, which in (("SpikeCounts", "sc"), ("BandPower", "bp")):
        for block in (1, 2):
            payload = _baked(out, block, etype)
            got = SpecialLoader.from_json(json.dumps(payload)).load()
            mu = np.asarray(payload["kwargs"]["mean"]).reshape(-1, 1)
            div = np.asarray(payload["kwargs"]["divisor"]).reshape(-1, 1)
            np.testing.assert_allclose(
                got, (_array(block, which) - mu) / div, rtol=1e-6, atol=1e-7
            )


def test_extractor_pairing_of_the_read_reproduces_the_builder_zscore(tmp_path):
    """Pair the rescaled stored bins: (S - block mean) / (last-block std + eps)."""
    out = JudeLastBlockRescale(scope=SCOPE)(_events(tmp_path))
    for etype, which in (("SpikeCounts", "sc"), ("BandPower", "bp")):
        for block in (1, 2):
            payload = _baked(out, block, etype)
            y = SpecialLoader.from_json(json.dumps(payload)).load()
            start, duration = CUES[0]
            a = int(round(start * RATE_HZ))
            e = min(a + int(round(duration * RATE_HZ)), N_BINS)
            n_pairs = (e - a) // 2
            pairs = y[:, a : a + 2 * n_pairs].reshape(N_CH, n_pairs, 2)
            # model the extractor: summed spikes get the native-rate factor 2
            # after pairing, averaged power gets none
            joined = (
                pairs.sum(axis=2) * 2.0
                if etype == "SpikeCounts"
                else pairs.mean(axis=2)
            )
            paired = _paired(block, which)
            block_mean = paired.mean(axis=1).reshape(-1, 1)
            divisor = (_paired(2, which).std(axis=1) + scaling.EPS).reshape(-1, 1)
            np.testing.assert_allclose(
                joined, (paired[:, :n_pairs] - block_mean) / divisor, rtol=1e-5
            )


def test_divisor_is_the_published_last_block_not_the_study_max(tmp_path):
    """Block 3 exists in the study and is higher; the builder never read it."""
    events = _events(tmp_path)
    out = JudeLastBlockRescale(scope=SCOPE)(events)
    for etype, which in (("SpikeCounts", "sc"), ("BandPower", "bp")):
        divisor = _baked(out, 1, etype)["kwargs"]["divisor"]
        block3 = (_paired(3, which).std(axis=1) + scaling.EPS).tolist()
        assert not np.allclose(divisor, block3)  # extras must not feed the fit


def test_out_of_scope_block_is_left_raw(tmp_path):
    events = _events(tmp_path)
    out = JudeLastBlockRescale(scope=SCOPE)(events)
    dense = events["type"].isin(["SpikeCounts", "BandPower"])
    extra_in = events[dense & (events["block"] == 3)]
    extra_out = out[dense & (out["block"] == 3)]
    assert len(extra_in) == 2  # one stored stream per type, none silently dropped
    assert extra_in["filepath"].tolist() == extra_out["filepath"].tolist()
    for raw in extra_out["filepath"]:
        assert "mean" not in json.loads(raw).get("kwargs", {})


def test_unknown_session_is_refused(tmp_path):
    events = _events(tmp_path).copy()
    events["session"] = "s_typo"
    with pytest.raises(
        ValueError,
        match=r"session 's_typo' is not in the "
        r"published scope",
    ):
        JudeLastBlockRescale(scope=SCOPE)(events)


def test_missing_published_last_block_is_refused(tmp_path):
    # block 2 in scope but absent from the frame: its std is this session's divisor
    with pytest.raises(
        ValueError,
        match=r"published last block \(2\) of session "
        r"'s1' is not in this frame",
    ):
        JudeLastBlockRescale(scope=SCOPE)(_events(tmp_path, blocks=(1, 3)))


def test_cueless_in_scope_block_is_refused(tmp_path):
    events = _events(tmp_path)
    cueless = events[~((events["block"] == 1) & (events["type"] == "Sentence"))]
    with pytest.raises(ValueError, match=r"has no 'Sentence' rows to pair"):
        JudeLastBlockRescale(scope=SCOPE)(cueless)


def test_plain_filepath_stream_is_refused(tmp_path):
    events = _events(tmp_path).copy()
    index = events.index[(events["block"] == 1) & (events["type"] == "SpikeCounts")][0]
    events.at[index, "filepath"] = "some/raw/path.npy"
    with pytest.raises(ValueError, match="SpecialLoader-backed"):
        JudeLastBlockRescale(scope=SCOPE)(events)


def test_power_cap_lands_before_pairing(tmp_path):
    """The builder capped at 50,000 *before* the pair average; the transform
    must fit its block mean on the capped values, not the raw ones."""

    class LoudBandPowerStudy(JudeFixtureStudy):
        """Same fixture with ch 0 of the power streams far above the cap."""

        def _load_stream(self, timeline, *, which):  # noqa: ANN001
            x = super()._load_stream(timeline, which=which)
            if which == "bp":
                x = x.copy()
                x[0] = 2 * CLIP
            return x

    study_obj = LoudBandPowerStudy(path=tmp_path)
    events = pd.concat(
        [
            study_obj._load_one({"subject": "T17", "session": "s1", "block": block})
            for block in (1, 2, 3)
        ],
        ignore_index=True,
    )
    out = JudeLastBlockRescale(scope=SCOPE)(events)
    payload = _baked(out, 1, "BandPower")
    loud = _array(1, "bp").copy()
    loud[0] = 2 * CLIP
    capped = _pair_array(loud, "bp", clip=CLIP).mean(axis=1)
    uncapped = _pair_array(loud, "bp", clip=np.inf).mean(axis=1)
    # ch 0: every capped pair average is exactly the cap, so the fitted mean
    # lands on CLIP; uncapped it would sit near 2*CLIP
    assert payload["kwargs"]["mean"][0] == pytest.approx(float(capped[0]))
    assert payload["kwargs"]["mean"][0] == pytest.approx(CLIP)
    assert payload["kwargs"]["mean"][0] != pytest.approx(float(uncapped[0]))
    # the channels under the cap are untouched by it
    np.testing.assert_allclose(payload["kwargs"]["mean"][1:], capped[1:], rtol=1e-12)


def test_default_scope_is_the_pinned_published_run(tmp_path):
    from brainwhisperer.data.splits import published_scope

    transform = JudeLastBlockRescale()
    assert transform.scope == published_scope("jude")
    # the recipe's five sessions are the scope's, with their published last blocks
    assert {s: max(b) for s, b in transform.scope.items()} == {
        "t17.2024.04.15": 13,
        "t17.2024.04.16": 13,
        "t17.2024.04.17": 6,
        "t17.2024.05.06": 10,
        "t17.2024.05.07": 10,
    }
