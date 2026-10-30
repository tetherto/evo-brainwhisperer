"""Competition recipes (brainwhisperer/data/recipes/competition.py).

1. parity: Willett's competition chain, run on the labelled ``test`` files,
   gives exactly the features the training recipe gives for the same 880
   trials (same order, same sentences);
2. submission sets: Card yields the 1,450 ``test`` trials and Willett the
   1,200 ``competitionHoldOut`` trials, sessions sorted and trials in file
   order, which is the row order the Kaggle and EvalAI scorers expect;
3. model shape: each competition recipe keeps its training recipe's day axis.

Checks 1-2 need the study data and skip without it. The order was checked
once against the pickles behind the accepted submissions (trial lengths
match one to one, for both sets).
"""

import numpy as np
import pytest
from neuralshed.datasets.card_2024 import utils as card_utils
from neuralshed.datasets.utils import dataset_dir

from brainwhisperer.data.recipes import COMPETITION_RECIPES, RECIPES

_CARD_ROOT = dataset_dir("card_2024") / "data" / "Card2024Brain2Text"
_WILLET_ROOT = dataset_dir("willet_2023") / "data" / "WillettCompetitionData"

needs_card = pytest.mark.skipif(not _CARD_ROOT.exists(), reason="no Card data")
needs_willet = pytest.mark.skipif(
    not _WILLET_ROOT.exists(), reason="no Willett competitionData"
)


@pytest.mark.parametrize("name", ["card", "willet"])
def test_model_shape_matches_training(name):
    ours, train = COMPETITION_RECIPES[name], RECIPES[name]
    assert ours.NUM_DAYS == train.NUM_DAYS
    assert ours.calendar_days() == train.calendar_days()
    assert ours.CHANNEL_FILL == train.CHANNEL_FILL


def test_card_submission_format(tmp_path):
    path = COMPETITION_RECIPES["card"].write_submission(tmp_path, ["a b", "c, d"])
    assert path.read_text().splitlines() == ["id,text", "0,a b", '1,"c, d"']


def test_willet_submission_format(tmp_path):
    path = COMPETITION_RECIPES["willet"].write_submission(tmp_path, ["a b", "c"])
    assert path.read_text() == "a b\nc\n"


def _sessions_sorted_then_file_order(meta):
    """Rows grouped by session in sorted order, file order inside each session."""
    sessions = list(meta["session"])
    assert sessions == sorted(sessions)
    for _, rows in meta.groupby("session", sort=False):
        keys = list(zip(rows["block"], rows["trial_id"]))
        assert keys == sorted(keys)


@needs_willet
def test_willet_matches_training_recipe():
    recipe = COMPETITION_RECIPES["willet"]
    ours = recipe.build_competition_dataset(recipe.competition_chain(partition="test"))
    train = RECIPES["willet"]
    ref_events = train.build_chain(split="val").run()
    ref = train.segmenter(max_label_len=256).apply(ref_events)
    ref.prepare()

    assert len(ours) == len(ref) == 880
    for i in range(len(ours)):
        np.testing.assert_array_equal(
            ours[i].data["neural"].numpy(), ref[i].data["neural"].numpy()
        )
        assert int(ours[i].data["day"]) == int(ref[i].data["day"])


@needs_willet
def test_willet_holdout_submission_set():
    dataset = COMPETITION_RECIPES["willet"].build_competition_dataset()
    assert len(dataset) == 1200
    _sessions_sorted_then_file_order(dataset.metadata)


@needs_card
def test_card_test_submission_set():
    dataset = COMPETITION_RECIPES["card"].build_competition_dataset()
    assert len(dataset) == 1450
    expected = [
        (ref.session, t.block, t.trial)
        for ref in card_utils.list_sessions(_CARD_ROOT)
        for t in card_utils.list_trials(ref, "test")
    ]
    meta = dataset.metadata
    got = list(zip(meta["session"], meta["block"], meta["trial_id"]))
    assert got == expected
