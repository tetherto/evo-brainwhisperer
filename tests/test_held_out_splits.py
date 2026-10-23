"""Pinned held-out val trials (brainwhisperer/data/splits/).

The published Wairagkar and Jude splits were random draws (per-block
``torch.randperm`` with seed 42, and ``np.random.RandomState(42)`` with fixed
counts), so the val trials are stored as JSON files. These tests check:

1. shape: the pinned files have the published split structure. Wairagkar
   holds out 5 of the 24 trials in each of the 10 blocks of ``t15_day482``;
   Jude holds out 1/1/1/1/50 per session, all inside the published block
   scope, with keys sorted;
2. assembly: ``val_query`` builds one clause per pinned key from the JSON;
3. train pin: the Wairagkar train pin is disjoint from the val pin, covers
   every trial, and is actually permuted.

All checks are pure: no study data needed.
"""

from collections import Counter

from brainwhisperer.data import splits


# ----------------------------------------------------------------------
# The pinned files carry the published split structure
# ----------------------------------------------------------------------


def test_wairagkar_val_shape():
    keys = splits.load_val_keys("wairagkar")
    assert len(keys) == 50
    assert {k["session"] for k in keys} == {"t15_day482"}
    per_block = sorted(Counter(k["block"] for k in keys).items())
    assert per_block == [(b, 5) for b in range(10)]  # 19 train / 5 val per day
    assert all(0 <= k["trial"] < 24 for k in keys)


def test_jude_val_shape():
    keys = splits.load_val_keys("jude")
    assert len(keys) == 54
    assert Counter(k["session"] for k in keys) == Counter(
        {
            "t17.2024.05.07": 50,
            "t17.2024.04.15": 1,
            "t17.2024.04.16": 1,
            "t17.2024.04.17": 1,
            "t17.2024.05.06": 1,
        }
    )
    # Every held-out trial lies inside the published block scope.
    scope = splits.published_scope("jude")
    assert all(k["block"] in scope[k["session"]] for k in keys)
    assert keys == sorted(keys, key=lambda k: (k["session"], k["block"], k["trial"]))


def test_wairagkar_has_no_block_scope():
    # Wairagkar used all 10 block files of its single session (240/240).
    assert splits.published_scope("wairagkar") == {}


# ----------------------------------------------------------------------
# Queries are assembled from the JSON files
# ----------------------------------------------------------------------


def test_queries_assembled_from_files():
    q = splits.val_query("jude")
    assert q.count("trial_id ==") == 54
    assert q.count(" or ") == 53
    q = splits.val_query("wairagkar")
    assert q.count("trial_id ==") == 50


# ----------------------------------------------------------------------
# The Wairagkar train pin is a complete, permuted partition
# ----------------------------------------------------------------------


def test_wairagkar_train_pin_integrity():
    train = splits.load_train_keys("wairagkar")
    val = splits.load_val_keys("wairagkar")
    train_set = {(k["block"], k["trial"]) for k in train}
    val_set = {(k["block"], k["trial"]) for k in val}
    assert len(train) == 190 and len(train_set) == 190
    assert train_set.isdisjoint(val_set)
    assert train_set | val_set == {(b, t) for b in range(10) for t in range(24)}
    assert all(k["session"] == "t15_day482" for k in train)
    # A sorted pin would make the reorder a no-op.
    assert train != sorted(train, key=lambda k: (k["block"], k["trial"]))
