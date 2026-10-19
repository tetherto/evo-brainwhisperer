"""Pinned held-out val trials for Wairagkar and Jude.

The splits used for the published results were pseudo-random dice rolls
reproducible only under the old pickle loading pipeline
(the one original adopted by the model, now swapped for one based on NeuralSet).
These JSON files pin the resuxlting held-out trials for the sake of reproducibility.
"""

import json
import typing as tp
from pathlib import Path

HERE = Path(__file__).parent


VAL_FILES = {
    "wairagkar": HERE / "wairagkar_val.json",
    "jude": HERE / "jude_val.json",
}

READER_COLUMNS = {"session": "session", "block": "block", "trial": "trial_id"}

SCOPE_FILES = {"jude": HERE / "jude_published_scope.json"}

TRAIN_FILES = {"wairagkar": HERE / "wairagkar_train.json"}


def load_val_keys(dataset: str) -> list[dict[str, tp.Any]]:
    """The pinned held-out trials (sorted by session/block/trial)."""
    with open(VAL_FILES[dataset], encoding="utf-8") as f:
        return json.load(f)


def load_train_keys(dataset: str) -> list[dict[str, tp.Any]]:
    """The pinned train trials in the published run's order (do not sort)."""
    with open(TRAIN_FILES[dataset], encoding="utf-8") as f:
        return json.load(f)


def val_query(dataset: str) -> str:
    """Pandas query selecting the pinned trials' rows in the reader events."""
    clauses = []
    for key in load_val_keys(dataset):
        terms = []
        for pinned, column in READER_COLUMNS.items():
            value = key[pinned]
            if isinstance(value, str):
                terms.append(f"{column} == {value!r}")
            else:
                terms.append(f"{column} == {value}")
        clauses.append("(" + " and ".join(terms) + ")")
    return " or ".join(clauses)


def published_scope(dataset: str) -> dict[str, list[int]]:
    """The published run's per-session block universe ({} if never pinned —
    Wairagkar used all 10 block files of its single session, 240/240)."""
    path = SCOPE_FILES.get(dataset)
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def old_scope_query(dataset: str) -> str:
    """Pandas query selecting the published run's builder blocks.

    Not applied by default (PLAN.md step 8, D4 — the reader's extra
    all-success task blocks are kept, per the Card precedent for its 3
    extra healthy trials). Step 14 applies it as the first diagnostic if
    its validation scores land outside the old runs' spread.
    """
    clauses = [
        f"(session == {session!r} and block in {sorted(blocks)})"
        for session, blocks in published_scope(dataset).items()
    ]
    return " or ".join(clauses)


def held_out_split(dataset: str):
    """``CreateColumn`` adding ``split``: ``train`` everywhere, ``val`` on
    the pinned trials' rows.

    Assembling the query here keeps the JSON files the single source of
    truth (PLAN.md step 8, D3). Applied in the step-11 recipes before the
    Segmenter, so segments inherit ``split`` from their trigger rows.
    """
    from neuralset.events.transforms.basic import CreateColumn

    return CreateColumn(
        column="split",
        query_row=val_query(dataset),
        default_value="train",
        query_value="val",
    )
