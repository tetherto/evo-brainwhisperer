"""Pinned held-out val trials for Wairagkar and Jude.

The published Wairagkar and Jude splits were drawn at random. These JSON
files store the resulting trials, so every run holds out the same ones.
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


def held_out_split(dataset: str):
    """``CreateColumn`` adding ``split``: ``train`` everywhere, ``val`` on
    the pinned trials' rows.

    The query is built from the JSON files, which stay the single source of
    truth. Recipes apply it before the Segmenter, so segments inherit
    ``split`` from their trigger rows.
    """
    from neuralset.events.transforms.basic import CreateColumn

    return CreateColumn(
        column="split",
        query_row=val_query(dataset),
        default_value="train",
        query_value="val",
    )
