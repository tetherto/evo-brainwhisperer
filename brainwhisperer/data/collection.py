"""The mixed-dataset training collection. ``member`` is the registered dataset name,
injected into every sample's metadata so `RaggedCollator` carries it through to the batch
and the model can route rows to the right embedder (and the collator can pick
the right channel fill). ``day`` is the recording-day index the sampler stratifies on.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace

import pandas as pd
import torch

from brainwhisperer.data.recipes import RECIPES, build_in_parallel

MEMBER = "member"


def _day_mapping(extractor) -> dict:
    """The day-index mapping a prepared DAY extractor carries."""
    mapping = getattr(extractor, "predefined_mapping", None)
    if mapping is None:
        mapping = getattr(extractor, "_label_to_ind", None)
    if not mapping:
        raise ValueError(
            f"Day extractor {type(extractor).__name__} carries no day-index "
            "mapping after prepare(); the collection cannot group days"
        )
    return dict(mapping)


def _day_rank(mapping: dict, value, dataset_name: str):
    """One row's day index: the trigger field looked up like the extractor does."""
    key = value.isoformat() if hasattr(value, "isoformat") else value
    if hasattr(key, "item"):  # numpy integer (Wairagkar's `block` column)
        key = int(key)
    try:
        return int(mapping[key])
    except KeyError:
        raise KeyError(
            f"Trigger field value {value!r} of dataset {dataset_name!r} is "
            f"not in its day-index mapping; the day extractor could not have "
            f"encoded this row either"
        ) from None


def day_indices(ds, name: str) -> list[int]:
    """Each trial's day index in a prepared recipe dataset, read from its
    DAY extractor's mapping (no features are extracted)."""
    day_extractor = ds.extractors[RECIPES[name].DAY]
    mapping = _day_mapping(day_extractor)
    raw_days = ds.triggers[day_extractor.event_field]
    return [_day_rank(mapping, value, name) for value in raw_days]


class MixedRaggedDataset(torch.utils.data.Dataset):
    """Concatenation of per-member ragged datasets, tagged with their name.
    ``datasets`` maps registered dataset name -> prepared `RaggedSegmentDataset`
    in the order the sampler must scan it (``cfg.data.datasets`` order).
    """

    def __init__(self, datasets: Mapping[str, torch.utils.data.Dataset]) -> None:
        if not datasets:
            raise ValueError("MixedRaggedDataset needs at least one member")
        self._names: list[str] = list(datasets)
        self._datasets = [datasets[name] for name in self._names]

        self._starts: list[int] = []
        start = 0
        frames = []
        for name, ds in zip(self._names, self._datasets):
            self._starts.append(start)
            frame = ds.metadata.copy()
            frame.insert(0, MEMBER, name)
            frame.insert(1, RECIPES[name].DAY, day_indices(ds, name))
            frames.append(frame)
            start += len(ds)
        self._metadata = pd.concat(frames, ignore_index=True)

    @property
    def metadata(self) -> pd.DataFrame:
        """The sampler's scan table: one row per trial, columns ``member``, ``day``."""
        return self._metadata

    def __len__(self) -> int:
        return len(self._metadata)

    def locate(self, index: int) -> tuple[str, int]:
        """Which member and which row within it a global index lands on."""
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} outside collection of {len(self)}")
        for pos in range(len(self._names) - 1, -1, -1):
            if index >= self._starts[pos]:
                return self._names[pos], index - self._starts[pos]
        raise AssertionError("unreachable: _starts[0] == 0")

    def __getitem__(self, index: int):
        name, local = self.locate(index)
        sample = self._datasets[self._names.index(name)][local]
        return replace(sample, metadata={**sample.metadata, MEMBER: name})


def build_train_collection(
    names: Sequence[str], max_label_len: int
) -> MixedRaggedDataset:
    """The different train recipes as one sampler-ready dataset."""
    datasets = build_in_parallel(
        names, lambda name: RECIPES[name].build_train_dataset(max_label_len)
    )
    return MixedRaggedDataset(datasets)
