import pickle

import numpy as np
from torch.utils.data import Dataset

from brainwhisperer.config.paths import JUDE_PATH
from brainwhisperer.data.datasets.base import BaseNeuralDataset
from brainwhisperer.data.registry import register_dataset

# Per-session holdouts for test trials
_DAY_NUM_TEST_TRIALS: dict[int, int] = {0: 1, 1: 1, 2: 1, 3: 1, 4: 50}


def jude_held_out_split(
    days: np.ndarray, seed: int = 42
) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    n = len(days)
    all_indices = np.arange(n)
    test_indices: list[int] = []

    for day, num_test_trials in _DAY_NUM_TEST_TRIALS.items():
        day_indices = np.where(days == day)[0]
        if len(day_indices) < num_test_trials:
            raise ValueError(
                f"day {day}: requested {num_test_trials} test trials, "
                f"but only found {len(day_indices)} trials."
            )
        sampled = rng.choice(day_indices, size=num_test_trials, replace=False)
        test_indices.extend(sampled.tolist())

    test_indices = sorted(test_indices)
    train_indices = np.setdiff1d(all_indices, np.array(test_indices)).tolist()
    return train_indices, test_indices


@register_dataset("jude")
class JudeDataset(Dataset):
    NUM_DAYS: int = 5
    VAL_SPLIT: str = "val"

    def __init__(self, split: str = "train", sbj_idx: int = 0):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self._sbj_idx = sbj_idx
        with open(JUDE_PATH, "rb") as f:
            raw = pickle.load(f)
        days = np.array(raw["dayIdx"])
        train_idxs, test_idxs = jude_held_out_split(days, seed=42)
        self._base = BaseNeuralDataset(raw, has_labels=True, source_dataset="jude")
        self._idxs = train_idxs if split == "train" else test_idxs

    def __len__(self) -> int:
        return len(self._idxs)

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[self._idxs[idx]]
        sample["sbj_idx"] = self._sbj_idx
        return sample
