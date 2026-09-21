import pickle

import torch
from torch.utils.data import Dataset

from brainwhisperer.config.paths import WAIRAGKAR_ROOT
from brainwhisperer.data.datasets.base import BaseNeuralDataset
from brainwhisperer.data.registry import register_dataset

_PATH = f"{WAIRAGKAR_ROOT}/data/Wairagkar_train_data.pkl"


def session_stratified_random_split(
    dataset: BaseNeuralDataset, fraction: float, seed: int = 1
) -> tuple[list[int], list[int]]:
    """Per-day stratified split via torch.randperm, one shared generator across days (not
    re-seeded per day)."""
    split_1_idxs: list[int] = []
    split_2_idxs: list[int] = []
    days = torch.tensor([dataset[idx]["day"] for idx in range(len(dataset))])
    all_idxs = torch.arange(len(dataset))
    for day in days.unique():
        day_idxs = all_idxs[days == day]
        num_samples = int(len(day_idxs) * fraction)
        perm = torch.randperm(
            len(day_idxs), generator=torch.Generator().manual_seed(seed)
        )
        split_1_idxs += day_idxs[perm[:num_samples]].tolist()
        split_2_idxs += day_idxs[perm[num_samples:]].tolist()
    return split_1_idxs, split_2_idxs


@register_dataset("wairagkar")
class WairagkarDataset(Dataset):
    NUM_DAYS: int = 10
    VAL_SPLIT: str = "val"

    def __init__(self, split: str = "train", sbj_idx: int = 0):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self._sbj_idx = sbj_idx
        with open(_PATH, "rb") as f:
            raw = pickle.load(f)
        base = BaseNeuralDataset(raw, has_labels=True, source_dataset="wairagkar")
        train_idxs, val_idxs = session_stratified_random_split(
            base, fraction=0.8, seed=42
        )
        self._base = base
        self._idxs = train_idxs if split == "train" else val_idxs

    def __len__(self) -> int:
        return len(self._idxs)

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[self._idxs[idx]]
        sample["sbj_idx"] = self._sbj_idx
        return sample
