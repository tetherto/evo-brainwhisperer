import pickle

from torch.utils.data import Dataset

from brainwhisperer.config.paths import NEJM_ROOT
from brainwhisperer.data.registry import register_dataset
from brainwhisperer.data.datasets.base import BaseNeuralDataset

_SPLITS = {
    "train": "data/cleaned_train_data.pkl",
    "val":   "data/cleaned_val_data.pkl",
    "test":  "data/cleaned_test_data.pkl",
}


@register_dataset("nejm")
class NEJMDataset(Dataset):
    SBJ_IDX = 0

    def __init__(self, split: str = "train"):
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {list(_SPLITS)}, got {split!r}")
        path = f"{NEJM_ROOT}/{_SPLITS[split]}"
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self._base = BaseNeuralDataset(raw, has_labels=split != "test", source_dataset="card")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[idx]
        sample["sbj_idx"] = self.SBJ_IDX
        return sample
