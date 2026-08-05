import pickle

from torch.utils.data import Dataset

from brainwhisperer.config.paths import CARD_ROOT
from brainwhisperer.data.datasets.base import BaseNeuralDataset
from brainwhisperer.data.registry import register_dataset
from brainwhisperer.utils.sessions import CARD_SESSIONS

_SPLITS = {
    "train": "data/cleaned_train_data.pkl",
    "val": "data/cleaned_val_data.pkl",
    "test": "data/cleaned_test_data.pkl",
}


@register_dataset("card")
class CardDataset(Dataset):
    NUM_DAYS: int = len(CARD_SESSIONS)

    def __init__(self, split: str = "train", sbj_idx: int = 0):
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {list(_SPLITS)}, got {split!r}")
        self._sbj_idx = sbj_idx
        path = f"{CARD_ROOT}/{_SPLITS[split]}"
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self._base = BaseNeuralDataset(
            raw, has_labels=split != "test", source_dataset="card"
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[idx]
        sample["sbj_idx"] = self._sbj_idx
        return sample
