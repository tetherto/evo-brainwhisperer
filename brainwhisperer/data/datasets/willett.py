import pickle

from torch.utils.data import Dataset

from brainwhisperer.config.paths import CARD_ROOT
from brainwhisperer.data.datasets.base import BaseNeuralDataset
from brainwhisperer.data.registry import register_dataset
from brainwhisperer.utils.sessions import WILLETT_SESSIONS

# Willet pickles live in CARD_ROOT alongside the CARD data.
_SPLITS = {
    "train": "data/willet_cleaned_train_data.pkl",
    "test": "data/willet_cleaned_test_data.pkl",
    "competition": "data/willet_cleaned_competition_data.pkl",
}


@register_dataset("willet")
class WilletDataset(Dataset):
    NUM_DAYS: int = len(WILLETT_SESSIONS)
    VAL_SPLIT: str = "test"  # Willet has no "val" split; use test for validation

    def __init__(self, split: str = "train"):
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {list(_SPLITS)}, got {split!r}")
        path = f"{CARD_ROOT}/{_SPLITS[split]}"
        with open(path, "rb") as f:
            raw = pickle.load(f)
        # competition split has no labels; train and test do
        self._base = BaseNeuralDataset(
            raw, has_labels=split != "competition", source_dataset="willet"
        )

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[idx]
        sample["dataset"] = self.NAME
        return sample
