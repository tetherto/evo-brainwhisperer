import pickle

from torch.utils.data import Dataset

from brainwhisperer.config.paths import KUNZ_ROOT
from brainwhisperer.data.registry import register_dataset
from brainwhisperer.data.datasets.base import BaseNeuralDataset
from brainwhisperer.utils.sessions import KUNZ_T12_SESSIONS, KUNZ_T15_SESSIONS

_BASE = "data/kunz_dataset_for_BaseNeuralTextDataset/padded_feature_dim"

_SPLITS: dict[str, dict[str, str]] = {
    "t12": {
        "train": f"{_BASE}/kunz_t12_train.pkl",
        "val":   f"{_BASE}/kunz_t12_val.pkl",
    },
    "t15": {
        "train": f"{_BASE}/kunz_t15_train.pkl",
        "val":   f"{_BASE}/kunz_t15_val.pkl",
    },
}


class KunzDataset(Dataset):
    SUBJECT: str   # set by subclasses
    NUM_DAYS: int  # set by subclasses

    def __init__(self, split: str = "train", sbj_idx: int = 0):
        splits = _SPLITS[self.SUBJECT]
        if split not in splits:
            raise ValueError(
                f"split must be one of {list(splits)}, got {split!r}"
            )
        self._sbj_idx = sbj_idx
        path = f"{KUNZ_ROOT}/{splits[split]}"
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self._base = BaseNeuralDataset(raw, has_labels=True, source_dataset="card")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[idx]
        sample["sbj_idx"] = self._sbj_idx
        return sample


@register_dataset("kunz_t12")
class KunzT12Dataset(KunzDataset):
    SUBJECT = "t12"
    NUM_DAYS: int = len(KUNZ_T12_SESSIONS)


@register_dataset("kunz_t15")
class KunzT15Dataset(KunzDataset):
    SUBJECT = "t15"
    NUM_DAYS: int = len(KUNZ_T15_SESSIONS)
