import pickle

from torch.utils.data import Dataset

from brainwhisperer.config.paths import KUNZ_ROOT
from brainwhisperer.data.registry import register_dataset
from brainwhisperer.data.datasets.base import BaseNeuralDataset

_BASE = "data/kunz_dataset_for_BaseNeuralTextDataset/padded_feature_dim"

_SUBJECT_CONFIG: dict[str, dict] = {
    "t12": {
        "sbj_idx": 2,
        "splits": {
            "train": f"{_BASE}/kunz_t12_train.pkl",
            "val":   f"{_BASE}/kunz_t12_val.pkl",
        },
    },
    "t15": {
        "sbj_idx": 3,
        "splits": {
            "train": f"{_BASE}/kunz_t15_train.pkl",
            "val":   f"{_BASE}/kunz_t15_val.pkl",
        },
    },
}


class KunzDataset(Dataset):
    SUBJECT: str  # set by subclasses

    def __init__(self, split: str = "train"):
        config = _SUBJECT_CONFIG[self.SUBJECT]
        if split not in config["splits"]:
            raise ValueError(
                f"split must be one of {list(config['splits'])}, got {split!r}"
            )
        self.sbj_idx: int = config["sbj_idx"]
        path = f"{KUNZ_ROOT}/{config['splits'][split]}"
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self._base = BaseNeuralDataset(raw, has_labels=True, source_dataset="card")

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict:
        sample = self._base[idx]
        sample["sbj_idx"] = self.sbj_idx
        return sample


@register_dataset("kunz_t12")
class KunzT12Dataset(KunzDataset):
    SUBJECT = "t12"


@register_dataset("kunz_t15")
class KunzT15Dataset(KunzDataset):
    SUBJECT = "t15"
