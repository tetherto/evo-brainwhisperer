import pickle

import torch.nn.functional as F
from torch.utils.data import Dataset

from brainwhisperer.config.paths import KUNZ_ROOT
from brainwhisperer.data.datasets.base import BaseNeuralDataset
from brainwhisperer.data.registry import register_dataset

_BASE = "data/kunz_dataset_for_BaseNeuralTextDataset/sentenceData_eval_split"

_SPLITS: dict[str, dict[str, str]] = {
    "t12": {
        "train": f"{_BASE}/kunz_t12_train.pkl",
        "val": f"{_BASE}/kunz_t12_val_50words.pkl",
    },
    "t15": {
        "train": f"{_BASE}/kunz_t15_train.pkl",
        "val": f"{_BASE}/kunz_t15_val_50words.pkl",
    },
}


class KunzDataset(Dataset):
    SUBJECT: str  # set by subclasses
    NUM_DAYS: int  # set by subclasses
    VAL_SPLIT: str = "val"

    def __init__(self, split: str = "train"):
        splits = _SPLITS[self.SUBJECT]
        if split not in splits:
            raise ValueError(f"split must be one of {list(splits)}, got {split!r}")
        self._split = split
        path = f"{KUNZ_ROOT}/{splits[split]}"
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self._base = BaseNeuralDataset(
            raw, has_labels=True, source_dataset=f"kunz_{self.SUBJECT}"
        )
        # Filter out trials longer than 3000 time steps (too long for Whisper).
        # For T12 val, length is checked post-interpolation (see _postprocess) --
        # matches REFERENCE, where the interpolating wrapper is applied before filtering.
        self._idxs = [
            i
            for i in range(len(self._base))
            if self._postprocess(self._base[i])["n_time_steps"] <= 3000
        ]

    def _postprocess(self, sample: dict) -> dict:
        # T12 closed-loop val trials run ~2x longer than open-loop ones; REFERENCE
        # interpolates them down to match, val only -- train and T15 are untouched.
        if self.SUBJECT != "t12" or self._split != "val":
            return sample
        n = sample["n_time_steps"] // 2
        seg = sample["neural_features"]
        seg = (
            F.interpolate(
                seg.T.unsqueeze(0), size=n, mode="linear", align_corners=False
            )
            .squeeze()
            .T
        )
        sample["neural_features"] = seg
        sample["n_time_steps"] = n
        return sample

    def __len__(self) -> int:
        return len(self._idxs)

    def __getitem__(self, idx: int) -> dict:
        sample = self._postprocess(self._base[self._idxs[idx]])
        sample["dataset"] = self.NAME
        return sample


@register_dataset("kunz_t12")
class KunzT12Dataset(KunzDataset):
    SUBJECT = "t12"
    NUM_DAYS: int = 1


@register_dataset("kunz_t15")
class KunzT15Dataset(KunzDataset):
    SUBJECT = "t15"
    NUM_DAYS: int = 3
