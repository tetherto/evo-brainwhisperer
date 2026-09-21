import torch
from torch.utils.data import Dataset


class BaseNeuralDataset(Dataset):
    """
    Wraps a raw pickle dict into per-sample dicts.

    source_dataset="willet": doubles the channel dim to match the 512-channel CARD layout.
    eval_type="val":  returns features + text labels (used for train and val splits).
    eval_type="test": returns features only, no labels.
    """

    def __init__(
        self, data_dict: dict, has_labels: bool = True, source_dataset: str = "card"
    ):
        self.neural_features = data_dict["neural_features"]
        self.n_time_steps = data_dict["n_time_steps"]
        self.seq_class_ids = data_dict["seq_class_ids"]
        self.seq_len = data_dict["seq_len"]
        self.sentence_label = data_dict["sentence_label"]
        self.transcriptions = data_dict["sentence_label"]
        self.days = torch.tensor(data_dict["dayIdx"], dtype=torch.long)
        self.has_labels = has_labels
        self.source_dataset = source_dataset

    def __len__(self) -> int:
        return len(self.neural_features)

    def __getitem__(self, idx: int) -> dict:
        neural_features = torch.tensor(self.neural_features[idx], dtype=torch.float32)

        # Willet electrodes cover 256 channels; duplicate to fill the 512-channel layout
        if self.source_dataset == "willet":
            neural_features = torch.cat([neural_features, neural_features], dim=1)

        # Zero-padding for Jude dataset
        if self.source_dataset == "jude":
            neural_features = torch.nn.functional.pad(neural_features, (0, 256))

        if self.has_labels:
            seq_class_ids = self.seq_class_ids[idx]
            seq_class_ids = [
                p.item() for p in seq_class_ids if p != 0
            ]  # strip zero padding
            return {
                "neural_features": neural_features,
                "n_time_steps": self.n_time_steps[idx],
                "seq_class_ids": torch.tensor(seq_class_ids, dtype=torch.long),
                "seq_len": torch.tensor(self.seq_len[idx], dtype=torch.long),
                "sentence_label": self.sentence_label[idx],
                "transcriptions": self.transcriptions[idx],
                "day": self.days[idx],
                "source_dataset": self.source_dataset,
            }
        else:  # no labels in this split
            return {
                "neural_features": neural_features,
                "n_time_steps": self.n_time_steps[idx],
                "day": self.days[idx],
                "source_dataset": self.source_dataset,
            }
