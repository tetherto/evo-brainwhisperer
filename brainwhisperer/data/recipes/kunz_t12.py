"""Kunz T12 recipe: the NeuralSet pipeline for the T12 eval line."""

import neuralset as ns
import torch
import torch.nn.functional as F
from tether_scratch import transforms
from tether_scratch.datasets.kunz_2025 import study as kunz_study
from tether_scratch.datasets.utils import dataset_dir
from tether_scratch.extractors import NeuralFeatureExtractor

from brainwhisperer.data.recipes._kunz import KunzRecipe
from brainwhisperer.data.recipes.base import NEURAL

_VAL_QUERY = (
    "subject == 'Kunz2025InnerSpeech/T12' "
    "and experiment == 'sentenceData' and original_split == 'eval'"
)
_TRAIN_QUERY = (
    "subject == 'Kunz2025InnerSpeech/T12' "
    "and experiment == 'sentenceData' and original_split == 'train'"
)

# Setting train file order explicitly:  ``imagined_50words``
# BEFORE ``attempted_50words`` (the Study's alphabetical session sort is the opposite).
_TRAIN_FILE_ORDER = (
    "t12.2023.11.28_imagined_50words",
    "t12.2023.11.28_attempted_50words",
)

_FEATURE_RATE_HZ = 50.0


class TimeHalving:
    """Halve ``(B, C, T)`` signals along time.
    The Segmenter itself has no transforms field, so this is attached to
    ``dataset.source.transforms`` before ``prepare()``; the ragged view
    applies it per item.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TimeHalving expects a (B, C, T) batch, got {x.shape}")
        return F.interpolate(
            x, size=x.shape[-1] // 2, mode="linear", align_corners=False
        )


class KunzT12Recipe(KunzRecipe):
    """T12: unpaired 256-channel extractor; halving + gate on val."""

    participant = "T12"
    train_file_order = _TRAIN_FILE_ORDER
    CHANNEL_FILL = "zeros"

    def build_chain(self, split: str = "val") -> ns.Chain:
        """Study + delay/go-window z-score + split assignment."""
        study_query = _VAL_QUERY if split == "val" else _TRAIN_QUERY
        mapping = {"eval": "val"} if split == "val" else {"train": "train"}
        return ns.Chain(
            steps={
                "study": kunz_study.Kunz2025InnerSpeech(
                    path=dataset_dir("kunz_2025") / "data",
                    query=study_query,
                ),
                "rescale": transforms.RescaleNeural(
                    stat_row_query="period in ('delay', 'go')",
                    guard="eps",
                ),
                "split": transforms.AssignSourceSplit(mapping=mapping),
            }
        )

    def neural_extractor(self) -> NeuralFeatureExtractor:
        """(256, T) per trial: 128 threshold crossings + 128 band-power
        channels, concatenated in event order."""
        return NeuralFeatureExtractor(
            target_unit="native",
            frequency=_FEATURE_RATE_HZ,
            aggregation="cat",
        )

    def postprocess_val(self, dataset, events):
        # Attach the halving before prepare so per-item views see it
        dataset.source.transforms[NEURAL] = TimeHalving()
        dataset.prepare()
        return self._gate(dataset)

    def postprocess_train(self, dataset, events):
        # No halving on train (it only applies to closed-loop val trials);
        dataset.prepare()
        return self._gate(self._reorder_to_legacy_table(dataset, events))
