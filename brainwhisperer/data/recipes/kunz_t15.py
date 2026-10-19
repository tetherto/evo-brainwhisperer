"""Kunz T15 recipe: the NeuralSet pipeline for the T15 eval line (50 trials)."""

import neuralset as ns
from tether_scratch import transforms
from tether_scratch.datasets.kunz_2025 import study as kunz_study
from tether_scratch.datasets.utils import dataset_dir
from tether_scratch.extractors import NeuralFeatureExtractor

from brainwhisperer.data.recipes._kunz import KunzRecipe
from brainwhisperer.data.recipes.replay import KUNZ_T15_PUBLISHED_CALENDAR_DAYS
from brainwhisperer.data.rescale import KunzPairedBlockRescale

_VAL_QUERY = (
    "subject == 'Kunz2025InnerSpeech/T15' "
    "and original_split == 'eval' and vocab == '50w'"
)

_TRAIN_QUERY = "subject == 'Kunz2025InnerSpeech/T15' and original_split == 'train'"

# Setting train file order explicitly
_TRAIN_FILE_ORDER = (
    "t15.2024.03.03_attempted_50words",
    "t15.2024.03.03_imagined_50words",
    "t15.2024.12.15_imagined_125kwords",
    "t15.2025.01.05_imagined_125kwords",
)

_FEATURE_RATE_HZ = 50.0

# Setting channel order over each stream's 256 rows (area 6v
# superior/inferior, then 44s, then the rest)
_BUILDER_CHANNEL_ORDER = (
    tuple(range(128, 192))
    + tuple(range(64, 128))
    + tuple(range(0, 64))
    + tuple(range(192, 256))
)


class KunzT15Recipe(KunzRecipe):
    """T15: paired 512-channel extractor with the builder's channel order."""

    participant = "T15"
    train_file_order = _TRAIN_FILE_ORDER
    CHANNEL_FILL = "none"

    def build_chain(self, split: str = "val") -> ns.Chain:
        """Study + paired-block rescale + split assignment."""
        study_query = _VAL_QUERY if split == "val" else _TRAIN_QUERY
        mapping = {"eval": "val"} if split == "val" else {"train": "train"}
        return ns.Chain(
            steps={
                "study": kunz_study.Kunz2025InnerSpeech(
                    path=dataset_dir("kunz_2025") / "data",
                    query=study_query,
                ),
                "rescale": KunzPairedBlockRescale(),
                "split": transforms.AssignSourceSplit(mapping=mapping),
            }
        )

    def neural_extractor(self) -> NeuralFeatureExtractor:
        """(512, T) per trial: 256 threshold crossings + 256 band-power
        channels in the builder's order, paired from each trial's delay
        start (sums for spikes, means for power), concatenated in event
        order."""
        return NeuralFeatureExtractor(
            target_unit="native",
            frequency=_FEATURE_RATE_HZ,
            pair_trigger_field="pair_start",
            channels=_BUILDER_CHANNEL_ORDER,
            aggregation="cat",
        )

    def calendar_days(self) -> list[int]:
        """REPLAY PIN, not data: the published embedder trained with all
        three T15 dates as calendar day 0 (see replay.py and ADR 0005).

        The session rule derives [0, 287, 308]; adopting it changes the
        month-bucket sizing from 1 to 11 slots and invalidates the
        published checkpoints — that is the planned retrain's decision,
        not this pipeline's.
        """
        return list(KUNZ_T15_PUBLISHED_CALENDAR_DAYS)

    def postprocess_val(self, dataset, events):
        dataset.prepare()
        # Gate on the paired length (paired == n_time_steps for T15).
        return self._gate(dataset)

    def postprocess_train(self, dataset, events):
        dataset.prepare()
        # The reorder runs before the gate, mirroring the old class
        # (gated, table-ordered pickle).
        return self._gate(self._reorder_to_legacy_table(dataset, events))
