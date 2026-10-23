"""Willett 2023 recipe"""

import neuralset as ns
from tether_scratch import transforms
from tether_scratch.datasets.utils import dataset_dir
from tether_scratch.datasets.willet_2023 import study as willet_study
from tether_scratch.datasets.willet_2023 import utils as willet_utils
from tether_scratch.extractors import NeuralFeatureExtractor
from tether_scratch.sessions import SessionRule, willet_session_rule

from brainwhisperer.data.recipes.base import Recipe

_TEST_QUERY = "original_split == 'test'"
_TRAIN_QUERY = "original_split == 'train'"


_FEATURE_RATE_HZ = 50.0

_CHANNELS_6V = tuple(
    int(j)
    for _, j in sorted(
        (int(col), j)
        for j, col in enumerate(willet_utils.CHANNEL_ORDER)
        if int(col) < 128
    )
)


class WilletRecipe(Recipe):
    """Willett 2023: tx1 + spike-band power, day axis = 24 sessions."""

    CHANNEL_FILL = "duplicate"

    def day_rule(self) -> SessionRule:
        return willet_session_rule()

    def build_chain(self, split: str = "val") -> ns.Chain:
        """Study + Willett's z-score + split assignment (test or train).
        ``RescaleNeural`` fits on the chain's own rows,
        """
        study_query = _TEST_QUERY if split == "val" else _TRAIN_QUERY
        mapping = {"test": "val"} if split == "val" else {"train": "train"}
        return ns.Chain(
            steps={
                "study": willet_study.Willett2023Sentences(
                    path=dataset_dir("willet_2023") / "data",
                    query=study_query,
                    threshold_variant="tx1",
                ),
                "rescale": transforms.RescaleNeural(
                    stat_row_query="type == 'Sentence'",
                    trim_leading_bins=1,
                    guard="eps",
                ),
                "split": transforms.AssignSourceSplit(mapping=mapping),
            }
        )

    def neural_extractor(self) -> NeuralFeatureExtractor:
        """(256, T) per trial: 128 area-6v threshold crossings (tx1) + 128
        spike-band power channels in file-column order, concatenated in
        event order (time-first; the collator pads
        time)."""
        return NeuralFeatureExtractor(
            target_unit="native",
            frequency=_FEATURE_RATE_HZ,
            channels=_CHANNELS_6V,
            trim_leading_bins=1,
            aggregation="cat",
        )
