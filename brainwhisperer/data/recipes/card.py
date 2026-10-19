"""Card 2024 recipe"""

import neuralset as ns
from tether_scratch import transforms
from tether_scratch.datasets.card_2024 import study as card_study
from tether_scratch.datasets.utils import dataset_dir
from tether_scratch.extractors import NeuralFeatureExtractor
from tether_scratch.sessions import SessionRule, card_session_rule

from brainwhisperer.data.recipes.base import Recipe

# The source's own split label (the column `AssignSourceSplit` reads).
_VAL_QUERY = "original_split == 'val'"
_TRAIN_QUERY = "original_split == 'train'"

# Card's pre-binned feature rate; `target_unit="native"` keeps the source
# values, so no rescaling happens here.
_FEATURE_RATE_HZ = 50.0


class CardRecipe(Recipe):
    """Card 2024: 512 channels native, day axis = 45 chronological sessions."""

    CHANNEL_FILL = "none"

    def day_rule(self) -> SessionRule:
        return card_session_rule()

    def build_chain(self, split: str = "val") -> ns.Chain:
        """Study + split assignment, filtered to Card's val or train trials."""
        query = _VAL_QUERY if split == "val" else _TRAIN_QUERY
        return ns.Chain(
            steps={
                "study": card_study.Card2024Brain2Text(
                    path=dataset_dir("card_2024") / "data",
                    query=query,
                ),
                "split": transforms.AssignSourceSplit(
                    mapping={"train": "train", "val": "val"}
                ),
            }
        )

    def neural_extractor(self) -> NeuralFeatureExtractor:
        """(512, T) per trial: 256 threshold crossings + 256 band-power
        channels, concatenated in event order (time-first; the collator pads the time axis)."""
        return NeuralFeatureExtractor(
            target_unit="native",
            frequency=_FEATURE_RATE_HZ,
            aggregation="cat",
        )
