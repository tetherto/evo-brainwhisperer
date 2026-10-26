"""Jude 2026 intended-speech recipe"""

import neuralset as ns
from neuralset.events.transforms.basic import QueryEvents
from neuralshed.datasets.jude_2026_speech.study import Jude2026IntendedSpeech
from neuralshed.datasets.utils import dataset_dir
from neuralshed.extractors import NeuralFeatureExtractor
from neuralshed.sessions import SessionRule, jude_session_rule

from brainwhisperer.data.recipes.base import Recipe
from brainwhisperer.data.rescale import JudeLastBlockRescale
from brainwhisperer.data.splits import held_out_split

# The five published-speech sessions
_SESSIONS = (
    "t17.2024.04.15",
    "t17.2024.04.16",
    "t17.2024.04.17",
    "t17.2024.05.06",
    "t17.2024.05.07",
)
_PUBLISHED_QUERY = "session in [" + ", ".join(repr(s) for s in _SESSIONS) + "]"

_FEATURE_RATE_HZ = 50.0

_CHANNELS_6V = tuple(range(128))


class JudeRecipe(Recipe):
    """Jude 2026: paired 256-channel extractor, day axis = 5 sessions."""

    CHANNEL_FILL = "zeros"

    def day_rule(self) -> SessionRule:
        return jude_session_rule()

    def build_chain(self, split: str = "val") -> ns.Chain:
        """Study, last-block rescale, and pinned split selection."""
        return ns.Chain(
            steps={
                "study": Jude2026IntendedSpeech(
                    path=dataset_dir("jude_2026_speech") / "data",
                    query=_PUBLISHED_QUERY,
                ),
                "rescale": JudeLastBlockRescale(),
                "split": held_out_split("jude"),
                # Segment triggers read the Sentence rows: keep every row
                # of this split's trials, drop only the other split's
                # Sentence rows.
                "drop_other_cues": QueryEvents(
                    query=f"type != 'Sentence' or split == {split!r}"
                ),
            }
        )

    def neural_extractor(self) -> NeuralFeatureExtractor:
        """(256, T) per trial: 128 threshold crossings + 128 band-power
        channels, paired from each trial's own cue start (sums for spikes,
        means for power), concatenated in event order (time-first)."""
        return NeuralFeatureExtractor(
            target_unit="native",
            frequency=_FEATURE_RATE_HZ,
            pair_trigger_field="start",
            channels=_CHANNELS_6V,
            aggregation="cat",
        )
