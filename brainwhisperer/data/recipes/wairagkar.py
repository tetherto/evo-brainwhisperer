"""Wairagkar 2025 control recipe"""

import neuralset as ns
from neuralset.events.transforms.basic import QueryEvents
from neuralset.extractors import LabelEncoder
from tether_scratch.datasets.utils import dataset_dir
from tether_scratch.datasets.wairagkar_2025.control_study import (
    Wairagkar2025ControlExperiment,
)
from tether_scratch.extractors import NeuralFeatureExtractor

from brainwhisperer.data.recipes.base import Recipe
from brainwhisperer.data.rescale import WairagkarBlockRescale
from brainwhisperer.data.splits import held_out_split, load_train_keys

#: The ten recording blocks; the day axis and its ordinal clock.
NUM_BLOCKS = 10


class BlockDayEncoder(LabelEncoder):
    """The Wairagkar day tag: the block number itself, dayIdx == block."""

    def prepare(self, obj) -> None:
        if self._label_to_ind:
            return
        labels = self._get_field_values(obj)
        self._label_to_ind = {int(label): int(label) for label in labels}
        self._n_classes = len(self._label_to_ind)
        events = self._event_types_helper.extract(obj)
        self(
            events[0],
            events[0].start,
            duration=0.001,
            trigger=events[0],
        )


class WairagkarRecipe(Recipe):
    """Wairagkar: no session rule (day == block); pinned train order."""

    # Sizes the embedder's day axis; there is no SessionRule to derive it
    # from (the day axis is the block number, not a recording session).
    NUM_DAYS = NUM_BLOCKS
    CHANNEL_FILL = "none"

    def day_rule(self):
        # The day axis is not calendar-backed; day_extractor/calendar_days
        # are overridden with the block-ordinal equivalents.
        return None

    def day_extractor(self):
        return BlockDayEncoder(event_types="Sentence", event_field="block")

    def calendar_days(self) -> list[int]:
        """The ordinal clock: block ``i`` stands in for calendar day ``i``."""
        return list(range(self.NUM_DAYS))

    def build_chain(self, split: str = "val") -> ns.Chain:
        """Study, block-pooled rescale, and pinned split selection."""
        return ns.Chain(
            steps={
                "study": Wairagkar2025ControlExperiment(
                    path=dataset_dir("wairagkar_2025"),
                ),
                "rescale": WairagkarBlockRescale(),
                "split": held_out_split("wairagkar"),
                "drop_other_cues": QueryEvents(
                    query=f"type != 'Sentence' or split == {split!r}"
                ),
            }
        )

    def neural_extractor(self) -> NeuralFeatureExtractor:
        """(512, T) per trial: 256 spike-band-power + 256 threshold
        crossings, concatenated in event order (time-first; the collator pads the time axis). The release's 100 Hz
        bins are consumed verbatim."""
        return NeuralFeatureExtractor(
            target_unit="native",
            frequency="native",
            aggregation="cat",
        )

    def postprocess_train(self, dataset, events):
        """Prepare, then reorder the block-major study rows into the
        published run's permuted train order.
        """
        dataset.prepare()

        pin = load_train_keys("wairagkar")
        assert len(pin) == len(dataset) == 190, (
            f"train pin has {len(pin)} rows, dataset has {len(dataset)}"
        )
        positions = {
            (k["block"], k["trial"]): i
            for i, k in enumerate(sorted(pin, key=lambda k: (k["block"], k["trial"])))
        }
        order = [positions[(k["block"], k["trial"])] for k in pin]
        return dataset.select(order)
