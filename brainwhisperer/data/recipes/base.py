"""Shared skeleton for the per-dataset recipes.

A `recipe` fixes the pipeline every dataset runs (study chain -> segmenter ->
postprocess -> prepared ragged segment dataset) and leaves the
dataset-specific steps as hooks:

- `build_chain(split)`     study + rescale + split assignment
- `neural_extractor()`     channel layout / pairing / rates
- `day_rule()`             the `tether_scratch` SessionRule behind the day axis
- `postprocess_val/train`  length gates, halving, row-order pins
- `calendar_days()`        per-day-index calendar metadata the embedders see

`NUM_DAYS` and `DAY` extraction derive from the session rule, so
`tether_scratch.sessions` is the single source of truth for the day axis;
a dataset whose day axis is not calendar-backed (Wairagkar: day == block)
overrides instead of faking a rule.
"""

import functools
from abc import ABC, abstractmethod

import neuralset as ns
from tether_scratch.dataloader import RaggedSegmenter
from tether_scratch.extractors import (
    NeuralFeatureExtractor,
    PhonemeEncoder,
    WhisperLabelEncoder,
)
from tether_scratch.sessions import SessionRule

from brainwhisperer.data.fields import DAY, NEURAL, PHONEMES, WHISPER


class Recipe(ABC):
    """The per-dataset pipeline: chain -> segmenter -> postprocess.

    Instances are stateless and registered in `recipes.__init__` under their
    dataset name; the model keys its embedders by that name.
    """

    # Batch field keys (see brainwhisperer/data/fields.py): the keys of
    # this recipe's batches' `data` dict, exposed for the adapter.
    NEURAL = NEURAL
    WHISPER = WHISPER
    PHONEMES = PHONEMES
    DAY = DAY

    #: Day-axis key read by the session rule: "session" (position in the
    #: chronological session list) or "date" (rank of the distinct date).
    day_field: str = "session"

    #: Fill-to-512 rule (ADR 0001): "none", "zeros" or "duplicate".
    CHANNEL_FILL: str = "none"

    # ------------------------------------------------------------------
    # Per-dataset hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def build_chain(self, split: str = "val") -> ns.Chain:
        """Study + rescale + split assignment, filtered to `split`.

        Exposed separately from `build_*_dataset` so parity checks can
        verify trial identity without extracting any features.
        """

    @abstractmethod
    def neural_extractor(self) -> NeuralFeatureExtractor:
        """This dataset's channel layout, pairing and rate as a segmenter field."""

    def day_rule(self) -> SessionRule | None:
        """The session rule behind this dataset's day axis, or None.

        Returning None means the day axis is not calendar-backed; such
        datasets override `day_extractor` and `calendar_days` themselves.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no day rule; override day_rule() "
            "or (calendar-less) day_extractor()/calendar_days()"
        )

    def day_extractor(self):
        """The session rule as the segmenter's DAY field extractor.

        Built fresh per dataset — `collection.py` reads the mapping back
        off each prepared dataset's extractors, so instances must not be
        shared across builds.
        """
        return self._rule.day_index_field(field=self.day_field)

    def calendar_days(self) -> list[int]:
        """Days since the first recording session, indexed by day index.

        Sizes the embedders' month buckets / sinusoidal day positions
        (`model.py` derives `session_to_month` from these), so the list is
        part of the model's parameter shape for saved checkpoints.
        """
        rule = self._rule
        if self.day_field == "date":
            dates = sorted(set(rule.dates))
            first = dates[0]
            return [(d - first).days for d in dates]
        return [rule.calendar_day[s] for s in rule.sessions]

    def postprocess_val(self, dataset, events):
        """Default: just prepare. Overrides attach gates/reorders BEFORE
        or AFTER `prepare()`, exactly where the old loader ran them."""
        dataset.prepare()
        return dataset

    def postprocess_train(self, dataset, events):
        dataset.prepare()
        return dataset

    # ------------------------------------------------------------------
    # Shared skeleton
    # ------------------------------------------------------------------
    @functools.cached_property
    def _rule(self) -> SessionRule:
        """Memoized day_rule(): rule construction scans dataset metadata,
        and the rule is immutable, so per-run derivations share one build."""
        return self.day_rule()

    @property
    def NUM_DAYS(self) -> int:
        """Day-axis size of this dataset's embedder, from the session rule."""
        rule = self._rule
        return rule.num_date_ranks if self.day_field == "date" else rule.num_days

    def segmenter(self, max_label_len: int) -> RaggedSegmenter:
        """Trial-windowed segmenter; the label fields are identical for
        every dataset, so only NEURAL and DAY come from the hooks.

        `max_label_len` is the run's `data.max_seq_len`: it reproduces the
        old collate's Whisper-label truncation.
        """
        return RaggedSegmenter(
            extractors={
                NEURAL: self.neural_extractor(),
                # Whisper ids with BOS stripped, truncated at max_label_len —
                # same contract as the old collate.py.
                WHISPER: WhisperLabelEncoder(
                    aggregation="trigger", max_length=max_label_len
                ),
                PHONEMES: PhonemeEncoder(aggregation="trigger"),
                DAY: self.day_extractor(),
            },
            trigger_query="type == 'Sentence'",
            duration=None,  # each segment inherits its sentence's trial window
            metadata_columns=("split",),
        )

    def build_val_dataset(self, max_label_len: int):
        """This dataset's val trials as a prepared ragged segment dataset."""
        events = self.build_chain(split="val").run()
        dataset = self.segmenter(max_label_len).apply(events)
        return self.postprocess_val(dataset, events)

    def build_train_dataset(self, max_label_len: int):
        """This dataset's train trials as a prepared ragged segment dataset."""
        events = self.build_chain(split="train").run()
        dataset = self.segmenter(max_label_len).apply(events)
        return self.postprocess_train(dataset, events)
