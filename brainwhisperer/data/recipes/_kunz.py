"""Shared parent for the Kunz 2025 T12/T15 recipes.
Both share two mechanics encoded here:

- a length gate on the postprocessed signal (`<= 3000` bins);
- a fixed train row order. A strict file-table order,
  then (block, trial) ascending within each file: no RNG anywhere.

The day axis is keyed by DATE, not session: several sentenceData files
share one recording date, so the day count is `num_date_ranks`.
"""

from tether_scratch.sessions import SessionRule, kunz_session_rule

from brainwhisperer.data.recipes.base import NEURAL, Recipe

MAX_TRIAL_LEN = 3000


class KunzRecipe(Recipe):
    """T12/T15 commonalities: date-keyed day axis, gate, legacy table order."""

    day_field = "date"

    # Participant tag for kunz_session_rule ("T12"/"T15")
    participant: str
    # Fixed train file table order
    train_file_order: tuple[str, ...]

    def day_rule(self) -> SessionRule:
        return kunz_session_rule(self.participant)

    def _gate(self, dataset):
        """Drop trials longer than MAX_TRIAL_LEN after postprocessing."""
        keep = [
            i
            for i in range(len(dataset))
            if dataset[i].data[NEURAL].shape[-1] <= MAX_TRIAL_LEN
        ]
        if len(keep) != len(dataset):
            dataset = dataset.select(keep)
        return dataset

    def _reorder_to_legacy_table(self, dataset, events):
        """Reorder the segmenter's study order."""
        sentence_rows = events[
            (events["type"] == "Sentence") & (events["split"] == "train")
        ]
        keys = [
            (str(s), int(b), int(t))
            for s, b, t in zip(
                sentence_rows["session"],
                sentence_rows["block"],
                sentence_rows["trial_id"],
                strict=True,
            )
        ]
        assert len(keys) == len(dataset), (
            f"segment rows ({len(dataset)}) diverge from the chain's train "
            f"Sentence rows ({len(keys)})"
        )
        unknown = {k[0] for k in keys} - set(self.train_file_order)
        assert not unknown, f"train files outside the builder table: {unknown}"
        positions = {k: i for i, k in enumerate(keys)}
        legacy = sorted(
            keys, key=lambda k: (self.train_file_order.index(k[0]), k[1], k[2])
        )
        return dataset.select([positions[k] for k in legacy])
