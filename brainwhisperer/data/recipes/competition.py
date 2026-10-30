"""Competition recipes: the held-out sets of the Brain-to-Text benchmarks.

- Card: the ``test`` split of Card 2024, scored on Kaggle (Brain-to-Text '25).
- Willett: ``competitionHoldOut`` of Willett 2023, scored on EvalAI
  (Brain-to-Text '24), read through `brainwhisperer.data.studies`.

Labels are withheld, so these sets have no ``Sentence`` events: segments
trigger on a per-trial marker and carry only the neural signal and the day
index.
"""

import csv
from pathlib import Path

import neuralset as ns
from neuralshed import transforms
from neuralshed.dataloader import RaggedSegmenter
from neuralshed.datasets.card_2024 import study as card_study
from neuralshed.datasets.utils import dataset_dir
from neuralshed.extractors import NeuralFeatureExtractor

from brainwhisperer.data.fields import DAY, NEURAL
from brainwhisperer.data.recipes.card import CardRecipe
from brainwhisperer.data.recipes.willet import (
    _CHANNELS_6V,
    _FEATURE_RATE_HZ,
    WilletRecipe,
)
from brainwhisperer.data.studies import WillettCompetitionData


class _CompetitionMixin:
    """Label-free segmenting shared by the competition recipes."""

    #: Event type marking one trial; segments trigger on it.
    TRIGGER: str
    #: Event type the day index is read from. The neural stream: one row per
    #: window, whereas back-to-back trial markers can touch a neighbour's
    #: window through float rounding and collide.
    DAY_SOURCE: str = "SpikeCounts"
    #: Trials the scorer expects, one prediction each.
    N_TRIALS: int

    def competition_chain(self) -> ns.Chain:
        raise NotImplementedError

    def write_submission(self, out_dir: Path, texts: list[str]) -> Path:
        """Write `texts` (row i = trial i) in the scorer's format; return the file."""
        raise NotImplementedError

    def competition_extractor(self) -> NeuralFeatureExtractor:
        """Defaults to the training extractor."""
        return self.neural_extractor()

    def competition_segmenter(self) -> RaggedSegmenter:
        return RaggedSegmenter(
            extractors={
                NEURAL: self.competition_extractor(),
                DAY: self._rule.day_index_field(
                    field=self.day_field, event_types=self.DAY_SOURCE
                ),
            },
            trigger_query=f"type == '{self.TRIGGER}'",
            duration=None,  # each segment inherits its trial window
            metadata_columns=("session", "block", "trial_id"),
        )

    def build_competition_dataset(self, chain: ns.Chain | None = None):
        """The held-out trials as a prepared ragged segment dataset, in
        submission order (sessions sorted, then file order)."""
        events = (chain or self.competition_chain()).run()
        dataset = self.competition_segmenter().apply(events)
        dataset.prepare()
        return dataset


class CardCompetitionRecipe(_CompetitionMixin, CardRecipe):
    """Card 2024 ``test`` split (1,450 trials)."""

    TRIGGER = "SpikeCounts"
    N_TRIALS = 1450

    def write_submission(self, out_dir: Path, texts: list[str]) -> Path:
        """Kaggle CSV: header ``id,text``, ids 0..N-1."""
        path = Path(out_dir) / "card_kaggle.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "text"])
            writer.writerows(enumerate(texts))
        return path

    def competition_chain(self) -> ns.Chain:
        # The source files are already z-scored, as for training: no rescale.
        return ns.Chain(
            steps={
                "study": card_study.Card2024Brain2Text(
                    path=dataset_dir("card_2024") / "data",
                    query="original_split == 'test'",
                ),
            }
        )


class WilletCompetitionRecipe(_CompetitionMixin, WilletRecipe):
    """Willett 2023 ``competitionHoldOut`` (1,200 trials)."""

    TRIGGER = "Stimulus"
    N_TRIALS = 1200

    def write_submission(self, out_dir: Path, texts: list[str]) -> Path:
        """EvalAI text file: one sentence per line."""
        path = Path(out_dir) / "willet_evalai.txt"
        path.write_text("".join(f"{t}\n" for t in texts), encoding="utf-8")
        return path

    def competition_chain(self, partition: str = "competitionHoldOut") -> ns.Chain:
        """``partition="test"`` reads the labelled test files, for parity checks."""
        return ns.Chain(
            steps={
                "study": WillettCompetitionData(
                    path=dataset_dir("willet_2023") / "data",
                    partition=partition,
                ),
                # Block-wise z-score as in training, pooled over the trial
                # markers; the files already drop the first bin of each trial.
                "rescale": transforms.RescaleNeural(
                    stat_row_query=f"type == '{self.TRIGGER}'",
                    trim_leading_bins=0,
                    guard="eps",
                ),
            }
        )

    def competition_extractor(self) -> NeuralFeatureExtractor:
        """The training extractor without its leading-bin trim."""
        return NeuralFeatureExtractor(
            target_unit="native",
            frequency=_FEATURE_RATE_HZ,
            channels=_CHANNELS_6V,
            trim_leading_bins=0,
            aggregation="cat",
        )
