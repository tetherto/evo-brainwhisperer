from dataclasses import replace

from tether_scratch.dataloader import PadSpec, RaggedCollator

from brainwhisperer.data.adapter import to_mixed_neural_batch, to_neural_batch
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.data.collection import MEMBER
from brainwhisperer.data.fields import DAY, NEURAL, PHONEMES, WHISPER
from brainwhisperer.utils.fill import fill_to_num_features


def default_ragged_collator() -> RaggedCollator:
    """The batch padding contract."""
    return RaggedCollator(
        fields={
            NEURAL: PadSpec(axis=-1, pad_value=0),
            # -100 = PyTorch CE ignore index; the old collate masked it in.
            WHISPER: PadSpec(axis=-1, pad_value=-100),
            PHONEMES: PadSpec(axis=-1, pad_value=0),
            # DAY is left unconfigured: every sample carries the same
            # (single-value) shape, so the collator just stacks it.
        }
    )


class MixedRecipeCollator:
    """Collates one batch of the mixed-dataset collection into a NeuralBatch.

    Each row's member (from `MixedRaggedDataset`'s metadata) picks both the
    fill rule (``channel_fills``, the same ``CHANNEL_FILL`` values the
    recipes declare) and its embedder. Padding comes from `default_ragged_collator()`.
    """

    # Shared batch field keys; the collection's rows carry the recipes'
    # data dicts, so these must be the base.py constants.
    NEURAL = NEURAL
    WHISPER = WHISPER
    PHONEMES = PHONEMES
    DAY = DAY

    def __init__(
        self,
        channel_fills: dict[str, str],
        num_features: int = 512,
    ) -> None:
        self._fills = dict(channel_fills)
        self._num_features = num_features
        self._ragged = default_ragged_collator()

    def __call__(self, batch: list) -> NeuralBatch:
        if not batch:
            raise ValueError("MixedRecipeCollator received an empty batch")
        filled = []
        for sample in batch:
            name = sample.metadata[MEMBER]
            if name not in self._fills:
                raise KeyError(
                    f"no CHANNEL_FILL rule for dataset {name!r}; the mixed "
                    f"collator knows {sorted(self._fills)}"
                )
            filled.append(
                replace(
                    sample,
                    data={
                        **sample.data,
                        self.NEURAL: fill_to_num_features(
                            sample.data[self.NEURAL],
                            self._fills[name],
                            self._num_features,
                        ),
                    },
                )
            )
        return to_mixed_neural_batch(
            self._ragged(filled),
            neural=self.NEURAL,
            whisper=self.WHISPER,
            phonemes=self.PHONEMES,
            day=self.DAY,
        )


class SingleRecipeCollator:
    """Collates batches that all belong to ONE recipe into a NeuralBatch.

    The counterpart of `MixedRecipeCollator`: this batch holds a single
    dataset, so the name is fixed at construction and no per-row
    `CHANNEL_FILL` is needed (one recipe has one width).
    """

    def __init__(self, dataset_name: str, recipe, collator: RaggedCollator) -> None:
        self._name = dataset_name
        self._recipe = recipe
        self._collator = collator

    def __call__(self, samples) -> NeuralBatch:
        return to_neural_batch(
            self._collator(samples),
            self._name,
            neural=self._recipe.NEURAL,
            whisper=self._recipe.WHISPER,
            phonemes=self._recipe.PHONEMES,
            day=self._recipe.DAY,
        )
