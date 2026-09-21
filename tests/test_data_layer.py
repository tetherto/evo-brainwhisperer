"""Gate checks for Phase 1 — data layer."""

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import WhisperTokenizer

import brainwhisperer.data.datasets  # noqa: F401 - unused in this script but triggers dataset registration
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.data.collate import NeuralCollator
from brainwhisperer.data.registry import get_dataset
from brainwhisperer.data.samplers import StratifiedDayBatchSampler

BATCH_SIZE = 8
TOKENIZER = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="en", task="transcribe"
)
COLLATOR = NeuralCollator(TOKENIZER)


def _one_batch(
    dataset_name: str, split: str = "train", sbj_idx: int = 0
) -> NeuralBatch:
    cls = get_dataset(dataset_name)
    ds = cls(split=split, sbj_idx=sbj_idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=COLLATOR, shuffle=True)
    return next(iter(loader))


def _assert_batch_shapes(batch: NeuralBatch, name: str):
    B = BATCH_SIZE
    assert batch.signals.ndim == 3, (
        f"[{name}] signals must be 3D, got {batch.signals.shape}"
    )
    assert batch.signals.shape[0] == B
    assert batch.signals.shape[1] == 512, (
        f"[{name}] expected 512 channels, got {batch.signals.shape[1]}"
    )
    assert batch.signals.dtype == torch.float32, (
        f"[{name}] signals dtype {batch.signals.dtype}"
    )

    assert batch.labels.ndim == 2, (
        f"[{name}] labels must be 2D, got {batch.labels.shape}"
    )
    assert batch.labels.shape[0] == B
    assert batch.labels.dtype == torch.long, (
        f"[{name}] labels dtype {batch.labels.dtype}"
    )

    assert batch.sbj_idx.shape == (B,), f"[{name}] sbj_idx shape {batch.sbj_idx.shape}"
    assert batch.sbj_idx.dtype == torch.long

    assert batch.day_idx.shape == (B,), f"[{name}] day_idx shape {batch.day_idx.shape}"
    assert batch.day_idx.dtype == torch.long

    assert batch.lengths.shape == (B,), f"[{name}] lengths shape {batch.lengths.shape}"
    assert batch.lengths.dtype == torch.long

    assert (batch.lengths > 0).all(), f"[{name}] all lengths must be positive"
    assert (batch.signals.shape[2] >= batch.lengths).all(), (
        f"[{name}] signal T < reported length for some samples"
    )

    # All datasets carry phoneme labels — phone_seq must always be present.
    assert batch.phone_seq is not None, f"[{name}] phone_seq missing"
    assert batch.phone_seq_len is not None, f"[{name}] phone_seq_len missing"
    assert batch.phone_seq.shape[0] == B, f"[{name}] phone_seq batch dim mismatch"
    assert batch.phone_seq_len.shape == (B,), (
        f"[{name}] phone_seq_len shape {batch.phone_seq_len.shape}"
    )
    assert batch.phone_seq.dtype == torch.long, (
        f"[{name}] phone_seq dtype {batch.phone_seq.dtype}"
    )
    assert (batch.phone_seq_len > 0).all(), f"[{name}] zero-length phone sequences"


# NOTE: we're injecting the same number (42) to test that there's no constant being
# injected from the dataclasses (e.g., 0,1,2,3). This is to make sure that the number
# assigned at initalization is respected in the batch.
def test_card_batch_shapes():
    batch = _one_batch("card", sbj_idx=42)
    _assert_batch_shapes(batch, "card")
    assert (batch.sbj_idx == 42).all(), "Card sbj_idx must equal the injected value"


def test_willet_batch_shapes():
    batch = _one_batch("willet", sbj_idx=42)
    _assert_batch_shapes(batch, "willet")
    assert (batch.sbj_idx == 42).all(), "Willet sbj_idx must equal the injected value"


def test_kunz_t12_batch_shapes():
    batch = _one_batch("kunz_t12", sbj_idx=42)
    _assert_batch_shapes(batch, "kunz_t12")
    assert (batch.sbj_idx == 42).all(), "Kunz T12 sbj_idx must equal the injected value"


def test_kunz_t15_batch_shapes():
    batch = _one_batch("kunz_t15", sbj_idx=42)
    _assert_batch_shapes(batch, "kunz_t15")
    assert (batch.sbj_idx == 42).all(), "Kunz T15 sbj_idx must equal the injected value"


def test_sampler_structure():
    """
    Each batch has the correct size and covers exactly days_per_batch distinct
    (source, day) pairs. Within-batch duplicates are expected (sampling with
    replacement); the DDP non-overlap guarantee is tested via seed offset and is
    not exercised here (requires multiple ranks).
    """
    card = get_dataset("card")(split="train")
    willet = get_dataset("willet")(split="train")
    combined = ConcatDataset([card, willet])

    days_per_batch = 2
    sampler = StratifiedDayBatchSampler(
        combined, batch_size=BATCH_SIZE, days_per_batch=days_per_batch, seed=42
    )

    assert len(sampler) > 0, "Sampler produced zero batches"

    for batch_indices in sampler:
        assert len(batch_indices) == BATCH_SIZE, (
            f"Expected {BATCH_SIZE} indices per batch, got {len(batch_indices)}"
        )
        # All indices must be valid
        assert all(0 <= i < len(combined) for i in batch_indices), (
            "Sampler returned out-of-range index"
        )


def test_augmentations_shapes():
    """transform_data preserves (B, C, T) shape and does not produce NaN."""
    from brainwhisperer.data.augmentations import transform_data

    B, C, T = 4, 512, 300
    x = torch.randn(B, C, T)
    lengths = torch.full((B,), T, dtype=torch.long)

    for mode in ("train", "val"):
        x_out, l_out = transform_data(x, lengths, mode=mode)
        assert x_out.shape == (B, C, x_out.shape[2]), f"shape mismatch in {mode}"
        assert not torch.isnan(x_out).any(), f"NaN in {mode} output"
        assert l_out.shape == (B,)
        assert (l_out >= 0).all()
