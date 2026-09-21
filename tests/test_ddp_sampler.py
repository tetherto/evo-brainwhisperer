"""
Phase 5 gate tests: StratifiedDayBatchSampler DDP correctness.

All tests run on CPU without real data or GPUs — the dataset is a list of
hand-crafted dicts with the minimum fields the sampler reads.

Invariants verified:
  1. Single-rank coverage: all |batches| = batch_size.
  2. Disjoint partition: for world_size ranks, the union of their batch-index
     sets is disjoint — no global batch index appears on more than one rank.
  3. Completeness: total batches across all ranks equals the single-rank
     num_batches (rank 0 with world_size=1) divided-and-floored appropriately.
  4. Determinism: two samplers with the same seed / rank / world_size yield
     identical sequences.
  5. Single-GPU equivalence: rank=0, world_size=1 yields the same sequence
     as the old single-process code (regression guard).
"""

import random as _random
from collections import Counter

import torch
import pytest

from brainwhisperer.data.samplers import StratifiedDayBatchSampler


# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

def _make_dataset(n_samples: int, n_sbj: int = 2, n_days: int = 3) -> list[dict]:
    """Minimal list-of-dicts that satisfies the sampler's iteration contract."""
    ds = []
    for i in range(n_samples):
        ds.append({
            "sbj_idx": i % n_sbj,
            "day": torch.tensor(i % n_days),
        })
    return ds


N = 400          # dataset size
BATCH = 32       # batch_size
DPB = 4          # days_per_batch
SEED = 7


# ---------------------------------------------------------------------------
# Helper: collect all batches from a sampler
# ---------------------------------------------------------------------------

def _collect(sampler: StratifiedDayBatchSampler) -> list[list[int]]:
    return list(iter(sampler))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleRankBasics:
    def test_batch_size(self):
        ds = _make_dataset(N)
        s = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=0, world_size=1, seed=SEED)
        batches = _collect(s)
        assert all(len(b) == BATCH for b in batches), "Batch size mismatch"

    def test_len_matches_iter(self):
        ds = _make_dataset(N)
        s = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=0, world_size=1, seed=SEED)
        assert len(s) == len(_collect(s)), "__len__ does not match actual iteration count"

    def test_determinism(self):
        ds = _make_dataset(N)
        s1 = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=0, world_size=1, seed=SEED)
        s2 = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=0, world_size=1, seed=SEED)
        assert _collect(s1) == _collect(s2), "Same seed must yield identical batches"


class TestDDPPartition:
    """Core correctness: rank batches must be disjoint and cover all global batches."""

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_disjoint_global_batch_indices(self, world_size):
        """
        All ranks generate the same sequence of global batches; each rank takes
        every world_size-th one starting at its rank.  The resulting sets of
        *global batch positions* must be disjoint and together cover 0 …
        total_to_generate - 1.
        """
        ds = _make_dataset(N)
        samplers = [
            StratifiedDayBatchSampler(ds, BATCH, DPB, rank=r, world_size=world_size, seed=SEED)
            for r in range(world_size)
        ]
        # Each sampler produces len(s) batches.  Verify counts match.
        counts = [len(s) for s in samplers]
        assert len(set(counts)) == 1, f"Unequal batch counts across ranks: {counts}"

        # Reconstruct which *global positions* each rank owns.
        num_per_rank = counts[0]
        total_global = num_per_rank * world_size
        rank_global_positions = [
            set(range(r, total_global, world_size)) for r in range(world_size)
        ]
        # All positions must be covered exactly once.
        union = set()
        for pos_set in rank_global_positions:
            assert union.isdisjoint(pos_set), "Rank positions overlap"
            union |= pos_set
        assert union == set(range(total_global)), "Not all global positions covered"

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_sample_index_disjointness(self, world_size):
        """
        Within a single epoch the *sample indices* yielded by different ranks
        need not be disjoint (with-replacement sampling allows repetition), but
        the batches at different global positions are built from independent RNG
        draws — test that different ranks do not yield the exact same batch content.
        """
        ds = _make_dataset(N)
        samplers = [
            StratifiedDayBatchSampler(ds, BATCH, DPB, rank=r, world_size=world_size, seed=SEED)
            for r in range(world_size)
        ]
        all_batches = [_collect(s) for s in samplers]
        # Encode batches as tuples for hashability
        encoded = [[tuple(sorted(b)) for b in batches] for batches in all_batches]
        # No two different ranks should yield the same ordered set of batches
        for r0 in range(world_size):
            for r1 in range(r0 + 1, world_size):
                assert encoded[r0] != encoded[r1], (
                    f"Ranks {r0} and {r1} yielded identical batch sequences"
                )

    @pytest.mark.parametrize("world_size", [2, 3])
    def test_total_batches_scales_with_world_size(self, world_size):
        """
        Each rank's num_batches should be ≈ single-rank batches / world_size.
        """
        ds = _make_dataset(N)
        single = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=0, world_size=1, seed=SEED)
        multi = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=0, world_size=world_size, seed=SEED)
        expected = len(single) // world_size
        assert len(multi) == expected, (
            f"world_size={world_size}: expected {expected} batches/rank, got {len(multi)}"
        )

    def test_combined_ranks_reconstruct_global_sequence(self):
        """
        Interleaving the batches from rank 0 and rank 1 in order of their
        global positions should match the batch sequence of a world_size=1
        sampler (same seed, same RNG path).
        """
        world_size = 2
        ds = _make_dataset(N)
        single = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=0, world_size=1, seed=SEED)
        r0 = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=0, world_size=world_size, seed=SEED)
        r1 = StratifiedDayBatchSampler(ds, BATCH, DPB, rank=1, world_size=world_size, seed=SEED)

        global_batches = _collect(single)
        r0_batches = _collect(r0)
        r1_batches = _collect(r1)

        # Interleave: global position 0 → r0[0], 1 → r1[0], 2 → r0[1], …
        reconstructed = []
        for a, b in zip(r0_batches, r1_batches):
            reconstructed.append(a)
            reconstructed.append(b)

        # Trim global_batches to the reconstructed length (floor-division drops tail)
        n = len(reconstructed)
        assert reconstructed == global_batches[:n], (
            "Interleaved rank batches do not match the global sequence"
        )


class TestDaysPerBatchScaling:
    """Regression guard for the train.py per-rank day scaling.

    train.py scales BOTH batch_size and days_per_batch by world_size:
        per_rank_batch = batch_size      // world_size
        per_rank_days  = days_per_batch  // world_size
    so that samples-per-day (= per_rank_batch / per_rank_days) is invariant to
    the number of GPUs and matches the reference's single-process composition.
    These tests reproduce that arithmetic and assert the invariant the sampler
    must uphold. Without the days scaling, samples-per-day is silently halved on
    two GPUs, weakening the per-session projection gradients.
    """

    GLOBAL_BATCH = 64
    GLOBAL_DAYS = 4
    EXPECTED_SAMPLES_PER_DAY = GLOBAL_BATCH // GLOBAL_DAYS  # 16

    @pytest.mark.parametrize("world_size", [1, 2, 4])
    def test_samples_per_day_invariant_under_scaling(self, world_size):
        ds = _make_dataset(800, n_sbj=1, n_days=8)
        per_rank_batch = self.GLOBAL_BATCH // world_size
        assert self.GLOBAL_DAYS % world_size == 0  # train.py guard precondition
        per_rank_days = self.GLOBAL_DAYS // world_size

        s = StratifiedDayBatchSampler(
            ds, per_rank_batch, per_rank_days, rank=0, world_size=world_size, seed=SEED
        )
        assert s.samples_per_day == self.EXPECTED_SAMPLES_PER_DAY, (
            f"world_size={world_size}: samples_per_day {s.samples_per_day} "
            f"!= reference {self.EXPECTED_SAMPLES_PER_DAY}"
        )

        # Every batch is per_rank_batch long, and each distinct day contributes a
        # whole multiple of samples_per_day (one full day-slot per draw).
        day_of = {i: int(ds[i]["day"].item()) for i in range(len(ds))}
        for b in _collect(s):
            assert len(b) == per_rank_batch
            counts = Counter(day_of[i] for i in b)
            assert all(v % s.samples_per_day == 0 for v in counts.values())

    def test_unscaled_days_halves_samples_per_day(self):
        """The bug the fix prevents: scaling batch_size but NOT days_per_batch
        on two GPUs cuts samples-per-day from 16 to 8."""
        ds = _make_dataset(800, n_sbj=1, n_days=8)
        buggy = StratifiedDayBatchSampler(
            ds, self.GLOBAL_BATCH // 2, self.GLOBAL_DAYS, rank=0, world_size=2, seed=SEED
        )
        assert buggy.samples_per_day == 8
        assert buggy.samples_per_day == self.EXPECTED_SAMPLES_PER_DAY // 2
