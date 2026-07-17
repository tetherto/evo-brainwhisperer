import os
import random

import torch
from torch.utils.data import Sampler


class StratifiedDayBatchSampler(Sampler):
    """
    Each batch contains exactly days_per_batch recording days, sampled
    proportional to each subject group's total size. Within each selected day,
    samples_per_day samples are drawn with replacement.

    Grouping is by sbj_idx (the model's routing key) rather than source_dataset
    string, so subject groups that share the same source string (e.g. NEJM and
    Kunz both report "card") are kept strictly separate.

    Multi-GPU: LOCAL_RANK offsets the seed so each rank produces a disjoint
    sequence of batches.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        days_per_batch: int,
        num_processes: int = 1,
        seed: int = 1,
    ):
        assert batch_size % days_per_batch == 0, (
            f"batch_size ({batch_size}) must be divisible by days_per_batch ({days_per_batch})"
        )

        rank_seed = seed + int(os.environ.get("LOCAL_RANK", 0))
        self.days_per_batch = days_per_batch
        self.samples_per_day = batch_size // days_per_batch
        self.trng = torch.Generator().manual_seed(rank_seed)
        self.rng = random.Random(rank_seed)

        # sbj_idx (int) → total sample count
        self.num_samples_per_sbj: dict[int, int] = {}
        # sbj_idx → day → [global dataset indices]
        self.day_to_indices: dict[int, dict[int, list[int]]] = {}

        for idx, sample in enumerate(dataset):
            sbj: int = int(sample["sbj_idx"])
            self.num_samples_per_sbj[sbj] = self.num_samples_per_sbj.get(sbj, 0) + 1
            self.day_to_indices.setdefault(sbj, {})
            day = int(sample["day"].item())
            self.day_to_indices[sbj].setdefault(day, []).append(idx)

        total = len(dataset)
        self.sbj_ids = list(self.num_samples_per_sbj.keys())
        self.sbj_probs = {
            sbj: n / total for sbj, n in self.num_samples_per_sbj.items()
        }
        self.days = {
            sbj: list(days.keys()) for sbj, days in self.day_to_indices.items()
        }
        self.num_batches = (total // batch_size) // num_processes

    def __iter__(self):
        probs = torch.tensor([self.sbj_probs[s] for s in self.sbj_ids])
        for _ in range(self.num_batches):
            batch: list[int] = []
            sbj_sample_ids = torch.multinomial(
                probs,
                num_samples=self.days_per_batch,
                replacement=True,
                generator=self.trng,
            )
            days_per_sbj = torch.bincount(sbj_sample_ids, minlength=len(self.sbj_ids))
            for sbj, n_days in zip(self.sbj_ids, days_per_sbj):
                chosen_days = self.rng.choices(self.days[sbj], k=int(n_days.item()))
                for day in chosen_days:
                    indices = self.rng.choices(
                        self.day_to_indices[sbj][day], k=self.samples_per_day
                    )
                    batch.extend(indices)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
