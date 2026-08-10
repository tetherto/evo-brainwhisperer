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
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        days_per_batch: int,
        rank: int = 0,
        world_size: int = 1,  # number of devices for DDP
        seed: int = 1,
    ):
        assert batch_size % days_per_batch == 0, (
            f"batch_size ({batch_size}) must be divisible by days_per_batch ({days_per_batch})"
        )

        self.rank = rank
        self.world_size = world_size
        self.days_per_batch = days_per_batch
        self.samples_per_day = batch_size // days_per_batch
        # Ensuring same seed on every rank for multi-device training
        self.trng = torch.Generator().manual_seed(seed)
        self.rng = random.Random(seed)

        self.num_samples_per_sbj: dict[int, int] = {}
        self.day_to_indices: dict[int, dict[int, list[int]]] = {}

        for idx, sample in enumerate(dataset):
            sbj: int = int(sample["sbj_idx"])
            self.num_samples_per_sbj[sbj] = self.num_samples_per_sbj.get(sbj, 0) + 1
            self.day_to_indices.setdefault(sbj, {})
            day = int(sample["day"].item())
            self.day_to_indices[sbj].setdefault(day, []).append(idx)

        total = len(dataset)
        self.sbj_ids = list(self.num_samples_per_sbj.keys())
        self.sbj_probs = {sbj: n / total for sbj, n in self.num_samples_per_sbj.items()}
        self.days = {
            sbj: list(days.keys()) for sbj, days in self.day_to_indices.items()
        }
        # Total batches this rank yields. Splitting batch size across devices
        total_batches = total // batch_size
        self.num_batches = total_batches // world_size
        self._total_to_generate = self.num_batches * world_size

    def __iter__(self):
        probs = torch.tensor([self.sbj_probs[s] for s in self.sbj_ids])
        for global_idx in range(self._total_to_generate):
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
            # Only yield batches assigned to this rank.
            if global_idx % self.world_size == self.rank:
                yield batch

    def __len__(self) -> int:
        return self.num_batches
