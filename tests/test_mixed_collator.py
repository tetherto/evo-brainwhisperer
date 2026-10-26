"""Mixed collation fill and collection metadata, without dataset builds.

1. `fill_to_num_features` on `(C, T)` samples and `MixedRecipeCollator`: the
   channel fill runs per sample, *before* time padding and stacking. Willet
   duplicates its channels (``torch.cat([x, x])`` on the channel axis), Jude
   appends zero channels.
2. `MixedRaggedDataset`'s sampler scan table: ``member``/``day`` columns
   derived from each DAY extractor's mapping, in the two shapes the real
   extractors produce: Kunz's date rank ``{'2023-11-28': 0}`` and
   Wairagkar's int-keyed block identity ``{0: 0, ... 9: 9}``.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from neuralshed.dataloader import SegmentSample

from brainwhisperer.data.collate import MixedRecipeCollator
from brainwhisperer.data.collection import MEMBER, MixedRaggedDataset
from brainwhisperer.utils.fill import fill_to_num_features

NUM_FEATURES = 8  # tiny stand-in for the model's 512-wide input


def make_sample(name: str, c: int, t: int, day: int, ids=(1, 2)):
    """A collection row as `MixedRaggedDataset.__getitem__` hands it over."""
    return SegmentSample(
        data={
            "neural": torch.arange(c * t, dtype=torch.float32).reshape(c, t),
            "whisper": torch.tensor(ids, dtype=torch.long),
            "phonemes": torch.tensor([1, 2, 3], dtype=torch.long),
            "day": torch.tensor([day], dtype=torch.long),
        },
        segment=None,
        metadata={"split": "train", MEMBER: name},
    )


# ---------------------------------------------------------------- fill rule


class TestFillSample:
    def test_duplicate_matches_willet_rule(self):
        # Expected: torch.cat([x, x], dim=1) on the (T, C) row, transposed
        # to (C, T).
        x = torch.arange(3 * 5, dtype=torch.float32).reshape(3, 5)
        out = fill_to_num_features(x, "duplicate", 6)
        expected = torch.cat([x.transpose(0, 1), x.transpose(0, 1)], dim=1).transpose(
            0, 1
        )
        assert torch.equal(out, expected)
        assert out.shape == (6, 5)

    def test_zeros_matches_jude_rule(self):
        # Expected: F.pad(x, (0, 256)) on the (T, C) row, i.e. zeros
        # appended at the END of the channel axis.
        x = torch.ones(3, 4)
        out = fill_to_num_features(x, "zeros", 5)
        assert torch.equal(out[:3], x)
        assert torch.equal(out[3:], torch.zeros(2, 4))

    def test_full_width_passes_through_all_modes(self):
        x = torch.ones(8, 2)
        for mode in ("duplicate", "zeros", "none"):
            assert fill_to_num_features(x, mode, 8) is x

    def test_wrong_widths_raise(self):
        with pytest.raises(ValueError, match="half-width"):
            fill_to_num_features(torch.ones(3, 2), "duplicate", 8)
        with pytest.raises(ValueError, match="cannot shrink"):
            fill_to_num_features(torch.ones(9, 2), "zeros", 8)
        with pytest.raises(ValueError, match="needs a CHANNEL_FILL"):
            fill_to_num_features(torch.ones(4, 2), "none", 8)
        with pytest.raises(ValueError, match="unknown channel fill mode"):
            fill_to_num_features(torch.ones(4, 2), "sideways", 8)


# ------------------------------------------------------------ mixed collator


class TestMixedRecipeCollator:
    def make_collator(self):
        # The CHANNEL_FILL values these datasets' recipes declare.
        return MixedRecipeCollator(
            {"card": "none", "willet": "duplicate", "jude": "zeros"},
            num_features=NUM_FEATURES,
        )

    def test_mixed_batch_layout(self):
        collator = self.make_collator()
        batch = collator(
            [
                make_sample("willet", 4, 5, day=3, ids=(10, 11)),
                make_sample("jude", 6, 3, day=0, ids=(12,)),
                make_sample("card", 8, 4, day=7, ids=(13, 14, 15)),
            ]
        )
        assert batch.signals.shape == (3, NUM_FEATURES, 5)  # (B, C, T_max)
        # Per-row routing names, the collection's member column:
        assert batch.datasets == ["willet", "jude", "card"]
        # day is flattened to (B,), not (B, 1):
        assert batch.day_idx.tolist() == [3, 0, 7]
        # Valid length per row from the ragged collator:
        assert batch.lengths.tolist() == [5, 3, 4]
        # Willet row: duplicated halves (duplicate rule) even in the padded tail.
        wil = batch.signals[0]
        assert torch.equal(wil[:4], wil[4:])
        # Jude row: zeros fill at the END of the channel axis.
        ju = batch.signals[1]
        assert torch.equal(ju[6:], torch.zeros(2, 5))
        assert torch.equal(ju[:6, 3:], torch.zeros(6, 2))  # padded time tail
        # Card row: untouched width-8 passthrough.
        assert not torch.equal(batch.signals[2][:4], batch.signals[2][4:])

    def test_labels_keep_the_minus100_contract(self):
        collator = self.make_collator()
        batch = collator(
            [
                make_sample("card", 8, 2, 0, ids=(5, 6)),
                make_sample("card", 8, 2, 0, ids=(7,)),
            ]
        )
        assert batch.labels.shape[1] == 2
        assert batch.labels[1, 1].item() == -100  # CE ignore index

    def test_unknown_member_or_fill_mode_raise(self):
        with pytest.raises(KeyError, match="no CHANNEL_FILL rule"):
            MixedRecipeCollator({"card": "none"}, num_features=NUM_FEATURES)(
                [make_sample("willet", 4, 2, 0)]
            )
        with pytest.raises(ValueError):
            MixedRecipeCollator({"card": "sideways"}, num_features=NUM_FEATURES)(
                [make_sample("card", 8, 2, 0)]
            )


# ------------------------------------------------------- collection metadata


class _FakeExtractor:
    def __init__(self, mapping, field, use_private=False):
        self.event_field = field
        if use_private:
            # BlockDayEncoder shape: pydantic refuses int keys, so the
            # mapping only exists as the prepare-time _label_to_ind.
            self.predefined_mapping = None
            self._label_to_ind = mapping
        else:
            self.predefined_mapping = mapping


class _FakeMemberDataset:
    """What MixedRaggedDataset reads of a member: extractors/triggers/metadata."""

    def __init__(self, extractor, triggers, split="train"):
        self.extractors = {"day": extractor}
        self.triggers = triggers
        self.metadata = pd.DataFrame({"split": [split] * len(triggers)})

    def __len__(self):
        return len(self.triggers)

    def __getitem__(self, i):
        return SegmentSample(
            data={}, segment=None, metadata=self.metadata.iloc[i].to_dict()
        )


class TestCollectionScanTable:
    def test_session_rank_and_boundaries(self):
        kunz = _FakeMemberDataset(
            # kunz_t12's real mapping (a single date).
            _FakeExtractor({"2023-11-28": 0}, "date"),
            pd.DataFrame({"date": ["2023-11-28"] * 3}),
        )
        wair = _FakeMemberDataset(
            _FakeExtractor({i: i for i in range(10)}, "block", use_private=True),
            # numpy ints, like the live `block` column (dtype int64):
            pd.DataFrame({"block": np.array([0, 0, 3, 9], dtype="int64")}),
        )
        collection = MixedRaggedDataset({"kunz_t12": kunz, "wairagkar": wair})

        assert len(collection) == 7
        assert collection.metadata[MEMBER].tolist() == (
            ["kunz_t12"] * 3 + ["wairagkar"] * 4
        )
        assert collection.metadata["day"].tolist() == [0, 0, 0, 0, 0, 3, 9]
        assert collection.locate(0) == ("kunz_t12", 0)
        assert collection.locate(3) == ("wairagkar", 0)
        assert collection.locate(6) == ("wairagkar", 3)
        assert collection[3].metadata[MEMBER] == "wairagkar"
        assert collection[0].metadata == {"split": "train", MEMBER: "kunz_t12"}

    def test_date_objects_resolve_through_isoformat(self):
        import datetime

        member = _FakeMemberDataset(
            _FakeExtractor({"2022-04-28": 0, "2022-05-05": 1}, "date"),
            pd.DataFrame(
                {"date": [datetime.date(2022, 5, 5), datetime.date(2022, 4, 28)]}
            ),
        )
        collection = MixedRaggedDataset({"kunz_t12": member})
        assert collection.metadata["day"].tolist() == [1, 0]

    def test_unmapped_row_raises(self):
        member = _FakeMemberDataset(
            _FakeExtractor({"2023-11-28": 0}, "date"),
            pd.DataFrame({"date": ["2023-99-99"]}),
        )
        with pytest.raises(KeyError, match="not in its day-index mapping"):
            MixedRaggedDataset({"kunz_t12": member})
