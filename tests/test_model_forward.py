"""
Phase 2 gate check — model layer.

Run: pytest tests/test_model_forward.py -v

Synthetic tests cover shape/dtype contracts without requiring data on disk.
TestBrainWhispererRealData builds its batch from the Card recipe, so it
requires the study data mount (neuralshed `dataset_dir("card_2024")`).
TestBrainWhisperer downloads openai/whisper-tiny.en (~150 MB) on first run.
"""

import dataclasses
import json
import shutil

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from brainwhisperer.config.schema import (
    BWEmbedderConfig,
    BWEncoderConfig,
    BWLoRAConfig,
    BWModelConfig,
)
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.data.collate import (
    SingleRecipeCollator,
    default_ragged_collator,
)
from brainwhisperer.data.recipes import RECIPES
from brainwhisperer.models.embedder import BWEmbedder
from brainwhisperer.models.encoder import BWEncoder
from brainwhisperer.models.build import model_card_kwargs
from brainwhisperer.models.model import HUB_FORMAT_VERSION, BrainWhisperer
from brainwhisperer.models.positional_encoding import PositionalEncoding
from brainwhisperer.utils.fill import FILL_NONE

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_FEATURES = 512
D_MODEL = 384
N_LAYERS = 6
N_HEADS = 6
B = 2
T_SIGNAL = 3000
STRIDE_2 = 2
UNFOLD_WIDTH = 4 // STRIDE_2  # 2
T_CTC = 1500 // UNFOLD_WIDTH  # 750
N_DAYS = 10


def _embedder_cfg(embed_type: str = "conv") -> BWEmbedderConfig:
    return BWEmbedderConfig(
        num_features=N_FEATURES,
        d_model=D_MODEL,
        embed_type=embed_type,
        kernel_size_1=7,
        stride_2=STRIDE_2,
        patch_size=5,
        encodings="sin",
        day_projections=True,
        day_proj_rank=4,  # small rank for fast tests
        dropout=0.0,  # deterministic forward
        day_layer_dropout=0.0,
        session_dropout=0.0,
    )


def _encoder_cfg() -> BWEncoderConfig:
    return BWEncoderConfig(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_ff=1536,
        dropout=0.0,
        last_phoneme_layer=4,
        attn_window_size=29,
        rope=False,
        max_source_pos=1500,
        temporal_pe_std=0.02,
    )


def _model_cfg() -> BWModelConfig:
    return BWModelConfig(
        whisper_name="openai/whisper-tiny.en",
        n_embedders=1,
        embedder=_embedder_cfg(),
        encoder=_encoder_cfg(),
        lora=BWLoRAConfig(
            r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]
        ),
    )


def _batch(n_embedders: int = 1) -> NeuralBatch:
    return NeuralBatch(
        signals=torch.randn(B, N_FEATURES, T_SIGNAL),
        labels=torch.randint(0, 200, (B, 10)),
        datasets=["e0"] * B,  # all samples route to embedder "e0"
        day_idx=torch.randint(0, N_DAYS, (B,)),
        lengths=torch.full((B,), T_SIGNAL, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=D_MODEL, max_len=1000)
        idx = torch.randint(0, 100, (B,))
        out = pe(idx)
        assert out.shape == (B, D_MODEL // 2)

    def test_sequence_lookup(self):
        pe = PositionalEncoding(d_model=D_MODEL, max_len=2000)
        idx = torch.arange(1500)
        out = pe(idx)
        assert out.shape == (1500, D_MODEL // 2)

    def test_no_parameters(self):
        pe = PositionalEncoding(d_model=D_MODEL)
        assert len(list(pe.parameters())) == 0

    def test_buffer_registered(self):
        pe = PositionalEncoding(d_model=D_MODEL)
        assert "pe" in dict(pe.named_buffers())

    def test_deterministic(self):
        pe = PositionalEncoding(d_model=D_MODEL)
        idx = torch.tensor([0, 1, 2])
        assert torch.equal(pe(idx), pe(idx))


# ---------------------------------------------------------------------------
# BWEmbedder
# ---------------------------------------------------------------------------


class TestBWEmbedder:
    def test_conv_output_shape(self):
        emb = BWEmbedder(_embedder_cfg("conv"), num_days=N_DAYS)
        out = emb(
            torch.randn(B, N_FEATURES, T_SIGNAL), torch.zeros(B, dtype=torch.long)
        )
        assert out.shape == (B, 1500, D_MODEL)

    def test_conv_short_signal_padded(self):
        emb = BWEmbedder(_embedder_cfg("conv"), num_days=N_DAYS)
        out = emb(torch.randn(B, N_FEATURES, 400), torch.zeros(B, dtype=torch.long))
        assert out.shape == (B, 1500, D_MODEL)

    def test_conv_long_signal_trimmed(self):
        emb = BWEmbedder(_embedder_cfg("conv"), num_days=N_DAYS)
        out = emb(torch.randn(B, N_FEATURES, 8000), torch.zeros(B, dtype=torch.long))
        assert out.shape == (B, 1500, D_MODEL)

    def test_patch_output_shape(self):
        cfg = _embedder_cfg("patch")
        emb = BWEmbedder(cfg, num_days=N_DAYS)
        signals = torch.randn(B, N_FEATURES, 3000)
        out = emb(signals, torch.zeros(B, dtype=torch.long))
        expected_patches = 3000 // cfg.patch_size
        assert out.shape == (B, expected_patches, D_MODEL)

    def test_different_day_indices(self):
        emb = BWEmbedder(_embedder_cfg("conv"), num_days=N_DAYS)
        day_idx = torch.arange(B) % N_DAYS
        out = emb(torch.randn(B, N_FEATURES, T_SIGNAL), day_idx)
        assert out.shape == (B, 1500, D_MODEL)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# BWEncoder
# ---------------------------------------------------------------------------


class TestBWEncoder:
    def test_output_last_hidden_state_shape(self):
        enc = BWEncoder(_encoder_cfg())
        out = enc(torch.randn(B, 1500, D_MODEL))
        assert out.last_hidden_state.shape == (B, 1500, D_MODEL)

    def test_hidden_states_count(self):
        enc = BWEncoder(_encoder_cfg())
        out = enc(torch.randn(B, 1500, D_MODEL))
        # hidden_states[0] = input, hidden_states[k] = output of layer k-1
        assert out.hidden_states is not None
        assert len(out.hidden_states) == N_LAYERS + 1

    def test_intermediate_layer_accessible(self):
        enc = BWEncoder(_encoder_cfg())
        out = enc(torch.randn(B, 1500, D_MODEL))
        last_phoneme_layer = _encoder_cfg().last_phoneme_layer
        intermediate = out.hidden_states[last_phoneme_layer + 1]
        assert intermediate.shape == (B, 1500, D_MODEL)

    def test_last_hidden_state_equals_final_layer(self):
        enc = BWEncoder(_encoder_cfg())
        out = enc(torch.randn(B, 1500, D_MODEL))
        assert torch.equal(out.last_hidden_state, out.hidden_states[-1])

    def test_no_nan(self):
        enc = BWEncoder(_encoder_cfg())
        out = enc(torch.randn(B, 1500, D_MODEL))
        assert not torch.isnan(out.last_hidden_state).any()


# ---------------------------------------------------------------------------
# BrainWhisperer  (downloads whisper-tiny.en on first run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    m = BrainWhisperer(_model_cfg(), ["e0"], num_days_per_embedder=[N_DAYS])
    m.eval()
    return m


class TestBrainWhisperer:
    def test_forward_keys(self, model):
        with torch.no_grad():
            out = model(_batch())
        assert "ce_loss" in out
        assert "phone_logits" in out
        assert "x_len" in out

    def test_ce_loss_scalar(self, model):
        with torch.no_grad():
            out = model(_batch())
        assert out["ce_loss"].shape == ()

    def test_phone_logits_shape(self, model):
        with torch.no_grad():
            out = model(_batch())
        assert out["phone_logits"].shape == (B, T_CTC, 41)

    def test_no_nan(self, model):
        with torch.no_grad():
            out = model(_batch())
        assert not torch.isnan(out["ce_loss"])
        assert not torch.isnan(out["phone_logits"]).any()

    def test_generate_shape(self, model):
        with torch.no_grad():
            tokens = model.generate(_batch(), max_new_tokens=10)
        assert tokens.ndim == 2
        assert tokens.shape[0] == B

    def test_generate_grouped_beams(self, model):
        # The paper's decoding: transformers 5 runs it only through our copy.
        with torch.no_grad():
            tokens = model.generate(
                _batch(),
                num_beams=4,
                num_beam_groups=2,
                diversity_penalty=0.25,
                max_new_tokens=10,
            )
        assert tokens.ndim == 2
        assert tokens.shape[0] == B

    def test_encoder_only_skips_decoder(self, model):
        # encoder_only=True must return CTC outputs but no ce_loss.
        with torch.no_grad():
            out = model(_batch(), encoder_only=True)
        assert "ce_loss" not in out, "encoder_only must not return ce_loss"
        assert "phone_logits" in out
        assert "x_len" in out
        assert out["phone_logits"].shape == (B, T_CTC, 41)
        assert out["x_len"].shape == (B,)


# ---------------------------------------------------------------------------
# BrainWhisperer — save_pretrained / from_pretrained round trip
# ---------------------------------------------------------------------------


# Shaped like brainwhisperer.models.build.inference_settings.
_HUB_INFERENCE = {
    "generation": {"num_beams": 4, "num_beam_groups": 2},
    "smoothing": {"smooth_kernel_size": 100, "smooth_sigma": 2.0},
    "max_label_len": 448,
    "measure_per": True,
}


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    """A two-dataset model, its save folder, and the model loaded back from it."""
    cfg = dataclasses.replace(
        _model_cfg(),
        n_embedders=2,
        whisper_dropout=0.1,  # so train mode has dropout to reproduce
        encoder=dataclasses.replace(_encoder_cfg(), dropout=0.1),
    )
    # As eval.py passes them: DictConfigs, one with a per-dataset override,
    # and day counts that may be numpy ints.
    embedder_cfgs = [
        OmegaConf.structured(cfg.embedder),
        OmegaConf.merge(cfg.embedder, {"day_proj_rank": 2}),
    ]
    m = BrainWhisperer(
        OmegaConf.structured(cfg),
        ["e0", "e1"],
        [np.int64(N_DAYS), 3],
        [list(range(N_DAYS)), [0, 40, 75]],
        embedder_cfgs,
        channel_fills=[FILL_NONE, FILL_NONE],
        inference=_HUB_INFERENCE,
    )
    folder = tmp_path_factory.mktemp("saved_model")
    m.save_pretrained(folder)
    return m, folder, BrainWhisperer.from_pretrained(folder)


def _two_dataset_batch() -> NeuralBatch:
    """One sample per embedder, so both the e0 and the shared e1 path run."""
    batch = _batch()
    batch.datasets = ["e0", "e1"]
    batch.day_idx = torch.tensor([N_DAYS - 1, 2])
    return batch


class TestHubRoundTrip:
    """save_pretrained -> from_pretrained gives back the same model, built from
    config.json alone (no dataset is read)."""

    def test_config(self, saved_model):
        _, folder, loaded = saved_model
        saved = json.loads((folder / "config.json").read_text())
        assert saved["format_version"] == HUB_FORMAT_VERSION
        assert saved["num_days_per_embedder"] == [N_DAYS, 3]
        assert saved["embedder_cfg_per_embedder"][1]["day_proj_rank"] == 2
        assert saved["pretrained_whisper"] is False
        assert loaded._hub_mixin_config == saved
        assert loaded.inference == _HUB_INFERENCE

    @pytest.mark.parametrize("train", [False, True])
    def test_same_forward(self, saved_model, train):
        m, _, loaded = saved_model
        batch = _two_dataset_batch()
        outs = []
        for model in (m, loaded):
            model.train(train)
            torch.manual_seed(0)  # same dropout masks in train mode
            with torch.no_grad():
                outs.append(model(batch))
        assert torch.equal(outs[0]["ce_loss"], outs[1]["ce_loss"])
        assert torch.equal(outs[0]["phone_logits"], outs[1]["phone_logits"])

    def test_same_generate(self, saved_model):
        m, _, loaded = saved_model
        batch = _two_dataset_batch()
        ids = []
        for model in (m, loaded):
            model.eval()
            with torch.no_grad():
                ids.append(model.generate(batch, num_beams=2, max_new_tokens=8))
        assert torch.equal(ids[0], ids[1])

    def test_shared_weights_relinked(self, saved_model):
        _, _, loaded = saved_model
        assert loaded.embedders["e1"].conv1 is loaded.embedders["e0"].conv1
        assert loaded.embedders["e1"].conv2 is loaded.embedders["e0"].conv2
        assert (
            loaded.whisper.proj_out.weight.data_ptr()
            == loaded.whisper.get_decoder().get_input_embeddings().weight.data_ptr()
        )

    def test_whisper_encoder_not_saved(self, saved_model):
        _, folder, _ = saved_model
        with safe_open(folder / "model.safetensors", "pt") as f:
            assert not [k for k in f.keys() if k.startswith("whisper.model.encoder.")]

    def test_missing_weight_fails(self, saved_model, tmp_path):
        _, folder, _ = saved_model
        shutil.copy(folder / "config.json", tmp_path)
        weights = load_file(folder / "model.safetensors")
        del weights["phone_head.weight"]
        save_file(weights, tmp_path / "model.safetensors")
        with pytest.raises(RuntimeError, match="phone_head.weight"):
            BrainWhisperer.from_pretrained(tmp_path)

    def test_rejects_other_format_version(self, saved_model, tmp_path):
        _, folder, _ = saved_model
        config = json.loads((folder / "config.json").read_text())
        (tmp_path / "config.json").write_text(
            json.dumps({**config, "format_version": 99})
        )
        with pytest.raises(ValueError, match="config format 99"):
            BrainWhisperer.from_pretrained(tmp_path)

    def test_model_card(self, tmp_path):
        """save_pretrained writes a README.md card with the Hub metadata and
        the template fully filled in."""
        m = BrainWhisperer(_model_cfg(), ["card"], [N_DAYS], inference=_HUB_INFERENCE)
        m.save_pretrained(tmp_path, model_card_kwargs=model_card_kwargs(m))
        card = (tmp_path / "README.md").read_text()
        assert "library_name: brainwhisperer" in card
        assert "{{" not in card and "{%" not in card


# ---------------------------------------------------------------------------
# BrainWhisperer — real CARD data  (requires .envrc env vars)
# ---------------------------------------------------------------------------

CARD_N_FEATURES = 512
CARD_N_DAYS = 45
CARD_BATCH_SIZE = 4
CARD_MAX_LABEL_LEN = 448  # seed 6's data.max_seq_len


@pytest.fixture(scope="module")
def card_batch():
    """First CARD_BATCH_SIZE val trials as a NeuralBatch, via the Card recipe."""
    ds = RECIPES["card"].build_val_dataset(max_label_len=CARD_MAX_LABEL_LEN)
    loader = DataLoader(
        ds,
        batch_size=CARD_BATCH_SIZE,
        collate_fn=SingleRecipeCollator(
            "card", RECIPES["card"], default_ragged_collator()
        ),
        shuffle=False,
    )
    return next(iter(loader))


@pytest.fixture(scope="module")
def card_model():
    cfg = BWModelConfig(
        whisper_name="openai/whisper-tiny.en",
        n_embedders=1,
        embedder=BWEmbedderConfig(
            num_features=CARD_N_FEATURES,
            d_model=D_MODEL,
            embed_type="conv",
            kernel_size_1=7,
            stride_2=STRIDE_2,
            patch_size=5,
            encodings="sin",
            day_projections=True,
            day_proj_rank=4,
            dropout=0.0,
            day_layer_dropout=0.0,
            session_dropout=0.0,
        ),
        encoder=_encoder_cfg(),
        lora=BWLoRAConfig(
            r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]
        ),
    )
    m = BrainWhisperer(cfg, ["card"], num_days_per_embedder=[CARD_N_DAYS])
    m.eval()
    return m


class TestBrainWhispererRealData:
    def test_forward_no_nan(self, card_model, card_batch):
        with torch.no_grad():
            out = card_model(card_batch)
        assert not torch.isnan(out["ce_loss"]), "ce_loss is NaN on real data"
        assert not torch.isnan(out["phone_logits"]).any(), "phone_logits has NaN"

    def test_forward_output_shapes(self, card_model, card_batch):
        B = CARD_BATCH_SIZE
        unfold_width = 4 // STRIDE_2
        T_ctc = 1500 // unfold_width
        with torch.no_grad():
            out = card_model(card_batch)
        assert out["ce_loss"].shape == ()
        assert out["phone_logits"].shape == (B, T_ctc, 41)

    def test_x_len_valid(self, card_model, card_batch):
        """x_len must be in (0, T_ctc] for every sample and match batch.lengths."""
        T_ctc = 1500 // (4 // STRIDE_2)
        with torch.no_grad():
            out = card_model(card_batch)
        x_len = out["x_len"]
        assert x_len.shape == (CARD_BATCH_SIZE,)
        assert (x_len > 0).all(), "x_len must be positive for all samples"
        assert (x_len <= T_ctc).all(), f"x_len must be <= T_ctc={T_ctc}"
        # longer raw signals must produce longer or equal x_len
        assert (x_len <= T_ctc).all()
        lengths = card_batch.lengths
        # monotone: sort by raw length, x_len must also be non-decreasing
        sorted_idx = lengths.argsort()
        assert (x_len[sorted_idx].diff() >= 0).all(), (
            "x_len must be monotone w.r.t. raw lengths"
        )

    def test_generate_returns_tokens(self, card_model, card_batch):
        with torch.no_grad():
            tokens = card_model.generate(card_batch, max_new_tokens=10)
        assert tokens.ndim == 2
        assert tokens.shape[0] == CARD_BATCH_SIZE
