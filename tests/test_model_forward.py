"""
Phase 2 gate check — model layer.

Run: pytest tests/test_model_forward.py -v

All tests run on CPU with random synthetic data.
TestBrainWhisperer downloads openai/whisper-tiny.en (~150 MB) on first run.
"""

import pytest
import torch

from brainwhisperer.config.schema import (
    BWEmbedderConfig,
    BWEncoderConfig,
    BWLoRAConfig,
    BWModelConfig,
)
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.models.positional_encoding import PositionalEncoding
from brainwhisperer.models.embedder import BWEmbedder
from brainwhisperer.models.encoder import BWEncoder
from brainwhisperer.models.model import BrainWhisperer

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
UNFOLD_WIDTH = 4 // STRIDE_2   # 2
T_CTC = 1500 // UNFOLD_WIDTH   # 750
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
        day_proj_rank=4,   # small rank for fast tests
        dropout=0.0,       # deterministic forward
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
    )


def _model_cfg() -> BWModelConfig:
    return BWModelConfig(
        whisper_name="openai/whisper-tiny.en",
        n_embedders=1,
        embedder=_embedder_cfg(),
        encoder=_encoder_cfg(),
        lora=BWLoRAConfig(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]),
    )


def _batch(n_embedders: int = 1) -> NeuralBatch:
    return NeuralBatch(
        signals=torch.randn(B, N_FEATURES, T_SIGNAL),
        labels=torch.randint(0, 200, (B, 10)),
        sbj_idx=torch.zeros(B, dtype=torch.long),   # all from embedder 0
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
        out = emb(torch.randn(B, N_FEATURES, T_SIGNAL), torch.zeros(B, dtype=torch.long))
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
    m = BrainWhisperer(_model_cfg(), num_days_per_embedder=[N_DAYS])
    m.eval()
    return m


class TestBrainWhisperer:
    def test_forward_keys(self, model):
        with torch.no_grad():
            out = model(_batch())
        assert "ce_loss" in out
        assert "phone_logits" in out
        assert "char_logits" in out

    def test_ce_loss_scalar(self, model):
        with torch.no_grad():
            out = model(_batch())
        assert out["ce_loss"].shape == ()

    def test_phone_logits_shape(self, model):
        with torch.no_grad():
            out = model(_batch())
        assert out["phone_logits"].shape == (B, T_CTC, 41)

    def test_char_logits_shape(self, model):
        with torch.no_grad():
            out = model(_batch())
        # char head vocab == whisper-tiny vocab size
        assert out["char_logits"].shape[0] == B
        assert out["char_logits"].shape[1] == T_CTC

    def test_no_nan(self, model):
        with torch.no_grad():
            out = model(_batch())
        assert not torch.isnan(out["ce_loss"])
        assert not torch.isnan(out["phone_logits"]).any()
        assert not torch.isnan(out["char_logits"]).any()

    def test_generate_shape(self, model):
        with torch.no_grad():
            tokens = model.generate(_batch(), max_new_tokens=10)
        assert tokens.ndim == 2
        assert tokens.shape[0] == B
