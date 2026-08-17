"""
Phase 3 gate test: one two-phase training step via BWLightningModule, real CARD data, CPU only.

Asserts:
  - BWLightningModule can be imported and instantiated (no import-chain incompatibility).
  - NeuralBatch from real CARD data carries phone_seq and phone_seq_len.
  - training_step returns a finite, non-negative scalar loss.
  - Gradients reach embedders, encoder, and phone_head.
  - No NaN in any gradient.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import WhisperTokenizer

from brainwhisperer.config.schema import (
    BWAugmentationConfig,
    BWEmbedderConfig,
    BWEncoderConfig,
    BWLoRAConfig,
    BWModelConfig,
    BWTrainingConfig,
)
from brainwhisperer.data.collate import NeuralCollator
from brainwhisperer.data.datasets.card import CardDataset
from brainwhisperer.models.model import BrainWhisperer
from brainwhisperer.training.lightning_module import BWLightningModule

# ---------------------------------------------------------------------------
# Fixtures (module-scoped for speed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return WhisperTokenizer.from_pretrained(
        "openai/whisper-small", language="en", task="transcribe"
    )


@pytest.fixture(scope="module")
def card_batch(tokenizer):
    dataset = CardDataset(split="train")
    collator = NeuralCollator(tokenizer, max_label_len=448)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator, shuffle=False)
    return next(iter(loader))


@pytest.fixture(scope="module")
def lightning_module(tokenizer):
    dataset = CardDataset(split="train")
    num_days = int(dataset._base.days.max().item()) + 1

    emb_cfg = BWEmbedderConfig(
        num_features=512,
        d_model=384,
        embed_type="conv",
        kernel_size_1=3,
        stride_2=2,
        patch_size=16,
        encodings="sin",
        day_projections=False,
        day_proj_rank=0,
        dropout=0.0,
        day_layer_dropout=0.0,
        session_dropout=0.0,
    )
    enc_cfg = BWEncoderConfig(
        n_layers=6,
        n_heads=6,
        d_model=384,
        d_ff=1536,
        dropout=0.0,
        last_phoneme_layer=4,
        attn_window_size=29,
        rope=False,
        max_source_pos=1500,
        temporal_pe_std=0.02,
    )
    lora_cfg = BWLoRAConfig(
        r=8, alpha=16, dropout=0.0, target_modules=["q_proj", "v_proj"]
    )
    model_cfg = BWModelConfig(
        whisper_name="openai/whisper-tiny.en",
        n_embedders=1,
        embedder=emb_cfg,
        encoder=enc_cfg,
        lora=lora_cfg,
    )
    training_cfg = BWTrainingConfig(
        max_epochs=10,
        lr=1e-3,
        lr_min=1e-5,
        weight_decay=1e-4,
        gradient_clip_val=1.0,
        ce_coeff=1.0,
        ctc_phone_coeff=1.0,
        ctc_char_coeff=0.0,
        scheduler="cosine",
        seed=42,
        augmentation=BWAugmentationConfig(
            gain_std=0.0,
            noise_std=1.0,
            offset_std=0.2,
            walk_std=0.0,
            max_cut=3,
            smooth_kernel_size=100,
            smooth_sigma=2.0,
            num_masks=10,
            max_mask_pct=0.075,
        ),
    )
    model = BrainWhisperer(model_cfg, num_days_per_embedder=[num_days])
    lm = BWLightningModule(
        model, training_cfg, tokenizer, val_dataloader_names=["card"]
    )
    lm.log = lambda *args, **kwargs: None
    return lm


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


class TestPhoneSeqInBatch:
    def test_phone_seq_present(self, card_batch):
        assert card_batch.phone_seq is not None, "phone_seq missing from CARD batch"
        assert card_batch.phone_seq_len is not None, "phone_seq_len missing"

    def test_phone_seq_nonzero_lengths(self, card_batch):
        assert (card_batch.phone_seq_len > 0).all(), "zero-length phone sequences"

    def test_phone_seq_shape(self, card_batch):
        B = card_batch.signals.shape[0]
        assert card_batch.phone_seq.shape[0] == B
        assert card_batch.phone_seq_len.shape == (B,)


class TestTrainingStep:
    def test_loss_is_scalar(self, lightning_module, card_batch):
        loss = lightning_module.training_step(card_batch, 0)
        assert loss.shape == ()

    def test_loss_finite_nonneg(self, lightning_module, card_batch):
        loss = lightning_module.training_step(card_batch, 0)
        assert torch.isfinite(loss), f"loss not finite: {loss}"
        assert loss >= 0.0, f"loss negative: {loss}"

    def test_gradients_flow(self, lightning_module, card_batch):
        lightning_module.model.zero_grad()
        loss = lightning_module.training_step(card_batch, 0)
        loss.backward()

        for tag, module in [
            ("embedders", lightning_module.model.embedders),
            ("encoder", lightning_module.model.encoder),
            ("phone_head", lightning_module.model.phone_head),
        ]:
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in module.parameters()
                if p.requires_grad
            )
            assert has_grad, f"no non-zero gradient in {tag}"

    def test_no_nan_in_gradients(self, lightning_module, card_batch):
        lightning_module.model.zero_grad()
        loss = lightning_module.training_step(card_batch, 0)
        loss.backward()

        for name, param in lightning_module.model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
