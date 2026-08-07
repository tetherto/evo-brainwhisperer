import torch
import torch.distributed.tensor
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from transformers import WhisperConfig, WhisperForConditionalGeneration

from brainwhisperer.config.schema import BWModelConfig
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.models.embedder import _MAX_SOURCE_POSITIONS, BWEmbedder
from brainwhisperer.models.encoder import BWEncoder

_N_PHONE_CLASSES = 41  # CMU phoneme set used in CARD/Willet


class BrainWhisperer(nn.Module):
    """
    Top-level model, receives neural signals and outputs transcribed text.
    BrainWhisperer (BW) follows a modular design, composed of:
        - one BWEmbedder per subject group, where embedding conv. weights are shared across all
        embedders.
        - a BWEncoder, introducing custom elements w.r.t the vanilla Whisper encoder
        - a WhisperForConditionalGeneration, used unmodified except for LoRA on the decoder.

    Usage: BrainWhisperer(cfg, num_days_per_embedder=[45, 24, ...])
    forward() returns:
        ce_loss:      scalar, cross-entropy from Whisper decoder
        phone_logits: (B, T_ctc, 41)
        char_logits:  (B, T_ctc, vocab_size)

    generate() returns token ids (B, L).
    """

    def __init__(self, cfg: BWModelConfig, num_days_per_embedder: list[int]):
        super().__init__()

        assert len(num_days_per_embedder) == cfg.n_embedders, (
            f"num_days_per_embedder has {len(num_days_per_embedder)} entries "
            f"but cfg.n_embedders={cfg.n_embedders}"
        )

        # Embedders
        self.embedders = nn.ModuleList(
            [BWEmbedder(cfg.embedder, num_days) for num_days in num_days_per_embedder]
        )
        # Share conv weights across subjects
        if cfg.n_embedders > 1:
            for emb in self.embedders[1:]:
                emb.conv1 = self.embedders[0].conv1
                emb.conv2 = self.embedders[0].conv2

        # Encoder
        self.encoder = BWEncoder(cfg.encoder)
        self._last_phoneme_layer = cfg.encoder.last_phoneme_layer

        # Original Whisper decoder + LoRA
        whisper_cfg = WhisperConfig.from_pretrained(cfg.whisper_name)
        if cfg.whisper_dropout > 0.0:
            whisper_cfg.dropout = cfg.whisper_dropout
            whisper_cfg.activation_dropout = cfg.whisper_dropout
            whisper_cfg.attention_dropout = cfg.whisper_dropout
        whisper = WhisperForConditionalGeneration.from_pretrained(
            cfg.whisper_name, config=whisper_cfg
        )
        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
        )
        self.whisper = get_peft_model(whisper, lora_cfg)

        # CTC heads
        # Unfold consecutive tokens into a single CTC frame (80 ms windows).
        self._unfold_width = 4 // cfg.embedder.stride_2  # 2 when stride_2=2
        self._stride_2 = cfg.embedder.stride_2
        d_model = cfg.embedder.d_model
        ctc_in = d_model * self._unfold_width
        vocab_size = whisper.config.vocab_size
        self.phone_head = nn.Linear(ctc_in, _N_PHONE_CLASSES)
        self.char_head = nn.Linear(ctc_in, vocab_size)

        self._d_model = d_model

    def forward(self, batch: NeuralBatch, encoder_only: bool = False) -> dict:
        """
        encoder_only=True: run embedder + BWEncoder + CTC heads only; skip the
        Whisper decoder.  Used for Phase 2 of two-phase training where CE loss
        is not needed and running the decoder would waste ~half the step cost.
        """
        embedded = self._embed(batch.signals, batch.sbj_idx, batch.day_idx)
        enc_out = self.encoder(embedded)

        # Intermediate hidden state for CTC heads
        intermediate = enc_out.hidden_states[self._last_phoneme_layer + 1]
        phone_logits = self.phone_head(self._ctc_unfold(intermediate))
        char_logits = self.char_head(self._ctc_unfold(intermediate))

        # CTC frame count per sample
        raw_len = batch.lengths
        x_len_emb = (raw_len // self._stride_2 + raw_len % self._stride_2).clamp(
            max=_MAX_SOURCE_POSITIONS
        )
        x_len = (x_len_emb // self._unfold_width).long()

        if encoder_only:
            return {
                "phone_logits": phone_logits,
                "char_logits": char_logits,
                "x_len": x_len,
            }

        whisper_out = self.whisper(
            encoder_outputs=enc_out,
            labels=batch.labels,
        )
        return {
            "ce_loss": whisper_out.loss,
            "phone_logits": phone_logits,
            "char_logits": char_logits,
            "x_len": x_len,
        }

    def generate(self, batch: NeuralBatch, **kwargs) -> Tensor:
        embedded = self._embed(batch.signals, batch.sbj_idx, batch.day_idx)
        enc_out = self.encoder(embedded)
        # whisper-tiny.en supplies its own decoder prompt via generation_config;
        # forcing multilingual language/task tokens corrupts free generation.
        return self.whisper.generate(encoder_outputs=enc_out, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, signals: Tensor, sbj_idx: Tensor, day_idx: Tensor) -> Tensor:
        """Route each sample in the batch to its subject-group embedder."""
        B = signals.shape[0]
        out = torch.empty(
            B, 1500, self._d_model, device=signals.device, dtype=signals.dtype
        )
        for s, embedder in enumerate(self.embedders):
            mask = sbj_idx == s
            if mask.any():
                out[mask] = embedder(signals[mask], day_idx[mask])
        return out

    def _ctc_unfold(self, x: Tensor) -> Tensor:
        """(B, 1500, d_model) → (B, T_ctc, d_model * unfold_width)"""
        B, T, D = x.shape
        W = self._unfold_width
        T_ctc = T // W
        return x[:, : T_ctc * W, :].reshape(B, T_ctc, W * D)

    def unfreeze_whisper_decoder(self) -> None:
        """Enable gradients for all Whisper decoder parameters.

        Called by UnfreezeWhisperDecoder callback at a configured epoch.
        With LoRA, only the adapter parameters would train by default.
        Unfreezing allows to train the full base-model decoder as well.
        """
        for name, param in self.whisper.named_parameters():
            if "decoder" in name:
                param.requires_grad_(True)
