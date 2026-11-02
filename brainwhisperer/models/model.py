from pathlib import Path

import torch
import torch.distributed.tensor
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_model, save_file
from torch import Tensor, nn
from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from brainwhisperer.config.schema import BWEmbedderConfig, BWModelConfig
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.models.embedder import BWEmbedder
from brainwhisperer.models.encoder import BWEncoder
from brainwhisperer.utils.fill import FILL_NONE, fill_to_num_features

_N_PHONE_CLASSES = 41  # CMU phoneme set used in CARD/Willet

# Layout version of the config.json written by save_pretrained.
HUB_FORMAT_VERSION = 1
# Whisper's own encoder is never run (BWEncoder replaces it), so
# save_pretrained leaves its weights out.
WHISPER_ENCODER_PREFIX = "whisper.model.encoder."


class BrainWhisperer(nn.Module, PyTorchModelHubMixin, library_name="brainwhisperer"):
    """
    Top-level model, receives neural signals and outputs transcribed text.
    BrainWhisperer (BW) follows a modular design, composed of:
        - one BWEmbedder per subject group, where embedding conv. weights are shared across all
        embedders.
        - a BWEncoder, introducing custom elements w.r.t the vanilla Whisper encoder
        - a WhisperForConditionalGeneration, used unmodified except for LoRA on the decoder.

    Usage: BrainWhisperer(cfg, embedder_names=["card", ...], num_days_per_embedder=[45, 24, ...])
    forward() returns:
        ce_loss:      scalar, cross-entropy from Whisper decoder
        phone_logits: (B, T_ctc, 41)
    generate() returns token ids (B, L).

    save_pretrained(folder) / BrainWhisperer.from_pretrained(folder or Hub name)
    store the weights plus a config.json holding every constructor argument.
    `inference` carries the settings that reproduce the reported scores and
    is available after loading as `model.inference`.
    """

    def __init__(
        self,
        cfg: BWModelConfig | dict,
        embedder_names: list[str],
        num_days_per_embedder: list[int],
        calendar_days_per_embedder: list[list[int]] | None = None,
        embedder_cfg_per_embedder: list[BWEmbedderConfig | dict] | None = None,
        channel_fills: list[str] | None = None,
        inference: dict | None = None,
        format_version: int = HUB_FORMAT_VERSION,
        pretrained_whisper: bool = True,
    ):
        super().__init__()

        if format_version != HUB_FORMAT_VERSION:
            raise ValueError(
                f"Saved model has config format {format_version}; this code "
                f"reads format {HUB_FORMAT_VERSION}"
            )
        # Configs arrive as dataclasses or DictConfigs from train/eval, and as
        # plain dicts from config.json. Merging onto the schema checks keys and
        # types the same way for all three.
        cfg = OmegaConf.merge(OmegaConf.structured(BWModelConfig), cfg)
        if embedder_cfg_per_embedder is not None:
            embedder_cfg_per_embedder = [
                OmegaConf.merge(OmegaConf.structured(BWEmbedderConfig), c)
                for c in embedder_cfg_per_embedder
            ]

        assert len(embedder_names) == len(num_days_per_embedder) == cfg.n_embedders, (
            f"embedder_names ({len(embedder_names)}) and "
            f"num_days_per_embedder ({len(num_days_per_embedder)}) must both "
            f"equal cfg.n_embedders={cfg.n_embedders}"
        )

        # Name-aligned, one BWEmbedderConfig per subject group. Defaults
        # to `cfg.embedder` for every group unless a per-dataset override is found
        embedder_cfgs = embedder_cfg_per_embedder or [cfg.embedder] * len(
            embedder_names
        )
        assert len(embedder_cfgs) == len(embedder_names), (
            f"embedder_cfg_per_embedder has {len(embedder_cfgs)} entries "
            f"but {len(embedder_names)} embedder names were given"
        )

        # Narrow datasets (willet 256, jude 256, kunz_t12 256) are widened
        # to the shared conv1 width before their embedder sees them.
        # Defaults to "none" (pass-through) for every group.
        fills = (
            channel_fills
            if channel_fills is not None
            else [FILL_NONE] * len(embedder_names)
        )
        assert len(fills) == len(embedder_names), (
            f"Channel_fills has {len(fills)} entries "
            f"but {len(embedder_names)} embedder names were given"
        )
        self._channel_fills = dict(zip(embedder_names, fills))
        self._num_channels = cfg.embedder.num_features

        # Derive calendar-based month grouping from the acquisition dates.
        cal_days_list = (
            calendar_days_per_embedder
            if calendar_days_per_embedder is not None
            else [None] * len(num_days_per_embedder)
        )
        self.embedders = nn.ModuleDict(
            {
                name: BWEmbedder(
                    emb_cfg,
                    num_days,
                    session_to_month=(
                        [d // 30 for d in cal_days] if cal_days is not None else None
                    ),
                    session_to_calendar_day=(
                        cal_days if emb_cfg.encodings == "sin" else None
                    ),
                )
                for name, emb_cfg, num_days, cal_days in zip(
                    embedder_names, embedder_cfgs, num_days_per_embedder, cal_days_list
                )
            }
        )
        # Share conv weights across subjects
        if cfg.n_embedders > 1:
            embedders = list(self.embedders.values())
            for emb in embedders[1:]:
                emb.conv1 = embedders[0].conv1
                emb.conv2 = embedders[0].conv2

        # Encoder
        self.encoder = BWEncoder(cfg.encoder)
        self._last_phoneme_layer = cfg.encoder.last_phoneme_layer

        # Original Whisper decoder + LoRA
        whisper_cfg = WhisperConfig.from_pretrained(cfg.whisper_name)
        if cfg.whisper_dropout > 0.0:
            whisper_cfg.dropout = cfg.whisper_dropout
            whisper_cfg.activation_dropout = cfg.whisper_dropout
            whisper_cfg.attention_dropout = cfg.whisper_dropout
        if pretrained_whisper:
            whisper = WhisperForConditionalGeneration.from_pretrained(
                cfg.whisper_name, config=whisper_cfg
            )
        else:
            whisper = WhisperForConditionalGeneration(whisper_cfg)
            whisper.generation_config = GenerationConfig.from_pretrained(
                cfg.whisper_name
            )
        # HF models load in eval() mode by default, explicitly calling `train()`
        # activates dropout in decoder during training.
        whisper.train()
        # Force the full decoder prompt [<|sot|>, <|en|>, <|transcribe|>, <|notimestamps|>].
        _prefix = WhisperTokenizer.from_pretrained(
            cfg.whisper_name, language="en", task="transcribe"
        ).prefix_tokens
        whisper.generation_config.forced_decoder_ids = [
            [i, tok] for i, tok in enumerate(_prefix[1:], start=1)
        ]
        lora_cfg = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
        )
        # Scopes LoRA to the decoder submodule only; avoid wasted VRAM.
        # Whisper's built-in encoder is always bypassed (BWEncoder replaces it); freeze it so
        # distributed training does not error on unused parameters.
        whisper.model.encoder.requires_grad_(False)
        whisper.model.decoder = get_peft_model(whisper.model.decoder, lora_cfg)
        self.whisper = whisper

        # CTC heads
        # Unfold consecutive tokens into a single CTC frame (80 ms windows).
        self._unfold_width = 4 // cfg.embedder.stride_2  # 2 when stride_2=2
        self._stride_2 = cfg.embedder.stride_2
        d_model = cfg.embedder.d_model
        ctc_in = d_model * self._unfold_width
        self.phone_head = nn.Linear(ctc_in, _N_PHONE_CLASSES)

        self._d_model = d_model
        self.max_source_pos = cfg.encoder.max_source_pos

        self.inference = inference
        # The tokenizer that decodes generate()'s token ids.
        self.whisper_name = cfg.whisper_name
        # What save_pretrained writes to config.json,
        # keyed by constructor argument
        self._hub_mixin_config = {
            "format_version": format_version,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "embedder_names": list(embedder_names),
            "num_days_per_embedder": [int(n) for n in num_days_per_embedder],
            "calendar_days_per_embedder": [
                None if days is None else [int(d) for d in days]
                for days in cal_days_list
            ],
            "embedder_cfg_per_embedder": [
                OmegaConf.to_container(c, resolve=True) for c in embedder_cfgs
            ],
            "channel_fills": list(fills),
            "inference": inference,
            "pretrained_whisper": False,
        }

    def forward(self, batch: NeuralBatch, encoder_only: bool = False) -> dict:
        """
        encoder_only=True: run embedder + BWEncoder + CTC heads only; skip the
        Whisper decoder.  Used for Phase 2 of two-phase training where CE loss
        is not needed and running the decoder would waste ~half the step cost.
        """
        embedded = self._embed(batch.signals, batch.datasets, batch.day_idx)
        enc_out = self.encoder(embedded)

        # Intermediate hidden state for CTC heads
        intermediate = enc_out.hidden_states[self._last_phoneme_layer + 1]
        phone_logits = self.phone_head(self._ctc_unfold(intermediate))

        # CTC frame count per sample
        raw_len = batch.lengths
        x_len_emb = (raw_len // self._stride_2 + raw_len % self._stride_2).clamp(
            max=self.max_source_pos
        )
        x_len = (x_len_emb // self._unfold_width).long()

        if encoder_only:
            return {
                "phone_logits": phone_logits,
                "x_len": x_len,
            }

        whisper_out = self.whisper(
            encoder_outputs=enc_out,
            labels=batch.labels,
        )
        return {
            "ce_loss": whisper_out.loss,
            "phone_logits": phone_logits,
            "x_len": x_len,
        }

    def generate(self, batch: NeuralBatch, **kwargs) -> Tensor:
        embedded = self._embed(batch.signals, batch.datasets, batch.day_idx)
        enc_out = self.encoder(embedded)
        return self.whisper.generate(encoder_outputs=enc_out, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, signals: Tensor, datasets: list[str], day_idx: Tensor) -> Tensor:
        """Route each sample in the batch to its subject-group embedder by name."""
        B = signals.shape[0]
        unknown = sorted({d for d in datasets} - set(self.embedders))
        if unknown:
            raise KeyError(
                f"Batch carries unknown dataset(s) {unknown}; "
                f"model embedders are keyed {list(self.embedders)}"
            )
        out = torch.empty(
            B, 1500, self._d_model, device=signals.device, dtype=signals.dtype
        )
        for name, embedder in self.embedders.items():
            mask = torch.tensor([d == name for d in datasets], device=signals.device)
            if mask.any():
                # Fill to the shared input width first: the day projection
                # (xi @ W) mixes channels, so it must see the same 512-wide
                # layout the checkpoint was trained on.
                x = fill_to_num_features(
                    signals[mask], self._channel_fills[name], self._num_channels
                )
                out[mask] = embedder(x, day_idx[mask])
        return out

    def _ctc_unfold(self, x: Tensor) -> Tensor:
        """(B, 1500, d_model) → (B, T_ctc, d_model * unfold_width)"""
        B, T, D = x.shape
        W = self._unfold_width
        T_ctc = T // W
        return x[:, : T_ctc * W, :].unfold(1, W, W).reshape(B, T_ctc, D * W)

    def unfreeze_whisper_decoder(self) -> None:
        """Enable gradients for all Whisper decoder parameters.

        Called by UnfreezeWhisperDecoder callback at a configured epoch.
        With LoRA, only the adapter parameters would train by default.
        Unfreezing allows to train the full base-model decoder as well.
        """
        for name, param in self.whisper.named_parameters():
            if "decoder" in name:
                param.requires_grad_(True)

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        """Load the weights of a checkpoint written by train.py (strict).

        Lightning stores this model under the ``model.`` prefix of the
        checkpoint's state dict (``BWLightningModule.model``).
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = {
            k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        self.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # save_pretrained / from_pretrained (PyTorchModelHubMixin hooks)
    # ------------------------------------------------------------------

    def _save_pretrained(self, save_directory: Path) -> None:
        """Write the weights, without Whisper's encoder, as safetensors."""
        state_dict = {}
        stored = set()
        for name, tensor in self.state_dict().items():
            # Shared conv weights and Whisper's tied output layer are stored
            # once; loading links the other names back to the same tensor.
            if name.startswith(WHISPER_ENCODER_PREFIX) or tensor.data_ptr() in stored:
                continue
            if tensor.numel():
                stored.add(tensor.data_ptr())
            state_dict[name] = tensor.contiguous()
        save_file(
            state_dict,
            Path(save_directory) / SAFETENSORS_SINGLE_FILE,
            metadata={"format": "pt"},
        )

    @classmethod
    def _from_pretrained(cls, *, strict: bool = True, **kwargs):
        # The mixin loads non-strictly by default; a missing weight must fail.
        return super()._from_pretrained(strict=strict, **kwargs)

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        missing, unexpected = load_model(
            model, model_file, strict=False, device=map_location
        )
        missing = [k for k in missing if not k.startswith(WHISPER_ENCODER_PREFIX)]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"{model_file} does not match the model: "
                f"missing {sorted(missing)}, unexpected {sorted(unexpected)}"
            )
        # Ready for inference, like a transformers model after from_pretrained.
        return model.eval()
