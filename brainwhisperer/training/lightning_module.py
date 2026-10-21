"""
Lightning wrapper for BrainWhisperer training.
"""

from __future__ import annotations

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchaudio.functional import edit_distance as _edit_distance
from torchmetrics.text import WordErrorRate
from transformers import PreTrainedTokenizer

from brainwhisperer.config.schema import BWTrainingConfig
from brainwhisperer.data.augmentations import (
    apply_noise,
    gaussian_smooth,
    random_time_cut,
    time_mask,
    transform_data,
)
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.models.model import BrainWhisperer
from brainwhisperer.training.losses import ctc_loss
from brainwhisperer.utils.text import build_normalizer


class BWLightningModule(LightningModule):
    """
    Lightning wrapper.

    Args:
        model:                BrainWhisperer instance, fully configured.
        cfg:                  BWTrainingConfig (from Hydra).
        tokenizer:            Whisper tokenizer, used to decode generated ids and
                              to recover target strings from NeuralBatch.labels.
        val_dataloader_names: One name per validation DataLoader, in order.
                              Used as suffixes on logged metrics, e.g. "val_wer_card".
    """

    def __init__(
        self,
        model: BrainWhisperer,
        cfg: BWTrainingConfig,
        tokenizer: PreTrainedTokenizer,
        val_dataloader_names: list[str],
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["model", "tokenizer"]
        )  # model and tokenizer are saved separately

        self.model = model
        self.cfg = cfg
        self._tokenizer = tokenizer
        self._val_names = val_dataloader_names

        a = cfg.augmentation
        self._aug = {
            "gain_std": a.gain_std,
            "noise_std": a.noise_std,
            "offset_std": a.offset_std,
            "walk_std": a.walk_std,
            "max_cut": a.max_cut,
            "smooth_kernel_size": a.smooth_kernel_size,
            "smooth_sigma": a.smooth_sigma,
            "num_masks": a.num_masks,
            "max_mask_pct": a.max_mask_pct,
        }

        self._normalizer = build_normalizer()
        self._wer = WordErrorRate()

        N = len(val_dataloader_names)
        self._val_hyps: list[list[str]] = [[] for _ in range(N)]
        self._val_refs: list[list[str]] = [[] for _ in range(N)]
        self._val_edit_phones: list[int] = [0] * N
        self._val_len_phones: list[int] = [0] * N

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        param_groups = [
            {
                # BWEmbedder
                "params": self.model.embedders.parameters(),
                "lr": self.cfg.lr,
                "weight_decay": self.cfg.weight_decay,
            },
            {
                # BWEncoder
                "params": self.model.encoder.parameters(),
                "lr": self.cfg.lr,
                "weight_decay": self.cfg.weight_decay,
            },
            {
                # Whisper decoder + LoRA adapter. LR is fixed at 1e-4, as in
                # the published runs.
                # Excluding frozen params (whisper.model.encoder) to avoid errors on multi-GPU.
                "params": [
                    p for p in self.model.whisper.parameters() if p.requires_grad
                ],
                "lr": 1e-4,
                "weight_decay": self.cfg.weight_decay,
            },
        ]
        if self.cfg.ctc_phone_coeff > 0.0:
            param_groups.append(
                {
                    "params": self.model.phone_head.parameters(),
                    "lr": self.cfg.lr,
                    "weight_decay": self.cfg.weight_decay,
                }
            )

        optimizer = Adam(param_groups)

        if self.cfg.scheduler == "cosine":
            sched = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.cfg.lr_min,
            )
        elif self.cfg.scheduler == "linear":
            sched = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=self.cfg.lr_min / self.cfg.lr,
                total_iters=self.trainer.max_epochs,
            )
        elif self.cfg.scheduler == "sequential":
            milestone = 100
            assert self.trainer.max_epochs > milestone, (
                f"Sequential scheduler requires `max_epochs` > {milestone}, "
                f"got {self.trainer.max_epochs}"
            )
            sched = SequentialLR(
                optimizer,
                schedulers=[
                    CosineAnnealingLR(
                        optimizer, T_max=milestone, eta_min=self.cfg.lr_min
                    ),
                    LinearLR(
                        optimizer,
                        start_factor=self.cfg.lr_min / self.cfg.lr,
                        end_factor=0.0,
                        total_iters=self.trainer.max_epochs - milestone,
                    ),
                ],
                milestones=[milestone],
            )
        else:
            raise ValueError(
                f"Unknown scheduler {self.cfg.scheduler!r}. "
                "Choose 'cosine', 'linear', or 'sequential'."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: NeuralBatch, batch_idx: int):
        x_noised = apply_noise(
            batch.signals,
            self._aug["gain_std"],
            self._aug["noise_std"],
            self._aug["offset_std"],
            self._aug["walk_std"],
        )

        # Phase 1 (CE loss on noise-augmented signal, uncut and unmasked).
        x_clean = gaussian_smooth(
            x_noised, self._aug["smooth_kernel_size"], self._aug["smooth_sigma"]
        )
        clean_batch = NeuralBatch(
            signals=x_clean,
            labels=batch.labels,
            datasets=batch.datasets,
            day_idx=batch.day_idx,
            lengths=batch.lengths,
        )

        # Disable dropout so CE loss is computed deterministically.
        self.model.embedders.eval()
        self.model.encoder.eval()
        out_clean = self.model(clean_batch)
        self.model.embedders.train()
        self.model.encoder.train()

        ce_loss = out_clean["ce_loss"]
        self.log("train_ce_loss", ce_loss, prog_bar=True)
        loss = self.cfg.ce_coeff * ce_loss

        # Phase 2 (CTC loss on the same noise-augmented signal, which is also cut, smoothed and masked).
        x_cut, len_cut = random_time_cut(x_noised, batch.lengths, self._aug["max_cut"])
        x_smoothed = gaussian_smooth(
            x_cut, self._aug["smooth_kernel_size"], self._aug["smooth_sigma"]
        )
        x_aug = time_mask(
            x_smoothed, len_cut, self._aug["num_masks"], self._aug["max_mask_pct"]
        )
        aug_batch = NeuralBatch(
            signals=x_aug,
            labels=batch.labels,
            datasets=batch.datasets,
            day_idx=batch.day_idx,
            lengths=len_cut,
        )
        out_aug = self.model(aug_batch, encoder_only=True)

        if self.cfg.ctc_phone_coeff > 0.0 and batch.phone_seq is not None:
            phone_loss = ctc_loss(
                logits=out_aug["phone_logits"],
                targets=batch.phone_seq,
                input_lengths=out_aug["x_len"],
                target_lengths=batch.phone_seq_len,
            )
            self.log("train_ctc_phone_loss", phone_loss, prog_bar=True)
            loss = loss + self.cfg.ctc_phone_coeff * phone_loss

        self.log("train_loss", loss, prog_bar=False)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: NeuralBatch, batch_idx: int, dataloader_idx: int = 0
    ):
        x_val, len_val = transform_data(
            batch.signals, batch.lengths, mode="val", **self._aug
        )
        val_batch = NeuralBatch(
            signals=x_val,
            labels=batch.labels,
            datasets=batch.datasets,
            day_idx=batch.day_idx,
            lengths=len_val,
        )
        out = self.model(val_batch)

        name = (
            self._val_names[dataloader_idx]
            if dataloader_idx < len(self._val_names)
            else str(dataloader_idx)
        )

        ce_loss = out["ce_loss"]
        self.log(
            f"val_ce_loss_{name}",
            ce_loss,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,  # Average across distributed devices
        )

        if self.cfg.ctc_phone_coeff > 0.0 and batch.phone_seq is not None:
            phone_loss = ctc_loss(
                logits=out["phone_logits"],
                targets=batch.phone_seq,
                input_lengths=out["x_len"],
                target_lengths=batch.phone_seq_len,
            )
            self.log(
                f"val_ctc_phone_loss_{name}",
                phone_loss,
                prog_bar=False,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            self._accumulate_per(
                out["phone_logits"],
                out["x_len"],
                batch.phone_seq,
                batch.phone_seq_len,
                dataloader_idx,
            )

        # WER: decode via generate() and accumulate for on_validation_epoch_end
        generated_ids = self.model.generate(val_batch)
        transcriptions = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        target_texts = self._labels_to_text(batch.labels)

        self._val_hyps[dataloader_idx].extend(transcriptions)
        self._val_refs[dataloader_idx].extend(target_texts)

    def on_validation_epoch_end(self):
        for i, name in enumerate(self._val_names):
            # All ranks see the full validation set,no gather or all_reduce needed.
            # sync_dist=True averages identical values. No-op silencing warnings.
            hyps = [self._normalizer(t) for t in self._val_hyps[i]]
            refs = [self._normalizer(t) for t in self._val_refs[i]]
            if hyps and refs:
                wer = self._wer(hyps, refs)
                self.log(f"val_wer_{name}", wer, prog_bar=(i == 0), sync_dist=True)

            if self._val_len_phones[i] > 0:
                per = self._val_edit_phones[i] / self._val_len_phones[i]
                self.log(f"val_per_{name}", per, prog_bar=(i == 0), sync_dist=True)

            # Reset accumulators
            self._val_hyps[i].clear()
            self._val_refs[i].clear()
            self._val_edit_phones[i] = 0
            self._val_len_phones[i] = 0

    # ------------------------------------------------------------------
    # Grad norm logging
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer):
        # Log decoder grad norms separately from the encoder.
        # The encoder receives no gradients in this architecture, so
        # logging the full PeftModel would dilute the decoder mean toward zero.
        whisper_decoder = self.model.whisper.model.decoder
        for tag, module in [
            ("embedders", self.model.embedders),
            ("encoder", self.model.encoder),
            ("whisper_decoder", whisper_decoder),
        ]:
            norms = grad_norm(module, norm_type=2)
            layer_norms = [v.cpu().item() for k, v in norms.items() if "total" not in k]
            if layer_norms:
                self.log(f"grad_norm/{tag}", np.mean(layer_norms), on_step=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _labels_to_text(self, labels: torch.Tensor) -> list[str]:
        """Decode NeuralBatch.labels (BOS-stripped, -100 padded) back to strings."""
        ids = labels.clone()
        ids[ids == -100] = self._tokenizer.pad_token_id
        return self._tokenizer.batch_decode(ids, skip_special_tokens=True)

    def _accumulate_per(
        self,
        phone_logits: torch.Tensor,
        x_len: torch.Tensor,
        phone_seq: torch.Tensor,
        phone_seq_len: torch.Tensor,
        dataloader_idx: int,
    ) -> None:
        """CTC-decode phone logits and accumulate edit distance for PER."""
        for i in range(phone_logits.shape[0]):
            decoded = torch.argmax(phone_logits[i, : x_len[i], :].detach(), dim=-1)
            decoded = torch.unique_consecutive(decoded).cpu().numpy()
            decoded = np.array([p for p in decoded if p != 0])  # remove blanks

            true_seq = phone_seq[i, : phone_seq_len[i]].cpu().numpy()

            self._val_edit_phones[dataloader_idx] += _edit_distance(
                decoded.tolist(), true_seq.tolist()
            )
            self._val_len_phones[dataloader_idx] += int(phone_seq_len[i])
