"""
Training callbacks for BrainWhisperer.

VizCallback is the only bridge between training and visualization.
All the graph logic is isolated from this submodule.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from lightning import Callback
from lightning.pytorch.callbacks import BaseFinetuning
from transformers import PreTrainedTokenizer, StoppingCriteria

from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.viz.embeddings import plot_day_encoding, plot_patch_embeddings
from brainwhisperer.viz.encoder import (
    plot_hidden_state_norms,
    plot_windowed_attention_mask,
)
from brainwhisperer.viz.training import plot_loss_curves, plot_wer_curve


class StopAtPeriodCriteria(StoppingCriteria):
    """Halts generation as soon as a period appears in the decoded output.
    Pass inside a StoppingCriteriaList to model.generate().
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, start_len: int):
        self.tokenizer = tokenizer
        self.start_len = start_len  # index where new tokens begin in input_ids[0]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        decoded = self.tokenizer.decode(
            input_ids[0][self.start_len :], skip_special_tokens=True
        )
        return "." in decoded


class UnfreezeWhisperDecoder(BaseFinetuning):
    """Unfreezes all Whisper decoder parameters at a given training epoch.
    Set unfreeze_at_epoch=-1 to never unfreeze.
    """

    def __init__(self, unfreeze_at_epoch: int):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module) -> None:
        # Decoder starts frozen via LoRA in BrainWhisperer; nothing to do here.
        pass

    def finetune_function(self, pl_module, current_epoch: int, optimizer) -> None:
        if current_epoch == self._unfreeze_at_epoch:
            pl_module.model.unfreeze_whisper_decoder()


class VizCallback(Callback):
    """Runs viz/ functions every N epochs and logs figures to W&B.

    This is the only place in training/ that imports from viz/.
    It holds a fixed held-out batch so visualisations are comparable
    across epochs without re-drawing from the DataLoader.

    Metrics accumulated internally:
        - train_ce_loss, train_ctc_phone_loss  (per epoch)
        - val_wer  (per validation epoch, first dataloader)

    Args:
        held_out_batch:    A NeuralBatch kept in CPU memory.
        log_every_n_epochs: How often to fire (counting from epoch 0).
    """

    def __init__(self, held_out_batch: NeuralBatch, log_every_n_epochs: int = 5):
        self._batch = held_out_batch
        self.log_every_n_epochs = log_every_n_epochs
        self._step_idxs: list[int] = []
        self._ce_losses: list[float] = []
        self._ctc_losses: list[float] = []
        self._wer_epochs: list[int] = []
        self._wer_values: list[float] = []

    # ------------------------------------------------------------------
    # Metric accumulation
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        self._step_idxs.append(epoch)
        self._ce_losses.append(float(metrics.get("train_ce_loss", float("nan"))))
        self._ctc_losses.append(
            float(metrics.get("train_ctc_phone_loss", float("nan")))
        )

        if (epoch + 1) % self.log_every_n_epochs == 0:
            self._log_figures(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        for key, val in trainer.callback_metrics.items():
            if "wer" in key:
                self._wer_epochs.append(epoch)
                self._wer_values.append(float(val))
                break

    # ------------------------------------------------------------------
    # Figure generation and W&B logging
    # ------------------------------------------------------------------

    def _log_figures(self, trainer, pl_module) -> None:
        # Non-zero ranks skip the forward pass and figure rendering
        # rather than doing it and discarding it.
        if not trainer.is_global_zero:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        device = pl_module.device
        model = pl_module.model
        model.eval()

        batch = NeuralBatch(
            signals=self._batch.signals.to(device),
            labels=self._batch.labels.to(device),
            sbj_idx=self._batch.sbj_idx.to(device),
            day_idx=self._batch.day_idx.to(device),
            lengths=self._batch.lengths.to(device),
        )

        figs: dict = {}

        with torch.no_grad():
            embedded = model._embed(batch.signals, batch.sbj_idx, batch.day_idx)
            enc_out = model.encoder(embedded)

        figs["embeddings/patch_embeddings"] = plot_patch_embeddings(embedded)
        figs["embeddings/day_encoding"] = plot_day_encoding(
            model.embedders[0],
            torch.arange(model.embedders[0].num_days, device=device),
        )
        figs["encoder/hidden_state_norms"] = plot_hidden_state_norms(enc_out)
        figs["encoder/windowed_attention_mask"] = plot_windowed_attention_mask(
            model.encoder
        )

        if self._step_idxs:
            figs["training/loss_curves"] = plot_loss_curves(
                self._step_idxs, self._ce_losses, self._ctc_losses
            )
        if self._wer_epochs:
            figs["training/wer_curve"] = plot_wer_curve(
                self._wer_epochs, self._wer_values
            )

        wandb.log(
            {k: wandb.Image(v) for k, v in figs.items()},
            step=trainer.global_step,
        )
        for fig in figs.values():
            plt.close(fig)

        model.train()
