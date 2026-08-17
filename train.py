import os
from pathlib import Path

import hydra
import lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from transformers import WhisperTokenizer

import brainwhisperer.config.schema
import brainwhisperer.data.datasets  # noqa: F401 - unused in this script but triggers dataset registration
from brainwhisperer.data.collate import NeuralCollator
from brainwhisperer.data.registry import get_dataset
from brainwhisperer.data.samplers import StratifiedDayBatchSampler
from brainwhisperer.models.model import BrainWhisperer
from brainwhisperer.training.callbacks import UnfreezeWhisperDecoder, VizCallback
from brainwhisperer.training.lightning_module import BWLightningModule
from brainwhisperer.utils.sessions import SESSIONS_BY_DATASET, sessions_to_calendar_days


@hydra.main(config_path="brainwhisperer/config", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    OmegaConf.to_container(cfg, throw_on_missing=True)

    run_dir = Path(HydraConfig.get().runtime.output_dir)

    torch.set_float32_matmul_precision(cfg.training.matmul_precision)

    # Seeting seed for reproducibility
    pl.seed_everything(cfg.training.seed, workers=True)

    # Set RANK and WORLD_SIZE for torchrun; default to 0 and 1 for single-GPU
    _rank = int(os.environ.get("RANK", 0))
    _world_size = int(os.environ.get("WORLD_SIZE", 1))

    assert cfg.data.batch_size % _world_size == 0, (
        f"Batch size ({cfg.data.batch_size}) must be divisible by world size ({_world_size})"
    )
    _per_rank_batch = cfg.data.batch_size // _world_size

    # Scale days_per_batch by world size alongside batch_size, so each rank keeps
    # the correct samples-per-day.
    assert cfg.data.days_per_batch % _world_size == 0, (
        f"days_per_batch ({cfg.data.days_per_batch}) must be divisible by "
        f"world size ({_world_size})"
    )
    _per_rank_days = cfg.data.days_per_batch // _world_size

    # Validate dataset and embedder counts match
    assert len(cfg.data.datasets) == cfg.model.n_embedders, (
        f"The number of datasets ({len(cfg.data.datasets)}) must equal "
        f"the number of embedders ({cfg.model.n_embedders}). "
        f"Each entry in data.datasets maps to one BWEmbedder at its list position."
    )

    # ------------------------------------------------------------------
    # Tokenizer + collator
    # ------------------------------------------------------------------
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.model.whisper_name, language="en", task="transcribe"
    )
    collator = NeuralCollator(tokenizer, max_label_len=cfg.data.max_seq_len)

    # ------------------------------------------------------------------
    # Datasets
    #
    # String position in cfg.data.datasets assing the `sbj_idx` to each dataset used.
    # The value is constructed with the index that matches its position so that
    # BrainWhisperer.embedders[i] always handles the dataset at position i.
    # ------------------------------------------------------------------
    dataset_classes = [get_dataset(name) for name in cfg.data.datasets]
    num_days_per_embedder = [cls.NUM_DAYS for cls in dataset_classes]
    calendar_days_per_embedder = [
        sessions_to_calendar_days(SESSIONS_BY_DATASET[name])
        for name in cfg.data.datasets
    ]
    val_names = list(cfg.data.datasets)

    train_datasets, val_dataloaders = [], []
    for i, cls in enumerate(dataset_classes):
        train_datasets.append(cls(split="train", sbj_idx=i))
        val_ds = cls(split=cls.VAL_SPLIT, sbj_idx=i)
        val_dataloaders.append(
            DataLoader(
                val_ds,
                batch_size=_per_rank_batch,
                collate_fn=collator,
                num_workers=cfg.data.num_workers,
                shuffle=False,
            )
        )

    # ------------------------------------------------------------------
    # Training DataLoader
    # ------------------------------------------------------------------
    train_combined = ConcatDataset(train_datasets)
    train_sampler = StratifiedDayBatchSampler(
        train_combined,
        batch_size=_per_rank_batch,
        days_per_batch=_per_rank_days,
        rank=_rank,
        world_size=_world_size,
        seed=cfg.training.seed,
    )
    train_dl = DataLoader(
        train_combined,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
    )

    # ------------------------------------------------------------------
    # Model + Lightning wrapper
    # ------------------------------------------------------------------
    model = BrainWhisperer(cfg.model, num_days_per_embedder, calendar_days_per_embedder)
    lm = BWLightningModule(model, cfg.training, tokenizer, val_names)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    ckpt_dir = run_dir / "checkpoints"
    primary_val_metric = f"val_wer_{val_names[0]}"

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=primary_val_metric,
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:04d}",
        ),
        UnfreezeWhisperDecoder(
            unfreeze_at_epoch=cfg.training.unfreeze_decoder_epoch,
        ),
    ]

    if cfg.viz.enabled:
        held_out = next(iter(val_dataloaders[0]))
        callbacks.append(
            VizCallback(held_out, log_every_n_epochs=cfg.viz.log_every_n_epochs)
        )

    # ------------------------------------------------------------------
    # Logger: CSVLogger by default. W&B set via experiment config
    # ------------------------------------------------------------------
    logger = CSVLogger(save_dir=str(run_dir), name="logs")

    # ------------------------------------------------------------------
    # Save the full config
    # ------------------------------------------------------------------
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        # Detects if single or multi-GPU
        devices=_world_size if _world_size > 1 else cfg.training.devices,
        # Avoids DDP errors for unused params
        strategy=DDPStrategy(static_graph=True) if _world_size > 1 else "auto",
        # Using the custom StratifiedDayBatchSampler forces a workaround;
        # all ranks run the full validation set, redundant but correct.
        use_distributed_sampler=False,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(lm, train_dataloaders=train_dl, val_dataloaders=val_dataloaders)


if __name__ == "__main__":
    main()
