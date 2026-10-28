import os
import subprocess
import sys
from pathlib import Path

import hydra
import lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from neuralshed.dataloader import StratifiedBatchSampler
from torch.utils.data import DataLoader
from transformers import WhisperTokenizer

import brainwhisperer.config.schema
from brainwhisperer.data.collate import (
    MixedRecipeCollator,
    SingleRecipeCollator,
    default_ragged_collator,
)
from brainwhisperer.data.collection import build_train_collection
from brainwhisperer.data.recipes import RECIPES, build_in_parallel
from brainwhisperer.models.model import BrainWhisperer
from brainwhisperer.training.callbacks import UnfreezeWhisperDecoder, VizCallback
from brainwhisperer.training.lightning_module import BWLightningModule


def _needs_torchrun(devices: int) -> bool:
    """Flag for multi-GPU training if distributed launcher is unset.
    `torchrun` and `torchelastic` set RANK and TORCHELASTIC_RUN_ID in every worker
    process. If those are set, the multi-GPU delegation needs to be skipped
    """
    already_distributed = "TORCHELASTIC_RUN_ID" in os.environ or "RANK" in os.environ
    return devices > 1 and not already_distributed


@hydra.main(config_path="brainwhisperer/config", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Unused config here; serving as guard throwing errors on missing values
    OmegaConf.to_container(cfg, throw_on_missing=True)

    # ------------------------------------------------------------------
    # Multi-GPU delegation
    # ------------------------------------------------------------------
    if _needs_torchrun(cfg.training.devices):
        run_dir = HydraConfig.get().runtime.output_dir
        overrides = list(HydraConfig.get().overrides.task)
        # torchrun's static rendezvous always binds port 29500 unless told
        # otherwise, so a second concurrent multi-GPU run dies immediately
        # with EADDRINUSE (seen on 2026-10-19 launching 4 seeds). Honour
        # MASTER_PORT when the caller sets one; leave torchrun's default
        # alone otherwise, so single runs are untouched.
        port_args = (
            [f"--master_port={os.environ['MASTER_PORT']}"]
            if "MASTER_PORT" in os.environ
            else []
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={cfg.training.devices}",
            *port_args,
            os.path.abspath(__file__),
            *overrides,
            f"hydra.run.dir={run_dir}",
            "hydra.output_subdir=null",
        ]
        raise SystemExit(
            subprocess.run(
                cmd, cwd=os.path.dirname(os.path.abspath(__file__))
            ).returncode
        )

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

    # Validate dataset and embedder counts match
    assert len(cfg.data.datasets) == cfg.model.n_embedders, (
        f"The number of datasets ({len(cfg.data.datasets)}) must equal "
        f"the number of embedders ({cfg.model.n_embedders}). "
        f"Each entry in data.datasets gets its own BWEmbedder, keyed by name."
    )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.model.whisper_name, language="en", task="transcribe"
    )

    # ------------------------------------------------------------------
    # Datasets
    #
    # Each recipe carries the model-shape metadata (NUM_DAYS, CHANNEL_FILL)
    # and tags its samples with its registered name. The model keys its
    # embedders by that name (BrainWhisperer.embedders[name]), so the
    # position in the config list never decides which embedder handles a
    # dataset.
    # ------------------------------------------------------------------
    num_days_per_embedder = [RECIPES[name].NUM_DAYS for name in cfg.data.datasets]
    # Per-day calendar metadata (day-encoding positions / month buckets),
    # derived from the neuralshed session rules. Whether the embedder
    # uses it depends on its config (see BrainWhisperer.__init__).
    calendar_days_per_embedder = [
        RECIPES[name].calendar_days() for name in cfg.data.datasets
    ]
    # Per-dataset override of model.embedder
    embedder_cfg_per_embedder = [
        OmegaConf.merge(cfg.model.embedder, cfg.data.embedder_overrides.get(name, {}))
        for name in cfg.data.datasets
    ]
    # Fill-to-512 rule per dataset: applied at
    # collation time for the mixed training batches (before the training-step
    # augmentations mix channels) and in the model's eval-mode forwards for
    # the validation batches.
    channel_fills_per_embedder = [
        RECIPES[name].CHANNEL_FILL for name in cfg.data.datasets
    ]
    val_names = list(cfg.data.datasets)

    # Prepared independently per dataset, so the six builds overlap instead
    # of queueing (see build_in_parallel). DataLoaders are cheap and stay
    # sequential; order follows val_names so metrics keep their names.
    val_datasets = build_in_parallel(
        val_names,
        lambda name: RECIPES[name].build_val_dataset(
            max_label_len=cfg.data.max_seq_len
        ),
    )
    val_dataloaders = [
        DataLoader(
            val_datasets[name],
            batch_size=_per_rank_batch,
            collate_fn=SingleRecipeCollator(
                name, RECIPES[name], default_ragged_collator()
            ),
            num_workers=cfg.data.num_workers,
            shuffle=False,
        )
        for name in val_names
    ]

    # ------------------------------------------------------------------
    # Training DataLoader
    # ------------------------------------------------------------------
    # The six train recipes as one collection, sampled by the
    # neuralshed sampler: each global batch draws `days_per_batch`
    # (member, day) groups, split across the ranks. "continuous" seeding
    # keeps one RNG stream for the whole run, so a seed fixes every
    # epoch's batch order.
    train_collection = build_train_collection(
        list(cfg.data.datasets), max_label_len=cfg.data.max_seq_len
    )
    train_sampler = StratifiedBatchSampler(
        train_collection,
        group_by=("member", "day"),
        weights_by=("member",),
        batch_size=cfg.data.batch_size,  # GLOBAL; the sampler splits per rank
        days_per_batch=cfg.data.days_per_batch,
        seed=cfg.training.seed,
        seeding="continuous",
        num_replicas=_world_size,
        rank=_rank,
    )
    train_dl = DataLoader(
        train_collection,
        batch_sampler=train_sampler,
        collate_fn=MixedRecipeCollator(
            channel_fills=dict(zip(cfg.data.datasets, channel_fills_per_embedder)),
        ),
        num_workers=cfg.data.num_workers,
    )

    # ------------------------------------------------------------------
    # Model + Lightning wrapper
    # ------------------------------------------------------------------
    model = BrainWhisperer(
        cfg.model,
        list(cfg.data.datasets),
        num_days_per_embedder,
        calendar_days_per_embedder,
        embedder_cfg_per_embedder,
        channel_fills=channel_fills_per_embedder,
    )
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

    # PER-selected checkpoint, for cascaded CTC+LM decoding (LightBeam)
    if cfg.training.ctc_phone_coeff > 0.0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=f"val_per_{val_names[0]}",
                mode="min",
                save_top_k=1,
                filename="best-per-{epoch:04d}",
            )
        )

    if cfg.viz.enabled:
        held_out = next(iter(val_dataloaders[0]))
        callbacks.append(
            VizCallback(held_out, log_every_n_epochs=cfg.viz.log_every_n_epochs)
        )

    # ------------------------------------------------------------------
    # Logger: CSVLogger always; WandbLogger appended when cfg.wandb.enabled.
    # ------------------------------------------------------------------
    logger = [CSVLogger(save_dir=str(run_dir), name="logs")]
    if cfg.wandb.enabled:
        from lightning.pytorch.loggers import WandbLogger

        logger.append(
            WandbLogger(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity or None,
                group=cfg.wandb.group or None,
                tags=list(cfg.wandb.tags) or None,
                name=run_dir.name,
                save_dir=str(run_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=cfg.wandb.mode,
            )
        )

    # ------------------------------------------------------------------
    # Save the full config (rank 0 only; all ranks share run_dir)
    # ------------------------------------------------------------------
    if _rank == 0:
        OmegaConf.save(cfg, run_dir / "config.yaml")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        # Detects if single or multi-GPU
        devices=_world_size if _world_size > 1 else cfg.training.devices,
        strategy=DDPStrategy(find_unused_parameters=True)
        if _world_size > 1
        else "auto",
        # Using the custom batch_sampler forces a workaround; all ranks run
        # the full validation set, redundant but correct.
        use_distributed_sampler=False,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(lm, train_dataloaders=train_dl, val_dataloaders=val_dataloaders)


if __name__ == "__main__":
    main()
