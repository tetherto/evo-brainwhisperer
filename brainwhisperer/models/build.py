"""
Utils function turning the model's config into constructor arguments and inference settings to reproduce paper's results.
(HuggingFace compatibility)
"""

from omegaconf import DictConfig, OmegaConf

from brainwhisperer.data.recipes import RECIPES


def model_kwargs(cfg: DictConfig) -> dict:
    """BrainWhisperer constructor arguments for the run's datasets."""
    names = list(cfg.data.datasets)
    return {
        "cfg": cfg.model,
        "embedder_names": names,
        "num_days_per_embedder": [RECIPES[name].NUM_DAYS for name in names],
        # Per-day calendar metadata (day-encoding positions / month buckets)
        "calendar_days_per_embedder": [RECIPES[name].calendar_days() for name in names],
        # Per-dataset override of model.embedder
        "embedder_cfg_per_embedder": [
            OmegaConf.merge(
                cfg.model.embedder, cfg.data.embedder_overrides.get(name, {})
            )
            for name in names
        ],
        # Fill-to-512 rule per dataset: applied at collation time for the
        # mixed training batches (before the training-step augmentations mix
        # channels) and in the model's eval-mode forwards for the validation
        # batches.
        "channel_fills": [RECIPES[name].CHANNEL_FILL for name in names],
    }


def inference_settings(cfg: DictConfig) -> dict:
    """The settings eval.py scores a model with, as plain JSON values."""
    aug = cfg.training.augmentation
    return {
        # Diverse beam search kwargs, forwarded to BrainWhisperer.generate().
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        # Val-mode preprocessing (gaussian smoothing only)
        "smoothing": {
            "smooth_kernel_size": aug.smooth_kernel_size,
            "smooth_sigma": aug.smooth_sigma,
        },
        # Whisper label length the validation sets are cut to.
        "max_label_len": cfg.data.max_seq_len,
        # Whether the model has a trained phoneme head to report PER from.
        "measure_per": cfg.training.ctc_phone_coeff > 0.0,
    }
