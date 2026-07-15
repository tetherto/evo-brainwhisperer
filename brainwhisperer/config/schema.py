from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class BWEmbedderConfig:
    n_channels: int = MISSING       # number of electrode channels
    patch_size: int = MISSING       # time steps per patch
    d_model: int = MISSING          # embedding dimension
    n_heads: int = MISSING
    dropout: float = MISSING


@dataclass
class BWEncoderConfig:
    n_layers: int = MISSING
    n_heads: int = MISSING
    d_model: int = MISSING          # must match Whisper encoder input dim
    d_ff: int = MISSING
    dropout: float = MISSING


@dataclass
class BWLoRAConfig:
    r: int = MISSING                # LoRA rank
    alpha: int = MISSING            # LoRA scaling factor
    dropout: float = MISSING
    target_modules: List[str] = MISSING


@dataclass
class BWModelConfig:
    whisper_name: str = MISSING     # HuggingFace model id, e.g. openai/whisper-small
    n_embedders: int = MISSING      # must equal the number of subject groups in the dataset
    embedder: BWEmbedderConfig = field(default_factory=BWEmbedderConfig)
    encoder: BWEncoderConfig = field(default_factory=BWEncoderConfig)
    lora: BWLoRAConfig = field(default_factory=BWLoRAConfig)


@dataclass
class BWDataConfig:
    datasets: List[str] = MISSING   # registered dataset names, e.g. [nejm, kunz]
    batch_size: int = MISSING
    num_workers: int = MISSING
    max_seq_len: int = MISSING      # max token length for Whisper decoder


@dataclass
class BWTrainingConfig:
    max_epochs: int = MISSING
    lr: float = MISSING
    weight_decay: float = MISSING
    warmup_steps: int = MISSING
    gradient_clip_val: float = MISSING
    ctc_weight: float = MISSING     # weight applied to CTC loss in phase 2
    seed: int = MISSING


@dataclass
class BWConfig:
    model: BWModelConfig = field(default_factory=BWModelConfig)
    data: BWDataConfig = field(default_factory=BWDataConfig)
    training: BWTrainingConfig = field(default_factory=BWTrainingConfig)


_cs = ConfigStore.instance()
_cs.store(name="base_config", node=BWConfig)
