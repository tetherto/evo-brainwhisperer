from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BWEmbedderConfig:
    num_features: int = (
        MISSING  # electrode channels (512 for CARD/Willet, padded for Kunz)
    )
    d_model: int = MISSING  # embedding dimension; must match Whisper decoder d_model
    embed_type: str = MISSING  # "conv" or "patch"
    kernel_size_1: int = MISSING  # conv1 kernel size (must be odd); conv mode only
    stride_2: int = MISSING  # conv2 stride; conv mode only
    patch_size: int = MISSING  # time steps per patch; patch mode only
    encodings: str = MISSING  # day encoding type: "sin" or "learnable"
    day_projections: bool = MISSING  # whether to apply per-day linear projection
    day_proj_rank: int = (
        MISSING  # low-rank delta rank (0 = full d_model×d_model per day)
    )
    dropout: float = MISSING


@dataclass
class BWEncoderConfig:
    n_layers: int = MISSING
    n_heads: int = MISSING
    d_model: int = MISSING  # must match BWEmbedderConfig.d_model
    d_ff: int = MISSING
    dropout: float = MISSING
    last_phoneme_layer: int = (
        MISSING  # layers 0..last_phoneme_layer-1 use windowed attention
    )
    attn_window_size: int = MISSING  # tokens each position can attend to left+right
    rope: bool = MISSING  # T5-style relative position bias (unused in logged runs)


@dataclass
class BWLoRAConfig:
    r: int = MISSING  # LoRA rank
    alpha: int = MISSING  # LoRA scaling factor
    dropout: float = MISSING
    target_modules: list[str] = MISSING


@dataclass
class BWModelConfig:
    whisper_name: str = MISSING  # HuggingFace model id, e.g. openai/whisper-small
    n_embedders: int = MISSING  # must equal the number of subject groups in the dataset
    whisper_dropout: float = 0.0  # 0.0 = pretrained default
    embedder: BWEmbedderConfig = field(default_factory=BWEmbedderConfig)
    encoder: BWEncoderConfig = field(default_factory=BWEncoderConfig)
    lora: BWLoRAConfig = field(default_factory=BWLoRAConfig)


@dataclass
class BWDataConfig:
    datasets: list[str] = MISSING  # registered dataset names, e.g. [card, kunz_t12]
    batch_size: int = MISSING
    num_workers: int = MISSING
    max_seq_len: int = MISSING  # max token length for Whisper decoder
    days_per_batch: int = MISSING  # distinct recording days per batch; batch_size must be divisible by this


@dataclass
class BWTrainingConfig:
    max_epochs: int = MISSING
    lr: float = MISSING
    lr_min: float = MISSING
    weight_decay: float = MISSING
    gradient_clip_val: float = MISSING
    ce_coeff: float = MISSING
    ctc_phone_coeff: float = MISSING
    ctc_char_coeff: float = MISSING
    scheduler: str = MISSING  # "cosine" | "linear" | "sequential"
    seed: int = MISSING
    unfreeze_decoder_epoch: int = (
        -1
    )  # epoch at which to unfreeze Whisper decoder; -1 = never
    devices: int = 1  # GPUs per node; override to >1 only when launching via torchrun


@dataclass
class BWVizConfig:
    enabled: bool = False
    log_every_n_epochs: int = 5


@dataclass
class BWConfig:
    model: BWModelConfig = field(default_factory=BWModelConfig)
    data: BWDataConfig = field(default_factory=BWDataConfig)
    training: BWTrainingConfig = field(default_factory=BWTrainingConfig)
    viz: BWVizConfig = field(default_factory=BWVizConfig)


_cs = ConfigStore.instance()
_cs.store(name="base_config", node=BWConfig)
