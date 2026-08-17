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
    dropout: float = MISSING  # patch-mode embedding dropout
    day_layer_dropout: float = MISSING  # dropout after the per-day projection
    session_dropout: float = MISSING  # dropout on the session encoding
    ref_half_layout: bool = False  # DEBUG: A/B flag: True → session enc on LAST half (REF convention); False → first half (CANDIDATE)


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
    max_source_pos: int = (
        MISSING  # Whisper encoder max positions; embedder output length
    )
    temporal_pe_std: float = MISSING
    attn_window_size: int = MISSING  # tokens each position can attend to left+right
    rope: bool = MISSING  # T5-style relative position bias (unused in logged runs)
    ref_half_layout: bool = False  # DEBUG: A/B flag: True → temporal PE on FIRST half (REF convention); False → last half (CANDIDATE)


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
    batch_size: int = MISSING  # global batch size across all GPUs; per-rank = batch_size // world_size
    num_workers: int = MISSING
    max_seq_len: int = MISSING  # max token length for Whisper decoder
    days_per_batch: int = MISSING  # distinct recording days per batch; per-rank batch_size must be divisible by this


@dataclass
class BWAugmentationConfig:
    gain_std: float = MISSING  # static cross-channel gain warp; 0.0 disables
    noise_std: float = MISSING  # white noise
    offset_std: float = MISSING  # per-channel constant offset
    walk_std: float = MISSING  # random walk along time; 0.0 disables
    max_cut: int = MISSING  # max frames cut from signal start
    smooth_kernel_size: int = MISSING
    smooth_sigma: float = MISSING
    num_masks: int = MISSING  # SpecAugment time masks per sample
    max_mask_pct: float = MISSING  # max mask width as fraction of valid length


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
    matmul_precision: str = MISSING
    augmentation: BWAugmentationConfig = field(default_factory=BWAugmentationConfig)


@dataclass
class BWVizConfig:
    enabled: bool = False
    log_every_n_epochs: int = 5


@dataclass
class BWGenerationConfig:
    """
    Beam search kwargs passed to BrainWhisperer.generate() at eval time.
    Training validation uses greedy decoding (no kwargs) for speed; eval.py
    uses this config to reproduce reported numbers.
    """

    num_beams: int = MISSING
    num_beam_groups: int = MISSING  # diversity-promoting Group Beam Search
    length_penalty: float = MISSING
    repetition_penalty: float = MISSING
    no_repeat_ngram_size: int = MISSING
    diversity_penalty: float = MISSING


@dataclass
class BWConfig:
    model: BWModelConfig = field(default_factory=BWModelConfig)
    data: BWDataConfig = field(default_factory=BWDataConfig)
    training: BWTrainingConfig = field(default_factory=BWTrainingConfig)
    generation: BWGenerationConfig = field(default_factory=BWGenerationConfig)
    viz: BWVizConfig = field(default_factory=BWVizConfig)


_cs = ConfigStore.instance()
_cs.store(name="base_config", node=BWConfig)
