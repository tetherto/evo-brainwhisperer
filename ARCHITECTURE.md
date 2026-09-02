# Architecture

A map of BrainWhisperer for developers to find their way around this modular package: what data flows through the system, what each module does, where to find it, and how the pieces connect at runtime.

---

## Repo layout

```
evo-brainwhisperer/
├── train.py                 # Entry point: cfg → datasets → model → BWLightningModule → Trainer.fit
├── eval.py                  # Entry point: load checkpoint → run inference → report WER/CER/PER
├── eval_lightbeam.py        # Entry point: cascaded CTC+KenLM+Llama-3.2 LoRA decoding eval (Card only)
├── tests/                   # One test file per module below, roughly 1:1 (see "Reading order")
├── notebooks/               # Exploratory notebooks, not part of the package
├── outputs/                 # Hydra run outputs (checkpoints, logs, saved configs) [gitignored]
│
└── brainwhisperer/            # Main package. Dependencies flow downward the graph: viz/ and training/ can
    │                          # import from data/ and models/; data/ and models/ never import viz/.
    ├── config/
    │   ├── base.yaml          # Hydra root config: defaults [base_config, _self_, optional experiment]
    │   ├── schema.py          # Typed dataclasses (BWConfig and its sub-configs) registered to Hydra's ConfigStore
    │   ├── paths.py           # Env-var-backed path constants: the only place paths are read, no hard-coded strings in the code
    │   └── experiment/        # YAML overrides layered on base.yaml (format `experiment=<name>`)
    │    
    │
    ├── data/
    │   ├── batch.py              # NeuralBatch, the data/model contract
    │   ├── registry.py           # Register datasets to their classes
    │   ├── datasets/             # One self-contained class per dataset 
    │   ├── samplers.py           # StratifiedDayBatchSampler
    │   ├── collate.py            # NeuralCollator: List[dict] → NeuralBatch
    │   └── augmentations.py      # Data processing functions
    │
    ├── models/
    │   ├── positional_encoding.py    # PositionalEncoding (sinusoidal table, shared by embedder + encoder)
    │   ├── embedder.py               # BWEmbedder (raw signals → projected + tokenised + day-encoded)
    │   ├── encoder.py                # BWEncoder (embedded patches → BaseModelOutput)
    │   └── model.py                  # BrainWhisperer, the full top-level assembly: embedders + encoder + Whisper+LoRA
    │
    ├── training/
    │   ├── lightning_module.py       # BWLightningModule (two-phase training_step, validation_step)
    │   ├── losses.py                 # CTC loss function
    │   ├── callbacks.py              # Training-related callbacks
    │   └── launcher.py               # SLURM sbatch / torchrun submission, standalone-runnable
    │
    ├── utils/
    │   ├── metrics.py                # Computes word error rate, char error rate and phoneme error rate
    │   ├── text.py                   # Text-processing utils
    │   └── sessions.py               # Session-handling utils
    │
    └── viz/                          # Visualization scripts, used in notebooks and training
        ├── data_pipeline.py          # Dataset structure and batch anatomy diagrams
        ├── augmentations.py          # Before/after augmentation figures
        ├── embeddings.py             # Conv. pipeline, patch embeddings, day-projection PCA plots
        ├── encoder.py                # Encoder/attention/FFN block diagrams
        ├── training.py               # Loss an WER curves
        └── animations.py             # Manim animations
```


---

## Suggested reading order

The fastest way to understand the architecture is to follow execution order, not directory alphabetization:

1. **`brainwhisperer/data/batch.py`**: `NeuralBatch`. Every module's input/output contract is expressed in terms of this one dataclass; its fields represent what every downstream stage receives.
2. **`brainwhisperer/config/schema.py`** shows all the dataclasses and their what's configurable parameters
3. **`train.py`** is the composition root, showing the  path through the whole system:  config → datasets → dataloaders → model → Lightning wrapper → callbacks → `Trainer.fit`.
   See the walkthrough below.
4. **`brainwhisperer/models/model.py`** (`BrainWhisperer`) and **`training/lightning_module.py`** (`BWLightningModule`) shape how batches are used for loss computation.
5. After understanding call trace and contracts from step 1 to 4, any leaf module can be read standalone as each is self-contained.

The `tests/` directory is also useful to understand the codebase: `test_data_layer.py` / `test_data_numerical.py` /
`test_ddp_sampler.py` show `data/` implementations, `test_embedder_numerical.py` / `test_encoder_windowed_attention.py`
/ `test_model_forward.py` / `test_init_parity.py` show `models/` implementations, `test_losses.py` / `test_training_step.py` show `training/` implementations.

---

## Configuration system

The configuration system is based on Hydra typed dataclasses and composed in layers. Nothing is inferred or defaulted at runtime:

1. **`config/schema.py`** defines `BWConfig` and its sub-configs (`BWModelConfig`,
   `BWDataConfig`, `BWTrainingConfig`, `BWAugmentationConfig`, `BWVizConfig`, `BWWandbConfig`,
   `BWGenerationConfig`, …) and registers `BWConfig` with Hydra's `ConfigStore` under the name
   `base_config`.
2. **`config/base.yaml`** is the entry Hydra reads (`@hydra.main(config_path="brainwhisperer/config", config_name="base")`
   in `train.py`).Hydra reading order starts from the schema's defaults, then overrides with the values written directly in
   `base.yaml`, then optionally layers one file from `config/experiment/`.
3. **`config/experiment/*.yaml`** files are named overrides selected from the CLI. For instance, running `python train.py experiment=cross_dataset` layers `config/experiment/cross_dataset.yaml` on top of `base.yaml`. Each experiment file is intentionally partial: it states only what differs from the base, for the sake of readability.
4. Any Hydra dotted override also works ad-hoc on the CLI. Setting `training.devices=4`, without a named experiment file overrides the number of devices listed in the base configuration for that specific run.

---

## How a training run actually executes

The call trace through `train.py`, works as follow:

1. `@hydra.main` (`train.py:37`) resolves `cfg` from `base.yaml` + schema defaults + any `experiment=` / CLI overrides.
2. **Multi-GPU delegation** (`train.py:44`): if `cfg.training.devices > 1` and the process wasn't launched by `torchrun`/`torchelastic` already (`_needs_torchrun`, `train.py:28`),
   `main()` re-execs itself under `torch.distributed.run` and returns. The actual training only happens in these newly spawned children processes. 
3. **Dataset construction** (`train.py:109-134`): `cfg.data.datasets` (a list of registry names, e.g. `[card, willet, kunz_t12, kunz_t15]`) is resolved via `get_dataset()` to classes; since datasets are mapped to their embedders through their list position, `cfg.model.n_embedders` must equal `len(cfg.data.datasets)` (asserted at `train.py:88`). Nore that this needs to be enforced *manually*.
4. **DataLoaders**: train datasets are wrapped in one `ConcatDataset` and driven by `StratifiedDayBatchSampler` (`train.py:139-153`); each val dataset gets its own plain `DataLoader` (`train.py:122-134`), so validation metrics stay per-dataset (`val_wer_<name>` etc., not pooled).
5. **Model + Lightning wrapper** (`train.py:158-164`): `BrainWhisperer(cfg.model, ...)` is built once, then wrapped in `BWLightningModule(model, cfg.training, tokenizer, val_names)`.  
Everything past this point is standard `pytorch_lightning.Trainer`: data augmentation, the two-phase CE/CTC step, and loss computation are all wrapped in  `BWLightningModule.training_step` / `.validation_step`.
6. **Callbacks** (`train.py:172-202`): `ModelCheckpoint` on `val_wer_<val_names[0]>` is always executed; a second `ModelCheckpoint` on `val_per_<val_names[0]>` is only executed if `cfg.training.ctc_phone_coeff > 0` (storing the best model for LightBeam eval); `UnfreezeWhisperDecoder` is always executed, parameterized by `cfg.training.unfreeze_decoder_epoch`; `VizCallback` only executed if visualization is enabled  (`cfg.viz.enabled`).
7. Finally, **`trainer.fit(...)`** (`train.py:248`) starts training execution.

---

## Data flow

```
Raw pickle files
│
│  brainwhisperer/data/datasets/
│  (CardDataset / WilletDataset / KunzT12Dataset / KunzT15Dataset /
│   JudeDataset / WairagkarDataset)
│
▼
Per-sample dict
  {neural_features: Tensor (T, C),  sentence_label: str,
   sbj_idx: int,  day: Tensor,  seq_class_ids: Tensor (optional, phone-level
   CTC targets — present only for datasets that provide phoneme labels), ...}
│
│  brainwhisperer/data/samplers.py
│  StratifiedDayBatchSampler — selects which samples form each batch,
│  stratified by recording day and subject group
│
▼
List[dict]  (one batch)
│
│  brainwhisperer/data/collate.py
│  NeuralCollator — pads variable-length signals, tokenises labels,
│  permutes signals to channels-first; collates seq_class_ids into
│  phone_seq/phone_seq_len only if every sample in the batch has one,
│  else leaves both None
│
▼
NeuralBatch
  signals:        (B, C, T)   raw neural signals, channels-first
  labels:         (B, L)      Whisper token ids
  sbj_idx:        (B,)        per-sample BWEmbedder routing key
  day_idx:        (B,)        per-sample session encoding index
  lengths:        (B,)        valid signal length per sample
  phone_seq:      (B, S)|None CTC phoneme targets; 0 = blank
  phone_seq_len:  (B,)|None   actual phone sequence lengths
│
│  brainwhisperer/data/augmentations.py
│  transform_data() — applied inside training_step, not the DataLoader:
│    train: gain noise, white noise, offset noise, random walk,
│           time cut, gaussian smoothing, time masking
│    val:   gaussian smoothing only
│
▼
Augmented NeuralBatch [flowing into BrainWhisperer model]
```

---

## Module map

| Stage | Directory / file | Key class or function |
|---|---|---|
| Dataset construction | `data/datasets/card.py` | `CardDataset` (registered `"card"`) |
| | `data/datasets/willett.py` | `WilletDataset` (registered `"willet"` — registry key keeps the old spelling) |
| | `data/datasets/kunz.py` | `KunzDataset` base; `KunzT12Dataset`, `KunzT15Dataset` |
| | `data/datasets/jude.py` | `JudeDataset` (registered `"jude"`) |
| | `data/datasets/wairagkar.py` | `WairagkarDataset` (registered `"wairagkar"`) |
| | `data/datasets/base.py` | `BaseNeuralDataset` |
| Dataset registry | `data/registry.py` | `register_dataset`, `get_dataset` |
| Batch sampling strategy | `data/samplers.py` | `StratifiedDayBatchSampler` |
| Batch assembly | `data/collate.py` | `NeuralCollator` |
| Data/model contract | `data/batch.py` | `NeuralBatch` |
| Signal augmentation | `data/augmentations.py` | `transform_data` |
| Session metadata | `utils/sessions.py` | `SESSIONS_BY_DATASET`; `CARD_SESSIONS`, `WILLETT_SESSIONS`, `KUNZ_T12_SESSIONS`, `KUNZ_T15_SESSIONS`, `WAIRAGKAR_SESSIONS`, `JUDE_SESSIONS` |
| Filesystem paths | `config/paths.py` | `CARD_ROOT`, `KUNZ_ROOT`, `WILLETT_ROOT`, `RNN_ARGS`, `JUDE_PATH`, `WAIRAGKAR_ROOT`; plus lazily-required, eval-only `BRAINAUDIO_ROOT`, `LIGHTBEAM_KENLM_PATH`, `LIGHTBEAM_ADAPTER_PATH` |
| Model (embedding) | `models/embedder.py` | `BWEmbedder` |
| Model (positional encoding) | `models/positional_encoding.py` | `PositionalEncoding` |
| Model (encoder) | `models/encoder.py` | `BWEncoder` — returns `BaseModelOutput`, feeds a stock `WhisperForConditionalGeneration` via `encoder_outputs=` |
| Model (top-level assembly) | `models/model.py` | `BrainWhisperer` — embedder + encoder + Whisper+LoRA; also owns `char_head` (CTC) |
| Training loop | `training/lightning_module.py` | `BWLightningModule` — two-phase training step (CE, then CTC on the same noise draw) |
| Training loss | `training/losses.py` | `ctc_loss` |
| Training callbacks | `training/callbacks.py` | `UnfreezeWhisperDecoder`, `VizCallback`, `StopAtPeriodCriteria` |
| Training launch (SLURM/torchrun) | `training/launcher.py` | — |
| Metrics / text | `utils/metrics.py` | `word_error_rate`, `char_error_rate`, `phoneme_error_rate` |
| | `utils/text.py` | `build_normalizer`, `remove_punctuation` |
| Visualization (pure, `Figure`-returning) | `viz/data_pipeline.py`, `viz/augmentations.py`, `viz/embeddings.py`, `viz/encoder.py`, `viz/training.py`, `viz/animations.py` | see one-liners in Repo layout above |
| Entry points | `train.py`, `eval.py` | Standard training / WER-CER-PER eval |
| | `eval_lightbeam.py` | Cascaded CTC + KenLM + Llama-3.2 LoRA (LightBeam) decoding eval, Card dataset only — bypasses the BrainWhisperer decoder entirely |

---

## What goes where?

- **A new dataset**: class definition goes in `data/datasets/<name>.py`, subclassing `BaseNeuralDataset`, and needs to be decorated with `@register_dataset("<name>")`.  
Its session list is added to `utils/sessions.py` (`SESSIONS_BY_DATASET["<name>"]`) and its root env var to `config/paths.py`.  
To use the new class, its reference `"<name>"` is added in `cfg.data.datasets` (and the total `cfg.model.n_embedders` needs to match the number of dataset names in config).
- **A new augmentation**: the transform function needs to be added to `data/augmentations.py` and wired into `transform_data()`'s `train`/`val` branches.  
Augmentations should be pure tensor-in/tensor-out functions, and their call should stay within `BWLightningModule.training_step`/`validation_step`, not within `DataLoader`.
- **A new loss**: loss functions are in `training/losses.py`. They should be implemented as pure functions taking tensors and returning a scalar, following `ctc_loss`'s shape. They are caled from `BWLightningModule`.
- **A new callback**: callbacks are added in `training/callbacks.py`, then appended to the `callbacks = [...]` list in `train.py` (`train.py:172-202`).  
Callbacka should be gated behind a new `cfg.training.*` / `cfg.viz.*` flag rather than left always-on.
- **A new config field**: fields should be added to their relevant dataclass in `config/schema.py` first (typed, with a default); only then referenced from `base.yaml` or an experiment override.  
Nothing should be read from `cfg` that isn't declared in the schema.
- **A new experiment variant**: experiments are declared with new YAML in `config/experiment/`, using `# @package _global_` at the top. Comments stating only the fields that differ from `base.yaml` and the reason why those have been changed are good practice. 
- **A new viz plot**: visualization scripts should include pure functions producing figures (`(tensors/arrays) -> Figure`), with no logging or file I/O inside it.  
These should be wired for training-time use only through `VizCallback` in `training/callbacks.py` for the sake of simplicity.
