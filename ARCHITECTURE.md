# Architecture

This page explains how BrainWhisperer is organised: what each part of the code does, how data moves through it, and where to make common changes.

## Key terms

- **Recipe**: the code that turns one public dataset into training and validation samples. There is one recipe per dataset (`card`, `willet`, `kunz_t12`, `kunz_t15`, `wairagkar`, `jude`).
- **`NeuralBatch`**: the batch format the model reads. Every data path ends in this one class.
- **Embedder**: the first block of the model. Each dataset has its own embedder, which turns that dataset's neural signals into tokens that Whisper's encoder can read.
- **Day**: a recording session (or block). The embedders learn small per-day corrections, because neural signals drift from one session to the next.

## Repository layout

```
evo-brainwhisperer/
├── train.py               # Train a model
├── eval.py                # Score a checkpoint: WER, CER and PER per dataset
├── eval_lightbeam.py      # Score a checkpoint with CTC beam search + language models (Card only)
├── tests/                 # Unit and regression tests
├── notebooks/             # Exploration notebooks (not part of the package)
├── outputs/               # Run folders: checkpoints, logs, saved config (not tracked by git)
└── brainwhisperer/
    ├── config/
    │   ├── schema.py          # Every config option, as typed dataclasses with defaults
    │   ├── base.yaml          # Default values used by train.py
    │   ├── experiment/        # Named overrides, e.g. the full six-dataset run
    │   └── paths.py           # Paths read from environment variables (LightBeam only)
    ├── data/
    │   ├── recipes/           # One recipe per dataset, plus the shared base class
    │   ├── rescale/           # Per-dataset normalisation used inside the recipes
    │   ├── splits/            # Fixed lists of held-out trials (JSON) for Wairagkar and Jude
    │   ├── fields.py          # Names of the fields in each sample (neural, whisper, phonemes, day)
    │   ├── collection.py      # Joins the six training sets into one dataset for the sampler
    │   ├── collate.py         # Turns a list of samples into a padded batch
    │   ├── adapter.py         # Converts that padded batch into a NeuralBatch
    │   ├── batch.py           # NeuralBatch
    │   └── augmentations.py   # Noise, time cuts, smoothing and masking, applied during training
    ├── models/
    │   ├── embedder.py            # BWEmbedder: neural signals -> tokens, with per-day corrections
    │   ├── encoder.py             # BWEncoder: Whisper's encoder with windowed attention
    │   ├── positional_encoding.py # Sinusoidal position table shared by embedder and encoder
    │   └── model.py               # BrainWhisperer: embedders + encoder + Whisper decoder (LoRA)
    ├── training/
    │   ├── lightning_module.py    # BWLightningModule: training and validation steps
    │   ├── losses.py              # CTC loss
    │   ├── callbacks.py           # Decoder unfreezing, training figures
    │   └── launcher.py            # Start runs with torchrun, SLURM, or several seeds in a row
    ├── prediction.py          # One batch in, text and phoneme predictions out (used by eval.py)
    ├── utils/
    │   ├── fill.py            # Widens narrow datasets to the model's 512 input channels
    │   ├── metrics.py         # WER, CER and PER
    │   └── text.py            # Text normalisation (english.json holds the spelling map)
    └── viz/                   # Figures for notebooks and training logs
```

Dependencies only point one way: `config/` depends on nothing, `models/` reads `config/` and `data/batch.py`, and `training/` and `viz/` may use anything else. `data/` never imports from `models/`.

## How the pieces fit

```
tether_scratch datasets on disk
        │
        ▼
recipes (data/recipes/)        one per dataset: select trials, normalise, cut into samples
        │
        ▼
collation (data/collate.py)    pad samples to equal length, widen to 512 channels
        │
        ▼
NeuralBatch (data/batch.py)
        │
        ▼
BrainWhisperer (models/)       embedder per dataset → encoder → Whisper decoder + CTC head
        │
        ▼
text and phoneme predictions → WER / CER / PER
```

The datasets themselves are downloaded and read by the `tether_scratch` library, which this package depends on. Everything that is specific to BrainWhisperer (which trials, which channels, how to normalise) lives in the recipes.

## Configuration

Configuration uses [Hydra](https://hydra.cc) with typed dataclasses.

1. `config/schema.py` defines every option and its default (`BWConfig` and its parts: model, data, training, generation, viz, wandb).
2. `config/base.yaml` sets the values `train.py` starts from. By default it trains on Card only.
3. `config/experiment/*.yaml` files hold named overrides. Each one lists only what differs from the base. `python train.py experiment=cross_dataset_full` trains the full six-dataset model from the paper; `cross_dataset_full_wandb` does the same and logs to Weights & Biases.
4. Any single option can be overridden on the command line, for example `python train.py training.devices=4`.

Every option must be declared in `schema.py`; the code never reads a key that the schema does not define. At the end of setup, `train.py` saves the full resolved config to `<run_dir>/config.yaml`. The evaluation scripts rebuild the model from that file.

## A training run, step by step

`train.py` is the best place to start reading: it connects every part of the system.

1. **Read the config.** Hydra merges the schema defaults, `base.yaml`, an optional experiment file and any command-line overrides.
2. **Start more processes for multi-GPU runs.** If `training.devices > 1` and the script is not already running under `torchrun`, it restarts itself with `torch.distributed.run`, one process per GPU, and the first process just waits.
3. **Ask each recipe for model settings.** For every name in `data.datasets`, the recipe in `RECIPES` gives the number of recording days (`NUM_DAYS`), the calendar date of each day (`calendar_days()`), and how to widen its channels to 512 (`CHANNEL_FILL`). `model.n_embedders` must equal the number of datasets.
4. **Build the validation loaders.** Each dataset gets its own loader, so validation scores are reported per dataset (`val_wer_card`, `val_wer_willet`, …). The six datasets are built in parallel.
5. **Build the training loader.** The six training sets are joined into one dataset (`build_train_collection`). A sampler from `tether_scratch` (`StratifiedBatchSampler`) fills each batch with trials from a fixed number of (dataset, day) groups (`data.days_per_batch`) and splits it across GPUs. `MixedRecipeCollator` pads the trials and widens each one to 512 channels.
6. **Build the model.** `BrainWhisperer` is created with one embedder per dataset and wrapped in `BWLightningModule`, which holds the training logic.
7. **Add callbacks and loggers.** A checkpoint is saved for the best WER on the first dataset (`best-epoch=….ckpt`), and, when the CTC loss is on, another for the best PER (`best-per-epoch=….ckpt`, used by LightBeam). `UnfreezeWhisperDecoder` unfreezes the Whisper decoder at a set epoch. Training figures are added when `viz.enabled` is true. Metrics go to a CSV log, and to Weights & Biases when `wandb.enabled` is true.
8. **Save the config and train.** The resolved config is written to `config.yaml`, then `trainer.fit` runs.

Each training step (`BWLightningModule.training_step`) has two parts that share the same noise:

- **Part 1, cross-entropy:** the noisy signal is smoothed and passed through the model with dropout off; the Whisper decoder predicts the sentence.
- **Part 2, CTC:** the same noisy signal is also cut in time, smoothed and masked; the CTC head predicts the phoneme sequence.

The loss is `ce_coeff × cross-entropy + ctc_phone_coeff × CTC`.

## The data path in detail

**Recipes.** Each recipe is a subclass of `Recipe` (`data/recipes/base.py`) and is registered in `RECIPES` (`data/recipes/__init__.py`). The base class runs the same pipeline for every dataset:

1. `build_chain(split)`: load the study, apply the dataset's normalisation, and mark trials as `train` or `val`.
2. A segmenter cuts one sample per trial with four fields: the neural signal, the Whisper token ids of the sentence, the phoneme ids, and the day index.
3. `postprocess_train` / `postprocess_val`: dataset-specific steps such as dropping trials that are too long or fixing the training order.

Each recipe fills in only what is specific to its dataset: which trials, which channels, and how the day index is defined (`day_rule()`, built from `tether_scratch` session rules). `build_val_dataset` and `build_train_dataset` return the finished datasets.

**Normalisation.** Card's data ships already normalised. Willet and Kunz T12 use the library's standard per-block rescaling. Kunz T15, Wairagkar and Jude need statistics that the library cannot compute directly, so `data/rescale/` computes them while the dataset is built and stores them with each recording.

**Held-out trials.** Card, Willet and both Kunz datasets use the validation split shipped with the data. Wairagkar and Jude had random splits in the published runs, so the chosen trials are stored in `data/splits/*.json` and read back by `held_out_split`.

**Collation.** Samples have different lengths and channel counts. Training uses `MixedRecipeCollator`, which first widens each sample to 512 channels (`utils/fill.py`: copy the channels twice, or add zero channels) and then pads in time. Validation uses `SingleRecipeCollator`, one dataset at a time; there the model widens the channels itself during evaluation. Both end in `adapter.py`, which builds the `NeuralBatch`.

**`NeuralBatch` fields:**

| Field | Shape | Meaning |
|---|---|---|
| `signals` | (B, C, T) | neural signals, channels first |
| `labels` | (B, L) | Whisper token ids of the sentence (padding is -100) |
| `datasets` | (B,) | dataset name of each row; picks its embedder |
| `day_idx` | (B,) | recording-day index of each row |
| `lengths` | (B,) | number of valid time steps |
| `phone_seq` | (B, S) or None | phoneme targets for CTC (0 is the blank) |
| `phone_seq_len` | (B,) or None | number of valid phonemes |

**Augmentation.** Noise, time cuts, smoothing and masking (`data/augmentations.py`) run inside the training step, not in the data loader, so the two parts of a step can share one noise sample. Validation only smooths.

## The model

`BrainWhisperer` (`models/model.py`) has three parts:

1. **Embedders** (`BWEmbedder`), one per dataset, stored by name. All embedders share their convolution weights. Each adds a learned per-day correction and a day encoding, so the model can adapt to drift between sessions.
2. **Encoder** (`BWEncoder`), built from Whisper's encoder layers. The first layers use windowed attention; the output of the last of those layers feeds the CTC phoneme head.
3. **Whisper decoder** with LoRA adapters, which writes the sentence.

`model.load_checkpoint(path)` loads the weights from a checkpoint saved by `train.py`.

## Evaluation

- **`eval.py`** takes a checkpoint and the run's `config.yaml`, rebuilds the model and the validation sets from the recipes, decodes with beam search (settings in `config.generation`) and prints WER, CER and PER for each dataset. `--set key=value` overrides config values without editing the saved file.
- **`eval_lightbeam.py`** decodes Card from the CTC phoneme head instead of the Whisper decoder: a beam search with a word-level KenLM model, rescored by a fine-tuned Llama-3.2-1B. It needs the environment variables listed at the top of the script.

Both scripts use `prediction.py` and the same text normalisation (`utils/text.py`) before scoring.

## Where to make changes

- **Add a dataset.** Write a `Recipe` subclass in `data/recipes/<name>.py` that implements `build_chain`, `neural_extractor` and `day_rule`, and sets `CHANNEL_FILL` if it has fewer than 512 channels. Register it in `RECIPES`. If its validation split is not shipped with the data, store the held-out trials in `data/splits/`. Then add the name to `data.datasets` and update `model.n_embedders`.
- **Add an augmentation.** Write a function that takes and returns tensors in `data/augmentations.py`, and call it from `transform_data()` (or `apply_noise()` for noise shared by both parts of a step). Keep augmentations inside the training step, not in the data loader.
- **Add a loss.** Write it as a function in `training/losses.py`, following `ctc_loss`, and call it from `BWLightningModule`.
- **Add a callback.** Write it in `training/callbacks.py`, add it to the `callbacks` list in `train.py`, and turn it on with a config flag.
- **Add a config option.** Declare it, with a type and a default, in the right dataclass in `config/schema.py`. Only then set it in `base.yaml` or an experiment file.
- **Add an experiment.** Create `config/experiment/<name>.yaml` starting with `# @package _global_`, list only the values that change, and add a short comment on why.
- **Add a figure.** Write a function that takes tensors or arrays and returns a matplotlib `Figure`, with no file writing or logging inside. To show it during training, call it from `VizCallback`.

## Tests

Run `pytest` from the repository root. `test_model_forward.py` and `test_training_step.py` read Card trials through its recipe, so they need the study data (`TETHER_DATA_ROOT`) and download `whisper-tiny.en` on first run. The other tests use synthetic data or skip what they cannot check.

| Area | Tests |
|---|---|
| Data | `test_held_out_splits.py`, `test_rescale_transforms.py`, `test_jude_rescale.py`, `test_fill.py`, `test_mixed_collator.py`, `test_data_numerical.py` |
| Model | `test_model_forward.py`, `test_encoder_windowed_attention.py`, `test_init_parity.py`, `test_session_calendar_mapping.py` |
| Training | `test_losses.py`, `test_training_step.py` |
| Evaluation | `test_prediction.py` |
| Config | `test_config_package.py` |
