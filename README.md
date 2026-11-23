# BrainWhisperer

**Paper:** [BrainWhisperer: Leveraging Large-Scale ASR Models for Neural Speech Decoding](https://arxiv.org/abs/2603.13321)

---

## What this is

BrainWhisperer decodes continuous speech from microelectrode array recordings of the motor cortex, targeting individuals with severe motor impairments who cannot speak. Three barriers make this hard in practice: labelled neural data is scarce, neural signals drift across recording sessions (non-stationarity), and the external language models used in state-of-the-art decoders are too large for local, private inference.  
This work addresses all three by adapting Whisper (pretrained on roughly 680,000 hours of speech) directly to intracortical signals through a convolutional front-end, hierarchical low-rank session projections, windowed self-attention in phoneme-selective encoder layers, and a multi-task CTC + cross-entropy training objective. Subject-specific embedders allow cross-participant training within a single unified architecture.  
On the publicly available BrainGate Utah array datasets, BrainWhisperer reaches **8.7% WER** on the Brain-to-Text '25 benchmark (Card et al. 2024) in end-to-end decoding, with no external language model. When the paper appeared, this was to the best of our knowledge the best end-to-end result reported on this benchmark. Cross-dataset training improves performance without any participant-specific fine-tuning; the six-dataset model trained with this repository scores 8.6%.

---

## Setup

**1. Install `neuralshed`**

BrainWhisperer reads its datasets through the [`neuralshed`](https://github.com/tetherto/neuralshed) library, which is not on PyPI yet. Clone the version BrainWhisperer is tested with and install it into a Python 3.12 environment:

```bash
git clone --branch v0.1.0 https://github.com/tetherto/neuralshed.git
pip install -e ./neuralshed
```

**2. Install BrainWhisperer**

```bash
git clone https://github.com/tetherto/evo-brainwhisperer.git
cd evo-brainwhisperer
pip install -e .
```

This is enough to run `eval.py`. Add extras for more:

```bash
pip install -e ".[train]"   # train.py and the launcher
pip install -e ".[viz]"     # the plotting and animation code in brainwhisperer/viz
pip install -e ".[dev]"     # pytest
```

`requirements.txt` pins the tested version of each dependency, for Linux (CUDA 12.6 builds of PyTorch) and macOS. To install them instead of the latest allowed versions, run `pip install -r requirements.txt` before `pip install -e .`.

Two things can break the install:

- **uv.** By default, uv takes `torch` only from the PyTorch index, which has no macOS builds. Add `--index-strategy unsafe-best-match` so it also checks PyPI: `uv pip install -r requirements.txt --index-strategy unsafe-best-match`.
- **The `viz` extra** (also in `requirements.txt`) builds `manim` from source, which needs the Cairo and Pango libraries. Install them first: `brew install cairo pango pkg-config` on macOS, or `apt install libcairo2-dev libpango1.0-dev pkg-config` on Debian/Ubuntu.

**3. Get the data**

The six datasets (Card, Willett, Kunz T12 and T15, Wairagkar, Jude) are not included. All six are hosted on Dryad, which needs a personal token to download. Set the token and the folder the datasets go in:

```bash
export NEURALFETCH_DRYAD_TOKEN=<your token>
export NEURALSHED_DATA_ROOT=/path/to/datasets
```

Then download them (about 120 GB unpacked; this might take hours):

```bash
python -c "from brainwhisperer.data.recipes import RECIPES; [RECIPES[n].build_chain('val').steps['study'].download() for n in ['card', 'willet', 'kunz_t12', 'wairagkar', 'jude']]"
```

Kunz T15 shares Kunz T12's files. The Wairagkar download includes a 26 GB folder BrainWhisperer does not use, `wairagkar_2025/Wairagkar2025Brain2Voice/t15_neural_data/`, which you can delete.

**4. Optional environment variables**

| Variable | Needed for |
|---|---|
| `WANDB_API_KEY` | logging to Weights & Biases (`wandb.enabled=true`) |
| `MASTER_PORT` | running more than one multi-GPU job on the same machine |
| `BRAINAUDIO_ROOT`, `LIGHTBEAM_KENLM_PATH`, `LIGHTBEAM_ADAPTER_PATH`, `HF_TOKEN` | `eval_lightbeam.py` only; see the top of that script |

---

## Usage

**Train**

```bash
python train.py                                   # default config: Card only, one GPU
python train.py experiment=cross_dataset_full     # the full six-dataset model from the paper (2 GPUs)
python train.py training.devices=4 training.seed=3  # change any config value on the command line
```

Each run writes its checkpoints, logs and full `config.yaml` to a folder under `outputs/`.

**Launch several seeds or a SLURM job**

```bash
python -m brainwhisperer.training.launcher --seeds 1,2,3 experiment=cross_dataset_full
python -m brainwhisperer.training.launcher --slurm training.devices=4
python -m brainwhisperer.training.launcher --slurm --dry-run training.devices=4  # print the job script only
```

**Evaluate**

```bash
python eval.py --checkpoint <run_dir>/checkpoints/best-epoch=XXXX.ckpt --config <run_dir>/config.yaml
```

This prints WER, CER and PER for every dataset in the run. Add `--set key=value` to change config values for the evaluation only, for example `--set data.batch_size=32`.

To share a model, convert its checkpoint into a folder holding only the weights and the settings needed to run it. Run this where the training data is:

```bash
python convert_checkpoint.py --checkpoint <run_dir>/checkpoints/best-epoch=XXXX.ckpt --config <run_dir>/config.yaml --out <folder>
```

Then evaluate from the folder (or, once published, from its Hugging Face name). Only the datasets you score need to be downloaded:

```bash
python eval.py --model <folder> --datasets card
```

It runs on the first available of CUDA, Apple's GPU (MPS) and CPU; choose one with `--device`. On a Mac, if PyTorch reports an operation that MPS does not support, set `PYTORCH_ENABLE_MPS_FALLBACK=1` to run it on the CPU.

`eval_lightbeam.py` takes the same two arguments and decodes Card from the CTC phoneme head with an n-gram and a large language model. Use the `best-per-epoch=XXXX.ckpt` checkpoint for it.

**Tests**

```bash
pytest
```

For how the code is organised, how data flows through it and where to make changes, see [ARCHITECTURE.md](ARCHITECTURE.md). [DESIGN.md](DESIGN.md) explains the less obvious design choices.

---

## Reproducing results

These steps reproduce the six-dataset model released on Hugging Face and the numbers on its [model card](MODEL_CARD.md): validation WER on every dataset, and the Brain-to-Text '25 score on Kaggle. The paper's Table 1 reports earlier models (Card only, and Card + Willett + Kunz) and a 5-gram decoder, which these steps do not cover.

**1. Train eight seeds** of the six-dataset model. The seeds run one after another, each on 2 GPUs, and each writes its own run folder under `outputs/`:

```bash
python -m brainwhisperer.training.launcher --seeds 1,2,3,4,5,6,7,8 experiment=cross_dataset_full
```

**2. Score each seed on the validation sentences.** This prints WER, CER and PER per dataset, the numbers behind the model card's "Per-dataset results" table (our logs are in `evals/20261023_seeds1-8/`):

```bash
python eval.py --checkpoint <run_dir>/checkpoints/best-epoch=XXXX.ckpt --config <run_dir>/config.yaml
```

**3. Score each seed on the benchmark test sets.** Their labels are withheld, so `--competition` writes submission files instead of scores: `card_kaggle.csv` for [Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25) on Kaggle, and `willet_evalai.txt` for the Willett benchmark on EvalAI. Upload `card_kaggle.csv` as a late submission on Kaggle to get the private and public scores (our files are in `evals/20261027_competition/`):

```bash
python eval.py --checkpoint <run_dir>/checkpoints/best-epoch=XXXX.ckpt --config <run_dir>/config.yaml --competition --out evals/<date>_competition/seed<N>
```

**4. Check the released model.** We released seed 6, the seed with the lowest Card validation WER. Scoring it from Hugging Face matches step 2 for that seed to one decimal place (on Card we get 8.80% WER, against 8.77% in our log; beam search can differ slightly with batch size and library versions):

```bash
python eval.py --model qvac/BrainWhisperer
```

Results vary between seeds: on Kaggle's private leaderboard, the eight seeds score from 8.5% to 9.6% WER (mean 9.1%).

---

## Citation

If you use this code, please cite:

```bibtex
@misc{boccato2026brainwhisperer,
  title         = {BrainWhisperer: Leveraging Large-Scale ASR Models for Neural Speech Decoding},
  author        = {Boccato, Tommaso and Olak, Michal and Ferrante, Matteo},
  year          = {2026},
  eprint        = {2603.13321},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2603.13321},
}
```

---

## License

The code and model weights are released by Tether Data, S.A. de C.V. under the [Apache 2.0 license](LICENSE). The model starts from [whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en), also Apache 2.0. The datasets it is trained on are not included; their authors release them on Dryad under CC0-1.0.
