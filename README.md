# BrainWhisperer

**Paper:** [BrainWhisperer: Leveraging Whisper for Speech Decoding in Neuroprosthetics](#) *(link added at publication)*

---

## What this is

BrainWhisperer decodes continuous speech from microelectrode array recordings of the motor cortex, targeting individuals with severe motor impairments who cannot speak. Three barriers make this hard in practice: labelled neural data is scarce, neural signals drift across recording sessions (non-stationarity), and the external language models used in state-of-the-art decoders are too large for local, private inference.  
This work addresses all three by adapting Whisper (pretrained on roughly 680,000 hours of speech) directly to intracortical signals through a convolutional front-end, hierarchical low-rank session projections, windowed self-attention in phoneme-selective encoder layers, and a multi-task CTC + cross-entropy training objective. Subject-specific embedders allow cross-participant training within a single unified architecture.  
On the publicly available BrainGate Utah array datasets, BrainWhisperer achieves **8.5% WER** on the Card benchmark in end-to-end decoding — to the best of our knowledge the best result reported in this setting — and cross-dataset training improves performance without any participant-specific fine-tuning.

---

## Setup

**1. Install `neuralshed`**

BrainWhisperer reads its datasets through the [`neuralshed`](https://github.com/tetherto/neuralshed) library, which is not on PyPI yet. Clone it and install it into your environment:

```bash
git clone https://github.com/tetherto/neuralshed.git
pip install -e ./neuralshed
```

**2. Install BrainWhisperer**

```bash
git clone <repo-url>
cd evo-brainwhisperer
pip install -e .
```

This is enough to run `eval.py`. Add extras for more:

```bash
pip install -e ".[train]"   # train.py and the launcher
pip install -e ".[viz]"     # the plotting and animation code in brainwhisperer/viz
pip install -e ".[dev]"     # pytest
```

`requirements-paper.txt` lists the exact package versions used for the paper's runs.

**3. Get the data**

The datasets are not included. Download the six datasets (Card, Willett, Kunz T12 and T15, Wairagkar, Jude) by following the [`neuralshed` docs](https://github.com/tetherto/neuralshed), then point it at the folder that holds them:

```bash
export NEURALSHED_DATA_ROOT=/path/to/datasets
```

Some datasets are hosted on Dryad, which needs a personal token to download (`NEURALFETCH_DRYAD_TOKEN`, see the `neuralshed` docs).

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

`eval_lightbeam.py` takes the same two arguments and decodes Card from the CTC phoneme head with an n-gram and a large language model. Use the `best-per-epoch=XXXX.ckpt` checkpoint for it.

**Tests**

```bash
pytest
```

For how the code is organised, how data flows through it and where to make changes, see [ARCHITECTURE.md](ARCHITECTURE.md). [DESIGN.md](DESIGN.md) explains the less obvious design choices.

---

## Reproducing results

*(Commands keyed to paper tables and figures — added at publication.)*

---

## Citation

If you use this code, please cite:

```bibtex
@article{brainwhisperer2025,
  title   = {BrainWhisperer: Leveraging Whisper for Speech Decoding in Neuroprosthetics},
  author  = {},
  journal = {},
  year    = {2025},
}
```
