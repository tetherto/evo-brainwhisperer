# BrainWhisperer

**Paper:** [BrainWhisperer: Leveraging Whisper for Speech Decoding in Neuroprosthetics](#) *(link added at publication)*

---

## What this is

BrainWhisperer decodes continuous speech from microelectrode array recordings of the motor cortex, targeting individuals with severe motor impairments who cannot speak. Three barriers make this hard in practice: labelled neural data is scarce, neural signals drift across recording sessions (non-stationarity), and the external language models used in state-of-the-art decoders are too large for local, private inference. This work addresses all three by adapting Whisper — pretrained on roughly 680,000 hours of speech — directly to intracortical signals through a convolutional front-end, hierarchical low-rank session projections, windowed self-attention in phoneme-selective encoder layers, and a multi-task CTC + cross-entropy training objective. Subject-specific embedders allow cross-participant training within a single unified architecture. On the publicly available BrainGate Utah array datasets, BrainWhisperer achieves **8.5% WER** on the Card benchmark in end-to-end decoding — to the best of our knowledge the best result reported in this setting — and cross-dataset training improves performance without any participant-specific fine-tuning.

---

## Setup

**1. Clone and install**

```bash
git clone <repo-url>
cd evo-brainwhisperer
pip install -e .
```

**2. Set data paths**

All dataset paths are read from environment variables — no paths are hardcoded. Add these to your shell profile or job script:

```bash
export NEJM_ROOT=/path/to/nejm-brain-to-text       # CARD participant data
export KUNZ_ROOT=/path/to/cross_participant_speech  # Kunz participant data
export WILLET_CODE=/path/to/speech_decoding_BCI     # Willet dataset
export RNN_ARGS=/path/to/rnn_args.yaml              # upstream model args
```

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
