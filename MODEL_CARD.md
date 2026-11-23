---
license: apache-2.0
language:
- en
library_name: brainwhisperer
tags:
- brain-computer-interface
- speech-decoding
- neuroscience
- intracortical
- neuroprosthetics
- whisper
base_model:
- openai/whisper-tiny.en
base_model_relation: finetune
metrics:
- wer
- cer
---

# BrainWhisperer

**BrainWhisperer** decodes attempted speech into English sentences from intracortical recordings of
the speech motor cortex. It replaces the audio encoder of
[Whisper](https://huggingface.co/openai/whisper-tiny.en) with a new encoder that reads neural
signals, and keeps Whisper's text decoder, whose knowledge of English comes from pretraining on
about 680,000 hours of transcribed audio. A single model is trained across six datasets from three
participants of the BrainGate2 clinical trial.

The model is described in
[*BrainWhisperer: Leveraging Large-Scale ASR Models for Neural Speech Decoding*](https://arxiv.org/abs/2603.13321)
(Boccato, Olak and Ferrante, 2026). This card is for the six-dataset model; training and
evaluation code is at
[github.com/tetherto/evo-brainwhisperer](https://github.com/tetherto/evo-brainwhisperer).

**Highlights:**

- **8.6% word error rate on the Brain-to-Text '25 benchmark**, end-to-end and with no external
  language model, on the hidden test set of [Card et al. (2024)](https://doi.org/10.1056/NEJMoa2314132)
  (Kaggle private leaderboard). The paper's model, trained on three of these datasets, scored
  8.7%. When the competition closed, and when the paper appeared on arXiv (March 2026), that was
  to our knowledge the best end-to-end result among published work on this benchmark. Later work
  has since improved on it.
- **One model, six datasets, three participants.** Each dataset has its own small input layer; the
  encoder and decoder are shared. Training across datasets improves accuracy with no
  participant-specific fine-tuning.
- **Small enough to run locally.** 62M parameters (249 MB in float32) and no external language
  model, where the best-performing decoders rely on n-gram and large language models at inference
  time.
- **Open weights** under Apache 2.0, with the training and evaluation code that produced the
  numbers below.

---

## Model details

| Property | Value |
| --- | --- |
| Developed by | [Tether AI Research](https://tether.io/) * |
| Model type | Sequence-to-sequence: neural recordings in, English text out |
| Parameters | 62.15M † |
| Base model | [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en): tokenizer and text decoder (frozen, adapted with LoRA). The audio encoder is replaced by a new encoder trained from scratch. |
| Input | Binned threshold crossings and spike-band power from Utah microelectrode arrays in the left precentral gyrus |
| Output | An English sentence, plus phoneme probabilities from an auxiliary CTC head (41 classes) |
| Language | English |
| Training data | Six public datasets from five studies (see [Training data](#training-data)) |
| Precision | float32 |
| Model size on disk | 249 MB |

&#42; References to Tether AI Research are references to Tether Data, S.A. de C.V.

† Saved parameters only: Whisper's audio encoder is replaced and not shipped. The count breaks down
into Whisper's decoder (29.87M), the BrainWhisperer encoder (10.93M), six dataset-specific input
layers (21.32M in total) and the phoneme head (0.03M).

### How it works

Whisper expects a log-mel spectrogram of audio. BrainWhisperer feeds it neural activity instead:

1. **Input layer, one per dataset.** A learned projection for each recording day corrects for the
   drift in neural signals from one day to the next (for Card, the largest dataset, it is low-rank
   and grouped by month). A convolutional front-end, with weights shared across datasets, then maps
   the recording to the sequence length and width Whisper's encoder expects, and an encoding of the
   recording day is added.
2. **Encoder.** Whisper's encoder architecture, trained from random weights and modified with
   windowed self-attention in the layers that are most selective for phonemes. A CTC head on these layers predicts phonemes and
   is trained alongside the decoder.
3. **Decoder.** Whisper's pretrained text decoder, fine-tuned with LoRA, writes the sentence.

The model is trained on a combined loss: cross-entropy on the decoder's text and CTC on the
phonemes. At inference it uses diverse beam search (4 beams in 2 groups) and Gaussian smoothing of
the input; both settings are stored in `config.json` and applied by the evaluation script.

---

## Benchmarks

### Brain-to-Text '25

Word error rate in percent on the hidden test set of
[Brain-to-Text '25](https://www.kaggle.com/competitions/brain-to-text-25) (Card et al. 2024),
end-to-end with no language model, scored by Kaggle after the competition closed. Kaggle splits
the test set in two: the public leaderboard scores one part, and the private leaderboard, which
gave the competition's final ranking, scores the rest.

| Model | Private leaderboard | Public leaderboard |
| --- | --- | --- |
| This model (seed 6) | **8.6** | **8.4** |
| Six-dataset model, 8 seeds (mean ± sd) | 9.1 ± 0.4 | 8.9 ± 0.4 |

Seed 6 was chosen by validation WER, not by its test score; across the eight seeds the private
score ranges from 8.5 to 9.6.

### Against prior decoders (from the paper)

Word error rate on the Brain-to-Text '25 competition split of Card, from Table 1 of
[the paper](https://arxiv.org/abs/2603.13321). These are the paper's models, not this checkpoint:
"cross" models are trained on Card, Willett and Kunz together. "With 5-gram" adds an n-gram
language model and rescoring on top of the phoneme predictions; "end-to-end" uses the model's own
output with no external language model. Lower is better; bold marks the best score in each
column.

| Model | WER, with 5-gram | WER, end-to-end |
| --- | --- | --- |
| RNN baseline ([Willett et al. 2023](https://doi.org/10.1038/s41586-023-06377-x); [Card et al. 2024](https://doi.org/10.1056/NEJMoa2314132)) | 6.7 | 16.0 |
| BIT ([Zhang et al. 2026](https://arxiv.org/abs/2511.21740)) | 5.2 | 12.2 |
| BIT, cross | **4.1** | 11.1 |
| BrainWhisperer, Card only | 5.9 | 10.2 |
| BrainWhisperer, cross | 5.2 | **8.7** |

BrainWhisperer gives the lowest end-to-end error, and is the first to go below 10% without an
external language model. The paper also measures the cost of each path: end-to-end decoding needs
under 2 GB of GPU memory and about 50 ms per sentence, against about 300 GB of RAM and 750 ms for
5-gram rescoring. An extension of the 5-gram pipeline placed 4th of 466 teams in the
Brain-to-Text '25 challenge.

### Per-dataset results

Word error rate (WER) and character error rate (CER) on each dataset's held-out validation
sentences, in percent; lower is better. We trained the six-dataset model with eight random seeds.
The first two columns are the mean and standard deviation across the eight; the last two are this
checkpoint (seed 6).

| Dataset | Validation sentences | WER, 8 seeds | CER, 8 seeds | WER, this model | CER, this model |
| --- | --- | --- | --- | --- | --- |
| Card | 1,426 | 9.1 ± 0.3 | 6.5 ± 0.2 | 8.8 | 6.2 |
| Willett | 880 | 19.3 ± 0.3 | 13.9 ± 0.2 | 18.8 | 13.5 |
| Kunz T12 | 100 | 11.5 ± 0.9 | 10.5 ± 0.9 | 10.1 | 9.2 |
| Kunz T15 | 50 | 7.2 ± 0.9 | 6.6 ± 0.8 | 7.4 | 6.4 |
| Wairagkar | 50 | 18.2 ± 3.2 | 16.4 ± 2.5 | 18.6 | 16.4 |
| Jude | 54 | 43.8 ± 2.7 | 30.2 ± 1.6 | 42.3 | 28.6 |

We published seed 6 because it had the lowest Card validation WER of the eight. Its own validation
scores are therefore slightly optimistic; the eight-seed mean is the fairer estimate. Wairagkar,
Kunz T15 and Jude have about 50 validation sentences each, so a single sentence moves their WER by
one to two points.

Text is normalised with Whisper's English normaliser before scoring.

---

## Training data

The model was trained on six datasets, released by the original authors on Dryad. All recordings
come from participants of the [BrainGate2](https://clinicaltrials.gov/study/NCT00912041) clinical
trial, who are identified here only by the trial IDs the original papers use.

| Dataset | Study | Participant | Task | Recording days | Source | Licence |
| --- | --- | --- | --- | --- | --- | --- |
| Card | [Card et al. 2024](https://doi.org/10.1056/NEJMoa2314132), *NEJM* | T15 | Attempted speech, vocalised or silent | 45 | [Dryad](https://doi.org/10.5061/dryad.dncjsxm85) | CC0-1.0 |
| Willett | [Willett et al. 2023](https://doi.org/10.1038/s41586-023-06377-x), *Nature* | T12 | Attempted speech | 24 | [Dryad](https://doi.org/10.5061/dryad.x69p8czpq) | CC0-1.0 |
| Kunz T12 | [Kunz et al. 2025](https://doi.org/10.1016/j.cell.2025.06.015), *Cell* | T12 | Sentence trials of the inner-speech study | 1 | [Dryad](https://doi.org/10.5061/dryad.gf1vhhn1j) | CC0-1.0 |
| Kunz T15 | [Kunz et al. 2025](https://doi.org/10.1016/j.cell.2025.06.015), *Cell* | T15 | Sentence trials of the inner-speech study | 3 | [Dryad](https://doi.org/10.5061/dryad.gf1vhhn1j) | CC0-1.0 |
| Wairagkar | [Wairagkar et al. 2025](https://doi.org/10.1038/s41586-025-09127-3), *Nature* | T15 | Attempted speech, 50-word vocabulary | 10 ‡ | [Dryad](https://doi.org/10.5061/dryad.2280gb64f) | CC0-1.0 |
| Jude | [Jude et al. 2026](https://doi.org/10.1016/j.celrep.2026.117162), *Cell Reports* | T17 | Intended speech | 5 | [Dryad](https://doi.org/10.5061/dryad.vq83bk481) | CC0-1.0 |

‡ The Wairagkar release is a single session of ten blocks; the model treats each block as a
recording day.

The datasets are used as published, de-identified by their authors. The training targets are the
sentences shown to the participants on screen, not their own words.

---

## Usage

| File | Size | What it is |
| --- | --- | --- |
| `model.safetensors` | 249 MB | The weights |
| `config.json` | 5 KB | Every constructor argument, and the inference settings behind the scores above |

BrainWhisperer runs through its own library, not the Hugging Face `pipeline`: its input is a batch
of neural recordings prepared by the dataset recipes in the code repository. Install
[`neuralshed`](https://github.com/tetherto/neuralshed) and BrainWhisperer as described in the
[repository README](https://github.com/tetherto/evo-brainwhisperer#setup), and download the
datasets you want to score (the README has a one-line command). Then score the model:

```bash
python eval.py --model qvac/BrainWhisperer --datasets card
```

This prints WER, CER and phoneme error rate for each dataset listed. Leave out `--datasets` to
score all six. It runs on CUDA, Apple silicon (MPS) or CPU.

Or load it in Python:

```python
from brainwhisperer.models.model import BrainWhisperer

model = BrainWhisperer.from_pretrained("qvac/BrainWhisperer").eval()
print(model.inference)  # beam search and smoothing settings used for the scores above
```

`brainwhisperer.prediction.predict_batch` runs the same decoding as `eval.py` on one batch from a
dataset recipe.

<!-- TODO: confirm the Hub repo name (qvac/BrainWhisperer is a placeholder). -->

---

## Intended use

BrainWhisperer is a **research release**. It is meant for researchers working on speech
brain-computer interfaces who want to reproduce the paper, compare against it on the public
benchmarks, or build on its architecture and training code.

It is **not a medical device**. It has not been evaluated for clinical use, and it must not be used
to make decisions about a person's care or to communicate on behalf of a participant.

## Limitations

- **Only the participants and recording days it was trained on.** Each input layer has one slot per
  training day of one dataset. The model cannot decode a new participant, a new array, or a new
  recording day without retraining; it is a pretrained starting point for those, not a
  plug-and-play decoder.
- **Cued sentences only.** Every training trial is a participant attempting or imagining a sentence
  shown on screen. Accuracy on spontaneous, conversational speech is unknown.
- **English only**, with the vocabulary of the training sentences. The Wairagkar data covers only a
  50-word vocabulary.
- **Accuracy varies widely between datasets**, from about 7% WER (Kunz T15) to about 44% (Jude).
  Datasets with fewer recording days and trials decode less accurately.
- **Offline decoding of complete trials**, not streaming.
- **No language model.** The reported scores are end-to-end. The code repository includes an
  optional decoder with an n-gram and a large language model (`eval_lightbeam.py`), which is not
  part of this release.

## Acknowledgements

We thank the BrainGate2 participants and their families, whose contribution made these datasets
possible, and the teams who recorded and openly released them: Card et al., Willett et al., Kunz et
al., Wairagkar et al. and Jude et al. BrainWhisperer is built on OpenAI's
[Whisper](https://github.com/openai/whisper), Hugging Face
[Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft),
and reads its data through [`neuralset`](https://pypi.org/project/neuralset/) and
[`neuralshed`](https://github.com/tetherto/neuralshed).

## Citation

**BrainWhisperer**

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

**Training data.** If you use this model, please also cite the studies whose data it was trained on.

```bibtex
@article{card2024speech,
  title   = {An accurate and rapidly calibrating speech neuroprosthesis},
  author  = {Card, Nicholas S. and Wairagkar, Maitreyee and Iacobacci, Carrina and Hou, Xianda and Singer-Clark, Tyler and Willett, Francis R. and Kunz, Erin M. and Fan, Chaofei and Vahdati Nia, Maryam and Deo, Darrel R. and Srinivasan, Aparna and Choi, Eun Young and Glasser, Matthew F. and Hochberg, Leigh R. and Henderson, Jaimie M. and Shahlaie, Kiarash and Stavisky, Sergey D. and Brandman, David M.},
  journal = {New England Journal of Medicine},
  volume  = {391},
  number  = {7},
  pages   = {609--618},
  year    = {2024},
  doi     = {10.1056/NEJMoa2314132},
}

@article{willett2023speech,
  title   = {A high-performance speech neuroprosthesis},
  author  = {Willett, Francis R. and Kunz, Erin M. and Fan, Chaofei and Avansino, Donald T. and Wilson, Guy H. and Choi, Eun Young and Kamdar, Foram and Glasser, Matthew F. and Hochberg, Leigh R. and Druckmann, Shaul and Shenoy, Krishna V. and Henderson, Jaimie M.},
  journal = {Nature},
  volume  = {620},
  number  = {7976},
  pages   = {1031--1036},
  year    = {2023},
  doi     = {10.1038/s41586-023-06377-x},
}

@article{kunz2025innerspeech,
  title   = {Inner speech in motor cortex and implications for speech neuroprostheses},
  author  = {Kunz, Erin M. and Abramovich Krasa, Benyamin and Kamdar, Foram and Avansino, Donald T. and Hahn, Nick and Yoon, Seonghyun and Singh, Akansha and Nason-Tomaszewski, Samuel R. and Card, Nicholas S. and Jude, Justin J. and Jacques, Brandon G. and Bechefsky, Payton H. and Iacobacci, Carrina and Hochberg, Leigh R. and Rubin, Daniel B. and Williams, Ziv M. and Brandman, David M. and Stavisky, Sergey D. and AuYong, Nicholas and Pandarinath, Chethan and Druckmann, Shaul and Henderson, Jaimie M. and Willett, Francis R.},
  journal = {Cell},
  volume  = {188},
  number  = {17},
  pages   = {4658--4673.e17},
  year    = {2025},
  doi     = {10.1016/j.cell.2025.06.015},
}

@article{wairagkar2025voice,
  title   = {An instantaneous voice-synthesis neuroprosthesis},
  author  = {Wairagkar, Maitreyee and Card, Nicholas S. and Singer-Clark, Tyler and Hou, Xianda and Iacobacci, Carrina and Miller, Lee M. and Hochberg, Leigh R. and Brandman, David M. and Stavisky, Sergey D.},
  journal = {Nature},
  volume  = {644},
  number  = {8075},
  pages   = {145--153},
  year    = {2025},
  doi     = {10.1038/s41586-025-09127-3},
}

@article{jude2026anarthria,
  title   = {Decoding intended speech with an intracortical brain-computer interface in a person with long-standing anarthria and locked-in syndrome},
  author  = {Jude, Justin J. and Haro, Stephanie and Levi-Aharoni, Hadar and Hashimoto, Hiroaki and Acosta, Alexander J. and Card, Nicholas S. and Wairagkar, Maitreyee and Brandman, David M. and Stavisky, Sergey D. and Williams, Ziv M. and Cash, Sydney S. and Simeral, John D. and Hochberg, Leigh R. and Rubin, Daniel B.},
  journal = {Cell Reports},
  volume  = {45},
  number  = {4},
  pages   = {117162},
  year    = {2026},
  doi     = {10.1016/j.celrep.2026.117162},
}
```

## Copyright

We will take appropriate actions in response to notices of copyright infringement. If you believe
your work has been used or copied in a manner that infringes upon your intellectual property
rights, please email data-apps@tether.io identifying and describing both the copyrighted work and
alleged infringing content.

## Licensing

This model, which was trained as described in the BrainWhisperer paper, is licensed by Tether Data,
S.A. de C.V. under the Apache 2.0 license. As described in the paper, this model is a version of
whisper-tiny.en (https://huggingface.co/openai/whisper-tiny.en), which is licensed under the Apache
2.0 license.

The Card et al. 2024 dataset (https://doi.org/10.5061/dryad.dncjsxm85), the Willett et al. 2023
dataset (https://doi.org/10.5061/dryad.x69p8czpq), the Kunz et al. 2025 dataset
(https://doi.org/10.5061/dryad.gf1vhhn1j), the Wairagkar et al. 2025 dataset
(https://doi.org/10.5061/dryad.2280gb64f) and the Jude et al. 2026 dataset
(https://doi.org/10.5061/dryad.vq83bk481) are made available under the CC0-1.0 license (Creative
Commons Zero v1.0 Universal).

As described in the paper, the whisper-tiny.en model was adapted and trained using the Card et al.
2024, Willett et al. 2023, Kunz et al. 2025 (participants T12 and T15), Wairagkar et al. 2025 and
Jude et al. 2026 datasets.
