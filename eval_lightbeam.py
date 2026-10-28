"""Cascaded CTC + word-n-gram-LM + LLM decoding (LightBeam), Card only.

Skips the BrainWhisperer decoder entirely: decodes straight from CTC phone
head via beam search + KenLM + Llama-3.2 LoRA rescoring.

Requires the following env variables:
    BRAINAUDIO_ROOT         e.g. checked-out copy of brainaudio Python package implementing CUDA beam-search CTC decoder
    LIGHTBEAM_KENLM_PATH    e.g. a binary KenLM 4-gram word-level language model file
    LIGHTBEAM_ADAPTER_PATH  e.g. Hugging Face-format LoRA adapter directory (config, safetensors weights, tokenizer files) fine-tuned on top of Llama-3.2-1B
    HF_TOKEN                HF token with meta-llama/Llama-3.2-1B access


Also requires `kenlm` installed in the virtual env.
"""

import argparse
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import WhisperTokenizer

from brainwhisperer.config.paths import (
    BRAINAUDIO_ROOT,
    LIGHTBEAM_ADAPTER_PATH,
    LIGHTBEAM_KENLM_PATH,
)
from brainwhisperer.config.schema import BWConfig
from brainwhisperer.data.collate import (
    SingleRecipeCollator,
    default_ragged_collator,
)
from brainwhisperer.data.recipes import RECIPES
from brainwhisperer.models.model import BrainWhisperer
from brainwhisperer.prediction import reference_texts
from brainwhisperer.utils.metrics import word_error_rate
from brainwhisperer.utils.text import remove_punctuation


def _install_nemo_shim() -> None:

    class Hypothesis:
        def __init__(
            self,
            score=None,
            y_sequence=None,
            timestamp=None,
            alignments=None,
            dec_state=None,
            **kw,
        ):
            self.score = score
            self.y_sequence = y_sequence
            self.timestamp = timestamp
            self.alignments = alignments
            self.dec_state = dec_state

    class NBestHypotheses:
        def __init__(self, n_best_hypotheses=None):
            self.n_best_hypotheses = n_best_hypotheses or []

    class PrettyStrEnum(str):
        pass

    def _fake_module(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    for name in [
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.parts",
        "nemo.collections.asr.parts.utils",
    ]:
        _fake_module(name)
    _fake_module(
        "nemo.collections.asr.parts.utils.rnnt_utils",
        Hypothesis=Hypothesis,
        NBestHypotheses=NBestHypotheses,
    )
    _fake_module("nemo.utils")
    _fake_module("nemo.utils.enum", PrettyStrEnum=PrettyStrEnum)


def _patch_transformers_dtype_kwarg() -> None:
    """Translate dtype to torch_dtype.
    Patch needed because of the old transformers version used for this model"""
    from transformers import AutoModelForCausalLM

    _orig = AutoModelForCausalLM.from_pretrained.__func__

    def _patched(cls, *args, **kwargs):
        if "dtype" in kwargs and "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return _orig(cls, *args, **kwargs)

    AutoModelForCausalLM.from_pretrained = classmethod(_patched)


def get_ctc_decoder(
    brainaudio_path,
    word_lm_path,
    hf_token,
    word_history,
    beam_size=1000,
    config="B2T_25",
    device="cpu",
    adapter_path=None,
):
    brainaudio_parent_dir = os.path.dirname(brainaudio_path)

    os.environ["B2T_BASE_PATH"] = brainaudio_parent_dir
    os.environ["B2T_BRAIN2TEXT_DIR"] = (
        brainaudio_parent_dir  # NOTE: just a placeholder to let the code run
    )

    sys.path.append(os.path.join(brainaudio_path, "scripts"))

    from decoder_config import _B2T_24, _B2T_25, BEAM_SEARCH, LLM
    from run_decoder import load_llm

    if config == "B2T_24":
        b2t_config = _B2T_24
    elif config == "B2T_25":
        b2t_config = _B2T_25
    else:
        raise Exception("Invalid config")

    BRAINAUDIO_PATH = brainaudio_path
    WORD_LM_PATH = word_lm_path
    HF_TOKEN = hf_token
    BEAM_SIZE = beam_size

    @dataclass
    class Config:
        disable_llm = False
        model = LLM["model"]
        hf_token = HF_TOKEN
        load_in_4bit = False
        no_adapter = False if adapter_path is not None else True
        lora_adapter = adapter_path
        llm_weight = LLM["llm_weight"]
        ngram_rescore_weight = LLM["ngram_rescore_weight"]
        scoring_chunk_size = LLM["scoring_chunk_size"]
        length_normalize = False
        word_lm_path = WORD_LM_PATH
        alpha_ngram = b2t_config["alpha_ngram"]
        tokens = os.path.join(
            BRAINAUDIO_PATH, "auxiliary_folders/shallow_fusion/units_pytorch.txt"
        )
        lexicon = os.path.join(
            BRAINAUDIO_PATH,
            "auxiliary_folders/shallow_fusion/vocab_lower_100k_pytorch_phoneme_with_variants.txt",
        )
        beam_size = BEAM_SIZE
        num_homophone_beams = BEAM_SEARCH["num_homophone_beams"]
        beam_prune_threshold = BEAM_SEARCH["beam_prune_threshold"]
        homophone_prune_threshold = BEAM_SEARCH["homophone_prune_threshold"]
        beam_beta = BEAM_SEARCH["beam_beta"]
        word_boundary_bonus = BEAM_SEARCH["word_boundary_bonus"]
        lm_rescore_interval = b2t_config["lm_rescore_interval"]
        score_combination = BEAM_SEARCH["score_combination"]

    args = Config()

    lm_fusion = load_llm(args, device=device)

    from brainaudio.inference.decoder import (
        BatchedBeamCTCComputer,
        VectorizedLexiconConstraint,
    )
    from brainaudio.inference.decoder.word_ngram_lm_optimized import FastNGramLM

    word_ngram_lm = FastNGramLM(args.word_lm_path, alpha=args.alpha_ngram)

    lexicon = VectorizedLexiconConstraint.from_file_paths(
        tokens_file=str(args.tokens),
        lexicon_file=str(args.lexicon),
        device=device,
    )

    return BatchedBeamCTCComputer(
        blank_index=lexicon.blank_index,
        beam_size=args.beam_size,
        lexicon=lexicon,
        lm_fusion=lm_fusion,
        allow_cuda_graphs=False,
        num_homophone_beams=args.num_homophone_beams,
        beam_threshold=args.beam_prune_threshold,
        homophone_prune_threshold=args.homophone_prune_threshold,
        beam_beta=args.beam_beta,
        word_boundary_bonus=args.word_boundary_bonus,
        word_ngram_lm=word_ngram_lm,
        word_history=word_history,
        lm_rescore_interval=args.lm_rescore_interval,
        score_combination=args.score_combination,
    )


def get_beams_and_scores(result, word_history):
    beams_list, scores_list = [], []
    for idx in range(result.scores.shape[0]):
        scores = result.scores[idx].cpu().numpy()
        valid_indices = [i for i, s in enumerate(scores) if s != -float("inf")]
        valid_indices_sorted = sorted(
            valid_indices, key=lambda i: scores[i], reverse=True
        )

        decoded_beams = []
        lm_scores = []
        for i in valid_indices_sorted:
            context_list = result.context_texts[idx][i]
            if context_list:
                lm_scores.append(context_list[0][0])
                hist_id = context_list[0][2]
                text = word_history.get_text(hist_id)
                decoded_beams.append(text)
            else:
                decoded_beams.append("")

        beams_list.append(decoded_beams)
        scores_list.append(lm_scores)

    return beams_list, scores_list


def get_best(beams_list):
    return [decoded_beams[0] if decoded_beams else "" for decoded_beams in beams_list]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--beam-size", type=int, default=100)
    parser.add_argument("--decoder-batch-size", type=int, default=64)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap val samples, for quick smoke tests"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.merge(OmegaConf.structured(BWConfig), OmegaConf.load(args.config))

    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.model.whisper_name, language="en", task="transcribe"
    )

    # Model-shape metadata (day-axis size, calendar days) comes from the recipes.
    num_days_per_embedder = [RECIPES[name].NUM_DAYS for name in cfg.data.datasets]
    calendar_days_per_embedder = [
        RECIPES[name].calendar_days() for name in cfg.data.datasets
    ]
    embedder_cfg_per_embedder = [
        OmegaConf.merge(cfg.model.embedder, cfg.data.embedder_overrides.get(name, {}))
        for name in cfg.data.datasets
    ]

    # Card's val trials from the recipe; the collator hands the loop
    # ready-made NeuralBatches (Card has no CHANNEL_FILL rule, so the
    # model is built without fills).
    val_ds = RECIPES["card"].build_val_dataset(max_label_len=cfg.data.max_seq_len)
    if args.limit:
        from torch.utils.data import Subset

        val_ds = Subset(val_ds, range(min(args.limit, len(val_ds))))
    val_loader = DataLoader(
        val_ds,
        batch_size=args.decoder_batch_size,
        collate_fn=SingleRecipeCollator(
            "card", RECIPES["card"], default_ragged_collator()
        ),
        shuffle=False,
    )

    model = BrainWhisperer(
        cfg.model,
        list(cfg.data.datasets),
        num_days_per_embedder,
        calendar_days_per_embedder,
        embedder_cfg_per_embedder,
    )
    model.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # --- Collect phone_logits for the whole val set first (cheap forward passes,
    # no grad); LightBeam decoding happens in a second pass since decoder construction
    # (loads the LLM + ~3.45GB KenLM model) is the expensive one-time setup cost. ---
    all_logits, all_lens, targets = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to_device(device)
            out = model(batch, encoder_only=True)
            all_logits.append(out["phone_logits"].cpu())
            all_lens.append(out["x_len"].cpu())
            targets += reference_texts(batch.labels, tokenizer)

    logits = torch.cat(all_logits).to(device)
    lens = torch.cat(all_lens).to(device)
    print(f"card val: n={logits.shape[0]}  phone_logits shape={tuple(logits.shape)}")

    # --- Build the LightBeam decoder (brainaudio_path/word_lm_path/adapter_path via
    # env vars, HF_TOKEN via env var -- never hardcoded, never in a config file). ---
    _install_nemo_shim()
    _patch_transformers_dtype_kwarg()
    sys.path.insert(0, f"{BRAINAUDIO_ROOT}/src")
    from brainaudio.inference.decoder.word_ngram_lm_optimized import WordHistory

    word_history = WordHistory()
    decoder = get_ctc_decoder(
        brainaudio_path=BRAINAUDIO_ROOT,
        word_lm_path=LIGHTBEAM_KENLM_PATH,
        hf_token=os.environ["HF_TOKEN"],
        word_history=word_history,
        beam_size=args.beam_size,
        config="B2T_25",
        device=str(device),
        adapter_path=LIGHTBEAM_ADAPTER_PATH,
    )

    # run_decoder.py's exact log-prob prep recipe: pad 2 all-zero frames on the time
    # axis, log_softmax(logits/temperature)*acoustic_scale. temperature/acoustic_scale
    # from brainaudio's own _B2T_25 config (decoder_config.py), not guessed here.
    temperature, acoustic_scale = 1.0, 0.4
    vocab_size = logits.shape[-1]
    padding = torch.zeros(
        (logits.shape[0], 2, vocab_size), device=device, dtype=logits.dtype
    )
    logits_padded = torch.cat([logits, padding], dim=1)
    lengths = lens + 2
    log_probs = (
        torch.nn.functional.log_softmax(logits_padded / temperature, dim=-1)
        * acoustic_scale
    )

    transcriptions = []
    for i in range(0, log_probs.shape[0], args.decoder_batch_size):
        result = decoder(
            log_probs[i : i + args.decoder_batch_size],
            lengths[i : i + args.decoder_batch_size],
        )
        beams_list, _ = get_beams_and_scores(result, word_history)
        transcriptions += get_best(beams_list)

    wer = word_error_rate(
        [remove_punctuation(t) for t in transcriptions],
        [remove_punctuation(t) for t in targets],
    )
    print(f"\nLightBeam cascaded WER (card, n={len(targets)}): {wer:.4f}")


if __name__ == "__main__":
    main()
