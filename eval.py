"""Evaluation entry point: load checkpoint → run inference → report WER / CER / PER.

With `--competition`, runs on the held-out benchmark sets instead (Card on
Kaggle, Willett on EvalAI) and writes their submission files: labels are
withheld, so the scores come from the scorers, not from here.
"""

import argparse
import datetime
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import WhisperTokenizer

from brainwhisperer.config.schema import BWConfig
from brainwhisperer.data.adapter import to_neural_batch
from brainwhisperer.data.collate import default_ragged_collator
from brainwhisperer.data.recipes import COMPETITION_RECIPES, RECIPES
from brainwhisperer.models.model import BrainWhisperer
from brainwhisperer.prediction import predict_batch
from brainwhisperer.utils.metrics import (
    char_error_rate,
    phoneme_error_rate,
    word_error_rate,
)
from brainwhisperer.utils.text import build_normalizer, remove_punctuation


def default_device() -> str:
    """The first available of CUDA, Apple's GPU (MPS) and CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="BrainWhisperer evaluation")
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to .ckpt file"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to config.yaml saved by train.py",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Device string, e.g. 'cuda', 'cuda:1', 'mps' or 'cpu' "
        "(default: the first available of cuda, mps, cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config (useful for memory-constrained eval)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Overrides merged after the saved config, e.g. "
        "--set data.batch_size=32 (leaves the saved config.yaml untouched)",
    )
    parser.add_argument(
        "--competition",
        action="store_true",
        help="Predict the held-out benchmark sets and write submission files "
        "instead of scoring the val sets",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Folder for the submission files (default: evals/<YYYYMMDD>_competition)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config: merge saved YAML onto the structured schema to restore
    # Python types that get lost with the config being saved to YAML and reloaded.
    # ------------------------------------------------------------------
    cfg = OmegaConf.merge(OmegaConf.structured(BWConfig), OmegaConf.load(args.config))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    # The resolved config goes to stderr so stdout holds only the scores.
    print(OmegaConf.to_yaml(cfg), file=sys.stderr)
    print(f"checkpoint: {args.checkpoint.resolve()}", file=sys.stderr)
    print(f"device: {args.device}", file=sys.stderr)
    batch_size = args.batch_size or cfg.data.batch_size
    device = torch.device(args.device)

    # Generation kwargs: diverse beam search params from the paper's Appendix C sweep.
    # Forwarded verbatim to BrainWhisperer.generate() → whisper.generate().
    gen_kwargs = dict(OmegaConf.to_container(cfg.generation, resolve=True))

    # Augmentation kwargs for val-mode preprocessing (smoothing only).
    aug = cfg.training.augmentation
    aug_kwargs = {
        "smooth_kernel_size": aug.smooth_kernel_size,
        "smooth_sigma": aug.smooth_sigma,
    }

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.model.whisper_name, language="en", task="transcribe"
    )
    normalizer = build_normalizer()

    # ------------------------------------------------------------------
    # Validation datasets
    # ------------------------------------------------------------------
    num_days_per_embedder = [RECIPES[name].NUM_DAYS for name in cfg.data.datasets]
    calendar_days_per_embedder = [
        RECIPES[name].calendar_days() for name in cfg.data.datasets
    ]
    # Ensuring a checkpoint trained with per-dataset overrides evaluates
    # with the same per-embedder encodings/day_proj_rank it was trained with.
    embedder_cfg_per_embedder = [
        OmegaConf.merge(cfg.model.embedder, cfg.data.embedder_overrides.get(name, {}))
        for name in cfg.data.datasets
    ]
    # Fill-to-512 rule per dataset: each recipe declares CHANNEL_FILL;
    # datasets without a fill rule declare "none".
    channel_fills_per_embedder = [
        RECIPES[name].CHANNEL_FILL for name in cfg.data.datasets
    ]
    if args.competition:
        # Only the checkpoint's datasets that have a held-out benchmark set.
        val_names = [n for n in cfg.data.datasets if n in COMPETITION_RECIPES]
        if not val_names:
            sys.exit(
                f"--competition: none of {list(cfg.data.datasets)} has a "
                f"competition set (have: {list(COMPETITION_RECIPES)})"
            )
        out_dir = args.out or Path(
            "evals", f"{datetime.date.today():%Y%m%d}_competition"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        val_names = list(cfg.data.datasets)

    # Every dataset comes from its recipe; batches use the default ragged
    # collator. eval.py adapts a ragged batch to a NeuralBatch in its own
    # loop (SingleRecipeCollator is the DataLoader-side twin of this two-step).
    val_loaders = []
    val_adapters = []
    for name in val_names:
        if args.competition:
            recipe = COMPETITION_RECIPES[name]
            val_ds = recipe.build_competition_dataset()
        else:
            recipe = RECIPES[name]  # KeyError = no recipe for this dataset
            val_ds = recipe.build_val_dataset(max_label_len=cfg.data.max_seq_len)
        val_adapters.append(
            lambda raw, n=name, r=recipe: to_neural_batch(
                raw,
                n,
                neural=r.NEURAL,
                whisper=r.WHISPER,
                phonemes=r.PHONEMES,
                day=r.DAY,
            )
        )
        val_loaders.append(
            DataLoader(
                val_ds,
                batch_size=batch_size,
                collate_fn=default_ragged_collator(labels=not args.competition),
                num_workers=cfg.data.num_workers,
                shuffle=False,
            )
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BrainWhisperer(
        cfg.model,
        list(cfg.data.datasets),
        num_days_per_embedder,
        calendar_days_per_embedder,
        embedder_cfg_per_embedder,
        channel_fills=channel_fills_per_embedder,
    )
    model.load_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    measure_per = cfg.training.ctc_phone_coeff > 0.0

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    for name, loader, adapt in zip(val_names, val_loaders, val_adapters):
        hyps: list[str] = []
        refs: list[str] = []
        phone_hyps: list[list[int]] = []
        phone_refs: list[list[int]] = []

        with torch.no_grad():
            for raw in loader:
                # Recipe batches are ragged; adapt to a NeuralBatch,
                # tensor-for-tensor.
                batch = adapt(raw)
                result = predict_batch(
                    model,
                    batch.to_device(device),
                    tokenizer=tokenizer,
                    gen_kwargs=gen_kwargs,
                    aug_kwargs=aug_kwargs,
                    measure_per=measure_per,
                )
                hyps.extend(result.hyps)
                refs.extend(result.refs)
                phone_hyps.extend(result.phone_hyps)
                phone_refs.extend(result.phone_refs)

        if args.competition:
            recipe = COMPETITION_RECIPES[name]
            if len(hyps) != recipe.N_TRIALS:
                sys.exit(
                    f"{name}: {len(hyps)} predictions, the scorer expects "
                    f"{recipe.N_TRIALS}; no file written"
                )
            # The scorers compare against their own references, so clean the
            # way the benchmarks do (lowercase, letters and apostrophes), not
            # with the WER normaliser, which rewrites e.g. "twenty" to "20".
            path = recipe.write_submission(
                out_dir, [remove_punctuation(t) for t in hyps]
            )
            print(f"Dataset: {name}  (n={len(hyps)})  wrote {path}")
            continue

        hyps_norm = [normalizer(t) for t in hyps]
        refs_norm = [normalizer(t) for t in refs]

        wer = word_error_rate(hyps_norm, refs_norm)
        cer = char_error_rate(hyps_norm, refs_norm)

        print(f"Dataset: {name}  (n={len(refs)})")
        print(f"  WER: {wer:.4f}")
        print(f"  CER: {cer:.4f}")
        if phone_hyps:
            per = phoneme_error_rate(phone_hyps, phone_refs)
            print(f"  PER: {per:.4f}")
        print()


if __name__ == "__main__":
    main()
