"""Evaluation entry point: load checkpoint → run inference → report WER / CER / PER."""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import WhisperTokenizer

import brainwhisperer.config.schema
import brainwhisperer.data.datasets  # noqa: F401 - unused in this script but triggers dataset registration
from brainwhisperer.config.schema import BWConfig
from brainwhisperer.data.augmentations import transform_data
from brainwhisperer.data.batch import NeuralBatch
from brainwhisperer.data.collate import NeuralCollator
from brainwhisperer.data.registry import get_dataset
from brainwhisperer.models.model import BrainWhisperer
from brainwhisperer.utils.metrics import (
    char_error_rate,
    phoneme_error_rate,
    word_error_rate,
)
from brainwhisperer.utils.sessions import SESSIONS_BY_DATASET, sessions_to_calendar_days
from brainwhisperer.utils.text import build_normalizer


def _ctc_decode_phones(
    phone_logits: torch.Tensor, x_len: torch.Tensor
) -> list[list[int]]:
    """Greedy CTC decode: argmax → collapse consecutive → strip blanks (class 0)."""
    result = []
    for i in range(phone_logits.shape[0]):
        decoded = torch.argmax(phone_logits[i, : x_len[i], :].detach(), dim=-1)
        decoded = torch.unique_consecutive(decoded).cpu().numpy()
        result.append([int(p) for p in decoded if p != 0])
    return result


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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string, e.g. 'cuda', 'cuda:1', or 'cpu'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config (useful for memory-constrained eval)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config: merge saved YAML onto the structured schema to restore
    # Python types that get lost with the config being saved to YAML and reloaded.
    # ------------------------------------------------------------------
    cfg = OmegaConf.merge(
        OmegaConf.structured(BWConfig),
        OmegaConf.load(args.config),
    )
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
    # Tokenizer + collator
    # ------------------------------------------------------------------
    tokenizer = WhisperTokenizer.from_pretrained(
        cfg.model.whisper_name, language="en", task="transcribe"
    )
    collator = NeuralCollator(tokenizer, max_label_len=cfg.data.max_seq_len)
    normalizer = build_normalizer()

    # ------------------------------------------------------------------
    # Validation datasets (same `sbj_idx` assignment as training)
    # ------------------------------------------------------------------
    dataset_classes = [get_dataset(name) for name in cfg.data.datasets]
    num_days_per_embedder = [cls.NUM_DAYS for cls in dataset_classes]
    calendar_days_per_embedder = [
        sessions_to_calendar_days(SESSIONS_BY_DATASET[name])
        for name in cfg.data.datasets
    ]
    val_names = list(cfg.data.datasets)

    val_loaders = []
    for i, cls in enumerate(dataset_classes):
        val_ds = cls(split=cls.VAL_SPLIT, sbj_idx=i)
        val_loaders.append(
            DataLoader(
                val_ds,
                batch_size=batch_size,
                collate_fn=collator,
                num_workers=cfg.data.num_workers,
                shuffle=False,
            )
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BrainWhisperer(cfg.model, num_days_per_embedder, calendar_days_per_embedder)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Lightning stores submodule weights under "model.*" (self.model = BrainWhisperer)
    state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    measure_per = cfg.training.ctc_phone_coeff > 0.0

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    for name, loader in zip(val_names, val_loaders):
        hyps: list[str] = []
        refs: list[str] = []
        phone_hyps: list[list[int]] = []
        phone_refs: list[list[int]] = []

        with torch.no_grad():
            for batch in loader:
                batch = NeuralBatch(
                    signals=batch.signals.to(device),
                    labels=batch.labels.to(device),
                    sbj_idx=batch.sbj_idx.to(device),
                    day_idx=batch.day_idx.to(device),
                    lengths=batch.lengths.to(device),
                    phone_seq=batch.phone_seq.to(device)
                    if batch.phone_seq is not None
                    else None,
                    phone_seq_len=batch.phone_seq_len.to(device)
                    if batch.phone_seq_len is not None
                    else None,
                )

                x_val, len_val = transform_data(
                    batch.signals, batch.lengths, mode="val", **aug_kwargs
                )
                val_batch = NeuralBatch(
                    signals=x_val,
                    labels=batch.labels,
                    sbj_idx=batch.sbj_idx,
                    day_idx=batch.day_idx,
                    lengths=len_val,
                    phone_seq=batch.phone_seq,
                    phone_seq_len=batch.phone_seq_len,
                )

                # WER: free generation with beam search.
                generated_ids = model.generate(val_batch, **gen_kwargs)
                transcriptions = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                label_ids = batch.labels.clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                target_texts = tokenizer.batch_decode(
                    label_ids, skip_special_tokens=True
                )
                hyps.extend(transcriptions)
                refs.extend(target_texts)

                # PER: encoder-only forward to obtain phone logits
                if measure_per and batch.phone_seq is not None:
                    out = model(val_batch, encoder_only=True)
                    decoded = _ctc_decode_phones(out["phone_logits"], out["x_len"])
                    phone_hyps.extend(decoded)
                    for i in range(batch.phone_seq.shape[0]):
                        seq_len = int(batch.phone_seq_len[i])
                        phone_refs.append(batch.phone_seq[i, :seq_len].cpu().tolist())

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
