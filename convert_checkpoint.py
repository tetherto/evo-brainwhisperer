"""Convert a train.py checkpoint into a folder that can be loaded from pretrained.
Writes model.safetensors, config.json and a default README.md,

Usage:
    python convert_checkpoint.py \\
    --checkpoint <run_dir>/checkpoints/best-epoch=XXXX.ckpt \\
    --config <run_dir>/config.yaml --out <folder>
"""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from brainwhisperer.config.schema import BWConfig
from brainwhisperer.models.build import inference_settings, model_kwargs
from brainwhisperer.models.model import WHISPER_ENCODER_PREFIX, BrainWhisperer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="The run's config.yaml"
    )
    parser.add_argument("--out", type=Path, required=True, help="Folder to write")
    args = parser.parse_args()

    cfg = OmegaConf.merge(OmegaConf.structured(BWConfig), OmegaConf.load(args.config))
    # The checkpoint overwrites every weight, so Whisper's pretrained weights
    # need not be downloaded.
    model = BrainWhisperer(
        **model_kwargs(cfg), inference=inference_settings(cfg), pretrained_whisper=False
    )
    model.load_checkpoint(args.checkpoint)
    model.save_pretrained(args.out)

    # Load the folder back: every saved weight must equal the checkpoint's.
    loaded = BrainWhisperer.from_pretrained(args.out).state_dict()
    differ = [
        name
        for name, tensor in model.state_dict().items()
        if not name.startswith(WHISPER_ENCODER_PREFIX)
        and not torch.equal(tensor, loaded[name])
    ]
    if differ:
        sys.exit(f"{len(differ)} weights differ after loading {args.out}: {differ}")

    size_mb = (args.out / "model.safetensors").stat().st_size / 1e6
    print(f"checkpoint: {args.checkpoint.resolve()}")
    print(f"wrote {args.out.resolve()} ({size_mb:.0f} MB of weights)")
    print("reloaded it: every weight matches the checkpoint")


if __name__ == "__main__":
    main()
