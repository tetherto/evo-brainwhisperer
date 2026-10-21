"""SLURM / torchrun submission helpers.

Standalone usage (this file as __main__):
    python -m brainwhisperer.training.launcher training.devices=2
    CUDA_VISIBLE_DEVICES=1,2 python -m brainwhisperer.training.launcher --seeds 1,2,3 training.devices=2
    python -m brainwhisperer.training.launcher --slurm training.devices=4 experiment=X
    python -m brainwhisperer.training.launcher --slurm --dry-run training.devices=4
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

from brainwhisperer.config.schema import BWConfig

REPO_ROOT = Path(__file__).parents[2]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _task_overrides() -> list[str]:
    """Return the raw Hydra task overrides that were passed on the command line."""
    from hydra.core.hydra_config import HydraConfig

    return list(HydraConfig.get().overrides.task)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def launch_torchrun(cfg: BWConfig) -> None:
    """Launch torchrun for a single-node multi-GPU run."""
    n = cfg.training.devices
    if n <= 1:
        raise ValueError(
            f"launch_torchrun requires training.devices > 1; got {n}. "
            "Set training.devices=N with N ≥ 2."
        )

    overrides = _task_overrides()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={n}",
        str(REPO_ROOT / "train.py"),
        *overrides,
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def submit_slurm(cfg: BWConfig, *, dry_run: bool = False) -> None:
    """Build and submit an sbatch script from cfg.

    The script exports TETHER_DATA_ROOT and WANDB_API_KEY from the current
    environment, then invokes torchrun with the full Hydra task overrides.
    """
    import os
    import shutil
    import tempfile

    # Fail early with an actionable message — sbatch only exists on the cluster.
    if not dry_run and shutil.which("sbatch") is None:
        raise RuntimeError(
            "sbatch not found on PATH.\n"
            "submit_slurm must be run on the SLURM head node, not a local machine.\n"
            "To inspect the generated script without submitting, add --dry-run."
        )

    n = cfg.training.devices
    overrides = _task_overrides()
    override_str = " ".join(shlex.quote(o) for o in overrides)

    required_vars = ["TETHER_DATA_ROOT", "WANDB_API_KEY"]
    env_exports = "\n".join(
        f"export {k}={shlex.quote(os.environ[k])}"
        for k in required_vars
        if k in os.environ
    )

    train_script = REPO_ROOT / "train.py"
    script = f"""\
#!/bin/bash
#SBATCH --job-name=brainwhisperer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{n}
#SBATCH --cpus-per-task={4 * n}
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

{env_exports}

torchrun --nproc_per_node={n} \\
    {train_script} {override_str}
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", prefix="brainwhisperer_slurm_", delete=False
    ) as fh:
        fh.write(script)
        script_path = fh.name

    if dry_run:
        print(f"[dry-run] sbatch script written to: {script_path}")
        print("--- script ---")
        print(script)
        return

    print(f"Submitting sbatch script: {script_path}")
    subprocess.run(["sbatch", script_path], check=True)


# ---------------------------------------------------------------------------
# Multi-seed sequential runner
# ---------------------------------------------------------------------------


def run_seeds(seeds: list[int], base_argv: list[str]) -> None:
    """Run multiple seeds sequentially in one invocation.
    Each seed is launched as a new subprocess (same Python, same args, with
    training.seed=N appended). Aborts on first failure to avoid stray GPU processes.
    """
    drain_seconds = 10

    for seed in seeds:
        cmd = (
            [sys.executable, "-m", "brainwhisperer.training.launcher"]
            + base_argv
            + [f"training.seed={seed}"]
        )
        print(f"\n=== seed {seed} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(
                f"Seed {seed} failed (exit {result.returncode}). Aborting sequence."
            )
        print(f"Seed {seed} done. Draining GPU memory ({drain_seconds}s)...")
        # Short drain period to ensure GPU memory fully restores
        time.sleep(drain_seconds)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Strip custom flags to avoid Hydra errors.
    _use_slurm = "--slurm" in sys.argv
    _dry_run = "--dry-run" in sys.argv

    # Running different seeds
    _seeds: list[int] = []
    _filtered: list[str] = []
    _argv = [a for a in sys.argv[1:] if a not in ("--slurm", "--dry-run")]
    i = 0
    while i < len(_argv):
        if _argv[i] == "--seeds":
            _seeds = [int(s) for s in _argv[i + 1].split(",")]
            i += 2
        else:
            _filtered.append(_argv[i])
            i += 1

    if _seeds:
        run_seeds(_seeds, _filtered)
        sys.exit(0)

    sys.argv = [sys.argv[0]] + _filtered

    import hydra
    from omegaconf import DictConfig, OmegaConf

    @hydra.main(
        config_path="../../brainwhisperer/config",
        config_name="base",
        version_base="1.3",
    )
    def _main(cfg: DictConfig) -> None:
        OmegaConf.to_container(cfg, throw_on_missing=True)
        if _use_slurm:
            submit_slurm(cfg, dry_run=_dry_run)
        else:
            launch_torchrun(cfg)

    _main()
