"""SLURM / torchrun submission helpers.

submit_slurm  — generates and submits an sbatch script; used when the target
                cluster uses SLURM for job scheduling.
launch_torchrun — wraps `torchrun --nproc_per_node` for interactive / non-SLURM
                  multi-GPU runs on a single node.
"""

from brainwhisperer.config.schema import BWConfig


def submit_slurm(cfg: BWConfig) -> None:
    """Build and submit an sbatch script from cfg.

    The script must set CARD_ROOT, KUNZ_ROOT, WILLET_CODE, and RNN_ARGS
    from the environment, then invoke `torchrun` with the right
    --nproc_per_node matching cfg.training.devices.
    """
    raise NotImplementedError("SLURM launcher not yet implemented")


def launch_torchrun(cfg: BWConfig) -> None:
    """Launch torchrun for a single-node multi-GPU run.

    Calls `torchrun --nproc_per_node=cfg.training.devices train.py`
    with the full Hydra overrides reconstructed from cfg.
    """
    raise NotImplementedError("torchrun launcher not yet implemented")
