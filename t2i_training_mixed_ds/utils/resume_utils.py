"""Checkpoint resumption and worktree saving."""

from __future__ import annotations

import os
import shutil
import subprocess
from glob import glob
from pathlib import Path
from typing import Optional

import torch

__all__ = [
    "find_resume_checkpoint",
    "load_training_checkpoint",
    "save_worktree",
]


def find_resume_checkpoint(experiment_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the experiment directory."""
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None

    # prefer "ep-last.pt" if present
    last = os.path.join(ckpt_dir, "ep-last.pt")
    if os.path.isfile(last):
        return last

    # find the latest epoch checkpoint
    pattern = os.path.join(ckpt_dir, "ep-*.pt")
    ckpts = sorted(glob(pattern))
    if ckpts:
        return ckpts[-1]

    # also check for step-based checkpoints
    pattern = os.path.join(ckpt_dir, "step-*.pt")
    ckpts = sorted(glob(pattern))
    if ckpts:
        return ckpts[-1]

    return None


def load_training_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    ema_state=None,
):
    """Load model/optimizer/scheduler from a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.module.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if ema_state is not None:
        named_params = dict(model.module.named_parameters())
        ckpt_ema = checkpoint.get("ema")
        ema_state.clear()
        if isinstance(ckpt_ema, dict):
            for name, p in named_params.items():
                if not p.requires_grad:
                    continue
                value = ckpt_ema.get(name)
                if torch.is_tensor(value):
                    ema_state[name] = value.to(device=p.device, dtype=p.dtype)
                else:
                    ema_state[name] = p.detach().clone()
        else:
            for name, p in named_params.items():
                if p.requires_grad:
                    ema_state[name] = p.detach().clone()
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def save_worktree(experiment_dir: str, cfg=None) -> None:
    """Save config and git info for reproducibility."""
    os.makedirs(experiment_dir, exist_ok=True)

    # save config
    if cfg is not None:
        try:
            from omegaconf import OmegaConf
            cfg_path = os.path.join(experiment_dir, "config.yaml")
            OmegaConf.save(cfg, cfg_path)
        except Exception:
            pass

    # save git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        with open(os.path.join(experiment_dir, "git_hash.txt"), "w") as f:
            f.write(git_hash + "\n")
    except Exception:
        pass
