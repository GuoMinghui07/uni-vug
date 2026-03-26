"""Optimizer and learning-rate scheduler builders."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    "build_optimizer",
    "build_scheduler",
    "get_autocast_scaler",
]


def build_optimizer(
    params: List[torch.nn.Parameter],
    cfg: Dict,
) -> Tuple[torch.optim.Optimizer, str]:
    """Build AdamW optimizer from config dict."""
    opt_cfg = cfg.get("optimizer", cfg)
    lr = float(opt_cfg.get("lr", 1e-4))
    betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    optimizer = AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    msg = f"Optimizer: AdamW(lr={lr}, betas={betas}, wd={weight_decay})"
    return optimizer, msg


def _cosine_schedule(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    final_lr: float,
) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = final_lr + (base_lr - final_lr) * cosine
    return lr / base_lr


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    cfg: Dict,
) -> Tuple[LambdaLR, str]:
    """Build LR scheduler from config dict."""
    sched_cfg = cfg.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine")

    warmup_epochs = float(sched_cfg.get("warmup_epochs", 1))
    decay_end_epoch = float(sched_cfg.get("decay_end_epoch", 100))
    base_lr = float(sched_cfg.get("base_lr", cfg.get("optimizer", {}).get("lr", 1e-4)))
    final_lr = float(sched_cfg.get("final_lr", 1e-6))
    warmup_from_zero = sched_cfg.get("warmup_from_zero", True)

    warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_steps = int(decay_end_epoch * steps_per_epoch)

    if sched_type == "cosine":
        fn = lambda step: _cosine_schedule(step, warmup_steps, total_steps, base_lr, final_lr)
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

    scheduler = LambdaLR(optimizer, lr_lambda=fn)
    msg = (
        f"Scheduler: {sched_type}(warmup={warmup_epochs}ep/{warmup_steps}steps, "
        f"decay_end={decay_end_epoch}ep/{total_steps}steps, "
        f"base_lr={base_lr}, final_lr={final_lr})"
    )
    return scheduler, msg


def get_autocast_scaler(args) -> Tuple[Optional[torch.amp.GradScaler], Dict]:
    """Return (scaler, autocast_kwargs) based on --precision flag."""
    precision = getattr(args, "precision", "bf16")
    if precision == "fp16":
        return torch.amp.GradScaler("cuda"), {"dtype": torch.float16, "enabled": True}
    elif precision == "bf16":
        return None, {"dtype": torch.bfloat16, "enabled": True}
    else:
        return None, {"dtype": torch.float32, "enabled": False}
