"""Weights & Biases logging utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

__all__ = [
    "init_wandb",
    "log",
    "log_image",
    "log_images",
    "finish",
]

_run = None


def init_wandb(
    project: str = "uni-vug",
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    host: str = "0.0.0.0",
    port: int = 7860,
    **kwargs,
) -> Any:
    """Initialize wandb run. Sets WANDB_BASE_URL for custom host if specified."""
    import os
    import wandb

    global _run
    if _run is not None:
        return _run

    _run = wandb.init(
        project=project,
        name=name,
        config=config,
        **kwargs,
    )
    return _run


def log(metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
    """Log scalar metrics."""
    import wandb

    if wandb.run is None:
        return
    wandb.log(metrics, step=step, commit=commit)


def log_image(
    image_tensor: torch.Tensor,
    step: Optional[int] = None,
    key: str = "samples",
    caption: Optional[str] = None,
    commit: bool = True,
) -> None:
    """Log a single image tensor (C, H, W) in [0, 1]."""
    import wandb
    import numpy as np

    if wandb.run is None:
        return
    if image_tensor.dim() == 3:
        img_np = image_tensor.permute(1, 2, 0).clamp(0, 1).cpu().float().numpy()
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = image_tensor.clamp(0, 1).cpu().float().numpy()
        img_np = (img_np * 255).astype(np.uint8)

    wandb.log(
        {key: wandb.Image(img_np, caption=caption)},
        step=step,
        commit=commit,
    )


def log_images(
    images: Dict[str, torch.Tensor],
    step: Optional[int] = None,
    commit: bool = True,
) -> None:
    """Log multiple named images."""
    import wandb
    import numpy as np

    if wandb.run is None:
        return
    payload = {}
    for key, img in images.items():
        if img.dim() == 3:
            img_np = img.permute(1, 2, 0).clamp(0, 1).cpu().float().numpy()
        else:
            img_np = img.clamp(0, 1).cpu().float().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        payload[key] = wandb.Image(img_np)
    wandb.log(payload, step=step, commit=commit)


def finish() -> None:
    """Finish wandb run."""
    import wandb

    global _run
    if wandb.run is not None:
        wandb.finish()
    _run = None
