"""Weights & Biases logging utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

__all__ = [
    "init_wandb",
    "log",
    "log_image",
    "log_images",
    "log_video",
    "log_videos",
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


def _video_to_uint8(video_tensor: torch.Tensor):
    import numpy as np

    if video_tensor.dim() != 4:
        raise ValueError(f"Expected video tensor (T, C, H, W), got {tuple(video_tensor.shape)}")
    if video_tensor.shape[1] in (1, 3):
        # Canonical input: (T, C, H, W)
        video_tchw = video_tensor
    elif video_tensor.shape[-1] in (1, 3):
        # Tolerate accidental (T, H, W, C), normalize to (T, C, H, W)
        video_tchw = video_tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            "Expected channel dimension to be either axis 1 (T,C,H,W) "
            f"or axis -1 (T,H,W,C), got shape {tuple(video_tensor.shape)}"
        )

    vid_np = video_tchw.clamp(0, 1).cpu().float().numpy()
    if vid_np.shape[1] == 1:
        vid_np = np.repeat(vid_np, 3, axis=1)
    vid_np = (vid_np * 255).astype(np.uint8, copy=False)
    return vid_np


def log_video(
    video_tensor: torch.Tensor,
    step: Optional[int] = None,
    key: str = "samples/video",
    caption: Optional[str] = None,
    fps: int = 8,
    commit: bool = True,
) -> None:
    """Log a single video tensor (T, C, H, W) in [0, 1]."""
    import wandb

    if wandb.run is None:
        return

    vid_np = _video_to_uint8(video_tensor)
    wandb.log(
        {key: wandb.Video(vid_np, fps=int(fps), format="mp4", caption=caption)},
        step=step,
        commit=commit,
    )


def log_videos(
    videos: Dict[str, torch.Tensor],
    step: Optional[int] = None,
    fps: int = 8,
    commit: bool = True,
) -> None:
    """Log multiple named videos. Each tensor must be (T, C, H, W)."""
    import wandb

    if wandb.run is None:
        return

    payload = {}
    for key, video_tensor in videos.items():
        vid_np = _video_to_uint8(video_tensor)
        payload[key] = wandb.Video(vid_np, fps=int(fps), format="mp4")
    wandb.log(payload, step=step, commit=commit)


def finish() -> None:
    """Finish wandb run."""
    import wandb

    global _run
    if wandb.run is not None:
        wandb.finish()
    _run = None
