from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _as_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list, got {type(value)}")
    return [int(v) for v in value]


def _as_float_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list, got {type(value)}")
    return [float(v) for v in value]


def validate_config(cfg: Dict[str, Any]) -> None:
    for section in ("data", "model", "training", "loss"):
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Missing or invalid config section: {section}")

    data = cfg["data"]
    training = cfg["training"]
    loss = cfg["loss"]

    data["stride_values"] = _as_int_list(data.get("stride_values", []), "data.stride_values")
    data["stride_probs"] = _as_float_list(data.get("stride_probs", []), "data.stride_probs")

    clip_frames = int(data.get("clip_frames", 0))
    if clip_frames != 64:
        raise ValueError(f"data.clip_frames must be 64, got {clip_frames}")

    chunk_frames = int(data.get("chunk_frames", 0))
    if chunk_frames <= 0 or chunk_frames % 2 != 0:
        raise ValueError("data.chunk_frames must be a positive even integer.")

    chunk_start_frames = int(data.get("chunk_start_frames", 0))
    if chunk_start_frames != 32:
        raise ValueError(
            f"data.chunk_start_frames must be 32 (sample from first 32 frames), got {chunk_start_frames}."
        )

    window_frames = int(data.get("window_frames", clip_frames))
    frame_stride = int(data.get("frame_stride", 1))
    if window_frames <= 0:
        raise ValueError("data.window_frames must be > 0")
    if frame_stride <= 0:
        raise ValueError("data.frame_stride must be > 0")
    sampled_frames = len(range(0, window_frames, frame_stride))
    if sampled_frames < clip_frames:
        raise ValueError(
            "data.window_frames/frame_stride produce too few frames: "
            f"sampled={sampled_frames}, require >= clip_frames({clip_frames})."
        )

    shuffle_buffer_size = int(data.get("shuffle_buffer_size", 0))
    if shuffle_buffer_size < 0:
        raise ValueError("data.shuffle_buffer_size must be >= 0")

    video_key = data.get("video_key", data.get("video_column", "video_blob"))
    if video_key is not None and str(video_key).strip() == "":
        raise ValueError("data.video_key must be non-empty when provided")

    video_keys = data.get("video_keys", ["video_blob", "video", "blob", "bytes"])
    if not isinstance(video_keys, (list, tuple)) or len(video_keys) == 0:
        raise ValueError("data.video_keys must be a non-empty list/tuple")

    stream_retry_cfg = data.get("stream_retry", {})
    if stream_retry_cfg is None:
        stream_retry_cfg = {}
    if not isinstance(stream_retry_cfg, dict):
        raise ValueError("data.stream_retry must be a mapping when provided")
    max_consecutive_failures = int(stream_retry_cfg.get("max_consecutive_failures", 0))
    base_sleep_seconds = float(stream_retry_cfg.get("base_sleep_seconds", 1.0))
    max_sleep_seconds = float(stream_retry_cfg.get("max_sleep_seconds", 30.0))
    if max_consecutive_failures < 0:
        raise ValueError("data.stream_retry.max_consecutive_failures must be >= 0")
    if base_sleep_seconds < 0 or max_sleep_seconds < 0:
        raise ValueError("data.stream_retry sleep values must be >= 0")

    data["window_frames"] = window_frames
    data["frame_stride"] = frame_stride
    data["shuffle_buffer_size"] = shuffle_buffer_size
    data["video_key"] = str(video_key) if video_key is not None else None
    data["video_keys"] = [str(v) for v in video_keys]
    data["stream_retry"] = {
        "max_consecutive_failures": max_consecutive_failures,
        "base_sleep_seconds": base_sleep_seconds,
        "max_sleep_seconds": max_sleep_seconds,
    }

    stride_values = data["stride_values"]
    stride_probs = data["stride_probs"]

    if len(stride_values) == 0:
        raise ValueError("data.stride_values must not be empty")
    if len(stride_values) != len(stride_probs):
        raise ValueError("data.stride_values and data.stride_probs must have the same length")

    if any(v <= 0 for v in stride_values):
        raise ValueError(f"stride_values must be >0, got {stride_values}")
    if any(v <= 0 for v in stride_probs):
        raise ValueError(f"stride_probs must be >0, got {stride_probs}")

    start_max = chunk_start_frames - 1
    max_stride = max(stride_values)
    max_index = start_max + (chunk_frames - 1) * max_stride
    if max_index >= clip_frames:
        raise ValueError(
            "Chunk sampling exceeds 64-frame clip: "
            f"max index = {max_index}, require < {clip_frames}."
        )

    global_batch_size = int(training.get("global_batch_size", 0))
    if global_batch_size <= 0:
        raise ValueError("training.global_batch_size must be > 0")

    steps_per_epoch = int(training.get("steps_per_epoch", 0))
    if steps_per_epoch <= 0:
        raise ValueError("training.steps_per_epoch must be > 0")

    precision = str(training.get("precision", "")).lower()
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"Unsupported precision: {precision}")

    training["betas"] = _as_float_list(training.get("betas", [0.9, 0.95]), "training.betas")
    if len(training["betas"]) != 2:
        raise ValueError("training.betas must have length 2")

    if float(loss.get("l1_weight", 0.0)) < 0 or float(loss.get("lpips_weight", 0.0)) < 0:
        raise ValueError("loss weights must be >= 0")

    wandb_cfg = cfg.get("wandb")
    if wandb_cfg is not None:
        if not isinstance(wandb_cfg, dict):
            raise ValueError("wandb must be a mapping when provided")

        mode = str(wandb_cfg.get("mode", "online")).lower()
        if mode not in ("online", "offline", "disabled"):
            raise ValueError(f"wandb.mode must be one of online/offline/disabled, got {mode}")

        log_interval_steps = int(wandb_cfg.get("log_interval_steps", 0))
        log_media_interval_steps = int(wandb_cfg.get("log_media_interval_steps", 0))
        log_video_fps = int(wandb_cfg.get("log_video_fps", 4))

        if log_interval_steps < 0:
            raise ValueError("wandb.log_interval_steps must be >= 0")
        if log_media_interval_steps < 0:
            raise ValueError("wandb.log_media_interval_steps must be >= 0")
        if log_video_fps <= 0:
            raise ValueError("wandb.log_video_fps must be > 0")

        if bool(wandb_cfg.get("enabled", False)):
            project = str(wandb_cfg.get("project", "")).strip()
            if project == "":
                raise ValueError("wandb.project must be set when wandb.enabled=true")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config root must be a mapping, got {type(cfg)}")

    validate_config(cfg)
    return cfg


def resolve_path(path: str, base: Optional[str] = None) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if base is None:
        return str(p)
    return str((Path(base) / p).resolve())
