#!/usr/bin/env python3
"""
Stage-2 DiT video training script (OpenVid-1M streaming).

Trains:
    - DiT backbone
    - MLLM meta query embeddings
    - MLLM connector

Frozen:
    - MLLM backbone (Qwen3-VL)
    - RAE encoder
    - RAE decoder

OpenVid sample policy (strict):
    1) Drop sample if source frame count < 64
    2) Randomly sample one 64-frame contiguous window
    3) Uniformly subsample by frame_stride (default 2 -> 32 frames)
    4) Resize frames to 256x256
    5) Encode/decode in video mode (never frame-wise image encode/decode)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tarfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:
    import httpcore
except Exception:  # pragma: no cover
    httpcore = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DiT import UniDiT
from qwen3_vl_rae_uncompressed import Qwen3VLRAE

from utils.dist_utils import setup_distributed, cleanup_distributed, barrier
from utils.model_utils import load_rae_decoder_weights
from utils.optim_utils import build_optimizer, build_scheduler, get_autocast_scaler
from utils.train_utils import (
    DEFAULT_OPENVID_TEXT_KEYS,
    DEFAULT_OPENVID_VIDEO_KEYS,
    build_openvid_streaming_dataloader,
    configure_experiment_dirs,
)
from utils.resume_utils import find_resume_checkpoint, save_worktree
from utils import wandb_utils


DEFAULT_OPENVID_DATASET_ID = "lance-format/openvid-lance"


def compute_shift_ratio_from_latent(latent: torch.Tensor, base_dim: int = 4096) -> float:
    """Compute schedule-shift ratio from latent grid size (B, T, H, W, C)."""
    if latent.dim() != 5:
        raise ValueError(f"Expected latent with shape (B, T, H, W, C), got {tuple(latent.shape)}")
    _, grid_t, grid_h, grid_w, channels = latent.shape
    input_dim = int(grid_t) * int(grid_h) * int(grid_w) * int(channels)
    if input_dim <= 0:
        raise ValueError(f"Invalid latent shape for shift-ratio computation: {tuple(latent.shape)}")
    return math.sqrt(float(input_dim) / base_dim)


def is_recoverable_stream_error(exc: Exception) -> bool:
    """Heuristic guard for transient streaming failures from remote dataset stream."""
    recoverable_types = (
        tarfile.ReadError,
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        BrokenPipeError,
        EOFError,
    )
    if httpx is not None:
        recoverable_types = recoverable_types + (httpx.RemoteProtocolError,)
    if httpcore is not None:
        recoverable_types = recoverable_types + (httpcore.RemoteProtocolError,)
    if isinstance(exc, recoverable_types):
        return True

    message = str(exc).lower()
    recoverable_tokens = (
        "readerror",
        "unexpected end of data",
        "timed out",
        "timeout",
        "connection reset by peer",
        "got disconnected from remote data host",
        "remote disconnected",
        "remote end closed connection",
        "remoteprotocolerror",
        "peer closed connection without sending complete message body",
        "broken pipe",
        "server disconnected",
        "temporary failure in name resolution",
        "name or service not known",
        "task was aborted",
        "external error",
        "lance_background_thread",
        "arrowinvalid",
        "io error",
    )
    return any(token in message for token in recoverable_tokens)


def load_latent_stats(
    stats_path: str,
    *,
    expected_channels: int,
    device: torch.device,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Load channel-wise latent mean/std from a stats .pt file."""
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Latent stats file not found: '{path}'.")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid latent stats format in '{path}': expected a dict.")

    mean = payload.get("mean")
    std = payload.get("std")
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise ValueError(
            f"Latent stats '{path}' must contain tensor entries 'mean' and 'std'."
        )

    mean = mean.to(torch.float32).flatten()
    std = std.to(torch.float32).flatten()
    if mean.numel() != expected_channels or std.numel() != expected_channels:
        raise ValueError(
            f"Latent stats channel mismatch in '{path}': "
            f"mean={mean.numel()}, std={std.numel()}, expected={expected_channels}."
        )

    std = torch.clamp(std, min=float(eps))
    mean_5d = mean.view(1, 1, 1, 1, -1).to(device=device)
    std_5d = std.view(1, 1, 1, 1, -1).to(device=device)

    global_mean = float(payload.get("global_mean", float(mean.mean().item())))
    global_std = float(payload.get("global_std", float(std.mean().item())))
    token_count = int(payload.get("token_count", 0))

    return {
        "path": str(path),
        "mean": mean_5d,
        "std": std_5d,
        "global_mean": global_mean,
        "global_std": global_std,
        "token_count": token_count,
    }


def normalize_latent(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply channel-wise latent normalization: (x - mean) / std."""
    if latent.dim() != 5:
        raise ValueError(f"Expected latent shape (B, T, H, W, C), got {tuple(latent.shape)}")
    dtype = latent.dtype
    return ((latent.to(torch.float32) - mean) / std).to(dtype=dtype)


def denormalize_latent(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Invert channel-wise latent normalization: x = x_norm * std + mean."""
    if latent.dim() != 5:
        raise ValueError(f"Expected latent shape (B, T, H, W, C), got {tuple(latent.shape)}")
    dtype = latent.dtype
    return (latent.to(torch.float32) * std + mean).to(dtype=dtype)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return inner module for DDP, otherwise model itself."""
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: nn.Module,
    ema_state: Optional[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    raw_model = unwrap_model(model)
    state = {
        "step": step,
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "ema": (
            {k: v.detach().cpu() for k, v in ema_state.items()}
            if ema_state is not None
            else None
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LambdaLR],
    ema_state: Optional[Dict[str, torch.Tensor]],
) -> Tuple[int, int]:
    """Load model/optimizer/scheduler from checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(checkpoint["model"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if ema_state is not None:
        named_params = dict(raw_model.named_parameters())
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

    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))


def init_ema_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


@torch.no_grad()
def update_ema_state(ema_state: Dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ema_p = ema_state.get(name)
        if ema_p is None:
            ema_state[name] = p.detach().clone()
            continue
        ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


@torch.no_grad()
def swap_model_with_ema(model: nn.Module, ema_state: Dict[str, torch.Tensor]) -> None:
    """In-place swap between model trainable params and ema_state."""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ema_p = ema_state.get(name)
        if ema_p is None:
            continue
        if ema_p.device != p.device or ema_p.dtype != p.dtype:
            ema_p = ema_p.to(device=p.device, dtype=p.dtype)
        current = p.detach().clone()
        p.copy_(ema_p)
        ema_state[name] = current


def maybe_autocast(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype, enabled=True)


@torch.no_grad()
def generate_sample_videos(
    model: UniDiT,
    rae: Qwen3VLRAE,
    prompts: List[str],
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
    dtype: torch.dtype,
    amp_enabled: bool,
    num_steps: int = 10,
    latent_stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Generate videos from text prompts for WandB visualization.

    Returns:
        Tensor (N, T, C, H, W) in [0, 1], one sample per prompt.
    """
    model.eval()
    all_videos: List[torch.Tensor] = []

    for prompt in prompts:
        tokenized = model.mllm_encoder.tokenize([prompt])
        tokenized = {
            k: (v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in tokenized.items()
        }

        with maybe_autocast(device, dtype, amp_enabled):
            context, _ = model.encode_condition(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                pixel_values=tokenized.get("pixel_values"),
                image_grid_thw=tokenized.get("image_grid_thw"),
                pixel_values_videos=tokenized.get("pixel_values_videos"),
                video_grid_thw=tokenized.get("video_grid_thw"),
                second_per_grid_ts=tokenized.get("second_per_grid_ts"),
            )
            latent = model.sample(
                context=context,
                grid_t=int(grid_t),
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                num_steps=int(num_steps),
            )

        if latent_stats is not None:
            latent = denormalize_latent(latent, latent_stats["mean"], latent_stats["std"])

        with maybe_autocast(device, dtype, amp_enabled):
            pixels = rae.decode(latent)

        if pixels.dim() == 4:
            pixels = pixels.unsqueeze(1)
        pixels = pixels.clamp(0, 1)
        all_videos.append(pixels[0])

    model.train()
    if not all_videos:
        return torch.empty(0, 0, 3, 1, 1, device=device, dtype=torch.float32)
    return torch.stack(all_videos, dim=0)


def make_video_panel(videos: torch.Tensor) -> torch.Tensor:
    """Convert (N, T, C, H, W) to one panel video (T, C, H, N*W)."""
    if videos.dim() != 5:
        raise ValueError(f"Expected (N, T, C, H, W), got {tuple(videos.shape)}")
    if videos.shape[2] not in (1, 3):
        raise ValueError(
            f"Expected channel dim at axis=2 with size 1/3, got shape {tuple(videos.shape)}"
        )
    n = int(videos.size(0))
    if n <= 0:
        raise ValueError("Cannot build panel from empty videos.")
    parts = [videos[i] for i in range(n)]
    return torch.cat(parts, dim=-1).contiguous()


def make_video_compare_panel(real: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
    """Stack real(top)/generated(bottom), then tile samples horizontally.

    Input:
        real: (N, T, C, H, W)
        generated: (N, T, C, H, W)
    Output:
        (T, C, 2H, N*W)
    """
    if real.dim() != 5 or generated.dim() != 5:
        raise ValueError(
            f"Expected (N, T, C, H, W), got real={tuple(real.shape)} generated={tuple(generated.shape)}"
        )
    if real.shape[2] not in (1, 3) or generated.shape[2] not in (1, 3):
        raise ValueError(
            "Expected channel dim at axis=2 with size 1/3, got "
            f"real={tuple(real.shape)} generated={tuple(generated.shape)}"
        )
    n = min(int(real.size(0)), int(generated.size(0)))
    if n <= 0:
        raise ValueError("No samples to compare.")

    t = min(int(real.size(1)), int(generated.size(1)))
    real = real[:n, :t]
    generated = generated[:n, :t]

    tiles = []
    for i in range(n):
        tile = torch.cat([real[i], generated[i]], dim=-2)  # (T, C, 2H, W)
        tiles.append(tile)
    return torch.cat(tiles, dim=-1).contiguous()  # (T, C, 2H, N*W)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 DiT video training (OpenVid stream).")
    parser.add_argument(
        "--config",
        type=str,
        default="t2v_training_single_ds/config/dit_training.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/scratch/e1539128/ckpt-video",
        help="Output directory.",
    )
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in experiment dir.")
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Resume from explicit checkpoint path. Overrides --resume auto-discovery.",
    )
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help=(
            "When resuming, load only model/EMA weights from checkpoint and keep optimizer/scheduler "
            "from current config."
        ),
    )
    parser.add_argument(
        "--resume-reset-progress",
        action="store_true",
        help=(
            "Only valid with --resume-model-only. After loading model/EMA, reset "
            "epoch/step to 0 and restart LR schedule from the beginning."
        ),
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional WandB run name override.")
    parser.add_argument(
        "--log-non-ema-samples",
        action="store_true",
        help=(
            "Also log non-EMA generated videos at visualization steps for side-by-side comparison. "
            "EMA videos remain logged to the original keys."
        ),
    )
    parser.add_argument(
        "--maintain-ema",
        dest="maintain_ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to maintain EMA weights during training. "
            "Use --no-maintain-ema to disable EMA maintenance."
        ),
    )
    parser.add_argument(
        "--sample-with-ema",
        dest="sample_with_ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether periodic sample visualization uses EMA weights. "
            "Use --no-sample-with-ema for non-EMA sampling."
        ),
    )
    parser.add_argument(
        "--latent-stats",
        type=str,
        default=None,
        help="Path to latent stats .pt with channel-wise mean/std (default: ckpts/latent_stats.pt).",
    )
    parser.add_argument(
        "--disable-latent-norm",
        action="store_true",
        help="Disable latent normalization even if a stats file is provided.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.resume_reset_progress and not args.resume_model_only:
        raise ValueError("--resume-reset-progress requires --resume-model-only.")

    rank, world_size, device = setup_distributed()
    is_ddp = dist.is_initialized()

    from omegaconf import OmegaConf

    config_path = Path(args.config)
    if not config_path.exists():
        fallback_cfg = Path(__file__).resolve().parent / "config" / config_path.name
        if fallback_cfg.exists():
            config_path = fallback_cfg
        else:
            raise FileNotFoundError(
                f"Config not found: '{args.config}'. Also tried '{fallback_cfg}'."
            )
    args.config = str(config_path)
    cfg = OmegaConf.load(str(config_path))
    model_cfg = cfg.get("model", {})
    rae_cfg = cfg.get("rae", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    wandb_cfg = cfg.get("wandb", {})
    cfg_section = cfg.get("cfg", {})
    shift_cfg = cfg.get("schedule_shift", {})
    latent_norm_cfg = cfg.get("latent_norm", {})

    # --- Training hyperparameters ---
    num_epochs = int(train_cfg.get("epochs", 50))
    accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("training.accumulation_steps must be > 0.")

    configured_global_batch_size = train_cfg.get("global_batch_size", 64)
    configured_global_batch_size = int(configured_global_batch_size)
    if configured_global_batch_size <= 0:
        raise ValueError("training.global_batch_size must be > 0.")

    if "micro_batch_size_per_gpu" in train_cfg or "micro_batch_size" in train_cfg:
        micro_batch_size = int(train_cfg.get("micro_batch_size_per_gpu", train_cfg.get("micro_batch_size")))
    else:
        denom = world_size * accumulation_steps
        if configured_global_batch_size % denom != 0:
            raise ValueError(
                "training.global_batch_size must be divisible by world_size * training.accumulation_steps."
            )
        micro_batch_size = configured_global_batch_size // denom

    if micro_batch_size <= 0:
        raise ValueError("training.micro_batch_size_per_gpu must be > 0.")

    global_batch_size = micro_batch_size * accumulation_steps * world_size
    if global_batch_size != configured_global_batch_size:
        raise ValueError(
            "Configured training.global_batch_size does not match "
            "micro_batch_size_per_gpu * accumulation_steps * world_size."
        )

    num_workers = int(train_cfg.get("num_workers", 8))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    if clip_grad <= 0:
        clip_grad = None
    log_interval = int(train_cfg.get("log_interval", 100))
    sample_every = int(train_cfg.get("sample_every", 1000))
    checkpoint_every_steps = int(train_cfg.get("checkpoint_every_steps", 2000))
    if checkpoint_every_steps <= 0:
        raise ValueError("training.checkpoint_every_steps must be > 0.")
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError("training.ema_decay must satisfy 0 <= ema_decay < 1.")

    maintain_ema_cfg = bool(train_cfg.get("maintain_ema", True))
    maintain_ema = maintain_ema_cfg if args.maintain_ema is None else bool(args.maintain_ema)
    sample_with_ema_cfg = bool(wandb_cfg.get("sample_with_ema", False))
    sample_with_ema = sample_with_ema_cfg if args.sample_with_ema is None else bool(args.sample_with_ema)
    sample_num_steps = int(wandb_cfg.get("sample_num_steps", 10))
    sample_fps = int(wandb_cfg.get("sample_fps", 8))
    if sample_num_steps <= 0:
        raise ValueError("wandb.sample_num_steps must be > 0.")
    if sample_fps <= 0:
        sample_fps = 8

    precision = getattr(args, "precision", train_cfg.get("precision", "bf16"))
    args.precision = precision

    # --- Video data policy ---
    dataset_id = str(data_cfg.get("dataset_id", DEFAULT_OPENVID_DATASET_ID))
    data_split = str(data_cfg.get("split", "train"))
    video_size = int(data_cfg.get("video_size", 256))
    window_frames = int(data_cfg.get("window_frames", 64))
    frame_stride = int(data_cfg.get("frame_stride", 2))
    if window_frames < 64:
        raise ValueError("data.window_frames must be >= 64 per OpenVid policy.")
    if frame_stride <= 0:
        raise ValueError("data.frame_stride must be > 0.")
    sampled_frames = len(range(0, window_frames, frame_stride))
    if sampled_frames <= 0:
        raise ValueError("Invalid frame settings produced zero sampled frames.")

    text_keys = list(data_cfg.get("text_keys", list(DEFAULT_OPENVID_TEXT_KEYS)))
    video_keys = list(data_cfg.get("video_keys", list(DEFAULT_OPENVID_VIDEO_KEYS)))
    video_key = data_cfg.get("video_key", "video_blob")
    if video_key is not None:
        video_key = str(video_key)
    shuffle_buffer_size = int(data_cfg.get("shuffle_buffer_size", 0))

    stream_retry_cfg = data_cfg.get("stream_retry", {})
    max_stream_failures = int(stream_retry_cfg.get("max_consecutive_failures", 0))
    stream_retry_base_sleep = float(stream_retry_cfg.get("base_sleep_seconds", 1.0))
    stream_retry_max_sleep = float(stream_retry_cfg.get("max_sleep_seconds", 30.0))
    if stream_retry_base_sleep < 0:
        raise ValueError("data.stream_retry.base_sleep_seconds must be >= 0.")
    if stream_retry_max_sleep < 0:
        raise ValueError("data.stream_retry.max_sleep_seconds must be >= 0.")
    if stream_retry_max_sleep < stream_retry_base_sleep:
        stream_retry_max_sleep = stream_retry_base_sleep

    # --- Seed ---
    default_seed = int(train_cfg.get("global_seed", 42))
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- Experiment dirs ---
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)

    # --- Noise schedule shift ---
    use_shift = bool(shift_cfg.get("enabled", False))
    base_dim = int(shift_cfg.get("base_dim", 4096))
    shift_ratio = 1.0

    # --- CFG dropout ---
    context_dropout_prob = float(cfg_section.get("context_dropout_prob", 0.0))

    # --- Latent normalization ---
    latent_norm_enabled = bool(latent_norm_cfg.get("enabled", True)) and (not args.disable_latent_norm)
    latent_stats_path = args.latent_stats
    if latent_stats_path is None:
        latent_stats_path = str(latent_norm_cfg.get("stats_path", "ckpts/latent_stats.pt"))
    latent_norm_eps = float(latent_norm_cfg.get("eps", 1e-6))
    latent_stats: Optional[Dict[str, Any]] = None
    if latent_norm_enabled:
        latent_stats = load_latent_stats(
            latent_stats_path,
            expected_channels=int(model_cfg.get("rae_latent_dim", 1024)),
            device=device,
            eps=latent_norm_eps,
        )

    # --- Build models ---
    logger.info("Building UniDiT model...")
    model = UniDiT(
        mllm_model_name=str(model_cfg.get("mllm_model_name", "Qwen/Qwen3-VL-4B-Instruct")),
        num_metaqueries=int(model_cfg.get("num_metaqueries", 256)),
        mllm_hidden_size=int(model_cfg.get("mllm_hidden_size", 2560)),
        connector_expansion=int(model_cfg.get("connector_expansion", 4)),
        rae_latent_dim=int(model_cfg.get("rae_latent_dim", 1024)),
        dit_dim=int(model_cfg.get("dit_dim", 1536)),
        dit_ffn_dim=int(model_cfg.get("dit_ffn_dim", 8960)),
        dit_freq_dim=int(model_cfg.get("dit_freq_dim", 256)),
        dit_num_heads=int(model_cfg.get("dit_num_heads", 12)),
        dit_num_layers=int(model_cfg.get("dit_num_layers", 30)),
        dit_patch_size=tuple(model_cfg.get("dit_patch_size", [1, 1, 1])),
        dit_max_seq_len=int(model_cfg.get("dit_max_seq_len", 1024)),
        dit_window_size=tuple(model_cfg.get("dit_window_size", [-1, -1])),
        fm_num_steps=int(model_cfg.get("fm_num_steps", 50)),
        fm_logit_normal_mean=float(model_cfg.get("fm_logit_normal_mean", 0.0)),
        fm_logit_normal_std=float(model_cfg.get("fm_logit_normal_std", 1.0)),
        fm_model_timestep_scale=float(model_cfg.get("fm_model_timestep_scale", 1000.0)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    ).to(device)

    model.flow_matching.use_schedule_shift = use_shift
    model.flow_matching.shift_ratio = shift_ratio

    logger.info("Building RAE (frozen)...")
    decoder_config_path = str(rae_cfg.get("decoder_config_path", "config/decoder_config.json"))
    decoder_cfg_path_obj = Path(decoder_config_path)
    if not decoder_cfg_path_obj.exists():
        fallback_decoder = Path(__file__).resolve().parent.parent / "config" / "decoder_config.json"
        if fallback_decoder.exists():
            logger.warning(
                f"decoder_config_path '{decoder_config_path}' not found, fallback to '{fallback_decoder}'."
            )
            decoder_config_path = str(fallback_decoder)
        else:
            raise FileNotFoundError(
                f"decoder_config_path not found: '{decoder_config_path}' and fallback "
                f"'{fallback_decoder}' also missing."
            )

    rae = Qwen3VLRAE(
        model_name_or_path=str(rae_cfg.get("model_name_or_path", "Qwen/Qwen3-VL-4B-Instruct")),
        decoder_config_path=decoder_config_path,
        noise_tau=float(rae_cfg.get("noise_tau", 0.0)),
        in_channels=int(rae_cfg.get("in_channels", 3)),
        denormalize_output=bool(rae_cfg.get("denormalize_output", True)),
        local_files_only=bool(rae_cfg.get("local_files_only", True)),
        do_resize=bool(rae_cfg.get("do_resize", False)),
    ).to(device)

    decoder_ckpt_path = str(rae_cfg.get("decoder_checkpoint_path", "")).strip()
    if decoder_ckpt_path:
        decoder_ckpt = Path(decoder_ckpt_path)
        if not decoder_ckpt.exists():
            raise FileNotFoundError(f"RAE decoder checkpoint not found: '{decoder_ckpt}'.")
        load_info = load_rae_decoder_weights(
            rae=rae,
            checkpoint_path=decoder_ckpt,
            source=str(rae_cfg.get("decoder_checkpoint_source", "ema")),
        )
        logger.info(
            f"Loaded RAE decoder from '{load_info['checkpoint_path']}' "
            f"(source='{load_info['source']}', decoder_keys={load_info['decoder_keys']}, "
            f"to_pixels_keys_in_ckpt={load_info['to_pixels_keys_in_ckpt']})."
        )
        if hasattr(rae, "to_pixels") and not load_info["loaded_to_pixels"]:
            logger.warning(
                "Checkpoint did not provide to_pixels weights; using current to_pixels initialization."
            )

    rae.requires_grad_(False)
    rae.eval()

    # Enforce encoder temporal chunk size 2 for video latentization.
    temporal_patch_size = int(getattr(rae, "temporal_patch_size", -1))
    if temporal_patch_size != 2:
        raise ValueError(
            f"Expected encoder temporal patch/chunk size == 2, but got {temporal_patch_size}."
        )
    spatial_patch = int(getattr(rae, "patch_size", -1))
    if rank == 0 and spatial_patch != 16:
        logger.warning(
            f"Encoder spatial patch size is {spatial_patch} (expected 16 from requested 16:16:2 setting)."
        )

    # Calibrate schedule shift from actual latent shape.
    if use_shift:
        with torch.no_grad():
            probe = torch.zeros(
                1,
                int(sampled_frames),
                int(rae_cfg.get("in_channels", 3)),
                int(video_size),
                int(video_size),
                device=device,
            )
            latent_probe, _ = rae.encode(probe)
        shift_ratio = compute_shift_ratio_from_latent(latent_probe, base_dim=base_dim)
        model.flow_matching.shift_ratio = shift_ratio

    # Count trainable params.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"UniDiT trainable: {trainable_params / 1e6:.2f}M / total: {total_params / 1e6:.2f}M")
    logger.info(f"Noise schedule shift: enabled={use_shift}, ratio={shift_ratio:.6f}")
    logger.info(
        f"Video policy: window_frames={window_frames}, frame_stride={frame_stride}, "
        f"sampled_frames={sampled_frames}, video_size={video_size}"
    )
    logger.info(
        f"Encoder downsample policy: spatial={spatial_patch}:{spatial_patch}, temporal={temporal_patch_size}"
    )
    if latent_stats is not None:
        logger.info(
            "Latent normalization: enabled=True, "
            f"path='{latent_stats['path']}', global_mean={latent_stats['global_mean']:.6f}, "
            f"global_std={latent_stats['global_std']:.6f}, token_count={latent_stats['token_count']}"
        )
    else:
        logger.info("Latent normalization: enabled=False")
    logger.info(f"CFG context dropout: {context_dropout_prob}")
    if maintain_ema:
        logger.info(f"EMA decay: {ema_decay}")
    else:
        logger.info("EMA maintenance: disabled.")
    if sample_with_ema and not maintain_ema:
        logger.warning("sample_with_ema=True but maintain_ema=False. Forcing sample_with_ema=False.")
        sample_with_ema = False

    # --- DDP ---
    if is_ddp:
        ddp_model: nn.Module = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    else:
        ddp_model = model
    model_core = unwrap_model(ddp_model)

    # --- Optimizer & scheduler ---
    trainable_named_params = [(n, p) for n, p in ddp_model.named_parameters() if p.requires_grad]
    trainable_param_list = [p for _, p in trainable_named_params]
    ema_state: Optional[Dict[str, torch.Tensor]] = (
        init_ema_state(model_core) if maintain_ema else None
    )

    query_token_params = [
        p
        for n, p in trainable_named_params
        if "mllm_encoder.query_token_embeddings" in n
    ]
    if query_token_params:
        wd_params = [
            p
            for n, p in trainable_named_params
            if "mllm_encoder.query_token_embeddings" not in n
        ]
        optim_groups = []
        if wd_params:
            optim_groups.append({"params": wd_params})
        optim_groups.append({"params": query_token_params, "weight_decay": 0.0})
        optimizer, optim_msg = build_optimizer(optim_groups, train_cfg)
        logger.info(optim_msg)
        logger.info("Optimizer override: set weight_decay=0.0 for trainable query-token embeddings.")
    else:
        optimizer, optim_msg = build_optimizer(trainable_param_list, train_cfg)
        logger.info(optim_msg)

    scaler, autocast_kwargs = get_autocast_scaler(args)
    autocast_dtype = autocast_kwargs.get("dtype", torch.bfloat16)
    amp_enabled = bool(autocast_kwargs.get("enabled", False))

    # --- Data loader ---
    logger.info("Configuring OpenVid streaming dataloader...")
    if rank == 0:
        max_failures_msg = "unlimited" if max_stream_failures <= 0 else str(max_stream_failures)
        logger.info(
            "Streaming retry policy: "
            f"max_consecutive_failures={max_failures_msg}, "
            f"base_sleep={stream_retry_base_sleep:.1f}s, "
            f"max_sleep={stream_retry_max_sleep:.1f}s."
        )

    def make_stream_loader(stream_epoch: int):
        return build_openvid_streaming_dataloader(
            dataset_id=dataset_id,
            split=data_split,
            video_size=video_size,
            window_frames=window_frames,
            frame_stride=frame_stride,
            batch_size=micro_batch_size,
            num_workers=num_workers,
            seed=global_seed,
            epoch=stream_epoch,
            prefetch_factor=prefetch_factor,
            shuffle_buffer_size=shuffle_buffer_size,
            text_keys=text_keys,
            video_key=video_key,
            video_keys=video_keys,
            pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

    if "steps_per_epoch" not in train_cfg:
        raise ValueError("training.steps_per_epoch must be set for streaming training.")
    estimated_steps_per_epoch = int(train_cfg.get("steps_per_epoch"))
    if estimated_steps_per_epoch <= 0:
        raise ValueError("training.steps_per_epoch must be > 0.")

    scheduler: Optional[LambdaLR] = None
    if train_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, estimated_steps_per_epoch, train_cfg)
        logger.info(sched_msg)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    resumed = False
    resume_path: Optional[str] = None
    if args.resume_path is not None:
        resume_path_arg = str(args.resume_path).strip()
        if not resume_path_arg:
            raise ValueError(
                "--resume-path is empty. Did you forget to set INIT_CKPT "
                "or pass a literal checkpoint path?"
            )
        explicit_resume = Path(resume_path_arg).expanduser()
        if explicit_resume.is_dir():
            raise IsADirectoryError(
                f"--resume-path points to a directory: '{explicit_resume}'. "
                "Expected a checkpoint file (.pt)."
            )
        if not explicit_resume.is_file():
            raise FileNotFoundError(f"--resume-path not found: '{explicit_resume}'.")
        resume_path = str(explicit_resume)
    elif args.resume:
        resume_path = find_resume_checkpoint(experiment_dir)
        if resume_path is None:
            logger.warning(
                f"--resume was set but no checkpoint found under '{experiment_dir}/checkpoints'; starting from scratch."
            )

    if resume_path is not None:
        logger.info(f"Resuming from {resume_path}...")
        load_optimizer = None if args.resume_model_only else optimizer
        load_scheduler = None if args.resume_model_only else scheduler
        start_epoch, global_step = load_checkpoint(
            resume_path,
            ddp_model,
            load_optimizer,
            load_scheduler,
            ema_state=ema_state,
        )
        if args.resume_model_only:
            if args.resume_reset_progress:
                start_epoch = 0
                global_step = 0
                logger.info(
                    "Resume mode: loaded model/EMA only; reset epoch/step to 0 and restarted scheduler."
                )
            else:
                if scheduler is not None:
                    scheduler.step(global_step)
                logger.info(
                    "Resume mode: loaded model/EMA only; optimizer/scheduler kept from current config."
                )
        logger.info(f"Resumed at epoch={start_epoch}, step={global_step}")
        resumed = True
    if rank == 0 and not resumed:
        save_worktree(experiment_dir, cfg)
        logger.info(f"Experiment dir: {experiment_dir}")

    # --- WandB init ---
    if args.wandb and rank == 0:
        run_name = args.wandb_name if args.wandb_name else Path(experiment_dir).name
        wandb_utils.init_wandb(
            project=str(wandb_cfg.get("project", "uni-vug-dit-video")),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb_utils.log({
            "system/world_size": world_size,
            "system/micro_batch_size_per_gpu": micro_batch_size,
            "system/batch_size_per_gpu": micro_batch_size,
            "system/accumulation_steps": accumulation_steps,
            "system/global_batch_size": global_batch_size,
            "system/trainable_params_M": trainable_params / 1e6,
            "system/shift_ratio": shift_ratio,
            "system/latent_norm_enabled": int(latent_stats is not None),
            "system/latent_norm_global_mean": (
                float(latent_stats["global_mean"]) if latent_stats is not None else 0.0
            ),
            "system/latent_norm_global_std": (
                float(latent_stats["global_std"]) if latent_stats is not None else 0.0
            ),
            "system/context_dropout": context_dropout_prob,
            "system/ema_decay": ema_decay,
            "system/maintain_ema": int(maintain_ema),
            "system/sample_with_ema": int(sample_with_ema),
            "system/sample_num_steps": sample_num_steps,
            "system/video_size": video_size,
            "system/window_frames": window_frames,
            "system/frame_stride": frame_stride,
            "system/sampled_frames": sampled_frames,
            "system/encoder_temporal_patch_size": temporal_patch_size,
            "system/encoder_spatial_patch_size": spatial_patch,
        }, step=0)

    sample_prompts = list(wandb_cfg.get("sample_prompts", [
        "A robot dancing under neon lights in rainy night streets.",
        "A snowy mountain landscape with flying birds and drifting clouds.",
        "A surfer riding a giant wave during sunset, cinematic slow motion.",
    ]))

    log_non_ema_samples = bool(wandb_cfg.get("log_non_ema_samples", False))
    if args.log_non_ema_samples:
        log_non_ema_samples = True
    if log_non_ema_samples and not sample_with_ema:
        if rank == 0:
            logger.warning(
                "log_non_ema_samples only has effect when sample_with_ema=True; disabling it."
            )
        log_non_ema_samples = False

    if rank == 0:
        logger.info(
            "Sample visualization: "
            f"sample_with_ema={sample_with_ema}, "
            f"log_non_ema_samples={log_non_ema_samples}, "
            f"sample_num_steps={sample_num_steps}, sample_fps={sample_fps}"
        )

    # =====================================================================
    # Training loop
    # =====================================================================
    logger.info(
        f"Starting training: {num_epochs} epochs, micro_bs={micro_batch_size}/gpu, "
        f"accum={accumulation_steps}, global_bs={global_batch_size}, "
        f"video_size={video_size}, sampled_frames={sampled_frames}, "
        f"steps_per_epoch={estimated_steps_per_epoch}"
    )
    if rank == 0:
        logger.info(f"Checkpoint cadence: every {checkpoint_every_steps} steps.")

    barrier()

    stream_epoch = start_epoch
    train_loader = None
    data_iter = None
    consecutive_stream_failures = 0

    def rebuild_stream_loader(
        *,
        reason: str,
        advance_epoch: bool,
        failure_count: int,
    ) -> int:
        nonlocal stream_epoch, train_loader, data_iter
        if advance_epoch:
            stream_epoch += 1

        build_failures = 0
        while True:
            try:
                train_loader = make_stream_loader(stream_epoch)
                data_iter = iter(train_loader)
                return failure_count
            except Exception as build_exc:
                if not is_recoverable_stream_error(build_exc):
                    raise

                build_failures += 1
                failure_count += 1
                if max_stream_failures > 0 and failure_count > max_stream_failures:
                    raise RuntimeError(
                        "Exceeded max consecutive streaming failures "
                        f"({max_stream_failures}) while rebuilding dataloader. "
                        f"Last error: {type(build_exc).__name__}: {build_exc}"
                    ) from build_exc

                backoff_power = min(failure_count - 1, 6)
                sleep_seconds = min(
                    stream_retry_base_sleep * (2 ** backoff_power),
                    stream_retry_max_sleep,
                )
                if rank == 0:
                    max_failures_msg = "unlimited" if max_stream_failures <= 0 else str(max_stream_failures)
                    logger.warning(
                        "Recoverable streaming error "
                        f"({failure_count}/{max_failures_msg}) while {reason} "
                        f"(build retry {build_failures}): "
                        f"{type(build_exc).__name__}: {build_exc}. "
                        f"Rebuilding dataloader in {sleep_seconds:.1f}s."
                    )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                stream_epoch += 1

    consecutive_stream_failures = rebuild_stream_loader(
        reason="initializing streaming dataloader",
        advance_epoch=False,
        failure_count=consecutive_stream_failures,
    )

    def next_stream_batch():
        nonlocal stream_epoch, train_loader, data_iter, consecutive_stream_failures
        while True:
            try:
                batch = next(data_iter)
                if consecutive_stream_failures > 0 and rank == 0:
                    logger.info(
                        f"Streaming dataloader recovered after {consecutive_stream_failures} retry attempt(s)."
                    )
                consecutive_stream_failures = 0
                return batch
            except StopIteration:
                consecutive_stream_failures = rebuild_stream_loader(
                    reason="advancing streaming epoch",
                    advance_epoch=True,
                    failure_count=consecutive_stream_failures,
                )
            except Exception as exc:
                if not is_recoverable_stream_error(exc):
                    raise

                consecutive_stream_failures += 1
                if max_stream_failures > 0 and consecutive_stream_failures > max_stream_failures:
                    raise RuntimeError(
                        "Exceeded max consecutive streaming failures "
                        f"({max_stream_failures}). Last error: {type(exc).__name__}: {exc}"
                    ) from exc

                backoff_power = min(consecutive_stream_failures - 1, 6)
                sleep_seconds = min(
                    stream_retry_base_sleep * (2 ** backoff_power),
                    stream_retry_max_sleep,
                )
                if rank == 0:
                    max_failures_msg = "unlimited" if max_stream_failures <= 0 else str(max_stream_failures)
                    logger.warning(
                        "Recoverable streaming error "
                        f"({consecutive_stream_failures}/{max_failures_msg}): "
                        f"{type(exc).__name__}: {exc}. "
                        f"Rebuilding dataloader in {sleep_seconds:.1f}s."
                    )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                consecutive_stream_failures = rebuild_stream_loader(
                    reason="recovering after streaming read failure",
                    advance_epoch=True,
                    failure_count=consecutive_stream_failures,
                )

    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for _ in range(estimated_steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device, dtype=torch.float32)
            step_mse = torch.zeros((), device=device, dtype=torch.float32)
            videos = None
            texts: List[str] = []
            latent_grid: Optional[Tuple[int, int, int]] = None

            for micro_idx in range(accumulation_steps):
                batch = next_stream_batch()
                videos = batch["video"].to(device, non_blocking=True)  # (B, T, C, H, W)
                texts = batch["prompt"]

                # ----- Encode ground-truth video latent with frozen RAE -----
                with torch.no_grad():
                    with maybe_autocast(device, autocast_dtype, amp_enabled):
                        latent, _ = rae.encode(videos)
                    if latent_stats is not None:
                        latent = normalize_latent(latent, latent_stats["mean"], latent_stats["std"])
                latent_grid = (int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))

                # ----- Encode condition via MLLM + connector -----
                tokenized = model_core.mllm_encoder.tokenize(texts)
                tokenized = {
                    k: (v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in tokenized.items()
                }

                if context_dropout_prob > 0 and ddp_model.training:
                    drop_mask = torch.rand(len(texts), device=device) < context_dropout_prob
                    if drop_mask.any():
                        cfg_texts = [("" if drop else t) for t, drop in zip(texts, drop_mask)]
                        tokenized = model_core.mllm_encoder.tokenize(cfg_texts)
                        tokenized = {
                            k: (v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v)
                            for k, v in tokenized.items()
                        }

                sync_context = (
                    ddp_model.no_sync()
                    if is_ddp and micro_idx < accumulation_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with maybe_autocast(device, autocast_dtype, amp_enabled):
                        losses = ddp_model(
                            latent,
                            input_ids=tokenized["input_ids"],
                            attention_mask=tokenized["attention_mask"],
                            pixel_values=tokenized.get("pixel_values"),
                            image_grid_thw=tokenized.get("image_grid_thw"),
                            pixel_values_videos=tokenized.get("pixel_values_videos"),
                            video_grid_thw=tokenized.get("video_grid_thw"),
                            second_per_grid_ts=tokenized.get("second_per_grid_ts"),
                        )
                        loss = losses["loss"] / accumulation_steps

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                step_loss += losses["loss"].detach().to(torch.float32)
                step_mse += losses.get("mse", losses["loss"]).detach().to(torch.float32)

            # ----- Optimizer update -----
            if scaler is not None:
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_param_list, clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_param_list, clip_grad)
                optimizer.step()

            if maintain_ema and ema_state is not None:
                update_ema_state(ema_state, model_core, ema_decay)

            if scheduler is not None:
                scheduler.step()

            step_loss /= accumulation_steps
            step_mse /= accumulation_steps
            loss_val = float(step_loss.item())
            epoch_loss_sum += loss_val
            epoch_steps += 1
            global_step += 1

            if rank == 0 and global_step % checkpoint_every_steps == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"step-{global_step:010d}.pt")
                save_checkpoint(ckpt_path, global_step, epoch, ddp_model, ema_state, optimizer, scheduler)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            do_sample = (sample_every > 0 and global_step % sample_every == 0)

            if log_interval > 0 and global_step % log_interval == 0:
                reduced_stats = torch.tensor(
                    [
                        float(step_loss.item()),
                        float(step_mse.item()),
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                if is_ddp:
                    dist.all_reduce(reduced_stats, op=dist.ReduceOp.SUM)
                    reduced_stats /= world_size

                if rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    stats = {
                        "train/loss": float(reduced_stats[0].item()),
                        "train/mse": float(reduced_stats[1].item()),
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    }
                    logger.info(
                        f"[Epoch {epoch} | Step {global_step}] "
                        + ", ".join(f"{k}: {v:.6f}" for k, v in stats.items())
                    )
                    if args.wandb:
                        wandb_utils.log(stats, step=global_step, commit=not do_sample)

            # ----- Video sample visualization -----
            if do_sample and rank == 0 and args.wandb and videos is not None and latent_grid is not None:
                logger.info("Generating sample videos for visualization...")
                swapped_to_ema = False
                non_ema_videos: Optional[torch.Tensor] = None
                try:
                    vis_n = min(3, int(videos.size(0)), len(texts))
                    vis_prompts = texts[:vis_n] if vis_n > 0 else sample_prompts[:3]
                    real_videos = videos[:vis_n].detach().to(torch.float32).clamp(0, 1) if vis_n > 0 else None
                    grid_t, grid_h, grid_w = latent_grid

                    # Optional non-EMA visualization first.
                    if sample_with_ema and log_non_ema_samples:
                        non_ema_videos = generate_sample_videos(
                            model=model_core,
                            rae=rae,
                            prompts=vis_prompts,
                            grid_t=grid_t,
                            grid_h=grid_h,
                            grid_w=grid_w,
                            device=device,
                            dtype=autocast_dtype,
                            amp_enabled=amp_enabled,
                            num_steps=sample_num_steps,
                            latent_stats=latent_stats,
                        )
                        if non_ema_videos.numel() > 0:
                            wandb_utils.log_video(
                                make_video_panel(non_ema_videos),
                                step=global_step,
                                key="samples/generated_non_ema",
                                fps=sample_fps,
                                commit=False,
                            )
                            if real_videos is not None:
                                wandb_utils.log_video(
                                    make_video_compare_panel(real_videos, non_ema_videos),
                                    step=global_step,
                                    key="samples/real_top_generated_bottom_non_ema",
                                    caption="Top: real videos; bottom: generated videos (non-EMA)",
                                    fps=sample_fps,
                                    commit=False,
                                )

                    if sample_with_ema and ema_state is not None:
                        swap_model_with_ema(model_core, ema_state)
                        swapped_to_ema = True

                    gen_videos = generate_sample_videos(
                        model=model_core,
                        rae=rae,
                        prompts=vis_prompts,
                        grid_t=grid_t,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        device=device,
                        dtype=autocast_dtype,
                        amp_enabled=amp_enabled,
                        num_steps=sample_num_steps,
                        latent_stats=latent_stats,
                    )

                    if gen_videos.numel() > 0:
                        primary_label = "EMA" if sample_with_ema else "non-EMA"
                        wandb_utils.log_video(
                            make_video_panel(gen_videos),
                            step=global_step,
                            key="samples/generated",
                            caption=f"Generated videos ({primary_label})",
                            fps=sample_fps,
                            commit=False,
                        )

                        if real_videos is not None:
                            wandb_utils.log_video(
                                make_video_compare_panel(real_videos, gen_videos),
                                step=global_step,
                                key="samples/real_top_generated_bottom",
                                caption=f"Top: real videos; bottom: generated videos ({primary_label})",
                                fps=sample_fps,
                                commit=True,
                            )
                        else:
                            wandb_utils.log_video(
                                make_video_panel(gen_videos),
                                step=global_step,
                                key="samples/real_top_generated_bottom",
                                caption=f"Generated videos ({primary_label})",
                                fps=sample_fps,
                                commit=True,
                            )

                        if (
                            sample_with_ema
                            and non_ema_videos is not None
                            and non_ema_videos.numel() > 0
                            and non_ema_videos.shape == gen_videos.shape
                        ):
                            wandb_utils.log_video(
                                make_video_compare_panel(non_ema_videos, gen_videos),
                                step=global_step,
                                key="samples/non_ema_top_ema_bottom",
                                caption="Top: non-EMA generated videos; bottom: EMA generated videos",
                                fps=sample_fps,
                                commit=False,
                            )
                    logger.info("Video sample generation done.")
                except Exception as e:
                    logger.warning(f"Video sample generation failed: {e}")
                finally:
                    if swapped_to_ema and ema_state is not None:
                        swap_model_with_ema(model_core, ema_state)
                    ddp_model.train()

        # ----- Epoch summary -----
        epoch_totals = torch.tensor(
            [float(epoch_loss_sum), float(epoch_steps)],
            device=device,
            dtype=torch.float64,
        )
        if is_ddp:
            dist.all_reduce(epoch_totals, op=dist.ReduceOp.SUM)

        global_epoch_loss_sum = float(epoch_totals[0].item())
        global_epoch_steps = int(epoch_totals[1].item())
        avg_loss = global_epoch_loss_sum / max(1, global_epoch_steps)

        if rank == 0 and global_epoch_steps > 0:
            per_rank_steps = global_epoch_steps // max(1, world_size)
            epoch_stats = {
                "epoch/loss": avg_loss,
                "epoch/steps": per_rank_steps,
                "epoch/number": epoch,
            }
            logger.info(f"[Epoch {epoch}] avg_loss: {avg_loss:.6f}, steps: {per_rank_steps}")
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)

    # ----- Final checkpoint -----
    if rank == 0:
        ckpt_path = os.path.join(checkpoint_dir, "ep-last.pt")
        save_checkpoint(ckpt_path, global_step, num_epochs, ddp_model, ema_state, optimizer, scheduler)
        logger.info(f"Saved final checkpoint: {ckpt_path}")

    if args.wandb and rank == 0:
        wandb_utils.finish()

    barrier()
    logger.info("Training complete.")
    cleanup_distributed()


if __name__ == "__main__":
    main()
