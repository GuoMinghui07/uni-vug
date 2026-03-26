#!/usr/bin/env python3
"""
Stage-2 DiT training script.

Trains: DiT backbone + MLLM meta query embeddings + MLLMConnector
Frozen: MLLM backbone (Qwen3-VL), RAE encoder, RAE decoder

Data flow:
    Image → RAE.encode() → latent (B, T, H, W, 1024)  [frozen, ground truth]
    Text+Image → MLLM(meta queries) → Connector → context (B, N_q, 1536)  [trainable]
    latent + noise → DiT → velocity prediction → flow matching loss
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency at import time
    httpx = None  # type: ignore[assignment]

try:
    import httpcore
except Exception:  # pragma: no cover - optional dependency at import time
    httpcore = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DiT import UniDiT
from qwen3_vl_rae_uncompressed import Qwen3VLRAE

from utils.dist_utils import setup_distributed, cleanup_distributed, is_main_process, barrier
from utils.model_utils import load_rae_decoder_weights
from utils.optim_utils import build_optimizer, build_scheduler, get_autocast_scaler
from utils.train_utils import (
    DEFAULT_MIXED_DATASETS,
    DEFAULT_STREAMING_IMAGE_KEYS,
    DEFAULT_STREAMING_TEXT_KEYS,
    build_streaming_dataloader,
    configure_experiment_dirs,
)
from utils.resume_utils import find_resume_checkpoint, load_training_checkpoint, save_worktree
from utils import wandb_utils


# =====================================================================
# Constants
# =====================================================================

UNIVIDEO_T2I_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn "
    "faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, walking backwards, computer-generated environment, weak dynamics, "
    "distorted and erratic motions, unstable framing and a disorganized composition."
)


# =====================================================================
# Noise schedule shift computation
# =====================================================================

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
    """Heuristic guard for transient streaming failures from remote webdataset shards."""
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
    )
    return any(token in message for token in recoverable_tokens)


def summarize_mixed_datasets(
    datasets: Sequence[Any],
) -> List[Tuple[str, str, float, float]]:
    """Normalize mixed dataset config into (name, url, weight, ratio)."""
    parsed: List[Tuple[str, str, float]] = []
    for idx, item in enumerate(datasets):
        if isinstance(item, Mapping):
            name = str(item.get("name", f"dataset_{idx}")).strip() or f"dataset_{idx}"
            url = str(item.get("url", "")).strip()
            weight = float(item.get("weight", 1.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            name = str(item[0]).strip() or f"dataset_{idx}"
            url = str(item[1]).strip()
            weight = float(item[2])
        else:
            raise ValueError(
                "Each mixed dataset entry must be {'name','url','weight'} or (name, url, weight)."
            )

        if not url:
            raise ValueError(f"Dataset[{idx}] missing url.")
        if weight <= 0:
            raise ValueError(f"Dataset[{idx}] weight must be > 0, got {weight}.")
        parsed.append((name, url, weight))

    if not parsed:
        raise ValueError("At least one mixed dataset must be configured.")

    total_weight = sum(weight for _, _, weight in parsed)
    return [
        (name, url, weight, weight / total_weight)
        for name, url, weight in parsed
    ]


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


# =====================================================================
# Checkpoint save / load
# =====================================================================

def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_state: Optional[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
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


# =====================================================================
# Sample generation for visualization
# =====================================================================

@torch.no_grad()
def generate_sample_images(
    model: UniDiT,
    rae: Qwen3VLRAE,
    prompts: List[str],
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 50,
    guidance_scale: float = 7.0,
    negative_prompt: str = UNIVIDEO_T2I_NEGATIVE_PROMPT,
    latent_stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Generate images from text prompts for WandB visualization.

    Returns:
        Tensor (N, C, H, W) in [0, 1], one sample per prompt.
    """
    model.eval()

    all_images = []
    for prompt in prompts:
        # Build inputs for MLLM encoder (text-only, no image/video condition)
        tokenized = model.mllm_encoder.tokenize([prompt])
        tokenized = {
            k: (v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in tokenized.items()
        }
        negative_tokenized: Optional[Dict[str, Any]] = None
        if guidance_scale > 1.0:
            negative_tokenized = model.mllm_encoder.tokenize([negative_prompt])
            negative_tokenized = {
                k: (v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in negative_tokenized.items()
            }

        # Encode condition
        with torch.amp.autocast("cuda", dtype=dtype):
            context, context_mask = model.encode_condition(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                pixel_values=tokenized.get("pixel_values"),
                image_grid_thw=tokenized.get("image_grid_thw"),
                pixel_values_videos=tokenized.get("pixel_values_videos"),
                video_grid_thw=tokenized.get("video_grid_thw"),
                second_per_grid_ts=tokenized.get("second_per_grid_ts"),
            )
            negative_context = None
            negative_context_lens = None
            if negative_tokenized is not None:
                negative_context, negative_context_mask = model.encode_condition(
                    input_ids=negative_tokenized["input_ids"],
                    attention_mask=negative_tokenized["attention_mask"],
                    pixel_values=negative_tokenized.get("pixel_values"),
                    image_grid_thw=negative_tokenized.get("image_grid_thw"),
                    pixel_values_videos=negative_tokenized.get("pixel_values_videos"),
                    video_grid_thw=negative_tokenized.get("video_grid_thw"),
                    second_per_grid_ts=negative_tokenized.get("second_per_grid_ts"),
                )
                negative_context_lens = negative_context_mask.sum(dim=1)

        # Compute RAE latent grid dimensions for target resolution
        # For 448×448: grid_h = 448/14 = 32, grid_w = 32, grid_t = 1
        patch_size = rae.patch_size
        grid_h = image_size // patch_size
        grid_w = image_size // patch_size
        grid_t = 1  # single image

        # Generate latent via flow matching sampling
        with torch.amp.autocast("cuda", dtype=dtype):
            latent = model.sample(
                context=context,
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                context_lens=context_mask.sum(dim=1),
                num_steps=num_steps,
                negative_context=negative_context,
                negative_context_lens=negative_context_lens,
                guidance_scale=guidance_scale,
            )
        if latent_stats is not None:
            latent = denormalize_latent(latent, latent_stats["mean"], latent_stats["std"])

        # Decode latent to pixels using RAE decoder
        with torch.amp.autocast("cuda", dtype=dtype):
            pixels = rae.decode(latent)  # (B, T, C, H, W) or (B, C, H, W)

        if pixels.dim() == 5:
            pixels = pixels[:, 0]  # take first frame
        pixels = pixels.clamp(0, 1)
        all_images.append(pixels[0])  # single sample per prompt

    if not all_images:
        model.train()
        return torch.empty(0, 3, image_size, image_size, device=device, dtype=torch.float32)

    out = torch.stack(all_images)
    model.train()
    return out


@torch.no_grad()
def generate_samples(
    model: UniDiT,
    rae: Qwen3VLRAE,
    prompts: List[str],
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 50,
    guidance_scale: float = 7.0,
    negative_prompt: str = UNIVIDEO_T2I_NEGATIVE_PROMPT,
    latent_stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Generate a visualization grid for text prompts."""
    images = generate_sample_images(
        model=model,
        rae=rae,
        prompts=prompts,
        image_size=image_size,
        device=device,
        dtype=dtype,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        latent_stats=latent_stats,
    )
    if images.numel() == 0:
        return torch.zeros(3, image_size, image_size, device=device, dtype=torch.float32)
    return make_grid(images, nrow=images.size(0))


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 DiT training.")
    parser.add_argument("--config", type=str, default="config/dit_training.yaml", help="YAML config path.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Output directory.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in experiment dir.")
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Resume from an explicit checkpoint path. Overrides --resume auto-discovery.",
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
            "Also log non-EMA samples at visualization steps for side-by-side comparison. "
            "EMA samples remain logged to the original keys."
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


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    if args.resume_reset_progress and not args.resume_model_only:
        raise ValueError("--resume-reset-progress requires --resume-model-only.")

    # --- Distributed setup ---
    rank, world_size, device = setup_distributed()

    # --- Load config ---
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

    # Training hyperparameters
    num_epochs = int(train_cfg.get("epochs", 100))
    has_micro_batch_cfg = ("micro_batch_size_per_gpu" in train_cfg) or ("micro_batch_size" in train_cfg)
    if "accumulation_steps" in train_cfg:
        accumulation_steps = int(train_cfg.get("accumulation_steps"))
    elif has_micro_batch_cfg:
        accumulation_steps = 3
    else:
        accumulation_steps = 1
    if accumulation_steps <= 0:
        raise ValueError("training.accumulation_steps must be > 0.")

    configured_global_batch_size = train_cfg.get("global_batch_size", None)
    if has_micro_batch_cfg:
        micro_batch_size = int(train_cfg.get("micro_batch_size_per_gpu", train_cfg.get("micro_batch_size")))
    elif configured_global_batch_size is not None:
        configured_global_batch_size = int(configured_global_batch_size)
        denom = world_size * accumulation_steps
        if configured_global_batch_size % denom != 0:
            raise ValueError(
                "training.global_batch_size must be divisible by "
                "world_size * training.accumulation_steps."
            )
        micro_batch_size = configured_global_batch_size // denom
    else:
        micro_batch_size = 24

    if micro_batch_size <= 0:
        raise ValueError("training.micro_batch_size_per_gpu must be > 0.")

    global_batch_size = micro_batch_size * accumulation_steps * world_size
    if configured_global_batch_size is not None and int(configured_global_batch_size) != global_batch_size:
        raise ValueError(
            "training.global_batch_size does not match "
            "micro_batch_size_per_gpu * accumulation_steps * world_size."
        )

    num_workers = int(train_cfg.get("num_workers", 16))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 4))
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    if clip_grad <= 0:
        clip_grad = None
    log_interval = int(train_cfg.get("log_interval", 100))
    sample_every = int(train_cfg.get("sample_every", 1000))
    checkpoint_every_steps = int(train_cfg.get("checkpoint_every_steps", 10000))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError("training.ema_decay must satisfy 0 <= ema_decay < 1.")
    maintain_ema_cfg = bool(train_cfg.get("maintain_ema", True))
    maintain_ema = maintain_ema_cfg if args.maintain_ema is None else bool(args.maintain_ema)
    resume_init_ema_from_model = bool(
        train_cfg.get("resume_init_ema_from_model", False)
    )
    sample_with_ema_cfg = bool(wandb_cfg.get("sample_with_ema", False))
    sample_with_ema = sample_with_ema_cfg if args.sample_with_ema is None else bool(args.sample_with_ema)
    sample_num_steps = int(wandb_cfg.get("sample_num_steps", 50))
    sample_guidance_scale = float(wandb_cfg.get("sample_guidance_scale", 7.0))
    sample_negative_prompt = str(wandb_cfg.get("sample_negative_prompt", UNIVIDEO_T2I_NEGATIVE_PROMPT))
    if sample_num_steps <= 0:
        raise ValueError("wandb.sample_num_steps must be > 0.")
    if sample_guidance_scale < 1.0:
        raise ValueError("wandb.sample_guidance_scale must be >= 1.0.")
    if checkpoint_every_steps <= 0:
        raise ValueError("training.checkpoint_every_steps must be > 0.")
    precision = getattr(args, "precision", train_cfg.get("precision", "bf16"))
    args.precision = precision

    # Seed
    default_seed = int(train_cfg.get("global_seed", 42))
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Experiment dirs
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)

    # --- Noise schedule shift ---
    image_size = int(data_cfg.get("image_size", 448))
    use_shift = bool(shift_cfg.get("enabled", False))
    base_dim = int(shift_cfg.get("base_dim", 4096))
    shift_ratio = 1.0

    # --- CFG dropout ---
    legacy_context_dropout = cfg_section.get("context_dropout_prob", None)
    uncondition_dropout_prob = float(
        cfg_section.get(
            "uncondition_dropout_prob",
            (0.1 if legacy_context_dropout is None else legacy_context_dropout),
        )
    )
    if not (0.0 <= uncondition_dropout_prob <= 1.0):
        raise ValueError("cfg.uncondition_dropout_prob must satisfy 0 <= p <= 1.")

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

    # Apply noise schedule shift (ratio is calibrated below from actual latent shape)
    model.flow_matching.use_schedule_shift = use_shift
    model.flow_matching.shift_ratio = shift_ratio

    # --- Build RAE (frozen) ---
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

    # Calibrate schedule shift from actual latent shape to avoid hard-coded assumptions.
    if use_shift:
        with torch.no_grad():
            probe = torch.zeros(
                1,
                int(rae_cfg.get("in_channels", 3)),
                image_size,
                image_size,
                device=device,
            )
            latent_probe, _ = rae.encode(probe)
        shift_ratio = compute_shift_ratio_from_latent(latent_probe, base_dim=base_dim)
        model.flow_matching.shift_ratio = shift_ratio

    # MLLM freeze policy is handled inside MLLMEncoder:
    # full backbone frozen, only new query-token embedding rows trainable.

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"UniDiT trainable: {trainable_params / 1e6:.2f}M / total: {total_params / 1e6:.2f}M")
    logger.info(f"Noise schedule shift: enabled={use_shift}, ratio={shift_ratio:.6f}")
    if latent_stats is not None:
        logger.info(
            "Latent normalization: enabled=True, "
            f"path='{latent_stats['path']}', global_mean={latent_stats['global_mean']:.6f}, "
            f"global_std={latent_stats['global_std']:.6f}, token_count={latent_stats['token_count']}"
        )
    else:
        logger.info("Latent normalization: enabled=False")
    logger.info(f"CFG uncondition dropout (null embedding): {uncondition_dropout_prob}")
    if maintain_ema:
        logger.info(f"EMA decay: {ema_decay}")
    else:
        logger.info("EMA maintenance: disabled (no EMA weights tracked during training).")
    if sample_with_ema and not maintain_ema:
        logger.warning(
            "sample_with_ema=True but EMA maintenance is disabled; forcing sample_with_ema=False."
        )
        sample_with_ema = False

    # --- DDP ---
    ddp_model = DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    # --- Optimizer & scheduler ---
    trainable_named_params = [(n, p) for n, p in ddp_model.named_parameters() if p.requires_grad]
    trainable_param_list = [p for _, p in trainable_named_params]
    ema_state: Optional[Dict[str, torch.Tensor]] = (
        init_ema_state(ddp_model.module) if maintain_ema else None
    )

    # Keep trainable query token vectors out of weight decay.
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
        logger.info(
            "Optimizer override: set weight_decay=0.0 for trainable query-token embeddings."
        )
    else:
        optimizer, optim_msg = build_optimizer(trainable_param_list, train_cfg)
        logger.info(optim_msg)

    # AMP
    scaler, autocast_kwargs = get_autocast_scaler(args)
    autocast_dtype = autocast_kwargs.get("dtype", torch.bfloat16)

    # --- Data loader ---
    logger.info("Configuring streaming data loader...")
    data_split = str(data_cfg.get("split", "train"))
    mixed_datasets = data_cfg.get("datasets", list(DEFAULT_MIXED_DATASETS))
    text_keys = list(data_cfg.get("text_keys", list(DEFAULT_STREAMING_TEXT_KEYS)))
    image_keys = list(data_cfg.get("image_keys", list(DEFAULT_STREAMING_IMAGE_KEYS)))
    shuffle_buffer_size = int(data_cfg.get("shuffle_buffer_size", 2000))
    loader_timeout = int(data_cfg.get("loader_timeout", 180))
    max_errors_per_worker = int(data_cfg.get("max_errors_per_worker", 5000))
    curl_cfg = data_cfg.get("curl", {})
    curl_connect_timeout = int(curl_cfg.get("connect_timeout", 10))
    curl_max_time = int(curl_cfg.get("max_time", 120))
    curl_retry = int(curl_cfg.get("retry", 15))
    curl_speed_time = int(curl_cfg.get("speed_time", 30))
    curl_speed_limit = int(curl_cfg.get("speed_limit", 1024))
    curl_show_errors = bool(curl_cfg.get("show_errors", False))
    stream_retry_cfg = data_cfg.get("stream_retry", {})
    max_stream_failures = int(stream_retry_cfg.get("max_consecutive_failures", 0))
    stream_retry_base_sleep = float(stream_retry_cfg.get("base_sleep_seconds", 1.0))
    stream_retry_max_sleep = float(stream_retry_cfg.get("max_sleep_seconds", 30.0))
    rebuild_after_consecutive_failures = int(
        stream_retry_cfg.get("rebuild_after_consecutive_failures", 1)
    )
    if data_split != "train":
        raise ValueError("Mixed webdataset loader currently supports data.split='train' only.")
    if loader_timeout < 0:
        raise ValueError("data.loader_timeout must be >= 0.")
    if max_errors_per_worker <= 0:
        raise ValueError("data.max_errors_per_worker must be > 0.")
    if stream_retry_base_sleep < 0:
        raise ValueError("data.stream_retry.base_sleep_seconds must be >= 0.")
    if stream_retry_max_sleep < 0:
        raise ValueError("data.stream_retry.max_sleep_seconds must be >= 0.")
    if rebuild_after_consecutive_failures < 0:
        raise ValueError(
            "data.stream_retry.rebuild_after_consecutive_failures must be >= 0."
        )
    if stream_retry_max_sleep < stream_retry_base_sleep:
        stream_retry_max_sleep = stream_retry_base_sleep
    if rank == 0:
        mixed_dataset_plan = summarize_mixed_datasets(mixed_datasets)
        max_failures_msg = "unlimited" if max_stream_failures <= 0 else str(max_stream_failures)
        rebuild_policy_msg = (
            "disabled"
            if rebuild_after_consecutive_failures == 0
            else f"{rebuild_after_consecutive_failures}"
        )
        logger.info(
            "Streaming retry policy: "
            f"max_consecutive_failures={max_failures_msg}, "
            f"rebuild_after_consecutive_failures={rebuild_policy_msg}, "
            f"base_sleep={stream_retry_base_sleep:.1f}s, "
            f"max_sleep={stream_retry_max_sleep:.1f}s."
        )
        logger.info(
            "Loader config: "
            f"num_workers={num_workers}, prefetch_factor={prefetch_factor}, "
            f"shuffle_buffer_size={shuffle_buffer_size}, loader_timeout={loader_timeout}s, "
            f"curl(connect_timeout={curl_connect_timeout}, max_time={curl_max_time}, "
            f"retry={curl_retry}, speed_time={curl_speed_time}, speed_limit={curl_speed_limit})"
        )
        logger.info(
            "Mixed dataset plan (configured weight -> normalized ratio):"
        )
        for idx, (name, url, weight, ratio) in enumerate(mixed_dataset_plan, start=1):
            logger.info(
                f"  [{idx}] {name}: weight={weight:g}, ratio={ratio * 100.0:.2f}%, url={url}"
            )

    def make_stream_loader(stream_epoch: int):
        return build_streaming_dataloader(
            datasets=mixed_datasets,
            split=data_split,
            image_size=image_size,
            batch_size=micro_batch_size,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            seed=global_seed,
            epoch=stream_epoch,
            shuffle_buffer_size=shuffle_buffer_size,
            prefetch_factor=prefetch_factor,
            text_keys=text_keys,
            image_keys=image_keys,
            loader_timeout=loader_timeout,
            max_errors_per_worker=max_errors_per_worker,
            curl_connect_timeout=curl_connect_timeout,
            curl_max_time=curl_max_time,
            curl_retry=curl_retry,
            curl_speed_time=curl_speed_time,
            curl_speed_limit=curl_speed_limit,
            curl_show_errors=curl_show_errors,
        )

    # Scheduler/loop length for streaming training
    if "steps_per_epoch" not in train_cfg:
        raise ValueError("training.steps_per_epoch must be set for streaming training.")
    estimated_steps_per_epoch = int(train_cfg.get("steps_per_epoch"))
    if estimated_steps_per_epoch <= 0:
        raise ValueError("training.steps_per_epoch must be > 0.")
    scheduler: Optional[LambdaLR] = None
    sched_msg = None
    if train_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, estimated_steps_per_epoch, train_cfg)
        logger.info(sched_msg)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    resumed = False
    resume_path: Optional[str] = None
    if args.resume_path is not None:
        explicit_resume = Path(args.resume_path)
        if not explicit_resume.exists():
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
        start_epoch, global_step = load_training_checkpoint(
            resume_path,
            ddp_model,
            load_optimizer,
            load_scheduler,
            ema_state=ema_state,
        )
        if maintain_ema and ema_state is not None and resume_init_ema_from_model:
            ema_state.clear()
            ema_state.update(init_ema_state(ddp_model.module))
            logger.info(
                "Resume EMA policy: reinitialized EMA from resumed model weights; "
                "EMA updates start from this run."
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
                    # Align fresh scheduler to resumed progress while honoring current config lr/base_lr.
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
            project=str(wandb_cfg.get("project", "uni-vug-dit")),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # Log system params
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
            "system/context_dropout": uncondition_dropout_prob,  # backward-compatible key
            "system/uncondition_dropout": uncondition_dropout_prob,
            "system/ema_decay": ema_decay,
            "system/maintain_ema": int(maintain_ema),
            "system/resume_init_ema_from_model": int(resume_init_ema_from_model),
            "system/sample_with_ema": int(sample_with_ema),
            "system/sample_num_steps": sample_num_steps,
            "system/sample_guidance_scale": sample_guidance_scale,
        }, step=0)

    sample_prompts = list(wandb_cfg.get("sample_prompts", [
        "A golden retriever puppy playing in a field of wildflowers under a blue sky.",
        "A futuristic cityscape at sunset with flying cars and neon signs.",
        "A cozy cabin in a snowy mountain landscape with smoke coming from the chimney.",
    ]))
    log_non_ema_samples = bool(wandb_cfg.get("log_non_ema_samples", False))
    if args.log_non_ema_samples:
        log_non_ema_samples = True
    if log_non_ema_samples and not sample_with_ema:
        if rank == 0:
            logger.warning(
                "log_non_ema_samples is only meaningful when sample_with_ema=True; disabling extra non-EMA sample logging."
            )
        log_non_ema_samples = False
    if rank == 0:
        logger.info(
            "Sample visualization: "
            f"sample_with_ema={sample_with_ema}, "
            f"log_non_ema_samples={log_non_ema_samples}, "
            f"sample_num_steps={sample_num_steps}, "
            f"sample_guidance_scale={sample_guidance_scale}"
        )

    # =====================================================================
    # Training loop
    # =====================================================================
    logger.info(
        f"Starting training: {num_epochs} epochs, micro_bs={micro_batch_size}/gpu, "
        f"accum={accumulation_steps}, global_bs={global_batch_size}, "
        f"image_size={image_size}, steps_per_epoch={estimated_steps_per_epoch}"
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
                if rank == 0:
                    logger.info(f"Building stream loader (stream_epoch={stream_epoch})...")
                train_loader = make_stream_loader(stream_epoch)
                if rank == 0:
                    logger.info("Creating stream iterator...")
                data_iter = iter(train_loader)
                if rank == 0:
                    logger.info("Stream iterator is ready.")
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
                # Move to a fresh shuffle stream to avoid repeatedly failing on the same
                # leading shard sequence during loader initialization.
                stream_epoch += 1

    consecutive_stream_failures = rebuild_stream_loader(
        reason="initializing streaming dataloader",
        advance_epoch=False,
        failure_count=consecutive_stream_failures,
    )

    def sync_stream_status(local_status: int) -> int:
        if not dist.is_initialized():
            return int(local_status)
        status = torch.tensor([int(local_status)], device=device, dtype=torch.int32)
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        return int(status.item())

    def next_stream_batch():
        nonlocal stream_epoch, train_loader, data_iter, consecutive_stream_failures
        while True:
            batch = None
            local_status = 0  # 0=ok, 1=StopIteration, 2=recoverable_error
            local_exc: Optional[Exception] = None
            try:
                batch = next(data_iter)
            except StopIteration:
                local_status = 1
            except Exception as exc:
                if not is_recoverable_stream_error(exc):
                    raise
                local_status = 2
                local_exc = exc

            global_status = sync_stream_status(local_status)

            if global_status == 0:
                if consecutive_stream_failures > 0 and rank == 0:
                    logger.info(
                        f"Streaming dataloader recovered after {consecutive_stream_failures} retry attempt(s)."
                    )
                consecutive_stream_failures = 0
                return batch

            if global_status == 1:
                consecutive_stream_failures = rebuild_stream_loader(
                    reason="advancing streaming epoch",
                    advance_epoch=True,
                    failure_count=consecutive_stream_failures,
                )
                continue

            # Any rank hit a recoverable streaming failure: recover in lockstep.
            if global_status != 2:
                raise RuntimeError(f"Unexpected synchronized stream status: {global_status}")

            if local_exc is None:
                local_exc = RuntimeError("peer rank reported a recoverable streaming failure")

            if rank == 0 and local_status == 0:
                logger.warning(
                    "Discarding a locally fetched batch to keep DDP ranks synchronized "
                    "after a peer streaming failure."
                )

            consecutive_stream_failures += 1
            if max_stream_failures > 0 and consecutive_stream_failures > max_stream_failures:
                raise RuntimeError(
                    "Exceeded max consecutive streaming failures "
                    f"({max_stream_failures}). Last error: {type(local_exc).__name__}: {local_exc}"
                ) from local_exc

            should_rebuild = (
                rebuild_after_consecutive_failures > 0
                and consecutive_stream_failures >= rebuild_after_consecutive_failures
            )
            if not should_rebuild:
                if rank == 0:
                    rebuild_policy_msg = (
                        "disabled"
                        if rebuild_after_consecutive_failures == 0
                        else f"threshold={rebuild_after_consecutive_failures}"
                    )
                    logger.warning(
                        "Recoverable streaming error "
                        f"({consecutive_stream_failures}/"
                        f"{'unlimited' if max_stream_failures <= 0 else max_stream_failures}): "
                        f"{type(local_exc).__name__}: {local_exc}. "
                        f"Skipping this batch without rebuilding ({rebuild_policy_msg})."
                    )
                continue

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
                    f"{type(local_exc).__name__}: {local_exc}. "
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
            step_start_time = time.perf_counter()
            step_data_wait = 0.0
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device, dtype=torch.float32)
            step_mse = torch.zeros((), device=device, dtype=torch.float32)
            images = None
            texts: List[str] = []

            for micro_idx in range(accumulation_steps):
                data_wait_start = time.perf_counter()
                batch = next_stream_batch()
                step_data_wait += time.perf_counter() - data_wait_start
                if isinstance(batch, dict):
                    images = batch["image"]
                    texts = batch["prompt"]
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, texts = batch[0], batch[1]
                else:
                    raise TypeError(f"Unexpected batch type: {type(batch)}")
                images = images.to(device, non_blocking=True)
                texts = list(texts)

                # ----- Encode ground truth latent with frozen RAE -----
                with torch.no_grad():
                    with torch.amp.autocast("cuda", **autocast_kwargs):
                        latent, _ = rae.encode(images)
                    if latent_stats is not None:
                        latent = normalize_latent(latent, latent_stats["mean"], latent_stats["std"])

                # ----- Encode condition via MLLM + connector -----
                tokenized = ddp_model.module.mllm_encoder.tokenize(texts)
                tokenized = {
                    k: (v.to(device=device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in tokenized.items()
                }

                sync_context = (
                    ddp_model.no_sync()
                    if dist.is_initialized() and micro_idx < accumulation_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with torch.amp.autocast("cuda", **autocast_kwargs):
                        context, context_mask = ddp_model.module.encode_condition(
                            input_ids=tokenized["input_ids"],
                            attention_mask=tokenized["attention_mask"],
                            pixel_values=tokenized.get("pixel_values"),
                            image_grid_thw=tokenized.get("image_grid_thw"),
                            pixel_values_videos=tokenized.get("pixel_values_videos"),
                            video_grid_thw=tokenized.get("video_grid_thw"),
                            second_per_grid_ts=tokenized.get("second_per_grid_ts"),
                        )
                        if uncondition_dropout_prob > 0 and ddp_model.training:
                            drop_mask = torch.rand(context.size(0), device=device) < uncondition_dropout_prob
                            if drop_mask.any():
                                null_tokenized = ddp_model.module.mllm_encoder.tokenize_null(
                                    batch_size=1,
                                    add_queries=True,
                                )
                                null_tokenized = {
                                    k: (
                                        v.to(device=device, non_blocking=True)
                                        if torch.is_tensor(v)
                                        else v
                                    )
                                    for k, v in null_tokenized.items()
                                }
                                null_context, null_context_mask = ddp_model.module.encode_condition(
                                    input_ids=null_tokenized["input_ids"],
                                    attention_mask=null_tokenized["attention_mask"],
                                    pixel_values=null_tokenized.get("pixel_values"),
                                    image_grid_thw=null_tokenized.get("image_grid_thw"),
                                    pixel_values_videos=null_tokenized.get("pixel_values_videos"),
                                    video_grid_thw=null_tokenized.get("video_grid_thw"),
                                    second_per_grid_ts=null_tokenized.get("second_per_grid_ts"),
                                )
                                null_context = null_context.expand(context.size(0), -1, -1)
                                null_context_mask = null_context_mask.expand(context_mask.size(0), -1)
                                context = torch.where(
                                    drop_mask[:, None, None],
                                    null_context,
                                    context,
                                )
                                context_mask = torch.where(
                                    drop_mask[:, None],
                                    null_context_mask,
                                    context_mask,
                                )

                        losses = ddp_model(
                            latent,
                            context=context,
                            context_lens=context_mask.sum(dim=1),
                        )
                        loss = losses["loss"] / accumulation_steps

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                step_loss += losses["loss"].detach().to(torch.float32)
                step_mse += losses.get("mse", losses["loss"]).detach().to(torch.float32)

            # Backward + optimizer update (one optimizer.step == one global step)
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
                update_ema_state(ema_state, ddp_model.module, ema_decay)

            if scheduler is not None:
                scheduler.step()

            step_loss /= accumulation_steps
            step_mse /= accumulation_steps
            step_wall_time = time.perf_counter() - step_start_time
            loss_val = float(step_loss.item())
            epoch_loss_sum += loss_val
            epoch_steps += 1
            global_step += 1

            if rank == 0 and global_step % checkpoint_every_steps == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"step-{global_step:010d}.pt")
                save_checkpoint(ckpt_path, global_step, epoch, ddp_model, ema_state, optimizer, scheduler)
                logger.info(f"Saved checkpoint: {ckpt_path}")

            # ----- Logging -----
            do_sample = (sample_every > 0 and global_step % sample_every == 0)

            if log_interval > 0 and global_step % log_interval == 0:
                reduced_stats = torch.tensor(
                    [
                        float(step_loss.item()),
                        float(step_mse.item()),
                        float(step_data_wait),
                        float(step_wall_time),
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                if dist.is_initialized():
                    dist.all_reduce(reduced_stats, op=dist.ReduceOp.SUM)
                    reduced_stats /= world_size

                if rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    data_wait_sec = float(reduced_stats[2].item())
                    step_wall_sec = float(reduced_stats[3].item())
                    stats = {
                        "train/loss": float(reduced_stats[0].item()),
                        "train/mse": float(reduced_stats[1].item()),
                        "train/data_wait_sec": data_wait_sec,
                        "train/step_wall_sec": step_wall_sec,
                        "train/data_wait_ratio": data_wait_sec / max(step_wall_sec, 1e-8),
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

            # ----- Sample visualization -----
            if do_sample and rank == 0 and args.wandb and images is not None:
                logger.info("Generating samples for visualization...")
                swapped_to_ema = False
                non_ema_images: Optional[torch.Tensor] = None
                primary_images: Optional[torch.Tensor] = None
                real_images: Optional[torch.Tensor] = None
                try:
                    num_vis = min(3, int(images.size(0)), len(texts))
                    vis_prompts = texts[:num_vis] if num_vis > 0 else sample_prompts[:3]

                    if sample_with_ema and log_non_ema_samples:
                        try:
                            non_ema_images = generate_sample_images(
                                model=ddp_model.module,
                                rae=rae,
                                prompts=vis_prompts,
                                image_size=image_size,
                                device=device,
                                dtype=autocast_dtype,
                                num_steps=sample_num_steps,
                                guidance_scale=sample_guidance_scale,
                                negative_prompt=sample_negative_prompt,
                                latent_stats=latent_stats,
                            )
                            if non_ema_images.numel() > 0:
                                non_ema_ncol = int(non_ema_images.size(0))
                                non_ema_grid = make_grid(non_ema_images, nrow=non_ema_ncol)
                                wandb_utils.log_image(
                                    non_ema_grid,
                                    step=global_step,
                                    key="samples/generated_non_ema",
                                    commit=False,
                                )

                                if num_vis > 0 and non_ema_ncol == num_vis:
                                    real_images = images[:num_vis].detach().to(
                                        device=non_ema_images.device,
                                        dtype=non_ema_images.dtype,
                                    ).clamp(0, 1)
                                    compare_non_ema = make_grid(
                                        torch.cat([real_images, non_ema_images], dim=0),
                                        nrow=num_vis,
                                    )
                                    wandb_utils.log_image(
                                        compare_non_ema,
                                        step=global_step,
                                        key="samples/real_top_generated_bottom_non_ema",
                                        caption="Top row: real images; bottom row: generated images (non-EMA)",
                                        commit=False,
                                    )
                        except Exception as non_ema_exc:
                            logger.warning(f"Non-EMA sample generation failed: {non_ema_exc}")

                    if sample_with_ema and ema_state is not None:
                        swap_model_with_ema(ddp_model.module, ema_state)
                        swapped_to_ema = True

                    primary_images = generate_sample_images(
                        model=ddp_model.module,
                        rae=rae,
                        prompts=vis_prompts,
                        image_size=image_size,
                        device=device,
                        dtype=autocast_dtype,
                        num_steps=sample_num_steps,
                        guidance_scale=sample_guidance_scale,
                        negative_prompt=sample_negative_prompt,
                        latent_stats=latent_stats,
                    )
                    if primary_images.numel() > 0:
                        primary_ncol = int(primary_images.size(0))
                        primary_grid = make_grid(primary_images, nrow=primary_ncol)
                        primary_label = "EMA" if sample_with_ema else "non-EMA"

                        if (
                            sample_with_ema
                            and non_ema_images is not None
                            and non_ema_images.numel() > 0
                            and non_ema_images.shape == primary_images.shape
                        ):
                            non_ema_vs_ema = make_grid(
                                torch.cat([non_ema_images, primary_images], dim=0),
                                nrow=primary_ncol,
                            )
                            wandb_utils.log_image(
                                non_ema_vs_ema,
                                step=global_step,
                                key="samples/non_ema_top_ema_bottom",
                                caption="Top row: non-EMA; bottom row: EMA",
                                commit=False,
                            )

                        # Keep existing logging keys for backward compatibility.
                        wandb_utils.log_image(
                            primary_grid,
                            step=global_step,
                            key="samples/generated",
                            commit=False,
                        )

                        if num_vis > 0 and primary_ncol == num_vis:
                            if real_images is None:
                                real_images = images[:num_vis].detach().to(
                                    device=primary_images.device,
                                    dtype=primary_images.dtype,
                                ).clamp(0, 1)
                            compare_grid = make_grid(
                                torch.cat([real_images, primary_images], dim=0),
                                nrow=num_vis,
                            )
                            wandb_utils.log_image(
                                compare_grid,
                                step=global_step,
                                key="samples/real_top_generated_bottom",
                                caption=f"Top row: real images; bottom row: generated images ({primary_label})",
                                commit=True,
                            )
                        else:
                            wandb_utils.log_image(
                                primary_grid,
                                step=global_step,
                                key="samples/real_top_generated_bottom",
                                caption=f"Generated samples ({primary_label})",
                                commit=True,
                            )
                    logger.info("Sample generation done.")
                except Exception as e:
                    logger.warning(f"Sample generation failed: {e}")
                finally:
                    if swapped_to_ema and ema_state is not None:
                        swap_model_with_ema(ddp_model.module, ema_state)
                    ddp_model.train()

        # ----- Epoch summary -----
        epoch_totals = torch.tensor(
            [float(epoch_loss_sum), float(epoch_steps)],
            device=device,
            dtype=torch.float64,
        )
        if dist.is_initialized():
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
