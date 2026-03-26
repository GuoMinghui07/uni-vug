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
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid

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
    DEFAULT_STREAMING_IMAGE_KEYS,
    DEFAULT_STREAMING_TEXT_KEYS,
    build_streaming_dataloader,
    configure_experiment_dirs,
)
from utils.resume_utils import find_resume_checkpoint, load_training_checkpoint, save_worktree
from utils import wandb_utils


# =====================================================================
# Streaming WebDataset loader
# =====================================================================

DEFAULT_DATA_FILES = (
    "https://huggingface.co/datasets/ma-xu/fine-t2i/resolve/main/"
    "synthetic_enhanced_prompt_square_resolution/train-*.tar"
)


class FixedSubsetDataset(Dataset):
    """In-memory tiny dataset for overfit sanity checks."""

    def __init__(self, samples: List[Dict[str, Any]]):
        if len(samples) == 0:
            raise ValueError("FixedSubsetDataset requires at least one sample.")
        self.images = torch.stack([s["image"].detach().cpu().to(torch.float32) for s in samples], dim=0)
        self.prompts = [str(s["prompt"]) for s in samples]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"image": self.images[idx], "prompt": self.prompts[idx]}


def collate_fixed_subset_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    return {"image": images, "prompt": prompts}


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


# =====================================================================
# Checkpoint save / load
# =====================================================================

def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: nn.Module,
    ema_state: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    raw_model = unwrap_model(model)
    state = {
        "step": step,
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "ema": {k: v.detach().cpu() for k, v in ema_state.items()},
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

        # Encode condition
        with torch.amp.autocast("cuda", dtype=dtype):
            context, _ = model.encode_condition(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                pixel_values=tokenized.get("pixel_values"),
                image_grid_thw=tokenized.get("image_grid_thw"),
                pixel_values_videos=tokenized.get("pixel_values_videos"),
                video_grid_thw=tokenized.get("video_grid_thw"),
                second_per_grid_ts=tokenized.get("second_per_grid_ts"),
            )

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
                num_steps=num_steps,
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
        latent_stats=latent_stats,
    )
    if images.numel() == 0:
        return torch.zeros(3, image_size, image_size, device=device, dtype=torch.float32)
    return make_grid(images, nrow=images.size(0))


# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 DiT sanity/overfit training.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/dit_sanity_overfit.yaml",
        help="YAML config path.",
    )
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Output directory.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in experiment dir.")
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help=(
            "When resuming, load only model/EMA weights from checkpoint and keep optimizer/scheduler "
            "from current config."
        ),
    )
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional WandB run name override.")
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
    parser.add_argument("--sanity", action="store_true", help="Run single forward pass and exit.")
    return parser.parse_args()


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()

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
    save_checkpoints = bool(train_cfg.get("save_checkpoints", False))
    checkpoint_every_steps = int(train_cfg.get("checkpoint_every_steps", 10000))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    if not (0.0 <= ema_decay < 1.0):
        raise ValueError("training.ema_decay must satisfy 0 <= ema_decay < 1.")
    if save_checkpoints and checkpoint_every_steps <= 0:
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
    context_dropout_prob = float(cfg_section.get("context_dropout_prob", 0.1))

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
    logger.info(f"CFG context dropout: {context_dropout_prob}")
    logger.info(f"EMA decay: {ema_decay}")

    # --- DDP ---
    if dist.is_initialized():
        ddp_model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        logger.info(f"DDP enabled: world_size={world_size}")
    else:
        ddp_model = model
        logger.info("DDP disabled: single-process mode.")
    model_core = unwrap_model(ddp_model)

    # --- Optimizer & scheduler ---
    trainable_named_params = [(n, p) for n, p in ddp_model.named_parameters() if p.requires_grad]
    trainable_param_list = [p for _, p in trainable_named_params]
    ema_state = init_ema_state(model_core)

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

    # --- Sanity check ---
    if args.sanity:
        logger.info("Running sanity check...")
        model_core.eval()
        with torch.no_grad():
            dummy_latent = torch.randn(2, 1, 16, 16, 1024, device=device, dtype=autocast_dtype)
            dummy_input_ids = torch.randint(0, 1000, (2, 128), device=device)
            dummy_attention_mask = torch.ones(2, 128, device=device)
            with torch.amp.autocast("cuda", **autocast_kwargs):
                # 使用新的参数调用方式
                out = model_core(
                    latent=dummy_latent,
                    input_ids=dummy_input_ids,
                    attention_mask=dummy_attention_mask,
                )
            logger.info(f"Sanity ok: loss={out['loss'].item():.4f}")
        barrier()
        cleanup_distributed()
        return

    # --- Data loader ---
    logger.info("Configuring streaming data loader...")
    data_files = str(data_cfg.get("data_files", DEFAULT_DATA_FILES))
    data_split = str(data_cfg.get("split", "train"))
    text_keys = list(data_cfg.get("text_keys", list(DEFAULT_STREAMING_TEXT_KEYS)))
    image_keys = list(data_cfg.get("image_keys", list(DEFAULT_STREAMING_IMAGE_KEYS)))
    shuffle_buffer_size = int(data_cfg.get("shuffle_buffer_size", 1000))
    overfit_subset_size = int(data_cfg.get("overfit_subset_size", 0))
    overfit_vis_count = int(data_cfg.get("overfit_vis_count", 3))

    def make_base_stream_loader(stream_epoch: int):
        return build_streaming_dataloader(
            data_files=data_files,
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
        )

    fixed_vis_images: Optional[torch.Tensor] = None
    fixed_vis_prompts: List[str] = []
    if overfit_subset_size > 0:
        logger.info(f"Building fixed overfit subset: target_size={overfit_subset_size} samples/rank.")

        collect_loader = make_base_stream_loader(stream_epoch=0)
        collect_iter = iter(collect_loader)
        subset_samples: List[Dict[str, Any]] = []
        collect_restarts = 0
        max_collect_restarts = 32
        while len(subset_samples) < overfit_subset_size:
            try:
                batch = next(collect_iter)
            except StopIteration:
                collect_restarts += 1
                if len(subset_samples) == 0 and collect_restarts > max_collect_restarts:
                    raise RuntimeError(
                        "Failed to collect overfit subset: this rank received no samples from streaming split. "
                        "Use single-GPU sanity run or increase dataset size."
                    )
                collect_loader = make_base_stream_loader(stream_epoch=collect_restarts)
                collect_iter = iter(collect_loader)
                continue

            images_b = batch["image"]
            prompts_b = batch["prompt"]
            for img, prompt in zip(images_b, prompts_b):
                subset_samples.append({"image": img.detach().cpu(), "prompt": str(prompt)})
                if len(subset_samples) >= overfit_subset_size:
                    break

        fixed_dataset = FixedSubsetDataset(subset_samples)
        if rank == 0:
            logger.info(f"Fixed overfit subset ready: {len(fixed_dataset)} samples.")

        vis_n = max(0, min(overfit_vis_count, len(fixed_dataset)))
        if vis_n > 0:
            fixed_vis_images = fixed_dataset.images[:vis_n].clone()
            fixed_vis_prompts = fixed_dataset.prompts[:vis_n]
            if rank == 0:
                logger.info(f"Fixed visualization set: {vis_n} samples from overfit subset.")

        def make_stream_loader(stream_epoch: int):
            sampler = DistributedSampler(
                fixed_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=global_seed + int(stream_epoch),
                drop_last=False,
            )
            loader_kwargs = {
                "batch_size": int(micro_batch_size),
                "sampler": sampler,
                "num_workers": int(num_workers),
                "pin_memory": True,
                "drop_last": False,
                "collate_fn": collate_fixed_subset_batch,
            }
            if int(num_workers) > 0:
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            return DataLoader(fixed_dataset, **loader_kwargs)

    else:
        make_stream_loader = make_base_stream_loader

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
    if args.resume:
        resume_path = find_resume_checkpoint(experiment_dir)
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
            if args.resume_model_only:
                if scheduler is not None:
                    # Align fresh scheduler to resumed progress while honoring current config lr/base_lr.
                    scheduler.step(global_step)
                logger.info(
                    "Resume mode: loaded model/EMA only; optimizer/scheduler kept from current config."
                )
            logger.info(f"Resumed at epoch={start_epoch}, step={global_step}")
            resumed = True
        else:
            logger.warning(
                f"--resume was set but no checkpoint found under '{experiment_dir}/checkpoints'; starting from scratch."
            )
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
            "system/context_dropout": context_dropout_prob,
            "system/ema_decay": ema_decay,
            "system/overfit_subset_size_per_rank": overfit_subset_size,
            "system/overfit_subset_enabled": int(overfit_subset_size > 0),
            "system/save_checkpoints": int(save_checkpoints),
        }, step=0)

    sample_prompts = list(wandb_cfg.get("sample_prompts", [
        "A golden retriever puppy playing in a field of wildflowers under a blue sky.",
        "A futuristic cityscape at sunset with flying cars and neon signs.",
        "A cozy cabin in a snowy mountain landscape with smoke coming from the chimney.",
    ]))

    # =====================================================================
    # Training loop
    # =====================================================================
    logger.info(
        f"Starting training: {num_epochs} epochs, micro_bs={micro_batch_size}/gpu, "
        f"accum={accumulation_steps}, global_bs={global_batch_size}, "
        f"image_size={image_size}, steps_per_epoch={estimated_steps_per_epoch}"
    )
    if rank == 0:
        if save_checkpoints:
            logger.info(f"Checkpoint cadence: every {checkpoint_every_steps} steps.")
        else:
            logger.info("Checkpoint saving disabled for sanity run.")

    barrier()

    stream_epoch = start_epoch
    train_loader = make_stream_loader(stream_epoch)
    data_iter = iter(train_loader)

    def next_stream_batch():
        nonlocal stream_epoch, train_loader, data_iter
        while True:
            try:
                return next(data_iter)
            except StopIteration:
                stream_epoch += 1
                train_loader = make_stream_loader(stream_epoch)
                data_iter = iter(train_loader)

    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for _ in range(estimated_steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device, dtype=torch.float32)
            step_mse = torch.zeros((), device=device, dtype=torch.float32)
            images = None
            texts: List[str] = []

            for micro_idx in range(accumulation_steps):
                batch = next_stream_batch()
                images = batch["image"].to(device, non_blocking=True)
                texts = batch["prompt"]

                # ----- Encode ground truth latent with frozen RAE -----
                with torch.no_grad():
                    with torch.amp.autocast("cuda", **autocast_kwargs):
                        latent, _ = rae.encode(images)
                    if latent_stats is not None:
                        latent = normalize_latent(latent, latent_stats["mean"], latent_stats["std"])

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
                    if dist.is_initialized() and micro_idx < accumulation_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with torch.amp.autocast("cuda", **autocast_kwargs):
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

            update_ema_state(ema_state, model_core, ema_decay)

            if scheduler is not None:
                scheduler.step()

            step_loss /= accumulation_steps
            step_mse /= accumulation_steps
            loss_val = float(step_loss.item())
            epoch_loss_sum += loss_val
            epoch_steps += 1
            global_step += 1

            if save_checkpoints and rank == 0 and global_step % checkpoint_every_steps == 0:
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
                    ],
                    device=device,
                    dtype=torch.float32,
                )
                if dist.is_initialized():
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

            # ----- Sample visualization -----
            if do_sample and rank == 0 and args.wandb:
                logger.info("Generating samples for visualization...")
                swapped_to_ema = False
                try:
                    swap_model_with_ema(model_core, ema_state)
                    swapped_to_ema = True
                    if len(fixed_vis_prompts) > 0:
                        vis_prompts = fixed_vis_prompts
                        num_vis = len(vis_prompts)
                    else:
                        num_vis = min(3, int(images.size(0)) if images is not None else 0, len(texts))
                        vis_prompts = texts[:num_vis] if num_vis > 0 else sample_prompts[:3]
                    gen_images = generate_sample_images(
                        model=model_core,
                        rae=rae,
                        prompts=vis_prompts,
                        image_size=image_size,
                        device=device,
                        dtype=autocast_dtype,
                        num_steps=int(model_cfg.get("fm_num_steps", 50)),
                        latent_stats=latent_stats,
                    )
                    if gen_images.numel() > 0:
                        ncol = int(gen_images.size(0))
                        gen_grid = make_grid(gen_images, nrow=ncol)
                        wandb_utils.log_image(
                            gen_grid,
                            step=global_step,
                            key="samples/generated",
                            commit=False,
                        )

                        if fixed_vis_images is not None and ncol <= int(fixed_vis_images.size(0)):
                            real_images = fixed_vis_images[:ncol].detach().to(
                                device=gen_images.device,
                                dtype=gen_images.dtype,
                            ).clamp(0, 1)
                            compare_grid = make_grid(
                                torch.cat([real_images, gen_images], dim=0),
                                nrow=ncol,
                            )
                            wandb_utils.log_image(
                                compare_grid,
                                step=global_step,
                                key="samples/real_top_generated_bottom",
                                caption="Top row: fixed-overfit real images; bottom row: generated images",
                                commit=True,
                            )
                        elif num_vis > 0 and images is not None and ncol == num_vis:
                            real_images = images[:num_vis].detach().to(
                                device=gen_images.device,
                                dtype=gen_images.dtype,
                            ).clamp(0, 1)
                            compare_grid = make_grid(
                                torch.cat([real_images, gen_images], dim=0),
                                nrow=num_vis,
                            )
                            wandb_utils.log_image(
                                compare_grid,
                                step=global_step,
                                key="samples/real_top_generated_bottom",
                                caption="Top row: real images; bottom row: generated images",
                                commit=True,
                            )
                        else:
                            wandb_utils.log_image(
                                gen_grid,
                                step=global_step,
                                key="samples/real_top_generated_bottom",
                                caption="Generated samples",
                                commit=True,
                            )
                    logger.info("Sample generation done.")
                except Exception as e:
                    logger.warning(f"Sample generation failed: {e}")
                finally:
                    if swapped_to_ema:
                        swap_model_with_ema(model_core, ema_state)
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
    if save_checkpoints and rank == 0:
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
