#!/usr/bin/env python3
"""
Video sanity-check training script.

Behavior:
1) Load exactly 64 samples from local `sanity-check-video-data`
2) Apply the same OpenVid sample policy (>=64 frames, random 64-frame window,
   frame_stride subsampling, resize to 256x256)
3) Build a fixed in-memory subset and train repeatedly on it
4) Visualize fixed 3 samples/prompts at every sampling step (never changing)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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
    FixedVideoSubsetDataset,
    collate_fixed_video_batch,
    configure_experiment_dirs,
    load_local_sanity_subset,
)
from utils.resume_utils import find_resume_checkpoint, save_worktree
from utils import wandb_utils

# Reuse shared helpers from train.py
from train import (
    compute_shift_ratio_from_latent,
    load_latent_stats,
    normalize_latent,
    unwrap_model,
    save_checkpoint,
    load_checkpoint,
    init_ema_state,
    update_ema_state,
    swap_model_with_ema,
    maybe_autocast,
    generate_sample_videos,
    make_video_panel,
    make_video_compare_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video sanity-check overfit training.")
    parser.add_argument(
        "--config",
        type=str,
        default="t2v_training_single_ds/config/sanity_check.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/scratch/e1539128/ckpt-video-sanity-check",
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
    num_epochs = int(train_cfg.get("epochs", 20))
    accumulation_steps = int(train_cfg.get("accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("training.accumulation_steps must be > 0.")

    configured_global_batch_size = int(train_cfg.get("global_batch_size", 64))
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

    num_workers = int(train_cfg.get("num_workers", 2))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    clip_grad = float(train_cfg.get("clip_grad", 1.0))
    if clip_grad <= 0:
        clip_grad = None
    log_interval = int(train_cfg.get("log_interval", 10))
    sample_every = int(train_cfg.get("sample_every", 50))
    checkpoint_every_steps = int(train_cfg.get("checkpoint_every_steps", 200))
    save_checkpoints = bool(train_cfg.get("save_checkpoints", True))
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    maintain_ema = bool(train_cfg.get("maintain_ema", False))
    sample_with_ema = bool(wandb_cfg.get("sample_with_ema", False))
    sample_num_steps = int(wandb_cfg.get("sample_num_steps", 10))
    sample_fps = int(wandb_cfg.get("sample_fps", 8))
    if sample_num_steps <= 0:
        raise ValueError("wandb.sample_num_steps must be > 0.")

    precision = getattr(args, "precision", train_cfg.get("precision", "bf16"))
    args.precision = precision

    # --- Fixed subset / video policy ---
    sanity_subset_dir = str(data_cfg.get("sanity_subset_dir", "sanity-check-video-data"))
    sanity_subset_size = int(data_cfg.get("sanity_subset_size", 64))
    fixed_vis_count = int(data_cfg.get("fixed_vis_count", 3))
    video_size = int(data_cfg.get("video_size", 256))
    window_frames = int(data_cfg.get("window_frames", 64))
    frame_stride = int(data_cfg.get("frame_stride", 2))
    sampled_frames = len(range(0, window_frames, frame_stride))
    if sanity_subset_size <= 0:
        raise ValueError("data.sanity_subset_size must be > 0.")
    if fixed_vis_count <= 0:
        raise ValueError("data.fixed_vis_count must be > 0.")

    # --- Seed ---
    default_seed = int(train_cfg.get("global_seed", 42))
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    fixed_vis_seed_raw = data_cfg.get("fixed_vis_seed", None)
    fixed_vis_seed = global_seed if fixed_vis_seed_raw is None else int(fixed_vis_seed_raw)

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

    rae.requires_grad_(False)
    rae.eval()

    temporal_patch_size = int(getattr(rae, "temporal_patch_size", -1))
    if temporal_patch_size != 2:
        raise ValueError(
            f"Expected encoder temporal patch/chunk size == 2, but got {temporal_patch_size}."
        )

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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"UniDiT trainable: {trainable_params / 1e6:.2f}M / total: {total_params / 1e6:.2f}M")
    logger.info(
        f"Sanity data policy: subset_size={sanity_subset_size}, fixed_vis_count={fixed_vis_count}, "
        f"fixed_vis_seed={fixed_vis_seed}, "
        f"window_frames={window_frames}, frame_stride={frame_stride}, sampled_frames={sampled_frames}, "
        f"video_size={video_size}"
    )

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

    # --- Optimizer / scheduler ---
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
    else:
        optimizer, optim_msg = build_optimizer(trainable_param_list, train_cfg)
        logger.info(optim_msg)

    if "steps_per_epoch" not in train_cfg:
        raise ValueError("training.steps_per_epoch must be set.")
    steps_per_epoch = int(train_cfg.get("steps_per_epoch"))
    if steps_per_epoch <= 0:
        raise ValueError("training.steps_per_epoch must be > 0.")

    scheduler: Optional[LambdaLR] = None
    if train_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, train_cfg)
        logger.info(sched_msg)

    scaler, autocast_kwargs = get_autocast_scaler(args)
    autocast_dtype = autocast_kwargs.get("dtype", torch.bfloat16)
    amp_enabled = bool(autocast_kwargs.get("enabled", False))

    # --- Load fixed sanity subset ---
    logger.info(f"Loading local sanity subset from '{sanity_subset_dir}'...")
    local_samples = load_local_sanity_subset(
        subset_dir=sanity_subset_dir,
        max_samples=sanity_subset_size,
        video_size=video_size,
        window_frames=window_frames,
        frame_stride=frame_stride,
        seed=global_seed,
    )
    fixed_dataset = FixedVideoSubsetDataset(local_samples)
    if len(fixed_dataset) != sanity_subset_size:
        raise RuntimeError(
            f"Expected {sanity_subset_size} sanity samples, got {len(fixed_dataset)}."
        )
    if rank == 0:
        logger.info(f"Loaded fixed sanity subset: {len(fixed_dataset)} samples.")

    vis_n = min(fixed_vis_count, len(fixed_dataset))
    vis_rng = random.Random(int(fixed_vis_seed))
    if vis_n < len(fixed_dataset):
        vis_indices = vis_rng.sample(range(len(fixed_dataset)), vis_n)
    else:
        vis_indices = list(range(len(fixed_dataset)))
    fixed_vis_videos = torch.stack([fixed_dataset.videos[i] for i in vis_indices], dim=0).clone()
    fixed_vis_prompts = [fixed_dataset.prompts[i] for i in vis_indices]
    fixed_vis_sample_ids = [fixed_dataset.sample_ids[i] for i in vis_indices]
    if rank == 0:
        logger.info(
            f"Fixed visualization samples (seed={fixed_vis_seed}): "
            f"indices={vis_indices}, sample_ids={fixed_vis_sample_ids}"
        )

    with torch.no_grad():
        probe_vis = fixed_vis_videos[:1].to(device=device, dtype=torch.float32)
        with maybe_autocast(device, autocast_dtype, amp_enabled):
            vis_latent_probe, _ = rae.encode(probe_vis)
    vis_grid_t, vis_grid_h, vis_grid_w = (
        int(vis_latent_probe.shape[1]),
        int(vis_latent_probe.shape[2]),
        int(vis_latent_probe.shape[3]),
    )

    def make_fixed_loader(epoch_seed: int) -> DataLoader:
        sampler = DistributedSampler(
            fixed_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=epoch_seed,
            drop_last=False,
        )
        loader_kwargs = {
            "batch_size": int(micro_batch_size),
            "sampler": sampler,
            "num_workers": int(num_workers),
            "pin_memory": device.type == "cuda",
            "drop_last": False,
            "collate_fn": collate_fixed_video_batch,
            "persistent_workers": int(num_workers) > 0,
        }
        if int(num_workers) > 0:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        return DataLoader(fixed_dataset, **loader_kwargs)

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

    # --- WandB ---
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
            "system/accumulation_steps": accumulation_steps,
            "system/global_batch_size": global_batch_size,
            "system/trainable_params_M": trainable_params / 1e6,
            "system/fixed_subset_size": len(fixed_dataset),
            "system/fixed_vis_count": vis_n,
            "system/fixed_vis_seed": int(fixed_vis_seed),
        }, step=0)

    # =====================================================================
    # Training loop
    # =====================================================================
    logger.info(
        f"Starting sanity training: epochs={num_epochs}, steps_per_epoch={steps_per_epoch}, "
        f"micro_bs={micro_batch_size}, accum={accumulation_steps}, global_bs={global_batch_size}"
    )
    if rank == 0:
        if save_checkpoints:
            logger.info(f"Checkpoint cadence: every {checkpoint_every_steps} steps.")
        else:
            logger.info("Checkpoint saving disabled.")

    barrier()

    stream_epoch = start_epoch
    train_loader = make_fixed_loader(global_seed + stream_epoch)
    data_iter = iter(train_loader)

    def next_batch():
        nonlocal stream_epoch, train_loader, data_iter
        while True:
            try:
                return next(data_iter)
            except StopIteration:
                stream_epoch += 1
                train_loader = make_fixed_loader(global_seed + stream_epoch)
                data_iter = iter(train_loader)

    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), device=device, dtype=torch.float32)
            step_mse = torch.zeros((), device=device, dtype=torch.float32)
            videos = None
            texts: List[str] = []

            for micro_idx in range(accumulation_steps):
                batch = next_batch()
                videos = batch["video"].to(device, non_blocking=True)
                texts = batch["prompt"]

                with torch.no_grad():
                    with maybe_autocast(device, autocast_dtype, amp_enabled):
                        latent, _ = rae.encode(videos)
                    if latent_stats is not None:
                        latent = normalize_latent(latent, latent_stats["mean"], latent_stats["std"])

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
            epoch_loss_sum += float(step_loss.item())
            epoch_steps += 1
            global_step += 1

            if save_checkpoints and rank == 0 and global_step % checkpoint_every_steps == 0:
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

            if do_sample and rank == 0 and args.wandb:
                logger.info("Generating fixed sanity videos for visualization...")
                swapped_to_ema = False
                try:
                    if sample_with_ema and ema_state is not None:
                        swap_model_with_ema(model_core, ema_state)
                        swapped_to_ema = True

                    gen_videos = generate_sample_videos(
                        model=model_core,
                        rae=rae,
                        prompts=fixed_vis_prompts,
                        grid_t=vis_grid_t,
                        grid_h=vis_grid_h,
                        grid_w=vis_grid_w,
                        device=device,
                        dtype=autocast_dtype,
                        amp_enabled=amp_enabled,
                        num_steps=sample_num_steps,
                        latent_stats=latent_stats,
                    )

                    if gen_videos.numel() > 0:
                        real_videos = fixed_vis_videos[: int(gen_videos.size(0))].to(
                            device=gen_videos.device,
                            dtype=gen_videos.dtype,
                        ).clamp(0, 1)
                        gen_panel = make_video_panel(gen_videos)
                        compare_panel = make_video_compare_panel(real_videos, gen_videos)
                        logger.info(
                            "Sample video tensor shapes: "
                            f"generated={tuple(gen_videos.shape)}, "
                            f"real={tuple(real_videos.shape)}, "
                            f"panel={tuple(gen_panel.shape)}, "
                            f"compare_panel={tuple(compare_panel.shape)}"
                        )
                        wandb_utils.log_video(
                            gen_panel,
                            step=global_step,
                            key="samples/generated",
                            caption="Fixed sanity prompts generated videos",
                            fps=sample_fps,
                            commit=False,
                        )
                        wandb_utils.log_video(
                            compare_panel,
                            step=global_step,
                            key="samples/real_top_generated_bottom",
                            caption="Top: fixed real sanity videos; bottom: generated videos",
                            fps=sample_fps,
                            commit=True,
                        )
                except Exception as e:
                    logger.warning(f"Sanity sample generation failed: {e}")
                finally:
                    if swapped_to_ema and ema_state is not None:
                        swap_model_with_ema(model_core, ema_state)
                    ddp_model.train()

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

    if save_checkpoints and rank == 0:
        ckpt_path = os.path.join(checkpoint_dir, "ep-last.pt")
        save_checkpoint(ckpt_path, global_step, num_epochs, ddp_model, ema_state, optimizer, scheduler)
        logger.info(f"Saved final checkpoint: {ckpt_path}")

    if args.wandb and rank == 0:
        wandb_utils.finish()

    barrier()
    logger.info("Sanity training complete.")
    cleanup_distributed()


if __name__ == "__main__":
    main()
