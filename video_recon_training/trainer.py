from __future__ import annotations

import os
import tarfile
import time
from argparse import Namespace
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:
    import httpcore
except Exception:  # pragma: no cover
    httpcore = None  # type: ignore[assignment]

from video_recon_training.checkpoint import (
    load_stage1_decoder_weights,
    load_training_checkpoint,
    save_training_checkpoint,
)
from video_recon_training.config import resolve_path
from video_recon_training.data import ChunkSampler, build_dataloader
from video_recon_training.disc.lpips import LPIPS
from video_recon_training.model import TemporalVideoRAE, build_temporal_video_rae
from video_recon_training.utils.dist_utils import cleanup_distributed, setup_distributed


def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank0_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _autocast_context(device: torch.device, precision: str):
    precision = str(precision).lower()
    if device.type != "cuda":
        return nullcontext()
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _build_optimizer(params, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    betas = training_cfg.get("betas", [0.9, 0.95])
    if len(betas) != 2:
        raise ValueError(f"training.betas must have two values, got {betas}")

    return torch.optim.AdamW(
        params,
        lr=float(training_cfg.get("lr", 2e-4)),
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        eps=1e-8,
    )


def _resolve_local_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"training.global_batch_size ({global_batch_size}) must be divisible by world_size ({world_size})"
        )
    return global_batch_size // world_size


def _maybe_all_reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if _is_dist_ready() and world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value


def _dist_barrier(device: torch.device) -> None:
    if not _is_dist_ready():
        return
    if device.type == "cuda" and device.index is not None:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def _is_recoverable_stream_error(exc: Exception) -> bool:
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
        "all video decode backends failed",
    )
    return any(token in message for token in recoverable_tokens)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return Namespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _setup_wandb(
    cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    output_dir: Path,
    rank: int,
) -> Dict[str, Any]:
    wandb_cfg = cfg.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        wandb_cfg = {}

    enabled = bool(wandb_cfg.get("enabled", False))
    scalar_interval = int(wandb_cfg.get("log_interval_steps", training_cfg.get("log_interval", 20)))
    media_interval = int(wandb_cfg.get("log_media_interval_steps", 0))
    video_fps = int(wandb_cfg.get("log_video_fps", 4))

    runtime = {
        "enabled": enabled,
        "scalar_interval": scalar_interval,
        "media_interval": media_interval,
        "video_fps": video_fps,
        "log": None,
        "log_image": None,
        "log_video": None,
    }
    if not enabled:
        return runtime

    mode = str(wandb_cfg.get("mode", "online")).strip().lower()
    if mode:
        os.environ["WANDB_MODE"] = mode

    from video_recon_training.utils import wandb_utils

    project = str(wandb_cfg.get("project", "video-recon-training")).strip()
    entity = wandb_cfg.get("entity", None)
    exp_name = wandb_cfg.get("exp_name", None)
    if exp_name is None or str(exp_name).strip() == "":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        exp_name = f"{output_dir.name}-{stamp}"

    wandb_utils.initialize(
        _to_namespace(cfg),
        entity=entity,
        exp_name=str(exp_name),
        project_name=project,
    )

    runtime["log"] = wandb_utils.log
    runtime["log_image"] = wandb_utils.log_image
    runtime["log_video"] = wandb_utils.log_video

    _rank0_print(
        rank,
        f"[wandb] enabled project={project} entity={entity} exp_name={exp_name} mode={mode if mode else 'online'}",
    )
    return runtime


def _finish_wandb(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def train(cfg: Dict[str, Any], *, config_path: str) -> None:
    rank, world_size, device = setup_distributed()

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    loss_cfg = cfg["loss"]

    seed = int(training_cfg.get("seed", 42)) + rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    global_batch_size = int(training_cfg["global_batch_size"])
    local_batch_size = _resolve_local_batch_size(global_batch_size, world_size)

    dataset, loader = build_dataloader(
        data_cfg,
        batch_size=local_batch_size,
        num_workers=int(training_cfg.get("num_workers", 4)),
        prefetch_factor=int(training_cfg.get("prefetch_factor", 2)),
        persistent_workers=bool(training_cfg.get("persistent_workers", True)),
        seed=int(training_cfg.get("seed", 42)),
        rank=rank,
        world_size=world_size,
        pin_memory=(device.type == "cuda"),
    )
    chunk_sampler = ChunkSampler(data_cfg)

    config_dir = str(Path(config_path).resolve().parent)
    model: TemporalVideoRAE = build_temporal_video_rae(
        model_cfg,
        chunk_frames=int(data_cfg["chunk_frames"]),
        config_dir=config_dir,
    )
    model = model.to(device)

    ckpt_path = resolve_path(str(model_cfg["stage1_checkpoint_path"]), config_dir)
    load_info = load_stage1_decoder_weights(
        model.rae,
        checkpoint_path=ckpt_path,
        source=str(model_cfg.get("stage1_checkpoint_source", "ema")),
        strict=True,
    )

    model.freeze_encoder_only_train_decoder()

    is_ddp = _is_dist_ready() and world_size > 1
    if is_ddp:
        ddp_model: torch.nn.Module = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    else:
        ddp_model = model

    model_core: TemporalVideoRAE = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
    trainable_params = [p for p in model_core.trainable_parameters() if p.requires_grad]

    optimizer = _build_optimizer(trainable_params, training_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and training_cfg.get("precision") == "fp16"))

    lpips = LPIPS().to(device)
    lpips.eval()
    lpips.requires_grad_(False)

    output_dir = Path(str(training_cfg.get("output_dir", "ckpts/video_recon_training"))).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        _rank0_print(rank, f"[init] output_dir={output_dir}")
        _rank0_print(rank, f"[init] loaded decoder ckpt: {load_info}")
        window_override = model_cfg.get("decoder_window_size")
        shift_override = model_cfg.get("decoder_shift_size")
        _rank0_print(
            rank,
            f"[init] attention window override window_size={window_override} shift_size={shift_override}",
        )

    wandb_rt = _setup_wandb(cfg, training_cfg, output_dir, rank)

    resume_path = training_cfg.get("resume_path", None)
    start_epoch = 0
    global_step = 0
    if resume_path:
        resume_path = resolve_path(str(resume_path), config_dir)
        if not Path(resume_path).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, global_step = load_training_checkpoint(
            resume_path,
            model=ddp_model,
            optimizer=optimizer,
            scheduler=None,
        )
        _rank0_print(rank, f"[resume] from {resume_path} epoch={start_epoch} step={global_step}")

    epochs = int(training_cfg["epochs"])
    steps_per_epoch = int(training_cfg["steps_per_epoch"])
    log_interval = int(training_cfg.get("log_interval", 20))
    save_interval_steps = int(training_cfg.get("save_interval_steps", 0))
    clip_grad = training_cfg.get("clip_grad", None)
    clip_grad = float(clip_grad) if clip_grad is not None else None

    l1_weight = float(loss_cfg.get("l1_weight", 1.0))
    lpips_weight = float(loss_cfg.get("lpips_weight", 1.0))
    precision = str(training_cfg.get("precision", "bf16")).lower()

    stream_retry_cfg = data_cfg.get("stream_retry", {})
    max_stream_failures = int(stream_retry_cfg.get("max_consecutive_failures", 0))
    stream_retry_base_sleep = float(stream_retry_cfg.get("base_sleep_seconds", 1.0))
    stream_retry_max_sleep = float(stream_retry_cfg.get("max_sleep_seconds", 30.0))
    if stream_retry_base_sleep < 0:
        raise ValueError("data.stream_retry.base_sleep_seconds must be >= 0")
    if stream_retry_max_sleep < 0:
        raise ValueError("data.stream_retry.max_sleep_seconds must be >= 0")
    if stream_retry_max_sleep < stream_retry_base_sleep:
        stream_retry_max_sleep = stream_retry_base_sleep

    data_iter = None
    checked_consistency = False
    consecutive_stream_failures = 0
    first_batch_ready = False
    first_batch_wait_start = time.time()

    if rank == 0:
        _rank0_print(rank, "[data] waiting first batch from streaming loader...")

    _dist_barrier(device)

    for epoch in range(start_epoch, epochs):
        model_core.train()
        model_core.rae.encoder.eval()
        dataset.set_epoch(epoch)
        data_iter = iter(loader)

        for _ in range(steps_per_epoch):
            while True:
                try:
                    clips = next(data_iter)
                    if not first_batch_ready:
                        first_batch_ready = True
                        if rank == 0:
                            _rank0_print(
                                rank,
                                f"[data] first batch ready after {time.time() - first_batch_wait_start:.1f}s",
                            )
                    if consecutive_stream_failures > 0 and rank == 0:
                        _rank0_print(
                            rank,
                            f"[stream] recovered after {consecutive_stream_failures} retry attempt(s)",
                        )
                    consecutive_stream_failures = 0
                    break
                except StopIteration:
                    data_iter = iter(loader)
                except Exception as exc:
                    if not _is_recoverable_stream_error(exc):
                        raise

                    consecutive_stream_failures += 1
                    if max_stream_failures > 0 and consecutive_stream_failures > max_stream_failures:
                        raise RuntimeError(
                            "Exceeded max consecutive streaming failures "
                            f"({max_stream_failures}). Last error: {type(exc).__name__}: {exc}"
                        ) from exc

                    backoff_power = min(consecutive_stream_failures - 1, 6)
                    sleep_s = min(stream_retry_base_sleep * (2 ** backoff_power), stream_retry_max_sleep)
                    if rank == 0:
                        max_failures_msg = "unlimited" if max_stream_failures <= 0 else str(max_stream_failures)
                        _rank0_print(
                            rank,
                            "[stream] recoverable error "
                            f"({consecutive_stream_failures}/{max_failures_msg}): "
                            f"{type(exc).__name__}: {exc}; retry in {sleep_s:.1f}s",
                        )
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    data_iter = iter(loader)

            clips = clips.to(device, non_blocking=True)
            sampled = chunk_sampler.sample(clips)
            chunk = sampled["chunks"].contiguous()
            target = chunk[:, -2:].contiguous()

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, precision):
                recon, latents = ddp_model(chunk, return_latents=True)
                l1_loss = F.l1_loss(recon, target)

                recon_4d = recon.reshape(-1, recon.shape[2], recon.shape[3], recon.shape[4])
                tgt_4d = target.reshape(-1, target.shape[2], target.shape[3], target.shape[4])
                lpips_loss = lpips(tgt_4d * 2.0 - 1.0, recon_4d * 2.0 - 1.0)

                total_loss = l1_weight * l1_loss + lpips_weight * lpips_loss

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
                if clip_grad is not None and clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None and clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, clip_grad)
                optimizer.step()

            if not checked_consistency:
                with torch.no_grad():
                    direct = model_core.decode_last_with_context(latents[:1])
                    cache = model_core.init_kv_cache()
                    streamed = None
                    for i in range(latents.shape[1]):
                        streamed = model_core.infer_step(latents[:1, i : i + 1], cache)
                    max_diff = float((direct - streamed).abs().max().item()) if streamed is not None else 0.0
                _rank0_print(rank, f"[consistency] direct_vs_cache_max_abs_diff={max_diff:.6e}")
                if wandb_rt["enabled"] and wandb_rt["log"] is not None:
                    wandb_rt["log"]({"debug/consistency_max_abs_diff": max_diff}, step=global_step, commit=True)
                checked_consistency = True

            global_step += 1

            should_console_log = log_interval > 0 and global_step % log_interval == 0
            should_wandb_scalar = (
                wandb_rt["enabled"]
                and wandb_rt["log"] is not None
                and int(wandb_rt["scalar_interval"]) > 0
                and global_step % int(wandb_rt["scalar_interval"]) == 0
            )
            should_wandb_media = (
                wandb_rt["enabled"]
                and wandb_rt["log_image"] is not None
                and wandb_rt["log_video"] is not None
                and int(wandb_rt["media_interval"]) > 0
                and global_step % int(wandb_rt["media_interval"]) == 0
            )

            if should_console_log or should_wandb_scalar:
                metrics = torch.stack(
                    [total_loss.detach(), l1_loss.detach(), lpips_loss.detach()],
                    dim=0,
                ).to(torch.float32)
                metrics = _maybe_all_reduce_mean(metrics, world_size)

                strides = sampled["strides"]
                stride_values = [int(v) for v in data_cfg["stride_values"]]
                stride_counts = torch.tensor(
                    [float((strides == v).sum().item()) for v in stride_values],
                    device=device,
                    dtype=torch.float32,
                )
                if _is_dist_ready() and world_size > 1:
                    dist.all_reduce(stride_counts, op=dist.ReduceOp.SUM)

                total_stride = float(stride_counts.sum().item())
                stride_ratio_values = [
                    float(stride_counts[i].item()) / max(1.0, total_stride) for i in range(len(stride_values))
                ]
                stride_ratio = [f"s{value}:{ratio:.3f}" for value, ratio in zip(stride_values, stride_ratio_values)]
                stride_info = " ".join(stride_ratio)

                if should_console_log:
                    _rank0_print(
                        rank,
                        (
                            f"[epoch {epoch} step {global_step}] "
                            f"loss={metrics[0].item():.6f} "
                            f"l1={metrics[1].item():.6f} "
                            f"lpips={metrics[2].item():.6f} "
                            f"strides={stride_info}"
                        ),
                    )

                if should_wandb_scalar:
                    wb_stats: Dict[str, float] = {
                        "train/loss": float(metrics[0].item()),
                        "train/l1": float(metrics[1].item()),
                        "train/lpips": float(metrics[2].item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "train/epoch": float(epoch),
                        "train/world_size": float(world_size),
                    }
                    for value, ratio in zip(stride_values, stride_ratio_values):
                        wb_stats[f"sampling/stride_ratio/s{value}"] = float(ratio)
                    wandb_rt["log"](wb_stats, step=global_step, commit=(not should_wandb_media))

            if should_wandb_media:
                with torch.no_grad():
                    sample_input = chunk[0].detach().to(torch.float32).clamp(0.0, 1.0).cpu()
                    sample_target = target[0].detach().to(torch.float32).clamp(0.0, 1.0).cpu()
                    sample_recon = recon[0].detach().to(torch.float32).clamp(0.0, 1.0).cpu()

                    # image grid: first two are GT, next two are reconstruction
                    image_panel = torch.cat([sample_target, sample_recon], dim=0)
                    # side-by-side video for two target frames vs two recon frames
                    compare_video = torch.cat([sample_target, sample_recon], dim=-1)

                wandb_rt["log_image"](image_panel, step=global_step, commit=False)
                wandb_rt["log_video"](
                    sample_input,
                    step=global_step,
                    fps=int(wandb_rt["video_fps"]),
                    name="samples/input_chunk",
                    commit=False,
                )
                wandb_rt["log_video"](
                    compare_video,
                    step=global_step,
                    fps=int(wandb_rt["video_fps"]),
                    name="samples/target_vs_recon",
                    commit=True,
                )

            if rank == 0 and save_interval_steps > 0 and global_step % save_interval_steps == 0:
                ckpt = checkpoint_dir / f"step-{global_step:08d}.pt"
                save_training_checkpoint(
                    str(ckpt),
                    step=global_step,
                    epoch=epoch,
                    model=ddp_model,
                    optimizer=optimizer,
                    scheduler=None,
                )
                _rank0_print(rank, f"[ckpt] saved {ckpt}")
                if wandb_rt["enabled"] and wandb_rt["log"] is not None:
                    wandb_rt["log"]({"train/checkpoint_step": float(global_step)}, step=global_step, commit=True)

    if rank == 0:
        final_ckpt = checkpoint_dir / "last.pt"
        save_training_checkpoint(
            str(final_ckpt),
            step=global_step,
            epoch=epochs,
            model=ddp_model,
            optimizer=optimizer,
            scheduler=None,
        )
        _rank0_print(rank, f"[done] final checkpoint: {final_ckpt}")

    _dist_barrier(device)

    _finish_wandb(bool(wandb_rt.get("enabled", False)))
    cleanup_distributed()
