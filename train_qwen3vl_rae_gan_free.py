# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE training script with reconstruction, LPIPS, and Gram losses.

This script adapts the training logic from the Kakao Brain VQGAN trainer while
targeting the RAE autoencoder architecture used in this repository.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from eval import evaluate_reconstruction_distributed
from disc import LPIPS, GramLossVGG19

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import *
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with reconstruction, LPIPS, and Gram losses.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing a stage_1 section.")
    parser.add_argument("--data-path", type=Path, required=True, help="HuggingFace datasets cache dir for ImageNet-1k.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=448, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")    
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging if set.')
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode, rae.forward).")
    parser.add_argument("--sanity", action="store_true", help="Run a single forward pass and exit.")
    parser.add_argument("--gram_weight", type=float, default=None, help="Weight for Gram loss.")
    parser.add_argument(
        "--gram_layers",
        type=str,
        default=None,
        help="Comma-separated VGG19 feature indices for Gram loss (e.g., '0,5,10,19,28').",
    )
    parser.add_argument(
        "--gram_layer_weights",
        type=str,
        default=None,
        help="Comma-separated layer weights for Gram loss (must match gram_layers length).",
    )
    parser.add_argument(
        "--gram_loss_type",
        type=str,
        choices=["l1", "l2"],
        default=None,
        help="Gram loss type (l1 or l2).",
    )
    parser.add_argument(
        "--gram_input_range",
        type=str,
        choices=["minus1_1", "0_1"],
        default=None,
        help="Input range for Gram loss preprocessing.",
    )
    parser.add_argument(
        "--gram_use_imagenet_norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply ImageNet normalization in Gram loss preprocessing.",
    )
    parser.add_argument(
        "--gram_vgg_weights",
        type=str,
        default=None,
        help="Torchvision VGG19 weights enum name (e.g., IMAGENET1K_V1). Use 'none' to disable.",
    )
    return parser.parse_args()


def _parse_list(value, cast):
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip() == "":
            return None
        return [cast(item.strip()) for item in value.split(",") if item.strip() != ""]
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(value)]


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _autocast(device: torch.device, autocast_kwargs: dict):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, **autocast_kwargs)
    return torch.cuda.amp.autocast(**autocast_kwargs)




def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def main():
    args = parse_args()
    #### Dist Init
    rank, world_size, device = setup_distributed()
    
    #### Config init
    full_cfg = OmegaConf.load(args.config)
    (rae_config, *_) = parse_configs(full_cfg)
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    loss_section = full_cfg.get("loss", None)
    loss_cfg = OmegaConf.to_container(loss_section, resolve=True) if loss_section is not None else {}
    loss_cfg = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}
    if not loss_cfg:
        gan_section = full_cfg.get("gan", None)
        gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
        gan_cfg = dict(gan_cfg) if isinstance(gan_cfg, dict) else {}
        loss_cfg = dict(gan_cfg.get("loss", {})) if isinstance(gan_cfg.get("loss", {}), dict) else {}

    gram_section = full_cfg.get("gram", None)
    gram_cfg = OmegaConf.to_container(gram_section, resolve=True) if gram_section is not None else {}
    gram_cfg = dict(gram_cfg) if isinstance(gram_cfg, dict) else {}

    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    lpips_start_epoch = float(loss_cfg.get("lpips_start", 0))

    gram_weight = args.gram_weight if args.gram_weight is not None else float(
        gram_cfg.get("weight", gram_cfg.get("gram_weight", 100.0))
    )
    gram_layers = _parse_list(
        args.gram_layers if args.gram_layers is not None else gram_cfg.get("layers", gram_cfg.get("gram_layers", None)),
        int,
    ) or [0, 5, 10, 19, 28]
    gram_layer_weights = _parse_list(
        args.gram_layer_weights
        if args.gram_layer_weights is not None
        else gram_cfg.get("layer_weights", gram_cfg.get("gram_layer_weights", None)),
        float,
    )
    gram_loss_type = args.gram_loss_type if args.gram_loss_type is not None else gram_cfg.get("loss_type", gram_cfg.get("gram_loss_type", "l1"))
    gram_input_range = args.gram_input_range if args.gram_input_range is not None else gram_cfg.get("input_range", gram_cfg.get("gram_input_range", "0_1"))
    gram_use_imagenet_norm = _coerce_bool(
        args.gram_use_imagenet_norm
        if args.gram_use_imagenet_norm is not None
        else gram_cfg.get("use_imagenet_norm", gram_cfg.get("gram_use_imagenet_norm", True)),
        True,
    )
    gram_vgg_weights = args.gram_vgg_weights if args.gram_vgg_weights is not None else gram_cfg.get("vgg_weights", gram_cfg.get("gram_vgg_weights", "IMAGENET1K_V1"))
    if gram_layer_weights is not None and len(gram_layer_weights) != len(gram_layers):
        raise ValueError("gram_layer_weights must have the same length as gram_layers.")
    batch_size = int(training_cfg.get("batch_size", 16))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
        batch_size = global_batch_size // world_size
    else:
        global_batch_size = batch_size * world_size
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 1250)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))
    eval_section = full_cfg.get("eval", None)
    
    if eval_section:
        do_eval = True
        eval_interval = int(eval_section.get("eval_interval", 5000))
        eval_model = eval_section.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
        eval_metrics = eval_section.get("metrics", ("rfid", "psnr", "ssim")) # by default eval all
        eval_data = eval_section.get("data_path", None)
        reference_npz_path = eval_section.get("reference_npz_path", None)
        assert eval_data, "eval.data_path must be specified to enable evaluation."
        assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
        assert len(eval_metrics) > 0, "eval.metrics must contain at least one metric to compute."
    else:
        do_eval = False
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    # update args as a dict to full_cfg
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir
    
    
    #### Model init
    rae = instantiate_from_config(rae_config).to(device)
    if args.compile:
        rae.encode = torch.compile(rae.encode)
        rae.forward = torch.compile(rae.forward)
    rae.encoder.eval()
    rae.decoder.train()
    if hasattr(rae, "demerge"):
        rae.demerge.train()
    if hasattr(rae, "to_pixels"):
        rae.to_pixels.train()
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)
    # only train decoder (+ demerge/to_pixels if present)
    rae.requires_grad_(False)
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)
    if hasattr(rae, "demerge"):
        rae.demerge.requires_grad_(True)
    if hasattr(rae, "to_pixels"):
        rae.to_pixels.requires_grad_(True)
    ddp_model = DDP(rae, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)  # type: ignore[arg-type]
    rae = ddp_model.module
    decoder = ddp_model.module.decoder
    
    lpips = LPIPS().to(device)
    lpips.eval()
    gram_loss_fn = GramLossVGG19(
        layers=gram_layers,
        layer_weights=gram_layer_weights,
        loss_type=gram_loss_type,
        input_range=gram_input_range,
        use_imagenet_norm=gram_use_imagenet_norm,
        vgg_weights=gram_vgg_weights,
        device=device,
    )
    gram_loss_fn.eval()
    
    #### Opt, Schedl init
    decoder_params = list(decoder.parameters())
    if hasattr(ddp_model.module, "demerge"):
        decoder_params += list(ddp_model.module.demerge.parameters())
    if hasattr(ddp_model.module, "to_pixels"):
        decoder_params += list(ddp_model.module.to_pixels.parameters())
    optimizer, optim_msg = build_optimizer(decoder_params, training_cfg)
    
    #### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    if args.sanity:
        rae.eval()
        with torch.no_grad():
            images = torch.zeros(2, 3, args.image_size, args.image_size, device=device)
            with _autocast(device, autocast_kwargs):
                recon = rae(images)
        logger.info(f"Sanity check ok: input {tuple(images.shape)} -> recon {tuple(recon.shape)}")
        dist.barrier()
        cleanup_distributed()
        return
    
    
    #### Data init
    first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
    stage1_transform = transforms.Compose(
        [
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )    
    loader, sampler = prepare_dataloader(
        args.data_path, batch_size, num_workers, rank, world_size, transform=stage1_transform, split="train"
    )
    # if do_eval:
    #     eval_dataset = ImageFolder(
    #         str(eval_data),
    #         transform=transforms.Compose([
    #             transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #             transforms.ToTensor(),
    #         ])
    #     )
    #     logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")
    if do_eval:
        eval_cache_dir = eval_data or args.data_path
        eval_dataset = ImageNetArrowDataset(
            cache_dir=eval_cache_dir,
            split="validation",
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ])
        )
        logger.info(
            f"Evaluation dataset loaded from {eval_cache_dir}, containing {len(eval_dataset)} images."
        )

    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")
    
    # Schedl init after knowing dataset length
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    
    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
    
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        logger.info(f"Stage-1 RAE trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Perceptual (LPIPS) weight: {perceptual_weight:.6f}")
        logger.info(f"Gram weight: {gram_weight:.6f}")
        logger.info(f"Gram layers: {gram_layers}")
        logger.info(f"Gram layer weights: {gram_layer_weights if gram_layer_weights is not None else 'uniform'}")
        logger.info(
            f"Gram loss type: {gram_loss_type}, input_range: {gram_input_range}, "
            f"ImageNet norm: {gram_use_imagenet_norm}, VGG weights: {gram_vgg_weights}"
        )
        logger.info(f"LPIPS loss starts at epoch {lpips_start_epoch}.")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler for generator.")
        logger.info(f"Training for {num_epochs} epochs, batch size {batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    lpips_start_step = lpips_start_epoch * steps_per_epoch
    dist.barrier()
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0  and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt" 
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
        for step, (images, _) in enumerate(loader):
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, autocast_kwargs):
                recon = ddp_model(images)  # keep gradient synced
                if recon.dim() == 5 or images.dim() == 5:
                    recon_frames = recon.unsqueeze(1) if recon.dim() == 4 else recon
                    target_frames = images.unsqueeze(1) if images.dim() == 4 else images
                    if target_frames.shape[1] == 1 and recon_frames.shape[1] > 1:
                        target_frames = target_frames.expand(-1, recon_frames.shape[1], -1, -1, -1)
                    if recon_frames.shape[1] == 1 and target_frames.shape[1] > 1:
                        recon_frames = recon_frames.expand(-1, target_frames.shape[1], -1, -1, -1)
                    if recon_frames.shape[1] != target_frames.shape[1]:
                        raise ValueError(
                            f"Mismatched temporal dims: recon {recon_frames.shape} vs target {target_frames.shape}."
                        )
                    l1_per_frame = (recon_frames - target_frames).abs().mean(dim=(2, 3, 4))
                    rec_loss = l1_per_frame.mean()
                    lpips_pred = recon_frames
                    lpips_tgt = target_frames
                else:
                    rec_loss = (recon - images).abs().mean()  # L1
                    lpips_pred = recon
                    lpips_tgt = images

                if use_lpips:
                    if lpips_pred.dim() == 5:
                        batch, frames, channels, height, width = lpips_pred.shape
                        lpips_pred_4d = lpips_pred.reshape(batch * frames, channels, height, width)
                        lpips_tgt_4d = lpips_tgt.reshape(batch * frames, channels, height, width)
                    else:
                        lpips_pred_4d = lpips_pred
                        lpips_tgt_4d = lpips_tgt
                    lpips_loss = lpips(lpips_tgt_4d * 2.0 - 1.0, lpips_pred_4d * 2.0 - 1.0)
                else:
                    lpips_loss = rec_loss.new_zeros(())

                if gram_weight > 0.0:
                    gram_loss = gram_loss_fn(recon, images)
                else:
                    gram_loss = rec_loss.new_zeros(())

                total_loss = rec_loss + perceptual_weight * lpips_loss + gram_weight * gram_loss
            total_loss.float()
            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            update_ema(ema_model, ddp_model.module, ema_decay)

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gram"] += gram_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            do_sample = (sample_every > 0 and global_step % sample_every == 0)
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gram": gram_loss.detach().item(),
                    "gram/weight": gram_weight,
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    # If we also log samples at this step, don't "commit" the step yet.
                    wandb_utils.log(stats, step=global_step, commit=not do_sample)
            if do_sample and rank == 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    # only keep first 4 sample
                    sample_images = images[:4]
                    with _autocast(device, autocast_kwargs):
                        latent, spec = ema_model.encode(sample_images)
                        samples = ema_model.decode(latent, spec)
                    if samples.dim() == 5:
                        samples = samples[:, 0]
                    # also concat input and reconstruction
                    comparison = torch.cat([sample_images, samples], dim=0).cpu().float()
                    # reshape to grid (sample at first row, recon at second row)
                    n = sample_images.size(0)
                    grid = make_grid(comparison, nrow=n)
                    if args.wandb:
                        wandb_utils.log_image(grid, step=global_step, commit=True)
                logger.info("Generating EMA samples done.")
            if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.info("Starting evaluation...")
                eval_models = [(ema_model, "ema")]
                if eval_model:
                    eval_models.append((ddp_model.module, "model"))
                for eval_mod, mod_name in eval_models:
                    eval_stats = evaluate_reconstruction_distributed(
                        eval_mod,
                        eval_dataset,
                        len(eval_dataset),
                        rank = rank,
                        world_size = world_size,
                        device = device,
                        batch_size = batch_size,
                        metrics_to_compute = eval_metrics,
                        experiment_dir = experiment_dir,
                        global_step = global_step,
                        autocast_kwargs = autocast_kwargs,
                        reference_npz_path = reference_npz_path
                    )
                    # log with prefix
                    eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats is not None else {}
                    if args.wandb:
                        wandb_utils.log(eval_stats, step=global_step)
                logger.info("Evaluation done.")
            global_step += 1
        if rank == 0 and num_batches > 0:
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_gram = (epoch_metrics["gram"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()
            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_gram": avg_gram,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
