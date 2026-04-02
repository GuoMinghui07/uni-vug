# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE training script with reconstruction, LPIPS, Gram, and optional GAN losses.

This script adapts the training logic from the Kakao Brain VQGAN trainer while
supporting the Qwen3-VL RAE architecture in this repository.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from contextlib import nullcontext
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
from disc import (
    DiffAug,
    LPIPS,
    GramLossVGG19,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import *
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage-1 RAE with reconstruction, LPIPS, Gram, and optional GAN losses."
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config containing a stage_1 section.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="HuggingFace datasets cache dir for ImageNet-1k.",
    )
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=448, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument(
        "--global-seed",
        type=int,
        default=None,
        help="Override training.global_seed from the config.",
    )
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging if set.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode, rae.forward).")
    parser.add_argument("--sanity", action="store_true", help="Run a single forward pass and exit.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint file path to resume from (can be from another experiment).",
    )

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


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: Optional[torch.nn.Module] = None,
    disc_optimizer: Optional[torch.optim.Optimizer] = None,
    disc_scheduler: Optional[LambdaLR] = None,
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    if disc is not None:
        state["disc"] = disc.state_dict()
    if disc_optimizer is not None:
        state["disc_optimizer"] = disc_optimizer.state_dict()
    if disc_scheduler is not None:
        state["disc_scheduler"] = disc_scheduler.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: Optional[torch.nn.Module] = None,
    disc_optimizer: Optional[torch.optim.Optimizer] = None,
    disc_scheduler: Optional[LambdaLR] = None,
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if disc is not None and checkpoint.get("disc") is not None:
        disc.load_state_dict(checkpoint["disc"])
    if disc_optimizer is not None and checkpoint.get("disc_optimizer") is not None:
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])

    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)


def _select_disc_frame(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        return tensor
    if tensor.dim() == 5:
        return tensor[:, 0]
    raise ValueError(f"Expected 4D/5D tensor for discriminator path, got {tuple(tensor.shape)}")


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

    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    gan_cfg = dict(gan_cfg) if isinstance(gan_cfg, dict) else {}
    disc_cfg = gan_cfg.get("disc", {}) if isinstance(gan_cfg.get("disc", {}), dict) else {}

    loss_section = full_cfg.get("loss", None)
    loss_cfg = OmegaConf.to_container(loss_section, resolve=True) if loss_section is not None else {}
    loss_cfg = dict(loss_cfg) if isinstance(loss_cfg, dict) else {}
    if not loss_cfg:
        loss_cfg = dict(gan_cfg.get("loss", {})) if isinstance(gan_cfg.get("loss", {}), dict) else {}

    gram_section = full_cfg.get("gram", None)
    gram_cfg = OmegaConf.to_container(gram_section, resolve=True) if gram_section is not None else {}
    gram_cfg = dict(gram_cfg) if isinstance(gram_cfg, dict) else {}

    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    lpips_start_epoch = float(loss_cfg.get("lpips_start", 0))

    # GAN controls (optional)
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_warmup_steps = int(float(loss_cfg.get("disc_start", 0)))
    disc_update_warmup_steps = int(float(loss_cfg.get("disc_upd_start", gan_warmup_steps)))
    if gan_warmup_steps < 0 or disc_update_warmup_steps < 0:
        raise ValueError("disc_start/disc_upd_start must be >= 0 warmup steps.")
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")
    gan_enabled = disc_weight > 0.0

    if gan_enabled and not disc_cfg:
        raise ValueError("disc_weight > 0 requires gan.disc configuration.")

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
    gram_loss_type = (
        args.gram_loss_type
        if args.gram_loss_type is not None
        else gram_cfg.get("loss_type", gram_cfg.get("gram_loss_type", "l1"))
    )
    gram_input_range = (
        args.gram_input_range
        if args.gram_input_range is not None
        else gram_cfg.get("input_range", gram_cfg.get("gram_input_range", "0_1"))
    )
    gram_use_imagenet_norm = _coerce_bool(
        args.gram_use_imagenet_norm
        if args.gram_use_imagenet_norm is not None
        else gram_cfg.get("use_imagenet_norm", gram_cfg.get("gram_use_imagenet_norm", True)),
        True,
    )
    gram_vgg_weights = (
        args.gram_vgg_weights
        if args.gram_vgg_weights is not None
        else gram_cfg.get("vgg_weights", gram_cfg.get("gram_vgg_weights", "IMAGENET1K_V1"))
    )
    if gram_layer_weights is not None and len(gram_layer_weights) != len(gram_layers):
        raise ValueError("gram_layer_weights must have the same length as gram_layers.")

    accumulate_steps = int(training_cfg.get("accumulate_steps", 2))
    if accumulate_steps < 1:
        raise ValueError("training.accumulate_steps must be >= 1.")

    batch_size = int(training_cfg.get("batch_size", 16))
    global_batch_size = training_cfg.get("global_batch_size", None)
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        if global_batch_size % accumulate_steps != 0:
            raise ValueError("global_batch_size must be divisible by accumulate_steps.")
        micro_global_batch_size = global_batch_size // accumulate_steps
        if micro_global_batch_size % world_size != 0:
            raise ValueError("global_batch_size / accumulate_steps must be divisible by world_size.")
        batch_size = micro_global_batch_size // world_size
    else:
        micro_global_batch_size = batch_size * world_size
        global_batch_size = micro_global_batch_size * accumulate_steps

    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None

    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 1250))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4))
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))

    eval_section = full_cfg.get("eval", None)
    if eval_section:
        do_eval = True
        eval_interval = int(eval_section.get("eval_interval", 5000))
        eval_model = eval_section.get("eval_model", False)
        eval_metrics = eval_section.get("metrics", ("rfid", "psnr", "ssim"))
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

    rae.requires_grad_(False)
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)
    if hasattr(rae, "demerge"):
        rae.demerge.requires_grad_(True)
    if hasattr(rae, "to_pixels"):
        rae.to_pixels.requires_grad_(True)

    ddp_model = DDP(
        rae,
        device_ids=[device.index],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )  # type: ignore[arg-type]
    rae = ddp_model.module
    decoder = ddp_model.module.decoder

    discriminator: Optional[torch.nn.Module] = None
    ddp_disc: Optional[DDP] = None
    disc_aug: Optional[DiffAug] = None
    disc_optimizer: Optional[torch.optim.Optimizer] = None
    disc_scheduler: Optional[LambdaLR] = None
    disc_optim_msg: Optional[str] = None
    disc_sched_msg: Optional[str] = None
    disc_loss_fn = None
    gen_loss_fn = None

    if gan_enabled:
        discriminator, disc_aug = build_discriminator(disc_cfg, device)
        ddp_disc = DDP(
            discriminator,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )  # type: ignore[arg-type]
        discriminator = ddp_disc.module
        discriminator.train()
        disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)

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

    #### Opt, Sched init
    decoder_params = list(decoder.parameters())
    if hasattr(ddp_model.module, "demerge"):
        decoder_params += list(ddp_model.module.demerge.parameters())
    if hasattr(ddp_model.module, "to_pixels"):
        decoder_params += list(ddp_model.module.to_pixels.parameters())
    optimizer, optim_msg = build_optimizer(decoder_params, training_cfg)

    if gan_enabled:
        assert discriminator is not None
        disc_params = [p for p in discriminator.parameters() if p.requires_grad]
        disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)

    #### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    disc_scaler: Optional[GradScaler] = GradScaler() if (gan_enabled and scaler is not None) else None
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
        args.data_path,
        batch_size,
        num_workers,
        rank,
        world_size,
        transform=stage1_transform,
        split="train",
    )

    if do_eval:
        eval_cache_dir = eval_data or args.data_path
        eval_dataset = ImageNetArrowDataset(
            cache_dir=eval_cache_dir,
            split="validation",
            transform=transforms.Compose(
                [
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                    transforms.ToTensor(),
                ]
            ),
        )
        logger.info(f"Evaluation dataset loaded from {eval_cache_dir}, containing {len(eval_dataset)} images.")

    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")

    gen_steps_per_epoch = (steps_per_epoch + accumulate_steps - 1) // accumulate_steps
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, gen_steps_per_epoch, training_cfg)
    if gan_enabled and disc_cfg.get("scheduler"):
        assert disc_optimizer is not None
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, gen_steps_per_epoch, disc_cfg)

    ### Resume and checkpoint
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = args.resume if args.resume is not None else find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        if args.resume is not None:
            logger.info(f"Explicit resume checkpoint provided: {maybe_resume_ckpt_path}")
        else:
            logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")

    ### Logging run details
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        logger.info(f"Stage-1 RAE trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Perceptual (LPIPS) weight: {perceptual_weight:.6f}")
        logger.info(f"Gram weight: {gram_weight:.6f}")
        logger.info(f"Gram layers: {gram_layers}")
        logger.info(
            f"Gram layer weights: {gram_layer_weights if gram_layer_weights is not None else 'uniform'}"
        )
        logger.info(
            f"Gram loss type: {gram_loss_type}, input_range: {gram_input_range}, "
            f"ImageNet norm: {gram_use_imagenet_norm}, VGG weights: {gram_vgg_weights}"
        )
        logger.info(f"LPIPS loss starts at epoch {lpips_start_epoch}.")

        if gan_enabled:
            assert discriminator is not None
            logger.info(f"Discriminator architecture:\n{discriminator}")
            disc_param_count = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
            logger.info(f"Discriminator trainable parameters: {disc_param_count/1e6:.2f}M")
            logger.info(f"Using {disc_loss_type} discriminator loss and {gen_loss_type} generator loss.")
            logger.info(
                f"GAN weight: {disc_weight:.6f}, GAN starts after {gan_warmup_steps} warmup steps, "
                f"discriminator updates start after {disc_update_warmup_steps} warmup steps."
            )
            logger.info(f"Discriminator updates per step: {disc_updates}, max adaptive weight: {max_d_weight}.")
            if disc_aug is not None:
                logger.info(f"Using DiffAug with policies: {disc_aug}")
            if disc_optim_msg is not None:
                logger.info(disc_optim_msg)
            print(disc_sched_msg if disc_sched_msg else "No LR scheduler for discriminator.")
        else:
            logger.info("GAN disabled (disc_weight <= 0).")

        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        logger.info(f"Gradient accumulation steps: {accumulate_steps}.")

        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler for generator.")
        logger.info(f"Training for {num_epochs} epochs, micro-batch size {batch_size} per GPU.")
        logger.info(f"Micro global batch size per step: {micro_global_batch_size}.")
        logger.info(f"Optimizer steps per epoch (after accumulation): {gen_steps_per_epoch}.")
        logger.info(f"Effective global batch size (with accumulation): {global_batch_size}.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")

    if hasattr(ddp_model.module, "to_pixels"):
        last_layer = ddp_model.module.to_pixels.weight
    else:
        last_layer = decoder.decoder_pred.weight

    run_start_step = global_step
    gan_start_step = run_start_step + gan_warmup_steps
    disc_update_step = run_start_step + disc_update_warmup_steps
    lpips_start_step = int(lpips_start_epoch * gen_steps_per_epoch)
    if rank == 0 and gan_enabled:
        logger.info(
            f"GAN absolute start step: {gan_start_step}, discriminator absolute start step: {disc_update_step}, "
            f"current start step: {run_start_step}."
        )

    dist.barrier()
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0

        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0 and rank == 0:
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
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )

        optimizer.zero_grad(set_to_none=True)
        if disc_optimizer is not None:
            disc_optimizer.zero_grad(set_to_none=True)
        remainder_steps = steps_per_epoch % accumulate_steps
        for step, (images, _) in enumerate(loader):
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            use_gan = gan_enabled and global_step >= gan_start_step and disc_weight > 0.0
            train_disc = gan_enabled and global_step >= disc_update_step and disc_weight > 0.0
            is_last_batch = (step + 1) == steps_per_epoch
            do_gen_step = ((step + 1) % accumulate_steps == 0) or is_last_batch
            do_disc_step = do_gen_step

            images = images.to(device, non_blocking=True)
            real_normed = _select_disc_frame(images) * 2.0 - 1.0

            if discriminator is not None:
                discriminator.eval()
                # Freeze D params during G update: keep grad path to fake inputs only.
                discriminator.requires_grad_(False)

            if remainder_steps > 0 and step >= steps_per_epoch - remainder_steps:
                grad_divisor = remainder_steps
            else:
                grad_divisor = accumulate_steps

            sync_context = nullcontext() if do_gen_step else ddp_model.no_sync()
            with sync_context:
                with _autocast(device, autocast_kwargs):
                    recon = ddp_model(images)

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
                        rec_loss = (recon - images).abs().mean()
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

                    recon_total = rec_loss + perceptual_weight * lpips_loss + gram_weight * gram_loss

                    if use_gan:
                        assert ddp_disc is not None and gen_loss_fn is not None
                        recon_for_gan = _select_disc_frame(lpips_pred)
                        recon_normed = recon_for_gan * 2.0 - 1.0
                        fake_aug = disc_aug.aug(recon_normed) if disc_aug is not None else recon_normed
                        logits_fake, _ = ddp_disc(fake_aug, None)
                        gan_loss = gen_loss_fn(logits_fake)
                    else:
                        gan_loss = rec_loss.new_zeros(())

                if use_gan:
                    adaptive_weight = calculate_adaptive_weight(recon_total, gan_loss, last_layer, max_d_weight)
                    total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
                else:
                    adaptive_weight = rec_loss.new_zeros(())
                    total_loss = recon_total

                scaled_total_loss = total_loss / float(grad_divisor)

                scaled_total_loss.float()
                if scaler:
                    scaler.scale(scaled_total_loss).backward()
                else:
                    scaled_total_loss.backward()

            if do_gen_step:
                if scaler:
                    if clip_grad is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                update_ema(ema_model, ddp_model.module, ema_decay)

            disc_metrics: Dict[str, torch.Tensor] = {}
            if train_disc:
                assert ddp_disc is not None
                assert discriminator is not None
                assert disc_optimizer is not None
                assert disc_loss_fn is not None

                ddp_model.eval()
                ddp_disc.train()
                discriminator.requires_grad_(True)
                disc_sync_context = nullcontext() if do_disc_step else ddp_disc.no_sync()
                with disc_sync_context:
                    for _ in range(disc_updates):
                        with _autocast(device, autocast_kwargs):
                            with torch.no_grad():
                                recon_disc = ddp_model.module(images)
                                recon_disc = _select_disc_frame(recon_disc)
                                recon_disc_normed = recon_disc * 2.0 - 1.0

                            fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                            fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0

                            fake_input = disc_aug.aug(fake_detached) if disc_aug is not None else fake_detached
                            real_input = disc_aug.aug(real_normed) if disc_aug is not None else real_normed
                            logits_fake, logits_real = ddp_disc(fake_input, real_input)
                            d_loss = disc_loss_fn(logits_real, logits_fake)
                            accuracy = (logits_real > logits_fake).float().mean()

                        scaled_d_loss = d_loss / float(grad_divisor * disc_updates)
                        scaled_d_loss.float()
                        active_disc_scaler = disc_scaler if disc_scaler is not None else scaler
                        if active_disc_scaler is not None:
                            active_disc_scaler.scale(scaled_d_loss).backward()
                        else:
                            scaled_d_loss.backward()

                        disc_metrics = {
                            "disc_loss": d_loss.detach(),
                            "logits_real": logits_real.detach().mean(),
                            "logits_fake": logits_fake.detach().mean(),
                            "disc_accuracy": accuracy.detach(),
                        }
                        epoch_metrics["disc_loss"] += d_loss.detach()
                        epoch_metrics["disc_accuracy"] += accuracy.detach()

                if do_disc_step:
                    active_disc_scaler = disc_scaler if disc_scaler is not None else scaler
                    if active_disc_scaler is not None:
                        active_disc_scaler.step(disc_optimizer)
                        active_disc_scaler.update()
                    else:
                        disc_optimizer.step()
                    disc_optimizer.zero_grad(set_to_none=True)

                    if disc_scheduler is not None:
                        disc_scheduler.step()

                ddp_disc.eval()
                discriminator.requires_grad_(False)
                ddp_model.train()

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gram"] += gram_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            do_sample = do_gen_step and sample_every > 0 and global_step % sample_every == 0
            if do_gen_step and log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gram": gram_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "gram/weight": gram_weight,
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if disc_metrics and disc_optimizer is not None:
                    stats.update(
                        {
                            "loss/disc": disc_metrics["disc_loss"].item(),
                            "disc/logits_real": disc_metrics["logits_real"].item(),
                            "disc/logits_fake": disc_metrics["logits_fake"].item(),
                            "disc/accuracy": disc_metrics["disc_accuracy"].item(),
                            "disc/weight": adaptive_weight.item(),
                            "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                        }
                    )

                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step, commit=not do_sample)

            if do_sample and rank == 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    sample_images = images[:4]
                    with _autocast(device, autocast_kwargs):
                        latent, spec = ema_model.encode(sample_images)
                        samples = ema_model.decode(latent, spec)
                    if samples.dim() == 5:
                        samples = samples[:, 0]

                    comparison = torch.cat([sample_images, samples], dim=0).cpu().float()
                    n = sample_images.size(0)
                    grid = make_grid(comparison, nrow=n)
                    if args.wandb:
                        wandb_utils.log_image(grid, step=global_step, commit=True)
                logger.info("Generating EMA samples done.")

            if do_gen_step and do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.info("Starting evaluation...")
                eval_models = [(ema_model, "ema")]
                if eval_model:
                    eval_models.append((ddp_model.module, "model"))
                for eval_mod, mod_name in eval_models:
                    eval_stats = evaluate_reconstruction_distributed(
                        eval_mod,
                        eval_dataset,
                        len(eval_dataset),
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        batch_size=batch_size,
                        metrics_to_compute=eval_metrics,
                        experiment_dir=experiment_dir,
                        global_step=global_step,
                        autocast_kwargs=autocast_kwargs,
                        reference_npz_path=reference_npz_path,
                    )
                    eval_stats = (
                        {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()}
                        if eval_stats is not None
                        else {}
                    )
                    if args.wandb:
                        wandb_utils.log(eval_stats, step=global_step)
                logger.info("Evaluation done.")

            if do_gen_step:
                global_step += 1

        if rank == 0 and num_batches > 0:
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_gram = (epoch_metrics["gram"] / num_batches).item()
            avg_gan = (epoch_metrics["gan"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()

            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_gram": avg_gram,
                "epoch/loss_gan": avg_gan,
            }
            if gan_enabled:
                epoch_stats["epoch/loss_disc"] = (epoch_metrics["disc_loss"] / num_batches).item()
                epoch_stats["epoch/disc_accuracy"] = (epoch_metrics["disc_accuracy"] / num_batches).item()

            logger.info(
                f"[Epoch {epoch}] " + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)

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
            discriminator,
            disc_optimizer,
            disc_scheduler,
        )

    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
