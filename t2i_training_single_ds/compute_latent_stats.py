#!/usr/bin/env python3
"""
Estimate latent mean/std for stage-2 DiT training.

This script reuses the same RAE encoder + streaming data pipeline style as
training/train.py, then computes channel-wise latent statistics:
    latent shape: (B, T, H, W, C)
    mean/std over all (B, T, H, W) positions for each channel C.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torchvision import transforms
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen3_vl_rae_uncompressed import Qwen3VLRAE
from utils.model_utils import load_rae_decoder_weights


TEXT_KEY_PRIORITY = ["txt", "prompt", "text", "caption", "re_caption", "description"]
IMAGE_KEY_PRIORITY = ["jpg", "jpeg", "png", "webp", "image"]


def _coerce_text(value: Any) -> Optional[str]:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace").strip()
        return text or None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _coerce_text(item)
            if text:
                return text
    return None


def _extract_text(sample: Dict[str, Any], text_keys: List[str]) -> Optional[str]:
    for key in text_keys:
        value = sample.get(key)
        text = _coerce_text(value)
        if text is not None:
            return text
    return None


def _extract_image(sample: Dict[str, Any], image_keys: List[str]):
    from PIL import Image

    for key in image_keys:
        payload = sample.get(key)
        if payload is None:
            continue
        try:
            if isinstance(payload, Image.Image):
                return payload.convert("RGB")
            if isinstance(payload, (bytes, bytearray)):
                return Image.open(io.BytesIO(payload)).convert("RGB")
            if isinstance(payload, dict):
                raw = payload.get("bytes")
                if isinstance(raw, (bytes, bytearray)):
                    return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
    return None


def _select_streaming_columns(
    ds: Any,
    text_keys: List[str],
    image_keys: List[str],
) -> Tuple[Any, List[str], List[str]]:
    """Keep only columns needed for training to avoid schema drift in metadata fields."""
    active_text_keys = list(text_keys)
    active_image_keys = list(image_keys)

    try:
        column_names = list(ds.column_names or [])
    except Exception:
        column_names = []

    if not column_names:
        return ds, active_text_keys, active_image_keys

    present_text = [k for k in text_keys if k in column_names]
    present_image = [k for k in image_keys if k in column_names]

    txt_like = [k for k in present_text if k.lower() in {"txt", "text"} or k.lower().endswith("txt")]
    if txt_like:
        active_text_keys = txt_like
    elif present_text:
        active_text_keys = [present_text[0]]

    if present_image:
        active_image_keys = present_image

    keep_columns = {"__key__", "__url__"}
    keep_columns.update(active_text_keys)
    keep_columns.update(active_image_keys)
    drop_columns = [name for name in column_names if name not in keep_columns]
    if drop_columns:
        ds = ds.remove_columns(drop_columns)

    return ds, active_text_keys, active_image_keys


def build_streaming_dataset(
    data_files: str,
    split: str = "train",
    text_keys: Optional[List[str]] = None,
    image_keys: Optional[List[str]] = None,
) -> Tuple[Any, List[str], List[str]]:
    from datasets import load_dataset

    ds = load_dataset("webdataset", data_files={split: data_files}, split=split, streaming=True)
    active_text_keys = list(text_keys) if text_keys is not None else list(TEXT_KEY_PRIORITY)
    active_image_keys = list(image_keys) if image_keys is not None else list(IMAGE_KEY_PRIORITY)
    ds, active_text_keys, active_image_keys = _select_streaming_columns(
        ds,
        active_text_keys,
        active_image_keys,
    )
    return ds, active_text_keys, active_image_keys


class StreamingTextImageDataset:
    def __init__(
        self,
        data_files: str,
        split: str,
        image_size: int,
        text_keys: Optional[List[str]] = None,
        image_keys: Optional[List[str]] = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.data_files = data_files
        self.split = split
        self.image_size = int(image_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self.text_key_priority = list(text_keys) if text_keys else list(TEXT_KEY_PRIORITY)
        self.image_key_priority = list(image_keys) if image_keys else list(IMAGE_KEY_PRIORITY)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, str]]:
        ds, active_text_keys, active_image_keys = build_streaming_dataset(
            self.data_files,
            self.split,
            text_keys=self.text_key_priority,
            image_keys=self.image_key_priority,
        )
        manual_shard = False
        if self.world_size > 1:
            if hasattr(ds, "shard"):
                ds = ds.shard(num_shards=self.world_size, index=self.rank, contiguous=True)
            else:
                manual_shard = True
        if self.split == "train":
            ds = ds.shuffle(buffer_size=10000, seed=self.seed + self.epoch * self.world_size + self.rank)

        for sample_idx, sample in enumerate(ds):
            if manual_shard and sample_idx % self.world_size != self.rank:
                continue
            text = _extract_text(sample, active_text_keys)
            if text is None:
                continue
            image = _extract_image(sample, active_image_keys)
            if image is None:
                continue
            try:
                image_tensor = self.transform(image)
            except Exception:
                continue
            yield image_tensor, text


def collate_streaming_batch(
    data_iter: Iterator[Tuple[torch.Tensor, str]],
    batch_size: int,
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    images: List[torch.Tensor] = []
    texts: List[str] = []
    for image, text in data_iter:
        images.append(image)
        texts.append(text)
        if len(images) >= batch_size:
            break
    if len(images) < batch_size:
        return None
    return torch.stack(images), texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute latent mean/std for stage-2 DiT.")
    parser.add_argument("--config", type=str, default="training/config/dit_training.yaml")
    parser.add_argument("--output", type=str, default="ckpts/latent_stats.pt")
    parser.add_argument("--num-batches", type=int, default=256, help="Number of batches to encode.")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size. 0 means from config.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'.")
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def _resolve_config_path(path_str: str) -> Path:
    cfg = Path(path_str)
    if cfg.exists():
        return cfg
    fallback = Path(__file__).resolve().parent / "config" / cfg.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Config not found: '{path_str}'. Also tried '{fallback}'.")


def _resolve_decoder_config_path(path_str: str) -> str:
    p = Path(path_str)
    if p.exists():
        return str(p)
    fallback = Path(__file__).resolve().parent.parent / "config" / "decoder_config.json"
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError(
        f"decoder_config_path not found: '{path_str}'. Also tried '{fallback}'."
    )


def _tensor_assignment_lines(name: str, values: List[float], per_line: int = 8) -> List[str]:
    lines = [f"{name} = torch.tensor(["]
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        row = ", ".join(f"{float(v):.10g}" for v in chunk)
        lines.append(f"    {row},")
    lines.append("], dtype=torch.float32)")
    return lines


def _write_python_stats_file(
    output_path: Path,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    global_mean: float,
    global_std: float,
    token_count: int,
    channel_dim: int,
    num_batches: int,
    batch_size: int,
    image_size: int,
    config_path: str,
    data_files: str,
    split: str,
    seed: int,
    precision: str,
) -> None:
    mean_values = mean.to(torch.float32).cpu().tolist()
    std_values = std.to(torch.float32).cpu().tolist()

    lines: List[str] = []
    lines.append('"""Auto-generated latent statistics."""')
    lines.append("import torch")
    lines.append("")
    lines.extend(_tensor_assignment_lines("MEAN", mean_values))
    lines.append("")
    lines.extend(_tensor_assignment_lines("STD", std_values))
    lines.append("")
    lines.append(f"GLOBAL_MEAN = {global_mean:.10g}")
    lines.append(f"GLOBAL_STD = {global_std:.10g}")
    lines.append(f"TOKEN_COUNT = {int(token_count)}")
    lines.append(f"CHANNEL_DIM = {int(channel_dim)}")
    lines.append(f"NUM_BATCHES = {int(num_batches)}")
    lines.append(f"BATCH_SIZE = {int(batch_size)}")
    lines.append(f"IMAGE_SIZE = {int(image_size)}")
    lines.append(f"CONFIG_PATH = {config_path!r}")
    lines.append(f"DATA_FILES = {data_files!r}")
    lines.append(f"SPLIT = {split!r}")
    lines.append(f"SEED = {int(seed)}")
    lines.append(f"PRECISION = {precision!r}")
    lines.append("")
    lines.append("def get_latent_stats(device=None, dtype=torch.float32):")
    lines.append("    mean = MEAN.to(device=device, dtype=dtype)")
    lines.append("    std = STD.to(device=device, dtype=dtype)")
    lines.append("    return {")
    lines.append('        "mean": mean,')
    lines.append('        "std": std,')
    lines.append('        "global_mean": GLOBAL_MEAN,')
    lines.append('        "global_std": GLOBAL_STD,')
    lines.append('        "token_count": TOKEN_COUNT,')
    lines.append('        "channel_dim": CHANNEL_DIM,')
    lines.append("    }")
    lines.append("")
    lines.append("LATENT_STATS = get_latent_stats(device='cpu', dtype=torch.float32)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    from omegaconf import OmegaConf

    config_path = _resolve_config_path(args.config)
    cfg = OmegaConf.load(str(config_path))
    data_cfg = cfg.get("data", {})
    rae_cfg = cfg.get("rae", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    is_main = rank == 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed mode requires CUDA.")
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    def log(msg: str) -> None:
        if is_main:
            print(msg)

    global_seed = int(train_cfg.get("global_seed", 42))
    if args.seed is not None:
        global_seed = int(args.seed)
    # Make per-rank RNG streams deterministic but distinct.
    torch.manual_seed(global_seed + rank)
    torch.cuda.manual_seed_all(global_seed + rank)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    if args.precision == "fp32":
        autocast_dtype = torch.float32
        autocast_enabled = False
    elif args.precision == "fp16":
        autocast_dtype = torch.float16
        autocast_enabled = True
    else:
        autocast_dtype = torch.bfloat16
        autocast_enabled = True

    image_size = int(data_cfg.get("image_size", 448))
    global_batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(train_cfg.get("global_batch_size", 32))
    if global_batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by world_size={world_size}."
        )
    batch_size = global_batch_size // world_size

    if args.num_batches <= 0:
        raise ValueError("num_batches must be > 0.")
    global_num_batches = int(args.num_batches)
    base_local_batches = global_num_batches // world_size
    remainder = global_num_batches % world_size
    local_num_batches = base_local_batches + (1 if rank < remainder else 0)

    log(f"[Info] Config: {config_path}")
    log(f"[Info] Device: {device}, precision={args.precision}")
    log(
        f"[Info] world_size={world_size}, global_batch_size={global_batch_size}, "
        f"local_batch_size={batch_size}, global_num_batches={global_num_batches}"
    )
    log(f"[Info] Image size: {image_size}")

    rae = Qwen3VLRAE(
        model_name_or_path=str(rae_cfg.get("model_name_or_path", "Qwen/Qwen3-VL-4B-Instruct")),
        decoder_config_path=_resolve_decoder_config_path(str(rae_cfg.get("decoder_config_path", "config/decoder_config.json"))),
        noise_tau=float(rae_cfg.get("noise_tau", 0.0)),
        in_channels=int(rae_cfg.get("in_channels", 3)),
        denormalize_output=bool(rae_cfg.get("denormalize_output", True)),
        local_files_only=bool(rae_cfg.get("local_files_only", True)),
        do_resize=bool(rae_cfg.get("do_resize", False)),
    ).to(device)
    rae.requires_grad_(False)
    rae.eval()

    decoder_ckpt_path = str(rae_cfg.get("decoder_checkpoint_path", "")).strip()
    if decoder_ckpt_path:
        info = load_rae_decoder_weights(
            rae=rae,
            checkpoint_path=decoder_ckpt_path,
            source=str(rae_cfg.get("decoder_checkpoint_source", "ema")),
        )
        log(
            "[Info] Loaded decoder checkpoint: "
            f"path={info['checkpoint_path']}, source={info['source']}, "
            f"decoder_keys={info['decoder_keys']}, to_pixels_keys={info['to_pixels_keys_in_ckpt']}"
        )

    dataset = StreamingTextImageDataset(
        data_files=str(data_cfg.get("data_files", "")),
        split=str(data_cfg.get("split", "train")),
        image_size=image_size,
        text_keys=list(data_cfg.get("text_keys", TEXT_KEY_PRIORITY)),
        image_keys=list(data_cfg.get("image_keys", IMAGE_KEY_PRIORITY)),
        rank=rank,
        world_size=world_size,
        seed=global_seed,
    )
    stream_epoch = 0
    dataset.set_epoch(stream_epoch)
    data_iter = iter(dataset)

    channels = int(model_cfg.get("rae_latent_dim", 1024))
    sum_c = torch.zeros(channels, device=device, dtype=torch.float64)
    sum_sq_c = torch.zeros(channels, device=device, dtype=torch.float64)
    token_count = 0

    pbar = None
    if is_main and tqdm is not None:
        pbar = tqdm(
            total=local_num_batches,
            desc="Encoding batches (rank0)",
            unit="batch",
            dynamic_ncols=True,
        )
    elif is_main:
        log("[Warn] tqdm is unavailable, fallback to text progress logs.")

    local_step_estimate = 0
    for step in range(local_num_batches):
        batch = collate_streaming_batch(data_iter, batch_size)
        while batch is None:
            stream_epoch += 1
            dataset.set_epoch(stream_epoch)
            data_iter = iter(dataset)
            batch = collate_streaming_batch(data_iter, batch_size)

        images, _ = batch
        images = images.to(device=device, non_blocking=True)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=autocast_enabled):
                latent, _ = rae.encode(images)  # (B, T, H, W, C)

        latent_f32 = latent.to(torch.float32)
        if int(latent_f32.shape[-1]) != channels:
            raise ValueError(
                f"Unexpected latent channels {latent_f32.shape[-1]} (expected {channels})."
            )
        flat = latent_f32.view(-1, channels)

        sum_c += flat.sum(dim=0, dtype=torch.float64)
        sum_sq_c += (flat * flat).sum(dim=0, dtype=torch.float64)
        token_count += int(flat.shape[0])

        if pbar is not None:
            pbar.update(1)
            if (step + 1) % max(1, args.log_interval) == 0 or (step + 1) == local_num_batches:
                pbar.set_postfix(tokens=token_count)
        elif is_main:
            local_step_estimate += 1
            if local_step_estimate % max(1, args.log_interval) == 0:
                log(f"[Progress] {local_step_estimate}/{local_num_batches} local batches (rank0)")

    if pbar is not None:
        pbar.close()

    count_tensor = torch.tensor([float(token_count)], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(sum_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_sq_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    token_count = int(count_tensor.item())
    if token_count <= 0:
        raise RuntimeError("No valid samples were processed.")

    mean = sum_c / float(token_count)
    var = (sum_sq_c / float(token_count)) - mean.pow(2)
    var = torch.clamp(var, min=1e-8)
    std = torch.sqrt(var)

    channel_dim = int(mean.numel())
    total_values = float(token_count * channel_dim)
    global_mean = float(sum_c.sum().item() / total_values)
    global_second_moment = float(sum_sq_c.sum().item() / total_values)
    global_var = max(1e-8, global_second_moment - global_mean * global_mean)
    global_std = math.sqrt(global_var)

    if is_main:
        print("[Result] Channel-wise stats")
        print(f"  mean_abs_avg: {float(mean.abs().mean().item()):.6f}")
        print(f"  std_avg:      {float(std.mean().item()):.6f}")
        print(f"  std_min/max:  {float(std.min().item()):.6f} / {float(std.max().item()):.6f}")
        print("[Result] Global stats")
        print(f"  mean: {global_mean:.6f}")
        print(f"  std:  {global_std:.6f}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mean": mean.to(torch.float32).cpu(),
                "std": std.to(torch.float32).cpu(),
                "global_mean": global_mean,
                "global_std": global_std,
                "token_count": int(token_count),
                "channel_dim": channel_dim,
                "num_batches": int(global_num_batches),
                "batch_size": int(global_batch_size),
                "local_batch_size": int(batch_size),
                "world_size": int(world_size),
                "image_size": int(image_size),
                "config_path": str(config_path),
                "data_files": str(data_cfg.get("data_files", "")),
                "split": str(data_cfg.get("split", "train")),
                "seed": int(global_seed),
                "precision": args.precision,
            },
            output_path,
        )

        if output_path.suffix:
            output_py_path = output_path.with_suffix(".py")
        else:
            output_py_path = Path(str(output_path) + ".py")

        _write_python_stats_file(
            output_path=output_py_path,
            mean=mean,
            std=std,
            global_mean=global_mean,
            global_std=global_std,
            token_count=int(token_count),
            channel_dim=channel_dim,
            num_batches=int(global_num_batches),
            batch_size=int(global_batch_size),
            image_size=int(image_size),
            config_path=str(config_path),
            data_files=str(data_cfg.get("data_files", "")),
            split=str(data_cfg.get("split", "train")),
            seed=int(global_seed),
            precision=args.precision,
        )

        print(f"[Done] Saved latent stats to: {output_path}")
        print(f"[Done] Saved Python latent stats to: {output_py_path}")

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
