"""Training utilities: EMA, experiment dirs, and mixed WebDataset loading helpers."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import webdataset as wds
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

__all__ = [
    "update_ema",
    "configure_experiment_dirs",
    "prepare_dataloader",
    "center_crop_arr",
    "ImageNetArrowDataset",
    "parse_configs",
    "build_streaming_dataloader",
    "DEFAULT_STREAMING_TEXT_KEYS",
    "DEFAULT_STREAMING_IMAGE_KEYS",
    "DEFAULT_MIXED_DATASETS",
]

Image.MAX_IMAGE_PIXELS = 1024 * 1024 * 256
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_STREAMING_TEXT_KEYS = ("txt", "json", "caption")
DEFAULT_STREAMING_IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")
DEFAULT_MIXED_DATASETS = (
    (
        "pd12m",
        "https://huggingface.co/datasets/Spawning/pd12m-full/resolve/main/{00155..02663}.tar",
        0.7,
    ),
    (
        "t2i2m",
        "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{000000..000046}.tar",
        0.2,
    ),
    (
        "fine_t2i",
        "https://huggingface.co/datasets/ma-xu/fine-t2i/resolve/main/synthetic_enhanced_prompt_square_resolution/train-{000000..001543}.tar",
        0.1,
    ),
)

_ERROR_COUNTER = Counter()


def _convert_to_rgb(image):
    return image.convert("RGB")


def _has_non_empty_text(sample: Tuple[Any, Any]) -> bool:
    return bool(sample[1])


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update exponential moving average parameters."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def configure_experiment_dirs(args, rank: int) -> Tuple[str, str, logging.Logger]:
    """Create experiment directory, checkpoint subdir, and logger."""
    results_dir = getattr(args, "results_dir", "ckpts")
    config_name = getattr(args, "config", "default")
    if isinstance(config_name, str):
        config_name = Path(config_name).stem

    experiment_dir = os.path.join(results_dir, config_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    if rank == 0:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    logger = logging.getLogger(f"train_rank{rank}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f"[Rank {rank}] %(message)s"))
        logger.addHandler(handler)
        if rank == 0:
            fh = logging.FileHandler(os.path.join(experiment_dir, "train.log"), mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s [Rank %(name)s] %(message)s"))
            logger.addHandler(fh)

    return experiment_dir, checkpoint_dir, logger


def prepare_dataloader(
    data_path,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
    transform=None,
    split: str = "train",
) -> Tuple[DataLoader, DistributedSampler]:
    """Build ImageFolder DataLoader with DistributedSampler."""
    dataset = ImageFolder(str(data_path), transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=(split == "train")
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


def center_crop_arr(pil_image, image_size: int):
    """Center-crop PIL image to square."""
    w, h = pil_image.size
    crop = min(w, h)
    left = (w - crop) // 2
    top = (h - crop) // 2
    return pil_image.crop((left, top, left + crop, top + crop)).resize(
        (image_size, image_size), Image.BICUBIC
    )


class ImageNetArrowDataset(torch.utils.data.Dataset):
    """Placeholder for arrow-based ImageNet dataset."""

    def __init__(self, cache_dir, split="validation", transform=None):
        from datasets import load_dataset

        self.ds = load_dataset("imagenet-1k", split=split, cache_dir=cache_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if self.transform:
            img = self.transform(img)
        return img, item.get("label", 0)


def parse_configs(cfg):
    """Extract stage configs from OmegaConf. Returns (stage_1_config, ...)."""
    stage_1 = cfg.get("stage_1", None)
    if stage_1 is None:
        raise ValueError("Config must define a 'stage_1' section.")
    return (stage_1,)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore").strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return _normalize_text(json.loads(text))
            except json.JSONDecodeError:
                return text
        return text
    if isinstance(value, Mapping):
        for key in ("caption", "prompt", "text", "txt", "description", "title"):
            if key in value and value[key] is not None:
                return _normalize_text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_normalize_text(v) for v in value if v is not None).strip()
    return str(value).strip()


class _ErrorHandler:
    def __init__(self, dataset_name: str, max_errors_per_worker: int) -> None:
        self.dataset_name = dataset_name
        self.max_errors_per_worker = int(max_errors_per_worker)

    def __call__(self, exc: Exception):
        _ERROR_COUNTER[f"{self.dataset_name}:{type(exc).__name__}"] += 1
        total_errors = sum(_ERROR_COUNTER.values())
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rank = int(os.getenv("RANK", "0"))

        if rank == 0 and worker_id == 0 and (total_errors <= 3 or total_errors % 100 == 0):
            message = " ".join(str(exc).split())[:220]
            print(
                f"[warn] {self.dataset_name} skip {type(exc).__name__}, total_errors={total_errors}, err={message}",
                flush=True,
            )

        if self.max_errors_per_worker > 0 and total_errors >= self.max_errors_per_worker:
            raise RuntimeError(
                f"[fatal] {self.dataset_name} too many data errors: {total_errors}"
            ) from exc
        return True


def _build_curl_url(
    *,
    url: str,
    connect_timeout: int,
    max_time: int,
    retry: int,
    speed_time: int,
    speed_limit: int,
    show_errors: bool,
) -> str:
    show_err_flag = "S" if show_errors else ""
    parts = [
        f"pipe:curl -s{show_err_flag}Lf",
        f"--connect-timeout {connect_timeout}",
        f"--retry {retry}",
        "--retry-delay 1",
    ]
    if int(max_time) > 0:
        parts.append(f"--max-time {int(max_time)}")
    if int(speed_time) > 0 and int(speed_limit) > 0:
        parts.append(f"--speed-time {int(speed_time)} --speed-limit {int(speed_limit)}")
    parts.append(str(url))
    return " ".join(parts)


def _parse_dataset_specs(
    datasets: Optional[Sequence[Any]],
    data_files: Optional[str],
) -> List[Tuple[str, str, float]]:
    raw: Sequence[Any]
    if datasets is not None:
        raw = datasets
    elif data_files:
        raw = [("default", str(data_files), 1.0)]
    else:
        raw = DEFAULT_MIXED_DATASETS

    parsed: List[Tuple[str, str, float]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, Mapping):
            name = str(item.get("name", f"dataset_{idx}")).strip()
            url = str(item.get("url", "")).strip()
            weight = float(item.get("weight", 1.0))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            name = str(item[0]).strip()
            url = str(item[1]).strip()
            weight = float(item[2])
        else:
            raise ValueError(
                "Each dataset entry must be {'name','url','weight'} or (name, url, weight)."
            )
        if not url:
            raise ValueError(f"Dataset[{idx}] missing url.")
        if weight <= 0:
            raise ValueError(f"Dataset[{idx}] weight must be > 0, got {weight}.")
        parsed.append((name or f"dataset_{idx}", url, weight))

    if not parsed:
        raise ValueError("At least one dataset must be configured for mixed training.")
    return parsed


def _build_stream(
    *,
    dataset_name: str,
    dataset_url: str,
    transform,
    shuffle_buffer_size: int,
    seed: int,
    max_errors_per_worker: int,
    image_keys: Sequence[str],
    text_keys: Sequence[str],
    curl_connect_timeout: int,
    curl_max_time: int,
    curl_retry: int,
    curl_speed_time: int,
    curl_speed_limit: int,
    curl_show_errors: bool,
):
    if not hasattr(wds, "split_by_node") or not hasattr(wds, "split_by_worker"):
        raise RuntimeError("Current webdataset must provide split_by_node and split_by_worker.")
    handler = _ErrorHandler(dataset_name, max_errors_per_worker)
    curl_url = _build_curl_url(
        url=dataset_url,
        connect_timeout=curl_connect_timeout,
        max_time=curl_max_time,
        retry=curl_retry,
        speed_time=curl_speed_time,
        speed_limit=curl_speed_limit,
        show_errors=curl_show_errors,
    )
    try:
        shards = wds.ResampledShards(curl_url, seed=seed, deterministic=True)
    except TypeError:
        try:
            shards = wds.ResampledShards(curl_url, seed=seed)
        except TypeError:
            shards = wds.ResampledShards(curl_url)

    return wds.DataPipeline(
        shards,
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=handler),
        wds.shuffle(shuffle_buffer_size),
        wds.decode("pil", handler=handler),
        wds.to_tuple(";".join(image_keys), ";".join(text_keys), handler=handler),
        wds.map_tuple(transform, _normalize_text),
        wds.select(_has_non_empty_text),
    )


def build_streaming_dataloader(
    *,
    data_files: Optional[str] = None,
    datasets: Optional[Sequence[Any]] = None,
    split: str = "train",
    image_size: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 16,
    seed: int = 42,
    epoch: int = 0,
    shuffle_buffer_size: int = 2000,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    text_keys: Optional[Sequence[str]] = None,
    image_keys: Optional[Sequence[str]] = None,
    loader_timeout: int = 180,
    max_errors_per_worker: int = 5000,
    curl_connect_timeout: int = 10,
    curl_max_time: int = 120,
    curl_retry: int = 15,
    curl_speed_time: int = 30,
    curl_speed_limit: int = 1024,
    curl_show_errors: bool = False,
) -> DataLoader:
    """Build mixed webdataset loader using RandomMix + ResampledShards."""
    if split != "train":
        raise ValueError("Mixed webdataset loader currently supports split='train' only.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0.")
    if shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be > 0.")

    _ = (rank, world_size, drop_last)

    active_image_keys = tuple(image_keys) if image_keys else DEFAULT_STREAMING_IMAGE_KEYS
    active_text_keys = tuple(text_keys) if text_keys else DEFAULT_STREAMING_TEXT_KEYS
    dataset_specs = _parse_dataset_specs(datasets, data_files)

    transform = transforms.Compose(
        [
            transforms.Lambda(_convert_to_rgb),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    streams = [
        _build_stream(
            dataset_name=name,
            dataset_url=url,
            transform=transform,
            shuffle_buffer_size=int(shuffle_buffer_size),
            seed=int(seed) + int(epoch),
            max_errors_per_worker=int(max_errors_per_worker),
            image_keys=active_image_keys,
            text_keys=active_text_keys,
            curl_connect_timeout=int(curl_connect_timeout),
            curl_max_time=int(curl_max_time),
            curl_retry=int(curl_retry),
            curl_speed_time=int(curl_speed_time),
            curl_speed_limit=int(curl_speed_limit),
            curl_show_errors=bool(curl_show_errors),
        )
        for name, url, _ in dataset_specs
    ]
    weights = [weight for _, _, weight in dataset_specs]

    mixed_pipeline = wds.DataPipeline(
        wds.RandomMix(streams, weights),
        wds.batched(int(batch_size), partial=False),
    )

    loader_kwargs = {
        "batch_size": None,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory and torch.cuda.is_available()),
        "persistent_workers": int(num_workers) > 0,
        "timeout": int(loader_timeout) if int(num_workers) > 0 else 0,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return wds.WebLoader(mixed_pipeline, **loader_kwargs)
