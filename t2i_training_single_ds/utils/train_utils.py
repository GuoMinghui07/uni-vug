"""Training utilities: EMA, experiment dirs, and data loading helpers."""

from __future__ import annotations

import logging
import os
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
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
]

DEFAULT_STREAMING_TEXT_KEYS = ("txt", "prompt", "text", "caption")
DEFAULT_STREAMING_IMAGE_KEYS = ("jpg", "jpeg", "png", "webp", "image")


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
    from PIL import Image
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


def _pick_sample_value(sample: Dict[str, Any], keys: Sequence[str], kind: str) -> Any:
    for key in keys:
        value = sample.get(key)
        if value is not None:
            return value
    raise KeyError(f"Missing {kind} in sample. Tried keys: {list(keys)}")


def _to_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _to_pil_image(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            return Image.open(io.BytesIO(payload)).convert("RGB")
    return value.convert("RGB")


class _StreamingPreprocess:
    def __init__(self, image_size: int, text_keys: Sequence[str], image_keys: Sequence[str]):
        self.text_keys = tuple(text_keys)
        self.image_keys = tuple(image_keys)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_value = _pick_sample_value(sample, self.image_keys, "image")
        text_value = _pick_sample_value(sample, self.text_keys, "text")
        image = _to_pil_image(image_value)
        prompt = _to_prompt(text_value).strip()
        return {"image": self.transform(image), "prompt": prompt}


def _collate_streaming_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    return {"image": images, "prompt": prompts}


def build_streaming_dataloader(
    *,
    data_files: str,
    split: str,
    image_size: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 16,
    seed: int = 42,
    epoch: int = 0,
    shuffle_buffer_size: int = 1000,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    text_keys: Optional[Sequence[str]] = None,
    image_keys: Optional[Sequence[str]] = None,
) -> DataLoader:
    """Build standard DDP + streaming DataLoader (HF webdataset backend)."""
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node

    os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
    # The hub defaults to 10s read timeout, which is often too aggressive for
    # large remote shard streams on shared clusters.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

    active_text_keys = tuple(text_keys) if text_keys else DEFAULT_STREAMING_TEXT_KEYS
    active_image_keys = tuple(image_keys) if image_keys else DEFAULT_STREAMING_IMAGE_KEYS

    dataset = load_dataset(
        "webdataset",
        data_files={split: data_files},
        split=split,
        streaming=True,
    )
    dataset = split_dataset_by_node(dataset, rank=int(rank), world_size=int(world_size))
    if split == "train":
        dataset = dataset.shuffle(seed=int(seed) + int(epoch), buffer_size=int(shuffle_buffer_size))
    dataset = dataset.map(_StreamingPreprocess(image_size, active_text_keys, active_image_keys))

    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "collate_fn": _collate_streaming_batch,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(dataset, **loader_kwargs)
