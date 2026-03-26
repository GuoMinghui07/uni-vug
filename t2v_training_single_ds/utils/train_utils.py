"""Training utilities for OpenVid video-stage training."""

from __future__ import annotations

import io
import itertools
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

__all__ = [
    "configure_experiment_dirs",
    "build_openvid_streaming_dataloader",
    "DEFAULT_OPENVID_TEXT_KEYS",
    "DEFAULT_OPENVID_VIDEO_KEYS",
    "FixedVideoSubsetDataset",
    "collate_fixed_video_batch",
    "load_local_sanity_subset",
    "sample_openvid_clip",
]


DEFAULT_OPENVID_TEXT_KEYS = (
    "caption",
    "prompt",
    "text",
    "txt",
    "description",
    "title",
    "json",
)
DEFAULT_OPENVID_VIDEO_KEYS = ("video_blob", "video", "blob", "bytes")


def _patch_torchcodec_audio_decoder() -> None:
    try:
        import torchcodec.decoders  # type: ignore
    except Exception:
        return

    if hasattr(torchcodec.decoders, "AudioDecoder"):
        return

    class AudioDecoder:  # pragma: no cover - compatibility shim
        pass

    torchcodec.decoders.AudioDecoder = AudioDecoder


_patch_torchcodec_audio_decoder()


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


def _to_prompt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except Exception:
                return text
            return _to_prompt(payload)
        return text
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, dict):
        for key in ("caption", "prompt", "text", "txt", "description", "title"):
            if key in value and value[key] is not None:
                return _to_prompt(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_to_prompt(v) for v in value]
        parts = [p for p in parts if p]
        return " ".join(parts).strip()
    return str(value).strip()


def _extract_prompt(sample: Dict[str, Any], text_keys: Sequence[str]) -> str:
    for key in text_keys:
        if key in sample and sample[key] is not None:
            prompt = _to_prompt(sample[key])
            if prompt:
                return prompt
    return ""


def _resolve_video_value(
    sample: Dict[str, Any],
    preferred_key: Optional[str],
    fallback_keys: Sequence[str],
) -> Any:
    if preferred_key and preferred_key in sample and sample[preferred_key] is not None:
        return sample[preferred_key]
    for key in fallback_keys:
        if key in sample and sample[key] is not None:
            return sample[key]
    return None


def _ensure_video_tensor(frames: torch.Tensor) -> torch.Tensor:
    if frames.dim() == 5 and frames.size(0) == 1:
        frames = frames[0]

    if frames.dim() == 3:
        if frames.shape[-1] in (1, 3, 4):
            frames = frames.permute(2, 0, 1).unsqueeze(0)
        elif frames.shape[0] in (1, 3, 4):
            frames = frames.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported frame tensor shape: {tuple(frames.shape)}")
    elif frames.dim() == 4:
        if frames.shape[-1] in (1, 3, 4):
            frames = frames.permute(0, 3, 1, 2)
        elif frames.shape[1] in (1, 3, 4):
            pass
        else:
            raise ValueError(f"Unsupported video tensor shape: {tuple(frames.shape)}")
    else:
        raise ValueError(f"Expected 3D/4D frame tensor, got shape {tuple(frames.shape)}")

    if frames.size(1) == 4:
        frames = frames[:, :3]

    frames = frames.to(torch.float32)
    if float(frames.max().item()) > 1.0:
        frames = frames / 255.0
    return frames.contiguous()


def _extract_video_source(value: Any) -> Tuple[Optional[str], Optional[bytes]]:
    if value is None:
        return None, None

    if isinstance(value, str):
        return value, None

    if isinstance(value, (bytes, bytearray)):
        return None, bytes(value)

    if isinstance(value, dict):
        path = value.get("path")
        blob = value.get("bytes")
        if not isinstance(path, str):
            path = None
        if not isinstance(blob, (bytes, bytearray)):
            blob = None
        return path, bytes(blob) if blob is not None else None

    path = getattr(value, "path", None)
    if isinstance(path, str):
        return path, None

    if hasattr(value, "read"):
        try:
            return None, value.read()
        except Exception:
            return None, None

    return None, None


def _metadata_num_frames(meta: Any) -> Optional[int]:
    if meta is None:
        return None

    candidates = (
        "num_frames",
        "num_frames_from_content",
        "num_frames_from_header",
        "frames",
        "frame_count",
    )
    for key in candidates:
        value = getattr(meta, key, None)
        if isinstance(value, int) and value > 0:
            return int(value)
        if torch.is_tensor(value) and value.numel() == 1:
            x = int(value.item())
            if x > 0:
                return x

    if isinstance(meta, dict):
        for key in candidates:
            value = meta.get(key)
            if isinstance(value, int) and value > 0:
                return int(value)

    return None


def _get_total_frames(video_value: Any) -> Optional[int]:
    if isinstance(video_value, (str, bytes, bytearray, os.PathLike)):
        return None

    if torch.is_tensor(video_value):
        if video_value.dim() == 4:
            if video_value.shape[0] >= 1 and video_value.shape[-1] in (1, 3, 4):
                return int(video_value.shape[0])
            if video_value.shape[1] in (1, 3, 4):
                return int(video_value.shape[0])
        if video_value.dim() == 3:
            return 1

    for key in ("num_frames", "frame_count", "frames"):
        value = getattr(video_value, key, None)
        if isinstance(value, int) and value > 0:
            return int(value)
        if torch.is_tensor(value) and value.numel() == 1:
            x = int(value.item())
            if x > 0:
                return x

    meta = getattr(video_value, "metadata", None)
    meta_count = _metadata_num_frames(meta)
    if meta_count is not None:
        return meta_count

    try:
        n = len(video_value)
        if isinstance(n, int) and n > 0:
            return int(n)
    except Exception:
        pass

    return None


def _decode_with_decord(
    source_path: Optional[str],
    source_bytes: Optional[bytes],
    start: int,
    end: int,
) -> torch.Tensor:
    import decord  # type: ignore

    decord.bridge.set_bridge("torch")
    tmp_path: Optional[Path] = None
    uri: str
    if source_path is not None:
        uri = source_path
    elif source_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(source_bytes)
            tmp_path = Path(tmp.name)
        uri = str(tmp_path)
    else:
        raise ValueError("Both source_path and source_bytes are None.")

    try:
        reader = decord.VideoReader(uri)
        total = len(reader)
        s = max(0, int(start))
        e = min(int(end), total)
        if e <= s:
            return torch.empty(0, 3, 1, 1, dtype=torch.float32)
        batch = reader.get_batch(list(range(s, e)))
        if torch.is_tensor(batch):
            frames = batch
        elif hasattr(batch, "asnumpy"):
            frames = torch.from_numpy(batch.asnumpy())
        else:
            frames = torch.as_tensor(batch)
        return _ensure_video_tensor(frames)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _decode_with_torchvision(
    source_path: Optional[str],
    source_bytes: Optional[bytes],
    start: int,
    end: int,
) -> torch.Tensor:
    from torchvision.io import read_video  # type: ignore

    tmp_path: Optional[Path] = None
    video_path: str
    if source_path is not None:
        video_path = source_path
    elif source_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(source_bytes)
            tmp_path = Path(tmp.name)
        video_path = str(tmp_path)
    else:
        raise ValueError("Both source_path and source_bytes are None.")

    try:
        # read_video decodes the full clip; this is a last-resort fallback path.
        frames, _, _ = read_video(video_path, pts_unit="sec")
        if not torch.is_tensor(frames):
            raise TypeError(f"Unexpected torchvision decoded frame type: {type(frames)}")
        frames = frames[int(start): int(end)]
        if frames.numel() == 0:
            return torch.empty(0, 3, 1, 1, dtype=torch.float32)
        return _ensure_video_tensor(frames)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _decode_with_torchcodec(
    source_path: Optional[str],
    source_bytes: Optional[bytes],
    start: int,
    end: int,
) -> torch.Tensor:
    backend_errors: List[Tuple[str, Exception]] = []

    try:
        import torchcodec.decoders  # type: ignore

        decoder_source: Any
        if source_path is not None:
            decoder_source = source_path
        elif source_bytes is not None:
            decoder_source = io.BytesIO(source_bytes)
        else:
            raise ValueError("Both source_path and source_bytes are None.")

        decoder = torchcodec.decoders.VideoDecoder(decoder_source)
        out = decoder.get_frames_in_range(int(start), int(end))
        data = out.data if hasattr(out, "data") else out
        if not torch.is_tensor(data):
            raise TypeError(f"Unexpected decoded frame type: {type(data)}")
        return _ensure_video_tensor(data)
    except Exception as exc:
        backend_errors.append(("torchcodec", exc))

    try:
        return _decode_with_decord(source_path, source_bytes, start, end)
    except Exception as exc:
        backend_errors.append(("decord", exc))

    try:
        return _decode_with_torchvision(source_path, source_bytes, start, end)
    except Exception as exc:
        backend_errors.append(("torchvision", exc))

    details = "; ".join(
        f"{name}={type(err).__name__}: {err}" for name, err in backend_errors
    )
    raise RuntimeError(f"All video decode backends failed: {details}")


def _decode_frame_range(video_value: Any, start: int, end: int) -> torch.Tensor:
    if end <= start:
        return torch.empty(0, 3, 1, 1, dtype=torch.float32)

    if torch.is_tensor(video_value):
        frames = _ensure_video_tensor(video_value)
        return frames[start:end]

    if hasattr(video_value, "get_frames_in_range"):
        out = video_value.get_frames_in_range(int(start), int(end))
        data = out.data if hasattr(out, "data") else out
        if torch.is_tensor(data):
            return _ensure_video_tensor(data)
        raise TypeError(f"Unexpected decoder output type: {type(data)}")

    src_path, src_bytes = _extract_video_source(video_value)
    if src_path is None and src_bytes is None:
        raise ValueError(f"Cannot decode video value of type: {type(video_value)}")
    return _decode_with_torchcodec(src_path, src_bytes, start, end)


def _resize_video(video: torch.Tensor, size: int) -> torch.Tensor:
    if video.dim() != 4:
        raise ValueError(f"Expected video shape (T, C, H, W), got {tuple(video.shape)}")
    if video.shape[-2] == size and video.shape[-1] == size:
        return video.contiguous()
    return F.interpolate(video, size=(size, size), mode="bilinear", align_corners=False).contiguous()


def sample_openvid_clip(
    video_value: Any,
    *,
    video_size: int,
    window_frames: int,
    frame_stride: int,
    rng: random.Random,
    random_window: bool = True,
) -> Optional[torch.Tensor]:
    """Sample video clip per OpenVid policy.

    1) Require source frames >= window_frames (default 64), otherwise drop.
    2) Randomly pick one contiguous window of `window_frames` frames.
    3) Uniformly subsample by `frame_stride` (2 => 32 frames from 64 window).
    4) Resize each frame to (video_size, video_size).
    """

    if window_frames <= 0:
        raise ValueError("window_frames must be > 0")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be > 0")

    total_frames = _get_total_frames(video_value)
    if total_frames is None:
        # Fallback for sources that don't expose metadata: decode first window only.
        clip = _decode_frame_range(video_value, 0, window_frames)
        if clip.shape[0] < window_frames:
            return None
    else:
        if total_frames < window_frames:
            return None
        max_start = total_frames - window_frames
        start = rng.randint(0, max_start) if (random_window and max_start > 0) else 0
        clip = _decode_frame_range(video_value, start, start + window_frames)
        if clip.shape[0] < window_frames:
            return None
        clip = clip[:window_frames]

    clip = clip[::frame_stride]
    if clip.shape[0] == 0:
        return None

    clip = _resize_video(clip, int(video_size))
    return clip.contiguous()


class StreamingOpenVidDataset(IterableDataset):
    """HF streaming IterableDataset for OpenVid clips."""

    def __init__(
        self,
        *,
        dataset_id: str,
        split: str,
        video_size: int,
        window_frames: int,
        frame_stride: int,
        seed: int,
        epoch: int,
        text_keys: Sequence[str],
        video_key: Optional[str],
        video_keys: Sequence[str],
        shuffle_buffer_size: int,
    ) -> None:
        super().__init__()
        self.dataset_id = str(dataset_id)
        self.split = str(split)
        self.video_size = int(video_size)
        self.window_frames = int(window_frames)
        self.frame_stride = int(frame_stride)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.text_keys = tuple(text_keys)
        self.video_key = str(video_key) if video_key else None
        self.video_keys = tuple(video_keys)
        self.shuffle_buffer_size = int(shuffle_buffer_size)

    @staticmethod
    def _rank_world_from_env() -> Tuple[int, int]:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world <= 0:
            world = 1
        rank = max(0, min(rank, world - 1))
        return rank, world

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        import datasets

        ds = datasets.load_dataset(self.dataset_id, split=self.split, streaming=True)
        if self.split == "train" and self.shuffle_buffer_size > 0:
            ds = ds.shuffle(
                seed=self.seed + self.epoch,
                buffer_size=self.shuffle_buffer_size,
            )

        rank, world_size = self._rank_world_from_env()
        iterator = iter(ds)
        if world_size > 1:
            iterator = itertools.islice(iterator, rank, None, world_size)

        worker = get_worker_info()
        worker_id = 0
        if worker is not None:
            worker_id = int(worker.id)
            iterator = itertools.islice(iterator, worker.id, None, worker.num_workers)

        rng_seed = self.seed + self.epoch * 1000003 + rank * 10007 + worker_id * 97
        rng = random.Random(rng_seed)

        for row in iterator:
            if not isinstance(row, dict):
                continue

            prompt = _extract_prompt(row, self.text_keys)
            if not prompt:
                continue

            video_value = _resolve_video_value(
                row,
                preferred_key=self.video_key,
                fallback_keys=self.video_keys,
            )
            if video_value is None:
                continue

            try:
                clip = sample_openvid_clip(
                    video_value,
                    video_size=self.video_size,
                    window_frames=self.window_frames,
                    frame_stride=self.frame_stride,
                    rng=rng,
                    random_window=True,
                )
            except Exception:
                continue

            if clip is None:
                continue

            yield {"video": clip, "prompt": prompt}


def _collate_openvid_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    videos = torch.stack([item["video"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    return {"video": videos, "prompt": prompts}


def build_openvid_streaming_dataloader(
    *,
    dataset_id: str,
    split: str,
    video_size: int,
    window_frames: int,
    frame_stride: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int,
    prefetch_factor: int = 2,
    shuffle_buffer_size: int = 0,
    text_keys: Optional[Sequence[str]] = None,
    video_key: Optional[str] = "video_blob",
    video_keys: Optional[Sequence[str]] = None,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    dataset = StreamingOpenVidDataset(
        dataset_id=dataset_id,
        split=split,
        video_size=int(video_size),
        window_frames=int(window_frames),
        frame_stride=int(frame_stride),
        seed=int(seed),
        epoch=int(epoch),
        text_keys=tuple(text_keys) if text_keys is not None else DEFAULT_OPENVID_TEXT_KEYS,
        video_key=video_key,
        video_keys=tuple(video_keys) if video_keys is not None else DEFAULT_OPENVID_VIDEO_KEYS,
        shuffle_buffer_size=int(shuffle_buffer_size),
    )

    loader_kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "persistent_workers": int(num_workers) > 0,
        "collate_fn": _collate_openvid_batch,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(dataset, **loader_kwargs)


class FixedVideoSubsetDataset(Dataset):
    """In-memory fixed video subset for sanity overfit checks."""

    def __init__(self, samples: List[Dict[str, Any]]):
        if len(samples) == 0:
            raise ValueError("FixedVideoSubsetDataset requires at least one sample.")
        self.videos = torch.stack([s["video"].detach().cpu().to(torch.float32) for s in samples], dim=0)
        self.prompts = [str(s["prompt"]) for s in samples]
        self.sample_ids = [str(s.get("sample_id", f"sample_{i:04d}")) for i, s in enumerate(samples)]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "video": self.videos[idx],
            "prompt": self.prompts[idx],
            "sample_id": self.sample_ids[idx],
        }


def collate_fixed_video_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    videos = torch.stack([item["video"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    return {"video": videos, "prompt": prompts, "sample_id": sample_ids}


def _load_manifest_entries(manifest_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def _load_caption_from_file(path: Path) -> str:
    try:
        return path.read_text("utf-8").strip()
    except Exception:
        return ""


def load_local_sanity_subset(
    *,
    subset_dir: str,
    max_samples: int,
    video_size: int,
    window_frames: int,
    frame_stride: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Load fixed local sanity subset from `sanity-check-video-data`."""

    root = Path(subset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Sanity subset dir not found: '{root}'.")

    manifest = root / "manifest.jsonl"
    entries: List[Dict[str, Any]] = []
    if manifest.exists():
        entries = _load_manifest_entries(manifest)

    if not entries:
        # Fallback: pair .mp4 with .caption.txt by stem.
        videos = sorted(root.glob("*.mp4"))
        for vp in videos:
            stem = vp.stem
            cp = root / f"{stem}.caption.txt"
            entries.append(
                {
                    "sample_id": stem,
                    "video_file": vp.name,
                    "caption": _load_caption_from_file(cp),
                    "caption_file": cp.name if cp.exists() else None,
                }
            )

    if not entries:
        raise RuntimeError(f"No sanity entries found under '{root}'.")

    rng = random.Random(int(seed))
    out: List[Dict[str, Any]] = []
    target = int(max_samples)
    skipped_missing_video = 0
    skipped_missing_prompt = 0
    skipped_decode_error = 0
    skipped_short_clip = 0
    first_decode_error: Optional[Exception] = None

    for entry in entries:
        if target > 0 and len(out) >= target:
            break

        sample_id = str(entry.get("sample_id", f"sample_{len(out):04d}"))
        video_name = entry.get("video_file")
        if not isinstance(video_name, str):
            skipped_missing_video += 1
            continue
        video_path = root / video_name
        if not video_path.exists():
            skipped_missing_video += 1
            continue

        prompt = _to_prompt(entry.get("caption"))
        if not prompt:
            caption_file = entry.get("caption_file")
            if isinstance(caption_file, str):
                prompt = _load_caption_from_file(root / caption_file)
        if not prompt:
            skipped_missing_prompt += 1
            continue

        try:
            clip = sample_openvid_clip(
                str(video_path),
                video_size=int(video_size),
                window_frames=int(window_frames),
                frame_stride=int(frame_stride),
                rng=rng,
                random_window=True,
            )
        except Exception as exc:
            skipped_decode_error += 1
            if first_decode_error is None:
                first_decode_error = exc
            continue

        if clip is None:
            skipped_short_clip += 1
            continue

        out.append({"sample_id": sample_id, "video": clip, "prompt": prompt})

    if target > 0 and len(out) < target:
        details = (
            f"entries={len(entries)}, kept={len(out)}, missing_video={skipped_missing_video}, "
            f"missing_prompt={skipped_missing_prompt}, decode_errors={skipped_decode_error}, "
            f"short_or_empty_clips={skipped_short_clip}"
        )
        if first_decode_error is not None:
            details += (
                f", first_decode_error={type(first_decode_error).__name__}: "
                f"{first_decode_error}"
            )
        raise RuntimeError(
            f"Only loaded {len(out)}/{target} valid sanity samples from '{root}' ({details})."
        )

    return out
