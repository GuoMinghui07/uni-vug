from __future__ import annotations

import io
import itertools
import multiprocessing as mp
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


DEFAULT_VIDEO_KEYS = ("video_blob", "video", "blob", "bytes")


try:
    import torchcodec.decoders

    if not hasattr(torchcodec.decoders, "AudioDecoder"):
        class AudioDecoder:
            pass

        torchcodec.decoders.AudioDecoder = AudioDecoder
except Exception:
    pass


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
    if frames.numel() > 0 and float(frames.max().item()) > 1.0:
        frames = frames / 255.0
    return frames.contiguous()


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

    details = "; ".join(f"{name}={type(err).__name__}: {err}" for name, err in backend_errors)
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


def sample_streaming_clip(
    video_value: Any,
    *,
    image_size: int,
    clip_frames: int,
    rng: random.Random,
    window_frames: int,
    frame_stride: int,
    random_window: bool = True,
) -> Optional[torch.Tensor]:
    if clip_frames <= 0:
        raise ValueError("clip_frames must be > 0")
    if window_frames <= 0:
        raise ValueError("window_frames must be > 0")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be > 0")

    total_frames = _get_total_frames(video_value)
    if total_frames is None:
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
    if clip.shape[0] < clip_frames:
        return None
    clip = clip[:clip_frames]

    clip = _resize_video(clip, int(image_size))
    return clip.contiguous()


class StreamingVideoClipDataset(IterableDataset):
    def __init__(
        self,
        *,
        dataset_id: str,
        split: str,
        video_key: Optional[str],
        video_keys: Sequence[str],
        image_size: int,
        clip_frames: int,
        window_frames: int,
        frame_stride: int,
        shuffle_buffer_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_id = str(dataset_id)
        self.split = str(split)
        self.video_key = str(video_key) if video_key else None
        self.video_keys = tuple(video_keys)
        self.image_size = int(image_size)
        self.clip_frames = int(clip_frames)
        self.window_frames = int(window_frames)
        self.frame_stride = int(frame_stride)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(max(1, world_size))
        self.epoch = 0
        self._shared_epoch = mp.Value("l", 0)

    def set_epoch(self, epoch: int) -> None:
        value = int(epoch)
        self.epoch = value
        self._shared_epoch.value = value

    def __iter__(self) -> Iterable[torch.Tensor]:
        import datasets

        current_epoch = int(self._shared_epoch.value)

        ds = datasets.load_dataset(self.dataset_id, split=self.split, streaming=True)
        if self.split == "train" and self.shuffle_buffer_size > 0:
            ds = ds.shuffle(seed=self.seed + current_epoch, buffer_size=self.shuffle_buffer_size)

        iterator = iter(ds)
        if self.world_size > 1:
            iterator = itertools.islice(iterator, self.rank, None, self.world_size)

        worker = get_worker_info()
        worker_id = 0
        if worker is not None:
            worker_id = int(worker.id)
            iterator = itertools.islice(iterator, worker.id, None, worker.num_workers)

        rng_seed = self.seed + current_epoch * 1_000_003 + self.rank * 10_007 + worker_id * 97
        rng = random.Random(rng_seed)

        for row in iterator:
            if not isinstance(row, dict):
                continue

            video_value = _resolve_video_value(
                row,
                preferred_key=self.video_key,
                fallback_keys=self.video_keys,
            )
            if video_value is None:
                continue

            try:
                clip = sample_streaming_clip(
                    video_value,
                    image_size=self.image_size,
                    clip_frames=self.clip_frames,
                    window_frames=self.window_frames,
                    frame_stride=self.frame_stride,
                    rng=rng,
                    random_window=True,
                )
            except Exception:
                continue

            if clip is None:
                continue

            if clip.shape[0] != self.clip_frames:
                continue

            yield clip


def collate_clips(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)


class ChunkSampler:
    def __init__(self, data_cfg: Dict[str, Any]) -> None:
        self.chunk_frames = int(data_cfg["chunk_frames"])
        self.start_frames = int(data_cfg["chunk_start_frames"])
        self.clip_frames = int(data_cfg["clip_frames"])

        stride_values = [int(v) for v in data_cfg["stride_values"]]
        stride_probs = [float(v) for v in data_cfg["stride_probs"]]
        self.stride_values = torch.tensor(stride_values, dtype=torch.long)
        probs = torch.tensor(stride_probs, dtype=torch.float32)
        self.stride_probs = probs / probs.sum()

    def sample(self, clips: torch.Tensor) -> Dict[str, torch.Tensor]:
        if clips.dim() != 5:
            raise ValueError(f"Expected clips shape (B,T,C,H,W), got {tuple(clips.shape)}")
        if clips.shape[1] != self.clip_frames:
            raise ValueError(f"Expected clip length {self.clip_frames}, got {clips.shape[1]}")

        batch = int(clips.shape[0])
        device = clips.device

        starts = torch.randint(
            low=0,
            high=self.start_frames,
            size=(batch,),
            device=device,
        )
        stride_ids = torch.multinomial(
            self.stride_probs.to(device=device),
            num_samples=batch,
            replacement=True,
        )
        strides = self.stride_values.to(device=device).index_select(0, stride_ids)

        offsets = torch.arange(self.chunk_frames, device=device).unsqueeze(0)
        indices = starts.unsqueeze(1) + strides.unsqueeze(1) * offsets

        batch_idx = torch.arange(batch, device=device).unsqueeze(1)
        chunks = clips[batch_idx, indices]

        return {
            "chunks": chunks.contiguous(),
            "indices": indices,
            "starts": starts,
            "strides": strides,
        }


def build_dataloader(
    data_cfg: Dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    seed: int,
    rank: int,
    world_size: int,
    pin_memory: bool,
) -> tuple[StreamingVideoClipDataset, DataLoader]:
    clip_frames = int(data_cfg["clip_frames"])
    frame_stride = int(data_cfg.get("frame_stride", 1))
    window_frames = int(data_cfg.get("window_frames", clip_frames * frame_stride))
    if frame_stride <= 0:
        raise ValueError("data.frame_stride must be > 0")
    if window_frames <= 0:
        raise ValueError("data.window_frames must be > 0")

    video_key = data_cfg.get("video_key", data_cfg.get("video_column", "video_blob"))
    video_keys = data_cfg.get("video_keys", list(DEFAULT_VIDEO_KEYS))
    if not isinstance(video_keys, (list, tuple)):
        raise TypeError("data.video_keys must be a list/tuple when provided")

    dataset = StreamingVideoClipDataset(
        dataset_id=str(data_cfg["dataset_id"]),
        split=str(data_cfg["split"]),
        video_key=(str(video_key) if video_key is not None else None),
        video_keys=[str(k) for k in video_keys],
        image_size=int(data_cfg["image_size"]),
        clip_frames=clip_frames,
        window_frames=window_frames,
        frame_stride=frame_stride,
        shuffle_buffer_size=int(data_cfg.get("shuffle_buffer_size", 0)),
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": True,
        "persistent_workers": bool(persistent_workers) and int(num_workers) > 0,
        "collate_fn": collate_clips,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = DataLoader(dataset, **kwargs)
    return dataset, loader
