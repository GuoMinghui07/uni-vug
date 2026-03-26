import argparse
import itertools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcodec.decoders
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# 兼容部分 datasets 版本对 AudioDecoder 的检查
if not hasattr(torchcodec.decoders, "AudioDecoder"):
    class AudioDecoder:
        pass

    torchcodec.decoders.AudioDecoder = AudioDecoder

import datasets


def parse_args():
    p = argparse.ArgumentParser(description="OpenVid 模拟训练脚本（统计每步加载耗时）")
    p.add_argument("--dataset-id", type=str, default="lance-format/openvid-lance")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


class TinyVideoModel(nn.Module):
    def __init__(self, in_channels=3, hidden=64, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden * 2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def decode_one_frame(video_value):
    if isinstance(video_value, torch.Tensor):
        frame = video_value[0] if video_value.ndim == 4 else video_value
    else:
        frame = video_value.get_frames_in_range(0, 1).data[0]
    return frame


def preprocess_frame(frame, image_size):
    x = frame.float()
    if x.max() > 1.0:
        x = x / 255.0
    x = F.interpolate(x.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)
    return x.squeeze(0).contiguous()


class StreamingVideoFrameDataset(IterableDataset):
    def __init__(self, dataset_id, split, image_size):
        super().__init__()
        self.dataset_id = dataset_id
        self.split = split
        self.image_size = image_size

    def __iter__(self):
        ds = datasets.load_dataset(self.dataset_id, split=self.split, streaming=True)
        worker = get_worker_info()
        if worker is None:
            iterator = iter(ds)
        else:
            iterator = itertools.islice(ds, worker.id, None, worker.num_workers)

        for row in iterator:
            video_value = row.get("video_blob")
            if video_value is None:
                continue
            try:
                frame = decode_one_frame(video_value)
                yield preprocess_frame(frame, self.image_size)
            except Exception:
                continue


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[init] device={device}, bs={args.batch_size}, workers={args.num_workers}, "
        f"steps={args.steps}, image_size={args.image_size}"
    )
    print("[init] 正在连接数据集...")

    ds = StreamingVideoFrameDataset(
        dataset_id=args.dataset_id,
        split=args.split,
        image_size=args.image_size,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(ds, **loader_kwargs)
    it = iter(loader)

    model = TinyVideoModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    load_times = []
    step_times = []

    for step in range(1, args.steps + 1):
        load_t0 = time.perf_counter()
        batch = next(it)  # [B, C, H, W]
        load_time = time.perf_counter() - load_t0

        step_t0 = time.perf_counter()
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)
        loss = pred.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_t0

        load_times.append(load_time)
        step_times.append(step_time)
        print(
            f"[step {step:03d}] load={load_time:.4f}s train={step_time:.4f}s ",
            f"total={load_time + step_time:.4f}s loss={loss.item():.6f}",
            f"size={batch.shape}",
            flush=True,
        )

    load_avg = sum(load_times) / len(load_times)
    load_total = sum(load_times)
    load_max = max(load_times)
    step_avg = sum(step_times) / len(step_times)
    print(
        f"\n[summary] load_total={load_total:.4f}s load_avg={load_avg:.4f}s "
        f"load_max={load_max:.4f}s train_avg={step_avg:.4f}s"
    )


if __name__ == "__main__":
    main()
