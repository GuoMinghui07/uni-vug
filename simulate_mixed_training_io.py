import argparse
import json
import os
import random
import time
from collections import Counter
from datetime import timedelta

import torch
import torch.distributed as dist
import webdataset as wds
from PIL import Image, ImageFile
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = 1024 * 1024 * 256
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    ("pd12m", "https://huggingface.co/datasets/Spawning/pd12m-full/resolve/main/{00155..02663}.tar", 0.7),
    ("t2i2m", "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{000000..000046}.tar", 0.2),
    ("fine_t2i", "https://huggingface.co/datasets/ma-xu/fine-t2i/resolve/main/synthetic_enhanced_prompt_square_resolution/train-{000000..000049}.tar", 0.1),
]
ERROR_COUNTER = Counter()


def parse_args():
    p = argparse.ArgumentParser(description="Mixed WebDataset throughput simulation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--shuffle-buffer", type=int, default=2000)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--simulate-ms", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--loader-timeout", type=int, default=180)
    p.add_argument("--max-errors-per-worker", type=int, default=5000)
    p.add_argument("--curl-connect-timeout", type=int, default=10)
    p.add_argument("--curl-max-time", type=int, default=120)
    p.add_argument("--curl-retry", type=int, default=15)
    p.add_argument("--curl-speed-time", type=int, default=30)
    p.add_argument("--curl-speed-limit", type=int, default=1024)
    p.add_argument("--curl-show-errors", action="store_true")
    a = p.parse_args()
    if a.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if a.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if a.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if a.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if a.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if a.log_every <= 0:
        raise ValueError("--log-every must be > 0")
    if a.image_size <= 0:
        raise ValueError("--image-size must be > 0")
    if a.simulate_ms < 0:
        raise ValueError("--simulate-ms must be >= 0")
    if a.loader_timeout < 0:
        raise ValueError("--loader-timeout must be >= 0")
    if a.max_errors_per_worker <= 0:
        raise ValueError("--max-errors-per-worker must be > 0")
    return a


def is_dist():
    return dist.is_available() and dist.is_initialized()


def init_dist():
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world > 1 and not is_dist():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def norm_text(v):
    if v is None:
        return ""
    if isinstance(v, bytes):
        s = v.decode("utf-8", errors="ignore").strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return norm_text(json.loads(s))
            except json.JSONDecodeError:
                return s
        return s
    if isinstance(v, dict):
        for k in ("caption", "prompt", "text", "txt"):
            if k in v and v[k] is not None:
                return norm_text(v[k])
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return " ".join(norm_text(x) for x in v if x is not None).strip()
    return str(v).strip()


def make_handler(name, max_errors):
    def _h(exc):
        ERROR_COUNTER[f"{name}:{type(exc).__name__}"] += 1
        total = sum(ERROR_COUNTER.values())
        winfo = torch.utils.data.get_worker_info()
        worker = winfo.id if winfo is not None else 0
        rank = int(os.getenv("RANK", "0"))
        if rank == 0 and worker == 0 and (total <= 3 or total % 100 == 0):
            msg = " ".join(str(exc).split())[:220]
            print(f"[warn] {name} skip {type(exc).__name__}, total_errors={total}, err={msg}", flush=True)
        if total >= max_errors:
            raise RuntimeError(f"[fatal] {name} too many data errors: {total}") from exc
        return True

    return _h


def build_curl_url(url, a):
    flag = "S" if a.curl_show_errors else ""
    return (
        f"pipe:curl -s{flag}Lf "
        f"--connect-timeout {a.curl_connect_timeout} "
        f"--max-time {a.curl_max_time} "
        f"--retry {a.curl_retry} --retry-delay 1 "
        f"--speed-time {a.curl_speed_time} --speed-limit {a.curl_speed_limit} "
        f"{url}"
    )


def build_stream(name, base_url, transform, a):
    handler = make_handler(name, a.max_errors_per_worker)
    url = build_curl_url(base_url, a)
    if not hasattr(wds, "split_by_node") or not hasattr(wds, "split_by_worker"):
        raise RuntimeError("Current webdataset must provide split_by_node and split_by_worker")
    try:
        shards = wds.ResampledShards(url, seed=a.seed, deterministic=True)
    except TypeError:
        try:
            shards = wds.ResampledShards(url, seed=a.seed)
        except TypeError:
            shards = wds.ResampledShards(url)
    stages = [
        shards,
        wds.split_by_node,
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=handler),
        wds.shuffle(a.shuffle_buffer),
        wds.decode("pil", handler=handler),
        wds.to_tuple("jpg;jpeg;png;webp", "txt;json;caption", handler=handler),
        wds.map_tuple(transform, norm_text),
        wds.select(lambda s: bool(s[1])),
    ]
    return wds.DataPipeline(*stages)


def build_loader(a):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(a.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((a.image_size, a.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    streams = [build_stream(n, u, transform, a) for n, u, _ in DATASETS]
    weights = [w for _, _, w in DATASETS]
    mixed = wds.DataPipeline(wds.RandomMix(streams, weights), wds.batched(a.batch_size, partial=False))

    kwargs = {
        "batch_size": None,
        "num_workers": a.num_workers,
        "timeout": a.loader_timeout if a.num_workers > 0 else 0,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": a.num_workers > 0,
    }
    if a.num_workers > 0:
        kwargs["prefetch_factor"] = a.prefetch_factor
    return wds.WebLoader(mixed, **kwargs)


def reduce_max(x, device):
    if not is_dist():
        return x
    t = torch.tensor(x, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def seed_all(seed, rank):
    s = seed + rank * 100003
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():
    a = parse_args()
    rank, world, local_rank = init_dist()
    seed_all(a.seed, rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    it = None
    step_times, load_times = [], []
    t0 = time.perf_counter()

    if rank == 0:
        print(
            f"[init] batch={a.batch_size} workers={a.num_workers} world={world} "
            f"shuffle={a.shuffle_buffer} timeout={a.loader_timeout}s"
        )
        print("[init] waiting for first batch...")

    try:
        loader = build_loader(a)
        it = iter(loader)
        for step in range(1, a.max_steps + 1):
            ts = time.perf_counter()
            try:
                images, _ = next(it)
            except StopIteration:
                it = iter(loader)
                images, _ = next(it)
            except RuntimeError as e:
                if "DataLoader timed out" in str(e):
                    raise RuntimeError(f"DataLoader timeout (> {a.loader_timeout}s)") from e
                raise

            load_t = time.perf_counter() - ts
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda._sleep(int(a.simulate_ms * 1e6))
                torch.cuda.synchronize(device)
            else:
                time.sleep(a.simulate_ms / 1000.0)
            step_t = time.perf_counter() - ts

            if step > a.warmup_steps:
                load_times.append(load_t)
                step_times.append(step_t)

            if step == 1 and rank == 0:
                print(f"[step {step:>4}] first batch shape={list(images.shape)} load={load_t:.2f}s")
            elif step <= a.warmup_steps and rank == 0:
                print(f"[step {step:>4}] warmup load={load_t:.3f}s")
            elif step > a.warmup_steps and step % a.log_every == 0:
                avg_load = sum(load_times) / len(load_times)
                avg_step = sum(step_times) / len(step_times)
                slowest = reduce_max(avg_step, device)  # all ranks must participate
                local_ips = a.batch_size / avg_step
                global_ips = (a.batch_size * world) / slowest
                if rank == 0:
                    print(
                        f"[step {step:>4}/{a.max_steps}] load={avg_load:.3f}s step={avg_step:.3f}s "
                        f"local={local_ips:.1f} imgs/s global={global_ips:.1f} imgs/s "
                        f"elapsed={time.perf_counter() - t0:.0f}s"
                    )

        avg_step = sum(step_times) / len(step_times) if step_times else 0.0
        slowest = reduce_max(avg_step, device)
        local_ips = a.batch_size / avg_step if avg_step > 0 else 0.0
        global_ips = (a.batch_size * world) / slowest if slowest > 0 else 0.0
        if rank == 0:
            print(f"[done] elapsed={time.perf_counter() - t0:.1f}s local={local_ips:.1f} imgs/s global={global_ips:.1f} imgs/s")
    except Exception as e:
        if rank == 0:
            print(f"[fatal] {e}")
        if is_dist() and world > 1:
            try:
                dist.abort()
            except Exception:
                pass
        raise
    finally:
        close_workers = getattr(it, "_shutdown_workers", None)
        if callable(close_workers):
            try:
                close_workers()
            except Exception:
                pass
        if is_dist():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
