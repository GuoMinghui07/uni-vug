#!/usr/bin/env python3
"""Compute OpenVid resolution stats via HF datasets streaming API."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple


def _to_int(v: Any) -> Optional[int]:
    try:
        x = int(v)
    except Exception:
        return None
    return x if x > 0 else None


def _pick_resolution_columns(names: Sequence[str]) -> Optional[Tuple[str, str]]:
    lower = {n.lower(): n for n in names}
    for w, h in (
        ("width", "height"),
        ("video_width", "video_height"),
        ("w", "h"),
        ("orig_width", "orig_height"),
    ):
        if w in lower and h in lower:
            return lower[w], lower[h]
    return None


def _pick_video_column(names: Sequence[str]) -> Optional[str]:
    lower = {n.lower(): n for n in names}
    for c in ("video", "video_blob", "blob", "bytes"):
        if c in lower:
            return lower[c]
    return None


def _extract_video_source(video_value: Any) -> Tuple[Optional[str], Optional[bytes]]:
    if video_value is None:
        return None, None
    if hasattr(video_value, "read"):
        try:
            return None, video_value.read()
        except Exception:
            return None, None
    if isinstance(video_value, (bytes, bytearray)):
        return None, bytes(video_value)
    if isinstance(video_value, str):
        return video_value, None
    if isinstance(video_value, dict):
        path = video_value.get("path")
        blob = video_value.get("bytes")
        if not isinstance(path, str):
            path = None
        if not isinstance(blob, (bytes, bytearray)):
            blob = None
        return path, bytes(blob) if blob is not None else None
    return None, None


def _resolution_from_video(av_mod, video_value: Any) -> Tuple[Optional[int], Optional[int]]:
    path, blob = _extract_video_source(video_value)
    if blob is not None:
        source = io.BytesIO(blob)
    elif path is not None:
        source = path
    else:
        return None, None
    try:
        with av_mod.open(source) as container:
            if not container.streams.video:
                return None, None
            stream = container.streams.video[0]
            return _to_int(stream.width), _to_int(stream.height)
    except Exception:
        return None, None


def _write_csv(path: Path, counter: Counter, total_valid: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["resolution", "width", "height", "count", "ratio"])
        for (w, h), c in counter.most_common():
            ratio = c / total_valid if total_valid > 0 else 0.0
            writer.writerow([f"{w}x{h}", w, h, c, f"{ratio:.8f}"])


def _write_json(
    path: Path,
    *,
    dataset_id: str,
    split: str,
    mode: str,
    processed_rows: int,
    valid_rows: int,
    invalid_rows: int,
    skipped_rows: int,
    restart_count: int,
    width_col: Optional[str],
    height_col: Optional[str],
    video_col: Optional[str],
    counter: Counter,
    top_k: int,
) -> None:
    top = []
    for (w, h), c in counter.most_common(top_k):
        ratio = c / valid_rows if valid_rows > 0 else 0.0
        top.append(
            {
                "resolution": f"{w}x{h}",
                "width": w,
                "height": h,
                "count": c,
                "ratio": ratio,
            }
        )
    payload = {
        "dataset_id": dataset_id,
        "split": split,
        "mode": mode,
        "processed_rows": processed_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "skipped_rows": skipped_rows,
        "restart_count": restart_count,
        "unique_resolutions": len(counter),
        "resolution_columns": {"width": width_col, "height": height_col},
        "video_column": video_col,
        "top_resolutions": top,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute OpenVid resolution stats.")
    p.add_argument("--dataset-id", type=str, default="lance-format/openvid-lance")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--mode", type=str, choices=("auto", "metadata", "decode"), default="auto")
    p.add_argument("--width-col", type=str, default=None)
    p.add_argument("--height-col", type=str, default=None)
    p.add_argument("--video-col", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=0, help="0 means full stream.")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--no-progress-bar", action="store_true")
    p.add_argument("--max-restarts", type=int, default=8, help="Max restart attempts after stream abort.")
    p.add_argument(
        "--skip-on-error",
        type=int,
        default=256,
        help="How many rows to skip when a stream abort happens.",
    )
    p.add_argument("--out-prefix", type=str, default="openvid_resolution_stats")
    return p.parse_args()


def _load_streaming_dataset(datasets_mod, dataset_id: str, split: str):
    candidates = [dataset_id]
    lid = dataset_id.lower()
    if lid == "lance-format/openvid-lance":
        candidates.append("lance-format/Openvid-1M")
    elif lid == "lance-format/openvid-1m":
        candidates.append("lance-format/openvid-lance")

    last_exc: Optional[Exception] = None
    for cid in candidates:
        try:
            ds = datasets_mod.load_dataset(cid, split=split, streaming=True)
            return ds, cid
        except Exception as exc:
            last_exc = exc
            if "No (supported) data files found" in str(exc):
                continue
            raise

    version = getattr(datasets_mod, "__version__", "unknown")
    raise SystemExit(
        "Failed to load OpenVid via datasets streaming.\n"
        f"Tried dataset ids: {candidates}\n"
        f"datasets version: {version}\n"
        f"python: {sys.executable}\n"
        f"last error: {type(last_exc).__name__}: {last_exc}\n\n"
        "This specific error means your local `datasets` cannot parse lance format in this runtime.\n"
        "Run in the same shell:\n"
        "  python -c \"import datasets,sys; print(datasets.__version__, sys.executable)\"\n"
        "  pip install -U datasets"
    )


def _is_retryable_stream_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    patterns = (
        "task was aborted",
        "external error",
        "lance_background_thread",
        "arrowinvalid",
        "io error",
    )
    return any(p in msg for p in patterns)


def _build_stream(datasets_mod, dataset_id: str, split: str, video_col: Optional[str]):
    ds = datasets_mod.load_dataset(dataset_id, split=split, streaming=True)
    if video_col is not None:
        try:
            ds = ds.cast_column(video_col, datasets_mod.Video(decode=False))
        except Exception:
            pass
    return ds


def main() -> None:
    args = parse_args()

    try:
        import datasets  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install with `pip install datasets`.") from exc
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: av. Install with `pip install av`.") from exc
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: tqdm. Install with `pip install tqdm`.") from exc

    hf_ds, resolved_dataset_id = _load_streaming_dataset(
        datasets_mod=datasets,
        dataset_id=args.dataset_id,
        split=args.split,
    )
    if resolved_dataset_id != args.dataset_id:
        print(
            f"[info] dataset_id alias fallback: {args.dataset_id} -> {resolved_dataset_id}",
            flush=True,
        )

    feature_names = list((hf_ds.features or {}).keys())
    width_col = args.width_col
    height_col = args.height_col
    video_col = args.video_col

    detected = _pick_resolution_columns(feature_names)
    if detected is not None:
        width_col = width_col or detected[0]
        height_col = height_col or detected[1]
    if video_col is None:
        video_col = _pick_video_column(feature_names)

    mode = args.mode
    if mode == "auto":
        mode = "metadata" if (width_col and height_col) else "decode"

    if mode == "metadata" and (width_col is None or height_col is None):
        raise SystemExit("Metadata mode requires --width-col and --height-col (or columns in dataset).")
    if mode == "decode" and video_col is None:
        raise SystemExit("Decode mode requires --video-col (or an auto-detected video column).")

    print(f"Dataset: {resolved_dataset_id}", flush=True)
    print(f"Split: {args.split}", flush=True)
    print(f"Feature columns: {', '.join(feature_names)}", flush=True)
    print(f"Mode: {mode}", flush=True)
    print(f"Resolution columns: {width_col}, {height_col}", flush=True)
    print(f"Video column: {video_col}", flush=True)

    counter: Counter = Counter()
    processed = 0
    invalid = 0
    skipped_rows = 0
    restart_count = 0

    total = int(args.num_samples) if args.num_samples > 0 else None
    pbar = tqdm(
        total=total,
        desc=f"scan({mode})",
        unit="it",
        dynamic_ncols=True,
        disable=bool(args.no_progress_bar),
    )

    while True:
        if args.num_samples > 0 and processed >= args.num_samples:
            break

        resume_offset = processed + skipped_rows
        hf_ds = _build_stream(
            datasets_mod=datasets,
            dataset_id=resolved_dataset_id,
            split=args.split,
            video_col=video_col if mode == "decode" else None,
        )
        if resume_offset > 0:
            hf_ds = hf_ds.skip(resume_offset)

        try:
            for row in hf_ds:
                if args.num_samples > 0 and processed >= args.num_samples:
                    break
                processed += 1

                if mode == "metadata":
                    w = _to_int(row.get(width_col))
                    h = _to_int(row.get(height_col))
                else:
                    w, h = _resolution_from_video(av, row.get(video_col))

                if w is None or h is None:
                    invalid += 1
                else:
                    counter[(w, h)] += 1
                pbar.update(1)
            break
        except Exception as exc:
            if not _is_retryable_stream_error(exc) or restart_count >= int(args.max_restarts):
                pbar.close()
                raise

            restart_count += 1
            skip_delta = max(0, int(args.skip_on_error))
            skipped_rows += skip_delta
            next_offset = processed + skipped_rows
            print(
                (
                    f"[warn] stream aborted: {type(exc).__name__}: {exc}\n"
                    f"[warn] restart {restart_count}/{int(args.max_restarts)}, "
                    f"skip_delta={skip_delta}, next_offset={next_offset}"
                ),
                file=sys.stderr,
                flush=True,
            )

    pbar.close()

    valid = sum(counter.values())
    out_prefix = Path(args.out_prefix)
    out_csv = out_prefix.with_suffix(".csv")
    out_json = out_prefix.with_suffix(".json")

    _write_csv(out_csv, counter, valid)
    _write_json(
        out_json,
        dataset_id=resolved_dataset_id,
        split=args.split,
        mode=mode,
        processed_rows=processed,
        valid_rows=valid,
        invalid_rows=invalid,
        skipped_rows=skipped_rows,
        restart_count=restart_count,
        width_col=width_col,
        height_col=height_col,
        video_col=video_col,
        counter=counter,
        top_k=int(args.top_k),
    )

    print("\nTop resolutions:", flush=True)
    for rank, ((w, h), c) in enumerate(counter.most_common(int(args.top_k)), start=1):
        ratio = c / valid if valid > 0 else 0.0
        print(f"{rank:>3}. {w}x{h:<8} count={c:<8} ratio={ratio:.4%}", flush=True)

    print("\nSummary:", flush=True)
    print(f"  processed_rows: {processed}", flush=True)
    print(f"  valid_rows: {valid}", flush=True)
    print(f"  invalid_rows: {invalid}", flush=True)
    print(f"  skipped_rows: {skipped_rows}", flush=True)
    print(f"  restart_count: {restart_count}", flush=True)
    print(f"  unique_resolutions: {len(counter)}", flush=True)
    print(f"  csv: {out_csv}", flush=True)
    print(f"  json: {out_json}", flush=True)


if __name__ == "__main__":
    main()
