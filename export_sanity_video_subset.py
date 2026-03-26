#!/usr/bin/env python3
"""Export a small random video+caption subset for sanity checks.

Example:
  python export_sanity_video_subset.py \
    --dataset-id lance-format/openvid-lance \
    --split train \
    --num-samples 64 \
    --output-dir sanity-check-video-data
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

try:
    import torchcodec.decoders  # type: ignore
except Exception:
    torchcodec = None
else:
    # Compatibility for some datasets versions that check AudioDecoder
    if not hasattr(torchcodec.decoders, "AudioDecoder"):
        class AudioDecoder:  # pragma: no cover
            pass

        torchcodec.decoders.AudioDecoder = AudioDecoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export random video+caption samples from a streaming dataset.")
    p.add_argument("--dataset-id", type=str, default="lance-format/openvid-lance")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--output-dir", type=str, default="sanity-check-video-data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=512,
        help="Streaming shuffle buffer; 0 disables shuffle for maximum stability.",
    )
    p.add_argument("--max-scan", type=int, default=200000, help="Max rows to scan before giving up.")
    p.add_argument("--max-restarts", type=int, default=8, help="Max stream restart attempts.")
    p.add_argument(
        "--skip-on-error",
        type=int,
        default=256,
        help="How many rows to skip forward when a stream abort happens.",
    )
    p.add_argument("--retry-sleep", type=float, default=1.0, help="Seconds to sleep before stream restart.")
    p.add_argument("--video-col", type=str, default=None)
    p.add_argument("--caption-key", type=str, default=None)
    p.add_argument("--overwrite", action="store_true", help="Delete output-dir first if it exists.")
    return p.parse_args()


def _pick_video_column(names: Sequence[str]) -> Optional[str]:
    lower = {n.lower(): n for n in names}
    for c in ("video_blob", "video", "blob", "bytes"):
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

    path = getattr(video_value, "path", None)
    if isinstance(path, str):
        return path, None

    return None, None


def _guess_video_ext(path: Optional[str], blob: Optional[bytes]) -> str:
    if path:
        suffix = Path(path).suffix.lower()
        if 0 < len(suffix) <= 10:
            return suffix

    if blob:
        if blob[:4] == b"\x1a\x45\xdf\xa3":
            return ".mkv"
        if blob[:4] == b"OggS":
            return ".ogv"
        if blob[:4] == b"RIFF" and blob[8:12] == b"AVI ":
            return ".avi"
        if len(blob) >= 8 and blob[4:8] == b"ftyp":
            return ".mp4"

    return ".mp4"


def _normalize_caption_value(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("{") and text.endswith("}"):
            try:
                payload = json.loads(text)
            except Exception:
                return text
            cap = _caption_from_mapping(payload)
            return cap if cap else text
        return text

    if isinstance(value, dict):
        return _caption_from_mapping(value)

    return None


def _caption_from_mapping(payload: dict[str, Any]) -> Optional[str]:
    for key in ("caption", "prompt", "text", "txt", "description", "title"):
        val = payload.get(key)
        if isinstance(val, str):
            txt = val.strip()
            if txt:
                return txt
    return None


def _extract_caption(row: dict[str, Any], preferred_key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    keys: list[str] = []
    if preferred_key:
        keys.append(preferred_key)
    keys.extend(["caption", "prompt", "text", "txt", "description", "title", "json"])

    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        if key not in row:
            continue
        caption = _normalize_caption_value(row.get(key))
        if caption:
            return caption, key
    return None, None


def _resolve_video_value(row: dict[str, Any], preferred_col: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    if preferred_col and preferred_col in row:
        return row[preferred_col], preferred_col

    for key in ("video_blob", "video", "blob", "bytes"):
        if key in row:
            return row[key], key

    return None, None


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
        "Failed to load dataset via streaming API.\\n"
        f"Tried dataset ids: {candidates}\\n"
        f"datasets version: {version}\\n"
        f"python: {sys.executable}\\n"
        f"last error: {type(last_exc).__name__}: {last_exc}"
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


def _build_stream(
    datasets_mod,
    dataset_id: str,
    split: str,
    video_col: Optional[str],
    *,
    shuffle_seed: int,
    shuffle_buffer_size: int,
):
    ds = datasets_mod.load_dataset(dataset_id, split=split, streaming=True)
    if video_col is not None:
        try:
            ds = ds.cast_column(video_col, datasets_mod.Video(decode=False))
        except Exception:
            pass
    if shuffle_buffer_size > 0:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)
    return ds


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"Output directory is not empty: {path}. "
                "Use --overwrite to replace it, or set another --output-dir."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be > 0")
    if args.shuffle_buffer_size < 0:
        raise SystemExit("--shuffle-buffer-size must be >= 0")
    if args.max_scan <= 0:
        raise SystemExit("--max-scan must be > 0")
    if args.max_restarts < 0:
        raise SystemExit("--max-restarts must be >= 0")
    if args.skip_on_error < 0:
        raise SystemExit("--skip-on-error must be >= 0")
    if args.retry_sleep < 0:
        raise SystemExit("--retry-sleep must be >= 0")

    try:
        import datasets  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Install with `pip install datasets`.") from exc

    out_dir = Path(args.output_dir)
    _prepare_output_dir(out_dir, overwrite=args.overwrite)

    probe_ds, resolved_dataset_id = _load_streaming_dataset(datasets, args.dataset_id, args.split)
    feature_names = list((probe_ds.features or {}).keys())
    video_col = args.video_col or _pick_video_column(feature_names)

    manifest_path = out_dir / "manifest.jsonl"
    meta_path = out_dir / "meta.json"

    saved = 0
    scanned = 0
    missing_caption = 0
    missing_video = 0
    save_fail = 0
    restart_count = 0
    skipped_rows = 0

    print(
        f"[init] dataset={resolved_dataset_id} split={args.split} num_samples={args.num_samples} "
        f"shuffle_buffer={args.shuffle_buffer_size} seed={args.seed} "
        f"max_restarts={args.max_restarts} skip_on_error={args.skip_on_error}",
        flush=True,
    )
    if video_col:
        print(f"[init] preferred video column: {video_col}", flush=True)

    with manifest_path.open("w", encoding="utf-8") as mf:
        while saved < args.num_samples and scanned < args.max_scan:
            resume_offset = scanned + skipped_rows
            ds = _build_stream(
                datasets_mod=datasets,
                dataset_id=resolved_dataset_id,
                split=args.split,
                video_col=video_col,
                shuffle_seed=args.seed,
                shuffle_buffer_size=args.shuffle_buffer_size,
            )
            if resume_offset > 0:
                ds = ds.skip(resume_offset)

            try:
                for row_idx, row in enumerate(ds, start=resume_offset):
                    if saved >= args.num_samples or scanned >= args.max_scan:
                        break
                    scanned += 1

                    caption, caption_key = _extract_caption(row, args.caption_key)
                    if caption is None:
                        missing_caption += 1
                        continue

                    video_value, resolved_video_col = _resolve_video_value(row, video_col)
                    if video_value is None:
                        missing_video += 1
                        continue

                    src_path, src_bytes = _extract_video_source(video_value)
                    ext = _guess_video_ext(src_path, src_bytes)

                    stem = f"sample_{saved:04d}"
                    video_name = f"{stem}{ext}"
                    caption_name = f"{stem}.caption.txt"
                    video_out = out_dir / video_name
                    caption_out = out_dir / caption_name

                    ok = False
                    if src_bytes is not None:
                        video_out.write_bytes(src_bytes)
                        ok = True
                    elif isinstance(src_path, str) and src_path:
                        p = Path(src_path)
                        if p.exists() and p.is_file():
                            shutil.copy2(p, video_out)
                            ok = True

                    if not ok:
                        save_fail += 1
                        continue

                    caption_out.write_text(caption + "\n", encoding="utf-8")

                    rec = {
                        "sample_id": stem,
                        "source_row_index": row_idx,
                        "video_file": video_name,
                        "caption_file": caption_name,
                        "caption": caption,
                        "video_column": resolved_video_col,
                        "caption_column": caption_key,
                    }
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    saved += 1
                    if saved % 8 == 0 or saved == args.num_samples:
                        print(f"[progress] saved={saved}/{args.num_samples} scanned={scanned}", flush=True)
                else:
                    # Exhausted stream without errors.
                    break
            except Exception as exc:
                if not _is_retryable_stream_error(exc) or restart_count >= int(args.max_restarts):
                    raise

                restart_count += 1
                skip_delta = int(args.skip_on_error)
                skipped_rows += skip_delta
                next_offset = scanned + skipped_rows
                print(
                    (
                        f"[warn] stream aborted: {type(exc).__name__}: {exc}\n"
                        f"[warn] restart {restart_count}/{int(args.max_restarts)}, "
                        f"skip_delta={skip_delta}, next_offset={next_offset}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if args.retry_sleep > 0:
                    time.sleep(float(args.retry_sleep))

    meta = {
        "dataset_id_requested": args.dataset_id,
        "dataset_id_resolved": resolved_dataset_id,
        "split": args.split,
        "num_samples_target": args.num_samples,
        "num_samples_saved": saved,
        "scanned_rows": scanned,
        "missing_caption_rows": missing_caption,
        "missing_video_rows": missing_video,
        "save_fail_rows": save_fail,
        "restart_count": restart_count,
        "skipped_rows": skipped_rows,
        "seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "max_scan": args.max_scan,
        "output_dir": str(out_dir),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if saved < args.num_samples:
        raise SystemExit(
            f"Only saved {saved}/{args.num_samples} samples after scanning {scanned} rows. "
            f"Try larger --max-scan / --shuffle-buffer-size, or set --caption-key/--video-col explicitly."
        )

    print(f"[done] Exported {saved} samples to: {out_dir}", flush=True)
    print(f"[done] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
