from __future__ import annotations

import argparse
from pathlib import Path

from video_recon_training.config import load_config, resolve_path
from video_recon_training.trainer import train


DEFAULT_CONFIG_PATH = str((Path(__file__).resolve().parent / "config.yaml").resolve())



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3VL video RAE with last-latent KV reconstruction.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override training.output_dir.")
    parser.add_argument("--resume-path", type=str, default=None, help="Optional checkpoint for resume.")
    parser.add_argument("--global-batch-size", type=int, default=None, help="Override training.global_batch_size.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    config_dir = str(Path(args.config).resolve().parent)

    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir

    if args.resume_path is not None:
        cfg["training"]["resume_path"] = resolve_path(args.resume_path, config_dir)

    if args.global_batch_size is not None:
        cfg["training"]["global_batch_size"] = int(args.global_batch_size)

    train(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
