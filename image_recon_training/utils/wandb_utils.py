import wandb
import torch
import os
import tempfile
import logging
from torchvision.utils import make_grid
try:
    from torchvision.io import write_video
except Exception:
    write_video = None
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math
import sys
import logging

def create_logger(logging_dir: str, logger_name: str) -> logging.Logger:
    """
    Create a logger that writes to a log file and stdout.
    Only rank 0 writes; other ranks get a dummy logger.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(logger_name)  # use provided logger name

    if rank == 0:
        # Make sure log dir exists
        os.makedirs(logging_dir, exist_ok=True)

        # Clear any existing handlers so we can reconfigure
        for h in list(logger.handlers):
            logger.removeHandler(h)

        logger.setLevel(logging.INFO)
        logger.propagate = False  # don't double-log via root

        fmt = logging.Formatter(
            '[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    else:
        # Dummy logger: no handlers, no output
        logger.setLevel(logging.CRITICAL + 1)
        logger.propagate = False
        for h in list(logger.handlers):
            logger.removeHandler(h)

    return logger

def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    if is_main_process():
        if "WANDB_KEY" in os.environ:
            wandb.login(key=os.environ["WANDB_KEY"])
        else:
            # assert already logged in
            pass
        wandb.init(
            entity=entity,
            project=project_name,
            name=exp_name,
            config=config_dict,
            id=generate_run_id(exp_name),
            resume="allow",
            reinit=True,
        )


def log(stats, step=None, commit: bool = True):
    if is_main_process():
        # print(f"WandB logging at step {step}: {stats}")
        wandb.log({k: v for k, v in stats.items()}, step=step, commit=commit)


def log_image(sample, step=None, commit: bool = True):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample)}, step=step, commit=commit)


def log_video(video, step=None, fps=4, name="samples/video", commit: bool = True, force_write: bool = False):
    if is_main_process():
        if not force_write:
            try:
                if isinstance(video, torch.Tensor):
                    video = video.detach().cpu().numpy()
                wandb.log({name: wandb.Video(video, fps=fps, format="mp4")}, step=step, commit=commit)
                return
            except Exception as exc:
                if "moviepy" not in str(exc).lower():
                    raise
                if write_video is None:
                    logging.warning("wandb.Video requires moviepy; torchvision.io.write_video unavailable.")
                    return
        # Fallback: write mp4 with torchvision and log file path
        if isinstance(video, torch.Tensor):
            v = video.detach().cpu()
        else:
            v = torch.from_numpy(video)
        if v.ndim == 4 and v.shape[-1] not in (1, 3, 4) and v.shape[1] in (1, 3, 4):
            v = v.permute(0, 2, 3, 1)
        v = v.contiguous()
        if v.dtype != torch.uint8:
            v = v.clamp(0, 255).to(torch.uint8)
        out_dir = None
        try:
            out_dir = wandb.run.dir if wandb.run is not None else None
        except Exception:
            out_dir = None
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        safe_name = name.replace("/", "_")
        out_path = os.path.join(out_dir, f"{safe_name}_step{step if step is not None else 'na'}.mp4")
        write_video(out_path, v, fps=fps)
        wandb.log({name: wandb.Video(out_path, fps=fps, format="mp4")}, step=step, commit=commit)


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0,1))
    x = x.clamp(0, 1).mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x
