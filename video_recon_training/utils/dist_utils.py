import os
from typing import Tuple

import torch
import torch.distributed as dist


def setup_distributed() -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)

            # Prefer explicit device binding when supported (newer PyTorch).
            try:
                dist.init_process_group(backend="nccl", device_id=device)
            except TypeError:
                dist.init_process_group(backend="nccl")
        else:
            local_rank = 0
            device = torch.device("cpu")
            dist.init_process_group(backend="gloo")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
