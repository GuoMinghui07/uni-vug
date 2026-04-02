from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from copy import deepcopy
from .dist_utils import setup_distributed
from datasets import load_dataset



def parse_configs(config: Union[DictConfig, str]) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    if isinstance(config, str):
        config = OmegaConf.load(config)
    rae_config = config.get("stage_1", None)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    eval_config = config.get("eval", None)
    return rae_config, stage2_config, transport_config, sampler_config, guidance_config, misc, training_config, eval_config

def none_or_str(value):
    if value == 'None':
        return None
    return value

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

# def prepare_dataloader(
#     data_path: Path,
#     batch_size: int,
#     workers: int,
#     rank: int,
#     world_size: int,
#     transform: List= None,
# ) -> Tuple[DataLoader, DistributedSampler]:
#     dataset = ImageFolder(str(data_path), transform=transform)
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         num_workers=workers,
#         pin_memory=True,
#         drop_last=True,
#     )
#     return loader, sampler

def get_autocast_scaler(args) -> Tuple[dict, torch.cuda.amp.GradScaler | None]:
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)
    
    return scaler, autocast_kwargs


class ImageNetArrowDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, split="train", transform=None):
        self.dataset = load_dataset(
            "ILSVRC/imagenet-1k", 
            split=split, 
            cache_dir=str(cache_dir),
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label

# 修改后的 prepare_dataloader
def prepare_dataloader(
    data_path: Path,      # 此时 data_path 对应你的 cache_dir (如 ./imagenet_data)
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    transform: any = None,
    split: str = "train"  # 新增 split 参数以区分训练和验证
) -> Tuple[DataLoader, DistributedSampler]:
    
    # 使用自定义的 Wrapper 代替 ImageFolder
    dataset = ImageNetArrowDataset(
        cache_dir=data_path, 
        split=split, 
        transform=transform
    )
    
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=(split == "train") # 只有训练集需要 shuffle
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=(split == "train"), # 训练集通常 drop_last 防止 batch size 不一
        persistent_workers=True if workers > 0 else False
    )
    return loader, sampler