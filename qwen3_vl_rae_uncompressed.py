from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import json

import torch
from torch import nn

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from rae_decoder.decoder import Decoder
from rae_decoder.decoder_config import DecoderConfig


def _safe_from_pretrained(cls, model_name: str, local_files_only: bool, **kwargs):
    try:
        return cls.from_pretrained(model_name, local_files_only=local_files_only, **kwargs)
    except Exception:
        return cls.from_pretrained(model_name, local_files_only=False, **kwargs)


@dataclass(frozen=True)
class _GridSpec:
    grid_t: int
    grid_h: int
    grid_w: int
    orig_t: int
    orig_h: int
    orig_w: int
    is_image: bool


class Qwen3VLDemerge(nn.Module):
    def __init__(self, hidden_size: int, out_hidden_size: int, merge_size: int) -> None:
        super().__init__()
        self.merge_size = int(merge_size)
        self.inner_dim = int(hidden_size) * self.merge_size * self.merge_size
        self.norm = nn.LayerNorm(out_hidden_size, eps=1e-6)
        self.fc1 = nn.Linear(out_hidden_size, self.inner_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.inner_dim, self.inner_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.norm.weight.dtype:
            x = x.to(self.norm.weight.dtype)
        x = self.fc2(self.act(self.fc1(self.norm(x))))
        b, n, _ = x.shape
        x = x.view(b, n, self.merge_size * self.merge_size, -1)
        return x.reshape(b, n * self.merge_size * self.merge_size, -1)


class Qwen3VLEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, local_files_only: bool = True, do_resize: bool = False) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.do_resize = bool(do_resize)

        model = _safe_from_pretrained(
            Qwen3VLForConditionalGeneration, model_name_or_path, local_files_only=local_files_only
        )
        self.visual = model.model.visual
        self.visual.requires_grad_(False)
        self.visual.eval()

        processor = _safe_from_pretrained(AutoProcessor, model_name_or_path, local_files_only=local_files_only)
        self.image_processor = processor.image_processor
        self.video_processor = processor.video_processor

        self.patch_size = int(self.visual.config.patch_size)
        self.temporal_patch_size = int(self.visual.config.temporal_patch_size)
        self.merge_size = int(self.visual.config.spatial_merge_size)
        self.hidden_size = int(self.visual.config.hidden_size)
        self.out_hidden_size = int(self.visual.config.out_hidden_size)

    def _to_device_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        ref = next(self.visual.parameters())
        return tensor.to(device=ref.device, dtype=ref.dtype)

    def _block_to_raster(
        self, tokens: torch.Tensor, grid_t: int, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        m = self.merge_size
        if m == 1:
            return tokens
        ghb, gwb = grid_h // m, grid_w // m
        b = tokens.shape[0]
        tokens = tokens.view(b, grid_t, ghb, gwb, m, m, -1)
        tokens = tokens.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return tokens.view(b, grid_t * grid_h * grid_w, -1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 4:
            processed = self.image_processor(
                images=x,
                do_rescale=False,
                do_normalize=False,
                do_resize=self.do_resize,
                return_tensors="pt",
            )
            pixel_values = processed["pixel_values"]
            grid_thw = processed["image_grid_thw"]
        elif x.dim() == 5:
            videos = [v.detach().cpu() for v in x]
            processed = self.video_processor(
                videos=videos,
                do_rescale=False,
                do_normalize=False,
                do_resize=self.do_resize,
                do_sample_frames=False,
                min_frames=1,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
                return_tensors="pt",
            )
            pixel_values = processed["pixel_values_videos"]
            grid_thw = processed["video_grid_thw"]
        else:
            raise ValueError(f"Expected input of shape (B, C, H, W) or (B, T, C, H, W), got {tuple(x.shape)}")

        if pixel_values.dim() == 3:
            seq = pixel_values.reshape(-1, pixel_values.shape[-1])
        elif pixel_values.dim() == 2:
            seq = pixel_values
        else:
            raise ValueError(f"Unexpected pixel_values shape: {tuple(pixel_values.shape)}")

        seq = self._to_device_dtype(seq)
        grid_thw = grid_thw.to(device=seq.device)

        out = self.visual(seq, grid_thw=grid_thw)
        hidden = None
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            hidden = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            hidden = out[0]
        else:
            hidden = out
        if not torch.is_tensor(hidden):
            raise TypeError(f"Unexpected visual output type: {type(out)}")
        split_sizes = grid_thw.prod(-1).tolist()
        if len(set(split_sizes)) != 1:
            raise ValueError("Mixed grid sizes in a batch are not supported for RAE training.")
        tokens = torch.stack(list(torch.split(hidden, split_sizes, dim=0)), dim=0)
        grid_t, grid_h, grid_w = (int(v) for v in grid_thw[0].tolist())
        tokens = self._block_to_raster(tokens, grid_t, grid_h, grid_w)
        return tokens, grid_thw


class Qwen3VLRAE(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        *,
        decoder_config: Optional[DecoderConfig] = None,
        decoder_config_path: Optional[str] = None,
        noise_tau: float = 0.8,
        in_channels: int = 3,
        denormalize_output: bool = True,
        local_files_only: bool = True,
        do_resize: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = Qwen3VLEncoder(
            model_name_or_path, local_files_only=local_files_only, do_resize=do_resize
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        self.patch_size = self.encoder.patch_size
        self.temporal_patch_size = self.encoder.temporal_patch_size
        self.merge_size = self.encoder.merge_size
        self.hidden_size = self.encoder.hidden_size
        self.out_hidden_size = self.encoder.out_hidden_size
        self.in_channels = int(in_channels)
        self.noise_tau = float(noise_tau)
        self.denormalize_output = bool(denormalize_output)

        img_mean = torch.tensor(self.encoder.image_processor.image_mean).view(1, 3, 1, 1)
        img_std = torch.tensor(self.encoder.image_processor.image_std).view(1, 3, 1, 1)
        vid_mean = torch.tensor(self.encoder.video_processor.image_mean).view(1, 1, 3, 1, 1)
        vid_std = torch.tensor(self.encoder.video_processor.image_std).view(1, 1, 3, 1, 1)
        self.register_buffer("image_mean", img_mean, persistent=False)
        self.register_buffer("image_std", img_std, persistent=False)
        self.register_buffer("video_mean", vid_mean, persistent=False)
        self.register_buffer("video_std", vid_std, persistent=False)

        if isinstance(decoder_config, dict):
            decoder_config = DecoderConfig(**decoder_config)
        if decoder_config is None and decoder_config_path is not None:
            cfg_path = Path(decoder_config_path)
            if cfg_path.is_dir():
                cfg_path = cfg_path / "config.json"
            with open(cfg_path, "r") as f:
                decoder_config = DecoderConfig(**json.load(f))
        if decoder_config is None:
            num_heads = getattr(self.encoder.visual.config, "num_heads", None)
            decoder_config = DecoderConfig(hidden_size=self.hidden_size, num_attention_heads=num_heads or 16)
        elif decoder_config.hidden_size != self.hidden_size:
            raise ValueError(
                f"decoder_config.hidden_size={decoder_config.hidden_size} must match encoder hidden_size={self.hidden_size}."
            )

        self.decoder = Decoder(decoder_config)
        self.patch_dim = self.temporal_patch_size * (self.patch_size ** 2) * self.in_channels
        self.to_pixels = nn.Linear(self.hidden_size, self.patch_dim, bias=True)

    def _normalize_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() == 4:
            mean, std = self.image_mean, self.image_std
        elif pixels.dim() == 5:
            mean, std = self.video_mean, self.video_std
        else:
            return pixels
        mean = mean.to(device=pixels.device, dtype=pixels.dtype)
        std = std.to(device=pixels.device, dtype=pixels.dtype)
        return (pixels - mean) / std

    def _denormalize_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() == 4:
            mean, std = self.image_mean, self.image_std
        elif pixels.dim() == 5:
            mean, std = self.video_mean, self.video_std
        else:
            return pixels
        mean = mean.to(device=pixels.device, dtype=pixels.dtype)
        std = std.to(device=pixels.device, dtype=pixels.dtype)
        return pixels * std + mean

    def _prepare_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, _GridSpec]:
        if x.dim() == 4:
            is_image = True
            _, c, h, w = x.shape
            t = 1
        elif x.dim() == 5:
            is_image = False
            _, t, c, h, w = x.shape
        else:
            raise ValueError(f"Expected input of shape (B, C, H, W) or (B, T, C, H, W), got {tuple(x.shape)}")
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}.")
        if not self.encoder.do_resize:
            if h % self.patch_size != 0 or w % self.patch_size != 0:
                raise ValueError(
                    f"Input H/W must be multiples of patch_size={self.patch_size}, got H={h}, W={w}."
                )
            if (h // self.patch_size) % self.merge_size != 0 or (w // self.patch_size) % self.merge_size != 0:
                raise ValueError(
                    f"Grid H/W must be divisible by merge_size={self.merge_size} (H={h}, W={w})."
                )

        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        grid_t = (t + self.temporal_patch_size - 1) // self.temporal_patch_size
        spec = _GridSpec(grid_t=grid_t, grid_h=grid_h, grid_w=grid_w, orig_t=t, orig_h=h, orig_w=w, is_image=is_image)
        return x, spec

    def _add_noise(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.noise_tau <= 0:
            return z
        sigma = self.noise_tau * torch.rand(
            (z.size(0),) + (1,) * (z.dim() - 1), device=z.device, dtype=z.dtype
        )
        return z + sigma * torch.randn_like(z)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, _GridSpec]:
        x, spec = self._prepare_input(x)
        x = self._normalize_pixels(x)
        with torch.no_grad():
            merged, grid_thw = self.encoder(x)
        grid_thw = grid_thw.to(device=merged.device)
        if (grid_thw != grid_thw[0]).any():
            raise ValueError("Mixed grid sizes in a batch are not supported for RAE training.")
        grid_t, grid_h, grid_w = (int(v) for v in grid_thw[0].tolist())
        if self.encoder.do_resize:
            orig_t = spec.orig_t
            orig_h = grid_h * self.patch_size
            orig_w = grid_w * self.patch_size
        else:
            orig_t, orig_h, orig_w = spec.orig_t, spec.orig_h, spec.orig_w
        spec = _GridSpec(
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            orig_t=orig_t,
            orig_h=orig_h,
            orig_w=orig_w,
            is_image=spec.is_image,
        )

        b = merged.shape[0]
        tokens = merged.view(b, grid_t, grid_h, grid_w, self.hidden_size)
        # tokens = self._add_noise(tokens)
        return tokens, spec

    def unpatchify(self, patch_tokens: torch.Tensor, grid_t: int, grid_h: int, grid_w: int) -> torch.Tensor:
        if patch_tokens.dim() == 3:
            b, n, c = patch_tokens.shape
            expected = grid_t * grid_h * grid_w
            if n != expected:
                raise ValueError(f"Token count {n} does not match grid {expected}.")
            patch_tokens = patch_tokens.view(b, grid_t, grid_h, grid_w, c)
        elif patch_tokens.dim() != 5:
            raise ValueError(f"Expected patch tokens with 3 or 5 dims, got {patch_tokens.dim()}.")

        if patch_tokens.shape[-1] != self.patch_dim:
            raise ValueError(f"Patch dim {patch_tokens.shape[-1]} does not match expected {self.patch_dim}.")

        b = patch_tokens.shape[0]
        patches = patch_tokens.view(
            b,
            grid_t,
            grid_h,
            grid_w,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 5, 4, 2, 6, 3, 7).contiguous()
        return patches.view(
            b,
            grid_t * self.temporal_patch_size,
            self.in_channels,
            grid_h * self.patch_size,
            grid_w * self.patch_size,
        )

    def decode(self, latent: torch.Tensor | tuple[torch.Tensor, _GridSpec], spec: Optional[_GridSpec] = None) -> torch.Tensor:
        if isinstance(latent, tuple):
            latent, spec = latent
        if latent.dim() != 5:
            raise ValueError(f"Expected latent grid of shape (B, T, H, W, C), got {tuple(latent.shape)}")
        decoded = self.decoder(latent)
        patch_tokens = self.to_pixels(decoded)

        if spec is None:
            grid_t, grid_h, grid_w = latent.shape[1:4]
            spec = _GridSpec(
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                orig_t=grid_t * self.temporal_patch_size,
                orig_h=grid_h * self.patch_size,
                orig_w=grid_w * self.patch_size,
                is_image=False,
            )

        pixels = self.unpatchify(patch_tokens, spec.grid_t, spec.grid_h, spec.grid_w)
        # if spec.is_image:
        #     pixels = pixels.mean(dim=1)
        # elif spec.orig_t < pixels.shape[1]:
        if not spec.is_image and spec.orig_t < pixels.shape[1]:
            pixels = pixels[:, : spec.orig_t]
        if pixels.shape[-2] != spec.orig_h or pixels.shape[-1] != spec.orig_w:
            pixels = pixels[..., : spec.orig_h, : spec.orig_w]
        if self.denormalize_output:
            pixels = self._denormalize_pixels(pixels)
        return pixels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, spec = self.encode(x)
        return self.decode(latent, spec)
