from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from qwen3_vl_rae_uncompressed import Qwen3VLRAE


class LatentKVCache:
    def __init__(self, max_history_latents: int) -> None:
        self.max_history_latents = int(max(0, max_history_latents))
        self._cache: Optional[torch.Tensor] = None

    def get_kv(self, current_latent: torch.Tensor) -> torch.Tensor:
        if self._cache is None:
            return current_latent
        return torch.cat([self._cache, current_latent], dim=1)

    def push(self, current_latent: torch.Tensor) -> None:
        current = current_latent.detach()
        if self.max_history_latents == 0:
            self._cache = None
            return

        if self._cache is None:
            self._cache = current
        else:
            self._cache = torch.cat([self._cache, current], dim=1)

        if self._cache.shape[1] > self.max_history_latents:
            self._cache = self._cache[:, -self.max_history_latents :]


class TemporalVideoRAE(nn.Module):
    """RAE wrapper for last-latent reconstruction with history KV context."""

    def __init__(self, rae: Qwen3VLRAE, chunk_frames: int) -> None:
        super().__init__()
        self.rae = rae
        self.chunk_frames = int(chunk_frames)

        if self.chunk_frames <= 0 or self.chunk_frames % 2 != 0:
            raise ValueError(f"chunk_frames must be positive and even, got {self.chunk_frames}")

        temporal_patch_size = int(getattr(self.rae, "temporal_patch_size", 0))
        if temporal_patch_size != 2:
            raise ValueError(f"Expected temporal_patch_size=2, got {temporal_patch_size}")

        self.latent_steps = self.chunk_frames // temporal_patch_size
        self.max_history_latents = max(0, self.latent_steps - 1)

    def freeze_encoder_only_train_decoder(self) -> None:
        self.rae.requires_grad_(False)
        self.rae.encoder.requires_grad_(False)
        self.rae.decoder.requires_grad_(True)
        if hasattr(self.rae, "to_pixels"):
            self.rae.to_pixels.requires_grad_(True)

    def trainable_parameters(self):
        params = list(self.rae.decoder.parameters())
        if hasattr(self.rae, "to_pixels"):
            params += list(self.rae.to_pixels.parameters())
        return params

    def encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        latents, _ = self.rae.encode(chunk)
        return latents

    def decode_last_with_context(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() != 5:
            raise ValueError(f"Expected latent shape (B,T,H,W,C), got {tuple(latents.shape)}")

        if not hasattr(self.rae.decoder, "forward_last"):
            raise RuntimeError("Decoder does not provide forward_last; required for query-last reconstruction.")

        decoded_last = self.rae.decoder.forward_last(latents)
        patch_tokens = self.rae.to_pixels(decoded_last)
        pixels = self.rae.unpatchify(
            patch_tokens,
            grid_t=1,
            grid_h=int(latents.shape[2]),
            grid_w=int(latents.shape[3]),
        )

        # temporal_patch_size is 2, so this exactly corresponds to the last two frames.
        pixels = pixels[:, :2]
        if bool(getattr(self.rae, "denormalize_output", False)):
            pixels = self.rae._denormalize_pixels(pixels)
        return pixels

    def forward(self, chunk: torch.Tensor, return_latents: bool = False):
        latents = self.encode_chunk(chunk)
        recon = self.decode_last_with_context(latents)
        if return_latents:
            return recon, latents
        return recon

    def init_kv_cache(self) -> LatentKVCache:
        return LatentKVCache(self.max_history_latents)

    def infer_step(self, current_latent: torch.Tensor, cache: LatentKVCache) -> torch.Tensor:
        kv_latents = cache.get_kv(current_latent)
        recon = self.decode_last_with_context(kv_latents)
        cache.push(current_latent)
        return recon

    def infer_sequence(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        if latent_sequence.dim() != 5:
            raise ValueError(f"Expected latent sequence shape (B,T,H,W,C), got {tuple(latent_sequence.shape)}")

        cache = self.init_kv_cache()
        outputs = []
        for idx in range(latent_sequence.shape[1]):
            current_latent = latent_sequence[:, idx : idx + 1]
            outputs.append(self.infer_step(current_latent, cache))
        return torch.stack(outputs, dim=1)



def _load_decoder_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Decoder config must be a dict, got {type(payload)}")
    return payload



def build_temporal_video_rae(
    model_cfg: Dict[str, Any],
    *,
    chunk_frames: int,
    config_dir: str,
) -> TemporalVideoRAE:
    decoder_cfg_path = Path(model_cfg["decoder_config_path"])
    if not decoder_cfg_path.is_absolute():
        decoder_cfg_path = (Path(config_dir) / decoder_cfg_path).resolve()

    if not decoder_cfg_path.exists():
        raise FileNotFoundError(f"Decoder config not found: {decoder_cfg_path}")

    decoder_cfg = _load_decoder_config(str(decoder_cfg_path))

    window_size = model_cfg.get("decoder_window_size", [4096, 8, 8])
    shift_size = model_cfg.get("decoder_shift_size", [0, 4, 4])
    decoder_cfg["window_size"] = [int(x) for x in window_size]
    decoder_cfg["shift_size"] = [int(x) for x in shift_size]

    rae = Qwen3VLRAE(
        model_name_or_path=str(model_cfg.get("model_name_or_path", "Qwen/Qwen3-VL-4B-Instruct")),
        decoder_config=decoder_cfg,
        noise_tau=float(model_cfg.get("noise_tau", 0.0)),
        in_channels=int(model_cfg.get("in_channels", 3)),
        denormalize_output=bool(model_cfg.get("denormalize_output", True)),
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        do_resize=bool(model_cfg.get("do_resize", False)),
    )

    return TemporalVideoRAE(rae=rae, chunk_frames=chunk_frames)
