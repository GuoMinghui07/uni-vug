from __future__ import annotations

import contextlib
from typing import Sequence

import torch
import torch.nn as nn
from torchvision import models

try:
    from torchvision.models import VGG19_Weights
except Exception:  # pragma: no cover - older torchvision
    VGG19_Weights = None


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _resolve_vgg19(weights: str | None) -> models.VGG:
    if VGG19_Weights is None:
        use_pretrained = weights is not None and str(weights).lower() not in {"", "none", "null"}
        return models.vgg19(pretrained=use_pretrained)

    if weights is None or str(weights).lower() in {"", "none", "null"}:
        weights_enum = None
    elif isinstance(weights, str):
        name = weights.strip()
        if name.upper() == "DEFAULT":
            weights_enum = VGG19_Weights.DEFAULT
        else:
            try:
                weights_enum = getattr(VGG19_Weights, name)
            except AttributeError as exc:
                raise ValueError(f"Unsupported VGG19 weights '{weights}'.") from exc
    else:
        weights_enum = weights

    return models.vgg19(weights=weights_enum)


class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layers: Sequence[int], vgg_weights: str | None = "IMAGENET1K_V1") -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must be a non-empty sequence of VGG feature indices.")
        self.layers = sorted({int(idx) for idx in layers})
        self.layer_set = set(self.layers)
        self.max_layer = max(self.layers)

        vgg = _resolve_vgg19(vgg_weights)
        self.features = vgg.features
        self.features.eval()
        self.features.requires_grad_(False)

    def forward(self, tensor: torch.Tensor) -> dict[int, torch.Tensor]:
        outputs: dict[int, torch.Tensor] = {}
        for idx, layer in enumerate(self.features):
            tensor = layer(tensor)
            if idx in self.layer_set:
                outputs[idx] = tensor
            if idx >= self.max_layer:
                break
        return outputs


class GramLossVGG19(nn.Module):
    def __init__(
        self,
        layers: Sequence[int] = (0, 5, 10, 19, 28),
        layer_weights: Sequence[float] | None = None,
        loss_type: str = "l1",
        input_range: str = "minus1_1",
        use_imagenet_norm: bool = True,
        vgg_weights: str | None = "IMAGENET1K_V1",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.layers = [int(idx) for idx in layers]
        if not self.layers:
            raise ValueError("layers must be a non-empty sequence of VGG feature indices.")

        loss_type = loss_type.lower()
        if loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type '{loss_type}'.")
        if input_range not in {"minus1_1", "0_1"}:
            raise ValueError(f"Unsupported input_range '{input_range}'.")

        self.loss_type = loss_type
        self.input_range = input_range
        self.use_imagenet_norm = bool(use_imagenet_norm)

        if layer_weights is None:
            weights = [1.0 / len(self.layers)] * len(self.layers)
        else:
            if len(layer_weights) != len(self.layers):
                raise ValueError("layer_weights must match the length of layers.")
            weights = [float(weight) for weight in layer_weights]
        self.register_buffer("layer_weights", torch.tensor(weights, dtype=torch.float32))

        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1))

        self.vgg = VGG19FeatureExtractor(self.layers, vgg_weights=vgg_weights)
        self.vgg.eval()
        self.vgg.requires_grad_(False)

        if device is not None:
            self.to(device)

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.input_range == "minus1_1":
            tensor = (tensor + 1.0) * 0.5
        tensor = tensor.clamp(0.0, 1.0)
        if self.use_imagenet_norm:
            tensor = (tensor - self.mean) / self.std
        return tensor

    @staticmethod
    def _gram_matrix(features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        features = features.view(batch, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (channels * height * width)
        return gram

    @staticmethod
    def _as_5d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4:
            return tensor.unsqueeze(1)
        return tensor

    def _align_video_inputs(
        self, x_hat: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_hat = self._as_5d(x_hat)
        x = self._as_5d(x)
        if x_hat.dim() != 5 or x.dim() != 5:
            raise ValueError("Expected 4D or 5D tensors for Gram loss inputs.")

        if x_hat.shape[1] == 1 and x.shape[1] > 1:
            x_hat = x_hat.expand(-1, x.shape[1], -1, -1, -1)
        if x.shape[1] == 1 and x_hat.shape[1] > 1:
            x = x.expand(-1, x_hat.shape[1], -1, -1, -1)

        if x_hat.shape[1] != x.shape[1]:
            raise ValueError(
                f"Mismatched temporal dims for Gram loss: {x_hat.shape} vs {x.shape}."
            )
        return x_hat, x

    def _gram_loss_4d(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_hat = x_hat.float()
        x = x.float()
        x_hat = self.preprocess(x_hat)
        x = self.preprocess(x)

        if x_hat.is_cuda:
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            autocast_ctx = contextlib.nullcontext()
        with autocast_ctx:
            feats_hat = self.vgg(x_hat)
            with torch.no_grad():
                feats_target = self.vgg(x)

        total = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
        for idx, weight in zip(self.layers, self.layer_weights):
            gram_hat = self._gram_matrix(feats_hat[idx])
            gram_target = self._gram_matrix(feats_target[idx]).detach()
            if self.loss_type == "l1":
                layer_loss = (gram_hat - gram_target).abs().mean()
            else:
                diff = gram_hat - gram_target
                layer_loss = (diff * diff).sum(dim=(1, 2)).mean()
            total = total + weight * layer_loss
        return total

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        if x_hat.dim() not in {4, 5} or x.dim() not in {4, 5}:
            raise ValueError("Gram loss expects 4D or 5D inputs.")

        if x_hat.dim() == 5 or x.dim() == 5:
            x_hat, x = self._align_video_inputs(x_hat, x)
            batch, frames, channels, height, width = x_hat.shape
            x_hat = x_hat.reshape(batch * frames, channels, height, width)
            x = x.reshape(batch * frames, channels, height, width)

        return self._gram_loss_4d(x_hat, x)
