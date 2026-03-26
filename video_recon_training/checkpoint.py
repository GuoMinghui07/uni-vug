from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch


def _pick_state_dict_from_checkpoint(
    checkpoint: Mapping[str, Any],
    source: str = "ema",
) -> Tuple[Mapping[str, Any], str]:
    source = str(source).strip()
    order = []
    if source:
        order.append(source)
    for key in ("ema", "model", "state_dict"):
        if key not in order:
            order.append(key)

    for key in order:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value, key

    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint, "root"

    raise KeyError(f"Cannot find a valid state dict. Tried keys: {order}")


def _extract_prefixed_state_dict(state_dict: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    nested_key = prefix[:-1] if prefix.endswith(".") else prefix
    nested = state_dict.get(nested_key)
    if isinstance(nested, Mapping):
        return dict(nested)

    extracted: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith(prefix):
            extracted[key[len(prefix) :]] = value
    return extracted


def load_stage1_decoder_weights(
    rae,
    checkpoint_path: str,
    source: str = "ema",
    strict: bool = True,
) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

    state_dict, used_source = _pick_state_dict_from_checkpoint(checkpoint, source=source)

    decoder_state = _extract_prefixed_state_dict(state_dict, "decoder.")
    if not decoder_state:
        raise KeyError("No decoder weights found in checkpoint.")
    rae.decoder.load_state_dict(decoder_state, strict=strict)

    to_pixels_state = _extract_prefixed_state_dict(state_dict, "to_pixels.")
    loaded_to_pixels = False
    if hasattr(rae, "to_pixels") and to_pixels_state:
        rae.to_pixels.load_state_dict(to_pixels_state, strict=strict)
        loaded_to_pixels = True

    return {
        "checkpoint_path": str(ckpt_path),
        "source": used_source,
        "decoder_keys": len(decoder_state),
        "to_pixels_keys_in_ckpt": len(to_pixels_state),
        "loaded_to_pixels": loaded_to_pixels,
    }


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_training_checkpoint(
    path: str,
    *,
    step: int,
    epoch: int,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
) -> None:
    state = {
        "step": int(step),
        "epoch": int(epoch),
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path_obj))


def load_training_checkpoint(
    path: str,
    *,
    model,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")

    _unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = int(checkpoint.get("epoch", 0))
    step = int(checkpoint.get("step", 0))
    return epoch, step
