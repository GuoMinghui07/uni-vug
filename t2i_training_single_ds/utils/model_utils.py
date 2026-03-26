"""Model instantiation utilities."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch

__all__ = [
    "instantiate_from_config",
    "load_rae_decoder_weights",
]


def instantiate_from_config(config) -> Any:
    """Instantiate a class from an OmegaConf-style config with 'target' and 'params'."""
    if hasattr(config, "target"):
        target = config.target
        params = dict(config.get("params", {}))
    elif isinstance(config, dict):
        target = config["target"]
        params = dict(config.get("params", {}))
    else:
        raise ValueError(f"Cannot parse config: {config}")

    module_path, _, class_name = target.rpartition(".")
    if not module_path:
        # target is just a module-level name, import the module itself
        module = importlib.import_module(class_name)
        return module
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**params)


def _pick_state_dict_from_checkpoint(
    checkpoint: Mapping[str, Any],
    source: str = "ema",
) -> Tuple[Mapping[str, Any], str]:
    source = str(source).strip()
    source_order = []
    if source:
        source_order.append(source)
    for key in ("ema", "model", "state_dict"):
        if key not in source_order:
            source_order.append(key)

    for key in source_order:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value, key

    if checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        return checkpoint, "root"

    raise KeyError(f"Cannot find a valid state dict. Tried keys: {source_order}.")


def _extract_prefixed_state_dict(state_dict: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    """Extract sub-state by prefix and strip prefix (e.g., 'decoder.')."""
    nested_key = prefix[:-1] if prefix.endswith(".") else prefix
    nested = state_dict.get(nested_key)
    if isinstance(nested, Mapping):
        return dict(nested)

    extracted = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith(prefix):
            extracted[key[len(prefix) :]] = value
    return extracted


def load_rae_decoder_weights(
    rae,
    checkpoint_path: str | Path,
    source: str = "ema",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load stage-1 decoder weights into a RAE-like module and freeze decoder-side params.

    Expected checkpoint formats:
    - {'ema': state_dict, 'model': state_dict, ...}
    - direct state_dict
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{ckpt_path}'.")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unexpected checkpoint format at '{ckpt_path}': {type(checkpoint)}")

    state_dict, used_source = _pick_state_dict_from_checkpoint(checkpoint, source=source)

    decoder_state = _extract_prefixed_state_dict(state_dict, "decoder.")
    if not decoder_state:
        raise KeyError(
            f"No 'decoder' weights found in checkpoint '{ckpt_path}' (source='{used_source}')."
        )
    rae.decoder.load_state_dict(decoder_state, strict=strict)

    to_pixels_state = _extract_prefixed_state_dict(state_dict, "to_pixels.")
    if hasattr(rae, "to_pixels") and to_pixels_state:
        rae.to_pixels.load_state_dict(to_pixels_state, strict=strict)

    # Always freeze decoder-side modules for stage-2 usage.
    if hasattr(rae, "decoder"):
        rae.decoder.requires_grad_(False)
    if hasattr(rae, "to_pixels"):
        rae.to_pixels.requires_grad_(False)
    if hasattr(rae, "demerge"):
        rae.demerge.requires_grad_(False)

    return {
        "checkpoint_path": str(ckpt_path),
        "source": used_source,
        "decoder_keys": len(decoder_state),
        "to_pixels_keys_in_ckpt": len(to_pixels_state),
        "loaded_to_pixels": bool(hasattr(rae, "to_pixels") and to_pixels_state),
    }
