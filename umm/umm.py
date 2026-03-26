from __future__ import annotations

import json
import math
import random
import re
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from transformers import TextIteratorStreamer


ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from DiT import UniDiT
from qwen3_vl_rae_uncompressed import Qwen3VLRAE
from t2i_training_single_ds.utils.model_utils import load_rae_decoder_weights


DEFAULT_CHECKPOINT = "/scratch/e1539128/ckpt-4gpu-new/dit_training/checkpoints/step-0000018000.pt"
ALLOWED_RESOLUTIONS = (256, 448, 512)
UNIVIDEO_T2I_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn "
    "faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, walking backwards, computer-generated environment, weak dynamics, "
    "distorted and erratic motions, unstable framing and a disorganized composition."
)


def resolve_dtype(precision: str) -> torch.dtype:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def compute_shift_ratio_from_dims(
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    channels: int,
    base_dim: int,
) -> float:
    input_dim = int(grid_t) * int(grid_h) * int(grid_w) * int(channels)
    if input_dim <= 0:
        raise ValueError(f"Invalid dims for shift ratio: {(grid_t, grid_h, grid_w, channels)}")
    return math.sqrt(float(input_dim) / float(base_dim))


def load_latent_stats(
    stats_path: str,
    *,
    expected_channels: int,
    device: torch.device,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Latent stats file not found: '{path}'.")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid latent stats format in '{path}': expected a dict.")

    mean = payload.get("mean")
    std = payload.get("std")
    if not torch.is_tensor(mean) or not torch.is_tensor(std):
        raise ValueError(f"Latent stats '{path}' must contain tensor entries 'mean' and 'std'.")

    mean = mean.to(torch.float32).flatten()
    std = std.to(torch.float32).flatten()
    if mean.numel() != expected_channels or std.numel() != expected_channels:
        raise ValueError(
            f"Latent stats channel mismatch in '{path}': "
            f"mean={mean.numel()}, std={std.numel()}, expected={expected_channels}."
        )

    std = torch.clamp(std, min=float(eps))
    return {
        "path": str(path),
        "mean": mean.view(1, 1, 1, 1, -1).to(device=device),
        "std": std.view(1, 1, 1, 1, -1).to(device=device),
    }


def denormalize_latent(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if latent.dim() != 5:
        raise ValueError(f"Expected latent shape (B, T, H, W, C), got {tuple(latent.shape)}")
    dtype = latent.dtype
    return (latent.to(torch.float32) * std + mean).to(dtype=dtype)


def to_uint8_image(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.detach().clamp(0, 1).to(torch.float32).permute(1, 2, 0).cpu().numpy()
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def sanitize_resolution(value: Any, default: int = 448) -> int:
    try:
        candidate = int(value)
    except Exception:
        candidate = int(default)
    if candidate in ALLOWED_RESOLUTIONS:
        return candidate
    return min(ALLOWED_RESOLUTIONS, key=lambda x: abs(x - candidate))


def sanitize_positive_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        candidate = int(value)
    except Exception:
        candidate = int(default)
    candidate = max(low, min(high, candidate))
    return int(candidate)


def ensure_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3:
            return Image.fromarray(arr).convert("RGB")
        raise ValueError(f"Unsupported numpy image shape: {arr.shape}")
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def _strip_json_fence(text: str) -> str:
    fenced = text.strip()
    if fenced.startswith("```"):
        fenced = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", fenced)
        fenced = re.sub(r"\s*```$", "", fenced)
    return fenced.strip()


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = _strip_json_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = cleaned.find("{")
    while start >= 0:
        depth = 0
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = cleaned[start : idx + 1]
                    try:
                        data = json.loads(chunk)
                        if isinstance(data, dict):
                            return data
                    except Exception:
                        break
        start = cleaned.find("{", start + 1)
    return None


def _parse_bool(value: Any, default: bool = True) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _image_prefix_tail_length(text: str) -> int:
    marker = "<image"
    max_tail = min(len(text), len(marker) - 1)
    for tail in range(max_tail, 0, -1):
        if marker.startswith(text[-tail:]):
            return tail
    return 0


def _find_tag_end(text: str) -> int:
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and (in_single or in_double):
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == ">" and (not in_single) and (not in_double):
            return idx
    return -1


@dataclass
class EngineConfig:
    config_path: str = str(ROOT / "training" / "config" / "dit_training.yaml")
    checkpoint_path: str = DEFAULT_CHECKPOINT
    device: str = "cuda:0"
    precision: Optional[str] = None
    sample_steps: Optional[int] = None
    latent_stats: Optional[str] = None
    disable_latent_norm: bool = False


class Stage2FMHead:
    def __init__(self, cfg: EngineConfig) -> None:
        self.config_path = Path(cfg.config_path)
        self.checkpoint_path = Path(cfg.checkpoint_path)
        self.device = torch.device(cfg.device)
        self.precision_override = cfg.precision
        self.sample_steps_override = cfg.sample_steps
        self.latent_stats_override = cfg.latent_stats
        self.disable_latent_norm = cfg.disable_latent_norm

        self.model: Optional[UniDiT] = None
        self.rae: Optional[Qwen3VLRAE] = None
        self.cfg: Optional[Mapping[str, Any]] = None
        self.autocast_dtype: torch.dtype = torch.bfloat16
        self.sample_steps_default: int = 10
        self.sample_guidance_scale_default: float = 7.0
        self.sample_negative_prompt: str = UNIVIDEO_T2I_NEGATIVE_PROMPT
        self.use_shift: bool = False
        self.shift_base_dim: int = 4096
        self.latent_channels: int = 1024
        self.latent_stats: Optional[Dict[str, Any]] = None

    def _resolve_existing_path(self, value: str) -> Path:
        raw = Path(value)
        if raw.exists():
            return raw
        candidate = ROOT / value
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Path not found: '{value}' and '{candidate}'.")

    def _to_device_inputs(self, tokenized: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: (v.to(device=self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in tokenized.items()
        }

    def load(self) -> None:
        if not self.config_path.exists():
            fallback = ROOT / "training" / "config" / self.config_path.name
            if fallback.exists():
                self.config_path = fallback
            else:
                raise FileNotFoundError(f"Config not found: '{self.config_path}'.")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: '{self.checkpoint_path}'.")

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            torch.cuda.set_device(self.device)

        cfg = OmegaConf.load(str(self.config_path))
        model_cfg = cfg.get("model", {})
        rae_cfg = cfg.get("rae", {})
        train_cfg = cfg.get("training", {})
        shift_cfg = cfg.get("schedule_shift", {})
        latent_norm_cfg = cfg.get("latent_norm", {})
        wandb_cfg = cfg.get("wandb", {})

        precision = self.precision_override if self.precision_override is not None else str(train_cfg.get("precision", "bf16"))
        self.autocast_dtype = resolve_dtype(precision)
        self.sample_steps_default = int(self.sample_steps_override) if self.sample_steps_override is not None else int(model_cfg.get("fm_num_steps", 10))
        self.sample_guidance_scale_default = float(wandb_cfg.get("sample_guidance_scale", 7.0))
        self.sample_negative_prompt = str(wandb_cfg.get("sample_negative_prompt", UNIVIDEO_T2I_NEGATIVE_PROMPT))

        model = UniDiT(
            mllm_model_name=str(model_cfg.get("mllm_model_name", "Qwen/Qwen3-VL-4B-Instruct")),
            num_metaqueries=int(model_cfg.get("num_metaqueries", 256)),
            mllm_hidden_size=int(model_cfg.get("mllm_hidden_size", 2560)),
            connector_expansion=int(model_cfg.get("connector_expansion", 4)),
            rae_latent_dim=int(model_cfg.get("rae_latent_dim", 1024)),
            dit_dim=int(model_cfg.get("dit_dim", 1536)),
            dit_ffn_dim=int(model_cfg.get("dit_ffn_dim", 8960)),
            dit_freq_dim=int(model_cfg.get("dit_freq_dim", 256)),
            dit_num_heads=int(model_cfg.get("dit_num_heads", 12)),
            dit_num_layers=int(model_cfg.get("dit_num_layers", 30)),
            dit_patch_size=tuple(model_cfg.get("dit_patch_size", [1, 1, 1])),
            dit_max_seq_len=int(model_cfg.get("dit_max_seq_len", 1024)),
            dit_window_size=tuple(model_cfg.get("dit_window_size", [-1, -1])),
            fm_num_steps=int(model_cfg.get("fm_num_steps", 50)),
            fm_logit_normal_mean=float(model_cfg.get("fm_logit_normal_mean", 0.0)),
            fm_logit_normal_std=float(model_cfg.get("fm_logit_normal_std", 1.0)),
            fm_model_timestep_scale=float(model_cfg.get("fm_model_timestep_scale", 1000.0)),
            local_files_only=bool(model_cfg.get("local_files_only", True)),
        ).to(self.device)

        self.use_shift = bool(shift_cfg.get("enabled", False))
        self.shift_base_dim = int(shift_cfg.get("base_dim", 4096))
        self.latent_channels = int(model_cfg.get("rae_latent_dim", 1024))
        model.flow_matching.use_schedule_shift = self.use_shift
        model.flow_matching.shift_ratio = 1.0

        decoder_cfg_path = self._resolve_existing_path(str(rae_cfg.get("decoder_config_path", "config/decoder_config.json")))
        rae = Qwen3VLRAE(
            model_name_or_path=str(rae_cfg.get("model_name_or_path", "Qwen/Qwen3-VL-4B-Instruct")),
            decoder_config_path=str(decoder_cfg_path),
            noise_tau=float(rae_cfg.get("noise_tau", 0.0)),
            in_channels=int(rae_cfg.get("in_channels", 3)),
            denormalize_output=bool(rae_cfg.get("denormalize_output", True)),
            local_files_only=bool(rae_cfg.get("local_files_only", True)),
            do_resize=bool(rae_cfg.get("do_resize", False)),
        ).to(self.device)

        decoder_ckpt_path = str(rae_cfg.get("decoder_checkpoint_path", "")).strip()
        if decoder_ckpt_path:
            load_rae_decoder_weights(
                rae=rae,
                checkpoint_path=self._resolve_existing_path(decoder_ckpt_path),
                source=str(rae_cfg.get("decoder_checkpoint_source", "ema")),
            )
        rae.requires_grad_(False)
        rae.eval()

        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        if not isinstance(ckpt, Mapping) or "model" not in ckpt or not isinstance(ckpt["model"], Mapping):
            raise KeyError("Checkpoint must be a dict containing 'model' state_dict.")
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        self.latent_stats = None
        latent_norm_enabled = bool(latent_norm_cfg.get("enabled", True)) and (not self.disable_latent_norm)
        if latent_norm_enabled:
            latent_stats_path = (
                self.latent_stats_override
                if self.latent_stats_override is not None
                else str(latent_norm_cfg.get("stats_path", "ckpts/latent_stats.pt"))
            )
            stats_path = self._resolve_existing_path(latent_stats_path)
            self.latent_stats = load_latent_stats(
                str(stats_path),
                expected_channels=self.latent_channels,
                device=self.device,
                eps=float(latent_norm_cfg.get("eps", 1e-6)),
            )

        self.cfg = cfg
        self.model = model
        self.rae = rae

    def _tokenize_condition(self, prompt: str, condition_images: Optional[Sequence[Image.Image]]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Image head not loaded.")
        mllm_encoder = self.model.mllm_encoder
        images = [ensure_pil_image(im) for im in (condition_images or [])]
        if not images:
            return mllm_encoder.tokenize([prompt], add_queries=True)

        content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})

        messages: List[Dict[str, Any]] = []
        if mllm_encoder.system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": mllm_encoder.system_prompt}]})
        messages.append({"role": "user", "content": content})
        if mllm_encoder.num_metaqueries > 0:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": mllm_encoder._query_token_text}],
                }
            )

        rendered = mllm_encoder.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = mllm_encoder.processor(
            text=[rendered],
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=mllm_encoder.max_input_text_tokens + mllm_encoder.num_metaqueries + 64,
        )
        if mllm_encoder.num_metaqueries > 0:
            mllm_encoder._validate_query_spans(inputs["input_ids"])
        return inputs

    @torch.no_grad()
    def generate_image(
        self,
        prompt: str,
        *,
        condition_images: Optional[Sequence[Image.Image]] = None,
        requested_size: int = 448,
        num_steps: Optional[int] = None,
        sample_guidance_scale: Optional[float] = None,
        seed: int = -1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.model is None or self.rae is None:
            raise RuntimeError("Image head is not loaded.")

        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Image prompt is empty.")

        requested_size = sanitize_resolution(requested_size, default=448)
        sample_steps = sanitize_positive_int(
            self.sample_steps_default if num_steps is None else num_steps,
            default=self.sample_steps_default,
            low=1,
            high=50,
        )
        guidance_scale = (
            self.sample_guidance_scale_default
            if sample_guidance_scale is None
            else float(sample_guidance_scale)
        )
        guidance_scale = max(0.0, min(10.0, guidance_scale))
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        patch_size = int(self.rae.patch_size)
        grid_h = max(1, requested_size // patch_size)
        grid_w = max(1, requested_size // patch_size)
        grid_t = 1
        internal_h = grid_h * patch_size
        internal_w = grid_w * patch_size

        if self.use_shift:
            self.model.flow_matching.shift_ratio = compute_shift_ratio_from_dims(
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                channels=self.latent_channels,
                base_dim=self.shift_base_dim,
            )

        random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        tokenized = self._tokenize_condition(prompt, condition_images)
        tokenized = self._to_device_inputs(tokenized)

        with autocast_ctx(self.device, self.autocast_dtype):
            context, context_mask = self.model.encode_condition(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                pixel_values=tokenized.get("pixel_values"),
                image_grid_thw=tokenized.get("image_grid_thw"),
                pixel_values_videos=tokenized.get("pixel_values_videos"),
                video_grid_thw=tokenized.get("video_grid_thw"),
                second_per_grid_ts=tokenized.get("second_per_grid_ts"),
            )
            negative_context = None
            negative_context_lens = None
            if abs(guidance_scale - 1.0) > 1e-6:
                negative_tokenized = self._tokenize_condition(self.sample_negative_prompt, condition_images)
                negative_tokenized = self._to_device_inputs(negative_tokenized)
                negative_context, negative_context_mask = self.model.encode_condition(
                    input_ids=negative_tokenized["input_ids"],
                    attention_mask=negative_tokenized["attention_mask"],
                    pixel_values=negative_tokenized.get("pixel_values"),
                    image_grid_thw=negative_tokenized.get("image_grid_thw"),
                    pixel_values_videos=negative_tokenized.get("pixel_values_videos"),
                    video_grid_thw=negative_tokenized.get("video_grid_thw"),
                    second_per_grid_ts=negative_tokenized.get("second_per_grid_ts"),
                )
                negative_context_lens = negative_context_mask.sum(dim=1)

            latent = self.model.sample(
                context=context,
                grid_t=grid_t,
                grid_h=grid_h,
                grid_w=grid_w,
                context_lens=context_mask.sum(dim=1),
                num_steps=sample_steps,
                negative_context=negative_context,
                negative_context_lens=negative_context_lens,
                guidance_scale=guidance_scale,
            )

        if self.latent_stats is not None:
            latent = denormalize_latent(latent, self.latent_stats["mean"], self.latent_stats["std"])

        with autocast_ctx(self.device, self.autocast_dtype):
            pixels = self.rae.decode(latent)

        if pixels.dim() == 5:
            pixels = pixels[:, 0]
        if pixels.shape[-2] != requested_size or pixels.shape[-1] != requested_size:
            pixels = F.interpolate(
                pixels.to(torch.float32),
                size=(requested_size, requested_size),
                mode="bicubic",
                align_corners=False,
            ).to(pixels.dtype)

        image = to_uint8_image(pixels[0])
        meta = {
            "requested_size": requested_size,
            "internal_size": (internal_h, internal_w),
            "output_size": (int(image.shape[0]), int(image.shape[1])),
            "num_steps": int(sample_steps),
            "sample_guidance_scale": float(guidance_scale),
            "seed": seed,
            "prompt": prompt,
            "num_condition_images": len(condition_images or []),
            "non_ema": True,
        }
        return image, meta


class InterleavedUMM:
    def __init__(self, cfg: EngineConfig) -> None:
        self.fm_head = Stage2FMHead(cfg)
        self.language_model = None
        self.processor = None
        self.tokenizer = None
        self.device = torch.device(cfg.device)

        self.planner_system_prompt = (
            "You are a multimodal assistant that can output interleaved text and images.\n"
            "Decide if image generation is needed. If needed, include image segments with highly specific prompts, if not, output only text.\n"
            "If you need to generate the image content, return ONLY JSON, no markdown, no explanation.\n"
            "Schema:\n"
            "{\n"
            '  "segments": [\n'
            '    {"type":"text","text":"..."} ,\n'
            '    {"type":"image","prompt":"detailed english prompt",'
            '"size":448,"steps":10,"seed":-1,"use_input_images":true}\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Keep segment order exactly as final response order.\n"
            "- If no image is needed, return only text segments.\n"
            "- At most 3 image segments.\n"
            "- For image prompt, be concrete about subject, composition, style, lighting, camera/lens.\n"
        )
        self.response_system_prompt = (
            "You are a multimodal assistant that can output interleaved text and images.\n"
            "Reply in natural markdown. Keep formulas and code blocks properly formatted.\n"
            "When you need an image at a specific position, emit exactly one inline tag at that position:\n"
            "<image prompt=\"detailed english prompt\" size=\"448\" steps=\"10\" seed=\"-1\" use_input_images=\"true\" />\n"
            "Rules:\n"
            "- Keep normal answer text outside the tag.\n"
            "- Prompt must be concrete and controllable.\n"
            "- At most 3 image tags.\n"
            "- If image is not needed, output text only.\n"
        )

    def load(self) -> None:
        self.fm_head.load()
        if self.fm_head.model is None:
            raise RuntimeError("Failed to initialize FM head model.")
        mllm_encoder = self.fm_head.model.mllm_encoder
        self.language_model = mllm_encoder.mllm
        self.processor = mllm_encoder.processor
        self.tokenizer = mllm_encoder._get_tokenizer()
        self.language_model.eval()

    def _history_to_transcript(self, history_state: Sequence[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for turn in history_state:
            role = str(turn.get("role", "assistant")).upper()
            segments = turn.get("segments", []) or []
            text_parts: List[str] = []
            for seg in segments:
                seg_type = str(seg.get("type", "text"))
                if seg_type == "text":
                    text = str(seg.get("text", "")).strip()
                    if text:
                        text_parts.append(text)
                elif seg_type == "image":
                    prompt = str(seg.get("prompt", "")).strip()
                    if prompt:
                        text_parts.append(f"<image prompt=\"{prompt}\">")
                    else:
                        text_parts.append("<image>")
            if text_parts:
                lines.append(f"{role}: " + " ".join(text_parts))
        return "\n".join(lines)

    def _build_planner_messages(
        self,
        *,
        user_text: str,
        user_images: Sequence[Image.Image],
        history_state: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        transcript = self._history_to_transcript(history_state)
        prompt_text = (
            "[Conversation so far]\n"
            f"{transcript if transcript else '(empty)'}\n\n"
            "[Current user request]\n"
            f"{user_text if user_text else '(empty)'}\n\n"
            "Now produce final response plan as JSON according to schema."
        )

        user_content: List[Dict[str, Any]] = [{"type": "image"} for _ in user_images]
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.planner_system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        return messages, list(user_images)

    def _build_response_messages(
        self,
        *,
        user_text: str,
        user_images: Sequence[Image.Image],
        history_state: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        transcript = self._history_to_transcript(history_state)
        prompt_text = (
            "[Conversation so far]\n"
            f"{transcript if transcript else '(empty)'}\n\n"
            "[Current user request]\n"
            f"{user_text if user_text else '(empty)'}\n\n"
            "Now answer directly. Use markdown. Emit <image ... /> tags only where an image should appear."
        )

        user_content: List[Dict[str, Any]] = [{"type": "image"} for _ in user_images]
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.response_system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        return messages, list(user_images)

    def _build_generation_kwargs(
        self,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        block_visual_tokens: bool = True,
    ) -> Dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        do_sample = temperature > 1e-5
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "temperature": float(max(temperature, 1e-5)),
            "top_p": float(min(max(top_p, 0.05), 1.0)),
        }
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = int(pad_id)
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = int(eos_id)
        if block_visual_tokens:
            if self.fm_head.model is None:
                raise RuntimeError("Image head model is not loaded.")
            mllm_encoder = self.fm_head.model.mllm_encoder
            start_id = int(mllm_encoder._query_token_start_id)
            end_id = int(mllm_encoder._query_token_end_id)
            if end_id > start_id:
                gen_kwargs["bad_words_ids"] = [[tok_id] for tok_id in range(start_id, end_id)]
        if not do_sample:
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
        return gen_kwargs

    @torch.no_grad()
    def _stream_response_raw(
        self,
        messages: List[Dict[str, Any]],
        images: Sequence[Image.Image],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        block_visual_tokens: bool = True,
    ) -> Iterator[str]:
        if self.language_model is None or self.processor is None or self.tokenizer is None:
            raise RuntimeError("InterleavedUMM is not loaded.")

        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        proc_inputs = self.processor(
            text=[rendered],
            images=list(images) if images else None,
            return_tensors="pt",
            padding=True,
        )
        proc_inputs = {
            k: (v.to(device=self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in proc_inputs.items()
        }
        gen_kwargs = self._build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            block_visual_tokens=block_visual_tokens,
        )
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs["streamer"] = streamer

        error_holder: Dict[str, Exception] = {}

        def _run_generate() -> None:
            try:
                with autocast_ctx(self.device, self.fm_head.autocast_dtype):
                    self.language_model.generate(**proc_inputs, **gen_kwargs)
            except Exception as exc:  # pragma: no cover - surfaced after thread join
                error_holder["error"] = exc
                try:
                    streamer.on_finalized_text("", stream_end=True)
                except Exception:
                    pass

        worker = threading.Thread(target=_run_generate, daemon=True)
        worker.start()
        try:
            for piece in streamer:
                if piece:
                    yield piece
        finally:
            worker.join()
        if "error" in error_holder:
            raise error_holder["error"]

    def _parse_image_tag(
        self,
        tag: str,
        *,
        default_size: int,
        default_steps: int,
        default_seed: int,
    ) -> Optional[Dict[str, Any]]:
        cleaned = tag.strip()
        match = re.match(r"^<image\b(.*?)/?>$", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        attrs_text = match.group(1).strip()
        attrs: Dict[str, str] = {}
        attr_pattern = re.compile(
            r"([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*(\".*?\"|'.*?'|[^\s\"'<>]+)",
            flags=re.DOTALL,
        )
        for attr_match in attr_pattern.finditer(attrs_text):
            key = str(attr_match.group(1)).strip().lower()
            value = str(attr_match.group(2)).strip()
            if len(value) >= 2 and (
                (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")
            ):
                value = value[1:-1]
            attrs[key] = value

        prompt = str(attrs.get("prompt", "")).strip()
        if not prompt:
            return None
        return {
            "type": "image",
            "prompt": prompt,
            "size": sanitize_resolution(attrs.get("size", default_size), default=default_size),
            "steps": sanitize_positive_int(attrs.get("steps", default_steps), default=default_steps, low=1, high=50),
            "seed": int(attrs.get("seed", default_seed)) if str(attrs.get("seed", default_seed)).lstrip("-").isdigit() else int(default_seed),
            "use_input_images": _parse_bool(attrs.get("use_input_images", "true"), default=True),
        }

    @torch.no_grad()
    def _generate_planner_raw(
        self,
        messages: List[Dict[str, Any]],
        images: Sequence[Image.Image],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        block_visual_tokens: bool = True,
    ) -> str:
        if self.language_model is None or self.processor is None or self.tokenizer is None:
            raise RuntimeError("InterleavedUMM is not loaded.")

        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        proc_inputs = self.processor(
            text=[rendered],
            images=list(images) if images else None,
            return_tensors="pt",
            padding=True,
        )
        proc_inputs = {
            k: (v.to(device=self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in proc_inputs.items()
        }

        do_sample = temperature > 1e-5
        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "temperature": float(max(temperature, 1e-5)),
            "top_p": float(min(max(top_p, 0.05), 1.0)),
        }
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = int(pad_id)
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = int(eos_id)
        if block_visual_tokens:
            if self.fm_head.model is None:
                raise RuntimeError("Image head model is not loaded.")
            mllm_encoder = self.fm_head.model.mllm_encoder
            start_id = int(mllm_encoder._query_token_start_id)
            end_id = int(mllm_encoder._query_token_end_id)
            if end_id > start_id:
                gen_kwargs["bad_words_ids"] = [[tok_id] for tok_id in range(start_id, end_id)]
        if not do_sample:
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)

        with autocast_ctx(self.device, self.fm_head.autocast_dtype):
            output_ids = self.language_model.generate(**proc_inputs, **gen_kwargs)

        prompt_len = int(proc_inputs["input_ids"].shape[1])
        new_tokens = output_ids[:, prompt_len:]
        text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text.strip()

    def _fallback_segments(
        self,
        *,
        user_text: str,
        planner_raw: str,
        resolution: int,
        num_steps: int,
        max_images: int,
    ) -> List[Dict[str, Any]]:
        raw = planner_raw.strip()
        lower_user = (user_text or "").lower()
        wants_image = any(
            token in lower_user
            for token in ("画", "图片", "图像", "生成图", "draw", "image", "illustrate", "render")
        )
        segments: List[Dict[str, Any]] = []
        if raw:
            segments.append({"type": "text", "text": raw})
        if wants_image and max_images > 0:
            segments.append(
                {
                    "type": "image",
                    "prompt": user_text.strip() or "A high quality image.",
                    "size": resolution,
                    "steps": num_steps,
                    "seed": -1,
                    "use_input_images": True,
                }
            )
        if not segments:
            segments.append({"type": "text", "text": "我准备好了，请告诉我你希望我生成什么内容。"})
        return segments

    def _plan_segments(
        self,
        *,
        user_text: str,
        user_images: Sequence[Image.Image],
        history_state: Sequence[Dict[str, Any]],
        resolution: int,
        num_steps: int,
        max_images: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        messages, planner_images = self._build_planner_messages(
            user_text=user_text,
            user_images=user_images,
            history_state=history_state,
        )
        planner_raw = self._generate_planner_raw(
            messages,
            planner_images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            block_visual_tokens=True,
        )
        plan = _extract_first_json_object(planner_raw)
        if not isinstance(plan, dict):
            return self._fallback_segments(
                user_text=user_text,
                planner_raw=planner_raw,
                resolution=resolution,
                num_steps=num_steps,
                max_images=max_images,
            ), planner_raw

        raw_segments = plan.get("segments")
        if not isinstance(raw_segments, list):
            return self._fallback_segments(
                user_text=user_text,
                planner_raw=planner_raw,
                resolution=resolution,
                num_steps=num_steps,
                max_images=max_images,
            ), planner_raw

        normalized: List[Dict[str, Any]] = []
        image_count = 0
        for seg in raw_segments:
            if not isinstance(seg, Mapping):
                continue
            seg_type = str(seg.get("type", "")).strip().lower()
            if seg_type == "text":
                text = str(seg.get("text", "")).strip()
                if text:
                    normalized.append({"type": "text", "text": text})
                continue

            if seg_type == "image" and image_count < max_images:
                prompt = str(seg.get("prompt", seg.get("text", ""))).strip()
                if not prompt:
                    continue
                image_count += 1
                normalized.append(
                    {
                        "type": "image",
                        "prompt": prompt,
                        "size": sanitize_resolution(seg.get("size", resolution), default=resolution),
                        "steps": sanitize_positive_int(seg.get("steps", num_steps), default=num_steps, low=1, high=50),
                        "seed": int(seg.get("seed", -1)) if str(seg.get("seed", "-1")).lstrip("-").isdigit() else -1,
                        "use_input_images": bool(seg.get("use_input_images", True)),
                    }
                )

        if not normalized:
            return self._fallback_segments(
                user_text=user_text,
                planner_raw=planner_raw,
                resolution=resolution,
                num_steps=num_steps,
                max_images=max_images,
            ), planner_raw
        return normalized, planner_raw

    def respond(
        self,
        *,
        user_text: str,
        user_images: Optional[Sequence[Any]],
        history_state: Optional[List[Dict[str, Any]]] = None,
        resolution: int = 448,
        num_steps: int = 10,
        sample_guidance_scale: float = 7.0,
        seed: int = -1,
        max_images: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.25,
        top_p: float = 0.9,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        final_payload: Optional[Dict[str, Any]] = None
        for payload in self.respond_stream(
            user_text=user_text,
            user_images=user_images,
            history_state=history_state,
            resolution=resolution,
            num_steps=num_steps,
            sample_guidance_scale=sample_guidance_scale,
            seed=seed,
            max_images=max_images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            text_stream_chars=128,
        ):
            if payload.get("event") == "final":
                final_payload = payload

        if not isinstance(final_payload, dict):
            raise RuntimeError("Streaming response finished without a final payload.")
        return (
            list(final_payload.get("ui_segments", [])),
            list(final_payload.get("history", [])),
            str(final_payload.get("planner_raw", "")),
        )

    def respond_stream(
        self,
        *,
        user_text: str,
        user_images: Optional[Sequence[Any]],
        history_state: Optional[List[Dict[str, Any]]] = None,
        resolution: int = 448,
        num_steps: int = 10,
        sample_guidance_scale: float = 7.0,
        seed: int = -1,
        max_images: int = 2,
        max_new_tokens: int = 512,
        temperature: float = 0.25,
        top_p: float = 0.9,
        text_stream_chars: int = 24,
    ) -> Iterator[Dict[str, Any]]:
        if self.language_model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        text = (user_text or "").strip()
        history = list(history_state or [])
        images = [ensure_pil_image(im) for im in (user_images or [])]

        resolution = sanitize_resolution(resolution, default=448)
        num_steps = sanitize_positive_int(num_steps, default=self.fm_head.sample_steps_default, low=1, high=50)
        sample_guidance_scale = float(max(0.0, min(10.0, sample_guidance_scale)))
        max_images = sanitize_positive_int(max_images, default=2, low=0, high=3)
        max_new_tokens = sanitize_positive_int(max_new_tokens, default=512, low=64, high=2048)
        temperature = float(max(0.0, min(2.0, temperature)))
        top_p = float(max(0.05, min(1.0, top_p)))
        _ = sanitize_positive_int(text_stream_chars, default=24, low=1, high=256)
        messages, stream_images = self._build_response_messages(
            user_text=text,
            user_images=images,
            history_state=history,
        )

        ui_segments: List[Dict[str, Any]] = []
        hist_segments: List[Dict[str, Any]] = []
        raw_text_parts: List[str] = []
        parser_buffer = ""
        emitted_images = 0

        seg_cursor = -1
        active_text_seg_idx: Optional[int] = None
        active_text_value = ""

        def _emit_text_delta(piece: str) -> Iterator[Dict[str, Any]]:
            nonlocal seg_cursor, active_text_seg_idx, active_text_value
            if not piece:
                return
            if active_text_seg_idx is None:
                seg_cursor += 1
                active_text_seg_idx = seg_cursor
                active_text_value = ""
            active_text_value += piece
            yield {
                "event": "text_delta",
                "segment_index": int(active_text_seg_idx),
                "text": active_text_value,
                "is_final": False,
            }

        def _finalize_active_text() -> Iterator[Dict[str, Any]]:
            nonlocal active_text_seg_idx, active_text_value
            if active_text_seg_idx is None:
                return
            final_text = active_text_value
            seg_idx = int(active_text_seg_idx)
            active_text_seg_idx = None
            active_text_value = ""
            if not final_text.strip():
                return
            yield {
                "event": "text_delta",
                "segment_index": seg_idx,
                "text": final_text,
                "is_final": True,
            }
            text_seg = {"type": "text", "text": final_text}
            ui_segments.append(text_seg)
            hist_segments.append(text_seg)

        for chunk in self._stream_response_raw(
            messages,
            stream_images,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            block_visual_tokens=True,
        ):
            raw_text_parts.append(chunk)
            yield {"event": "raw_delta", "raw": "".join(raw_text_parts)}

            parser_buffer += chunk
            while parser_buffer:
                marker_pos = parser_buffer.find("<image")
                if marker_pos < 0:
                    keep_tail = _image_prefix_tail_length(parser_buffer)
                    flush_text = parser_buffer[:-keep_tail] if keep_tail > 0 else parser_buffer
                    parser_buffer = parser_buffer[-keep_tail:] if keep_tail > 0 else ""
                    if flush_text:
                        for evt in _emit_text_delta(flush_text):
                            yield evt
                    break

                if marker_pos > 0:
                    prefix = parser_buffer[:marker_pos]
                    parser_buffer = parser_buffer[marker_pos:]
                    for evt in _emit_text_delta(prefix):
                        yield evt
                    continue

                tag_end = _find_tag_end(parser_buffer)
                if tag_end < 0:
                    break

                raw_tag = parser_buffer[: tag_end + 1]
                parser_buffer = parser_buffer[tag_end + 1 :]
                image_req = self._parse_image_tag(
                    raw_tag,
                    default_size=resolution,
                    default_steps=num_steps,
                    default_seed=int(seed),
                )
                if image_req is None or emitted_images >= max_images:
                    for evt in _emit_text_delta(raw_tag):
                        yield evt
                    continue

                for evt in _finalize_active_text():
                    yield evt

                emitted_images += 1
                seg_cursor += 1
                image_seg_idx = int(seg_cursor)
                seg_seed = int(image_req.get("seed", -1))
                if seg_seed < 0 and int(seed) >= 0:
                    seg_seed = int(seed)
                cond_images = images if bool(image_req.get("use_input_images", True)) else None

                yield {
                    "event": "image_start",
                    "segment_index": image_seg_idx,
                    "prompt": str(image_req["prompt"]),
                    "size": int(image_req["size"]),
                    "steps": int(image_req["steps"]),
                    "sample_guidance_scale": float(sample_guidance_scale),
                }
                try:
                    image, meta = self.fm_head.generate_image(
                        str(image_req["prompt"]),
                        condition_images=cond_images,
                        requested_size=int(image_req["size"]),
                        num_steps=int(image_req["steps"]),
                        sample_guidance_scale=float(sample_guidance_scale),
                        seed=int(seg_seed),
                    )
                    image_seg = {
                        "type": "image",
                        "image": image,
                        "prompt": str(image_req["prompt"]),
                        "meta": meta,
                    }
                    ui_segments.append(image_seg)
                    hist_segments.append(
                        {
                            "type": "image",
                            "prompt": str(image_req["prompt"]),
                            "meta": meta,
                        }
                    )
                    yield {
                        "event": "image_done",
                        "segment_index": image_seg_idx,
                        "segment": image_seg,
                    }
                except Exception as exc:
                    err_text = f"[image generation error] {type(exc).__name__}: {exc}"
                    ui_segments.append({"type": "text", "text": err_text})
                    hist_segments.append({"type": "text", "text": err_text})
                    yield {
                        "event": "image_error",
                        "segment_index": image_seg_idx,
                        "error": err_text,
                    }

        if parser_buffer:
            for evt in _emit_text_delta(parser_buffer):
                yield evt
            parser_buffer = ""
        for evt in _finalize_active_text():
            yield evt

        if not ui_segments:
            fallback = "我现在无法构造有效响应，请换一种描述再试一次。"
            fallback_seg = {"type": "text", "text": fallback}
            ui_segments.append(fallback_seg)
            hist_segments.append(fallback_seg)
            yield {
                "event": "text_delta",
                "segment_index": -1,
                "text": fallback,
                "is_final": True,
            }

        user_hist_segments: List[Dict[str, Any]] = []
        if text:
            user_hist_segments.append({"type": "text", "text": text})
        for _ in images:
            user_hist_segments.append({"type": "image", "prompt": "<user_uploaded_image>"})

        history.append({"role": "user", "segments": user_hist_segments})
        history.append({"role": "assistant", "segments": hist_segments})
        yield {
            "event": "final",
            "ui_segments": ui_segments,
            "history": history,
            "planner_raw": "".join(raw_text_parts),
        }


def build_default_engine(
    *,
    config_path: str = str(ROOT / "training" / "config" / "dit_training.yaml"),
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    device: str = "cuda:0",
    precision: Optional[str] = None,
    sample_steps: Optional[int] = None,
    latent_stats: Optional[str] = None,
    disable_latent_norm: bool = False,
) -> InterleavedUMM:
    cfg = EngineConfig(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        precision=precision,
        sample_steps=sample_steps,
        latent_stats=latent_stats,
        disable_latent_norm=disable_latent_norm,
    )
    engine = InterleavedUMM(cfg)
    engine.load()
    return engine
