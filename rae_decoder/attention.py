import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _apply_rope_1d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


def _get_3d_pos_indices(
    t: int, h: int, w: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_idx = torch.arange(t, device=device)
    h_idx = torch.arange(h, device=device)
    w_idx = torch.arange(w, device=device)
    grid_t, grid_h, grid_w = torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij")
    return grid_t.reshape(-1), grid_h.reshape(-1), grid_w.reshape(-1)


def _normalize_window_args(
    window_size: Sequence[int], shift_window: Optional[Sequence[int]]
) -> Tuple[int, int, int, int, int, int]:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if len(window_size) == 2:
        win_t = 1
        win_h, win_w = window_size
        if shift_window is None:
            shift_window = (0, 0)
        if isinstance(shift_window, int):
            shift_window = (shift_window, shift_window)
        if len(shift_window) != 2:
            raise ValueError("shift_window must have 2 values for 2D window attention.")
        shift_t = 0
        shift_h, shift_w = shift_window
    elif len(window_size) == 3:
        win_t, win_h, win_w = window_size
        if shift_window is None:
            shift_window = (0, 0, 0)
        if isinstance(shift_window, int):
            shift_window = (shift_window, shift_window, shift_window)
        if len(shift_window) != 3:
            raise ValueError("shift_window must have 3 values for 3D window attention.")
        shift_t, shift_h, shift_w = shift_window
    else:
        raise ValueError("window_size must have 2 or 3 values.")

    if win_t <= 0 or win_h <= 0 or win_w <= 0:
        raise ValueError("window_size values must be positive.")
    if shift_t < 0 or shift_h < 0 or shift_w < 0:
        raise ValueError("shift_window values must be non-negative.")

    return win_t, win_h, win_w, shift_t, shift_h, shift_w


def full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_backend: str,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if attn_backend == "xformers":
        import xformers.ops as xops

        return xops.memory_efficient_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_bias=None,
            p=dropout_p,
            scale=scale,
        )

    if attn_backend == "flash_attn":
        import flash_attn

        if q.shape[1] == k.shape[1]:
            qkv = torch.stack((q, k, v), dim=2).contiguous()
            return flash_attn.flash_attn_qkvpacked_func(
                qkv, dropout_p=dropout_p, softmax_scale=scale
            )
        return flash_attn.flash_attn_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            dropout_p=dropout_p,
            softmax_scale=scale,
            causal=False,
        )

    if attn_backend in ("sdpa", "torch", "torch_sdpa"):
        q_t = q.permute(0, 2, 1, 3).contiguous()
        k_t = k.permute(0, 2, 1, 3).contiguous()
        v_t = v.permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=None,
            dropout_p=dropout_p,
            scale=scale,
        )
        return out.permute(0, 2, 1, 3).contiguous()

    raise ValueError(f"Unknown attention module: {attn_backend}")


def window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t: int,
    h: int,
    w: int,
    window_size: Sequence[int],
    shift_window: Optional[Sequence[int]],
    attn_backend: str,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    *,
    pos_idx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if attn_backend == "xformers":
        import xformers.ops as xops
    elif attn_backend == "flash_attn":
        import flash_attn
    elif attn_backend in ("sdpa", "torch", "torch_sdpa"):
        xops = None
        flash_attn = None
    else:
        raise ValueError(f"Unknown attention module: {attn_backend}")

    win_t, win_h, win_w, shift_t, shift_h, shift_w = _normalize_window_args(window_size, shift_window)
    device = q.device

    if pos_idx is None:
        t_idx, h_idx, w_idx = _get_3d_pos_indices(t, h, w, device)
    else:
        t_idx, h_idx, w_idx = pos_idx

    t_win = (t_idx + shift_t) // win_t
    h_win = (h_idx + shift_h) // win_h
    w_win = (w_idx + shift_w) // win_w

    num_win_t = (t - 1 + shift_t) // win_t + 1
    num_win_h = (h - 1 + shift_h) // win_h + 1
    num_win_w = (w - 1 + shift_w) // win_w + 1
    num_windows_per_sample = num_win_t * num_win_h * num_win_w

    window_id = (t_win * (num_win_h * num_win_w) + h_win * num_win_w + w_win).to(torch.int64)

    bsz, n_tokens, num_heads, head_dim = q.shape

    if attn_backend in ("sdpa", "torch", "torch_sdpa"):
        mask = (window_id.unsqueeze(1) == window_id.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        q_t = q.permute(0, 2, 1, 3).contiguous()
        k_t = k.permute(0, 2, 1, 3).contiguous()
        v_t = v.permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=mask,
            dropout_p=dropout_p,
            scale=scale,
        )
        return out.permute(0, 2, 1, 3).contiguous()

    window_id = window_id.unsqueeze(0) + (
        torch.arange(bsz, device=device, dtype=window_id.dtype).view(bsz, 1) * num_windows_per_sample
    )
    window_id = window_id.reshape(-1)

    q_flat = q.reshape(bsz * n_tokens, num_heads, head_dim).contiguous()
    k_flat = k.reshape(bsz * n_tokens, num_heads, head_dim).contiguous()
    v_flat = v.reshape(bsz * n_tokens, num_heads, head_dim).contiguous()

    sorted_idx = torch.argsort(window_id)
    window_id_sorted = window_id[sorted_idx]
    q_sorted = q_flat[sorted_idx].contiguous()
    k_sorted = k_flat[sorted_idx].contiguous()
    v_sorted = v_flat[sorted_idx].contiguous()

    _, counts = torch.unique_consecutive(window_id_sorted, return_counts=True)
    if attn_backend == "xformers":
        attn_bias = xops.fmha.BlockDiagonalMask.from_seqlens(counts.tolist())
        out_sorted = xops.memory_efficient_attention(
            q_sorted.unsqueeze(0),
            k_sorted.unsqueeze(0),
            v_sorted.unsqueeze(0),
            attn_bias=attn_bias,
            p=dropout_p,
            scale=scale,
        ).squeeze(0)
    else:
        counts_i32 = counts.to(torch.int32)
        cu_seqlens = torch.empty(counts_i32.numel() + 1, device=device, dtype=torch.int32)
        cu_seqlens[0] = 0
        cu_seqlens[1:] = torch.cumsum(counts_i32, dim=0)
        max_seqlen = int(counts_i32.max().item())
        qkv_sorted = torch.stack((q_sorted, k_sorted, v_sorted), dim=1).contiguous()
        out_sorted = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_sorted,
            cu_seqlens,
            max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=scale,
        )

    out_flat = torch.empty_like(q_flat)
    out_flat[sorted_idx] = out_sorted
    return out_flat.reshape(bsz, n_tokens, num_heads, head_dim)


def spatial_window_cross_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: int,
    w: int,
    window_size: Sequence[int],
    shift_window: Optional[Sequence[int]],
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    *,
    h_idx_q: torch.Tensor,
    w_idx_q: torch.Tensor,
    h_idx_k: torch.Tensor,
    w_idx_k: torch.Tensor,
) -> torch.Tensor:
    _, win_h, win_w, _, shift_h, shift_w = _normalize_window_args(window_size, shift_window)

    num_win_w = (w - 1 + shift_w) // win_w + 1

    q_window = ((h_idx_q + shift_h) // win_h) * num_win_w + ((w_idx_q + shift_w) // win_w)
    k_window = ((h_idx_k + shift_h) // win_h) * num_win_w + ((w_idx_k + shift_w) // win_w)

    mask = (q_window.unsqueeze(1) == k_window.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    q_t = q.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=mask,
        dropout_p=dropout_p,
        scale=scale,
    )
    return out.permute(0, 2, 1, 3).contiguous()


class RotaryEmbedding3D(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_heads = getattr(config, "num_attention_heads", None)
        if num_heads is None:
            num_heads = getattr(config, "num_heads", None)
        if num_heads is None:
            raise ValueError("config must define num_attention_heads or num_heads.")
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads.")
            head_dim = hidden_size // num_heads

        self.head_dim = head_dim
        self.base = getattr(config, "rope_base", 10000.0)
        rope_sections = getattr(config, "rope_sections", None)

        if rope_sections is None:
            per_axis = head_dim // 3
            per_axis -= per_axis % 2
            rope_sections = (per_axis, per_axis, per_axis)
        if len(rope_sections) != 3:
            raise ValueError("rope_sections must have 3 values for t, h, w.")
        if any(x < 0 or x % 2 != 0 for x in rope_sections):
            raise ValueError("rope_sections values must be non-negative and even.")

        self.rope_sections = tuple(int(x) for x in rope_sections)
        self.rotary_dim = sum(self.rope_sections)
        if self.rotary_dim > head_dim:
            raise ValueError("Sum of rope_sections must be <= head_dim.")

        self.register_buffer("inv_freq_t", self._build_inv_freq(self.rope_sections[0]), persistent=False)
        self.register_buffer("inv_freq_h", self._build_inv_freq(self.rope_sections[1]), persistent=False)
        self.register_buffer("inv_freq_w", self._build_inv_freq(self.rope_sections[2]), persistent=False)

        self._cache = {
            "t": {"len": 0, "cos": None, "sin": None, "device": None, "dtype": None},
            "h": {"len": 0, "cos": None, "sin": None, "device": None, "dtype": None},
            "w": {"len": 0, "cos": None, "sin": None, "device": None, "dtype": None},
        }

    def _build_inv_freq(self, dim: int) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(0)
        return 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def _get_cos_sin(
        self, axis: str, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cache = self._cache[axis]
        cos = cache["cos"]
        sin = cache["sin"]
        if (
            cos is not None
            and sin is not None
            and cache["len"] >= seq_len
            and cache["device"] == device
            and cache["dtype"] == dtype
        ):
            return cos[:seq_len], sin[:seq_len]

        inv_freq = getattr(self, f"inv_freq_{axis}")
        if inv_freq.numel() == 0:
            return None, None
        inv_freq = inv_freq.to(device=device, dtype=torch.float32)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        cache["len"] = seq_len
        cache["cos"] = cos
        cache["sin"] = sin
        cache["device"] = device
        cache["dtype"] = dtype
        return cos, sin

    def apply_to(
        self,
        x: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
        t: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        if self.rotary_dim == 0:
            return x

        t_dim, h_dim, w_dim = self.rope_sections
        parts = []
        offset = 0

        if t_dim > 0:
            cos_t, sin_t = self._get_cos_sin("t", t, x.device, x.dtype)
            cos_t = cos_t.index_select(0, t_idx).unsqueeze(0).unsqueeze(2)
            sin_t = sin_t.index_select(0, t_idx).unsqueeze(0).unsqueeze(2)
            x_t = _apply_rope_1d(x[..., offset : offset + t_dim], cos_t, sin_t)
            parts.append(x_t)
            offset += t_dim

        if h_dim > 0:
            cos_h, sin_h = self._get_cos_sin("h", h, x.device, x.dtype)
            cos_h = cos_h.index_select(0, h_idx).unsqueeze(0).unsqueeze(2)
            sin_h = sin_h.index_select(0, h_idx).unsqueeze(0).unsqueeze(2)
            x_h = _apply_rope_1d(x[..., offset : offset + h_dim], cos_h, sin_h)
            parts.append(x_h)
            offset += h_dim

        if w_dim > 0:
            cos_w, sin_w = self._get_cos_sin("w", w, x.device, x.dtype)
            cos_w = cos_w.index_select(0, w_idx).unsqueeze(0).unsqueeze(2)
            sin_w = sin_w.index_select(0, w_idx).unsqueeze(0).unsqueeze(2)
            x_w = _apply_rope_1d(x[..., offset : offset + w_dim], cos_w, sin_w)
            parts.append(x_w)
            offset += w_dim

        if offset < x.shape[-1]:
            parts.append(x[..., offset:])

        return torch.cat(parts, dim=-1).contiguous()

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        t_idx: torch.Tensor,
        h_idx: torch.Tensor,
        w_idx: torch.Tensor,
        t: int,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.apply_to(q, t_idx, h_idx, w_idx, t, h, w),
            self.apply_to(k, t_idx, h_idx, w_idx, t, h, w),
        )


class MultiHeadVideoAttention(nn.Module):
    def __init__(self, config, rope) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, "num_attention_heads", None)
        if self.num_heads is None:
            self.num_heads = getattr(config, "num_heads", None)
        if self.num_heads is None:
            raise ValueError("config must define num_attention_heads or num_heads.")

        self.head_dim = getattr(config, "head_dim", None)
        if self.head_dim is None:
            if self.hidden_size % self.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads.")
            self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim.")

        attn_dropout = getattr(config, "attention_probs_dropout_prob", None)
        if attn_dropout is None:
            attn_dropout = getattr(config, "attention_dropout", None)
        if attn_dropout is None:
            attn_dropout = getattr(config, "attn_dropout", 0.0)

        proj_dropout = getattr(config, "hidden_dropout_prob", None)
        if proj_dropout is None:
            proj_dropout = getattr(config, "proj_dropout", 0.0)

        self.scale = self.head_dim**-0.5
        self.attn_dropout = attn_dropout
        self.attn_backend = getattr(config, "attn_backend", "flash_attn")

        qkv_bias = getattr(config, "qkv_bias", True)
        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.rope = rope
        self._pos_cache = {}

    def _get_pos_indices(
        self, t: int, h: int, w: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (t, h, w, device)
        cached = self._pos_cache.get(key)
        if cached is not None:
            return cached
        pos = _get_3d_pos_indices(t, h, w, device)
        self._pos_cache[key] = pos
        return pos

    def _reshape_qkv(
        self, tensor: torch.Tensor, bsz: int, t: int, h: int, w: int
    ) -> torch.Tensor:
        if tensor.dim() == 5 and tensor.shape[-1] == self.hidden_size:
            return tensor.reshape(bsz, t, h, w, self.num_heads, self.head_dim)
        if tensor.dim() == 6 and tensor.shape[-2] == self.num_heads and tensor.shape[-1] == self.head_dim:
            return tensor
        raise ValueError("q/k/v must be (B, T, H, W, C) or (B, T, H, W, heads, head_dim).")

    def _split_qkv(
        self, qkv: torch.Tensor, bsz: int, t: int, h: int, w: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if qkv.dim() == 5 and qkv.shape[-1] == 3 * self.hidden_size:
            qkv = qkv.reshape(bsz, t, h, w, 3, self.num_heads, self.head_dim)
        elif qkv.dim() == 7 and qkv.shape[-3] == 3 and qkv.shape[-2] == self.num_heads:
            pass
        elif qkv.dim() == 7 and qkv.shape[-2] == 3 and qkv.shape[-3] == self.num_heads:
            qkv = qkv.permute(0, 1, 2, 3, 5, 4, 6)
        else:
            raise ValueError(
                "qkv must be (B, T, H, W, 3*C) or (B, T, H, W, 3, heads, head_dim) or (B, T, H, W, heads, 3, head_dim)."
            )

        q = qkv[..., 0, :, :]
        k = qkv[..., 1, :, :]
        v = qkv[..., 2, :, :]
        return q, k, v

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        *,
        qkv: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        attn_mode: str = "full",
        window_size: Optional[Sequence[int]] = None,
        shift_window: Optional[Sequence[int]] = None,
        attn_backend: Optional[str] = None,
        query_last_only: bool = False,
    ) -> torch.Tensor:
        if query_last_only:
            if x is None:
                raise ValueError("x is required when query_last_only=True")
            bsz, t, h, w, _ = x.shape
            if t <= 0:
                raise ValueError("Temporal length must be > 0.")

            qkv_query = self.qkv(x[:, -1:])
            q_query, _, _ = self._split_qkv(qkv_query, bsz, 1, h, w)
            q = q_query.reshape(bsz, h * w, self.num_heads, self.head_dim).contiguous()

            qkv_kv = self.qkv(x)
            _, k_full, v_full = self._split_qkv(qkv_kv, bsz, t, h, w)
            k = k_full.reshape(bsz, t * h * w, self.num_heads, self.head_dim).contiguous()
            v = v_full.reshape(bsz, t * h * w, self.num_heads, self.head_dim).contiguous()

            _, h_idx_q, w_idx_q = self._get_pos_indices(1, h, w, q.device)
            t_idx_q = torch.full((h * w,), t - 1, device=q.device, dtype=torch.int64)
            t_idx_k, h_idx_k, w_idx_k = self._get_pos_indices(t, h, w, q.device)

            q = self.rope.apply_to(q, t_idx_q, h_idx_q, w_idx_q, t, h, w)
            k = self.rope.apply_to(k, t_idx_k, h_idx_k, w_idx_k, t, h, w)

            dropout_p = self.attn_dropout if self.training else 0.0
            backend = attn_backend or self.attn_backend
            if attn_mode == "full":
                attn_out = full_attention(q, k, v, backend, dropout_p=dropout_p, scale=self.scale)
            elif attn_mode in ("swin", "window"):
                if window_size is None:
                    raise ValueError("window_size is required for window attention.")
                attn_out = spatial_window_cross_attention(
                    q,
                    k,
                    v,
                    h,
                    w,
                    window_size,
                    shift_window,
                    dropout_p=dropout_p,
                    scale=self.scale,
                    h_idx_q=h_idx_q,
                    w_idx_q=w_idx_q,
                    h_idx_k=h_idx_k,
                    w_idx_k=w_idx_k,
                )
            else:
                raise ValueError(f"Unknown attention mode: {attn_mode}")

            out = attn_out.reshape(bsz, 1, h, w, self.hidden_size).contiguous()
            out = self.proj(out)
            return self.proj_dropout(out)

        if x is not None:
            bsz, t, h, w, _ = x.shape
        elif qkv is not None:
            bsz, t, h, w = qkv.shape[:4]
        elif q is not None:
            bsz, t, h, w = q.shape[:4]
        else:
            raise ValueError("Provide x, qkv, or q/k/v.")

        if qkv is not None:
            q, k, v = self._split_qkv(qkv, bsz, t, h, w)
        elif q is not None and k is not None and v is not None:
            q = self._reshape_qkv(q, bsz, t, h, w)
            k = self._reshape_qkv(k, bsz, t, h, w)
            v = self._reshape_qkv(v, bsz, t, h, w)
        else:
            if x is None:
                raise ValueError("x is required when qkv or q/k/v are not provided.")
            qkv = self.qkv(x)
            q, k, v = self._split_qkv(qkv, bsz, t, h, w)

        n_tokens = t * h * w
        q = q.reshape(bsz, n_tokens, self.num_heads, self.head_dim).contiguous()
        k = k.reshape(bsz, n_tokens, self.num_heads, self.head_dim).contiguous()
        v = v.reshape(bsz, n_tokens, self.num_heads, self.head_dim).contiguous()

        t_idx, h_idx, w_idx = self._get_pos_indices(t, h, w, q.device)
        q, k = self.rope.apply(q, k, t_idx, h_idx, w_idx, t, h, w)

        dropout_p = self.attn_dropout if self.training else 0.0
        backend = attn_backend or self.attn_backend
        if attn_mode == "full":
            attn_out = full_attention(q, k, v, backend, dropout_p=dropout_p, scale=self.scale)
        elif attn_mode in ("swin", "window"):
            if window_size is None:
                raise ValueError("window_size is required for window attention.")
            attn_out = window_attention(
                q,
                k,
                v,
                t,
                h,
                w,
                window_size,
                shift_window,
                backend,
                dropout_p=dropout_p,
                scale=self.scale,
                pos_idx=(t_idx, h_idx, w_idx),
            )
        else:
            raise ValueError(f"Unknown attention mode: {attn_mode}")

        out = attn_out.reshape(bsz, t, h, w, self.hidden_size).contiguous()
        out = self.proj(out)
        return self.proj_dropout(out)
