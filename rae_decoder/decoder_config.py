import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence
import torch.nn as nn


@dataclass
class DecoderConfig:
    hidden_size: int = 1280
    num_attention_heads: int = 16
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: int = 3420
    hidden_act: str = "silu"
    qkv_bias: bool = True
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    attn_backend: str = "flash_attn"
    rope_base: float = 10000.0
    rope_sections: Optional[Tuple[int, int, int]] = None
    norm_type: str = "rmsnorm"
    rms_norm_eps: float = 1e-6
    norm_eps: Optional[float] = None
    layer_norm_eps: Optional[float] = None
    window_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = (2, 8, 8)
    shift_size: Optional[Union[int, Tuple[int, int], Tuple[int, int, int]]] = (1, 4, 4)
    block_num: int = 27
    full_attn_index: Optional[Union[int, Sequence[int]]] = (6, 13, 20, 26)

    def __post_init__(self) -> None:
        if self.num_heads is None:
            self.num_heads = self.num_attention_heads

        if self.head_dim is None:
            if self.hidden_size % self.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads.")
            self.head_dim = self.hidden_size // self.num_heads
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_heads * head_dim.")

        if isinstance(self.window_size, int):
            self.window_size = (self.window_size, self.window_size)
        if len(self.window_size) not in (2, 3):
            raise ValueError("window_size must have 2 or 3 values.")
        self.window_size = tuple(int(x) for x in self.window_size)
        if any(w <= 0 for w in self.window_size):
            raise ValueError("window_size values must be positive.")

        if self.shift_size is None:
            self.shift_size = tuple(w // 2 for w in self.window_size)
        elif isinstance(self.shift_size, int):
            self.shift_size = (self.shift_size,) * len(self.window_size)
        if len(self.shift_size) != len(self.window_size):
            raise ValueError("shift_size must match window_size dims.")
        self.shift_size = tuple(int(s) for s in self.shift_size)
        for s, w in zip(self.shift_size, self.window_size):
            if s < 0 or s >= w:
                raise ValueError("shift_size must satisfy 0 <= shift < window_size.")

        if self.block_num <= 0:
            raise ValueError("block_num must be positive.")
        if self.full_attn_index is None:
            self.full_attn_index = ()
        elif isinstance(self.full_attn_index, int):
            self.full_attn_index = (self.full_attn_index,)
        else:
            self.full_attn_index = tuple(int(x) for x in self.full_attn_index)
        for idx in self.full_attn_index:
            if not (0 <= idx < self.block_num):
                raise ValueError("full_attn_index entries must be in [0, block_num).")

        norm_type = str(self.norm_type).lower()
        if norm_type not in ("layernorm", "layer_norm", "ln", "rmsnorm", "rms_norm", "rms"):
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
