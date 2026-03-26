import torch
import torch.nn as nn
from transformers.activations import get_activation

from .attention import MultiHeadVideoAttention, RotaryEmbedding3D


class RMSNorm(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        eps = getattr(config, "rms_norm_eps", None)
        if eps is None:
            eps = getattr(config, "norm_eps", None)
        if eps is None:
            eps = getattr(config, "layer_norm_eps", 1e-6)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LayerNorm(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        eps = getattr(config, "layer_norm_eps", None)
        if eps is None:
            eps = getattr(config, "norm_eps", None)
        if eps is None:
            eps = getattr(config, "rms_norm_eps", 1e-6)
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

    def extra_repr(self):
        return f"{tuple(self.norm.normalized_shape)}, eps={self.norm.eps}"


def _resolve_norm_cls(config):
    norm_type = getattr(config, "norm_type", None) or "layernorm"
    norm_type = str(norm_type).lower()
    if norm_type in ("layernorm", "layer_norm", "ln"):
        return LayerNorm
    if norm_type in ("rmsnorm", "rms_norm", "rms"):
        return RMSNorm
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class DecoderBlock(nn.Module):
    def __init__(self, config, mode, rope, shift=False) -> None:
        super().__init__()
        self.mode = mode
        self.attn = MultiHeadVideoAttention(config, rope)
        norm_cls = _resolve_norm_cls(config)
        self.norm1 = norm_cls(config)
        self.mlp = MLP(config)
        self.norm2 = norm_cls(config)
        self.window_size = config.window_size
        self.shift = shift
        self.shift_size = config.shift_size

    def _resolve_shift_window(self):
        if not self.shift:
            return None
        if self.shift_size is None:
            return None
        if len(self.shift_size) == 3:
            # Temporal dimension keeps non-shifted behavior.
            return (0, int(self.shift_size[1]), int(self.shift_size[2]))
        if len(self.shift_size) == 2:
            return (int(self.shift_size[0]), int(self.shift_size[1]))
        raise ValueError(f"Unsupported shift_size: {self.shift_size}")

    def _attend(self, x_norm: torch.Tensor, query_last_only: bool) -> torch.Tensor:
        if self.mode == "full":
            return self.attn(x_norm, attn_mode="full", query_last_only=query_last_only)

        if self.mode in ("swin", "window"):
            return self.attn(
                x_norm,
                attn_mode="swin",
                window_size=self.window_size,
                shift_window=self._resolve_shift_window(),
                query_last_only=query_last_only,
            )

        raise ValueError(f"Unsupported attention mode: {self.mode}")

    def forward(self, x: torch.Tensor, query_last_only: bool = False) -> torch.Tensor:
        attn_out = self._attend(self.norm1(x), query_last_only=query_last_only)

        if query_last_only:
            hidden_last = x[:, -1:] + attn_out
            mlp_out = self.mlp(self.norm2(hidden_last))
            updated_last = hidden_last + mlp_out
            if x.shape[1] == 1:
                return updated_last
            return torch.cat([x[:, :-1], updated_last], dim=1)

        hidden_states = x + attn_out
        mlp_out = self.mlp(self.norm2(hidden_states))
        return hidden_states + mlp_out


class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.rope = RotaryEmbedding3D(config)
        self.full_attn_index = config.full_attn_index
        self.block_num = config.block_num
        self.final_norm = _resolve_norm_cls(config)(config)

        decoder_block = []
        for i in range(self.block_num):
            if i in self.full_attn_index:
                decoder_block.append(DecoderBlock(config, "full", self.rope))
            else:
                decoder_block.append(DecoderBlock(config, "swin", self.rope, shift=(i % 2 != 0)))
        self.decoder_block = nn.ModuleList(decoder_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.block_num):
            x = self.decoder_block[i](x, query_last_only=False)
        return self.final_norm(x)

    def forward_last(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected latent shape (B,T,H,W,C), got {tuple(x.shape)}")
        for i in range(self.block_num):
            x = self.decoder_block[i](x, query_last_only=True)
        return self.final_norm(x[:, -1:])
