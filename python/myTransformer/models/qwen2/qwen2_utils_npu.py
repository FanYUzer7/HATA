"""
NPU (Ascend910B) 版 Qwen2 工具算子，替换 qwen2_utils.py 中依赖
flashinfer / flash-attn 的部分。

包含：
- CustomQwen2RotaryEmbeddingNPU : 替换 flashinfer.apply_rope / apply_llama31_rope
- CustomQwen2RMSNormNPU         : 替换 flashinfer.norm.rmsnorm
- SiLUAndMulNPU                 : 替换 flashinfer.activation.silu_and_mul

实现策略：优先使用 torch_npu 提供的融合算子，若不可用则降级为纯 torch 等价实现。
RoPE 以 HuggingFace Qwen2 原生 (NeoX / rotate_half, interleave=False) 为基准，
并复刻 flashinfer ragged 布局 (indptr / offsets) 的位置计算语义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False


# ----------------------------------------------------------------------
# RoPE
# ----------------------------------------------------------------------
def _compute_inv_freq(head_dim: int, rope_theta: float,
                      device, dtype=torch.float32) -> torch.Tensor:
    """标准 inv_freq = 1 / theta^(2i/d)，i in [0, d/2)。"""
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (idx / head_dim))
    return inv_freq.to(dtype)


def _apply_llama3_freq_scaling(inv_freq: torch.Tensor,
                               low_freq_factor: float,
                               high_freq_factor: float,
                               old_context_len: int,
                               rope_scale: float) -> torch.Tensor:
    """Llama3.1 频率平滑（与 flashinfer.apply_llama31_rope 等价）。"""
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / inv_freq
    # 高频段保持不变；低频段除以 rope_scale；中间段平滑插值
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen,
                                 inv_freq / rope_scale, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / \
        (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / rope_scale + \
        smooth_factor * inv_freq_llama
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NeoX 风格 rotate_half（与 HF apply_rotary_pos_emb 一致）。"""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _positions_from_ragged(indptr: torch.Tensor, offsets: torch.Tensor,
                           total_tokens: int, device) -> torch.Tensor:
    """根据 flashinfer ragged 布局 (indptr, offsets) 计算每个 token 的绝对位置。

    indptr:  [B+1]，第 i 个请求覆盖 token [indptr[i], indptr[i+1])
    offsets: [B]，第 i 个请求的位置起点
    返回 positions: [total_tokens]
    """
    indptr = indptr.to(torch.long)
    offsets = offsets.to(torch.long)
    positions = torch.zeros(total_tokens, dtype=torch.long, device=device)
    B = offsets.shape[0]
    for b in range(B):
        start = int(indptr[b].item())
        end = int(indptr[b + 1].item())
        n = end - start
        if n <= 0:
            continue
        local = torch.arange(n, device=device, dtype=torch.long)
        positions[start:end] = offsets[b] + local
    return positions


class CustomQwen2RotaryEmbeddingNPU(nn.Module):
    """替换 qwen2_utils.CustomQwen2RotaryEmbedding。

    输入 / 输出与原实现保持一致：
        forward(query_states, key_states, past_key_values)
        query_states: [total_tokens, num_heads, head_dim]
        key_states:   [total_tokens, num_kv_heads, head_dim]
    """

    def __init__(self, config):
        super().__init__()
        assert config is not None

        if "rope_scaling" in config and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        assert self.rope_type in ["default", "llama3", "linear"]

        self.rope_theta = config.rope_theta
        self.rope_scale = 1.0
        self.llama3_kwargs = None

        if self.rope_type == "linear":
            self.rope_scale = config.rope_scaling["factor"]
        elif self.rope_type == "llama3":
            self.rope_scale = config.rope_scaling["factor"]
            self.llama3_kwargs = dict(
                low_freq_factor=config.rope_scaling["low_freq_factor"],
                high_freq_factor=config.rope_scaling["high_freq_factor"],
                old_context_len=config.rope_scaling[
                    "original_max_position_embeddings"],
            )

        self._inv_freq_cache = {}

    def _get_inv_freq(self, head_dim, device):
        key = (head_dim, str(device))
        if key not in self._inv_freq_cache:
            inv_freq = _compute_inv_freq(head_dim, self.rope_theta, device)
            if self.rope_type == "linear":
                # 线性缩放：等价于位置除以 scale，这里改为频率不变、
                # 在下方对 position 做缩放（见 forward）。
                pass
            elif self.rope_type == "llama3":
                inv_freq = _apply_llama3_freq_scaling(
                    inv_freq, rope_scale=self.rope_scale,
                    **self.llama3_kwargs)
            self._inv_freq_cache[key] = inv_freq
        return self._inv_freq_cache[key]

    def forward(self, query_states, key_states, past_key_values):
        device = query_states.device
        head_dim = query_states.shape[-1]
        total_tokens = query_states.shape[0]

        indptr, offsets = past_key_values.get_rope_metadata(device)
        positions = _positions_from_ragged(indptr, offsets,
                                            total_tokens, device).float()

        if self.rope_type == "linear":
            positions = positions / self.rope_scale

        inv_freq = self._get_inv_freq(head_dim, device).float()  # [d/2]

        # freqs[t, j] = position[t] * inv_freq[j]
        freqs = torch.outer(positions, inv_freq)        # [T, d/2]
        # NeoX：cos/sin 在前后半段重复拼接
        emb = torch.cat((freqs, freqs), dim=-1)         # [T, d]
        cos = emb.cos()[:, None, :]                     # [T, 1, d]
        sin = emb.sin()[:, None, :]

        def _apply(x):
            xf = x.float()
            out = xf * cos + _rotate_half(xf) * sin
            return out.to(x.dtype)

        q_rope = _apply(query_states)
        k_rope = _apply(key_states)
        return q_rope, k_rope


# ----------------------------------------------------------------------
# RMSNorm
# ----------------------------------------------------------------------
class CustomQwen2RMSNormNPU(nn.Module):
    """替换 qwen2_utils.CustomQwen2RMSNorm（flashinfer.norm.rmsnorm）。"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if _HAS_TORCH_NPU and hasattr(torch_npu, "npu_rms_norm"):
            # torch_npu 融合 RMSNorm，返回 (output, rstd)
            out = torch_npu.npu_rms_norm(hidden_states, self.weight,
                                         epsilon=self.variance_epsilon)[0]
            return out
        # 纯 torch 降级
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x.to(input_dtype))


# ----------------------------------------------------------------------
# SiLU and Mul
# ----------------------------------------------------------------------
class SiLUAndMulNPU(nn.Module):
    """替换 flashinfer.activation.silu_and_mul。

    输入最后一维为 2 * intermediate_size，前半为 gate，后半为 up，
    输出 silu(gate) * up。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if _HAS_TORCH_NPU and hasattr(torch_npu, "npu_swiglu"):
            return torch_npu.npu_swiglu(x, dim=-1)
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
