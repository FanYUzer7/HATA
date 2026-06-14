"""
NPU (Ascend910B) 版 Qwen2 HATA hash 模型，替换 modeling_qwen2_hash.py。

与原实现的差异：
1. flash_attn_with_kvcache -> 本文件内 _flash_attn_prefill / npu_ops.flash_decode
2. KVLib.flash_index_decode -> npu_ops.flash_index_decode
3. 使用 HashStaticCacheNPU 与 qwen2_utils_npu 中的 NPU 算子
4. 修正原文件笔误：self.sacle -> self.scale

仅用于精度验证，不考虑性能。
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
)
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...cache.kvcache_hash_npu import (
    HashStaticCacheNPU,
    prepare_cache_for_generation,
)
from ... import npu_ops
from .qwen2_utils_npu import (
    CustomQwen2RotaryEmbeddingNPU,
    SiLUAndMulNPU,
    CustomQwen2RMSNormNPU,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

logger = logging.get_logger(__name__)


def _flash_attn_prefill(query, key, value, scale, seqlen_k):
    """等价于 prefill 阶段的 flash_attn_with_kvcache(causal=True)。

    query: [B, q_len, NUM_HEAD, D]
    key/value: [B, S_buf, NUM_KV_HEAD, D]（取前 seqlen_k 个有效）
    返回: [B, q_len, NUM_HEAD, D]

    causal 语义：cache 中前 (seqlen_k - q_len) 个 token 是历史（全可见），
    最后 q_len 个 token 与当前 query 一一对应，按下三角因果遮挡。
    """
    B, q_len, num_head, D = query.shape
    k = key[:, :seqlen_k, :, :]
    v = value[:, :seqlen_k, :, :]

    k = npu_ops._gqa_expand(k, num_head)   # [B, Sk, H, D]
    v = npu_ops._gqa_expand(v, num_head)

    qf = query.to(torch.float32).permute(0, 2, 1, 3)  # [B, H, q_len, D]
    kf = k.to(torch.float32).permute(0, 2, 1, 3)      # [B, H, Sk, D]
    vf = v.to(torch.float32).permute(0, 2, 1, 3)

    logits = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # [B,H,q_len,Sk]

    # 构造 causal mask：query 第 i 个（绝对位置 past + i）可见 key [0, past+i]
    past = seqlen_k - q_len
    q_pos = torch.arange(q_len, device=query.device).view(q_len, 1) + past
    k_pos = torch.arange(seqlen_k, device=query.device).view(1, seqlen_k)
    causal = k_pos <= q_pos                              # [q_len, Sk]
    logits = logits.masked_fill(~causal.view(1, 1, q_len, seqlen_k), float('-inf'))

    attn = torch.softmax(logits, dim=-1)
    out = torch.matmul(attn, vf)                        # [B, H, q_len, D]
    out = out.permute(0, 2, 1, 3).contiguous().to(query.dtype)
    return out


class CustomQwen2AttentionNPU(Qwen2FlashAttention2):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rotary_emb = CustomQwen2RotaryEmbeddingNPU(config)
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HashStaticCacheNPU] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size = past_key_value.curr_batch_size
        q_len = past_key_value.get_cur_q_len()
        _, hidden_size = hidden_states.size()

        is_prefill = q_len > 1

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim)
        query_states, key_states = self.rotary_emb(query_states, key_states,
                                                   past_key_value)

        query_states = query_states.view(batch_size, -1, self.num_heads,
                                         self.head_dim)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads,
                                     self.head_dim)
        value_states = value_states.view(batch_size, -1,
                                         self.num_key_value_heads, self.head_dim)

        if is_prefill:
            cached_keys, cache_values, kvcache_len = past_key_value.append_prefill(
                key_states, value_states, self.layer_idx)

            if self.layer_idx >= past_key_value.get_num_skip_layers():
                past_key_value.prefill_encode_hash(self.layer_idx, key_states)

            attn_output = _flash_attn_prefill(
                query_states, cached_keys, cache_values, self.scale, kvcache_len)
        else:
            cached_keys, cache_values, kvcache_len = past_key_value.append_decode(
                key_states, value_states, self.layer_idx)

            if self.layer_idx >= past_key_value.get_num_skip_layers():
                # must after append_decode
                encoded_query = past_key_value.decode_encode_hash(
                    key_states, query_states, self.layer_idx)

                topk_indices = past_key_value.compute_topk(
                    encoded_query, kvcache_len, self.layer_idx)

                attn_output, _ = npu_ops.flash_index_decode(
                    query_states, cached_keys, cache_values, topk_indices,
                    self.scale)
            else:
                attn_output = npu_ops.flash_decode(
                    query_states, cached_keys, cache_values, self.scale,
                    kvcache_len)

        attn_output = attn_output.view(-1, hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class CustomQwen2MLPNPU(Qwen2MLP):

    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = config.torch_dtype
        self.hidden_act = config.hidden_act
        self.converted = False
        assert self.hidden_act in ["silu"]

    def convert_fusion_exec(self):
        if not self.converted:
            device = self.down_proj.weight.device
            self.gate_up_proj = nn.Linear(self.hidden_size,
                                          self.intermediate_size * 2,
                                          bias=False,
                                          dtype=self.torch_dtype,
                                          device=device)
            self.gate_up_proj.weight.data[:self.intermediate_size, :] = \
                self.gate_proj.weight.data
            self.gate_up_proj.weight.data[self.intermediate_size:, :] = \
                self.up_proj.weight.data
            self.act_fn = SiLUAndMulNPU()
            del self.gate_proj
            del self.up_proj
            self.converted = True

    def forward(self, x):
        self.convert_fusion_exec()
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class CustomQwen2DecoderLayerNPU(Qwen2DecoderLayer):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomQwen2AttentionNPU(config, layer_idx)
        self.input_layernorm = CustomQwen2RMSNormNPU(config.hidden_size,
                                                     eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomQwen2RMSNormNPU(
            config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CustomQwen2MLPNPU(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HashStaticCacheNPU] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # NPU 设备设置：跟随当前 hidden_states 所在设备
        if hidden_states.device.type == "npu":
            try:
                import torch_npu  # noqa: F401
                if hidden_states.device.index != torch.npu.current_device():
                    torch.npu.set_device(hidden_states.device)
            except (ImportError, AttributeError):
                pass

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, )
        if output_attentions:
            outputs += (self_attn_weights, )
        if use_cache:
            outputs += (present_key_value, )
        return outputs


class CustomQwen2ModelNPU(Qwen2Model):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            CustomQwen2DecoderLayerNPU(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CustomQwen2RMSNormNPU(config.hidden_size,
                                          eps=config.rms_norm_eps)
        self.rotary_emb = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HashStaticCacheNPU] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        assert inputs_embeds is None, "inputs_embeds is not supported"
        output_attentions = False
        use_cache = True

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the "
                "same time, and must specify either one")

        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError(
                "`use_cache=True` is incompatible with gradient checkpointing.")

        # chunk prefill
        CHUNK_SIZE = 1024
        for chunk_start in range(0, input_ids.shape[1], CHUNK_SIZE):
            chunk_input_ids = input_ids[:, chunk_start:chunk_start + CHUNK_SIZE]

            chunk_inputs_embeds = self.embed_tokens(chunk_input_ids)
            hidden_states = chunk_inputs_embeds
            bsz, q_len, _ = hidden_states.shape

            past_key_values.alloc(q_len)

            all_hidden_states = None
            all_self_attns = None
            next_decoder_cache = None
            kwargs = {}

            hidden_states = hidden_states.view(bsz * q_len, -1)

            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=None,
                    position_embeddings=None,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]
                if use_cache:
                    next_decoder_cache = layer_outputs[
                        2 if output_attentions else 1]

        hidden_states = hidden_states.view(bsz, q_len, -1)[:, -1, :].view(bsz, -1)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bsz, 1, -1)

        next_cache = next_decoder_cache

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomQwen2ForCausalLM(Qwen2ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen2ModelNPU(config)
        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = \
            prepare_cache_for_generation
