"""
NPU (Ascend910B) 版 Hash KV Cache，替换原 kvcache_hash.py。

与原实现的主要差异：
1. hash code 以 ±1 (float16) 存储，宽度为完整 RBIT（不打包成 int32）。
   因此 hash_dim = hash_rbits，hash cache dtype = self.dtype。
2. 所有 KVLib.* 调用替换为 npu_ops.* 等价实现。
3. hash encode 使用 kernels/npu_hash_encode.py 的 torch 版本。
4. compute_topk 中由于 hamming_score 返回的是“符号点积”分数
   （越大越接近），topk 取 largest=True（原实现取最小汉明距离 largest=False）。

基类 CustomStaticCache 复用自 kvcache_fa.py（纯 torch，无 CUDA 调用）。
"""

from typing import Dict, Optional, Union
import os
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig

from ..kernels.npu_hash_encode import (
    prefill_multi_hash_encode,
    decode_multi_hash_encode,
)
from .kvcache_fa import CustomStaticCache
from .. import npu_ops


class HashStaticCacheNPU(CustomStaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        hash_rbits: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_gpu_cache_memory_size: int = 1000000000,
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
        sparse_ratio: float = 0.1,
        num_skip_layers: int = 2,
        hash_weights_path: str = None,
        num_sink: int = 0,
        num_recent: int = 0,
        max_batch_size: int = 16,
    ) -> None:
        super().__init__(config, device, dtype, max_gpu_cache_memory_size,
                         layer_device_map)
        self.hash_rbits = hash_rbits
        self.sparse_ratio = sparse_ratio
        self.hash_weights_path = hash_weights_path
        self.num_skip_layers = num_skip_layers

        self.num_sink = num_sink
        self.num_recent = num_recent
        self.max_batch_size = max_batch_size

        # hash code 以 float16 (self.dtype) 存储，每位占 itemsize 字节
        self.max_gpu_cache_memory_size -= 2 * self.num_layers * self.max_batch_size * (
            self.num_sink + self.num_recent
        ) * self.num_key_value_heads * self.head_dim * self.dtype.itemsize

        self.gqa_size = self.num_heads // self.num_key_value_heads

    def build_cache(self):
        self.layer_caches = []
        self.max_layer_caches = []

        self.layer_hash_caches = []
        self.max_layer_hash_caches = []

        self.hash_weights = []

        assert self.hash_rbits % 32 == 0
        # 与原实现不同：不打包，hash_dim 即完整 RBIT 宽度
        self.hash_dim = self.hash_rbits

        per_token_per_head_kv_size = self.dtype.itemsize * self.head_dim * 2
        # hash code 存 float16，宽度 hash_dim(=rbit)
        per_token_per_head_hash_size = self.dtype.itemsize * self.hash_dim

        all_layer_per_token_per_head_kv_size = per_token_per_head_kv_size * self.num_layers
        all_layer_per_token_per_head_hash_size = per_token_per_head_hash_size * (
            self.num_layers - self.num_skip_layers)
        all_layer_per_token_per_head_size = all_layer_per_token_per_head_kv_size + \
            all_layer_per_token_per_head_hash_size

        self.max_kv_cache_size = self.max_gpu_cache_memory_size * \
            all_layer_per_token_per_head_kv_size / all_layer_per_token_per_head_size
        self.max_hash_cache_size = self.max_gpu_cache_memory_size * \
            all_layer_per_token_per_head_hash_size / all_layer_per_token_per_head_size

        self.each_layer_max_kv_cache = self.max_kv_cache_size / self.num_layers
        self.each_layer_max_hash_cache = self.max_hash_cache_size / (
            self.num_layers - self.num_skip_layers)

        kv_numel = int(self.each_layer_max_kv_cache / self.dtype.itemsize)
        self.each_layer_max_kv_numel = kv_numel
        hash_numel = int(self.each_layer_max_hash_cache / self.dtype.itemsize)
        self.each_layer_max_hash_numel = hash_numel

        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]

            self.layer_caches.append(None)
            self.layer_hash_caches.append(None)

            self.max_layer_caches.append(
                torch.zeros((kv_numel, ),
                            dtype=self.dtype,
                            device=layer_device))

            if l >= self.num_skip_layers:
                self.max_layer_hash_caches.append(
                    torch.zeros((hash_numel, ),
                                dtype=self.dtype,
                                device=layer_device))
                if self.hash_weights_path is None:
                    hash_weight = torch.randn((self.num_key_value_heads,
                                               self.head_dim, self.hash_rbits),
                                              dtype=self.dtype,
                                              device=layer_device)
                else:
                    hash_weight = torch.load(
                        os.path.join(self.hash_weights_path,
                                     f"hash_weight_layer_{l:02d}.pt")).to(
                                         layer_device).to(self.dtype)
                self.hash_weights.append(hash_weight)
            else:
                self.max_layer_hash_caches.append(None)
                self.hash_weights.append(None)

        self.max_seq_len = 0
        self.curr_batch_size = 0
        self.seq_len = 0
        self.layer_cache_lens = [0 for _ in range(self.num_layers)]

        # 符号点积方案不需要 packbit 辅助张量，保留占位以兼容接口
        self.query_code_buffers = None

    def reset(self, batch_size):
        self.curr_batch_size = batch_size
        self.seq_len = 0
        self.layer_cache_lens = [0 for _ in range(self.num_layers)]

        kv_max_seq_len = self.each_layer_max_kv_numel // (
            self.num_key_value_heads * self.head_dim * self.curr_batch_size * 2)
        hash_max_seq_len = self.each_layer_max_hash_numel // (
            self.num_key_value_heads * self.hash_dim * self.curr_batch_size)
        self.max_seq_len = min(kv_max_seq_len, hash_max_seq_len)

        numel = 2 * batch_size * self.max_seq_len * \
            self.num_key_value_heads * self.head_dim
        hash_numel = batch_size * self.max_seq_len * \
            self.num_key_value_heads * self.hash_dim

        for i in range(self.num_layers):
            self.layer_caches[i] = self.max_layer_caches[i][:numel]
            self.layer_caches[i] = self.layer_caches[i].view(
                2, batch_size, self.max_seq_len, self.num_key_value_heads,
                self.head_dim)

            if i >= self.num_skip_layers:
                self.layer_hash_caches[i] = self.max_layer_hash_caches[
                    i][:hash_numel]
                self.layer_hash_caches[i] = self.layer_hash_caches[i].view(
                    batch_size, self.max_seq_len, self.num_key_value_heads,
                    self.hash_dim)

        # query code buffer：±1 float16，宽度 RBIT
        self.query_code_buffers = {}
        for device in self.unique_devices:
            self.query_code_buffers[device] = torch.empty(
                batch_size, 1, self.num_heads, self.hash_dim,
                dtype=self.dtype, device=device)

    def append_prefill(self, key_states, value_states, layer_idx):
        q_len = key_states.shape[1]
        seq_start = self.layer_cache_lens[layer_idx]

        self.layer_caches[layer_idx][0, :,
                                     seq_start:seq_start + q_len, :, :] = key_states
        self.layer_caches[layer_idx][1, :,
                                     seq_start:seq_start + q_len, :, :] = value_states

        self.layer_cache_lens[layer_idx] += q_len
        if layer_idx == self.num_layers - 1:
            self.seq_len += q_len

        key = self.layer_caches[layer_idx][0]
        value = self.layer_caches[layer_idx][1]
        return key, value, self.layer_cache_lens[layer_idx]

    def append_decode(self, key_states, value_states, layer_idx):
        npu_ops.kvcache_append(self.layer_caches[layer_idx], key_states,
                               value_states, self.layer_cache_lens[layer_idx])

        self.layer_cache_lens[layer_idx] += 1
        if layer_idx == self.num_layers - 1:
            self.seq_len += 1

        key = self.layer_caches[layer_idx][0]
        value = self.layer_caches[layer_idx][1]
        return key, value, self.layer_cache_lens[layer_idx]

    def prefill_encode_hash(self, layer_idx, key):
        assert layer_idx >= self.num_skip_layers, \
            f"hash topk is not enabled in layer{layer_idx}!"
        seq_start = self.layer_cache_lens[layer_idx] - key.shape[1]

        prefill_multi_hash_encode(
            key, self.hash_weights[layer_idx],
            self.layer_hash_caches[layer_idx],
            None, seq_start)

    def decode_encode_hash(self, key, query, layer_idx):
        assert layer_idx >= self.num_skip_layers, \
            f"hash topk is not enabled in layer{layer_idx}!"

        seq_start = self.layer_cache_lens[layer_idx] - 1

        decode_multi_hash_encode(
            key,
            self.hash_weights[layer_idx],
            self.layer_hash_caches[layer_idx],
            query,
            self.query_code_buffers[self.layer_devices[layer_idx]],
            None,
            seq_start)

        return self.query_code_buffers[self.layer_devices[layer_idx]]

    def compute_topk(self, encoded_query, seq_len, layer_idx):
        assert layer_idx >= self.num_skip_layers, \
            f"hash topk is not enabled in layer{layer_idx}!"

        score = npu_ops.hamming_score(self.layer_hash_caches[layer_idx],
                                      encoded_query,
                                      self.hash_rbits,
                                      seq_len,
                                      sink=self.num_sink,
                                      recent=self.num_recent)

        if self.sparse_ratio < 1:
            fetch_num = int(seq_len * self.sparse_ratio)
        else:
            fetch_num = min(int(self.sparse_ratio), seq_len)

        # 符号点积分数越大越接近 -> largest=True（原实现汉明距离取最小）
        largest = True
        topk_indices = npu_ops.batch_topk(score, fetch_num, largest)

        return topk_indices

    def get_num_skip_layers(self):
        return self.num_skip_layers


"""
===================================================
Hugging Face api reload
===================================================
"""


def prepare_cache_for_generation(
    self,
    generation_config: GenerationConfig,
    model_kwargs: Dict,
    assistant_model,
    batch_size: int,
    max_cache_length: int,
    device: torch.device,
) -> bool:
    if not hasattr(self, "_cache"):

        def get_layer_device_map(execution_device_map: Optional[dict] = None):
            if execution_device_map is None or len(execution_device_map) <= 1:
                return None
            layer_device_map = {}
            for layer in execution_device_map:
                for idx in range(self.config.num_hidden_layers):
                    if f".{idx}." in f"{layer}.":
                        layer_device_map[idx] = execution_device_map[layer]
                        break
            for idx in range(self.config.num_hidden_layers):
                if idx not in layer_device_map:
                    raise RuntimeError(
                        f"layer {idx} has not been mapped to a device.")
            return layer_device_map

        execution_device_map = None
        if hasattr(self, "hf_device_map"):
            main_device = [
                d for d in self.hf_device_map.values()
                if d not in ["cpu", "disk"]
            ][0]
            execution_device_map = {
                name: main_device if device in ["cpu", "disk"] else device
                for name, device in self.hf_device_map.items()
            }

        layer_device_map = get_layer_device_map(execution_device_map)
        self._cache = HashStaticCacheNPU(
            config=self.config.get_text_config(),
            hash_rbits=generation_config.hash_rbits,
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
            sparse_ratio=generation_config.sparse_ratio,
            hash_weights_path=generation_config.hash_weights_path,
            num_sink=generation_config.num_sink,
            num_recent=generation_config.num_recent,
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
