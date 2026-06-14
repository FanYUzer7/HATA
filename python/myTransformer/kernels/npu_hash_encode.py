"""
NPU (Ascend910B) 版 hash encode，替换原 Triton kernel
（python/myTransformer/kernels/triton_hash_encode.py）。

仅用于精度验证，不考虑性能，用朴素 torch 算子实现。

============================================================
与原实现的存储差异
============================================================
原 Triton 实现：把投影结果取符号后按 32bit 打包成 int32 存储，
hash code 形状 [B, S, NUM_HEAD, RBIT//32]，dtype int32。

本实现：直接以 ±1 存储符号位，不打包，
hash code 形状 [B, S, NUM_HEAD, RBIT]，dtype float16。
这样配合 npu_ops.hamming_score 的“符号点积”即可还原汉明距离序，
精度与原实现等价（见 npu_ops.py 顶部说明）。

hash_weights 形状：[NUM_KV_HEAD, HEAD_DIM, RBIT]
（与原 build_cache 中 torch.randn((num_kv_head, head_dim, rbit)) 一致）。
"""

import torch


def _sign_pm1(x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """投影值取符号，得到 ±1。

    与原实现 `acc = (data @ weight) > 0` 对齐：>0 记为 1（这里 +1），
    否则记为 0（这里 -1）。返回 out_dtype。
    """
    return torch.where(x > 0,
                       torch.tensor(1.0, dtype=torch.float32, device=x.device),
                       torch.tensor(-1.0, dtype=torch.float32,
                                    device=x.device)).to(out_dtype)


def prefill_multi_hash_encode(data: torch.Tensor,
                              hash_weights: torch.Tensor,
                              data_code_output: torch.Tensor,
                              packbit_aux_tensor: torch.Tensor,
                              seq_start: int) -> None:
    """等价于原 prefill_multi_hash_encode（编码一段 prefill 的 K）。

    参数：
        data:             [B, SEQ, NUM_KV_HEAD, HEAD_DIM]
        hash_weights:     [NUM_KV_HEAD, HEAD_DIM, RBIT]
        data_code_output: [B, S_buf, NUM_KV_HEAD, RBIT]，±1，原地写入
        packbit_aux_tensor: 未使用（保留参数以对齐接口）
        seq_start:        写入起始位置
    """
    B, SEQ, NUM_KV_HEAD, HEAD_DIM = data.shape
    RBIT = hash_weights.shape[2]
    out_dtype = data_code_output.dtype

    # proj[b, s, h, r] = sum_d data[b,s,h,d] * W[h,d,r]
    proj = torch.einsum('bshd,hdr->bshr',
                        data.to(torch.float32),
                        hash_weights.to(torch.float32))   # [B, SEQ, KVH, RBIT]
    code = _sign_pm1(proj, out_dtype)

    data_code_output[:, seq_start:seq_start + SEQ, :, :] = code


def decode_multi_hash_encode(key_data: torch.Tensor,
                             hash_weights: torch.Tensor,
                             key_code_output: torch.Tensor,
                             query_data: torch.Tensor,
                             query_code_output: torch.Tensor,
                             packbit_aux_tensor: torch.Tensor,
                             cur_seq: int) -> None:
    """等价于原 decode_multi_hash_encode（同时编码当前 1 个 K 和 Q）。

    参数：
        key_data:          [B, 1, NUM_KV_HEAD, HEAD_DIM]
        hash_weights:      [NUM_KV_HEAD, HEAD_DIM, RBIT]
        key_code_output:   [B, S_buf, NUM_KV_HEAD, RBIT]，±1，原地写入
        query_data:        [B, 1, NUM_HEAD, HEAD_DIM]
        query_code_output: [B, 1, NUM_HEAD, RBIT]，±1，原地写入
        packbit_aux_tensor: 未使用（保留接口）
        cur_seq:           当前 key 写入位置
    """
    B, _, NUM_KV_HEAD, HEAD_DIM = key_data.shape
    NUM_HEAD = query_data.shape[2]
    RBIT = hash_weights.shape[2]
    group = NUM_HEAD // NUM_KV_HEAD

    w = hash_weights.to(torch.float32)   # [KVH, HEAD_DIM, RBIT]

    # ---- 编码 key ----
    k = key_data[:, 0, :, :].to(torch.float32)            # [B, KVH, HEAD_DIM]
    k_proj = torch.einsum('bhd,hdr->bhr', k, w)           # [B, KVH, RBIT]
    k_code = _sign_pm1(k_proj, key_code_output.dtype)     # [B, KVH, RBIT]
    key_code_output[:, cur_seq, :, :] = k_code

    # ---- 编码 query ----
    # query head 按 GQA group 折叠到对应 kv head：head h 用 kv head h//group 的权重
    q = query_data[:, 0, :, :].to(torch.float32)          # [B, NUM_HEAD, HEAD_DIM]
    q = q.view(B, NUM_KV_HEAD, group, HEAD_DIM)           # [B, KVH, G, HEAD_DIM]
    q_proj = torch.einsum('bhgd,hdr->bhgr', q, w)         # [B, KVH, G, RBIT]
    q_code = _sign_pm1(q_proj, query_code_output.dtype)
    query_code_output[:, 0, :, :] = q_code.reshape(B, NUM_HEAD, RBIT)
