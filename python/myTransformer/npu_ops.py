"""
NPU (Ascend910B) 等价算子库，用于替换原 CUDA 扩展 KVLib。

设计目标：仅用于在 Ascend910B 上做精度验证，不考虑性能，因此全部用
朴素的 torch / torch_npu 算子组合实现，语义与原 KVLib 算子保持等价。

原 KVLib 暴露的算子及其在本文件中的等价实现：
    KVLib.hamming_score        -> hamming_score
    KVLib.batch_topk           -> batch_topk
    KVLib.flash_index_decode   -> flash_index_decode
    KVLib.flash_decode         -> flash_decode
    KVLib.kvcache_append       -> kvcache_append
    KVLib.kvcache_append2      -> kvcache_append2
    KVLib.create_tensor        -> create_tensor

============================================================
关于 hamming_score 的“符号点积简化”等价性说明
============================================================
原实现：把 Q/K 投影后取符号得到 hash code，按 32bit 打包成 int32，
汉明距离 d = popcount(q_code XOR k_code)，topk 取 **最小** 汉明距离。

本实现：hash code 直接以 ±1 (float/int8) 存储，不打包。
设 RBIT 为 hash 位数，q_code、k_code 取值 {+1, -1}，则
    q_code · k_code = (匹配位数) - (不匹配位数) = RBIT - 2 * d
因此 “汉明距离最小” 完全等价于 “符号点积最大”。
所以 score 用点积表示，topk 时取 largest=True，索引集合与原实现一致。
"""

import torch
import torch.nn.functional as F


# 用一个足够大的正数表示“被屏蔽/不参与 topk”的位置。
# 在符号点积语义下分数越大越优先，被屏蔽位置应当得到最小分数，
# 因此屏蔽值取一个很小的负数。
_MASK_SCORE = -1e30


def hamming_score(key_code: torch.Tensor,
                  query_code: torch.Tensor,
                  rbit: int,
                  seq_len: int,
                  sink: int = 0,
                  recent: int = 0) -> torch.Tensor:
    """等价于 KVLib.hamming_score（符号点积简化版）。

    参数（与原 CUDA 接口对齐）：
        key_code:   [B, S_buf, NUM_KV_HEAD, RBIT]，取值 {+1, -1}（float）
                    注意：这里第 4 维是 **展开后的符号位**，而非打包的 int32。
        query_code: [B, 1, NUM_HEAD, RBIT]，取值 {+1, -1}（float）
        rbit:       hash 位数
        seq_len:    有效序列长度（只对前 seq_len 个 token 打分）
        sink:       前 sink 个 token 强制不被选中（置最低分）
        recent:     后 recent 个 token 强制不被选中（置最低分）

    返回：
        score: [B, NUM_KV_HEAD, seq_len]，float。
               分数越大代表与 query 越接近（汉明距离越小）。
    """
    B = key_code.shape[0]
    num_kv_head = key_code.shape[2]
    num_head = query_code.shape[2]
    kv_group = num_head // num_kv_head

    # 只取有效长度
    k = key_code[:, :seq_len, :, :].to(torch.float32)  # [B, S, KVH, R]
    q = query_code.to(torch.float32)                   # [B, 1, H, R]

    # 将 query 的 head 按 GQA group 折叠到对应的 kv head 上：
    # query 排布为 [.., H, R]，H = KVH * kv_group，
    # 同一个 kv head 对应连续的 kv_group 个 query head。
    q = q.view(B, num_kv_head, kv_group, rbit)         # [B, KVH, G, R]

    # k: [B, S, KVH, R] -> [B, KVH, S, R]
    k = k.permute(0, 2, 1, 3).contiguous()             # [B, KVH, S, R]

    # 点积得分：对每个 kv head，q 的 group 内多个 head 与 key 做点积后求和，
    # 与原 CUDA kernel 中对 KVGroup 求和的行为一致。
    # score[b, kvh, s] = sum_g sum_r q[b,kvh,g,r] * k[b,kvh,s,r]
    score = torch.einsum('bhgr,bhsr->bhs', q, k)       # [B, KVH, S]

    # sink / recent 屏蔽
    if sink > 0 or recent > 0:
        idx = torch.arange(seq_len, device=score.device)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=score.device)
        if sink > 0:
            mask |= idx < sink
        if recent > 0:
            mask |= idx >= (seq_len - recent)
        score = score.masked_fill(mask.view(1, 1, seq_len), _MASK_SCORE)

    return score


def batch_topk(data: torch.Tensor, k: int, largest: bool) -> torch.Tensor:
    """等价于 KVLib.batch_topk。

    参数：
        data:    [B, NUM_HEAD, SEQ]
        k:       取 topk 的数量
        largest: 是否取最大的 k 个

    返回：
        indices: [B, NUM_HEAD, k]，int32
    """
    k = min(int(k), data.shape[-1])
    indices = torch.topk(data, k, dim=-1, largest=largest).indices
    return indices.to(torch.int32)


def _gqa_expand(x: torch.Tensor, num_head: int) -> torch.Tensor:
    """把 KV (num_kv_head) 在 head 维广播到 num_head（GQA）。

    x: [B, S, NUM_KV_HEAD, D] -> [B, S, NUM_HEAD, D]
    """
    B, S, num_kv_head, D = x.shape
    if num_kv_head == num_head:
        return x
    group = num_head // num_kv_head
    x = x.unsqueeze(3).expand(B, S, num_kv_head, group, D)
    return x.reshape(B, S, num_head, D)


def flash_decode(q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 softmax_scale: float,
                 seqlen_k: int):
    """等价于 KVLib.flash_decode（普通 decode attention，无 causal mask，
    因为 decode 时 query 在序列末尾，可看到所有已缓存 key）。

    参数：
        q: [B, 1, NUM_HEAD, D]
        k: [B, S_buf, NUM_KV_HEAD, D]
        v: [B, S_buf, NUM_KV_HEAD, D]
        seqlen_k: 有效 key 长度

    返回：
        out: [B, 1, NUM_HEAD, D]
    """
    B, q_len, num_head, D = q.shape
    k = k[:, :seqlen_k, :, :]
    v = v[:, :seqlen_k, :, :]

    k = _gqa_expand(k, num_head)   # [B, Sk, H, D]
    v = _gqa_expand(v, num_head)

    qf = q.to(torch.float32).permute(0, 2, 1, 3)   # [B, H, 1, D]
    kf = k.to(torch.float32).permute(0, 2, 1, 3)   # [B, H, Sk, D]
    vf = v.to(torch.float32).permute(0, 2, 1, 3)   # [B, H, Sk, D]

    attn = torch.matmul(qf, kf.transpose(-1, -2)) * softmax_scale  # [B,H,1,Sk]
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, vf)                   # [B, H, 1, D]

    out = out.permute(0, 2, 1, 3).contiguous().to(q.dtype)  # [B,1,H,D]
    return out


def flash_index_decode(q: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       idx: torch.Tensor,
                       softmax_scale: float):
    """等价于 KVLib.flash_index_decode（top-k 稀疏 decode attention）。

    只在 idx 指定的 key/value 位置上做 attention。

    参数：
        q:   [B, 1, NUM_HEAD, D]
        k:   [B, S_buf, NUM_KV_HEAD, D]
        v:   [B, S_buf, NUM_KV_HEAD, D]
        idx: [B, NUM_KV_HEAD, S_gather]，int，每个 kv head 选中的 token 下标
        softmax_scale: 缩放系数

    返回：
        (out, lse):
            out: [B, 1, NUM_HEAD, D]
            lse: [B, NUM_HEAD, 1]  (log-sum-exp，与原接口返回 softmax_lse 对齐)
    """
    B, q_len, num_head, D = q.shape
    num_kv_head = k.shape[2]
    group = num_head // num_kv_head
    S_gather = idx.shape[2]

    idx = idx.to(torch.long)  # [B, KVH, G]

    # gather key/value：对每个 (b, kv_head) 取 idx 指定的 token
    # k: [B, S, KVH, D] -> [B, KVH, S, D]
    k_t = k.permute(0, 2, 1, 3)   # [B, KVH, S, D]
    v_t = v.permute(0, 2, 1, 3)
    gather_idx = idx.unsqueeze(-1).expand(B, num_kv_head, S_gather, D)
    k_sel = torch.gather(k_t, 2, gather_idx)  # [B, KVH, S_gather, D]
    v_sel = torch.gather(v_t, 2, gather_idx)

    # GQA：把 kv head 广播到 query head
    if group > 1:
        k_sel = k_sel.unsqueeze(2).expand(B, num_kv_head, group, S_gather, D)
        k_sel = k_sel.reshape(B, num_head, S_gather, D)
        v_sel = v_sel.unsqueeze(2).expand(B, num_kv_head, group, S_gather, D)
        v_sel = v_sel.reshape(B, num_head, S_gather, D)

    qf = q.to(torch.float32).permute(0, 2, 1, 3)   # [B, H, 1, D]
    kf = k_sel.to(torch.float32)                   # [B, H, S_gather, D]
    vf = v_sel.to(torch.float32)

    logits = torch.matmul(qf, kf.transpose(-1, -2)) * softmax_scale  # [B,H,1,Sg]

    lse = torch.logsumexp(logits, dim=-1)          # [B, H, 1]
    attn = torch.softmax(logits, dim=-1)
    out = torch.matmul(attn, vf)                   # [B, H, 1, D]

    out = out.permute(0, 2, 1, 3).contiguous().to(q.dtype)  # [B,1,H,D]
    lse = lse.to(torch.float32)                    # [B, H, 1]
    return out, lse


def kvcache_append(kv_cache_tensor: torch.Tensor,
                   key_tensor: torch.Tensor,
                   value_tensor: torch.Tensor,
                   insert_pos: int) -> None:
    """等价于 KVLib.kvcache_append（把一个 decode token 的 K/V 写入 cache）。

    kv_cache_tensor: [2, B, max_seq, NUM_KV_HEAD, D]
    key_tensor:      [B, 1, NUM_KV_HEAD, D]
    value_tensor:    [B, 1, NUM_KV_HEAD, D]
    insert_pos:      写入的序列位置
    """
    kv_cache_tensor[0, :, insert_pos, :, :] = key_tensor[:, 0, :, :]
    kv_cache_tensor[1, :, insert_pos, :, :] = value_tensor[:, 0, :, :]


def kvcache_append2(dst_kv_cache_tensor: torch.Tensor,
                    src_kv_cache_tensor: torch.Tensor,
                    dst_pos: int,
                    src_pos: int) -> None:
    """等价于 KVLib.kvcache_append2（cache 间逐 token 拷贝 K 和 V）。

    两个 cache 形状均为 [2, B, max_seq, NUM_KV_HEAD, D]
    """
    dst_kv_cache_tensor[:, :, dst_pos, :, :] = \
        src_kv_cache_tensor[:, :, src_pos, :, :]


def create_tensor(size, dtype: int) -> torch.Tensor:
    """等价于 KVLib.create_tensor。

    原实现用 cudaMallocHost 分配 pinned host 内存以加速 H2D 拷贝；
    精度测试不关心性能，这里直接分配普通 tensor，返回展平的一维张量
    （与原实现返回 {num_elements} 形状一致）。

    dtype: 16 -> float16, 否则 float32
    """
    num_elements = 1
    for d in size:
        num_elements *= int(d)
    torch_dtype = torch.float16 if dtype == 16 else torch.float32
    return torch.empty(num_elements, dtype=torch_dtype)
