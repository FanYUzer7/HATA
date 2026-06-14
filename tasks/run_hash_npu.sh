#!/bin/bash
# HATA Ascend910B (NPU) 精度测试脚本 —— 仅 Qwen2.5 + hash 方法
#
# 说明：
# - 本脚本用于在昇腾 910B 上做精度验证，不考虑性能。
# - 使用 ASCEND_RT_VISIBLE_DEVICES 指定可见 NPU 卡（替代 CUDA_VISIBLE_DEVICES）。
# - 14B 模型 + 长上下文需多卡 PP：设 --pp_num 为单进程使用的卡数（>=2 触发跨卡）。
# - --mp_num 为数据并行的进程数；总卡数需 >= mp_num * pp_num。
#
# 显存参考（hash 方法 KV cache 仍存全量 K/V）：
#   Qwen2.5-7B  + 130K ctx  ≈ 22GB  -> 单卡（pp_num=1）可放下
#   Qwen2.5-14B + 130K ctx  ≈ 52GB  -> 需 pp_num>=2 跨卡

LONGBENCH_PATH=/nfs/shared_LLM_dataset/LongBench
QWEN2_PATH=/nfs/shared_LLM_model/Qwen/Qwen2.5-7B-Instruct-1M/

# 7B 单卡示例
ASCEND_RT_VISIBLE_DEVICES=0 python3 run_pred_npu.py \
    --model_name Qwen2.5-7B-Instruct-1M \
    --model_name_or_path ${QWEN2_PATH} \
    --model_maxlen 130816 \
    --dataset_path ${LONGBENCH_PATH} \
    --config_file ../configs/hash_qwen2.5_npu.ini \
    --dataset_name longbench \
    --output_dir ./preds/npu_test \
    --method hash --write_in_time --mp_num 1 --pp_num 1 --min_seq_len 0 --e

# 14B 多卡 PP 示例（2 卡拆一个模型）：
# QWEN2_14B_PATH=/nfs/shared_LLM_model/Qwen/Qwen2.5-14B-Instruct-1M/
# ASCEND_RT_VISIBLE_DEVICES=0,1 python3 run_pred_npu.py \
#     --model_name Qwen2.5-14B-Instruct-1M \
#     --model_name_or_path ${QWEN2_14B_PATH} \
#     --model_maxlen 130816 \
#     --dataset_path ${LONGBENCH_PATH} \
#     --config_file ../configs/hash_qwen2.5_npu.ini \
#     --dataset_name longbench \
#     --output_dir ./preds/npu_test_14b \
#     --method hash --write_in_time --mp_num 1 --pp_num 2 --min_seq_len 0 --e
