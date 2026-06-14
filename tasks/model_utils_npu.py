"""
NPU (Ascend910B) 版 model_utils，基于 tasks/model_utils.py 裁剪。

仅保留 Qwen2/Qwen2.5 的 hash 方法（HATA），加载 NPU 版模型实现。
其余 tokenizer / chat_template 逻辑与原版一致。
"""

import torch
from transformers import AutoConfig, AutoTokenizer

# 复用原 model_utils 的通用工具，避免重复
from model_utils import (
    qwen2_apply_chat_template,
    comm_generate,  # noqa: F401  供 run_pred_npu 导入转发
)


def get_model_type_arch(model_name_or_path):
    name = model_name_or_path.lower()
    if any(x in name for x in ["qwen2", "qwen2.5"]):
        print("run qwen2 model (NPU)")
        return "qwen2", "qwen2"
    raise ValueError(
        "run_pred_npu 仅支持 Qwen2/Qwen2.5，收到: " + model_name_or_path)


def load_config_and_tokenizer(args, task_config, model_name_or_path):
    dtype = torch.float16
    model_config = AutoConfig.from_pretrained(model_name_or_path,
                                              trust_remote_code=True)
    # NPU 上不使用 flash_attention_2，attention 由自定义实现接管
    model_config._attn_implementation = "eager"
    model_config.torch_dtype = dtype

    generate_kwargs = {}
    method = args.method.lower()

    model_type, model_arch = get_model_type_arch(model_name_or_path)
    assert model_type == "qwen2"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              fast_tokenizer=True,
                                              use_fast=True)
    apply_chat_template = qwen2_apply_chat_template

    if method == "hash":
        generate_config = {
            "max_gpu_cache_memory":
            float(task_config.get('device', 'NPU_MEM')) * 1024 * 1024 * 1024,
            "hash_rbits": int(task_config.get('hata', 'RBIT')),
            "hash_weights_path": task_config.get('hata', 'HASH_WEIGHTS_PATH'),
            "sparse_ratio": float(task_config.get('dataset', 'TOPK_RATIO')),
            "with_bias": False,
            "num_sink": int(task_config.get('hata', 'NUM_SINK')),
            "num_recent": int(task_config.get('hata', 'NUM_RECENT')),
        }
    else:
        raise ValueError(
            f"run_pred_npu 仅支持 method=hash，收到: {method}")

    model_meta = (method, model_arch, generate_config)
    return model_meta, model_config, tokenizer, generate_kwargs, apply_chat_template


def load_model(model_meta, model_config, model_name_or_path):
    method, model_arch, generate_config = model_meta
    assert method == "hash" and model_arch == "qwen2"

    from myTransformer.models.qwen2.modeling_qwen2_hash_npu import (
        CustomQwen2ForCausalLM,
    )
    model = CustomQwen2ForCausalLM.from_pretrained(model_name_or_path,
                                                   config=model_config)

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    dtype = torch.float16
    model = model.to(dtype).eval()

    for key, value in generate_config.items():
        setattr(model.generation_config, key, value)

    return model
