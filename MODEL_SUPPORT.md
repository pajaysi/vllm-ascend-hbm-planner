# 模型支持范围

“vLLM Ascend 可以运行”与“HBM 模型达到源码级精度”是两件事。本工具按以下
等级描述建模能力。

| 等级 | 模型/输入 | KV 方法 | 说明 |
|---|---|---|---|
| source-exact | DeepSeek-V4-Flash W8A8 + MTP | v0.23 异构 BlockPool | 启动最小 KV、权重放置和 Q 缓冲已建模 |
| built-in-mla | DeepSeek-V3/R1 | latent + RoPE | KV 结构可算，算子 Workspace 需校准 |
| built-in-gqa | Qwen/Llama 代表规格 | 标准 GQA/MHA | 标准 Decoder KV 可确定计算 |
| built-in-gqa-moe | Qwen3 MoE | GQA + EP 权重近似 | 推荐解析实际 Safetensors |
| hf-config-auto | 其他 Decoder | 从 config.json 判断 | 依赖字段完整性 |
| manual | 混合 Attention/SSM | 手工每 token/总 KV | 可信度由输入决定 |

## 内置 Profile

- `deepseek-v4-flash`
- `deepseek-v3`
- `deepseek-r1`
- `qwen3-8b`
- `qwen3-32b`
- `qwen3-30b-a3b`
- `qwen2.5-7b`
- `qwen2.5-72b`
- `llama-3.1-8b`
- `llama-3.1-70b`

运行 `python vllm_ascend_hbm_calculator.py --list-models` 查看完整列表。

## 自动接入

标准 Decoder 至少需要：

- `num_hidden_layers`
- `hidden_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `vocab_size`
- 总参数量或本地 Safetensors

MLA 还需要 `kv_lora_rank` 与 `qk_rope_head_dim`。Qwen3-Next、Mamba、RWKV
等混合 Cache 模型在新增专用 Adapter 前应使用 manual KV。

多模态模型可以解析语言主干，但视觉/音频 encoder 的激活和 Workspace 必须
实测或手工输入。Embedding/Reranker 可设置 `kv_cache_strategy=none`。

官方运行兼容范围会持续变化，请以 vLLM Ascend Supported Models 文档为准；
本文件只描述本工具的 HBM 建模精度。
