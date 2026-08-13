# 从原单文件版本迁移

## 保持兼容的内容

- 原 `schema_version: 1` 会在内存中迁移为 v2；
- `operation=estimate/recommend`、硬件、并行、Scheduler、工作负载、校准、
  不确定性和推荐字段保持原含义；
- `deepseek-v4-flash` 的 v0.20 专用 KV 逻辑和推荐回归结果保持一致；
- 旧入口文件名 `dsv4_total_hbm_calculator.py` 仍可执行。

## 新增字段

- `model.config_path`
- `model.family` / `architecture`
- `model.num_key_value_heads` / `head_dim`
- `model.kv_lora_rank` / `qk_rope_head_dim`
- `model.kv_cache_strategy`
- `kv_cache.*`

## 推荐迁移步骤

1. 把旧 JSON 复制到 `configs/`；
2. 将 `schema_version` 改成 2；
3. 保留 `model.profile=deepseek-v4-flash`，先做回归；
4. 对其他模型改为内置 Profile 或 `profile=auto + config_path`；
5. 添加实际模型目录 `model_path`，让权重从 safetensors 解析；
6. 用目标节点日志填写 `profile_calibration`；
7. 在推荐点和相邻 Frontier 点完成 OOM、TTFT、TPOT 和吞吐压测。
