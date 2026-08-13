# vLLM Ascend HBM Planner

面向 vLLM Ascend 部署的、可解释的单 Rank/单逻辑 NPU HBM 容量规划器。
工具根据硬件、模型、并行策略和 vLLM Ascend 参数，给出两套结果：

1. **启动极限**：按 `vllm serve` 初始化阶段的最小 KV 准入检查，判断服务能否启动。
2. **运行安全推荐**：在启动通过的基础上，再计入运行期 KV、激活、
   Workspace、图缓存、运行时、碎片和安全余量。

当前对 **DeepSeek-V4-Flash W8A8 + MTP / vllm-ascend 0.23.0rc1 /
910C** 提供源码级理论模型；其他模型使用通用 GQA、MLA、Safetensors
解析或手工 KV Adapter。

运行环境：Python 3.9 或更高版本；代码未使用 `zip(strict=...)`。

## 快速开始

Windows PowerShell：

```powershell
python vllm_ascend_hbm_calculator.py `
  --config configs\deepseek_v4_flash_910c.json
```

固定参数估算：

```powershell
python vllm_ascend_hbm_calculator.py `
  --config configs\deepseek_v4_flash_910c.json `
  --operation estimate
```

查看支持的模型：

```powershell
python vllm_ascend_hbm_calculator.py --list-models
```

## 复现九组启动边界验证

纯理论口径只使用源码可确定项，不隐藏经验常数：

```powershell
python vllm_ascend_hbm_calculator.py `
  --config configs\deepseek_v4_flash_910c.json `
  --validate-boundaries configs\dsv4_v023_startup_boundaries.json
```

加入由 TP2 成功日志拆出的 `1.08 GiB` CANN/ACL 固定基线：

```powershell
python vllm_ascend_hbm_calculator.py `
  --config configs\deepseek_v4_flash_910c_v023_calibrated.json `
  --validate-boundaries configs\dsv4_v023_startup_boundaries.json
```

支持 `--format text|json|csv`。验证输出同时给出：

- 实测成功/失败区间；
- 理论临界 Q；
- 是否落入实测区间；
- 由实测区间反推的“尚未建模内存区间”。

完整结果见 [验证报告](docs/DSV4_V023_VALIDATION.md)，公式和代码映射见
[建模说明](docs/THEORY_MODELING.md)。

三种拓扑均已有已校准配置：

```powershell
python vllm_ascend_hbm_calculator.py `
  --config configs\deepseek_v4_flash_910c_v023_tp2_profiled.json
```

- DP16/TP1：`deepseek_v4_flash_910c_v023_tp1_profiled.json`
- DP8/TP2：`deepseek_v4_flash_910c_v023_tp2_profiled.json`
- DP4/TP4：`deepseek_v4_flash_910c_v023_tp4_profiled.json`

联合验证九组边界：

```powershell
python vllm_ascend_hbm_calculator.py `
  --config configs\deepseek_v4_flash_910c_v023_topology_profiled.json `
  --validate-boundaries configs\dsv4_v023_startup_boundaries.json
```

结果为严格命中 8/9；唯一未命中点与失败边界相差 926 token，小于一次测试步长。

## JSON 的主要输入

```json
{
  "platform": {
    "device": "910c",
    "vllm_ascend_version": "0.23.0rc1",
    "hbm_gib_per_die": 64.0,
    "visible_hbm_gib_per_die": 61.27,
    "startup_free_hbm_gib_per_die": 60.89,
    "gpu_memory_utilization": 0.9
  },
  "vllm_ascend": {
    "enable_shared_expert_dp": true,
    "enable_flashcomm1": true,
    "hccl_buffsize_mib": 1024,
    "hccl_communication_domains_per_rank": 1
  },
  "model": {
    "profile": "deepseek-v4-flash"
  },
  "scheduler": {
    "block_size": 128,
    "max_model_len": 32768,
    "max_num_batched_tokens": 45056,
    "max_num_seqs": 64
  },
  "parallelism": {
    "dp_size": 8,
    "tp_size": 2,
    "pp_size": 1,
    "ep_size": 16,
    "pcp_size": 1,
    "dcp_size": 1
  }
}
```

`hbm_gib_per_die=64` 是标称容量；容量检查优先使用 NPU 实际可见的
`visible_hbm_gib_per_die=61.27`。因此本例的 vLLM 目标预算是：

```text
61.27 × 0.9 = 55.143 GiB/rank
```

## 校准方式

理论模型始终保留。实测值通过 `profile_calibration` 显式覆盖，并输出理论残差：

```json
{
  "profile_calibration": {
    "profiled_max_num_batched_tokens": 45056,
    "weight_gib_per_rank": 27.17,
    "peak_activation_gib_per_rank": 7.27,
    "non_torch_gib_per_rank": 3.08,
    "graph_gib_per_rank": 1.97
  }
}
```

也可以直接传启动日志：

```powershell
python vllm_ascend_hbm_calculator.py `
  --config configs\deepseek_v4_flash_910c.json `
  --profile-log D:\logs\run.log
```

已提供参考 Q 时，候选 Q 的权重常驻缓冲按理论增量调整，激活按 Q 线性缩放。

## 代码结构

```text
src/vllm_ascend_hbm/
├── capacity.py                  # 标称/可见/启动空闲 HBM 口径
├── config.py                    # JSON Schema、加载和校验
├── engine.py                    # 运行期总 HBM 汇总
├── startup.py                   # vllm serve 启动生命周期
├── recommender.py               # Q / max_num_seqs 候选搜索
├── validation.py                # 实测边界验证
├── weight_models/
│   ├── deepseek_v4_w8a8.py      # DSV4 W8A8 张量放置模型
│   └── persistent_buffers.py    # MTP、Top-k、RoPE 常驻缓冲
└── kv/
    ├── deepseek_v4_v023.py      # v0.23 最小 KV 准入公式
    ├── deepseek_v4_flash.py     # 异构物理 BlockPool
    └── homogeneous.py           # 通用 GQA/MLA/manual KV
```

## 测试

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m compileall -q src
```

## 项目资料

- [理论建模说明](docs/THEORY_MODELING.md)
- [DeepSeek-V4 v0.23 验证报告](docs/DSV4_V023_VALIDATION.md)
- [上游源码版本引用](docs/UPSTREAM_SOURCES.md)
- [HBM 建模汇报 PPT](docs/assets/vllm_ascend_hbm_model_report.pptx)
- [HBM 技术总览图](docs/assets/vllm_ascend_hbm_technology_overview.png)

仓库不保存历史 ZIP、wheel 或完整的 vLLM/vLLM Ascend 源码快照。发布包应由
Git tag 对应的提交重新构建，上游源码使用文档中记录的 release 和 commit 定位。

该工具是容量规划器，不替代目标节点压测。模型权重和 KV 可以做到确定性
或接近确定性计算；CANN/ACL 基线、通信域、算子 Workspace 和图捕获仍可能
随版本、拓扑和算子选择改变，应使用邻近边界的启动日志校准。
