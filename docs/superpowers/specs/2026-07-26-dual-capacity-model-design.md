# vLLM Ascend 双口径 HBM 容量模型设计

日期：2026-07-26  
目标版本：vLLM Ascend HBM Planner v0.3  
首个精确适配对象：DeepSeek-V4-Flash W8A8 + MTP、910C/A3、vLLM Ascend 0.23.0rc1

## 1. 背景

现有工具把权重、运行时激活、KV BlockPool、图缓存和运行时开销累加成单个峰值，再据此推荐
`max_num_batched_tokens`（下文记为 Q）和 `max_num_seqs`（下文记为 S）。

实测表明这种口径混合了不同生命周期的内存：

1. `profile_run` 使用 Q 个 dummy token 测量非 KV 峰值；
2. vLLM 根据 profile 结果计算可用于 KV cache 的内存；
3. KV planner 检查是否至少能容纳一个 `max_model_len` 请求；
4. KV BlockPool 分配完成后，再进行 Decode 图捕获；
5. 服务启动成功并不代表真实请求运行时一定安全。

因此工具必须区分：

- **启动极限**：只判断 `vllm serve` 能否完成初始化；
- **运行安全值**：考虑真实请求、并发、上下文分布和安全余量后的推荐值。

## 2. 实测依据

已确认平台输入：

- 逻辑 Die 名义 HBM：64 GiB；
- vLLM 实际识别的设备总 HBM：61.27 GiB；
- 启动时可用 HBM：60.89 GiB；
- `gpu_memory_utilization=0.9`；
- vLLM Ascend 0.23.0rc1；
- DeepSeek-V4-Flash W8A8 + 1 层 MTP；
- `block_size=128`、`max_num_seqs=64`；
- `FULL_DECODE_ONLY`、异步调度、EP、FlashComm1、共享专家 DP。

对于 `max_model_len=32768`、DP8/TP2：

| Q | 结果 | 关键证据 |
|---:|---|---|
| 45056 | 成功 | 权重 27.17 GiB，激活 7.27 GiB，non-Torch 3.08 GiB，Graph 1.97 GiB，实际 KV pool 17.63 GiB |
| 47104 | 失败 | 最低 KV 需求 17.26 GiB，可用 KV 17.20 GiB，缺口 0.06 GiB |

成功点在最低 KV 检查前满足：

\[
61.27 \times 0.9 - 27.17 - 7.27 - 3.08
\approx 17.62\ \text{GiB}
\]

这与实际 KV pool 17.63 GiB 一致。失败发生在最低 KV 容量检查阶段，不是 Graph 捕获阶段。

现有模型在同一成功点存在误差抵消：

- 权重估算约 19.95 GiB，比实测少约 7.22 GiB；
- KV planner 估算约 24.11 GiB，比实测最低需求多约 6.85 GiB；
- 激活估算约 7.20 GiB，与实测 7.27 GiB 接近；
- runtime/non-Torch 估算约 0.28 GiB，比实测少约 2.80 GiB。

## 3. 目标与非目标

### 3.1 目标

1. 同时输出启动极限和运行安全推荐；
2. 对每个结论给出内存分解、限制阶段、余量和可信度；
3. 支持无日志时的解析估算，也支持用启动日志校准；
4. 将模型、平台和 vLLM Ascend 版本差异封装为适配器；
5. 将实测成功/失败边界作为区间验证，而不是伪造精确临界点；
6. 保持现有 JSON 配置兼容，并提供明确的 schema 迁移。

### 3.2 非目标

1. 不使用这 9 组仅启动数据证明真实请求运行安全；
2. 不把全部实测点拟合后再将同一批数据报告为独立验证；
3. 不承诺仅凭名义 HBM、模型名称和并行度精确还原 CANN/HCCL Workspace；
4. 不在本阶段扩展新的模型家族，已有通用模型适配器只做兼容性保护。

## 4. 方案选择

采用**解析结构 + 版本化实测校准**的混合方案。

- 解析模型负责生命周期、并行切分、KV 布局和候选搜索；
- 日志校准负责实际权重、profile 激活、non-Torch、Graph 和可见 HBM；
- 无实测项使用解析值，并增大不确定性；
- 纯数据回归只用于诊断残差，不直接替代组件模型。

## 5. 总体架构

新增四个职责明确的子系统：

```text
输入配置 / 启动日志 / 实测边界
              |
              v
      Capacity & Calibration
              |
       +------+------+
       |             |
       v             v
 Startup Model   Runtime Model
       |             |
       +------+------+
              v
       Search & Validation
              |
              v
 双口径推荐、分解、余量、可信度
```

建议模块：

- `capacity.py`：名义 HBM、可见 HBM、启动空闲 HBM和规划预算；
- `startup.py`：profile、最低 KV、Graph 三阶段启动模型；
- `runtime.py`：真实 workload 下的稳态和瞬态峰值模型；
- `calibration.py`：日志解析、组件覆盖、样本拟合和适用范围；
- `validation.py`：成功/失败区间、交叉验证和误差报告；
- `engine.py`：只负责组合结果，不再内嵌各阶段公式；
- `recommender.py`：分别搜索启动极限和运行安全 frontier。

## 6. 容量口径

平台容量分为三个字段：

```json
{
  "platform": {
    "nominal_hbm_gib_per_die": 64.0,
    "visible_hbm_gib_per_die": 61.27,
    "startup_free_hbm_gib_per_die": 60.89,
    "gpu_memory_utilization": 0.9
  }
}
```

优先级：

1. 启动日志解析的数值；
2. 用户显式填写的 `visible_hbm_gib_per_die` 和 `startup_free_hbm_gib_per_die`；
3. 平台/版本 profile 的默认值；
4. 名义 HBM。

若只能使用名义 HBM，结果必须标记容量未经实测，并降低可信度。

启动 KV 规划预算为：

\[
M_{\text{requested}}
=U\times M_{\text{visible-HBM}}
\]

物理 OOM 检查使用：

\[
M_{\text{startup-free-HBM}}
\]

两者不可混用。

## 7. 启动极限模型

### 7.1 阶段 A：profile 与最低 KV 检查

\[
M_{\text{available-KV}}(Q,S)
=M_{\text{requested}}
-M_{\text{weight}}
-M_{\text{profile-activation}}(Q,S,TP,EP)
-M_{\text{non-Torch}}(Q,S,TP,EP)
\]

必须满足：

\[
M_{\text{required-min-KV}}(L,Q,S,\text{layout})
\le M_{\text{available-KV}}(Q,S)
\]

DeepSeek-V4 的最低 KV 需求仍可能通过 hybrid cache spec 间接受 Q 影响，因此不能简单视为
仅与 L 有关。该计算由版本化 KV adapter 提供，profile dummy run 的 Q 分配与最低 KV planner
是两个不同概念。

失败结果应包含：

- `limiting_stage="minimum_kv_check"`；
- 可用 KV、最低需求和缺口；
- 使用的组件来源：解析、日志或校准。

### 7.2 阶段 B：KV BlockPool 物理分配

检查分组、page padding、block rounding 和 worker 间最小 block 数后形成的实际 pool tensor 是否能
完成分配。该阶段区分：

- planner 最低需求；
- requested KV budget；
- 实际分配的 pool tensor；
- 因异构分组和取整导致的超额分配。

### 7.3 阶段 C：Decode 图捕获

图缓存独立检查：

\[
M_{\text{post-KV-live}}
+M_{\text{decode-graph}}(S,TP,MTP)
\le M_{\text{startup-free-HBM}}
\]

`FULL_DECODE_ONLY` 只建模 Decode capture sizes。MTP 的 `enforce_eager=true` 不等价于整个服务
禁用图模式。

### 7.4 启动输出

```json
{
  "startup_limit": {
    "max_num_batched_tokens": 45056,
    "max_num_seqs": 64,
    "limiting_stage": "minimum_kv_check",
    "requested_memory_gib": 55.14,
    "available_kv_gib": 17.62,
    "required_min_kv_gib": 17.26,
    "headroom_gib": 0.36,
    "confidence": "measured"
  }
}
```

数值仅示意字段结构；最终值由候选搜索和取整规则产生。

## 8. 运行安全模型

运行模型不以“服务能启动”为成功标准。它按 workload 场景计算：

\[
M_{\text{runtime-peak}}
=M_{\text{weight}}
+M_{\text{physical-KV-pool}}
+M_{\text{live-KV/state}}
+M_{\text{prefill/decode-activation}}
+M_{\text{operator-workspace}}
+M_{\text{graph}}
+M_{\text{runtime}}
+M_{\text{fragmentation}}
\]

场景至少包括：

- 新请求大 Prefill；
- 长上下文续 Prefill；
- 高并发 Decode；
- MTP Decode；
- 用户指定的上下文长度分布。

安全推荐必须满足用户配置的最小物理余量和不确定组件 reserve。运行推荐不得高于启动极限。

### 8.1 运行输出

```json
{
  "runtime_safe": {
    "max_num_batched_tokens": 32768,
    "max_num_seqs": 32,
    "estimated_peak_hbm_gib": 56.1,
    "physical_headroom_gib": 4.79,
    "binding_scenario": "late_prefill",
    "confidence": "analytical_with_partial_calibration"
  }
}
```

## 9. 校准模型

### 9.1 日志解析

解析以下稳定字段：

- visible/free HBM；
- desired utilization 和 requested memory；
- weights；
- peak activation；
- non-Torch；
- NPU graph；
- current KV cache；
- minimum required KV 和 available KV；
-失败阶段及 traceback 类型。

原始日志值和解析值都进入输出，避免无声覆盖。

### 9.2 校准作用域

校准键至少包含：

```text
device
vllm_ascend_version
model_profile
quantization
TP / EP / PP / PCP / DCP
graph_mode
relevant additional-config flags
```

校准值不得跨不兼容作用域复用。

### 9.3 缺失数据

- 有日志：使用实测组件；
- 有多个 Q 样本：拟合 profile activation/non-Torch 的分段线性或单调插值；
- 只有一个样本：校准截距，斜率保留解析模型并扩大不确定性；
- 没有日志：使用解析模型和保守上界。

不允许静默使用零值表示未知 Workspace。

## 10. 推荐搜索

搜索过程对每个 `(Q,S)` 产生两个判断：

```text
startup_feasible
runtime_safe
```

约束关系：

```text
runtime_safe => startup_feasible
```

推荐结果包括：

- 启动 frontier；
- 运行安全 frontier；
- 固定 S 时最大 Q；
- 固定 Q 时最大 S；
- 每个失败候选的第一限制阶段；
- 统一配置与分场景配置。

## 11. 实测验证

边界样本格式：

```json
{
  "max_model_len": 32768,
  "dp_size": 8,
  "tp_size": 2,
  "max_num_seqs": 64,
  "max_success_mnbt": 45056,
  "first_fail_mnbt": 47104
}
```

真实临界点位于：

\[
Q^* \in [Q_{\text{success}},Q_{\text{fail}})
\]

验证规则：

1. 预测最大成功 Q 落入该区间则通过；
2. 报告与区间两端的距离，不伪造单点误差；
3. 组件日志样本用于组件校准；
4. 其余上下文或并行组合用于留出验证；
5. 补充 leave-one-context-length-out 结果，防止只记住三个长度。

当前 9 组数据全部固定 `S=64`，因此：

- 可验证 Q 的启动边界；
- 不能实证验证 S 的缩放；
- 不能实证验证真实请求运行安全值。

对未验证维度必须显示 `extrapolated` 或较低可信度。

## 12. 兼容与迁移

现有 schema v2 继续接受：

```json
"hbm_gib_per_die": 64.0
```

加载时迁移为 `nominal_hbm_gib_per_die`。现有 `operation=estimate/recommend` 保留，并增加明确口径：

```json
{
  "operation": "recommend",
  "recommendation": {
    "outputs": ["startup_limit", "runtime_safe"]
  }
}
```

旧配置默认输出双口径，并在文本结果中解释差异。

## 13. 测试策略

测试必须先于生产代码修改：

1. 日志解析单元测试；
2. visible HBM 与 nominal HBM 不混用测试；
3. 45056 成功样本的组件重建测试；
4. 47104 在 minimum KV check 阶段失败测试；
5. Graph 不参与该 KV 可用量计算的测试；
6. 运行安全值不超过启动极限的性质测试；
7. schema v2 兼容测试；
8. 9 组边界区间验证；
9. 未校准配置的置信度降级测试；
10. 现有 DeepSeek、Qwen、manual adapter 回归测试。

## 14. 验收标准

1. 成功日志重建误差：
   - 权重、激活、non-Torch、Graph：日志输入时精确复现；
   - available KV：误差不超过 0.05 GiB；
2. DP8/TP2、32K 的预测边界落入 `[45056,47104)`；
3. 9 组边界分别报告通过/失败，不隐藏未匹配项；
4. 运行安全推荐始终不高于启动极限；
5. 无日志时输出完整解析结果、未知项和可信度；
6. 全部现有测试继续通过；
7. README、配置示例和建模文档同步更新。

## 15. 风险与后续数据

主要风险：

- 不同 TP 下权重副本、共享专家 DP 和通信 buffer 差异较大；
- `HCCL_BUFFSIZE=1024`、FlashComm1 和 CANN 版本会改变 non-Torch；
- Graph capture sizes 可能随 S、TP、MTP 和版本变化；
- 当前没有真实请求压测，运行模型只能是解析加保守校准。

为提高 TP1/TP4 精度，后续优先补充每种 TP 一个成功启动 summary。为验证 S，至少固定
`max_model_len` 和 TP，再测试两个不同 `max_num_seqs`。为验证运行安全值，需要加入实际 Prefill
和 Decode 压测峰值。
