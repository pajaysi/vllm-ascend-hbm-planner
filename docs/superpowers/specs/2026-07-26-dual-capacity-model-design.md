# vLLM Ascend 双口径 HBM 容量模型设计（理论优先版）

日期：2026-07-26  
目标版本：vLLM Ascend HBM Planner v0.3  
首个精确适配对象：DeepSeek-V4-Flash W8A8 + MTP、910C/A3、vLLM Ascend 0.23.0rc1

## 1. 设计原则

工具采用“理论模型为主、实测日志为验证和残差诊断”的方法，不用经验拟合替代理论公式。

优先级如下：

1. 直接复现指定 vLLM/vLLM Ascend 版本的源码计算；
2. 根据 checkpoint 张量、量化格式和模块并行规则计算；
3. 根据模型结构参数进行解析计算；
4. 对 CANN/HCCL、算子 workspace、图缓存等难以完全静态确定的部分使用版本化实测值；
5. 日志校准只能覆盖明确的未知项，不能覆盖已经可以理论计算的组件。

每项输出必须标明来源：

- `source_exact`：逐行等价于指定版本源码；
- `tensor_exact`：来自 checkpoint header、dtype、shape 和确定的切分规则；
- `architecture_analytical`：来自模型结构公式；
- `measured`：来自启动日志或专项测量；
- `residual`：理论值与实测值的未解释差额。

## 2. 双口径输出

工具同时输出两个推荐结果：

### 2.1 启动极限

回答：

> 给定配置下，`vllm serve` 能否完成 profile、KV 规划和图捕获？

该口径用于复现实测表中的 `max_success_mnbt` 和 `first_fail_mnbt`。

### 2.2 运行安全值

回答：

> 服务启动后，在指定请求长度、并发和 Prefill/Decode 场景下是否仍有足够物理 HBM？

运行安全值不得高于启动极限。当前九组数据只验证启动极限，不能单独证明运行安全值。

## 3. 已确认的实测依据

平台与配置：

- 逻辑 Die 名义 HBM：64 GiB；
- vLLM 可见总 HBM：61.27 GiB；
- Worker 启动时空闲 HBM：60.89 GiB；
- `gpu_memory_utilization=0.9`；
- vLLM Ascend 0.23.0rc1；
- DeepSeek-V4-Flash W8A8 + 1 层 MTP；
- `block_size=128`、`max_num_seqs=64`；
- EP、FlashComm1、shared expert DP、异步调度；
- `FULL_DECODE_ONLY`。

DP8/TP2、`max_model_len=32768` 的日志：

| Q | 结果 | 关键证据 |
|---:|---|---|
| 45056 | 成功 | 权重 27.17 GiB，激活 7.27 GiB，non-Torch 3.08 GiB，Graph 1.97 GiB，KV pool 17.63 GiB |
| 47104 | 失败 | 最低 KV 需求 17.26 GiB，可用 KV 17.20 GiB，估算最大长度 32648 |

可用 KV 的日志口径为：

\[
M_{\text{available-KV}}
=M_{\text{visible-HBM}}\times U
-M_{\text{model-load}}
-M_{\text{profile-activation}}
-M_{\text{non-Torch}}
\]

代入成功日志：

\[
61.27\times0.9-27.17-7.27-3.08
\approx17.62\ \text{GiB}
\]

与实际 KV pool 17.63 GiB 一致。Graph 在该次 KV 预算之后分配，不参与这条最低 KV 检查公式。

## 4. KV Planner：必须源码等价，不做经验校准

### 4.1 v0.23.0rc1 的页面布局

当 `block_size=128`：

| 缓存类型 | 内部 block | page bytes |
|---|---:|---:|
| C4/C128 MLA、SWA、C128 state、MTP | 128/32 | 131072 |
| C4 indexer、C4 index state | 128/8 | 16640 |

标准 layer tuple：

\[
S_{\text{tuple}}=131072+16640=147712\ \text{bytes}
\]

DeepSeek-V4-Flash 的**最低 KV 准入检查**使用 22 个 layer tuple。物理
KV pool 的创建路径会把 MTP 作为独立的大页面 tensor，并可能使用 23 个
allocation tuple。两条源码路径的 tuple 口径不同，不能共用一个常量。

### 4.2 最低 KV 需求公式

令：

- \(L\)：`max_model_len`；
- \(Q\)：`max_num_batched_tokens`；
- \(B=128\)；
- SWA window \(W=128\)。

各 group 的最大 page 数：

\[
P_{\text{C4-history}}
=\left\lceil\frac{L}{4B}\right\rceil
\]

\[
P_{\text{C128-history}}
=\left\lceil\frac{L}{128B}\right\rceil
\]

\[
P_{\text{SWA-one-group}}
=\left\lceil
\frac{\min(W-1+Q,L)}{B}
\right\rceil+1
\]

DSV4 有两个 SWA group：

\[
P_{\text{SWA-total}}=2P_{\text{SWA-one-group}}
\]

\[
P_{\text{C4-state}}
=\left\lceil
\frac{\min(7+Q,L)}{8}
\right\rceil+1
\]

\[
P_{\text{C128-state}}
=\left\lceil
\frac{\min(127+Q,L)}{32}
\right\rceil+1
\]

C4 主 compressor state 与 C4 indexer compressor state 属于同一 block-size group，组容量取两者最大值，不相加。

最低 KV 需求：

\[
M_{\text{required-min-KV}}
=22\times147712\times
\left(
P_{\text{C4-history}}
+P_{\text{C128-history}}
+P_{\text{SWA-total}}
+P_{\text{C4-state}}
+P_{\text{C128-state}}
\right)
\]

### 4.3 对失败日志的精确复现

当 \(L=32768\)，且 \(Q\ge L\)：

```text
C4 history        64 pages
C128 history       2 pages
SWA              514 pages
C4 state        4097 pages
C128 state      1025 pages
合计            5702 pages
```

\[
5702\times22\times147712
=17.2570199966\ \text{GiB}
\]

日志显示 17.26 GiB，理论误差小于 0.01 GiB。因此这部分不需要经验拟合。

### 4.4 现有代码的三项根因

现有代码得到约 24.11 GiB，主要因为：

1. C4/C128 state 按 Q 直接计算，没有使用 `min(state_window + Q, max_model_len)` 截断；
2. 用物理 pool 分配路径的 23 个 allocation tuple 代替最低准入检查的
   22 个 tuple；
3. 模型 profile 中的 sliding window 写成 4096，官方配置和源码实际为 128。

其中第 1 项在 `Q > max_model_len` 时造成最大误差。修复后应以源码公式为唯一事实来源。

### 4.5 需要拆开的三个 KV 数值

输出不得再混用：

- `required_min_kv_bytes`：启动时至少容纳一个最大长度请求的源码检查值；
- `allocated_kv_pool_bytes`：available memory 经 block/page 取整后实际创建的池；
- `runtime_live_kv_bytes`：真实请求和并发场景使用的逻辑 live KV/state。

## 5. 权重：从“总参数除并行度”改为逐张量理论模型

### 5.1 现有公式的问题

现有解析公式：

```text
routed experts / EP
其余全部 / TP
所有权重统一按 8 bit
最后乘固定 3% overhead
```

对本场景得到约 19.95 GiB，日志为 27.17 GiB。差距不是单纯的“权重 overhead”，而是建模分类错误：

1. W8A8 checkpoint 中仍有大量 BF16/FP32 张量；
2. `ReplicatedLinear` 不除 TP；
3. Column/Row Parallel 张量才按 TP 切分；
4. routed experts 按实际 EP group 和 local expert mapping 切分；
5. `enable_shared_expert_dp=true` 会改变 shared expert 的权重放置；
6. W8A8 dynamic 额外保存 weight scale、offset 和部分 FP32 scale 副本；
7. `model_memory_usage` 包含模型构造期间创建的持久 buffer，不只是 checkpoint payload；
8. MTP 有独立权重、embedding/head 和持久状态。

### 5.2 模型元数据事实

公开 W8A8 MTP checkpoint 的 index 声明：

```text
total_size = 300002377534 bytes = 279.399 GiB
70 个 safetensors shard
103176 个 tensor 名称
```

量化描述明确区分：

- routed/shared expert 的 W1/W2/W3：`W8A8_DYNAMIC`；
- WQ_A/WQ_B/WKV：`W8A8_DYNAMIC`；
- WO_A/WO_B：FLOAT；
- compressor WKV/WGate：FLOAT；
- indexer 的部分投影：W8A8，部分为 FLOAT；
- router、norm、mHC、embedding、head 和 MTP 中存在 BF16/FP32 张量。

因此，`total_parameters × 1 byte` 不是该模型的权重 HBM。

### 5.3 新的逐张量计算路径

优先读取：

```text
config.json
quant_model_description.json
*.safetensors header
model.safetensors.index.json 或 quant_model_weights.safetensors.index.json
```

对每个目标 rank，执行：

```text
checkpoint tensor
  -> vLLM 参数名映射
  -> 模块类型
  -> checkpoint dtype/shape
  -> load 后 dtype/shape
  -> TP/EP/PP/local-expert placement
  -> post-load transform/duplicate
  -> 本 rank 持久字节
```

必须显式支持：

- `ReplicatedLinear`；
- `ColumnParallelLinear`；
- `RowParallelLinear`；
- `VocabParallelEmbedding`；
- `ParallelLMHead`；
- `FusedMoE` local experts；
- shared expert DP；
- MTP draft layer；
- W8A8 dynamic scale/offset；
- `weight_scale_fp32` 等 post-load 副本；
- PP 首尾 stage 的 embedding/head 放置。

### 5.4 模型加载内存分解

日志中的 `weights` 改名解释为 `model_load_persistent`，并拆成：

\[
M_{\text{model-load}}
=M_{\text{loaded-tensors}}
+M_{\text{post-load-duplicates}}
+M_{\text{model-owned-buffers}}(Q,S)
+M_{\text{allocator-residual}}
\]

DeepSeek-V4-Flash MTP 中可理论计算的 Q 相关持久 buffer 至少包括：

\[
M_{\text{mtp-hidden}}
=Q\times(hc\_mult\times hidden\_size)\times dtype\_bytes
\]

当 \(Q=45056\)、`hc_mult=4`、`hidden_size=4096`、BF16：

\[
M_{\text{mtp-hidden}}=1.375\ \text{GiB}
\]

target 和 MTP 各有一个 indexer top-k buffer：

\[
M_{\text{topk-buffers}}
=2\times Q\times index\_topk\times4
\]

当 `index_topk=512`：

\[
M_{\text{topk-buffers}}\approx0.172\ \text{GiB}
\]

这些内存在 vLLM 的 MemoryProfiler 中计入 model load，不能放进普通 activation。

### 5.5 当前理论复核进展

仅用公开配置、量化描述和 v0.23.0rc1 模块定义，已经可将 DP8/TP2、Q=45056 的模型加载内存从旧公式约 19.95 GiB 修正到约 25.9 GiB。

与日志 27.17 GiB 的剩余差额约 1.25 GiB，当前标记为待解释残差，不能直接设为校准常数。实现阶段继续通过：

- safetensors header 的精确 dtype/shape；
- post-load 参数和 buffer 清单；
- ACL format padding；
- shared expert overlap 的持久 buffer；
- allocator reserved/allocated 差异；

逐项闭合。只有确实无法静态确定的剩余项才进入版本化 reserve。

## 6. 激活、non-Torch 和 Graph

### 6.1 Profile activation

激活继续采用结构化算子峰值模型。当前成功点理论约 7.20 GiB，日志 7.27 GiB，说明主项方向正确。

实现时仍需：

- 按 profile dummy-run 的 token 分配规则生成 shape；
- 区分 target 与 MTP；
- 区分相加的持久 buffer 和时间复用的临时 tensor；
- 使用峰值 live-set，而不是把所有层的 activation 简单求和。

### 6.2 non-Torch

non-Torch 3.08 GiB 不能默认为零，也不能全部回归到 Q：

- HCCL buffer；
- CANN/ACL context；
- 算子 workspace；
- 通信域和 stream；
- `HCCL_BUFFSIZE=1024`；
- FlashComm1 和 multistream。

其中能由配置确定的 buffer 先解析计算；其余按“平台 + CANN + HCCL + 并行配置 + 运行开关”建立测量 profile。

### 6.3 Graph

`FULL_DECODE_ONLY` 的 Graph memory 在 KV pool 规划后产生，单独用于启动阶段 C 的物理 OOM 检查，不回填最低 KV 检查。

## 7. 启动极限计算流程

对每个候选 `(Q,S)`：

### 阶段 A：profile 后可用 KV

\[
M_{\text{available-KV}}(Q,S)
=M_{\text{requested}}
-M_{\text{model-load}}(Q,S)
-M_{\text{profile-activation}}(Q,S)
-M_{\text{non-Torch}}(Q,S)
\]

### 阶段 B：最低 KV 检查

\[
M_{\text{required-min-KV}}(L,Q,\text{layout})
\le M_{\text{available-KV}}(Q,S)
\]

### 阶段 C：KV pool 分配

按指定版本源码的 group、page、block rounding 和 worker-min 规则计算实际 pool tensor。

### 阶段 D：Graph 捕获

\[
M_{\text{post-KV-live}}+M_{\text{decode-graph}}
\le M_{\text{startup-free-HBM}}
\]

输出第一个失败阶段，不用一个总 HBM 数值代替生命周期判断。

## 8. 运行安全模型

运行时按 workload 场景计算：

\[
M_{\text{runtime-peak}}
=M_{\text{model-load}}
+M_{\text{allocated-KV-pool}}
+M_{\text{runtime-live-state}}
+M_{\text{activation-live-set}}
+M_{\text{workspace}}
+M_{\text{graph}}
+M_{\text{runtime-reserve}}
\]

至少覆盖：

- fresh prefill；
- late prefill；
- high-concurrency decode；
- MTP decode；
- 用户指定的上下文长度分布。

运行安全推荐需要物理 HBM 余量和未知项 reserve。

## 9. 校准边界

日志的作用限定为：

1. 验证理论分项是否与 vLLM 日志口径一致；
2. 标出未解释残差；
3. 为无法静态确定的 non-Torch/workspace/graph 建立版本化测量项；
4. 验证预测临界点是否落在实测区间。

禁止：

- 用一个总修正系数同时修正权重和 KV；
- 在源码公式已经可精确复现时继续拟合 KV；
- 把成功/失败九组数据全部用于拟合后，再把同一批数据报告为独立验证；
- 用实测权重直接覆盖错误的理论权重分类而不报告根因。

## 10. 九组边界验证

每组真实临界点为：

\[
Q^*\in[Q_{\text{success}},Q_{\text{fail}})
\]

验收规则：

1. 理论预测的最大成功 Q 落入区间；
2. 报告距离区间上下界的 token 数；
3. 报告限制阶段；
4. 报告理论分项和未解释残差；
5. 先完成零拟合的九点验证；
6. 若必须引入 measured profile，采用留一法验证，不能污染保留样本。

当前九组均固定 `S=64`，只能验证 Q 的启动边界，不能验证 S 的缩放或运行安全并发。

## 11. 代码结构

计划新增或重构：

```text
src/vllm_ascend_hbm/
  capacity.py
  startup.py
  runtime.py
  validation.py
  logs.py
  weights/
    manifest.py
    placement.py
    modelslim_w8a8.py
    persistent_buffers.py
  kv/
    deepseek_v4_flash_v020.py
    deepseek_v4_flash_v023.py
```

原则：

- `engine.py` 只组合，不内嵌版本公式；
- KV adapter 逐版本复现上游源码；
- 权重 adapter 使用张量 manifest 和 placement rule；
- 实测 profile 独立存放，不修改理论 adapter；
- 输出同时包含数值、公式来源和适用条件。

## 12. 测试优先顺序

实施前先写失败测试：

1. `L=32768,Q=47104` 的最低 KV 必须为约 17.25702 GiB；
2. Q 超过 L 后，state pages 必须因 `min(...,L)` 进入平台期；
3. 最低 KV 检查必须使用 22 tuple，物理 pool 分配路径单独验证 23
   allocation tuple；
4. sliding window 必须来自 config，当前模型为 128；
5. C4 主/indexer state 同组取最大值；
6. replicated/TP/EP/shared-expert-DP placement 单元测试；
7. W8A8 scale、offset、FP32 duplicate 测试；
8. Q 相关 MTP hidden/top-k buffer 测试；
9. 45056 成功与 47104 最低 KV 失败日志重建；
10. 九组区间验证；
11. 现有其他模型 profile 回归测试。

## 13. 本阶段验收标准

1. KV 最低需求以源码公式复现，单点误差不超过 0.01 GiB；
2. 权重不再使用“总参数统一 8 bit 后除 TP/EP”作为精确路径；
3. 有本地模型目录时输出逐张量精确放置清单；
4. 无模型目录时使用内置 tensor manifest，并显示版本；
5. 模型加载内存的未解释残差单独显示，不静默乘 overhead；
6. 九组边界首先进行无拟合验证；
7. 双口径输出和失败阶段清晰可解释；
8. README、配置模板和建模说明同步更新。
