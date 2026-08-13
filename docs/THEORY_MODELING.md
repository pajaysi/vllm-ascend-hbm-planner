# 理论优先的 HBM 建模说明

## 1. 两个容量口径

### 1.1 启动极限

`vllm serve` 的最小 KV 准入检查使用：

```text
requested_memory
  = visible_hbm × gpu_memory_utilization

available_kv
  = requested_memory
  - model_load
  - profile_peak_activation
  - non_torch_memory

startup_pass
  = minimum_kv(max_model_len, max_num_batched_tokens)
    <= available_kv
```

NPU Graph 在该日志路径中单独报告，不从 `available_kv` 再减一次。图捕获失败
属于后续阶段。因此代码把“最小 KV 检查”和“图捕获”分开，避免重复扣除。

### 1.2 运行安全

运行安全口径更保守：

```text
runtime_planning
  = weights
  + KV BlockPool planner capacity
  + peak activation
  + operator workspace
  + graph cache
  + runtime/non-torch
  + allocator fragmentation
  + safety reserve
```

推荐参数必须同时满足启动准入和运行预算。输出中的：

- `startup_limit_recommended`：只按启动生命周期筛选；
- `runtime_safe_recommended`：再通过运行期保守上界；
- `single_service_recommended`：兼容旧接口，等同运行安全推荐。

## 2. DeepSeek-V4 v0.23 最小 KV

适用条件：

- DeepSeek-V4-Flash；
- vllm-ascend 0.23.0rc1；
- `block_size=128`；
- 无 PCP/DCP；
- MTP 1 层。

页面布局：

```text
small page = 16,640 B
large page = 131,072 B
minimum-admission layer tuple count = 22
bytes per global page = (16,640 + 131,072) × 22
```

页数由五部分组成：

```text
C4 history
C128 history
2 × SWA group
C4 compressor state
C128 compressor state
```

源码等价公式位于
`src/vllm_ascend_hbm/kv/deepseek_v4_v023.py`。以
`L=32768, Q=47104` 为例：

```text
total pages = 5,702
minimum KV
  = 5,702 × (16,640 + 131,072) × 22
  = 18,529,584,128 B
  = 17.2570199966 GiB
```

实测失败日志为 `17.26 GiB`，误差只来自日志保留两位小数。因此此前 KV
Planner 的大偏差不是页面公式问题。

当 Q 大于单请求 `max_model_len` 时，状态窗口受 L 限制，最小 KV 会进入平台；
但权重阶段创建的 Q 相关缓冲和 profile activation 仍会随 Q 增长，因此启动
临界值仍会变化。

## 3. W8A8 权重与模型常驻缓冲

不能用“总参数 × 1 byte ÷ TP”计算 DeepSeek-V4：

- routed expert 按 EP 分片，不按 TP 再除；
- 开启 shared expert DP 后，共享专家在 Rank 上复制；
- 部分 attention 投影是 TP 分片，部分是 replicated；
- compressor 投影由量化描述标记为 FLOAT/FP32；
- router 同时保留 BF16 参数和 FP32 副本；
- W8A8 linear 还有 scale、offset 和加载后的 FP32 scale；
- MTP embedding 独立，MTP LM head 与 target head 共享；
- RoPE 创建普通 theta 和压缩 theta 两套完整 FP32 cos/sin cache；
- MTP hidden、Top-k、RoPE runtime buffer 随 Q 增长。

TP2、EP16、Q=45056 的理论分解：

| 分项 | GiB/rank |
|---|---:|
| Routed experts | 16.532 |
| Shared expert | 1.034 |
| TP-sharded attention | 3.443 |
| Replicated attention | 0.258 |
| Compressor | 0.974 |
| Indexer | 0.340 |
| Router | 0.267 |
| mHC | 0.129 |
| Embedding/head | 1.479 |
| MTP projection/norm | 0.063 |
| MTP/Top-k/RoPE buffers | 2.547 |
| 其他小项 | 0.001 |
| **理论合计** | **27.066** |
| **启动日志** | **27.170** |
| **残差** | **0.104（0.38%）** |

权重实现位于
`src/vllm_ascend_hbm/weight_models/deepseek_v4_w8a8.py`。

## 4. Profile activation

激活模型显式计算 mHC residual、hidden I/O、attention query/indexer/top-k、
MoE dispatch 和 expert intermediate，并按分支同时存活系数聚合。

TP2、Q=45056：

```text
理论 activation = 7.204 GiB
日志 activation = 7.270 GiB
残差             = 0.066 GiB（0.91%）
```

FlashComm1 对 TP>2 的 sequence-parallel 有效 token 数仍是模型中不确定性较高的
部分。TP1/TP4 的成功日志可以直接验证这部分，而无需修改 KV 公式。

新增日志验证结果：

```text
TP1, Q=23552：理论 3.766 GiB，实测 4.740 GiB
TP4, Q=39936：理论 3.193 GiB，实测 3.380 GiB
```

TP1 的 activation 残差明显大于 TP2/TP4，说明同一个
`branch_live_fraction` 不能作为跨拓扑的精确常数。代码保留结构理论值，同时
允许每种 TP 使用一份 profile 日志校准；没有反向调整 KV 页面公式。

## 5. HCCL、CANN 与 ACL

HCCL 文档给出的单通信域缓存是：

```text
HCCL memory = 2 × HCCL_BUFFSIZE × communication_domains
```

`HCCL_BUFFSIZE=1024 MiB`、一个通信域时为 `2.00 GiB/rank`。

TP2 成功日志：

```text
non-torch measured = 3.08 GiB
known HCCL          = 2.00 GiB
remaining baseline = 1.08 GiB
```

这 1.08 GiB 在校准配置中被显式命名为 CANN/ACL fixed baseline。默认理论配置
不包含任意经验常数，`base_persistent_gib_per_rank=0`。

通信域数量可能随 TP/DP/EP 拓扑变化。代码要求显式输入
`hccl_communication_domains_per_rank`，不会擅自把所有 communicator 都当成
完整 1 GiB 双向缓存。

## 6. 实测校准但不覆盖理论

提供 profile 日志或 `profile_calibration` 后：

- 输出继续保留 theoretical weight；
- 输出 measured-theory residual；
- Q 改变时，权重使用理论 Q-buffer 增量；
- activation 以实测参考 Q 做线性缩放；
- KV 仍使用确定性的源码公式。

这样校准只补偿无法从公开配置确定的内存，不会把错误 KV/权重公式藏在拟合
系数中。

## 7. 模型扩展

模型支持分为四级：

1. `source-exact`：版本锁定的异构 KV 和张量放置模型；
2. `parsed`：解析本地 `config.json` 和 Safetensors header；
3. `structural`：通用 GQA/MHA 或 MLA 公式；
4. `manual`：混合 Attention/SSM 使用手工 KV Adapter。

“vLLM Ascend 能运行某模型”不等于“本工具已对该模型达到 source-exact”。
具体范围见仓库根目录的 `MODEL_SUPPORT.md`。
