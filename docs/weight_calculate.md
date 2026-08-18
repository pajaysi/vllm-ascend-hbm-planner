这四个函数组成了一个逐级降级的“权重容量解析器”：

```text
找到模型文件
    ↓
读取每个 tensor 的元数据
    ↓
按 tensor 名称推断它落在哪些 Rank
    ↓
得到每 Rank 权重 HBM

如果没有模型文件：
    ↓
根据总参数量和并行参数做粗粒度估算
```

它们位于 [weights.py](D:/code/chatgpt/LLM_inference/vllm-ascend-hbm-planner/src/vllm_ascend_hbm/weights.py:24)。

---

## 1. `resolve_safetensor_files()`

函数入口：

```python
def resolve_safetensor_files(model_path: str) -> list[Path]:
```

它解决的是：

> 用户传入的路径对应哪些 `.safetensors` 文件？

输入可以是三种形式。

### 单个 Safetensors 文件

```text
D:/models/model.safetensors
```

直接返回：

```python
[Path("D:/models/model.safetensors")]
```

### Safetensors index 文件

```text
D:/models/model.safetensors.index.json
```

索引文件通常类似：

```json
{
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00010.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00010.safetensors",
    "model.layers.40.mlp.down_proj.weight": "model-00009-of-00010.safetensors",
    "lm_head.weight": "model-00010-of-00010.safetensors"
  }
}
```

函数读取 `weight_map`，取出所有不同的分片文件名：

```python
sorted(set(weight_map.values()))
```

最终返回：

```text
model-00001-of-00010.safetensors
model-00002-of-00010.safetensors
...
model-00010-of-00010.safetensors
```

### 模型目录

```text
D:/models/Qwen3-32B/
```

函数优先寻找：

```text
model.safetensors.index.json
*.safetensors.index.json
```

如果没有 index，则枚举目录下所有：

```text
*.safetensors
```

### 这个函数不做什么

它不读取权重，也不计算内存，只负责确定文件集合。

可以把它理解成：

> 找到仓库目录和装箱清单。

---

## 2. `read_safetensor_header()`

函数入口：

```python
def read_safetensor_header(path: Path) -> list[TensorRecord]:
```

它解决的是：

> 每个 Safetensors 分片里有哪些 Tensor，每个 Tensor 占多少字节？

Safetensors 文件可以简化为：

```text
┌──────────────────────┐
│ 前8字节：header长度   │
├──────────────────────┤
│ JSON header          │
├──────────────────────┤
│ Tensor payload       │
│ 真正的权重数据        │
└──────────────────────┘
```

函数首先读取 8 字节：

```python
raw = stream.read(8)
header_len = struct.unpack("<Q", raw)[0]
```

`<Q` 表示：

- 小端序；
- unsigned 64-bit integer。

随后只读取 JSON Header：

```python
header = json.loads(stream.read(header_len))
```

不会读取后面的 Tensor Payload。

### Header 中的内容

一个 Tensor 的 Header 类似：

```json
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "I8",
    "shape": [4096, 4096],
    "data_offsets": [0, 16777216]
  }
}
```

函数把它转换为：

```python
TensorRecord(
    name="model.layers.0.self_attn.q_proj.weight",
    dtype="I8",
    shape=(4096, 4096),
    nbytes=16777216,
    file="model-00001-of-00010.safetensors"
)
```

字节数直接根据 offset 计算：

\[
M_{\text{tensor}}
=
\text{data\_offset}_{end}
-
\text{data\_offset}_{start}
\]

### 为什么不直接用 shape × dtype

用 `data_offsets` 的好处是获得 checkpoint 中真实存储字节数，包括：

- INT8 权重；
- BF16/FP16/FP32 权重；
- 量化 scale；
- 量化 zero point；
- 特殊压缩 Tensor。

不需要自己维护所有 dtype 的字节映射。

### 为什么只读取 Header

对于 300 GB 模型，Header 可能只有几十 MB。这样可以：

- 不把权重加载到 CPU 内存；
- 不需要 NPU；
- 不依赖 PyTorch；
- 快速统计整个 checkpoint；
- Windows 机器上也可以分析 Ascend 模型文件。

可以把它理解成：

> 只看每个箱子的标签、尺寸和编号，不打开箱子搬货。

---

## 3. `parsed_weight_estimate()`

函数入口：

```python
def parsed_weight_estimate(
    c: dict[str, Any],
    model_path: str,
) -> WeightEstimate:
```

它解决的是：

> 已经知道 checkpoint 中每个 Tensor 的精确字节数，如何推断每个 TP/EP/PP Rank 实际加载多少？

这一步分为四个阶段。

### 3.1 获取所有 Tensor

```python
files = resolve_safetensor_files(model_path)

records = [
    record
    for file in files
    for record in read_safetensor_header(file)
]
```

到这里已经获得整个 checkpoint 的 Tensor 清单：

```text
Tensor 名称
dtype
shape
nbytes
所在文件
```

因此：

```python
sum(record.nbytes for record in records)
```

是精确的 checkpoint payload 大小。

但单 Rank HBM 还需要根据张量放置方式进行切分。

---

### 3.2 判断 Tensor 属于哪个 PP Stage

函数 `_tensor_stage()` 使用名称中的层号：

```text
model.layers.0...
model.layers.1...
model.layers.12...
```

提取正则：

```python
layers\.(\d+)
```

然后近似均匀分配到 PP Stage：

\[
PP_{\text{stage}}
=
\left\lfloor
\frac{\text{layer index}\times PP}
{\text{total layers}}
\right\rfloor
\]

例如 40 层、`PP=2`：

```text
layer 0～19  → PP stage 0
layer 20～39 → PP stage 1
```

特殊规则：

```text
lm_head
output_layer
model.norm
```

分到最后一个 PP Stage。

没有层号的其他 Tensor 默认分到 Stage 0，例如 embedding。

这是一种通用近似，不一定完全等价于某个模型自定义的 PP layer assignment。

---

### 3.3 将 Tensor 分成三种放置类型

#### Routed Expert Tensor

识别规则：

```python
.experts.
```

但排除：

```python
.shared_expert.
.shared_experts.
```

例如：

```text
model.layers.12.mlp.experts.3.gate_proj.weight
model.layers.12.mlp.experts.3.down_proj.weight
```

认为按 EP 切分：

\[
M_{\text{expert/rank}}
=
\frac{M_{\text{expert,total}}}{EP}
\]

注意这里不除以 TP。

#### Replicated Tensor

识别规则包括：

```text
norm
layernorm
rms_norm
e_score_correction_bias
```

例如：

```text
model.layers.12.input_layernorm.weight
model.layers.12.post_attention_layernorm.weight
model.norm.weight
```

认为每个 Rank 都保留完整副本：

\[
M_{\text{replicated/rank}}
=
M_{\text{replicated}}
\]

不除以 TP 或 EP。

#### 其他 Tensor

其他 Tensor 统一认为按 TP 切分：

\[
M_{\text{sharded/rank}}
=
\frac{M_{\text{sharded,total}}}{TP}
\]

例如：

```text
embedding
attention projection
普通 dense MLP
lm_head
shared expert
```

在通用模型中都走这个分支。

---

### 3.4 计算每个 PP Stage 的最大 Rank 占用

对于第 \(i\) 个 PP Stage：

\[
M_i
=
\left(
\frac{M_{\text{expert},i}}{EP}
+
\frac{M_{\text{sharded},i}}{TP}
+
M_{\text{replicated},i}
\right)
\]

然后加入两个修正因子：

\[
M_i'
=
M_i
\times
(1+F_{\text{checkpoint-to-HBM}})
\times
F_{\text{PP imbalance}}
\]

其中：

- `checkpoint_to_hbm_overhead_fraction`：checkpoint 到加载后 HBM 的额外开销；
- `pp_imbalance_factor`：PP Stage 层数、Embedding、LM Head 不均衡的修正。

最后取最重的 PP Stage：

\[
\boxed{
M_{\text{per-rank}}
=
\max_i M_i'
}
\]

之所以取最大值，是因为部署容量由占用最大的 Rank 决定，不是由所有 Stage 的平均值决定。

### 返回结果

```python
WeightEstimate(
    per_rank_bytes=...,
    total_checkpoint_bytes=...,
    routed_expert_checkpoint_bytes=...,
    dense_sharded_checkpoint_bytes=...,
    replicated_checkpoint_bytes=...,
    max_pp_stage=...,
    source="safetensors-header+name-based-sharding"
)
```

这里应这样理解：

- checkpoint 总字节数：精确；
- 每个 Tensor 的字节数：精确；
- Tensor 属于 expert/replicated/sharded：根据名称推断；
- TP/EP/PP 放置：通用近似；
- 加载后额外副本：通过 overhead 粗略修正。

因此代码注释写的是：

```text
checkpoint bytes are exact;
rank placement is inferred from names
```

---

## 4. `analytical_weight_estimate()`

函数入口：

```python
def analytical_weight_estimate(c) -> WeightEstimate:
```

它解决的是：

> 如果本地没有 Safetensors 文件，只知道模型总参数量和结构，如何估算单 Rank 权重？

这是精度最低但适用范围最广的兜底路径。

### 4.1 估算 Routed Expert 参数量

如果模型有：

```text
num_routed_experts
num_moe_layers
hidden_size
moe_intermediate_size
```

则估算：

\[
P_{\text{routed}}
=
N_{\text{MoE layers}}
\times
N_{\text{experts}}
\times
3
\times
H
\times
I_{\text{MoE}}
\]

这里的 3 对应：

- Gate projection；
- Up projection；
- Down projection。

为了避免超过模型总参数量：

\[
P_{\text{routed}}
=
\min(P_{\text{total}},P_{\text{routed}})
\]

### 4.2 剩余参数作为 Dense 参数

\[
P_{\text{dense}}
=
\max(0,P_{\text{total}}-P_{\text{routed}})
\]

这里的 Dense 是广义分类，包含：

- Attention；
- Embedding；
- LM Head；
- Norm；
- Shared Expert；
- Router；
- 非 Routed Expert 参数。

### 4.3 根据量化 bit 数转成字节

Routed Expert：

\[
M_{\text{routed}}
=
P_{\text{routed}}
\times
\frac{B_{\text{expert}}}{8}
\]

Dense：

\[
M_{\text{dense}}
=
P_{\text{dense}}
\times
\frac{B_{\text{dense}}}{8}
\]

例如 W8A8：

```text
routed_expert_weight_bits = 8
dense_weight_bits = 8
```

则每个参数按 1 byte 计算。

如果 Dense 权重仍是 BF16：

```text
dense_weight_bits = 16
```

则每个参数按 2 byte 计算。

### 4.4 根据并行方式切分

\[
M_{\text{local}}
=
\frac{
M_{\text{routed}}/EP
+
M_{\text{dense}}/TP
}{PP}
\]

然后加入 overhead 和 PP imbalance：

\[
\boxed{
M_{\text{rank}}
=
\left(
\frac{
M_{\text{routed}}/EP
+
M_{\text{dense}}/TP
}{PP}
\right)
(1+F_{\text{overhead}})
F_{\text{PP imbalance}}
}
\]

---

## 5. 一个 MoE 示例

假设：

```text
总参数量     = 100B
Routed 参数  = 80B
其他参数     = 20B
全部为 INT8
TP           = 4
EP           = 16
PP           = 1
```

则：

\[
M_{\text{routed/rank}}
=
80/16
=
5\text{ GB}
\]

\[
M_{\text{dense/rank}}
=
20/4
=
5\text{ GB}
\]

所以：

\[
M_{\text{rank}}
\approx10\text{ GB}
\]

再加入加载开销和 PP 不均衡修正。

不能直接用：

\[
100/TP=25\text{ GB}
\]

因为 Routed Expert 是按 EP 分配的。

也不能直接用：

\[
100/EP=6.25\text{ GB}
\]

因为 Attention、Embedding、LM Head 等不是全部按 EP 分配。

---

## 6. 两种估计方式的区别

| 特性 | `parsed_weight_estimate()` | `analytical_weight_estimate()` |
|---|---|---|
| 需要本地模型文件 | 是 | 否 |
| checkpoint 总字节数 | 精确 | 参数量×bit |
| 每个 Tensor 大小 | 精确 | 不知道 |
| Routed Expert 识别 | 根据名称 | 根据结构公式 |
| Replicated Tensor | 部分识别 | 基本无法区分 |
| TP 切分 | 根据 Tensor 分类 | 所有非 Routed 参数统一除 TP |
| PP 分布 | 根据层号近似 | 直接整体除 PP |
| 加载后 FP32 副本 | 只通过 overhead | 只通过 overhead |
| 精度 | 中等 | 较低 |
| 适用场景 | 有模型目录 | 部署前只有模型规格 |

两者都不是 vLLM 加载后的绝对精确内存模型。

---

## 7. 为什么 DSV4 不优先使用这两个通用函数

DeepSeek-V4-Flash W8A8 存在很多通用分类无法准确表达的情况：

- Routed Expert 按 EP；
- Shared Expert 的放置取决于 `enable_shared_expert_dp`；
- 一部分 Attention Tensor replicated；
- 一部分 Attention Tensor TP-sharded；
- Router 保留 BF16 和 FP32 两份；
- Compressor 投影实际使用 FP32；
- MTP embedding/head 存在 alias；
- RoPE Cache 在模型构造时创建；
- MTP hidden buffer 随 Q 变化；
- top-k Buffer 随 Q 变化。

Safetensors Header 能看到 checkpoint Tensor，却看不到很多加载后产生的 Buffer 和副本。

因此 DSV4 v0.23 优先调用专用的：

```python
estimate_deepseek_v4_w8a8()
```

而不是 `parsed_weight_estimate()`。

---

## 8. 最终选择顺序

真正的统一入口是 [estimate_weights()](D:/code/chatgpt/LLM_inference/vllm-ascend-hbm-planner/src/vllm_ascend_hbm/weights.py:165)。

选择顺序为：

```text
1. 是否有 vLLM profile 实测权重？
   └─ 有：使用实测值
      DSV4 同时保留理论值和 residual

2. 是否配置 manual_gib_per_rank？
   └─ 有：使用手工值

3. 是否为 DSV4 Flash + vLLM Ascend v0.23？
   └─ 是：使用 estimate_deepseek_v4_w8a8()

4. 是否有 model_path？
   └─ 是：使用 parsed_weight_estimate()

5. 都没有
   └─ 使用 analytical_weight_estimate()
```

所以四个函数的关系可以简单记为：

```text
resolve_safetensor_files
    = 找到有哪些权重文件

read_safetensor_header
    = 查清有哪些 Tensor、每个多大

parsed_weight_estimate
    = 根据 Tensor 名称推断它们如何分到各 Rank

analytical_weight_estimate
    = 没有 Tensor 清单时，根据总参数量粗估
```

最关键的区别是：

> Safetensors Header 能精确回答“checkpoint 中存了多少字节”，但不一定能精确回答“vLLM 加载完成后每个 Rank 占多少 HBM”；后者还受运行时副本、模块放置、并行策略、alias 和模型常驻 Buffer 影响。