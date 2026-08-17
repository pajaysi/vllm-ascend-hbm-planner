# Windows EXE 使用说明

## 1. 文件组成

发布包包含：

```text
vllm-ascend-hbm.exe
configs/
  hardware_910c_1node.json
  deepseek_v4_flash_910c_inference.json
  ...
```

`vllm-ascend-hbm.exe` 是 Windows x64 控制台程序，不要求目标机器安装
Python。它用于部署前容量估算和参数推荐，不在昇腾计算节点上执行模型推理。

## 2. 推荐的双 JSON 输入

硬件库存和推理配置分别输入：

```powershell
.\vllm-ascend-hbm.exe `
  --hardware-config .\configs\hardware_910c_1node.json `
  --config .\configs\deepseek_v4_flash_910c_inference.json
```

硬件 JSON 保存相对稳定的事实：

- 设备型号；
- 服务器数量；
- 每服务器物理卡数；
- 每卡逻辑 Die 数；
- 每服务器逻辑设备数；
- 标称、运行时可见和启动时空闲 HBM。

推理 JSON 保存部署相关参数：

- vLLM Ascend 版本；
- `gpu_memory_utilization`；
- 模型和量化配置；
- DP、TP、PP、EP、PCP、DCP；
- `max_model_len`、`max_num_batched_tokens` 和 `max_num_seqs`；
- 推荐目标和候选参数范围。

工具会验证：

```text
physical_cards_per_server * dies_per_card
    == logical_devices_per_server

DP * TP * PP
    <= server_count * logical_devices_per_server
```

硬件文件中的硬件容量字段优先于推理配置中的同名旧字段。没有提供
`--hardware-config` 时，原有单 JSON 配置仍然可用。

## 3. 常用命令

查看帮助：

```powershell
.\vllm-ascend-hbm.exe --help
```

查看内置模型：

```powershell
.\vllm-ascend-hbm.exe --list-models
```

固定参数内存估算：

```powershell
.\vllm-ascend-hbm.exe `
  --hardware-config .\configs\hardware_910c_1node.json `
  --config .\configs\deepseek_v4_flash_910c_inference.json `
  --operation estimate
```

输出 JSON：

```powershell
.\vllm-ascend-hbm.exe `
  --hardware-config .\configs\hardware_910c_1node.json `
  --config .\configs\deepseek_v4_flash_910c_inference.json `
  --format json
```

使用本地模型目录解析权重：

```powershell
.\vllm-ascend-hbm.exe `
  --hardware-config .\configs\hardware_910c_1node.json `
  --config .\configs\deepseek_v4_flash_910c_inference.json `
  --model-path D:\models\DeepSeek-V4-Flash-w8a8-mtp
```

使用启动日志校准：

```powershell
.\vllm-ascend-hbm.exe `
  --hardware-config .\configs\hardware_910c_1node.json `
  --config .\configs\deepseek_v4_flash_910c_inference.json `
  --profile-log D:\logs\run.log
```

## 4. 在任意目录调用

如果不把 EXE 所在目录加入 `PATH`，需要使用完整路径或 `.\`：

```powershell
D:\tools\vllm-ascend-hbm\vllm-ascend-hbm.exe --list-models
```

把发布目录加入 Windows 用户 `PATH` 后，可以直接执行：

```powershell
vllm-ascend-hbm --list-models
```

## 5. 兼容性

- 构建目标：Windows x64；
- 原有 `--config` 单 JSON 调用保持兼容；
- Windows EXE 不能直接复制到 Linux/昇腾服务器运行；
- Linux 如需独立可执行文件，应在对应 Linux 构建环境单独打包。

## 6. 从源码重新构建

构建机需要 Windows、Python 和 PyInstaller。仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

脚本会执行完整单元测试、PyInstaller 单文件构建、EXE 冒烟测试以及
Python/EXE JSON 输出一致性检查。构建结果位于：

```text
release/vllm-ascend-hbm-windows-x64-v<版本>/
release/vllm-ascend-hbm-windows-x64-v<版本>.zip
```
