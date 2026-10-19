# vLLM 版本与硬件、CUDA 要求

整理日期：2026-10-19。以下以 **Linux + NVIDIA GPU** 为主，选取近期版本和历史代表版本，按发布时间倒序排列。“热门版本”没有统一的官方下载排名，这里按版本节点选取，不表示下载量排名，也不穷举所有补丁版本。

## 1. 版本对照表

日期按北京时间记录；版本链接指向官方 Release，可核对发布时间与安装资产。GPU 列是该版本安装文档给出的最低 Compute Capability（CC），不保证所有模型、量化和注意力后端都支持该显卡。

| 发布日期 | 代表版本 | Python | NVIDIA GPU 文档门槛 | CUDA：安装文档 / 默认镜像构建 | 安装文档 |
| --- | --- | --- | --- | --- | --- |
| 2026-10-18 | [0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) | 3.11–3.14 | CC ≥ 7.5 | **Release 明确：PyPI 默认 13.0，另有 12.9 包及镜像**；安装文档仍写 12.9 | [文档](https://docs.vllm.ai/en/v0.32.0/getting_started/installation/gpu/) |
| 2026-08-11 | [0.27.0](https://github.com/vllm-project/vllm/releases/tag/v0.27.0) | 3.10–3.14¹ | CC ≥ 7.5 | 文档 12.9 / Dockerfile 13.0.3 | [文档](https://docs.vllm.ai/en/v0.27.0/getting_started/installation/gpu/) |
| 2026-05-29 | [0.22.0](https://github.com/vllm-project/vllm/releases/tag/v0.22.0) | 3.10–3.14¹ | CC ≥ 7.5 | 文档 12.9 / Dockerfile 13.0.2 | [文档](https://docs.vllm.ai/en/v0.22.0/getting_started/installation/gpu/) |
| 2026-04-28 | [0.20.0](https://github.com/vllm-project/vllm/releases/tag/v0.20.0) | 3.10–3.14¹ | CC ≥ 7.5 | **Release 明确：PyPI 和默认镜像切换到 13.0**；安装文档仍写 12.9 | [文档](https://docs.vllm.ai/en/v0.20.0/getting_started/installation/gpu/) |
| 2026-01-29 | [0.15.0](https://github.com/vllm-project/vllm/releases/tag/v0.15.0) | 3.10–3.13 | CC ≥ 7.0² | 安装段落 12.9 / Dockerfile 12.9.1；文档概述仍写 12.8 | [文档](https://docs.vllm.ai/en/v0.15.0/getting_started/installation/gpu/) |
| 2025-10-03 | [0.11.0](https://github.com/vllm-project/vllm/releases/tag/v0.11.0) | 3.9–3.13¹ | CC ≥ 7.0² | 文档 12.8 / Dockerfile 12.8.1 | [文档](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu/) |
| 2025-09-13 | [0.10.2](https://github.com/vllm-project/vllm/releases/tag/v0.10.2) | 3.9–3.13¹ | CC ≥ 7.0² | 文档 12.8 / Dockerfile 12.8.1 | [文档](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html) |
| 2025-07-08 | [0.9.2](https://github.com/vllm-project/vllm/releases/tag/v0.9.2) | 3.9–3.12 | CC ≥ 7.0² | 文档 12.8 / Dockerfile 12.8.1 | [源码文档](https://github.com/vllm-project/vllm/blob/v0.9.2/docs/getting_started/installation/gpu/cuda.inc.md) |
| 2025-04-29 | [0.8.5](https://github.com/vllm-project/vllm/releases/tag/v0.8.5) | 3.9–3.12 | CC ≥ 7.0² | 文档 12.4 / Dockerfile 12.4.1 | [文档](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html) |
| 2025-02-21 | [0.7.3](https://github.com/vllm-project/vllm/releases/tag/v0.7.3) | 文档 3.9–3.12 | CC ≥ 7.0² | 文档 12.1 / Dockerfile 12.4.1³ | [文档](https://docs.vllm.ai/en/v0.7.3/getting_started/installation/gpu/) |
| 2024-12-27 | [0.6.6](https://github.com/vllm-project/vllm/releases/tag/v0.6.6) | 3.9–3.12 | CC ≥ 7.0² | 文档 12.1 / Dockerfile 12.4.1³ | [文档](https://docs.vllm.ai/en/v0.6.6/getting_started/installation.html) |
| 2024-08-24 | [0.5.5](https://github.com/vllm-project/vllm/releases/tag/v0.5.5) | 源码文档 3.8–3.12 | CC ≥ 7.0² | 源码文档 12.1 / Dockerfile 12.4.1³ | [源码文档](https://github.com/vllm-project/vllm/blob/v0.5.5/docs/source/getting_started/installation.rst) |
| 2024-06-01 | [0.4.3](https://github.com/vllm-project/vllm/releases/tag/v0.4.3) | 3.8–3.11 | CC ≥ 7.0² | 文档 12.1；另列 CUDA 11.8 安装方式 | [源码文档](https://github.com/vllm-project/vllm/blob/v0.4.3/docs/source/getting_started/installation.rst) |

¹ Python 范围依据对应标签的 `pyproject.toml` 中 `requires-python`；部分安装文档仍写更窄范围。它表示包声明的范围，具体平台是否有 wheel、依赖是否可安装还需核对发布资产。例如：[0.22.0 包配置](https://github.com/vllm-project/vllm/blob/v0.22.0/pyproject.toml)、[0.10.2 包配置](https://github.com/vllm-project/vllm/blob/v0.10.2/pyproject.toml)。

² 历史文档写 CC ≥ 7.0，不代表每个版本的 V1 引擎、PyTorch wheel 和后端内核都能运行 V100。使用 V100 等旧卡时，必须进一步核对引擎和后端要求，不能仅按此列选版本。

³ 这些版本的安装文档与 Docker 构建配置不同，故分别列出。Dockerfile 的 CUDA 版本也不能直接作为 PyPI wheel 的版本证明。复现环境时，以实际下载的 wheel、配套 PyTorch 和镜像标签为准。构建配置示例：[0.7.3 Dockerfile](https://github.com/vllm-project/vllm/blob/v0.7.3/Dockerfile)、[0.15.0 Dockerfile](https://github.com/vllm-project/vllm/blob/v0.15.0/docker/Dockerfile)、[0.27.0 Dockerfile](https://github.com/vllm-project/vllm/blob/v0.27.0/docker/Dockerfile)。

## 2. 显卡型号与算力对照

| GPU 架构 | 常见型号 | CC | 选版本时关注什么 |
| --- | --- | --- | --- |
| Blackwell | RTX 5090 / 5080、RTX PRO 6000 Blackwell | 12.0 | 必须核对 wheel 和后端是否包含 SM120 支持 |
| Blackwell | B200 / GB200 | 10.0 | 官方安装文档明确要求 CUDA ≥ 12.8；仍需配套驱动与 PyTorch |
| Hopper | H100 / H200 / GH200 | 9.0 | 按所需模型、量化和 CUDA 构建选择 |
| Ada | RTX 4090、L4 / L40 / L40S | 8.9 | 同上；显存容量另行核算 |
| Ampere | RTX 3090、A10 / A40 | 8.6 | 同上 |
| Ampere | A100 / A30 | 8.0 | 同上 |
| Turing | T4、RTX 2080 / 2080 Ti | 7.5 | 达到较新文档的最低算力要求，但部分内核和数据类型不可用 |
| Volta | V100 | 7.0 | 不满足上表较新版本的 CC ≥ 7.5 门槛；旧版也需确认引擎兼容性 |

型号和算力依据 [NVIDIA GPU 算力表](https://developer.nvidia.com/cuda/gpus) 与 [历史 GPU 算力表](https://developer.nvidia.com/cuda-legacy-gpus)。B200 / GB200 的 CUDA 要求见 [vLLM 0.10.2 安装说明](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html)。

## 3. CUDA、驱动和本机 Toolkit 怎么区分

- **NVIDIA 驱动**：由操作系统提供，必须支持所选 CUDA 运行库。`nvidia-smi` 顶部的 CUDA Version 表示驱动支持的 CUDA 版本上限，不是本机安装的 Toolkit 版本。
- **CUDA 运行库**：通常随 PyTorch / NVIDIA Python 依赖或容器提供。上表列出的 CUDA 构建版本主要属于这一层。
- **CUDA Toolkit**：含 `nvcc`，用于本地编译 CUDA 代码。安装预编译包时通常不必先安装同版本的系统 Toolkit；源码编译、某些运行时编译路径或额外组件可能需要它。
- **预编译扩展**：还依赖特定 PyTorch 构建及 ABI，不能仅看 CUDA 大版本相同就随意替换 PyTorch。

NVIDIA 给出的 CUDA 大版本兼容下限如下；这是通用兼容规则，**不是 vLLM 整套依赖的运行保证**。PTX/JIT 或新特性可能要求更新驱动。

| CUDA 系列 | NVIDIA 通用驱动分支下限 |
| --- | --- |
| CUDA 13.x | R580 |
| CUDA 12.x | R525 |
| CUDA 11.x | R450；Linux 小版本兼容规则需至少 450.80.02 |

来源：[NVIDIA CUDA 小版本兼容规则及限制](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

## 4. 显存要求与其他硬件

vLLM 没有适用于所有模型的统一“最低显存”。显存需要容纳权重、KV cache、运行时工作区及 CUDA Graph 等；模型大小、量化、上下文长度和并发都会影响占用。

仅估算权重：`参数量 × 每参数字节数`。例如 7B 模型的 FP16/BF16 权重约 14 GB（十进制），还没有计入 KV cache 和运行开销，因此不能据此断定 16 GB 显卡一定能运行目标配置。量化还会有 scale 等额外数据。

AMD GPU 使用 ROCm，Intel GPU 使用 XPU 软件栈，CPU 使用对应 CPU 构建，均不适用上述 NVIDIA CUDA 列。以 [0.22.0 硬件安装文档](https://docs.vllm.ai/en/v0.22.0/getting_started/installation/gpu/) 为例，AMD 文档列 ROCm ≥ 6.3，MI350 需 ≥ 7.0；实际预编译包要求可能更高。Apple Silicon 则需核对独立的 vLLM-Metal 插件。

## 5. 查看当前环境

```bash
# 查看 GPU 和驱动
nvidia-smi

# 查看本机 CUDA Toolkit（如果已安装）
/usr/local/cuda-12.8/bin/nvcc --version

# 查看虚拟环境中的版本和 GPU 算力
.venv/bin/python - <<'PY'
import torch
import vllm

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index),
          "CC:", torch.cuda.get_device_capability(index))
PY
```

你此前日志中的 `0.22.1rc1.dev2919+gfe1c31715.precompiled` 是开发构建，不等同于稳定版 `0.22.0` 或 `0.22.1`。开发环境还应记录 Git 提交、预编译 wheel 来源和实际 `torch.version.cuda`，不能只按版本号前缀套用稳定版要求。
