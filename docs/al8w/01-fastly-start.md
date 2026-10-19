# 使用 venv 安装 vLLM

以下命令在 vLLM 仓库根目录执行，适用于 NVIDIA CUDA 环境。

仓库所在的 `/media/l8w/Linux118` 磁盘当前约有 44 GB 可用空间，
根分区约有 16 GB。先把安装缓存和临时文件目录指定到该挂载磁盘，
避免 uv 默认缓存目录 `~/.cache/uv` 和默认临时目录 `/tmp` 占满根分区。

```bash
# 将缓存、下载解压和构建临时文件放到挂载磁盘
VLLM_INSTALL_STORAGE="/media/l8w/Linux118/.cache/vllm-install"
export UV_CACHE_DIR="$VLLM_INSTALL_STORAGE/uv"
export PIP_CACHE_DIR="$VLLM_INSTALL_STORAGE/pip"
export TMPDIR="$VLLM_INSTALL_STORAGE/tmp"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# 缓存和 .venv 在同一文件系统，使用硬链接减少重复占用
export UV_LINK_MODE=hardlink

# 环境显示名称为 vllm，保存目录为 .venv
/home/l8w/miniconda3/envs/cp312/bin/python -m venv --prompt vllm .venv
source .venv/bin/activate

# 安装当前仓库源码；使用预编译组件避免本地编译
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# 验证安装
.venv/bin/python -c "import vllm; print(vllm.__version__)"
```

以后进入仓库，运行 `source .venv/bin/activate` 即可重新使用该环境。
上述环境变量只对当前终端及其子进程生效；在新终端安装或升级依赖前，
需重新执行缓存和临时目录的设置命令。已有的根分区缓存不会自动迁移或删除。

## 使用本地 Qwen3-0.6B 测试

以下命令在仓库根目录执行，使用 `../00-COMMON/Qwen/Qwen3-0.6B`，
无需启动 API 服务。先把运行时缓存也放到挂载磁盘：

```bash
source .venv/bin/activate
export XDG_CACHE_HOME="/media/l8w/Linux118/.cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv/ComputeCache"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
```

### 1. 验证模型能正常回答

```bash
.venv/bin/python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="../00-COMMON/Qwen/Qwen3-0.6B",
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.7,
)
outputs = llm.chat(
    [{"role": "user", "content": "请用一句话解释什么是大语言模型。"}],
    sampling_params=SamplingParams(temperature=0, max_tokens=128),
    chat_template_kwargs={"enable_thinking": False},
)
print(outputs[0].outputs[0].text)
PY
```

### 2. 简单评估离线吞吐量

使用 32 个随机请求，每个输入 128 tokens、输出 128 tokens：

```bash
.venv/bin/vllm bench throughput \
  --model ../00-COMMON/Qwen/Qwen3-0.6B \
  --backend vllm \
  --dataset-name random \
  --num-prompts 32 \
  --input-len 128 \
  --output-len 128 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7
```

关注输出的请求吞吐量（requests/s）和输出 token 吞吐量（output tokens/s）。
随机输入用于性能测试，不评估回答准确率；32 个请求仅适合快速验证，
正式比较时应增加请求数，并保持硬件、输入输出长度和运行参数一致。
首次运行可能需要编译内核，耗时会较长。若显存不足，可降低
`--max-model-len` 或 `--gpu-memory-utilization`，并确保没有其他进程占用 GPU。
