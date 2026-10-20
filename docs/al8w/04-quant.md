
## 简单评估离线吞吐量

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

Throughput: 9.28 requests/s, 10695.38 total tokens/s, 1188.38 output tokens/s
Total num prompt tokens:  32768
Total num output tokens:  4096

## AWQ

我想采用 vllm 模型来评估awq的效果，如何实现？
/media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B-AWQ/


### 1. 本地模型与评估目标

已检查该目录的 `config.json`：模型架构是 `Qwen3ForCausalLM`，量化方法
为 `awq`，4 bit、group size 128、`zero_point=true`、GEMM 格式，
标注激活类型为 FP16。这是现成的 AWQ 权重，无需在 vLLM 中再执行校准。
本机 RTX 4060 Ti 16 GB 的容量适合先做该 4B 模型的单卡短上下文测试。

评估分成三件事：模型能否正确加载回答、固定负载下的吞吐、相对原模型的质量损失。
上面的 Qwen3-0.6B 可用于检查环境，但参数规模不同，不能用来归因 AWQ 的收益。
严谨对照需要这份 AWQ 检查点对应的 Qwen3-4B 原始模型及相同修订。

以下命令在 **vLLM 仓库** 执行，使用该仓库的 `.venv`，不使用隔壁 Omni 环境。
命令根据本地源码核对，尚未实际运行模型评估，不预设加速或准确率结果。

### 2. 先验证回答

```bash
cd /media/l8w/Linux118/PROJECTS/29-vllm-serials/vllm
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="/media/l8w/Linux118/.cache"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv/ComputeCache"
mkdir -p "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" results/awq

.venv/bin/python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="/media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B-AWQ",
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.7,
    seed=42,
)
outputs = llm.chat(
    [{"role": "user", "content": "请计算 17×23，并用一句话说明计算过程。"}],
    sampling_params=SamplingParams(temperature=0, max_tokens=128),
    chat_template_kwargs={"enable_thinking": False},
)
print(outputs[0].outputs[0].text)
PY
```

正确答案是 391。这仅是加载与输出的冒烟测试，不能证明量化质量。
默认从模型配置自动识别 AWQ，不必手动加 `quantization="awq"`。
在适用硬件和权重格式下，vLLM 可能自动选择 AWQ Marlin；记录日志中的实际后端。
AWQ 是量化格式/方法，Marlin 是执行内核，二者并不矛盾。

### 3. 固定负载下评估离线吞吐

```bash
.venv/bin/vllm bench throughput \
  --model /media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B-AWQ \
  --backend vllm \
  --dataset-name random \
  --num-prompts 32 \
  --num-warmups 4 \
  --input-len 128 \
  --output-len 128 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --seed 42 \
  --output-json results/awq/qwen3-4b-awq-128-128.json \
  2>&1 | tee results/awq/qwen3-4b-awq-128-128.log

mkdir -p results/baseline results/smq
set -o pipefail

.venv/bin/vllm bench throughput \
  --model /media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B \
  --backend vllm \
  --dataset-name random \
  --num-prompts 32 \
  --num-warmups 4 \
  --input-len 128 \
  --output-len 128 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --seed 42 \
  --output-json results/baseline/qwen3-4b-fp16-128-128.json \
  2>&1 | tee results/baseline/qwen3-4b-fp16-128-128.log

.venv/bin/vllm bench throughput \
  --model /media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B-smQ \
  --backend vllm \
  --dataset-name random \
  --num-prompts 32 \
  --num-warmups 4 \
  --input-len 128 \
  --output-len 128 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --seed 42 \
  --output-json results/smq/qwen3-4b-smq-128-128.json \
  2>&1 | tee results/smq/qwen3-4b-smq-128-128.log

mkdir -p results/fp8
set -o pipefail

.venv/bin/vllm bench throughput \
  --model /media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B-FP8 \
  --backend vllm \
  --dataset-name random \
  --num-prompts 32 \
  --num-warmups 4 \
  --input-len 128 \
  --output-len 128 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --seed 42 \
  --output-json results/fp8/qwen3-4b-fp8-128-128.json \
  2>&1 | tee results/fp8/qwen3-4b-fp8-128-128.log
```

先用 32 请求确认流程；正式比较建议增加到 256，并重复至少 3 次取中位数。
同时测试长输入短输出（如 1024/128）和短输入长输出（如 128/512），
因为量化对 prefill 和 decode 的影响不同。输入输出总长须不超过设定上下文。
随机 token 测试只用于性能，不用于判断回答质量。

对非量化 Qwen3-4B 重复同一命令，只替换 `--model` 和输出文件名。
两次测试分别启动、串行执行，保持 dtype、请求数、长度、seed、上下文限制、
GPU 内存比例以及其他参数一致，并记录软件版本和实际量化内核。
如果原模型 OOM，应先把两边都调整成能够运行的共同负载，再比较；
“某配置下 AWQ 能运行而原模型 OOM”也可以作为容量结果单独记录。

| 维度 | 怎么看 |
| --- | --- |
| 吞吐 | requests/s、output tokens/s；同一口径下 AWQ / 原模型 |
| 显存 | 权重加载占用、KV cache 容量及运行峰值分别记录 |
| 质量 | 同一真实任务数据集、提示模板、生成参数下的准确率或任务分数 |

`gpu_memory_utilization=0.7` 会影响 KV cache 规划，因此 `nvidia-smi` 中
总显存占用相近，并不意味着 AWQ 没有减少权重显存。优先查看日志中的模型权重
占用和可用 KV cache 容量；不要把权重文件大小直接当作显存峰值。

### 4. 如何验证质量损失

使用同一批固定题目，分别让原模型和 AWQ 模型在 `temperature=0`、
`enable_thinking=False`、相同 `max_tokens` 下生成并评分。
可以从有标准答案的算术、事实问答和你的实际任务开始；记录整体正确率和失败样本，
不要求两个模型逐字一致。正式评估应使用固定版本的公开评测集或业务测试集，
记录题目数量、评分方式和 chat template。

本地已存在 `/media/l8w/Linux118/MODEL/LLM/Qwen/Qwen3-4B`，可作为候选原模型。
将第 3 节命令的模型路径改成该目录，并将输出文件名中的 `awq` 改为 `fp16`，
即可建立对照。正式归因前还需确认该目录与 AWQ 检查点的源模型修订一致；
尚未实际测量，不能给出速度提升倍数或质量损失结论。


### 实验数据

#### NAIVE

Throughput: 1.58 requests/s, 1822.38 total tokens/s, 202.49 output tokens/s
Total num prompt tokens:  32768
Total num output tokens:  4096

#### FP8

Throughput: 2.75 requests/s, 3162.39 total tokens/s, 351.38 output tokens/s
Total num prompt tokens:  32768
Total num output tokens:  4096

#### AWQ

Throughput: 3.13 requests/s, 3608.93 total tokens/s, 400.99 output tokens/s
Total num prompt tokens:  32768
Total num output tokens:  4096

#### SMQ

Throughput: 4.02 requests/s, 4625.64 total tokens/s, 513.96 output tokens/s
Total num prompt tokens:  32768
Total num output tokens:  4096
