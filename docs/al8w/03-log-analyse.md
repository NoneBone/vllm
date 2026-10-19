# Qwen3-0.6B 推理日志分析

分析对象：[chat.qwen3.log](log/chat.qwen3.log)。这是一次本地模型、单请求的离线聊天测试，模型成功输出中文回答。日志中的引擎版本为 `0.32.1rc1.dev62+g283d48f4e`。

最明确的性能优化是 **FlashAttention 2、torch.compile/Inductor 编译、CUDA Graph 捕获及半精度推理**。此外，前缀缓存、分块预填充已开启，但单次短请求无法证明它们带来了收益。本日志没有优化前后的对照实验，不能计算各项技术的加速倍数。

## 1. 这次运行的基本配置

| 字段 | 日志中的值 | 含义 |
| --- | --- | --- |
| `model` / `tokenizer` | `../00-COMMON/Qwen/Qwen3-0.6B` | 从本地目录加载模型及分词器 |
| `Resolved architecture` | `Qwen3ForCausalLM` | 使用 Qwen3 因果语言模型实现 |
| 引擎 / 执行器 | V1 LLM engine / V2 Model Runner | 引擎架构版本与模型执行器版本是不同层次，并不冲突 |
| `device_config` | `cuda` | 在 NVIDIA GPU 上执行；日志没有直接提供显卡型号 |
| `dtype` | `torch.float16` | 权重和主要模型计算采用 FP16；部分内部运算仍可能使用更高精度 |
| `max_seq_len` | `2048` | 每条序列的上下文上限，包含输入和生成部分 |
| `gpu_memory_utilization` | `0.7` | 用于规划本引擎显存预算，不是 GPU 计算利用率，也不是硬性的显存占用上限 |
| `temperature` / `max_tokens` | `0` / `128` | 贪心生成，最多生成 128 tokens；遇到结束条件可提前停止 |
| `enable_thinking` | `False` | 通过聊天模板关闭思考模式，可能减少生成长度；不代表更换推理内核 |
| `world_size` / TP / PP / DP | 均为 `1` | 单个分布式 rank，无多卡张量、流水线或数据并行加速 |
| `seed` | `0` | 随机种子；不能据此保证跨硬件、跨版本逐位一致 |
| `disable_log_stats` | `True` | 关闭周期性统计，不会关闭全部启动日志 |

`MainProcess` 是用户脚本所在进程，`EngineCore` 是执行引擎进程。`pid` 是进程号；`[文件.py:行号]` 指日志输出的位置，方便定位实现。

## 2. 采用了哪些性能优化技术

### 2.1 FlashAttention 2：实际选中的注意力后端

证据：`Using FLASH_ATTN attention backend`、`Using FlashAttention version 2`。

本次选中的是 FlashAttention 2。它通过分块计算和融合注意力操作，减少中间张量及 GPU 显存读写。日志列出的 `FLASHINFER`、`TRITON_ATTN`、`FLEX_ATTENTION` 是候选后端，不表示四种后端同时运行。

日志足以确定后端和版本，但不能确定每个阶段实际调用了哪个 CUDA 内核，也不能测出该后端相对其他后端的收益。

### 2.2 torch.compile + Inductor：实际进行了编译

证据：`mode=VLLM_COMPILE`、`backend=inductor`、`Starting torch.compile`，最终报告编译耗时 `12.94 s`。

编译器捕获模型计算图，生成优化代码，减少 Python 调度开销，并有机会融合相邻算子以减少 kernel launch 和中间结果读写。这里的收益机制来自编译路径；具体融合了哪些算子、减少了多少次访存，需要查看生成代码或 profiler，不能只凭配置断言。

主要字段：

- `Dynamo bytecode transform time: 3.78 s`：Dynamo 转换 Python 字节码阶段的耗时。
- `compile range (1, 8192)`、`6.88 s`：为相应 token 数范围编译计算图；不是本次输入有 8192 tokens，也不是上下文上限变成了 8192。
- `splitting_ops`：编译图的切分边界名单；其中出现 MLA、Mamba、MoE 等名称，不代表 Qwen3-0.6B 使用了这些结构。
- `combo_kernels=True`：开启组合内核相关编译配置；不等于日志已经证明某个具体组合内核执行。
- `collected artifacts`：收集了 29 个条目、3 个编译产物，总计 5,910,951 字节；这是编译产物大小，不是模型权重大小。
- `saved AOT compiled function`：编译结果已保存，有利于后续兼容配置复用；本次仍有编译耗时，不能描述为完整缓存命中。

`Not enough SMs to use max_autotune_gemm mode` 表示该次编译没有使用这一 GEMM 自动调优模式，并不表示 torch.compile 失败，也不能据此推断 GPU 的具体型号。

### 2.3 CUDA Graph：完成了捕获

证据：`enforce_eager=False`、`cudagraph_mode=FULL_AND_PIECEWISE`，且 PIECEWISE 与 FULL 的捕获进度均完成。

CUDA Graph 将 GPU 操作序列记录下来，后续可重放，以减少 CPU 逐个提交内核的开销。自回归生成每步计算较小，这类开销值得关注；实际收益仍需对照测量。

日志先出现一轮用于显存分析的捕获，随后又出现正式初始化阶段的捕获。第一轮报告 `0.30 GiB`，后续报告 `0.21 GiB`，不能将两者简单相加为最终图显存。最终日志明确给出图池实际使用 `0.21 GiB`，估计值 `0.34 GiB`。

`cudagraph_capture_sizes` 配置了从 1 到 512 的多个捕获规模，捕获进度显示 PIECEWISE 51 项、后续 FULL 35 项。这些数字不是实际处理了多少请求。该日志证明捕获成功，但没有逐步重放统计，不能确定每一步都走 FULL 图。

### 2.4 KV cache：缓存历史 token 的注意力键和值

证据：`Using LBNHC KV cache layout`、`Available KV cache memory: 8.89 GiB`、`GPU KV cache size: 83,232 tokens`。

KV cache 避免生成每个新 token 时重新计算全部历史 token 的 K/V。日志中的 `LBNHC` 是缓存内存布局名称，描述层、缓存块、块内 token、KV head 和 head 通道维度的组织方式；布局会影响后端访问数据的方式。它不表示启用了量化。

结合本仓库的块式 KV cache 管理实现，可理解为使用 vLLM 的分页缓存机制；但这条日志本身没有展示块大小、碎片率或具体分页注意力内核名，不能把它等同于某个名为 PagedAttention 的内核已经执行。

`kv_cache_dtype=auto` 表示由配置推导缓存类型；在此次 FP16、未指定 KV 量化的配置下，按模型精度使用缓存。日志没有启用 FP8/INT8 KV cache 的证据。

### 2.5 前缀缓存：开启，但没有命中证据

证据：`enable_prefix_caching=True`。

多个请求具有相同的可复用前缀时，可复用前缀对应的 KV，减少重复 prefill。本次只有一次聊天请求，没有命中率记录，因此只能确认功能开启，不能声称它加速了本次请求。

### 2.6 Chunked Prefill：开启，但短请求不一定分块

证据：`enable_chunked_prefill=True`，以及 `max_num_batched_tokens=8192`。

分块预填充允许将长 prompt 的 prefill 分成多个调度步骤，便于与 decode 等工作共同调度。`8192` 是调度步的批量 token 预算，不是单请求长度；单请求上限仍是 2048。

本次只有一个短 prompt，无证据表明实际拆成了多个 prefill 块，也没有持续并发流量来体现连续批处理的收益。

### 2.7 FP16：实际使用，但不是低比特量化

证据：`dtype=torch.float16`、`Casting torch.bfloat16 to torch.float16`，模型加载占用约 `1.12 GiB`。

FP16 相比 FP32 通常可减少权重存储和数据传输量，并允许硬件使用相应的半精度计算路径。但本模型原始权重是 BF16，BF16 与 FP16 都是 16 位，所以这次类型转换本身不能带来“权重减半”的收益。Tensor Core 是否被具体内核使用，需要进一步分析。

`quantization=None`、`quantization_config=None` 表明没有配置 AWQ、GPTQ、FP8、NVFP4 或 SVDQuant 等权重量化。

### 2.8 采样、预热和 CPU 线程调整

日志显示 `Using FlashInfer for top-p & top-k sampling`，随后还构建了 `sampling` JIT 模块。这说明采样后端已初始化，**不表示注意力后端也换成了 FlashInfer**。

本请求 `temperature=0`。本仓库采样实现对全贪心请求直接返回 argmax 结果，因此不能把初始化时的 top-p/top-k 采样日志当作该请求实际使用随机采样的证据。JIT 模块构建仍可能发生在初始化或预热阶段。

`JIT kernel warmup`、`Initial profiling/warmup run` 将部分准备工作提前到服务请求前；`JIT monitor ... mode=warn` 用于提示推理期间出现的新编译，本身不是加速算法。配置中的 `enable_flashinfer_autotune=True`、`enable_cutedsl_warmup=True` 也不能证明所有相关内核都实际运行。

`Reducing Torch threads from 10 to 1` 是将 Torch CPU 线程数调低，目的是避免 GPU 推理场景中的 CPU 线程竞争；是否更快需要结合工作负载验证。

## 3. 没有启用或无法确认的优化

| 字段或日志 | 可以得出的结论 |
| --- | --- |
| `speculative_config=None` | 没有启用推测解码 |
| TP、PP、DP 均为 1 | 没有多卡并行收益；初始化 NCCL 不代表发生了多卡通信 |
| `disable_custom_all_reduce=False` | 没有禁用该能力，但单 rank 不需要跨卡 all-reduce |
| `pass_config` 中列出的融合开关均为 `False` | 这些指定融合 pass 未开启；不能进一步推断所有算子融合均被禁用 |
| `custom_ops=['none']` | 编译配置中的自定义算子策略；不等于进程不使用 CUDA 扩展或注意力专用内核 |
| `rms_norm=['native']` 等 IR 优先级 | 算子实现的选择优先级；其中 `native` 不能直接解释为“在 CPU 上计算” |
| `gelu_and_mul_sparse=['triton', 'native']` | 通用算子候选优先级，不证明当前 Qwen3 使用稀疏 GELU 或 MoE |
| `monokernel=False` | 未开启该单内核执行模式 |
| `compile_mm_encoder=False` 等 | 未开启这些多模态编码器优化，本次本来也是纯文本测试 |
| Auto-prefetch disabled | 没有自动预取权重，原因是 `FUSEBLK` 未被识别为 NFS/Lustre 等网络文件系统 |

## 4. 显存字段应如何理解

| 日志数值 | 含义 |
| --- | --- |
| checkpoint `1.40 GiB` | 磁盘上的检查点大小，不等于 GPU 权重占用 |
| Available RAM `22.87 GiB` | 加载时可用的主机内存，不是显存 |
| Model loading `1.12 GiB` | 模型加载阶段记录的显存增量 |
| 启动时空闲/总显存 `15.13 / 15.58 GiB` | GPU 初始化时的显存状态 |
| 目标预算 `10.91 GiB` | 总显存约乘以 `0.7` |
| consumed memory `1.35 GiB` | 后续统计的权重与非 Torch 占用，统计口径比模型加载增量更宽 |
| peak activation `0.66 GiB` | 分析阶段记录的峰值激活占用 |
| 最终 CUDA Graph `0.21 GiB` | 实际图池占用 |
| 当前 KV cache `8.89 GiB` | 当前已规划并使用的 KV 缓存容量，不代表本次短请求消耗了这么多有效历史 token |

日志建议将 KV cache 显式设为 `8.54 GiB` 来更贴近此次目标预算；如果充分使用空闲显存则给出 `12.76 GiB`。这些是运行时估算建议，不是必须照搬的设置。`0.7` 不应理解为严格显存配额，本次日志也说明了估计与实际占用的差异。

`Maximum concurrency ... 40.64x` 来自 `83,232 / 2,048 ≈ 40.64`，是按每请求 2048 tokens 估算的 KV 容量比例。它不是实测并发吞吐量，也不是保证能同时稳定服务 40 或 41 个请求；还受调度限制、工作区和实际序列长度影响。

日志中的 `0.6782`、`0.7218` 用于说明启用 CUDA Graph 显存分析后，与旧统计方式的预算换算，不是 GPU 计算利用率或加速比。图池估计误差的 `65.7%` 也不是整卡显存浪费率。

## 5. 启动时间和推理速度

模型初始化耗时 `2.70 s`，权重加载 `2.45 s`，模型加载阶段整体 `6.46 s`。这些阶段存在包含关系，不能全部相加。引擎报告初始化耗时 `75.37 s`，其中编译 `12.94 s`，其余还包含显存分析、KV cache 创建、预热、图捕获等工作。

从 `20:43:05` 开始构建 FlashInfer sampling JIT 模块，到 `20:44:00` 出现后续图捕获完成日志，中间约 55 秒。这个时间段包含 JIT 等初始化工作，日志没有单独给出 sampling 编译耗时，因此不能将全部 55 秒归因于它。

末尾进度条：

```text
1/1 [00:00<00:00, 7.47it/s,
     est. speed input: 157.44 toks/s, output: 194.89 toks/s]
```

这说明一个请求完成；`7.47it/s` 对应本次处理阶段约 `0.134 s` 的平均完成时间。输入/输出 token 速度是进度条对该处理阶段的估算，不是独立测量的 prefill 吞吐量、逐 token decode 速度，也不包含前面的完整冷启动开销。

所以，“输出约 195 tokens/s”可以作为这一次短请求的观察值，但不能据此宣称长期服务吞吐量为 195 tokens/s。日志没有提供 TTFT、TPOT、P50/P99、多用户并发测试或回答准确率。

## 6. 警告与磁盘缓存问题

大量 BF16→FP16 的权重警告来自明确设置的 `dtype="float16"`。这次最终生成成功，说明它们没有阻止运行；但类型转换可能影响数值范围和精度。改用 BF16 前需确认显卡和后端支持，并以同一精度做性能比较。

`torch._opaque_base`、`register_opaque_type`、`cuda.cudart`、`cuda.nvrtc` 的 deprecated 警告表示相关接口计划迁移，此日志没有显示它们导致异常。

本次编译缓存实际写入：

```text
/home/l8w/.cache/vllm/torch_compile_cache/...
```

虽然分布式初始化临时文件已经位于 `/media/l8w/Linux118/.cache/vllm-install/tmp`，**uv 安装缓存的位置不会自动决定 vLLM 运行时编译缓存的位置**。为避免继续占用根分区，启动 Python 前可显式设置：

```bash
export XDG_CACHE_HOME="/media/l8w/Linux118/.cache"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv/ComputeCache"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
```

这不会迁移或删除已有缓存，也不保证重定向所有第三方库的缓存。改变缓存位置后若新位置没有兼容产物，首次运行可能重新编译。

最后的 SIGTERM 出现在模型回答完成、脚本退出之后，前面明确记录了主动关闭引擎。虽然客户端又输出 `engine core exited unexpectedly` 警告，但本日志没有推理失败的 traceback，不能据此判定为推理崩溃；若长期服务期间提前出现同类退出，则需另查原因。

## 7. 下一步如何评估优化收益

复用同一个 `LLM` 实例执行预热和多次请求，分别记录冷启动与预热后的性能。通过 `enforce_eager=True` 做对照可以观察关闭编译/图执行路径后的整体差异，但不能只将差异归因于 CUDA Graph。前缀缓存需使用重复前缀并比较命中情况；分块预填充和连续批处理需使用长输入与并发请求。

本次最值得继续测量的是编译与图执行对小模型单请求延迟的影响，以及并发请求时的吞吐量。若要确定具体融合算子、Tensor Core 使用和内存瓶颈，再使用 Nsight Systems/Compute 查看时间线与内核指标。

解释中涉及的本地实现可参考：[采样分支](../../vllm/v1/sample/sampler.py)、[运行时缓存路径](../../vllm/envs.py)、[CUDA Graph 显存分析](../../vllm/v1/worker/gpu_worker.py)。
