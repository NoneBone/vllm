<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
简单、快捷、便宜的LLM服务，人人适用
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>文档</b></a> | <a href="https://blog.vllm.ai/"><b>博客</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>纸</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>用户论坛</b></a> | <a href="https://slack.vllm.ai"><b>开发者 Slack</b></a> |
</p>

🔥 我们已建立 vLLM 网站，帮助您快速上手 vLLM。请访问 [vllm.ai](https://vllm.ai) 了解更多信息。
活动详情请访问 [vllm.ai/events](https://vllm.ai/events) 加入我们。

---

## 关于

vLLM 是一个快速易用的 LLM 推理和服务库。

最初在 [天空计算实验室](https://sky.cs.berkeley.edu) 在加州大学伯克利分校，vLLM 已发展成为最活跃的开源人工智能项目之一，由来自 2000 多名贡献者的数十个学术机构和公司组成的多元化社区构建和维护。

vLLM 速度很快，具体表现为：

- 最先进的服务吞吐量
- 高效管理注意力、关键信息和价值记忆 [**已翻页**](https://blog.vllm.ai/2023/06/20/vllm.html)
- 连续批量处理传入请求、分块预填充、前缀缓存
- 使用分段和完整 CUDA/HIP 图快速灵活地执行模型
- 量化方式：FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、压缩张量、ModelOpt、TorchAO 等 [更多的](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- 优化的注意力内核包括 FlashAttention、FlashInfer、TRTLLM-GEN、FlashMLA 和 Triton。
- 使用 CUTLASS、TRTLLM-GEN 和 CuTeDSL 针对不同精度优化了 GEMM/MoE 内核
- 推测性解码，包括 n-gram、后缀、EAGLE、DFlash
- 使用 torch.compile 自动生成内核并进行图级转换
- 分解式预填充、解码和编码

vLLM 灵活易用，可与以下设备配合使用：

- 与流行的 Hugging Face 模型无缝集成
- 采用多种解码算法实现高吞吐量服务，包括*并行采样*、*波束搜索*等。
- 用于分布式推理的张量、管道、数据、专家和上下文并行性
- 流媒体输出
- 使用 xgrammar 或指南生成结构化输出
- 工具调用和推理解析器
- 兼容 OpenAI 的 API 服务器，以及 Anthropic Messages API 和 gRPC 支持
- 高效的多层RA支持，适用于致密层和MoE层
- 支持 NVIDIA GPU、AMD GPU、Intel GPU 以及 x86/ARM/PowerPC CPU。此外，还支持多种硬件插件，例如 Google TPU、Intel Gaudi、IBM Spyre、华为 Ascend、Rebellions NPU、Apple Silicon、MetaX GPU 等。

vLLM 无缝支持 Hugging Face 上的 200 多种模型架构，包括：

- 仅解码器 LLM（例如，Llama、Qwen、Gemma）
- 混合专家学习学习模型（例如 Mixtral、DeepSeek-V3、Qwen-MoE、GPT-OSS）
- 混合注意力与状态空间模型（例如，Mamba、Qwen3.5）
- 多模态模型（例如，LLaVA、Qwen-VL、Pixtral）
- 嵌入和检索模型（例如，E5-Mistral、GTE、ColBERT）
- 奖励和分类模型（例如，Qwen-Math）

查找所有支持的型号列表 [这里](https://docs.vllm.ai/en/latest/models/supported_models.html)。

## 入门

使用以下方式安装 vLLM [`uv`](https://docs.astral.sh/uv/) （推荐）或 `pip`：

```bash
uv pip install vllm

# 创建虚拟环境
uv venv --python 3.12 --seed --managed-python
# 激活
source .venv/bin/activate
# 安装 torch
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# 依赖安装
grep -v '^torch==' requirements/build/cuda.txt | uv pip install -r -
```

或者 [从源代码构建](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) 用于发展。

访问我们的网站 [文档](https://docs.vllm.ai/en/latest/) 了解更多信息。

- [安装](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [快速入门](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [支持的型号列表](https://docs.vllm.ai/en/latest/models/supported_models.html)

## 贡献

我们欢迎并重视任何形式的贡献与合作。
请查看 [为 vLLM 做出贡献](https://docs.vllm.ai/en/latest/contributing/index.html) 了解如何参与。

## 引用

如果您在研究中使用 vLLM，请引用我们的数据。 [纸](https://arxiv.org/abs/2309.06180)：

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 联系我们

<!-- --8<-- [start:contact-us] -->
- 如有技术问题或功能请求，请使用 GitHub。 [问题](https://github.com/vllm-project/vllm/issues)
- 如需与其他用户讨论，请使用 [vLLM论坛](https://discuss.vllm.ai)
- 为了协调贡献和开发，请使用 [松弛](https://slack.vllm.ai)
- 对于安全披露，请使用 GitHub 的 [安全公告](https://github.com/vllm-project/vllm/security/advisories) 特征
- 如需洽谈合作或伙伴关系，请联系我们： [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## 媒体资料包

- 如果您想使用 vLLM 的徽标，请参考以下内容： [我们的媒体资料库](https://github.com/vllm-project/media-kit)
