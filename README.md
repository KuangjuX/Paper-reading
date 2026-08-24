<h1 align="center">📚 Paper Reading</h1>

<p align="center">
  <em>A curated collection of research papers on AI systems, compilers, architecture, and systems software.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Papers-260-blue?style=flat-square" alt="Papers">
  <img src="https://img.shields.io/badge/Read-70-green?style=flat-square" alt="Read">
  <img src="https://img.shields.io/badge/To_Read-190-orange?style=flat-square" alt="To Read">
</p>

---

## Table of Contents

- [Deep Learning Compiler](#-deep-learning-compiler)
- [LLM Inference](#-llm-inference)
- [LLM Training](#-llm-training)
- [Deep Learning](#-deep-learning)
- [LLM Evaluation & Safety](#-llm-evaluation--safety)
- [LLM for Kernel Optimization](#-llm-for-kernel-optimization)
- [Agent Systems](#-agent-systems)
- [GPU Microarchitecture](#-gpu-microarchitecture)
- [Math Foundations](#-math-foundations)
- [Compiler](#-compiler)
- [Operating Systems](#-operating-systems)
- [Hypervisor & Virtualization](#-hypervisor--virtualization)
- [RISC-V](#-risc-v)

> **Legend:** ✅ = Read &nbsp;|&nbsp; ⬜ = To Read &nbsp;|&nbsp; 📝 = Note Available

> **Learning roadmap:** [LLM Training, Inference & Algorithms — Recommended Reading Path](notes/llm/learning-path.md)

---

## 🔧 Deep Learning Compiler

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **The Deep Learning Compiler: A Comprehensive Survey** | — | [Paper](https://arxiv.org/pdf/2002.03794.pdf) / [Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/ai-compiler/The-Deep-Learning-Compiler-A-Comprehensive-Survey.md) |
| ✅ | **MLIR: Scaling Compiler Infrastructure for Domain Specific Computation** | CGO'21 | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9370308) / [Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/ai-compiler/MLIR-Scaling-Compiler-Infrastructure-for-Domain-Specific-Computation.md) |
| ✅ | **TIRAMISU: A Polyhedral Compiler for Expressing Fast and Portable Code** | CGO'19 | [Paper](papers/mlsys/polyhedral/Tiramisu-CGO.pdf) / [Note](notes/compiler/tiramisu-cgo/tiramisu.md) |
| ✅ | **Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks** | OSDI'20 | [Paper](https://www.usenix.org/system/files/osdi20-ma.pdf) / [Note](https://github.com/KuangjuX/paper-reading/issues/22) |
| ✅ | **ROLLER: Fast and Efficient Tensor Compilation for Deep Learning** | OSDI'22 | [Paper](https://www.usenix.org/system/files/osdi22-zhu.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/24) |
| ✅ | **BOLT: Bridging The Gap Between Auto-Tuners and Hardware-Native Performance** | MLSys'22 | [Paper](http://yibozhu.com/doc/bolt-mlsys22.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/28) |
| ✅ | **AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures** | ASPLOS'22 | [Paper](https://dl.acm.org/doi/10.1145/3503222.3507723) / [Note](https://github.com/KuangjuX/Paper-reading/issues/26) |
| ✅ | **AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction** | ISCA'22 | [Paper](papers/mlsys/IR/AMOS-ISCA.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/31) |
| ✅ | **Welder: Scheduling Deep Learning Memory Access via Tile-graph** | OSDI'23 | [Paper](https://www.usenix.org/system/files/osdi23-shi.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/25) |
| ✅ | **Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators** | OSDI'23 | [Paper](papers/mlsys/soft-hard-co-design/osdi23-zhao.pdf) |
| ✅ | **Cocktailer: Analyzing and Optimizing Dynamic Control Flow in Deep Learning** | OSDI'23 | [Paper](https://www.usenix.org/system/files/osdi23-zhang-chen.pdf) / [Note](https://github.com/KuangjuX/paper-reading/issues/21) |
| ✅ | **Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion** | HPCA'23 | [Paper](papers/mlsys/fusion/Chimera_An_Analytical_Optimizing_Framework_for_Effective_Compute-intensive_Operators_Fusion.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/30) |
| ✅ | **Graphene: An IR for Optimized Tensor Computations on GPUs** | ASPLOS'23 | [Paper](https://dl.acm.org/doi/pdf/10.1145/3582016.3582018) / [Note](https://github.com/KuangjuX/Paper-reading/issues/27) |
| ✅ | **Uncovering Nested Data Parallelism and Data Reuse in DNN Computation with FractalTensor** | SOSP'24 | [Paper](https://dl.acm.org/doi/10.1145/3694715.3695961) |
| ✅ | **ThunderKittens: Simple, Fast, and Adorable AI Kernels** | — | [Paper](papers/mlsys/ThunderKittens.pdf) |
| ✅ | **Mirage: A Multi-Level Superoptimizer for Tensor Programs** | OSDI'25 | [Paper](https://www.usenix.org/system/files/osdi25-wu-mengdi.pdf) |
| ✅ | **PipeThreader: Software-Defined Pipelining for Efficient DNN Execution** | OSDI'25 | [Paper](https://www.usenix.org/system/files/osdi25-cheng.pdf) |
| ✅ | **TileLang: A Composable Tiled Programming Model for AI Systems** | — | [Paper](https://arxiv.org/pdf/2504.17577) |
| ✅ | **Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References** | arXiv'25 | [Paper](https://arxiv.org/pdf/2510.14719) |
| ✅ | **KPerfIR: Towards an Open and Compiler-centric Ecosystem for GPU Kernel Performance Tooling on Modern AI Workloads** | OSDI'25 | [Paper](https://www.usenix.org/system/files/osdi25-guan.pdf) |

## 🚀 LLM Inference

### Decoding Algorithms

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **The Curious Case of Neural Text Degeneration** | 系统比较 greedy、beam、top-k 与 nucleus sampling，理解生成质量、随机性和退化 | [Paper](https://arxiv.org/abs/1904.09751) |
| ⬜ | **Fast Inference from Transformers via Speculative Decoding** | draft–verify 与拒绝采样；在保持目标分布不变的前提下并行生成多个 token | [Paper](https://proceedings.mlr.press/v202/leviathan23a.html) |
| ⬜ | **Accelerating Large Language Model Decoding with Speculative Sampling** | speculative sampling 的独立推导与工程验证，适合和上一论文对照阅读 | [Paper](https://arxiv.org/abs/2302.01318) |
| ⬜ | **SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification** | 用候选树提高并行验证宽度，连接解码算法与 serving batch | [Paper](https://arxiv.org/abs/2305.09781) |
| ⬜ | **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads** | 不依赖独立 draft model 的多头预测与 tree attention | [Paper](https://arxiv.org/abs/2401.10774) |
| ⬜ | **Break the Sequential Dependency of LLM Inference Using Lookahead Decoding** | 从 Jacobi iteration 理解无额外模型的并行候选生成 | [Paper](https://arxiv.org/abs/2402.02057) |
| ⬜ | **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty** | 在 feature space 自回归预测，理解高接受率 drafter 的训练方法 | [Paper](https://arxiv.org/abs/2401.15077) |
| ⬜ | **Better & Faster Large Language Models via Multi-token Prediction** | 训练时预测多个未来 token，将训练目标与推理解码并行性连接起来 | [Paper](https://arxiv.org/abs/2404.19737) |

### General

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **A Survey of LLM Inference Systems** | — | [Paper](https://arxiv.org/pdf/2506.21901) / [Note](https://github.com/KuangjuX/Notes/blob/main/PaperNotes/LLM/Inference/llm_inference_survey.pdf) |
| ⬜ | **WaferLLM: Large Language Model Inference at Wafer Scale** | OSDI'25 | [Paper](https://www.usenix.org/system/files/osdi25-he.pdf) |

### Long Context Inference

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Training-Free Long-Context Scaling of Large Language Models** | ICML'24 | [Paper](https://arxiv.org/pdf/2402.17463) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/DCA.md) |
| ✅ | **Efficient Streaming Language Models with Attention Sinks** | ICLR'24 | [Paper](https://arxiv.org/pdf/2309.17453) |
| ✅ | **Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference** | ICML'24 | [Paper](https://arxiv.org/pdf/2406.10774) |
| ✅ | **DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads** | ICLR'25 | [Paper](https://arxiv.org/pdf/2410.10819v1) |
| ✅ | **MiniMax Sparse Attention** | arXiv'26 | [Paper](https://arxiv.org/abs/2606.13392) / [Note](notes/llm/minimax-msa/msa.md) |

### LLM Serving

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ⬜ | **Orca: A Distributed Serving System for Transformer-Based Generative Models** | OSDI'22 | [Paper](https://www.usenix.org/conference/osdi22/presentation/yu) |
| ⬜ | **Efficient Memory Management for Large Language Model Serving with PagedAttention** | SOSP'23 | [Paper](https://arxiv.org/abs/2309.06180) |
| ⬜ | **FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU** | ICML'23 | [Paper](https://proceedings.mlr.press/v202/sheng23a.html) |
| ⬜ | **AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving** | OSDI'23 | [Paper](https://www.usenix.org/conference/osdi23/presentation/li-zhouhan) |
| ⬜ | **FastServe: Fast Distributed Inference Serving for Large Language Models** | arXiv'23 | [Paper](https://arxiv.org/abs/2305.05920) |
| ⬜ | **InferCept: Efficient Interleaving of Inference and Decoding for LLM Serving** | ICML'24 | [Paper](https://arxiv.org/abs/2307.07694) |
| ⬜ | **Punica: Multi-Tenant LoRA Serving** | MLSys'24 | [Paper](https://arxiv.org/abs/2310.18547) |
| ⬜ | **S-LoRA: Serving Thousands of Concurrent LoRA Adapters** | MLSys'24 | [Paper](https://arxiv.org/abs/2311.03285) |
| ⬜ | **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** | ISCA'24 | [Paper](https://arxiv.org/abs/2311.18677) |
| ⬜ | **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** | OSDI'24 | [Paper](https://arxiv.org/abs/2403.02310) |
| ⬜ | **Llumnix: Dynamic Scheduling for Large Language Model Serving** | OSDI'24 | [Paper](https://arxiv.org/abs/2406.03243) |
| ⬜ | **Preble: Efficient Distributed Prompt Scheduling for LLM Serving** | arXiv'24 | [Paper](https://arxiv.org/abs/2407.00023) |
| ✅ | **SGLang: Efficient Execution of Structured Language Model Programs** | — | [Paper](papers/mlsys/sglang.pdf) |
| ✅ | **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving** | — | [Paper](papers/mlsys/2501.01005v1.pdf) |
| ⬜ | **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** | OSDI'24 | [Paper](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) |
| ⬜ | **LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism** | SOSP'24 | [Paper](https://dl.acm.org/doi/pdf/10.1145/3694715.3695948) |
| ⬜ | **Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot** | FAST'25 | [Paper](https://www.usenix.org/system/files/fast25-qin.pdf) |
| ⬜ | **NanoFlow: Towards Optimal Large Language Model Serving Throughput** | OSDI'25 | [Paper](https://www.usenix.org/system/files/osdi25-zhu-kan.pdf) |

### Quantization & Compression

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale** | 混合精度分解 activation outlier，是理解 LLM INT8 的起点 | [Paper](https://arxiv.org/abs/2208.07339) |
| ⬜ | **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** | 基于近似二阶信息的 one-shot weight-only PTQ | [Paper](https://arxiv.org/abs/2210.17323) |
| ⬜ | **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** | 用等价缩放把 activation 量化难度迁移到 weight，形成 W8A8 路线 | [Paper](https://arxiv.org/abs/2211.10438) |
| ⬜ | **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** | 用 activation 识别显著通道并保护关键权重，形成主流 W4A16 路线 | [Paper](https://arxiv.org/abs/2306.00978) |
| ⬜ | **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression** | 将少量 outlier 与低比特权重分离，理解稀疏异常值处理 | [Paper](https://arxiv.org/abs/2306.03078) |
| ⬜ | **OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models** | 通过可学习的 clipping 与等价变换改善低比特 PTQ | [Paper](https://arxiv.org/abs/2308.13137) |
| ⬜ | **AQLM: Extreme Compression of Large Language Models via Additive Quantization** | additive codebook 与极低 bit/weight，连接压缩算法和解码 kernel | [Paper](https://arxiv.org/abs/2401.06118) |
| ⬜ | **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** | 区分 K/V 的通道与 token 统计特性，理解 KV cache 低比特化 | [Paper](https://arxiv.org/abs/2402.02750) |
| ⬜ | **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** | 用 Hadamard rotation 消除 outlier，适合从算法追到融合 kernel | [Paper](https://arxiv.org/abs/2404.00456) |
| ⬜ | **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits** | BitNet b1.58 与三值权重，理解量化感知训练的架构路线 | [Paper](https://arxiv.org/abs/2402.17764) |

### MegaKernel

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B** | Blog | [Paper](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) |
| ✅ | **Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs** | arXiv'25 | [Paper](https://arxiv.org/pdf/2512.22219) |
| ✅ 📝 | **Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel** | MLSys'26 | [Paper](https://arxiv.org/abs/2604.13327) / [Note](notes/llm/event-tensor/event-tensor.md) |
| ✅ | **TileRT: Tile-Based Runtime for Ultra-Low-Latency LLM Inference** | — | [Paper](https://github.com/tile-ai/TileRT) |
| ✅ | **SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations** | arXiv'25 | [Paper](https://arxiv.org/pdf/2512.14080) |
| ✅ 📝 | **MegaMoE** (fused MoE megakernel, DeepGEMM) | DeepGEMM PR | [Source](https://github.com/deepseek-ai/DeepGEMM/pull/304) / [Note](notes/llm/megamoe/megamoe.md) |
| ✅ | **Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference** | Blog | [Paper](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17) |

**MegaMoE 摘要（非正式论文，见 DeepGEMM PR #304）**：把 MoE 前向中分发、两层分组 GEMM、SwiGLU、合并压进**单一持久化 CUDA 核**；用对称显存布局与 NVLink 在核内做专家并行词元交换，并以波次调度、L1/L2 词元池上的细粒度到达计数 / 掩码，把通信与计算流水重叠；SM100 上按分发、TMA+MMA、尾声与合并划分线程束角色与寄存器预算。

## 🏋️ LLM Training

### Training Numerics & Memory

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour** | global batch、线性学习率缩放与 warmup；理解数据并行扩展为何会改变优化行为 | [Paper](https://arxiv.org/abs/1706.02677) |
| ⬜ | **Mixed Precision Training** | FP16 master weights、loss scaling 与数值范围；连接 Tensor Core 吞吐和收敛 | [Paper](https://arxiv.org/abs/1710.03740) |
| ⬜ | **Training Deep Nets with Sublinear Memory Cost** | activation checkpointing/rematerialization 的经典计算–显存交换 | [Paper](https://arxiv.org/abs/1604.06174) |
| ⬜ | **Reducing Activation Recomputation in Large Transformer Models** | sequence parallelism 与 selective recomputation，解释 Megatron 的 activation 内存优化 | [Paper](https://arxiv.org/abs/2205.05198) |
| ⬜ | **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection** | 对梯度做低秩投影以降低 optimizer state 和训练显存 | [Paper](https://arxiv.org/abs/2403.03507) |

### Distributed Training

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ⬜ | **PyTorch Distributed: Experiences on Accelerating Data Parallel Training** | VLDB'20 | [Paper](https://www.vldb.org/pvldb/vol13/p3005-li.pdf) |
| ⬜ | **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism** | NeurIPS'19 | [Paper](https://arxiv.org/abs/1811.06965) |
| ⬜ | **PipeDream: Generalized Pipeline Parallelism for DNN Training** | SOSP'19 | [Paper](https://dl.acm.org/doi/10.1145/3341301.3359646) |
| ⬜ | **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** | arXiv'19 | [Paper](https://arxiv.org/abs/1909.08053) |
| ⬜ | **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** | SC'20 | [Paper](https://arxiv.org/abs/1910.02054) |
| ⬜ | **ZeRO-Offload: Democratizing Billion-Scale Model Training** | USENIX ATC'21 | [Paper](https://www.usenix.org/conference/atc21/presentation/ren-jie) |
| ⬜ | **Memory-Efficient Pipeline-Parallel DNN Training** | ICML'21 | [Paper](https://proceedings.mlr.press/v139/narayanan21a.html) |
| ⬜ | **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning** | SC'21 | [Paper](https://arxiv.org/abs/2104.07857) |
| ⬜ | **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** | SC'21 | [Paper](https://arxiv.org/abs/2104.04473) |
| ⬜ | **GSPMD: General and Scalable Parallelization for ML Computation Graphs** | arXiv'21 | [Paper](https://arxiv.org/abs/2105.04663) |
| ⬜ | **Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning** | OSDI'22 | [Paper](https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin) |
| ⬜ | **DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models** | arXiv'23 | [Paper](https://arxiv.org/abs/2309.14509) |
| ⬜ | **Ring Attention with Blockwise Transformers for Near-Infinite Context** | ICLR'24 | [Paper](https://arxiv.org/abs/2310.01889) |
| ⬜ | **Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates** | SOSP'23 | [Paper](https://arxiv.org/abs/2309.08125) |
| ⬜ | **MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs** | NSDI'24 | [Paper](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng) |
| ⬜ | **LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism** | — | [Paper](https://arxiv.org/pdf/2406.18485) |
| ✅ 📝 | **PithTrain: A Compact and Agent-Native MoE Training System** | arXiv'26 | [Paper](https://arxiv.org/abs/2605.31463) / [Note](notes/llm/pithtrain/pithtrain.md) / [Code Guide](notes/llm/pithtrain/pithtrain-code-guide.md) |

### Mixture-of-Experts Training

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** | 现代稀疏 MoE 的起点：top-k gate、稀疏激活与负载均衡 | [Paper](https://arxiv.org/abs/1701.06538) |
| ⬜ | **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** | 将 MoE、SPMD sharding 和大规模 Transformer 训练结合 | [Paper](https://arxiv.org/abs/2006.16668) |
| ⬜ | **BASE Layers: Simplifying Training of Large, Sparse Models** | 用 balanced assignment 避免额外负载均衡损失 | [Paper](https://arxiv.org/abs/2103.16716) |
| ⬜ | **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts** | 大规模稀疏语言模型的质量、计算与能耗权衡 | [Paper](https://arxiv.org/abs/2112.06905) |
| ⬜ | **DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale** | expert parallel、通信与 MoE inference/training 系统化设计 | [Paper](https://arxiv.org/abs/2201.05596) |
| ⬜ | **FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models** | 动态路由下的 shadowing 与拓扑感知通信优化 | [Paper](https://arxiv.org/abs/2202.11436) |
| ⬜ | **Tutel: Adaptive Mixture-of-Experts at Scale** | 自适应并行、all-to-all 与 kernel 优化的完整 MoE 系统 | [Paper](https://arxiv.org/abs/2206.03382) |
| ⬜ | **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts** | 将 token dropping 问题转成 block-sparse GEMM，最贴近算子视角 | [Paper](https://arxiv.org/abs/2211.15841) |

### RL Training

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning** | arXiv'25 | [Paper](https://arxiv.org/pdf/2511.14617) |

### Fine-Tuning & Alignment

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Fine-Tuning Language Models from Human Preferences** | 将 reward model 与 PPO 用于语言模型偏好优化的早期完整方案 | [Paper](https://arxiv.org/abs/1909.08593) |
| ⬜ | **Learning to Summarize from Human Feedback** | 展示偏好数据、reward model 和 RL 在真实生成任务中的规模化 | [Paper](https://arxiv.org/abs/2009.01325) |
| ⬜ | **Prefix-Tuning: Optimizing Continuous Prompts for Generation** | 冻结主模型，仅训练可学习 prefix 的参数高效微调 | [Paper](https://arxiv.org/abs/2101.00190) |
| ⬜ | **The Power of Scale for Parameter-Efficient Prompt Tuning** | soft prompt 与模型规模关系，适合理解 PEFT 的表达能力 | [Paper](https://arxiv.org/abs/2104.08691) |
| ⬜ | **LoRA: Low-Rank Adaptation of Large Language Models** | 冻结权重并注入低秩更新，连接矩阵秩、训练显存和多租户推理 | [Paper](https://arxiv.org/abs/2106.09685) |
| ⬜ | **Proximal Policy Optimization Algorithms** | RLHF 所需的最低限度策略优化基础：ratio、advantage 与 clipping | [Paper](https://arxiv.org/abs/1707.06347) |
| ⬜ | **Finetuned Language Models Are Zero-Shot Learners** | FLAN 与 instruction tuning，说明任务混合如何产生泛化能力 | [Paper](https://arxiv.org/abs/2109.01652) |
| ⬜ | **Training Language Models to Follow Instructions with Human Feedback** | InstructGPT 的 SFT → reward model → PPO 三阶段链路 | [Paper](https://arxiv.org/abs/2203.02155) |
| ⬜ | **Constitutional AI: Harmlessness from AI Feedback** | self-critique、revision 与 RLAIF，把安全原则引入后训练 | [Paper](https://arxiv.org/abs/2212.08073) |
| ⬜ | **Self-Instruct: Aligning Language Models with Self-Generated Instructions** | 自生成、过滤和扩增指令数据的经典流程 | [Paper](https://arxiv.org/abs/2212.10560) |
| ⬜ | **The Flan Collection: Designing Data and Methods for Effective Instruction Tuning** | 系统研究任务混合、模板、CoT 数据与 instruction tuning 配方 | [Paper](https://arxiv.org/abs/2301.13688) |
| ⬜ | **QLoRA: Efficient Finetuning of Quantized LLMs** | NF4、double quantization、paged optimizer 与 LoRA 的组合 | [Paper](https://arxiv.org/abs/2305.14314) |
| ⬜ | **LIMA: Less Is More for Alignment** | 少量高质量监督数据与大规模预训练知识之间的分工 | [Paper](https://arxiv.org/abs/2305.11206) |
| ⬜ | **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** | 将显式 reward model + PPO 化为稳定的 pairwise classification loss | [Paper](https://arxiv.org/abs/2305.18290) |
| ⬜ | **RRHF: Rank Responses to Align Language Models with Human Feedback** | 用候选排序损失统一多种反馈来源 | [Paper](https://arxiv.org/abs/2304.05302) |
| ⬜ | **A General Theoretical Paradigm to Understand Learning from Human Preferences** | IPO 与偏好优化的理论视角，理解 DPO 类方法的过拟合和正则化 | [Paper](https://arxiv.org/abs/2310.12036) |
| ⬜ | **KTO: Model Alignment as Prospect Theoretic Optimization** | 只需 desirable/undesirable 标签的非成对偏好学习 | [Paper](https://arxiv.org/abs/2402.01306) |
| ⬜ | **ORPO: Monolithic Preference Optimization without Reference Model** | 将 SFT 与偏好约束合并，移除独立 reference model | [Paper](https://arxiv.org/abs/2403.07691) |
| ⬜ | **SimPO: Simple Preference Optimization with a Reference-Free Reward** | reference-free、长度归一化 reward 与 margin objective | [Paper](https://arxiv.org/abs/2405.14734) |
| ⬜ | **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** | GRPO、数学数据与可验证奖励，是 reasoning RL 的关键前置 | [Paper](https://arxiv.org/abs/2402.03300) |
| ⬜ | **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** | 大规模 reasoning RL、冷启动数据与蒸馏路线 | [Paper](https://arxiv.org/abs/2501.12948) |

### Compute-Communication Overlap

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Flux: Fast Software-based Communication Overlap on GPUs through Kernel Fusion** | — | [Paper](https://arxiv.org/pdf/2406.06858v1) |
| ✅ | **DeepEP: An Efficient Expert-Parallel Communication Library** | — | [Paper](https://github.com/deepseek-ai/DeepEP) |
| ⬜ | **Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning** | ASPLOS'24 | [Paper](https://dl.acm.org/doi/pdf/10.1145/3620666.3651379) |
| ⬜ | **Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts** | — | [Paper](https://arxiv.org/pdf/2502.19811) |
| ⬜ | **TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives** | MLSys'25 | [Paper](https://arxiv.org/pdf/2503.20313) |
| ⬜ | **Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler** | — | [Paper](https://arxiv.org/pdf/2504.19442) |
| ⬜ | **FlashOverlap: A Lightweight Design for Efficiently Overlapping Communication and Computation** | EuroSys'25 | [Paper](https://arxiv.org/pdf/2504.19519) |
| ⬜ | **TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference** | — | [Paper](https://arxiv.org/pdf/2505.11329) |

## 🧠 Deep Learning

### Foundations & Optimization

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Learning Representations by Back-Propagating Errors** | 从局部算子 backward 上升到计算图链式法则、梯度流和 activation 保存 | [Paper](https://www.nature.com/articles/323533a0) |
| ⬜ | **Understanding the Difficulty of Training Deep Feedforward Neural Networks** | Xavier initialization 与方差传播，理解初始化为何影响深层网络稳定性 | [Paper](https://proceedings.mlr.press/v9/glorot10a.html) |
| ⬜ | **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** | 经典正则化、train/eval 行为差异与随机 mask | [Paper](https://jmlr.org/papers/v15/srivastava14a.html) |
| ⬜ | **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** | 对比 batch statistics 与 LayerNorm，理解同步 BN 和训练/推理差异 | [Paper](https://proceedings.mlr.press/v37/ioffe15.html) |
| ⬜ | **Deep Residual Learning for Image Recognition** | 残差连接与深层梯度传播，是 Transformer residual stream 的结构前置 | [Paper](https://arxiv.org/abs/1512.03385) |
| ⬜ | **Adam: A Method for Stochastic Optimization** | 一阶/二阶矩、bias correction 与 optimizer state 显存 | [Paper](https://arxiv.org/abs/1412.6980) |
| ⬜ | **Decoupled Weight Decay Regularization** | 区分 L2 regularization 与 AdamW 的 decoupled weight decay | [Paper](https://arxiv.org/abs/1711.05101) |
| ⬜ | **Layer Normalization** | 单样本归一化、训练稳定性与 Transformer 中的 reduction/fusion | [Paper](https://arxiv.org/abs/1607.06450) |
| ⬜ | **Root Mean Square Layer Normalization** | 省去 re-centering 的 RMSNorm，连接现代 LLM 结构与高效 kernel | [Paper](https://arxiv.org/abs/1910.07467) |
| ⬜ | **Adafactor: Adaptive Learning Rates with Sublinear Memory Cost** | 对二阶矩做 factored approximation，理解 optimizer memory 优化 | [Paper](https://proceedings.mlr.press/v80/shazeer18a.html) |
| ⬜ | **Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes** | LAMB 的 layer-wise scaling 与超大 batch 训练 | [Paper](https://arxiv.org/abs/1904.00962) |
| ⬜ | **Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer** | μP 与跨模型规模超参数迁移，连接 scaling experiment 和训练配方 | [Paper](https://arxiv.org/abs/2203.03466) |
| ⬜ | **Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training** | 低成本二阶曲率估计与 per-coordinate clipping | [Paper](https://arxiv.org/abs/2305.14342) |

### Language Modeling, Tokenization & Modern LLMs

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **A Neural Probabilistic Language Model** | embedding、条件概率与 next-token language modeling 的早期完整形式 | [Paper](https://www.jmlr.org/papers/v3/bengio03a.html) |
| ⬜ | **Sequence to Sequence Learning with Neural Networks** | 自回归分解、teacher forcing、EOS 与 beam search 的基础 | [Paper](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html) |
| ⬜ | **Neural Machine Translation by Jointly Learning to Align and Translate** | additive attention 与 encoder–decoder alignment，理解 Transformer 之前的问题 | [Paper](https://arxiv.org/abs/1409.0473) |
| ⬜ | **Neural Machine Translation of Rare Words with Subword Units** | BPE、词表大小、序列长度和输出 softmax 成本之间的关系 | [Paper](https://aclanthology.org/P16-1162/) |
| ⬜ | **SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing** | 从 raw text 训练 BPE/unigram tokenizer，理解现代 tokenizer pipeline | [Paper](https://aclanthology.org/D18-2012/) |
| ⬜ | **Improving Language Understanding by Generative Pre-Training** | GPT-1 与 decoder-only 预训练–微调范式 | [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| ⬜ | **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** | encoder-only、masked LM 与 pretrain–finetune 范式 | [Paper](https://aclanthology.org/N19-1423/) |
| ⬜ | **Language Models are Unsupervised Multitask Learners** | GPT-2、zero-shot transfer 与 WebText 数据路线 | [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| ⬜ | **Language Models are Few-Shot Learners** | GPT-3、in-context learning 与 decoder-only scaling | [Paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) |
| ⬜ | **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** | T5、span corruption、C4 与统一 text-to-text interface | [Paper](https://jmlr.org/papers/v21/20-074.html) |
| ⬜ | **GLU Variants Improve Transformer** | SwiGLU/GEGLU 与 gated FFN，解释现代 LLM 中三路投影和逐元素乘 | [Paper](https://arxiv.org/abs/2002.05202) |
| ⬜ | **RoFormer: Enhanced Transformer with Rotary Position Embedding** | RoPE 的旋转与相对位置语义，以及 Q/K 融合实现 | [Paper](https://arxiv.org/abs/2104.09864) |
| ⬜ | **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** | ALiBi 与无需显式位置 embedding 的长度外推 | [Paper](https://arxiv.org/abs/2108.12409) |
| ⬜ | **PaLM: Scaling Language Modeling with Pathways** | 大规模 dense LLM、Pathways 并行与训练行为分析 | [Paper](https://arxiv.org/abs/2204.02311) |
| ⬜ | **LLaMA: Open and Efficient Foundation Language Models** | 将 RMSNorm、SwiGLU、RoPE、tokenizer、数据与训练配方串成现代 LLM | [Paper](https://arxiv.org/abs/2302.13971) |
| ⬜ | **Llama 2: Open Foundation and Fine-Tuned Chat Models** | 预训练、SFT、RLHF、安全评测与 chat model 的完整技术报告 | [Paper](https://arxiv.org/abs/2307.09288) |
| ⬜ | **Mistral 7B** | sliding-window attention、GQA 与 rolling buffer KV cache | [Paper](https://arxiv.org/abs/2310.06825) |
| ⬜ | **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models** | shared expert、细粒度 expert segmentation 与稀疏计算配比 | [Paper](https://arxiv.org/abs/2401.06066) |
| ⬜ | **Mixtral of Experts** | 实用 sparse MoE LLM 的结构、路由和质量–计算权衡 | [Paper](https://arxiv.org/abs/2401.04088) |
| ⬜ | **OLMo: Accelerating the Science of Language Models** | 开放数据、训练代码、checkpoint、日志和评测的端到端案例 | [Paper](https://aclanthology.org/2024.acl-long.841/) |

### Scaling Laws, Data & Pretraining

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Scaling Laws for Neural Language Models** | 建立 loss 与参数量、数据量、训练算力之间的幂律关系 | [Paper](https://arxiv.org/abs/2001.08361) |
| ⬜ | **Training Compute-Optimal Large Language Models** | Chinchilla scaling 与 compute-optimal 参数/token 配比 | [Paper](https://arxiv.org/abs/2203.15556) |
| ⬜ | **Scaling Data-Constrained Language Models** | 数据受限和重复 epoch 下的 scaling behavior | [Paper](https://arxiv.org/abs/2305.16264) |
| ⬜ | **The Pile: An 800GB Dataset of Diverse Text for Language Modeling** | 多域预训练语料组成、治理与 benchmark contamination | [Paper](https://arxiv.org/abs/2101.00027) |
| ⬜ | **Deduplicating Training Data Makes Language Models Better** | 去重对记忆、评测污染、训练效率和质量的影响 | [Paper](https://arxiv.org/abs/2107.06499) |
| ⬜ | **Data Selection for Language Models via Importance Resampling** | DSIR：用目标分布重要性重采样选择预训练数据 | [Paper](https://arxiv.org/abs/2302.03169) |
| ⬜ | **DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining** | 将多域数据配比转化为 group DRO 优化问题 | [Paper](https://arxiv.org/abs/2305.10429) |
| ⬜ | **The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only** | 大规模网页过滤与去重的生产级案例 | [Paper](https://arxiv.org/abs/2306.01116) |
| ⬜ | **Textbooks Are All You Need** | 合成高质量数据、数据质量与小模型能力的交换关系 | [Paper](https://arxiv.org/abs/2306.11644) |
| ⬜ | **Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research** | 开放语料的来源、过滤、去重、PII 与治理流程 | [Paper](https://arxiv.org/abs/2402.00159) |
| ⬜ | **DataComp-LM: In Search of the Next Generation of Training Sets for Language Models** | 受控比较过滤、去重和数据混合策略 | [Paper](https://arxiv.org/abs/2406.11794) |
| ⬜ | **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale** | 现代网页清洗、质量过滤与 FineWeb-Edu 配方 | [Paper](https://arxiv.org/abs/2406.17557) |

### Attention Mechanisms & Variants

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Attention Is All You Need** | NeurIPS'17 | [Paper](papers/DL/Attention-NIPS.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/32) |
| ✅ | **Big Bird: Transformers for Longer Sequences** | NeurIPS'20 | [Paper](papers/DL/bigbird.pdf) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/SparseAttention.md) |
| ✅ | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | NeurIPS'22 | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashAttention.md) |
| ✅ | **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | arXiv | [Paper](https://arxiv.org/pdf/2307.08691.pdf) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashAttention.md) |
| ✅ | **Flash-Decoding for Long-Context Inference** | Blog | [Paper](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashDecoding.md) |
| ✅ | **A Survey of Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention** | — | [Paper](https://attention-survey.github.io/files/Attention_Survey.pdf) |

### Sparse Attention Algorithms & Training

这一分类覆盖稀疏注意力本身的架构、选择算法与训练方法。MSA 和 DSA 是其中的现代案例，而不是分类边界。

#### Architecture and Modern Sparse Attention

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Fast Transformer Decoding: One Write-Head is All You Need** | MQA 原始论文；理解所有 Query heads 共享 KV 与 decode 带宽瓶颈 | [Paper](https://arxiv.org/abs/1911.02150) |
| ⬜ | **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** | 建立 MHA–GQA–MQA 的连续关系；理解 MSA 为什么按 GQA group 独立选择 | [Paper](https://arxiv.org/abs/2305.13245) |
| ⬜ | **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** | MLA、latent KV compression、decoupled RoPE，以及 `uk_proj` / `uo_proj` 的矩阵吸收 | [Paper](https://arxiv.org/abs/2405.04434) |
| ⬜ | **SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs** | 可学习的 block gate、自蒸馏与 block-sparse kernel；最接近 MSA Indexer 的对照之一 | [Paper](https://arxiv.org/abs/2410.13276) |
| ⬜ | **Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention** | 同时研究压缩、选择、局部窗口与硬件对齐；理解 DSA 的算法–kernel 协同背景 | [Paper](https://arxiv.org/abs/2502.11089) |
| ⬜ | **MoBA: Mixture of Block Attention for Long-Context LLMs** | MoE 风格的 block routing；适合对比 MSA 的 block max-pooling 与 group-specific selection | [Paper](https://arxiv.org/abs/2502.13189) |
| ⬜ | **MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention** | 无需重新训练的动态稀疏 prefill；对比学习式 Indexer 与预设 attention pattern | [Paper](https://arxiv.org/abs/2407.02490) |
| ⬜ | **SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference** | 两阶段在线过滤与 softmax-aware pruning；理解推理期稀疏化的另一条路线 | [Paper](https://arxiv.org/abs/2502.18137) |

**现代架构阅读顺序：** MQA → GQA → DeepSeek-V2/MLA → SeerAttention → Native Sparse Attention → MoBA → [Quest](https://arxiv.org/abs/2406.10774) → MInference → SpargeAttn。

#### Algorithm and Training Foundations

这组论文用于从算子实现反向补齐稀疏注意力依赖的算法与训练概念：稀疏 pattern、内容路由、不可微 Top-k、知识蒸馏、Router 稳定性和稀疏归一化。

##### Sparse Pattern and Content Routing

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Generating Long Sequences with Sparse Transformers** | 固定 factorized sparse pattern 与早期 block-sparse kernel；理解“规则但不自适应”的稀疏性 | [Paper](https://arxiv.org/abs/1904.10509) |
| ⬜ | **Longformer: The Long-Document Transformer** | local window + global token；理解局部先验和全局信息通路 | [Paper](https://arxiv.org/abs/2004.05150) |
| ⬜ | **Reformer: The Efficient Transformer** | LSH attention、bucket、排序和 reversible layers；理解动态候选集带来的数据重排成本 | [Paper](https://arxiv.org/abs/2001.04451) |
| ⬜ | **Efficient Content-Based Sparse Attention with Routing Transformers** | online k-means 内容路由；连接固定 sparse pattern 与 learned Indexer | [Paper](https://arxiv.org/abs/2003.05997) |

> **Big Bird** 已在 [Attention Mechanisms & Variants](#attention-mechanisms--variants) 中记录并标为已读，不在这里重复计数。

##### Differentiable Top-k and Discrete Selection

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Sparser is Faster and Less is More: Efficient Sparse Attention for Long-Range Transformers** | SparseK scoring network + differentiable Top-k；直接对比 MSA/DSA 的 hard Top-k + KL 路线 | [Paper](https://arxiv.org/abs/2406.16747) |
| ⬜ | **Differentiable Top-k Operator with Optimal Transport** | 用熵正则 Optimal Transport 平滑 Top-k，理解连续松弛及其梯度 | [Paper](https://arxiv.org/abs/2002.06504) |
| ⬜ | **Fast, Differentiable and Sparse Top-k: A Convex Analysis Perspective** | 从凸优化构造可微且真正稀疏的 Top-k，并讨论 GPU/TPU-friendly 算法 | [Paper](https://arxiv.org/abs/2302.01425) |
| ⬜ | **Categorical Reparameterization with Gumbel-Softmax** | 离散采样的经典连续松弛；理解 temperature、annealing 与 soft-train/hard-inference 差异 | [Paper](https://arxiv.org/abs/1611.01144) |

##### Distillation and Router Training

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ✅ 📝 | **Distilling the Knowledge in a Neural Network** | soft target、temperature 与 KL；理解主 attention 如何作为 Indexer teacher | [Paper](https://arxiv.org/abs/1503.02531) / [Note](notes/llm/knowledge-distillation/distilling-the-knowledge-in-a-neural-network.md) |
| ⬜ | **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** | hard Top-1 routing、capacity 与 load-balancing loss；将 Indexer 理解成 memory router | [Paper](https://arxiv.org/abs/2101.03961) |
| ⬜ | **ST-MoE: Designing Stable and Transferable Sparse Expert Models** | Router 稳定性、辅助损失与 router z-loss；理解小型路由器如何影响整个模型训练 | [Paper](https://arxiv.org/abs/2202.08906) |
| ⬜ | **Mixture-of-Experts with Expert Choice Routing** | expert 选择 token 而非 token 选择 expert；类比 q2k→k2q reverse index 与 KV-owner 调度 | [Paper](https://arxiv.org/abs/2202.09368) |

##### Sparse Normalization

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification** | 在概率 simplex 上产生精确零值并保留可计算 Jacobian | [Paper](https://arxiv.org/abs/1602.02068) |
| ⬜ | **Adaptively Sparse Transformers** | 使用可学习的 $\alpha$-entmax 让不同 attention heads 自适应选择稠密或稀疏分布 | [Paper](https://arxiv.org/abs/1909.00015) |

**算法/训练补课顺序：** Sparse Transformer → Routing Transformer → SparseK Attention → Differentiable Top-k → Knowledge Distillation → Switch Transformer / ST-MoE → 回看 SeerAttention、NSA、MSA 与 DSA。

### New Architectures

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ⬜ | **Hyena Hierarchy: Towards Larger Convolutional Language Models** | ICML'23 | [Paper](https://arxiv.org/abs/2302.10866) |
| ⬜ | **RWKV: Reinventing RNNs for the Transformer Era** | EMNLP'23 | [Paper](https://arxiv.org/abs/2305.13048) |
| ⬜ | **Retentive Network: A Successor to Transformer for Large Language Models** | arXiv'23 | [Paper](https://arxiv.org/abs/2307.08621) |
| ⬜ | **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** | arXiv'23 | [Paper](https://arxiv.org/abs/2312.00752) |
| ⬜ | **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality** | ICML'24 | [Paper](https://arxiv.org/abs/2405.21060) |
| ⬜ | **xLSTM: Extended Long Short-Term Memory** | NeurIPS'24 | [Paper](https://arxiv.org/abs/2405.04517) |
| ✅ | **Gated Linear Attention Transformers with Hardware-Efficient Training** | arXiv | [Paper](papers/DL/GLA.pdf) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/LinearAttention.md) |
| ✅ | **Kimi Linear Attention: An Expressive, Efficient Attention Architecture** | arXiv'25 | [Paper](https://arxiv.org/pdf/2510.26692) |
| ✅ | **DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models** | arXiv'25 | [Paper](https://arxiv.org/pdf/2512.02556) |
| ✅ 📝 | **DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence** | Tech report'26 | [Paper](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf) / [Note](notes/llm/deepseek-v4/deepseek-v4.md) |

### On-Device / Mobile

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **On-Device Training Under 256KB Memory** | NeurIPS'22 | [Paper](https://arxiv.org/pdf/2206.15472.pdf) |
| ✅ | **PockEngine: Sparse and Efficient Fine-tuning in a Pocket** | MICRO'23 | [Paper](papers/mlsys/on-device/pockengine.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/29) |

## 📊 LLM Evaluation & Safety

### Capability & Quality Evaluation

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **Measuring Massive Multitask Language Understanding** | MMLU：用多学科考试题衡量知识与问题求解能力 | [Paper](https://arxiv.org/abs/2009.03300) |
| ⬜ | **Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models** | BIG-bench：异构任务与能力随规模变化 | [Paper](https://arxiv.org/abs/2206.04615) |
| ⬜ | **Holistic Evaluation of Language Models** | HELM：统一准确率、校准、鲁棒性、公平性、毒性和效率 | [Paper](https://arxiv.org/abs/2211.09110) |
| ⬜ | **TruthfulQA: Measuring How Models Mimic Human Falsehoods** | 将事实真实性与普通知识准确率区分开 | [Paper](https://arxiv.org/abs/2109.07958) |
| ⬜ | **Training Verifiers to Solve Math Word Problems** | GSM8K 与 verifier 路线，连接推理生成和可验证结果 | [Paper](https://arxiv.org/abs/2110.14168) |
| ⬜ | **Measuring Mathematical Problem Solving With the MATH Dataset** | 竞赛数学、分步推理与严格答案评测 | [Paper](https://arxiv.org/abs/2103.03874) |
| ⬜ | **Evaluating Large Language Models Trained on Code** | HumanEval 与 pass@k，理解代码生成评测 | [Paper](https://arxiv.org/abs/2107.03374) |
| ⬜ | **GPQA: A Graduate-Level Google-Proof Q&A Benchmark** | 高难度、抗搜索污染的专家级科学问答 | [Paper](https://arxiv.org/abs/2311.12022) |
| ⬜ | **Instruction-Following Evaluation for Large Language Models** | IFEval：用可验证约束衡量 instruction following | [Paper](https://arxiv.org/abs/2311.07911) |
| ⬜ | **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** | LLM judge、位置偏差、冗长偏差和 pairwise evaluation | [Paper](https://arxiv.org/abs/2306.05685) |
| ⬜ | **Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference** | 真实用户盲测、成对偏好与 Elo/Bradley–Terry 排名 | [Paper](https://arxiv.org/abs/2403.04132) |
| ⬜ | **RewardBench: Evaluating Reward Models for Language Modeling** | 对话、推理、安全等场景的 reward model 基准 | [Paper](https://arxiv.org/abs/2403.13787) |
| ⬜ | **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** | 真实仓库、issue、patch 与测试驱动的软件工程评测 | [Paper](https://arxiv.org/abs/2310.06770) |
| ⬜ | **LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code** | 持续更新题目与时间切分，降低代码评测污染 | [Paper](https://arxiv.org/abs/2403.07974) |
| ⬜ | **Lessons from the Trenches on Reproducible Evaluation of Language Models** | prompt、tokenizer、版本和实现细节对评测复现的影响 | [Paper](https://arxiv.org/abs/2405.14782) |

### Safety, Robustness & Red Teaming

| Status | Paper | Why It Matters | Links |
|:------:|-------|----------------|-------|
| ⬜ | **RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models** | 开放式生成中的 toxicity 测量与 prompt 条件效应 | [Paper](https://arxiv.org/abs/2009.11462) |
| ⬜ | **Red Teaming Language Models with Language Models** | 用模型自动生成和筛选攻击，建立可扩展 red teaming | [Paper](https://arxiv.org/abs/2202.03286) |
| ⬜ | **Universal and Transferable Adversarial Attacks on Aligned Language Models** | GCG 对抗后缀与可迁移 jailbreak 的基础工作 | [Paper](https://arxiv.org/abs/2307.15043) |
| ⬜ | **XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models** | 同时衡量合理拒绝与过度拒绝 | [Paper](https://arxiv.org/abs/2308.01263) |
| ⬜ | **SafetyBench: Evaluating the Safety of Large Language Models** | 多类别、多语言的安全知识与行为评测 | [Paper](https://arxiv.org/abs/2309.07045) |
| ⬜ | **Towards Understanding Sycophancy in Language Models** | 研究模型迎合用户观点的表现及训练信号来源 | [Paper](https://arxiv.org/abs/2310.13548) |
| ⬜ | **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** | 研究条件触发的欺骗策略能否熬过安全训练 | [Paper](https://arxiv.org/abs/2401.05566) |
| ⬜ | **HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal** | jailbreak、攻击方法与稳健拒绝的标准化评测 | [Paper](https://arxiv.org/abs/2402.04249) |
| ⬜ | **A StrongREJECT for Empty Jailbreaks** | jailbreak 评测必须衡量实际危害、完成度和拒绝质量 | [Paper](https://arxiv.org/abs/2402.10260) |
| ⬜ | **The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning** | 高风险双用途知识评测及其与 unlearning 的关系 | [Paper](https://arxiv.org/abs/2403.03218) |

## 🤖 LLM for Kernel Optimization

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **AVO: Agentic Variation Operators for Autonomous Evolutionary Search** | arXiv'26 | [Paper](https://arxiv.org/abs/2603.24517) / [Note](notes/llm-for-kernel/avo.md) |
| ✅ 📝 | **CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution** | arXiv'26 | [Paper](https://arxiv.org/abs/2608.16292) / [Note](notes/llm-for-kernel/CAKE.md) |

## 🧩 Agent Systems

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ 📝 | **SkVM: Revisiting Language VM for Skills across Heterogenous LLMs and Harnesses** | arXiv'26 | [Paper](https://arxiv.org/abs/2604.03088) / [Note](notes/agent/skvm/skvm.md) |

## 🖥️ GPU Microarchitecture

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ⬜ | **Understanding Latency Hiding on GPUs** | — | [Paper](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf) |

## 📐 Math Foundations

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ⬜ | **Categorical Foundations for CuTe Layouts** | — | [Paper](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) |

## ⚙️ Compiler

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Honeycomb: Secure and Efficient GPU Executions via Static Validation** | OSDI'23 | [Paper](papers/hypervisor/osdi23-mai.pdf) / [Note](notes/hypervisor/honeycomb/honeycomb.md) |
| ✅ | **HIDA: A Hierarchical Dataflow Compiler for High-Level Synthesis** | ASPLOS'24 | [Paper](papers/compiler/asplos24-hida.pdf) / [Note](notes/compiler/hida/hida.md) |

## 🐧 Operating Systems

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **RedLeaf: Isolation and Communication in a Safe Operating System** | OSDI'20 | [Paper](https://www.usenix.org/system/files/osdi20-narayanan_vikram.pdf) / [Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/os/RedLeaf.md) |
| ✅ | **Theseus: an Experiment in Operating System Structure and State Management** | OSDI'20 | [Paper](https://www.usenix.org/system/files/osdi20-boos.pdf) |
| ✅ | **Unikraft: Fast, Specialized Unikernels the Easy Way** | EuroSys'21 | [Paper](https://dl.acm.org/doi/pdf/10.1145/3447786.3456248) / [Note](https://github.com/KuangjuX/paper-reading/issues/9) |
| ✅ | **The Demikernel Datapath OS Architecture for Microsecond-scale Datacenter Systems** | SOSP'21 | [Paper](https://irenezhang.net/papers/demikernel-sosp21.pdf) / [Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/os/Demikernel.md) |

## 🛡️ Hypervisor & Virtualization

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **HyperBench: A Benchmark Suite for Virtualization Capabilities** | — | [Paper](https://dl.acm.org/doi/pdf/10.1145/3341617.3326138) / [Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/hypervisor/Hyperbench-A-Benchmark-Suite-for-Virtualization-Capabilities.md) |
| ✅ | **DuVisor: a User-level Hypervisor Through Delegated Virtualization** | arXiv'22 | [Paper](https://arxiv.org/pdf/2201.09652.pdf) |
| ✅ | **AvA: Accelerated Virtualization of Accelerators** | ASPLOS'22 | [Paper](https://dl.acm.org/doi/pdf/10.1145/3373376.3378466) |
| ✅ | **Security and Performance in the Delegated User-level Virtualization** | OSDI'23 | [Paper](https://www.usenix.org/system/files/osdi23-chen.pdf) / [Note](notes/hypervisor/duvisor/duvisor.md) |
| ✅ | **System Virtualization for Neural Processing Units** | HotOS'23 | [Paper](https://sigops.org/s/conferences/hotos/2023/papers/xue.pdf) |
| ✅ | **Nephele: Extending Virtualization Environments for Cloning Unikernel-based VMs** | EuroSys'23 | [Paper](http://nets.cs.pub.ro/~costin/files/nephele.pdf) / [Note](notes/hypervisor/nephele/nephele.md) |
| ✅ | **Honeycomb: Secure and Efficient GPU Executions via Static Validation** | OSDI'23 | [Paper](papers/hypervisor/osdi23-mai.pdf) / [Note](notes/hypervisor/honeycomb/honeycomb.md) |

## 🔬 RISC-V

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **A First Look at RISC-V Virtualization from an Embedded Systems Perspective** | TC'21 | [Paper](https://arxiv.org/pdf/2103.14951.pdf) |
| ✅ | **CVA6 RISC-V Virtualization: Architecture, Microarchitecture, and Design Space Exploration** | arXiv'23 | [Paper](https://arxiv.org/pdf/2302.02969.pdf) |

---

<p align="center">
  <em>If you find this list helpful, feel free to ⭐ star this repo!</em>
</p>
