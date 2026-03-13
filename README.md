<h1 align="center">📚 Paper Reading</h1>

<p align="center">
  <em>A curated collection of research papers on AI systems, compilers, architecture, and systems software.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Papers-75-blue?style=flat-square" alt="Papers">
  <img src="https://img.shields.io/badge/Read-59-green?style=flat-square" alt="Read">
  <img src="https://img.shields.io/badge/To_Read-16-orange?style=flat-square" alt="To Read">
</p>

---

## Table of Contents

- [Deep Learning Compiler](#-deep-learning-compiler)
- [LLM Inference](#-llm-inference)
- [LLM Training](#-llm-training)
- [Deep Learning](#-deep-learning)
- [GPU Microarchitecture](#-gpu-microarchitecture)
- [Math Foundations](#-math-foundations)
- [Compiler](#-compiler)
- [Operating Systems](#-operating-systems)
- [Hypervisor & Virtualization](#-hypervisor--virtualization)
- [RISC-V](#-risc-v)

> **Legend:** ✅ = Read &nbsp;|&nbsp; ⬜ = To Read &nbsp;|&nbsp; 📝 = Note Available

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

### LLM Serving

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **SGLang: Efficient Execution of Structured Language Model Programs** | — | [Paper](papers/mlsys/sglang.pdf) |
| ✅ | **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving** | — | [Paper](papers/mlsys/2501.01005v1.pdf) |
| ⬜ | **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** | OSDI'24 | [Paper](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) |
| ⬜ | **LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism** | SOSP'24 | [Paper](https://dl.acm.org/doi/pdf/10.1145/3694715.3695948) |
| ⬜ | **Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot** | FAST'25 | [Paper](https://www.usenix.org/system/files/fast25-qin.pdf) |
| ⬜ | **NanoFlow: Towards Optimal Large Language Model Serving Throughput** | OSDI'25 | [Paper](https://www.usenix.org/system/files/osdi25-zhu-kan.pdf) |

### MegaKernel

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B** | Blog | [Paper](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) |
| ✅ | **Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs** | arXiv'25 | [Paper](https://arxiv.org/pdf/2512.22219) |
| ✅ | **TileRT: Tile-Based Runtime for Ultra-Low-Latency LLM Inference** | — | [Paper](https://github.com/tile-ai/TileRT) |
| ✅ | **SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations** | arXiv'25 | [Paper](https://arxiv.org/pdf/2512.14080) |
| ✅ | **Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference** | Blog | [Paper](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17) |

## 🏋️ LLM Training

### Distributed Training

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ⬜ | **LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism** | — | [Paper](https://arxiv.org/pdf/2406.18485) |

### RL Training

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning** | arXiv'25 | [Paper](https://arxiv.org/pdf/2511.14617) |

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

### Attention Mechanisms & Variants

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Attention Is All You Need** | NeurIPS'17 | [Paper](papers/DL/Attention-NIPS.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/32) |
| ✅ | **Big Bird: Transformers for Longer Sequences** | NeurIPS'20 | [Paper](papers/DL/bigbird.pdf) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/SparseAttention.md) |
| ✅ | **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | NeurIPS'22 | [Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashAttention.md) |
| ✅ | **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | arXiv | [Paper](https://arxiv.org/pdf/2307.08691.pdf) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashAttention.md) |
| ✅ | **Flash-Decoding for Long-Context Inference** | Blog | [Paper](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashDecoding.md) |
| ✅ | **A Survey of Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention** | — | [Paper](https://attention-survey.github.io/files/Attention_Survey.pdf) |

### New Architectures

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **Gated Linear Attention Transformers with Hardware-Efficient Training** | arXiv | [Paper](papers/DL/GLA.pdf) / [Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/LinearAttention.md) |
| ✅ | **Kimi Linear Attention: An Expressive, Efficient Attention Architecture** | arXiv'25 | [Paper](https://arxiv.org/pdf/2510.26692) |
| ✅ | **DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models** | arXiv'25 | [Paper](https://arxiv.org/pdf/2512.02556) |

### On-Device / Mobile

| Status | Paper | Venue | Links |
|:------:|-------|-------|-------|
| ✅ | **On-Device Training Under 256KB Memory** | NeurIPS'22 | [Paper](https://arxiv.org/pdf/2206.15472.pdf) |
| ✅ | **PockEngine: Sparse and Efficient Fine-tuning in a Pocket** | MICRO'23 | [Paper](papers/mlsys/on-device/pockengine.pdf) / [Note](https://github.com/KuangjuX/Paper-reading/issues/29) |

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
