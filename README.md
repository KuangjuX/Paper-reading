# Paper Reading

## Machine Learning System

### Deep Learning Compiler

- Survey
  - [x] The Deep Learning Compiler: A Comprehensive Survey: [[Paper](https://arxiv.org/pdf/2002.03794.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/ai-compiler/The-Deep-Learning-Compiler-A-Comprehensive-Survey.md)]

- Graph Optimization

  - [x] Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-ma.pdf)] [[Note](https://github.com/KuangjuX/paper-reading/issues/22)]

- Efficient Compute-intensive Kernel Generation
  - [x] ROLLER: Fast and Efficient Tensor Compilation for Deep Learning[OSDI'22]: [[Paper](https://www.usenix.org/system/files/osdi22-zhu.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/24)]

- Efficient Compute-intensive Kernel Fusion
  - [x] BOLT: Brinding The Gap Between Auto-Tunners and Hardware-Native Performance[MLSys'22]: [[Paper](http://yibozhu.com/doc/bolt-mlsys22.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/28)]

- Efficient Memory-intensive Kernel Fusion
  - [x] AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures[ASPLOS'22]: [[Paper](https://dl.acm.org/doi/10.1145/3503222.3507723)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/26)]
  - [x] Welder: Scheduling Deep Learning Memory Access via Tile-graph[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-shi.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/25)]

- Dataflow Analytical
  - [x] Cocktailer: Analyzing and Optimizing Dynamic Control Flow in Deep Learning[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-zhang-chen.pdf)] [[Note](https://github.com/KuangjuX/paper-reading/issues/21)]
  - [x] Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion[HPCA'23]: [[Paper](papers/mlsys/fusion/Chimera_An_Analytical_Optimizing_Framework_for_Effective_Compute-intensive_Operators_Fusion.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/30)]

- Efficient Hardware Mapping
    - [x] AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction[ISCA'22]: [[Paper](papers/mlsys/IR/AMOS-ISCA.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/31)]

- IR Design

  - [x] MLIR: Scaling Compiler Infrastructure for Domain Specifific Computation[CGO'21]: [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9370308)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/ai-compiler/MLIR-Scaling-Compiler-Infrastructure-for-Domain-Specific-Computation.md)]
  - [x] Graphene: An IR for Optimized Tensor Computations on GPUs[ASPLOS'23]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3582016.3582018)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/27)]




- Polyhedral Optimization

  - [x] TIRAMISU: A Polyhedral Compiler for Expressing Fast and Portable Code[CGO'19]: [[Paper](papers/mlsys/polyhedral/Tiramisu-CGO.pdf)] [[Note](notes/compiler/tiramisu-cgo/tiramisu.md)]
  - [x] Effectively Scheduling Computational Graphs of Deep Neural Networks
toward Their Domain-Specific Accelerators[OSDI'23]: [[Paper](papers/mlsys/soft-hard-co-design/osdi23-zhao.pdf)]



### System for Edge and Mobile

- [x] On-Device Training Under 256KB Memory[NeurIPS'22]: [[Paper](https://arxiv.org/pdf/2206.15472.pdf)]
- [x] PockEngine: Sparse and Efficient Fine-tuning in a Pocket[MICRO'23]: [[Paper](papers/mlsys/on-device/pockengine.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/29)]

### System for SIMT GPU Programming

- [x] FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving[Arxiv'2024]: [[Paper](papers/mlsys/2501.01005v1.pdf)]
- [x] ThunderKittens: Simple, Fast, and Adorable AI Kernels[Arxiv'2024]: [[Paper](papers/mlsys/ThunderKittens.pdf)]
- [x] SGLang: Efficient Execution of Structured language Model Programs[Arxiv'24]: [[Paper](papers/mlsys/sglang.pdf)]


## Deep Learning

### Attention with Variants

- [x] Attention is all you need[NIPS'17]: [[Paper](papers/DL/Attention-NIPS.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/32)]
- [x] Big bird: transformers for longer sequences[NIPS'20]: [[Paper](papers/DL/bigbird.pdf)] [[Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/SparseAttention.md)]
- [x] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness[NIPS'22]: [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)] [[Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashAttention.md)]
- [x] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning[Arxiv]: [[Paper](https://arxiv.org/pdf/2307.08691.pdf)] [[Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashAttention.md)]
- [x] Flash-Decoding for long-context inference[Blog]: [[Paper](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)] [[Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/FlashDecoding.md)]

### New Architecture for LLM

- [x] Gated Linear Attention Transformers with Hardware-Efficient Training[arxiv]: [[Paper](papers/DL/GLA.pdf)] [[Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/LinearAttention.md)]




## Compiler

- [x] Honeycomb: Secure and Efficient GPU Executions via Static Validation[OSDI'23]: [[Paper](papers/hypervisor/osdi23-mai.pdf)] [[Note](notes/hypervisor/honeycomb/honeycomb.md)]
- [x] HIDA: A Hierarchical Dataflow Compiler for High-Level Synthesis[ASPLOS'24]: [[Paper](papers/compiler/asplos24-hida.pdf)] [[Note](notes/compiler/hida/hida.md)]

## OS
- [x] RedLeaf: Isolation and Communication in a Safe Operating System[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-narayanan_vikram.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/os/RedLeaf.md)]
- [x] Theseus: an Experiment in Operating System Structure and State Management[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-boos.pdf)]
- [x] Unikraft: Fast, Specialized Unikernels the Easy Way[EuroSys'21]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3447786.3456248)] [[Note](https://github.com/KuangjuX/paper-reading/issues/9)]
- [x] The Deminkernel Datapath OS Architecture for Microsecond-scale Datacenter Systems[SOSP'21]: [[Paper](https://irenezhang.net/papers/demikernel-sosp21.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/os/Demikernel.md)]


## Hypervisor
- [x] HyperBench: A Benchmark Suite for Virtualization Capabilities: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3341617.3326138)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/hypervisor/Hyperbench-A-Benchmark-Suite-for-Virtualization-Capabilities.md)]
- [x] DuVisor: a User-level Hypervisor Through Delegated Virtualization[arxiv'22]: [[Paper](https://arxiv.org/pdf/2201.09652.pdf)]
- [x] AvA: Accelerated Virtualization of Accelerators[ASPLOS'22]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3373376.3378466)]
- [x] Security and Performance in the Delegated User-level Virtualization[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-chen.pdf)] [[Note](notes/hypervisor/duvisor/duvisor.md)]
- [x] System Virtualization for Neural Processing Units[HotOS'23]: [[Paper](https://sigops.org/s/conferences/hotos/2023/papers/xue.pdf)]
- [x] Nephele: Extending Virtualization Environments for Cloning Unikernel-based VMs[EuroSys'23]: [[Paper](http://nets.cs.pub.ro/~costin/files/nephele.pdf)] [[Note](notes/hypervisor/nephele/nephele.md)]
- [x] Honeycomb: Secure and Efficient GPU Executions via Static Validation[OSDI'23]: [[Paper](papers/hypervisor/osdi23-mai.pdf)] [[Note](notes/hypervisor/honeycomb/honeycomb.md)]

## RISC-V
- [x] A First Look at RISC-V Virtualization from an Embedded Systems Perspective[TC'21]: [[Paper](https://arxiv.org/pdf/2103.14951.pdf)]
- [x] CVA6 RISC-V Virtualization: Architecture, Microarchitecture, and Design Space Exploration[arxiv'23]: [[Paper](https://arxiv.org/pdf/2302.02969.pdf)]
