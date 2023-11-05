# Paper Reading

## MLSys

### DNN Compiler

#### Survey

- [x] The Deep Learning Compiler: A Comprehensive Survey: [[Paper](https://arxiv.org/pdf/2002.03794.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/ai-compiler/The-Deep-Learning-Compiler-A-Comprehensive-Survey.md)]

#### Graph Optimization

- [ ] TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions[SOSP'19]: [[Paper](https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf)]
- [ ] Ansor: Generating High-Performance Tensor Programs for Deep Learning[OSDI'20]:   [[Paper](https://www.usenix.org/system/files/osdi20-zheng.pdf)]
- [x] Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-ma.pdf)] [[Note](https://github.com/KuangjuX/paper-reading/issues/22)]
- [ ] Pet: Optimizing Tensor Programs with Partially Equivalent Transformations and
  Automated Corrections[OSDI'21]: [[Paper](https://www.usenix.org/system/files/osdi21-wang-haojie.pdf)]
- [ ] DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion[PLDI'21]: [[Paper](https://arxiv.org/pdf/2108.13342.pdf)]
- [ ] Graphiler: A Compiler for Graph Neural Networks[MLSys'22]: [[Paper](https://gnnsys.github.io/papers/GNNSys21_paper_10.pdf)]
- [ ] EinNet: Optimizing Tensor Programs with Derivation-Based Transformations [OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-zheng.pdf)]

#### Kernel Generation

- [ ] Automatic Kernel Generation for Volta Tensor Cores[arxiv'20][Nvidia]: [[Paper](https://arxiv.org/abs/2006.12645)]
- [x] AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures[ASPLOS'22]: [[Paper](https://dl.acm.org/doi/10.1145/3503222.3507723)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/26)]
- [ ] Tacker: Tensor-CUDA Core Kernel Fusion for Improving the GPU Utilization while Ensuring QoS[HPCA'22]: [[Paper](paper/mlsys/Kernel_Generation/Tacker_Tensor-CUDA_Core_Kernel_Fusion_for_Improving_the_GPU_Utilization_while_Ensuring_QoS.pdf)]

- [ ] BOLT: BRIDGING THE GAP BETWEEN AUTO-TUNERS AND HARDWARE-NATIVE PERFORMANCE[MLSys'22]: [[Paper](http://yibozhu.com/doc/bolt-mlsys22.pdf)]
- [x] ROLLER: Fast and Efficient Tensor Compilation for Deep Learning[OSDI'22]: [[Paper](https://www.usenix.org/system/files/osdi22-zhu.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/24)]
- [x] Cocktailer: Analyzing and Optimizing Dynamic Control Flow in Deep Learning[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-zhang-chen.pdf)] [[Note](https://github.com/KuangjuX/paper-reading/issues/21)]
- [x] Welder: Scheduling Deep Learning Memory Access via Tile-graph[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-shi.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/25)]

#### Compiler Design

- [ ] The Tensor Algebra Compiler[OOPSLA'17]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3133901)]
- [ ] DISTAL: The Distributed Tensor Algebra Compiler[PLDI'22]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3519939.3523437)]

#### IR Design

- [x] MLIR: Scaling Compiler Infrastructure for Domain Specifific Computation[CGO'21]: [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9370308)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/ai-compiler/MLIR-Scaling-Compiler-Infrastructure-for-Domain-Specific-Computation.md)]
- [ ] SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning[ASPLOS'23]: [[Paper](https://arxiv.org/pdf/2207.04606.pdf)]
- [x] Graphene: An IR for Optimized Tensor Computations on GPUs[ASPLOS'23]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3582016.3582018)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/27)]

### Distrubited Training

- [ ] Ray: A Distributed Framework for Emerging AI Applications[OSDI'18]: [[Paper](https://www.usenix.org/system/files/osdi18-moritz.pdf)]
- [ ] Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning[OSDI'22]: [[Paper](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)]
- [ ] Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling[SOSP'23]: [[Paper](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sia_sosp23-final.pdf)]

### Accelerator Architecture

- [ ] Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences[OSDI'22]: [[Paper](https://www.usenix.org/system/files/osdi22-han.pdf)]
- [ ] FLASHATTENTION: Fast and Memory-Efficient Exact Attention with IO-Awareness[NIPS'22]: [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)]
- [ ] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning[arxiv]: [[Paper](https://arxiv.org/pdf/2307.08691.pdf)]

## OS

- [x] Unikraft: Fast, Specialized Unikernels the Easy Way[EuroSys'19]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3447786.3456248)] [[Note](https://github.com/KuangjuX/paper-reading/issues/9)]
- [x] The Deminkernel Datapath OS Architecture for Microsecond-scale Datacenter Systems[SOSP'21]: [[Paper](https://irenezhang.net/papers/demikernel-sosp21.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/os/Demikernel.md)]
- [x] RedLeaf: Isolation and Communication in a Safe Operating System[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-narayanan_vikram.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/os/RedLeaf.md)]
- [ ] Theseus: an Experiment in Operating System Structure and State Management[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-boos.pdf)]

## Hypervisor

- [x] HyperBench: A Benchmark Suite for Virtualization Capabilities: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3341617.3326138)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/hypervisor/Hyperbench-A-Benchmark-Suite-for-Virtualization-Capabilities.md)]
- [ ] HyperEnclave: An Open and Cross-platform Trusted Execution Environment[USENIX ATC'22]: [[Paper](https://www.usenix.org/system/files/atc22-jia-yuekai.pdf)]
- [x] Security and Performance in the Delegated User-level Virtualization[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-chen.pdf)]
- [x] System Virtualization for Neural Processing Units[HotOS'23]: [[Paper](https://sigops.org/s/conferences/hotos/2023/papers/xue.pdf)]
- [ ] Nephele: Extending Virtualization Environments for Cloning Unikernel-based VMs[EuroSys'23]: [[Paper](http://nets.cs.pub.ro/~costin/files/nephele.pdf)]

## Deep Learning

- [x] On-Device Training Under 256KB Memory[NeurIPS'22]: [[Paper](https://arxiv.org/pdf/2206.15472.pdf)]
- [ ] Full Stack Optimization of Transformer Inference: a Survey[arxiv]: [[Paper](https://arxiv.org/pdf/2302.14017.pdf)]
