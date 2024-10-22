# Paper Reading

## MLSys
### DL Compiler
#### Survey
- [x] The Deep Learning Compiler: A Comprehensive Survey: [[Paper](https://arxiv.org/pdf/2002.03794.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/ai-compiler/The-Deep-Learning-Compiler-A-Comprehensive-Survey.md)]

#### Graph Optimization
- [ ] TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions[SOSP'19]: [[Paper](https://www.cs.cmu.edu/~zhihaoj2/papers/sosp19.pdf)]
- [x] Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-ma.pdf)] [[Note](https://github.com/KuangjuX/paper-reading/issues/22)]
- [ ] Pet: Optimizing Tensor Programs with Partially Equivalent Transformations and
  Automated Corrections[OSDI'21]: [[Paper](https://www.usenix.org/system/files/osdi21-wang-haojie.pdf)]
- [ ] Graphiler: A Compiler for Graph Neural Networks[MLSys'22]: [[Paper](https://gnnsys.github.io/papers/GNNSys21_paper_10.pdf)]
- [ ] EinNet: Optimizing Tensor Programs with Derivation-Based Transformations [OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-zheng.pdf)]


#### Kernel Fusion
- [ ] Automatic Kernel Generation for Volta Tensor Cores[arxiv'20][Nvidia]: [[Paper](https://arxiv.org/abs/2006.12645)]
- [ ] DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion[PLDI'21]: [[Paper](https://arxiv.org/pdf/2108.13342.pdf)]
- [ ] Automatic Horizontal Fusion for GPU Kernels[CGO'22]: [[Paper](papers/mlsys/fusion/CGO53902.2022.9741270.pdf)]
- [x] AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures[ASPLOS'22]: [[Paper](https://dl.acm.org/doi/10.1145/3503222.3507723)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/26)]
- [ ] Tacker: Tensor-CUDA Core Kernel Fusion for Improving the GPU Utilization while Ensuring QoS[HPCA'22]: [[Paper](papers/mlsys/fusion/Tacker_Tensor-CUDA_Core_Kernel_Fusion_for_Improving_the_GPU_Utilization_while_Ensuring_QoS.pdf)]
- [x] BOLT: BRIDGING THE GAP BETWEEN AUTO-TUNERS AND HARDWARE-NATIVE PERFORMANCE[MLSys'22]: [[Paper](http://yibozhu.com/doc/bolt-mlsys22.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/28)]
- [x] ROLLER: Fast and Efficient Tensor Compilation for Deep Learning[OSDI'22]: [[Paper](https://www.usenix.org/system/files/osdi22-zhu.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/24)]
- [x] Cocktailer: Analyzing and Optimizing Dynamic Control Flow in Deep Learning[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-zhang-chen.pdf)] [[Note](https://github.com/KuangjuX/paper-reading/issues/21)]
- [x] Welder: Scheduling Deep Learning Memory Access via Tile-graph[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-shi.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/25)]
- [x] Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion[HPCA'23]: [[Paper](papers/mlsys/fusion/Chimera_An_Analytical_Optimizing_Framework_for_Effective_Compute-intensive_Operators_Fusion.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/30)]
- [x] Effectively Scheduling Computational Graphs of Deep Neural Networks
toward Their Domain-Specific Accelerators[OSDI'23]: [[Paper](papers/mlsys/soft-hard-co-design/osdi23-zhao.pdf)]
- [ ] Operator Fusion in XLA: Analysis and Evaluation: [[Paper](papers/mlsys/compiler-design/XLA.pdf)]

#### Compiler Design
- [ ] The Tensor Algebra Compiler[OOPSLA'17]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3133901)]
- [ ] Bring Your Own Codegen to Deep Learning Compiler[arxiv]: [[Paper](https://arxiv.org/pdf/2105.03215.pdf)]
- [ ] DISTAL: The Distributed Tensor Algebra Compiler[PLDI'22]: [[Paper](papers/mlsys/compiler-design/distal.pdf)]

#### IR Design
- [ ] Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations[MAPL'19]: [[Paper](papers/mlsys/IR/2019-triton.pdf)]
- [x] MLIR: Scaling Compiler Infrastructure for Domain Specifific Computation[CGO'21]: [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9370308)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/ai-compiler/MLIR-Scaling-Compiler-Infrastructure-for-Domain-Specific-Computation.md)]
- [x] AMOS: Enabling Automatic Mapping for Tensor Computations On Spatial Accelerators with Hardware Abstraction[ISCA'22]: [[Paper](papers/mlsys/IR/AMOS-ISCA.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/31)]
- [ ] SparseTIR: Composable Abstractions for Sparse Compilation in Deep Learning[ASPLOS'23]: [[Paper](https://arxiv.org/pdf/2207.04606.pdf)]
- [ ] TensorIR: An Abstraction for Automatic Tensorized Program Optimization[ASPLOS'23]: [[Paper](https://arxiv.org/pdf/2207.04296.pdf)]
- [x] Graphene: An IR for Optimized Tensor Computations on GPUs[ASPLOS'23]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3582016.3582018)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/27)]
- [ ] Relax: Composable Abstractions for End-to-End Dynamic Machine Learning[arxiv]: [[Paper](https://arxiv.org/pdf/2311.02103.pdf)]

#### Auto Schedule
- [ ] Ansor: Generating High-Performance Tensor Programs for Deep Learning[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-zheng.pdf)]
- [ ] FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System[ASPLOS'20]: [[Paper](https://ceca.pku.edu.cn/docs/20200915213803856105.pdf)]

#### Space Exploration
- [ ] Analytical Characterization and Design Space Exploration for Optimization of CNNs[ASPLOS'21]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3445814.3446759)]
- [ ] Mind Mappings: Enabling Efficient Algorithm-Accelerator Mapping Space Search[ASPLOS'21]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3445814.3446762)]

#### Software-Hardware Co-Design
- [ ] HASCO: Towards Agile Hardware and Software CO-design for Tensor Computation[ISCA'21]: [[Paper](papers/mlsys/soft-hard-co-design/HASCO-ISCA.pdf)]
- [ ] CoSA: Scheduling by Constrained Optimization for Spatial Accelerators[ISCA'21]: [[Paper](papers/mlsys/soft-hard-co-design/CoSA-ISCA.pdf)]
- [ ] SARA: Scaling a Reconfigurable Dataflow Accelerator[ISCA'21]: [[Paper](papers/mlsys/soft-hard-co-design/SARA-ISCA.pdf)]
- [ ] AutoSA: A Polyhedral Compiler for High-Performance Systolic Arrays on FPGA[FPGA'21]: [[Paper](papers/mlsys/soft-hard-co-design/AutoSA-FPGA.pdf)]

#### Polyhedral
- [ ] TIRAMISU: A Polyhedral Compiler for Expressing Fast and Portable Code[CGO'19]: [[Paper](papers/mlsys/polyhedral/Tiramisu-CGO.pdf)]
- [ ] AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations[PLDI'21]: [[Paper](papers/mlsys/polyhedral/AKG-PLDI.pdf)]

### Inter-Operator Optimization
- [ ] FLASHATTENTION: Fast and Memory-Efficient Exact Attention with IO-Awareness[NIPS'22]: [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)]
- [ ] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning[arxiv]: [[Paper](https://arxiv.org/pdf/2307.08691.pdf)]
- [ ] FLASHDECODING++: FASTER LARGE LANGUAGE MODEL INFERENCE ON GPUS[arxiv]: [[Paper](https://arxiv.org/pdf/2311.01282.pdf)]


### Distrubited Training
- [ ] Ray: A Distributed Framework for Emerging AI Applications[OSDI'18]: [[Paper](https://www.usenix.org/system/files/osdi18-moritz.pdf)]
- [ ] Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning[OSDI'22]: [[Paper](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)]
- [ ] AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf)]
- [ ] Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling[SOSP'23]: [[Paper](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sia_sosp23-final.pdf)]

### Accelerator Architecture
- [ ] Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences[OSDI'22]: [[Paper](https://www.usenix.org/system/files/osdi22-han.pdf)]
- [ ] UGache: A Unified GPU Cache for Embedding-based Deep Learning[SOSP'23]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3600006.3613169)]


### Sparse Model
- [ ] PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation[SOSP'23]: [[Paper](https://arxiv.org/pdf/2301.10936.pdf)]

### On-device Training
- [x] On-Device Training Under 256KB Memory[NeurIPS'22]: [[Paper](https://arxiv.org/pdf/2206.15472.pdf)]
- [x] PockEngine: Sparse and Efficient Fine-tuning in a Pocket[MICRO'23]: [[Paper](papers/mlsys/on-device/pockengine.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/29)]

### LLM Infer
- [ ] Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time[ICML'23]: [[Paper](papers/mlsys/LLM/liu23am.pdf)]
- [ ] Efficient Memory Management for Large Language Model Serving with PagedAttention[SOSP'23]: [[Paper](papers/mlsys/LLM/vllm-sosp.pdf)]
- [ ] LLM in a flash: Efficient Large Language Model Inference with Limited Memory[arxiv]: [[Paper](papers/mlsys/LLM/LLM-In-a-Flash.pdf)]
- [ ] PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU[arxiv]: [[Paper](papers/mlsys/LLM/powerinfer-20231219.pdf)]

## Deep Learning
- [x] Attention is all you need[NIPS'17]: [[Paper](papers/DL/Attention-NIPS.pdf)] [[Note](https://github.com/KuangjuX/Paper-reading/issues/32)]
- [x] Big bird: transformers for longer sequences[NIPS'20]: [[Paper](papers/DL/bigbird.pdf)] [[Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/SparseAttention.md)]
- [ ] Full Stack Optimization of Transformer Inference: a Survey[arxiv]: [[Paper](https://arxiv.org/pdf/2302.14017.pdf)]
- [ ] Llama 2: Open Foundation and Fine-Tuned Chat Models[arxiv]: [[Paper](papers/DL/llama2.pdf)]
- [x] Gated Linear Attention Transformers with Hardware-Efficient Training[arxiv]: [[Paper](papers/DL/GLA.pdf)] [[Note](https://github.com/KuangjuX/Notes/blob/main/DeepLearning/LinearAttention.md)]

## HPC/CUDA
- [ ] POSTER: Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU[PPOPP'23]: [[Paper](papers/hpc/stream-k.pdf)]

## Parallelism
- [ ] A Loop Transformation Theory and an Algorithm to Maximize Parallelism[TPDS'1991]: [[Paper](papers/parallelism/1991-tpds-wolf-unimodular.pdf)]

## Computer Architecutre
- [ ] Clockhands: Rename-free Instruction Set Architecture for Out-of-order Processors[MICRO'23]: [[Paper](papers/arch/clockhands.pdf)]

## Compiler
- [ ] egg: Fast and Extensible Equality Saturation[POPL'21]: [[Paper](papers/compiler/egg-PLDI-21.pdf)]
- [ ] Reticle: A Virtual Machine for Programming Modern FPGAs[PLDI'21]: [[Paper](papers/compiler/pldi21reticle.pdf)]
- [x] Honeycomb: Secure and Efficient GPU Executions via Static Validation[OSDI'23]: [[Paper](papers/hypervisor/osdi23-mai.pdf)] [[Note](notes/hypervisor/honeycomb/honeycomb.md)]
- [x] HIDA: A Hierarchical Dataflow Compiler for High-Level Synthesis[ASPLOS'24]: [[Paper](papers/compiler/asplos24-hida.pdf)] [[Note](notes/compiler/hida/hida.md)]

## OS
- [x] RedLeaf: Isolation and Communication in a Safe Operating System[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-narayanan_vikram.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/os/RedLeaf.md)]
- [ ] Theseus: an Experiment in Operating System Structure and State Management[OSDI'20]: [[Paper](https://www.usenix.org/system/files/osdi20-boos.pdf)]
- [x] Unikraft: Fast, Specialized Unikernels the Easy Way[EuroSys'21]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3447786.3456248)] [[Note](https://github.com/KuangjuX/paper-reading/issues/9)]
- [x] The Deminkernel Datapath OS Architecture for Microsecond-scale Datacenter Systems[SOSP'21]: [[Paper](https://irenezhang.net/papers/demikernel-sosp21.pdf)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/os/Demikernel.md)]
- [ ] FlexOS: towards flexible OS isolation[ASPLOS'22]: [[Paper](papers/os/flexos.pdf)]
- [ ] Unikernel Linux (UKL)[EuroSys'23]: [[Paper](papers/os/ukl.pdf)]
- [ ] TreeSLS: A Whole-system Persistent Microkernel with Tree-structured State Checkpoint on NVM[SOSP'23 Best Paper]: [[Paper](papers/os/treesls.pdf)]

## Hypervisor
- [x] HyperBench: A Benchmark Suite for Virtualization Capabilities: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3341617.3326138)] [[Note](https://github.com/KuangjuX/system-notes/blob/main/paper-notes/hypervisor/Hyperbench-A-Benchmark-Suite-for-Virtualization-Capabilities.md)]
- [ ] TwinVisor: Hardware-isolated Confidential Virtual Machines for ARM[SOSP'21]: [[Paper](papers/hypervisor/twinvisor.pdf)]
- [ ] HyperEnclave: An Open and Cross-platform Trusted Execution Environment[USENIX ATC'22]: [[Paper](https://www.usenix.org/system/files/atc22-jia-yuekai.pdf)]
- [ ] DuVisor: a User-level Hypervisor Through Delegated Virtualization[arxiv'22]: [[Paper](https://arxiv.org/pdf/2201.09652.pdf)]
- [x] AvA: Accelerated Virtualization of Accelerators[ASPLOS'22]: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3373376.3378466)]
- [x] Security and Performance in the Delegated User-level Virtualization[OSDI'23]: [[Paper](https://www.usenix.org/system/files/osdi23-chen.pdf)] [[Note](notes/hypervisor/duvisor/duvisor.md)]
- [x] System Virtualization for Neural Processing Units[HotOS'23]: [[Paper](https://sigops.org/s/conferences/hotos/2023/papers/xue.pdf)]
- [x] Nephele: Extending Virtualization Environments for Cloning Unikernel-based VMs[EuroSys'23]: [[Paper](http://nets.cs.pub.ro/~costin/files/nephele.pdf)] [[Note](notes/hypervisor/nephele/nephele.md)]
- [x] Honeycomb: Secure and Efficient GPU Executions via Static Validation[OSDI'23]: [[Paper](papers/hypervisor/osdi23-mai.pdf)] [[Note](notes/hypervisor/honeycomb/honeycomb.md)]

## RISC-V
- [x] A First Look at RISC-V Virtualization from an Embedded Systems Perspective[TC'21]: [[Paper](https://arxiv.org/pdf/2103.14951.pdf)]
- [x] CVA6 RISC-V Virtualization: Architecture, Microarchitecture, and Design Space Exploration[arxiv'23]: [[Paper](https://arxiv.org/pdf/2302.02969.pdf)]
