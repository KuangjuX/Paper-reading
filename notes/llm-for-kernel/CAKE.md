---
tags:
  - papers/llm-for-kernel
  - papers/gpu-compilers
  - papers/agent-systems
aliases:
  - "CAKE"
  - "Compiler-Agent Co-Design"
date: 2026-08-12
arxiv: 2608.16292
---

# CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution

## 核心信息

- **标题**：CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution
- **作者**：Zihao Ye, Yingyi Huang, Hongyi Jin, Bohan Hou, Junru Shao, Zhongming Yu, Jinqi Chen, Meghan Cowan, Shiyi Cao, Shanli Xing, Hanfeng Chen, Vinod Grover, Tianqi Chen, Luis Ceze
- **机构**：NVIDIA、Carnegie Mellon University
- **发表时间**：2026-08-12
- **类型**：arXiv preprint
- **论文链接**：https://arxiv.org/abs/2608.16292
- **领域**：GPU Kernel Optimization / Compiler / Coding Agent

## 一句话总结

CAKE 不再让智能体只在固定编译器上反复修改 GPU kernel，而是让 **kernel、面向智能体的 IR、静态验证器、性能模型和编译器能力共同演化**，把一次性的错误与优化经验沉淀为可以跨 kernel 复用的编译器知识。

## 研究动机

现有 GPU kernel 智能体通常采用黑盒式循环：生成 CUDA/PTX 或 DSL 代码，编译，检查数值正确性，测量延迟，再根据结果修改。这个流程能够完成局部调优，但反馈粒度不足：

- 编译错误或运行崩溃通常不能指出具体违反了哪项同步、资源或硬件约束；
- 数值错误难以定位到特定的数据流或 producer-consumer 交接；
- 单一延迟数字不能解释瓶颈来自计算、访存、同步、流水线还是资源占用；
- 当前沿 workload 需要一种语言尚未支持的调度结构时，智能体无法通过修改 kernel 本身补齐语言能力。

现有 GPU DSL 又存在两难：Triton、TileLang、cuTile 等 tile-level DSL 易于使用，但会隐藏 warp specialization、barrier choreography 和 memory-tier placement；CUDA/PTX、CuTe DSL 等低层表示能暴露硬件控制，却要求开发者掌握复杂的地址、同步和 layout calculus。

CAKE 的目标是在二者之间提供一个适合智能体搜索的表示与反馈环境：保留影响性能的物理调度决策，同时由编译器承担机械性的底层元数据生成和合法性验证。

## 核心贡献

1. **Cake IR**：一种经过类型检查、硬件决策显式、但不要求智能体操作 layout algebra 的 GPU schedule IR。
2. **结构化 compiler harness**：在实际编译和 GPU 运行前提供局部化的安全、硬件一致性、数据一致性和调度语义诊断，并用成本模型过滤和排序候选。
3. **Compiler-agent co-evolution**：将反复出现的失败转化为 verifier 规则、IR primitive、lowering 支持、成本模型校准或可复用优化策略。
4. **从单 shape 到 library portfolio 的独立泛化阶段**：显式构建 dispatcher、shape bucket 和 fallback，并用未见 shape 检查泛化。
5. **生产级实证**：在 Flash-KMeans、Kimi Delta Attention、TinyGEMM、Alpha-MoE、KNN/KMeans portfolio 以及 11 个已知专家 kernel 上进行验证，并形成四项上游 PR。

## Cake IR

### 显式表达物理调度

Cake IR 记录“如何驱动硬件”，包括：

- 哪些 warp 或 warp group 承担 load、MMA、epilogue 等角色；
- shared memory、tensor memory 等缓冲区的形状、数据类型、所有权与生命周期；
- pipeline 的 stage 数；
- producer-consumer 之间由哪个 barrier 完成交接；
- 使用哪一种矩阵乘、内存传输或同步指令；
- grid、persistent scheduling 和 multi-CTA coordination。

但 barrier 地址、phase bit、TMEM offset、TMA descriptor encoding 和 warp identity 等机械元数据由 lowering 自动推导。因此 Cake IR 的分工是：

> 调度声明做什么，lowering 推导具体怎么编码。

### 四个关键属性

- **Type-checked vocabulary**：计算、访存、同步、数学和 warp 控制来自固定操作集合。
- **Declared resources**：缓冲区、同步对象、warp role 和 pipeline 统一声明。
- **Explicit roles**：跨 warp role 的数据交接直接出现在程序中。
- **Auto-derived metadata**：底层地址、phase 和 descriptor 等不由智能体手写。

### 不把 Layout 作为一等抽象

与 CuTe layout algebra、Triton linear layout 和 Axe named-axis abstraction 不同，Cake IR 不要求智能体构造独立的 layout 对象。智能体直接写下具体硬件承诺，例如：

- SMEM view offset；
- operand byte offset；
- TMEM column range；
- swizzle tag；
- TMA descriptor coordinate。

编译器沿数据流检查这些承诺是否相互一致，并验证它们是否满足目标指令和资源约束。这样既保留硬件控制，又缩小了智能体必须操作的抽象表面。

## Compiler Harness

Harness 向智能体暴露以下分析能力：

| 类别 | 处理方式 | 目的 |
|---|---|---|
| Program safety | 编译前硬门禁 | 检查同步、执行顺序和内存使用风险 |
| Hardware conformance | 编译前硬门禁 | 检查资源、指令和目标架构约束 |
| Data consistency | 编译前硬门禁 | 检查数据流和 producer-consumer 表示兼容性 |
| Schedule semantics | 编译前硬门禁 | 检查角色、pipeline 和资源声明的结构不变量 |
| Numerical validation | 执行阶段硬门禁 | 与权威外部参考实现比较 |
| Performance analysis | 报告 | 估算成本并归因主要瓶颈 |
| Optimization guidance | 提示 | 建议可能有效、但不阻塞编译的修改 |

静态检查和成本模型只负责提前过滤、排序和解释候选；最终正确性与性能仍以真实 GPU 执行、外部 oracle 和 CUPTI 计时为准。

## Compiler 与 Kernel 的共同演化

### Kernel 内循环

1. 生成结构上不同的 Cake IR 候选；
2. 经过 IR 构造检查、verifier 硬门禁和成本模型排序；
3. 对幸存候选执行编译、数值验证、benchmark 和 profiling；
4. 根据证据判断应该修改候选、verifier、成本模型还是 IR 词汇表。

### Compiler-Harness 外循环

编译器演化有两条证据来源：

- **主动路径**：从生产 kernel 和硬件文档中识别新的 Blackwell 指令、资源、descriptor、同步习惯和 schedule pattern。
- **反馈路径**：从 sanitizer、运行失败、正确性 mismatch 和性能模型误判中抽取反复出现或代价较高的问题。

典型转化包括：

- 反复出现的运行时崩溃 → 新的静态 verifier 规则；
- 经常非法的 lowering pattern → 编译前检查；
- 系统性的性能误判 → 成本模型校准任务；
- 不能表达的生产调度 → 新 IR primitive 和 lowering 支持。

所有 compiler change 都需要经过 kernel corpus 回归测试和人工 merge gate。论文固定了基础模型与 agent scaffold，演化对象主要是领域编译器环境，而不是智能体模型本身。

## Cake IR 的形成过程

Cake IR 不是预先完整设计的，而是从生产 kernel 自底向上演化出来：

1. 收集生产级 CUDA kernel；其他 DSL kernel 先由智能体翻译成 CUDA/inline PTX。
2. 识别 barrier choreography、pipeline staging、warp-role partitioning、TMA descriptor 和 TMEM accumulator lifecycle 等重复模式。
3. 结合 Blackwell 编程模型，由人类知识引导抽象设计。
4. 按八项原则检查候选抽象：易编辑、性能透明、规范化、静态类型化、易分析、测试门禁、分析一致、硬件语义明确。
5. 不断移植新 kernel；移植失败就暴露新的 IR 或 lowering 缺口。

这种设计意味着 Cake IR 已经编码了生产 kernel corpus 与专家硬件知识，它并不是知识中立的表示。

## Agent Workflow 与实验协议

每个任务由稳定的 workload contract 约束，包括 shape、正确性 oracle、数值容差、硬件和允许访问的参考资料。论文中的任务统一使用：

- **模型**：GPT-5.6-sol；
- **推理强度**：xhigh；
- **主要硬件**：NVIDIA B200；
- **测量**：GPU 上正确性检查、CUPTI timing，每个 timed sample 前刷新 L2 cache。

在 clean-start 和 frontier-kernel synthesis 中，智能体可以查看数学描述、评测契约、正确性 oracle 和高层代码，但不能查看已有 CUDA、PTX、SASS 或等价生成代码；外部实现只可作为黑盒性能基线。已知 kernel reproduction 则允许查看参考实现。

## 核心实验结果

### 1. Flash-KMeans Clean Start

目标是 Flash-KMeans 的 `assign` kernel：`B=32, N=65,536, K=1,024, D=128`，BF16 输入、FP32 累加。调优后的 FlashML Triton 基线为 0.938 ms。

实验固定模型、agent scaffold、任务、oracle、benchmark 和单一目标 shape，分别让智能体编写 Cake IR 或直接编写 CUDA C++/inline PTX。每组进行三次独立运行，预算为 8000 万 token。

| 指标 | Cake IR | Direct CUDA/PTX |
|---|---:|---:|
| 80M token 前达到预设 plateau | 3/3 | 0/3 |
| 最佳性能中位数 | 1.144× baseline | 0.928× baseline |
| 最佳性能范围 | 1.041–1.205× | 0.852–1.151× |
| Active evolve time 中位数 | 1.89 h | 3.73 h |

Cake IR 三次运行的平均性能约在 5500 万 token 时越过基线，之后继续提升；CUDA/PTX 组的平均值在 8000 万 token 截止时仍低于基线。结果表明，结构化表示与局部诊断可以减少无效搜索，但实验规模只有一个 workload、一个 shape 和每组三次运行。

### 2. Kimi Delta Attention

智能体不能查看 FlashKDA 的低层实现，只能将其作为黑盒性能基线：

- KDA prefill 覆盖 fixed、packed-variable 和 tail 输入，在六个 B200 BF16 shape 上取得 **2.05× 几何平均加速**；
- 满足论文验证契约下的 bitwise correctness；
- 在 SGLang 的 Kimi-K3 端到端 serving 中完成验证；
- KDA decode 在 30 个公共 API shape 上相对上游 FlashInfer 获得 **1.14× 几何平均加速**。

KDA 包含需要跨 chunk 保持的 recurrent state，比普通 GEMM + epilogue 更能检验 IR 对复杂状态与调度的表达能力。

### 3. TinyGEMM

从 FlashInfer 中 TensorRT-LLM 派生的小 M BF16 kernel 出发，智能体构建了浅/深 pipeline、PDL 变体和 batch size 小于 8 的专门路径：

- 35 个标准 shape 上 kernel time 几何平均降低 **18%–23%**；
- GPT-OSS-20B/120B greedy decoding 保持 bitwise identical；
- SGLang GPT-OSS-120B、concurrency 128、TP1 下最高报告 **7.6% output throughput 提升**；
- TP4 下差异位于测量噪声范围。

### 4. Alpha-MoE Blackwell Rewrite

智能体把原本面向 Hopper 的 Alpha-MoE W8A8 megakernel 重写到 Blackwell，将 routed gather、两个 projection、activation、requantization 和 route-weighted output accumulation 融入一个设备程序。

相对 FlashInfer 的 TensorRT-LLM-derived pre-routed API：

- `N=256` 时 API 级加速 **6.204×**，GPU span 加速 **1.215×**；
- `N=512` 时 API 级加速 **4.025×**，GPU span 加速 **1.170×**。

API 与 GPU-span 数字的差距说明，大幅端到端提升还来自减少 GPU activity、调度间隙和 workspace 管理，而不只是 kernel 算术本身更快。

### 5. 已知专家 Kernel 复现

论文对比 TensorRT-LLM、CUTLASS、DeepGEMM、FlashAttention-4 和 FlashInfer 的 11 个 LLM 关键 kernel：

- 10/11 达到或超过参考实现；
- 唯一未达到参考的 DSv4 sparse MLA FP8 decode 达到 **0.9649×**；
- FP8/FP4 MQA indexer 分别达到约 **1.270×/1.273×**；
- CUTLASS MLA decode 达到 **1.2174×**；
- FlashAttention-4 BF16 forward/backward 分别达到 **1.0045×/1.0470×**。

所有 Cake IR 实现的物理代码行数都少于审计的参考 device code，但作者明确指出语言和统计范围不同，不能据此直接声称更高的开发效率或可读性。

## 从单一 Shape 到 Library Portfolio

CAKE 将单 shape 优化与通用 library 构建视为两个目标不同的阶段。强单 shape seed 产生后，泛化阶段才会：

1. 将 shape 划入 bucket；
2. 产生共享或专用 physical schedule；
3. 在显式 fallback 前组织 dispatcher guard；
4. 检查代表性输入、held-out 输入、边界/tail、guard 重叠或遗漏以及 fallback；
5. 用包含 dispatcher 开销的固定 workload 进行整体评分。

有效 shape domain 在调优前声明，dispatcher 不能为了结果临时引入有利的评测 shape。GB200 上的完整 portfolio 结果为：

| Kernel family | Shape 数量 | Dispatcher-inclusive 几何平均加速 |
|---|---:|---:|
| KNN build | 112 | 1.418× |
| KNN search | 198 | 2.116× |
| KMeans | 124 | 1.803× |

所有输出均正确，KNN recall 为 1.0。论文强调这些结果与 Flash-KMeans 单 shape clean-start 使用不同主机、shape 分布、baseline 和协议，二者差异不能被解释为“泛化成本”。

## 系统覆盖

- 超过 400 个静态分析和编译案例；
- 399 个 GPU correctness case；
- 约 28 个 kernel family；
- 超过 100 个 TensorRT-LLM port；
- 覆盖 attention、linear attention、dense/grouped/quantized GEMM、MoE、normalization、quantization、Top-K、KNN、KMeans、state-space model 和 fused graph kernel；
- 支持 NVIDIA Ampere 到 Blackwell，包括 A100、H100/H200、B200/B300、RTX 5090 和 GB10 等目标；
- KDA prefill、KDA decode、TinyGEMM2 和 Alpha-MoE 四项修改形成上游 PR。

编译器要求 exact target match，不会静默降级到其他架构。性能模型只对 B200 和 H100 做了证据支持的校准，其他平台明确报告 coverage limitation。

## 核心洞察

### 1. 将隐性 GPU 专家知识转化为可执行知识

传统专家经验通常存在于人的直觉、代码习惯或调试过程里。CAKE 将这些经验编码为 IR primitive、类型规则、verifier、lowering、成本模型和可复用 tactic，使其能够被自动检查并跨任务复用。

### 2. 优化智能体所处的环境，而不只是智能体本身

相关工作通常优化模型、搜索算法、变异策略或历史记忆。CAKE 固定基础模型与 agent scaffold，将重点放在“被搜索的表示”和“环境返回的证据”上。它的长期价值来自环境持续吸收任务经验。

### 3. 生成便宜后，验证和理解成为新瓶颈

当智能体可以大量生成候选程序时，关键问题不再只是生成能力，而是能否快速判断合法性、定位失败原因、理解性能，并把经验沉淀下来。CAKE 本质上是在构建这样一个验证与知识吸收层。

## 局限与审慎解读

1. **主要证据集中在 B200**：跨 Ampere–Blackwell 的语言与 lowering 覆盖不等于跨平台性能证据；非 NVIDIA 后端尚未测量，移植需要重建 backend。
2. **静态分析不是正确性证明**：只覆盖已建模的语义，可能有 false positive/negative，也无法捕获全部微架构行为，GPU 执行仍是最终依据。
3. **Clean-start 对照规模较小**：只有一个 workload、一个固定 shape、每组三次运行，不能据此证明所有 kernel 上都有同等优势。
4. **“Clean start”不等于环境没有先验**：智能体看不到目标低层实现，但 Cake IR/harness 已从生产 kernel corpus 中演化出来；优势来自表示与环境中累积的领域知识，而不只是语法更简洁。
5. **演化成本高**：Flash-KMeans 每次使用 8000 万 token，KDA 轨迹达到十亿级累计 token；论文较少讨论推理成本和维护 compiler harness 的人力如何摊销。
6. **不同案例的 speedup 不可直接横向比较**：硬件、shape、baseline、API/GPU-span 口径和协议不同，尤其不能把 Alpha-MoE 的 API 级 6.204× 当作 kernel 执行本身的加速。
7. **仍有人工门控**：compiler evolution 并非完全自主，IR 和分析更新仍需 corpus test 与 human merge gate。

## 我的评价

CAKE 最重要的贡献不是某一个 kernel 的 1.2× 或 2× 加速，而是提出了一个适合长期积累的系统边界：把 kernel agent 的失败从一次性的搜索噪声，转化为编译器可以验证、复用和回归测试的领域知识。

其最有说服力的证据是 Cake IR 与直接 CUDA/PTX 的 matched clean-start 对照，以及 KDA、TinyGEMM 和 Alpha-MoE 的生产集成案例；最需要谨慎之处则是实验样本量、巨大的 token 预算，以及 Cake 环境已经吸收了生产 kernel 先验这一事实。

可以将 CAKE 理解为对 GPU kernel agent 研究方向的一次重心转移：

> 不仅搜索更好的程序，也持续改造让程序可表达、可验证、可解释和可优化的环境。

## 相关笔记

- [[notes/llm-for-kernel/avo|AVO: Agentic Variation Operators for Autonomous Evolutionary Search]]

