

# AVO: Agentic Variation Operators for Autonomous Evolutionary Search

- arXiv:2603.24517v1 [cs.LG]（2026年3月25日发布
- **作者**：Terry Chen, Zhifan Ye, Bing Xu, Zihao Ye, Timmy Liu, Ali Hassani, Tianqi Chen, Andrew Kerr, Haicheng Wu, Yang Xu, Yu-Jung Chen, Hanfeng Chen, Aditya Kane, Ronny Krashinsky, Ming-Yu Liu, Vinod Grover, Luis Ceze, Roger Bringmann, John Tran, Wei Liu, Fung Xie, Michael Lightstone, Humphrey Shi (NVIDIA) [cite: 3]

---

## 2. 论文目的
该论文旨在引入一种名为**智能体变异算子 (AVO)** 的新型进化算法组件。其核心目的是打破传统进化搜索中 LLM 仅作为“单轮候选代码生成器”的局限性。通过将变异过程提升为具备自主规划、工具使用和自我修正能力的**深度智能体循环**，AVO 能够自主发现关键的微架构级优化方案，从而在当前最先进的 GPU 硬件（NVIDIA Blackwell）上产生超越人类专家手工编写的高性能算子内核。

---

## 3. 详细工作流 (Workflow)
AVO 将传统的变异和交叉过程替换为一个持续运行的自主循环，具体流程如下：

* **输入阶段**：系统接收当前的解谱系（Population $\mathcal{P}_t$）、领域知识库（$K$，包含 CUDA/PTX 文档和参考代码）以及评分函数（$f$，用于验证正确性和测量吞吐量）。
* **智能体循环 (Main Agent Loop)**：
    * **规划 (Planning)**：智能体查阅硬件手册并分析既有方案的 Profiling 数据，提出编辑建议。
    * **实施 (Implementation)**：自主编写包含内联 PTX 代码的 CUDA 内核。
    * **评估 (Evaluation)**：通过自动化工具链运行编译，并调用评分函数 $f$ 进行数值正确性和性能测试。
    * **纠错与优化 (Bug-Fixing)**：若测试失败或性能未达标，智能体将利用 Profiler 反馈诊断问题、修复 Bug 并调整优化策略，循环直至提交更优版本。
* **持续进化与持久化**：每一个被确认的改进版本都会作为 Git 提交进行记录，确保存储完整的演化路径。
* **自我监督 (Self-Supervision)**：引入一个 **Supervisor Agent** 监控演化轨迹。当主智能体进入死循环或探索停滞时，监督者会介入并强行引导至新的搜索方向。

---

## 4. 关键微架构优化 (Micro-architectural Optimizations)
AVO 通过自主探索发现了三个关键的、涉及硬件底层推理的优化方案：

* **无分支累加器重缩放 (Branchless Accumulator Rescaling)**：
    * **改动**：在 Online Softmax 计算中，将原本带有 `if` 条件判断的分支改为无分支的谓词选择路径（Predicated Select）。
    * **原理**：虽然增加了少量计算成本，但消除了 Warp 同步开销，并允许使用更轻量级的非阻塞内存屏障（Non-blocking fence）。
* **校正与 MMA 流水线重叠 (Correction/MMA Pipeline Overlap)**：
    * **改动**：重构计算流水线，使校正 Warp 在第一阶段的 PV GEMM 计算完成后立即开始标准化处理。
    * **原理**：将串行执行转化为流水线重叠执行，大幅减少了校正 Warp 的空闲时间。
* **寄存器重平衡 (Register Rebalancing)**：
    * **改动**：在 Blackwell 架构的固定寄存器预算下，将部分寄存器配额从 Softmax Warp 组重新分配给校正 Warp 组。
    * **原理**：缓解了关键路径上的校正 Warp 因寄存器压力导致的局部内存溢出（Spilling to local memory），从而减少了访存停顿。

---

## 5. 最终结果
在 NVIDIA Blackwell B200 GPU 上的基准测试显示：

* **多头注意力 (MHA)**：
    * 相比 **cuDNN**，性能提升高达 **3.5%**。
    * 相比 **FlashAttention-4**，性能提升高达 **10.5%**。
    * 峰值吞吐量达到 **1668 TFLOPS** (BF16)。
* **组查询注意力 (GQA)**：
    * 通过对演化出的 MHA 内核进行仅 **30 分钟** 的自主适配，即可获得支持。
    * 相比 cuDNN 提升达 **7.0%**，相比 FlashAttention-4 提升达 **9.3%**。
* **进化效率**：在 7 天的连续运行中，智能体内部探索了超过 **500 个** 优化方向，提交了 **40 个** 改进版本。
