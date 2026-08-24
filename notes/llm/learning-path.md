# LLM Training, Inference & Algorithms — Recommended Reading Path

这份路线面向已有 GPU kernel、编译器或体系结构经验，但希望系统补齐现代 LLM 算法、训练、推理与评测知识的工程师。完整论文索引见仓库根目录的 [README](../../README.md)；本文件解决的是“先读什么、依赖什么、读完要能做什么”。

## 优先级与使用方式

- **P0 — 主干**：不掌握就很难正确理解现代训练或推理系统。12 周计划以 P0 为主。
- **P1 — 深化**：帮助比较不同设计，适合在对应周追加一篇。
- **P2 — 专题**：面向具体工作再读，例如 MoE、极低比特量化、长上下文或安全。

不要以“看完 PDF”为完成标准。每篇至少回答下面五个问题：

1. 优化目标、输入输出和张量形状是什么？
2. forward、backward 或 decode 分别保存哪些状态？
3. 计算量、显存量、通信量和关键同步点在哪里？
4. 论文用什么指标证明有效，baseline 是否公平？
5. 如果落到 GPU kernel/runtime，最可能改变哪个算子、layout、collective 或 scheduler？

推荐每周投入 6–8 小时：两篇精读、两到三篇结构化速读，以及一个可验证的小产出。

## 依赖关系

```text
反向传播 / 优化器 / 数值精度
        │
        ├── 自回归语言建模 ── Transformer ── 现代 LLM block
        │                                 │
        │                                 ├── Scaling law / 数据 / 预训练
        │                                 └── LoRA / SFT / 偏好优化 / RL
        │
        ├── activation 与 optimizer state
        │       └── DP / TP / PP / ZeRO ── 3D 并行 ── MoE / 长上下文
        │
        └── prefill / decode / KV cache
                ├── batching / scheduling / P-D 分离
                ├── speculative decoding
                └── weight / activation / KV-cache quantization

所有路线最终汇合到：质量评测、效率评测、安全评测与可复现实验。
```

## 12 周主线

### Week 1 — 从算子 backward 到“训练算法”

**P0 论文**

1. [Learning Representations by Back-Propagating Errors](https://www.nature.com/articles/323533a0)
2. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
3. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
4. [Layer Normalization](https://arxiv.org/abs/1607.06450)

**要回答**：参数、activation、gradient、master weight、Adam `m/v` 分别在何时产生和释放？AdamW 为什么不是简单的 L2 loss？LayerNorm backward 需要哪些 reduction？

**实践产出**：画一张单个 Transformer block 训练步骤的数据流图，并按 dtype 列出参数、梯度、optimizer state 和 activation 的显存公式。

### Week 2 — 自回归语言模型与现代 Transformer block

**P0 论文**

1. [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html)
2. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
4. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
5. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

**要回答**：teacher forcing 和 autoregressive decode 的区别是什么？tokenizer 如何改变 sequence length 和词表 GEMM？Pre-Norm/Post-Norm、RMSNorm、SwiGLU 分别改变哪些算子和 activation？

**实践产出**：为一个 LLaMA-style block 写出逐算子的 shape table，覆盖 prefill、单 token decode 和 backward。

### Week 3 — 位置编码、Scaling Law、数据与现代 LLM 配方

**P0 论文**

1. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
2. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
3. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
4. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
5. [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)

**要回答**：RoPE 为什么只作用于 Q/K？固定训练 FLOPs 时参数和 token 如何分配？模型质量、训练成本和推理成本为什么不是同一个最优点？数据去重如何影响记忆和评测污染？

**实践产出**：选择一个 7B 模型，估算一次完整预训练所需 token、FLOPs、GPU-days，并写明所有假设。

### Week 4 — 混合精度、activation 与训练显存

**P0 论文**

1. [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
2. [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
3. [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
4. [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)

**要回答**：loss scaling 解决哪种数值问题？BF16 与 FP16 的风险有何不同？checkpoint boundary 如何改变峰值显存和额外 FLOPs？selective recomputation 为什么优于整层重算？

**实践产出**：用表格比较 baseline、full checkpoint、selective checkpoint 和低秩 optimizer 四种方案的显存与额外计算。

### Week 5 — DP、TP、PP 与 ZeRO 的基本构件

**P0 论文**

1. [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://www.vldb.org/pvldb/vol13/p3005-li.pdf)
2. [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
3. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
4. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

**要回答**：DP 的 AllReduce、TP 的 AllReduce/AllGather、PP 的 send/recv 分别位于哪里？ZeRO-1/2/3 各切分什么？microbatch 数量如何影响 bubble 与 activation memory？

**实践产出**：为 `DP=8, TP=4, PP=2` 的模型画 rank topology，并标出一个 iteration 的所有 collective。

### Week 6 — 3D 并行、自动切分与万卡训练

**P0 论文**

1. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
2. [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663)
3. [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin)
4. [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://www.usenix.org/conference/nsdi24/presentation/jiang-ziheng)

**要回答**：并行维度应如何映射 NVLink 域和跨节点网络？sharding propagation 能自动解决什么、不能解决什么？MFU 下降究竟来自 kernel、通信、bubble、数据还是故障？

**实践产出**：写一份 64/512/4096 GPU 三种规模的并行配置建议，说明拓扑、batch 和容错假设。

### Week 7 — MoE：算法路由到 block-sparse kernel

**P0 论文**

1. [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
2. [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
3. [Switch Transformers](https://arxiv.org/abs/2101.03961)
4. [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
5. [DeepSeekMoE](https://arxiv.org/abs/2401.06066)

**要回答**：capacity factor、token dropping、load-balancing loss 和 expert parallel 分别解决什么？dispatch/combine 为什么会变成 all-to-all？MegaBlocks 如何把不规则 token 数映射到 block-sparse GEMM？

**实践产出**：针对一个 top-2 MoE 层，写出 route → permute → all-to-all → grouped GEMM → combine 的 shape 与通信量。

### Week 8 — SFT、参数高效微调与指令数据

**P0 论文**

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
3. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
4. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
5. [Self-Instruct](https://arxiv.org/abs/2212.10560)

**要回答**：预训练、SFT 与 preference tuning 的 loss 和数据分别是什么？LoRA rank 改变哪些矩阵乘？QLoRA 的 NF4、double quantization 与 paged optimizer 分别省什么？

**实践产出**：设计一个小规模 SFT 实验，列出数据 schema、packing、loss mask、LoRA target modules 和显存预算。

### Week 9 — PPO、DPO 与 reasoning RL

**P0 论文**

1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
3. [DeepSeekMath](https://arxiv.org/abs/2402.03300)
4. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

**要回答**：policy、reference、reward、value/critic 和 rollout 各是什么模型？PPO clipping 与 KL penalty 各限制什么？DPO 为什么不需要在线 rollout？GRPO 如何移除独立 critic？

**实践产出**：画出 SFT、PPO-RLHF、DPO、GRPO 四条训练 pipeline，并比较模型副本数、显存和同步点。

### Week 10 — 解码与量化

**P0 论文**

1. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
2. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)
3. [SmoothQuant](https://arxiv.org/abs/2211.10438)
4. [AWQ](https://arxiv.org/abs/2306.00978)
5. [KIVI](https://arxiv.org/abs/2402.02750)

**要回答**：temperature、top-k、top-p 如何改变采样分布？speculative decoding 为何保持目标分布？W8A8、W4A16 与 KV2 分别受 compute、bandwidth 还是容量限制？

**实践产出**：为一个具体 batch/sequence/model 配置做 roofline 判断，给出 FP16、W8A8、W4A16、KV2 的预期瓶颈。

### Week 11 — LLM Serving：从 continuous batching 到 P/D 分离

**P0 论文**

1. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
2. [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
3. [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
4. [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
5. [SGLang](https://arxiv.org/abs/2312.07104)

**要回答**：TTFT、TPOT、goodput、throughput 和 tail latency 如何冲突？continuous batching 的调度单位是什么？PagedAttention 管理的究竟是逻辑 block 还是物理 page？chunked prefill 与 P/D 分离分别处理什么干扰？

**实践产出**：写一个离散事件 scheduler 草图，至少模拟 arrival、prefill、decode、KV allocation、preemption 和 completion。

### Week 12 — 评测、安全与可复现结题

**P0 论文**

1. [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
2. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
3. [RewardBench](https://arxiv.org/abs/2403.13787)
4. [HarmBench](https://arxiv.org/abs/2402.04249)
5. [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782)

**要回答**：quality 和 system efficiency 应怎样联合报告？LLM judge 有哪些位置、长度和自我偏差？如何控制 contamination、prompt template、tokenizer 与随机性？安全评测为什么必须同时测 under-refusal 和 over-refusal？

**实践产出**：完成一份结题设计文档：选择一个模型和 workload，给出训练/后训练/Serving 方案、成本模型、指标、baseline、风险与复现实验清单。

## P1：每条主线的下一步

| 主线 | 推荐论文 | 何时追加 |
|------|----------|----------|
| 优化与训练配方 | [Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html)、[μP](https://arxiv.org/abs/2203.03466)、[Sophia](https://arxiv.org/abs/2305.14342) | 需要研究 optimizer memory、超参迁移或收敛速度时 |
| 数据工程 | [DoReMi](https://arxiv.org/abs/2305.10429)、[RefinedWeb](https://arxiv.org/abs/2306.01116)、[Dolma](https://arxiv.org/abs/2402.00159)、[DataComp-LM](https://arxiv.org/abs/2406.11794)、[FineWeb](https://arxiv.org/abs/2406.17557) | 需要构建或比较预训练数据 pipeline 时 |
| 分布式训练 | [ZeRO-Offload](https://www.usenix.org/conference/atc21/presentation/ren-jie)、[ZeRO-Infinity](https://arxiv.org/abs/2104.07857)、[Oobleck](https://arxiv.org/abs/2309.08125)、[Ulysses](https://arxiv.org/abs/2309.14509) | 研究 offload、容错或长序列训练时 |
| MoE 系统 | [GLaM](https://arxiv.org/abs/2112.06905)、[DeepSpeed-MoE](https://arxiv.org/abs/2201.05596)、[FasterMoE](https://arxiv.org/abs/2202.11436)、[Tutel](https://arxiv.org/abs/2206.03382) | 开始优化 expert parallel 和 all-to-all 时 |
| 后训练 | [Constitutional AI](https://arxiv.org/abs/2212.08073)、[LIMA](https://arxiv.org/abs/2305.11206)、[IPO](https://arxiv.org/abs/2310.12036)、[KTO](https://arxiv.org/abs/2402.01306)、[SimPO](https://arxiv.org/abs/2405.14734) | 比较数据质量、RLAIF 与 reference-free preference loss 时 |
| 解码 | [SpecInfer](https://arxiv.org/abs/2305.09781)、[Medusa](https://arxiv.org/abs/2401.10774)、[EAGLE](https://arxiv.org/abs/2401.15077)、[Lookahead Decoding](https://arxiv.org/abs/2402.02057) | 优化低 batch latency 或 speculative tree verification 时 |
| 量化 | [LLM.int8()](https://arxiv.org/abs/2208.07339)、[GPTQ](https://arxiv.org/abs/2210.17323)、[SpQR](https://arxiv.org/abs/2306.03078)、[QuaRot](https://arxiv.org/abs/2404.00456) | 需要比较 outlier、校准和 kernel 可实现性时 |
| Serving | [FlexGen](https://proceedings.mlr.press/v202/sheng23a.html)、[AlpaServe](https://www.usenix.org/conference/osdi23/presentation/li-zhouhan)、[Punica](https://arxiv.org/abs/2310.18547)、[S-LoRA](https://arxiv.org/abs/2311.03285)、[Splitwise](https://arxiv.org/abs/2311.18677) | 研究 offload、多模型、多 LoRA 或异构 P/D 集群时 |
| 评测与安全 | [TruthfulQA](https://arxiv.org/abs/2109.07958)、[IFEval](https://arxiv.org/abs/2311.07911)、[Chatbot Arena](https://arxiv.org/abs/2403.04132)、[XSTest](https://arxiv.org/abs/2308.01263)、[WMDP](https://arxiv.org/abs/2403.03218) | 建立特定产品或研究任务的评测矩阵时 |

## P2：按工作专题选读

- **非 Transformer 架构**：[Hyena](https://arxiv.org/abs/2302.10866) → [RWKV](https://arxiv.org/abs/2305.13048) → [RetNet](https://arxiv.org/abs/2307.08621) → [Mamba](https://arxiv.org/abs/2312.00752) → [Mamba-2](https://arxiv.org/abs/2405.21060)。
- **长上下文训练与推理**：[Ring Attention](https://arxiv.org/abs/2310.01889)、README 中的 MQA/GQA/MLA、LoongTrain、LoongServe、Quest、NSA、MSA 与 DSA。
- **极低比特与新数值格式**：[AQLM](https://arxiv.org/abs/2401.06118)、[BitNet b1.58](https://arxiv.org/abs/2402.17764)、[QuaRot](https://arxiv.org/abs/2404.00456)。
- **推理集群与 KV 基础设施**：README 中的 Mooncake、NanoFlow、FlashInfer、SGLang、Preble 与 megakernel 系列。
- **安全攻防**：[GCG](https://arxiv.org/abs/2307.15043)、[Sleeper Agents](https://arxiv.org/abs/2401.05566)、[StrongREJECT](https://arxiv.org/abs/2402.10260)、[WMDP](https://arxiv.org/abs/2403.03218)。

## 三个毕业项目

### A. 训练闭环：从 tokenizer 到 checkpoint

训练一个 50M–300M 参数的 decoder-only 模型。自己确定 tokenizer、数据去重、sequence packing、AdamW、学习率、混合精度和 checkpoint 策略；至少比较一次是否重算、两种 batch size 和两种精度。最终报告 loss curve、tokens/s、峰值显存、MFU 估计、失败/恢复流程和评测污染检查。

### B. 推理闭环：KV cache 与调度模拟器

实现一个离散事件模拟器或最小 runtime，支持 continuous batching、paged KV、chunked prefill、preemption 和 P/D worker。输入真实或合成 arrival trace，报告 TTFT、TPOT、P50/P99 latency、goodput、KV 利用率和调度公平性；再加入一种 speculative decoding 或 KV quantization 策略比较收益。

### C. 系统设计：70B 模型从预训练到线上服务

在给定 GPU 型号、数量、网络拓扑和 SLO 下，完成一份可评审设计：数据/token 预算、DP/TP/PP/EP、ZeRO、checkpoint/容错、SFT/偏好优化、量化、Serving 架构、容量规划、质量/安全评测与总成本。所有数字必须能追溯到公式、profile 或论文实验，而不是经验口号。

## 算子工程师的论文笔记模板

```markdown
# Paper Title

## 1. 一句话结论
- 它解决什么问题，核心机制是什么？

## 2. 前置与 baseline
- 前置论文/概念：
- 最强 baseline：
- 论文真正改变的变量：

## 3. 算法
- 目标函数：
- 输入/输出：
- 关键公式：
- 训练与推理差异：

## 4. Shape 与数据流
| 阶段 | Tensor | Shape | DType | 保存到 backward/KV? |
|------|--------|-------|-------|----------------------|

## 5. 成本模型
- FLOPs：
- HBM bytes：
- 峰值显存：
- 通信量与 collective：
- 同步点 / pipeline bubble：

## 6. Kernel / Runtime 映射
- 对应算子：
- layout / tiling：
- 可融合位置：
- scheduler / cache 影响：
- 预计瓶颈：compute / bandwidth / latency / communication：

## 7. 实验可信度
- 硬件、模型、batch/sequence：
- 指标与 SLO：
- baseline 是否同精度、同质量、同硬件：
- 缺失的 ablation：

## 8. 复现计划
- 最小实验：
- 预期结果：
- 失败判据：

## 9. 未解决问题
- 
```

## 进度 Checklist

- [ ] Week 1：能解释 AdamW、optimizer state 与 LayerNorm backward
- [ ] Week 2：能写出 LLaMA block 的训练/prefill/decode shape
- [ ] Week 3：能估算训练 FLOPs、token budget 和数据去重影响
- [ ] Week 4：能比较 loss scaling、checkpoint 和 selective recomputation
- [ ] Week 5：能画出 DP/TP/PP/ZeRO 的通信拓扑
- [ ] Week 6：能为给定集群选择 3D parallel 配置并解释 MFU
- [ ] Week 7：能推导 MoE dispatch、all-to-all 与 grouped GEMM
- [ ] Week 8：能设计 LoRA/QLoRA SFT 数据和显存预算
- [ ] Week 9：能比较 PPO、DPO 与 GRPO 的模型副本和同步点
- [ ] Week 10：能判断 speculative decoding 与量化方案的适用瓶颈
- [ ] Week 11：能解释 continuous batching、PagedAttention、chunked prefill 和 P/D 分离
- [ ] Week 12：能给出同时覆盖质量、效率、安全和复现性的实验设计

完成 P0 后，不必按顺序读完所有 P1/P2。应以实际项目的瓶颈为入口回到 README：遇到通信问题读训练系统，遇到 decode latency 读 speculative decoding，遇到容量问题读量化/KV cache，遇到质量问题先检查数据、训练目标和评测，而不是直接优化 kernel。
