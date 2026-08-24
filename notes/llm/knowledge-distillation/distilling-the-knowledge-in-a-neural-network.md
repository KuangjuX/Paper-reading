---
tags:
  - papers/LLM
  - topics/knowledge-distillation
  - topics/model-compression
aliases:
  - "Knowledge Distillation"
  - "Distilling the Knowledge"
date: 2026-08-24
doi: 10.48550/arXiv.1503.02531
---

# Distilling the Knowledge in a Neural Network

> Geoffrey Hinton, Oriol Vinyals, Jeff Dean，arXiv v1，2015-03-09。
> 原文：[arXiv](https://arxiv.org/abs/1503.02531) / [PDF](https://arxiv.org/pdf/1503.02531)；Roadmap：[Distillation and Router Training](../../../README.md#distillation-and-router-training)。

## 一句话结论

知识蒸馏不是复制教师参数，而是让更易部署的学生拟合教师的**完整输出分布**。soft targets 暴露了 hard label 丢掉的类别关系，temperature 控制这些关系显露多少；训练时把 soft-target CE（等价于 forward KL）和 hard-label CE 组合，部署时只保留学生并恢复普通 $T=1$ softmax。[论文 §1–2, pp. 1–3](https://arxiv.org/pdf/1503.02531#page=1)

## 1. 先把 teacher、student 和流程说清楚

论文从一个工程矛盾出发：训练阶段可以容忍很大计算量，部署阶段却受延迟和资源约束。因此可以先训练一个 cumbersome teacher——例如多个模型的 ensemble，或一个强正则化的大模型——再把它的行为转给单个 student。[论文 §1, pp. 1–2](https://arxiv.org/pdf/1503.02531#page=1)

这里的“知识”不是 teacher 的参数，而是它学到的输入到输出映射。teacher 与 student 因而不必同构。

```text
阶段 A：训练 teacher / ensemble

阶段 B：对 transfer inputs 运行 teacher
        teacher logits ──high-T softmax──► soft targets

阶段 C：训练 student
        同时匹配 soft targets，并在有标签时学习 hard labels

部署：input ─► student ─► softmax(T=1)
      teacher 不再参与推理
```

transfer set 可以是原训练集、另一批数据，甚至完全无标签。teacher 若是 ensemble，soft targets 可以取成员预测分布的算术平均或几何平均。[论文 §1, p. 2](https://arxiv.org/pdf/1503.02531#page=2)

## 2. Hard target 与 soft target 到底差什么

对三分类样本，hard target 只有 $y=[1,0,0]$。

它告诉 student 第一类正确，却没有告诉它第二类是否“差一点也可能”，第三类是否“完全不像”。

teacher 的 soft target 可能是 $p=[0.87,0.12,0.01]$。

这仍然认定第一类最可能，但同时编码了“第二类比第三类更接近”的关系。论文用 BMW 的例子说明：即使 BMW 被误判为垃圾车和胡萝卜的概率都很低，前者仍应远高于后者。这些错误类的相对概率，就是 hard label 丢掉而蒸馏想保留的信息。[论文 §1, p. 2](https://arxiv.org/pdf/1503.02531#page=2)

soft targets 还让每个样本提供多类监督，作者认为这会降低样本间梯度方差，使 student 能使用更少数据和更高学习率训练。

## 3. Logits 与 temperature：为什么要“把分布变软”

logits 是 softmax 前的分数。设 logits 为 $z_i$，temperature softmax 为：

$$
q_i^{(T)}=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}.
$$

$T=1$ 是普通 softmax；$T>1$ 把 logit 差距除小，使概率分布更平，但不会改变类别排序。[论文 Eq. 1 与 §2, pp. 2–3](https://arxiv.org/pdf/1503.02531#page=2)

用我们讨论过的 logits $[10,6,2]$：

| Temperature | softmax 概率（约） |
|---:|---|
| $T=1$ | $[0.9817,\ 0.0180,\ 0.0003]$ |
| $T=2$ | $[0.8668,\ 0.1173,\ 0.0159]$ |

$T=1$ 时，后两类几乎都像零，二者的关系很难影响训练；$T=2$ 后，第二类明显高于第三类，teacher 的相似性判断变得可学习。

训练时，teacher 生成 targets 与 student 匹配 targets 必须使用同一个高温 $T$；部署时 student 恢复 $T=1$。这不是 temperature annealing，而是训练/推理使用不同的读出温度。[论文 §2, p. 3](https://arxiv.org/pdf/1503.02531#page=3)

## 4. Soft-target CE 为什么等价于 forward KL

记 teacher 分布为 $p^{(T)}$，student 分布为 $q^{(T)}$。soft-target loss 是 cross-entropy：

$$
H(p^{(T)},q^{(T)})=-\sum_i p_i^{(T)}\log q_i^{(T)}.
$$

它与 KL 的关系是：

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q).
$$

teacher 已固定，所以 $H(p)$ 对 student 是常数。最小化 soft-target CE 与最小化

$$
D_{\mathrm{KL}}(p_{teacher}\|q_{student})
$$

有完全相同的 student 最优点。原文主要写 CE；roadmap 中说的 KL 是这个等价形式。

方向也有含义：forward KL 的期望权重来自 teacher。teacher 认为重要的类别若被 student 给出过低概率，会受到大惩罚；teacher 几乎不关心的类别权重也很小。其 logit 梯度直接写成：

$$
\frac{\partial C}{\partial z_i}=\frac{1}{T}\left(q_i^{(T)}-p_i^{(T)}\right).
$$

student 高估某类时梯度为正、更新会压低该 logit；低估 teacher 看重的类时梯度为负、更新会抬高它。[论文 Eq. 2, p. 3](https://arxiv.org/pdf/1503.02531#page=3)

## 5. 完整 loss：为什么还要 hard labels，为什么乘 $T^2$

若 transfer set 有真实标签，论文发现只把标签混进 soft targets 不如直接优化两项 loss。可写成：

$$
\mathcal L
=\alpha T^2H\!\left(p^{(T)},q^{(T)}\right)
+\beta H\!\left(y,q^{(1)}\right).
$$

- soft 分支学习 teacher 的完整分布；
- hard 分支在 student 无法完美拟合 teacher 时，把误差稍微推向正确答案；
- 两个分支共享 student logits，soft 分支用 $T$，hard 分支用 $T=1$；
- 原文没有规定 $\alpha,\beta$ 的统一符号，只说 hard-target 项通常取明显较低权重。[论文 §2, p. 3](https://arxiv.org/pdf/1503.02531#page=3)

为什么 soft 项前要乘 $T^2$？梯度公式先显式带来一个 $1/T$；高温下 $q^{(T)}-p^{(T)}$ 又大约缩小一个 $1/T$，所以 soft 梯度总体约按 $1/T^2$ 变小。乘回 $T^2$ 后，调 temperature 主要改变分布软硬，而不会顺便大幅改变 soft/hard 两项的相对权重。

## 6. 高温极限为何变成 logit matching

令 teacher logits 为 $v_i$、student logits 为 $z_i$。当 $T$ 远大于 logits 的幅值，并且对每个样本分别满足

$$
\sum_j z_j=\sum_j v_j=0,
$$

论文把梯度近似为：

$$
\frac{\partial C}{\partial z_i}\approx\frac{1}{NT^2}(z_i-v_i).
$$

因此高温蒸馏等价于最小化 teacher/student logits 的平方差，只差一个整体缩放。[论文 §2.1, Eq. 3–4, p. 3](https://arxiv.org/pdf/1503.02531#page=3)

零均值条件不能省略：softmax 对“所有 logits 同加一个常数”不敏感，概率本来无法确定 logits 的公共偏置；先零均值，才有逐项匹配的明确含义。有限温度还会少关注极负 logits，这可能过滤噪声，也可能丢失信息，所以最佳 $T$ 与 student 容量有关。

## 7. 实验：数字说明了什么

### 7.1 MNIST

teacher 是两个 1200-unit ReLU 隐层，使用 dropout、权重约束和最多 2 像素平移；student 是两个 800-unit ReLU 隐层。[论文 §3, pp. 3–4](https://arxiv.org/pdf/1503.02531#page=3)

| 模型 | Test errors |
|---|---:|
| 大 teacher | 67 |
| 普通 student | 146 |
| 蒸馏 student，$T=20$ | 74 |

student 几乎拿回 teacher 的全部优势，也学到了 teacher 从平移增强中获得、transfer set 没直接展示的泛化方式。student 每层至少 300 units 时，$T>8$ 结果相近；缩到每层 30 units 后，$T=2.5\sim4$ 最好，说明容量很小时不必强行匹配所有尾部 logits。

### 7.2 Transfer set 缺失类别

删掉所有数字 3 后，student 总计 206 次错误，其中 1010 个测试 3 错 133 个。作者把类别 3 的 bias 增加 3.5 后，总错误降到 109，测试 3 只错 14 个，即识别率 98.6%。只用 7、8 作为 transfer set 时，错误率为 47.3%；把 7/8 bias 各降低 7.6 后降到 13.2%。[论文 §3, p. 4](https://arxiv.org/pdf/1503.02531#page=4)

解释是：非 3 样本上的 teacher soft probabilities 仍泄露了“什么像 3”的结构，但没有 3 样本就难以学对类别先验/bias。必须强调，bias 数值是直接按**测试集表现**选的，因此这是机制展示，不是严格无偏的泛化结果。

### 7.3 语音：10-model ensemble 压成单模型

每个模型约 85M 参数，8 个 2560-unit ReLU 隐层、14,000 个 HMM-state 输出；数据约 2000 小时、700M 样本。ensemble 的 10 个成员架构相同，只改变随机初始化。试验 $T\in\{1,2,5,10\}$，最佳 $T=2$，hard-target CE 相对权重 0.5。[论文 §4–4.1, pp. 4–5](https://arxiv.org/pdf/1503.02531#page=4)

| System | Frame accuracy | WER |
|---|---:|---:|
| 单模型 baseline | 58.9% | 10.9% |
| 10-model ensemble | 61.1% | 10.7% |
| distilled single model | 60.8% | 10.7% |

蒸馏拿回 frame-accuracy 增益的约 86.4%，并按表中舍入值保留全部 WER 增益。这里压缩的是 **10×85M ensemble → 1×85M**；student 与单个成员同尺寸，所以它强力证明 ensemble compression，不等于证明单个 85M 模型还能任意缩小。

### 7.4 3% 数据：soft targets 像强正则器

| 训练方式 | 数据 | Train frame acc. | Test frame acc. |
|---|---:|---:|---:|
| hard targets | 100% | 63.4% | 58.9% |
| hard targets | 3% | 67.3% | 44.5% |
| soft targets | 3% | 65.4% | 57.0% |

3% hard-target 模型严重过拟合并需要 early stopping；soft-target 模型无需 early stopping，测试结果只比全数据 baseline 低 1.9 个百分点。[论文 §6 与 Table 5, pp. 7–8](https://arxiv.org/pdf/1503.02531#page=7)

但这不是“只靠 3% 信息恢复全量知识”：soft targets 来自看过 100% 数据的 teacher，全量数据中的规律通过 target 通道进入了 student。

### 7.5 JFT specialists

JFT 有 100M 图像、15,000 类。作者训练 61 个独立 specialists，每个负责 300 个易混类和一个 dustbin class：总体 top-1 从 25.0% 到 26.1%（相对 +4.4%），specialist 类条件准确率从 43.1% 到 45.9%。[论文 §5, Tables 2–4, pp. 5–7](https://arxiv.org/pdf/1503.02531#page=5)

这部分证明 specialists 能便宜、并行地增强 generalist，却**没有**把它们蒸馏回单模型。多模型输出还需逐样本优化一个 KL 合并目标；论文结尾明确承认 specialist distillation 尚未闭环。[论文 §5.4、§8, pp. 6–8](https://arxiv.org/pdf/1503.02531#page=6)

## 8. 简要局限

- 语音与 JFT 是内部数据，训练细节不足以完整复现；论文没有置信区间、student 重训方差或系统的 loss-weight 消融。
- MNIST 缺类实验用测试集调 bias，只能说明 soft targets 含结构信息，不能作为无偏性能估计。
- student 容量仍是硬上限；30-unit 实验已经显示温度敏感性。
- 蒸馏节省部署计算，不会消除先训练大 teacher / ensemble 的成本。
- 语音任务优化 frame CE，最终指标却是 WER；61.1%→60.8% 的 frame 差距与 WER 同为 10.7% 也说明目标并不完全一致。
- specialist ensemble 的最终蒸馏只是设想，没有实验结果。[论文 §8, p. 8](https://arxiv.org/pdf/1503.02531#page=8)

## 9. 与 MSA / DSA 的连接

把蒸馏中的“类别”换成“候选 key/token 位置”，就得到 roadmap 想建立的直觉：

```text
经典蒸馏：teacher 类别分布 ──forward KL──► student 类别分布
MSA 路由：Main Attention 位置分布 ──forward KL──► Indexer 位置分布
```

在 $T=1$ 时，Indexer logit 的核心梯度仍是：

$$
\frac{\partial\mathcal L}{\partial z_j^{index}}
=P_j^{index}-P_j^{main}.
$$

因此 Main Attention 认为重要而 Indexer 低估的位置会被抬高，反之会被压低。差别在于经典蒸馏通常是离线的大模型 → 小模型；MSA 是同一注意力层里的位置级蒸馏，用 soft distribution 绕过 hard Top-k 的不可导选择，而且推理时 Main Attention 仍负责正式计算。具体梯度边界见 [MSA KL 对齐笔记](../minimax-msa/msa-kl-distillation.md)。
