---
tags:
  - papers/LLM
  - topics/knowledge-distillation
  - topics/sparse-attention
aliases:
  - "MSA KL"
  - "MSA Knowledge Distillation"
date: 2026
---

# MSA 中的 KL 对齐与局部知识蒸馏

> 关联论文：[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)
>
> 讨论来源：[ChatGPT 共享对话：KL 散度解释](https://chatgpt.com/s/t_6a8276767a7c8191b061cea2e62c5ca0)
>
> 主笔记：[MiniMax Sparse Attention (MSA)](msa.md)

## 一句话结论

MSA 不是用一个完整大模型去蒸馏另一个小模型，而是在**同一注意力层内部**建立局部教师—学生关系：

- **Main Branch 是教师**：它的精确 attention probability 表示模型真正需要哪些上下文 token；
- **Index Branch 是学生**：它用低成本 score 预测哪些 token 所在的 KV block 值得进入 Top-k；
- **KL loss 是教学信号**：在选中的 token 支撑集上，让 Index Branch 的分布拟合 Main Branch；
- **推理时没有 KL**：Indexer 只负责选块，Main Branch 只在选中块上执行精确 softmax attention。

更准确的称呼是：**层内、在线、注意力位置级的自蒸馏**。

---

## 1. 为什么 LM Loss 训练不了 Index Branch

MSA 的路由路径是：

```text
Index scores → block max → Top-k block indices → gather K/V → Main Attention
```

Top-k 的输出是离散整数下标。只要排名不变，小幅改变 block score 不会改变输出；跨过排名边界时，输出又会突然跳变。因此普通链式法则无法提供有用的 Top-k 梯度。

语言模型损失可以正常更新：

```text
LM Loss → Main output → Main Q/K/V/O → backbone
```

但不能沿下面的路径训练索引投影：

```text
LM Loss ⇏ integer Top-k indices ⇏ W_q^idx, W_k^idx
```

KL 的作用不是训练语言模型输出，而是绕过不可导的 Top-k，给 Index Branch 单独建立一条可微监督路径。

---

## 2. 教师分布和学生分布

对 query 位置 $i$、GQA group $r$，记选中 block 所诱导的因果可见 token 集合为：

$$
\mathcal J_i^{(r)} = \mathcal I_{i,\mathrm{tok}}^{(r)}.
$$

### 2.1 学生：Index Branch

Indexer 的 token-level logits 为：

$$
S_{i,j}^{\mathrm{idx},(r)}
=
\frac{Q_i^{\mathrm{idx},(r)}(K_j^{\mathrm{idx}})^\top}
{\sqrt{d_{\mathrm{idx}}}}.
$$

在 $\mathcal J_i^{(r)}$ 上归一化：

$$
P_{i,j}^{\mathrm{idx},(r)}
=
\frac{\exp S_{i,j}^{\mathrm{idx},(r)}}
{\sum_{u\in\mathcal J_i^{(r)}}\exp S_{i,u}^{\mathrm{idx},(r)}}.
$$

这表示 Index Branch 对“当前 query 应关注哪个历史 token”的预测。

### 2.2 教师：Main Branch

同一个 GQA group 中有 $G=H_q/H_{kv}$ 个 query heads，但它们必须共享一份 block 选择。论文先让每个主 attention head 在同一支撑集上分别做 softmax，再在**概率层面**平均：

$$
P_{i,j}^{(r)}
=
\frac{1}{G}\sum_{\ell\in\mathcal H_r}
\frac{\exp S_{i,j}^{(\ell)}}
{\sum_{u\in\mathcal J_i^{(r)}}\exp S_{i,u}^{(\ell)}}.
$$

这里不是先平均 logits 再 softmax。概率级平均使 group-level teacher 同时反映组内多个 query heads 的关注模式。

需要特别区分：

- **路由选择**使用 block 内 token score 的最大值，再做 block Top-k；
- **KL 对齐**使用选中 block 内所有 token 的 token-level softmax。

---

## 3. KL 的数学含义

每层辅助损失为：

$$
\mathcal L_{\mathrm{KL}}
=
\frac{1}{NH_{kv}}
\sum_{i=1}^{N}\sum_{r=1}^{H_{kv}}
D_{\mathrm{KL}}\!\left(
\operatorname{stopgrad}(P_i^{(r)})
\,\|\,
P_i^{\mathrm{idx},(r)}
\right).
$$

总训练目标为：

$$
\mathcal L
=
\mathcal L_{\mathrm{LM}}
+\lambda\sum_{\mathrm{layers}}\mathcal L_{\mathrm{KL}}.
$$

因为教师分布被 detach：

$$
D_{\mathrm{KL}}(P\|P^{\mathrm{idx}})
=
-\sum_jP_j\log P_j^{\mathrm{idx}}-H(P),
$$

其中 $H(P)$ 对 Index Branch 是常数。因此，训练 Index Branch 时，最小化 KL 等价于：

> 使用 Main Branch 的 attention probability 作为 soft label，最小化一项 soft-label cross entropy。

传统知识蒸馏通常在词表类别上对齐；MSA 把“类别”换成了上下文中的 key-token 位置。

### 最关键的梯度

若 Index logits 为 $z_j=S_{i,j}^{\mathrm{idx},(r)}$，则：

$$
\frac{\partial D_{\mathrm{KL}}(P\|P^{\mathrm{idx}})}{\partial z_j}
=
P_j^{\mathrm{idx}}-P_j.
$$

所以：

- 教师认为重要、Indexer 低估的 token：梯度下降会提高它的 index logit；
- 教师认为不重要、Indexer 高估的 token：梯度下降会降低它的 index logit；
- token logit 升高后，其所在 block 的 max score 更可能进入 Top-k。

这就是 token-level KL 最终转化为 block-level routing 能力的机制。

### 关于 KL 方向的严谨解释

论文使用 $D_{\mathrm{KL}}(P^{\mathrm{main}}\|P^{\mathrm{idx}})$。当教师在某位置分配较大概率、学生却给出极小概率时，$P_j\log(P_j/P_j^{\mathrm{idx}})$ 会产生较大惩罚，这符合稀疏检索不希望漏掉重要位置的直觉。

但不能把 forward-KL 与 reverse-KL 的**单个位置项**直接拿来比较倍数：KL 只有对完整分布求和后才保证非负，reverse-KL 的单项可以为负。这里最可靠、最直接的依据仍是完整目标以及梯度 $P^{\mathrm{idx}}-P^{\mathrm{main}}$；论文没有报告 KL 方向的专项消融。

---

## 4. 梯度究竟流向哪里

论文同时使用两种 detach：

$$
Q^{\mathrm{idx}}=\operatorname{stopgrad}(X)W_q^{\mathrm{idx}},
\qquad
K^{\mathrm{idx}}=\operatorname{stopgrad}(X)W_k^{\mathrm{idx}},
$$

以及：

$$
D_{\mathrm{KL}}(\operatorname{stopgrad}(P)\|P^{\mathrm{idx}}).
$$

最终梯度路由是：

| 对象 | LM Loss | KL Loss |
|---|:---:|:---:|
| Main $W_q,W_k,W_v,W_o$ | ✓ | — |
| Backbone / hidden states $X$ | ✓ | — |
| Index $W_q^{\mathrm{idx}},W_k^{\mathrm{idx}}$ | — | ✓ |
| Top-k block indices | — | — |

两条分支在 forward 上相互依赖，在 backward 上隔离：

- Index Branch 决定 Main Branch 看见什么；
- Main Branch 提供 Index Branch 应该模仿的注意力分布；
- LM 不通过 Top-k 更新 Indexer；
- KL 不改变 teacher 或 backbone。

如果不 detach，backbone 可以通过“让 Main attention 更平滑、更容易模仿”来降低 KL，而不是真正提升 Indexer。论文的消融观察到梯度尖峰、LM loss 不稳定和短上下文能力退化。

---

## 5. 为什么必须 Warmup

从随机初始化直接开启 sparse Top-k 会形成闭环故障：

```text
随机 Indexer
→ 选错 blocks
→ Main Branch 看不到有意义的上下文
→ teacher 分布质量变差
→ Indexer 得到更差的监督
```

因此训练分成两阶段：

| 阶段 | Main Branch | KL 支撑集 | Indexer 是否控制路由 |
|---|---|---|---|
| Warmup | Full Attention | 完整因果上下文 | 否 |
| Sparse training | Top-k sparse attention | 已选 block 内的 token | 是 |

Warmup 先让 Indexer 在完整 teacher 分布上建立较好的全局初始化，再把路由权交给它。

### 稀疏阶段的“支撑集盲区”

打开 Top-k 后，Main Branch 和 KL 都只计算已选 token。如果 Indexer 漏掉一个真正重要的 block，当前 step 的 teacher 没有计算那个 block，KL 因而不能直接告诉 Indexer“你漏了它”。

所以 sparse 阶段的 KL 更擅长：

- 维持 warmup 已建立的对齐；
- 调整已选 support 内的相对重要性；
- 跟踪训练过程中 Main attention 的变化。

这也是为什么不能只看 KL loss，还要同时看 block recall、score recall 和长上下文任务表现。

---

## 6. Kernel 为什么只需要 LSE

softmax probability 可以写为：

$$
P_j^{\mathrm{main}}=\exp(S_j^{\mathrm{main}}-\mathrm{LSE}_{\mathrm{main}}),
$$

$$
P_j^{\mathrm{idx}}=\exp(S_j^{\mathrm{idx}}-\mathrm{LSE}_{\mathrm{idx}}).
$$

论文的优化是：

1. Main sparse-attention forward 顺便输出最终 $\mathrm{LSE}_{\mathrm{main}}$；
2. Index forward 保存 per-block LSE，再对 Top-k blocks 归约得到 $\mathrm{LSE}_{\mathrm{idx}}$；
3. KL backward 根据 logits 和两个 LSE 重建概率；
4. 直接形成 $dS^{\mathrm{idx}}\propto P^{\mathrm{idx}}-P^{\mathrm{main}}$。

因此可以跳过一个独立、重复计算 softmax 的 KL-forward kernel。这里省掉的是**专门的 KL forward 计算阶段**；训练系统仍通过自定义 backward 注入相应的辅助梯度。

由于不同 KV tiles 对应的 query 数量高度不均匀，KL backward 使用 persistent grid 和全局 atomic work counter 做动态负载均衡。

---

## 7. 与传统知识蒸馏的区别

| 维度 | 传统知识蒸馏 | MSA 的 KL 对齐 |
|---|---|---|
| 教师 | 通常是独立大模型 | 同一层的 Main Branch |
| 学生 | 通常是较小模型 | 轻量 Index Branch |
| 对齐对象 | 词表输出概率 | 上下文位置概率 |
| 目的 | 压缩整个模型 | 训练不可导的 Top-k 路由器 |
| 教师状态 | 常被冻结 | 持续被 LM loss 更新，但对 KL detach |
| 推理时教师 | 通常移除 | Main Branch 仍负责正式 attention |
| 推理时 KL | 不需要 | 不需要 |

因此 MSA 不需要在部署时额外运行一个 teacher LLM。

---

## 8. 如何阅读训练指标

`msa_kl_loss` 低，只表示**在当前 KL 支撑集上**，Indexer distribution 接近 Main attention distribution。它不等价于：

- Top-k 只错了某个百分比；
- block recall 很高；
- 模型输出误差很小；
- 没有漏掉支撑集外的重要 block。

更可靠的联合判断是：

```text
KL loss
+ block recall
+ score recall
+ long-context retrieval evaluation
+ LM loss / gradient norm
```

其中：

- **Block recall**：Main Branch 理想 Top-k blocks 中，有多少被 Indexer 找到；
- **Score recall**：找到的 blocks 覆盖了多少 Main attention probability mass；
- score recall 高于 block recall 并不矛盾：可能漏了若干低质量 block，但保住了最重要的 attention mass。

---

## 记忆模型

```text
Main Branch：如果真正做 attention，我会关注这些 token。

Index Branch：我要用更便宜的计算，提前猜出你会关注哪些 token。

KL：你的预测分布与 Main Branch 的真实注意力分布相差多少。
```

MSA 中 KL 的本质是：

> **使用精确主注意力产生的位置级软标签，训练一个便宜的稀疏检索器。**

