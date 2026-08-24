---
tags:
  - papers/LLM
  - topics/sparse-attention
  - topics/indexer
aliases:
  - "MSA vs DSA Indexer"
  - "MSA Indexer 与 Lightning Indexer"
date: 2026
---

# MSA Indexer 与 DSA Lightning Indexer 的差别

> MSA 主笔记：[MiniMax Sparse Attention (MSA)](msa.md)
>
> DSA 完整流程：[DeepSeek Sparse Attention (DSA) 算法流程](../deepseek-dsa/deepseek-sparse-attention-dsa.md)
>
> 论文：[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)；[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)

## 一句话结论

两者都是“轻量 Indexer 先选候选，主注意力再做精确 softmax”，但它们把选择能力放在了不同维度：

- **DSA 在序列轴更细**：逐 token 选择 2048 个 latent KV entries，但所有主注意力 Query heads 共用同一份 token 清单；
- **MSA 在 head/group 轴更细**：每个 GQA group 独立选择，但一次选择整个 128-token KV block。

因此，“DSA 比 MSA 更细粒度”只是在说 **token vs block**，不是说 DSA 在所有维度上都更精细。

---

## 1. 两者共享的基本框架

MSA 与 DSA 都把稀疏注意力拆成两步：

```text
轻量 Indexer：对历史位置做近似相关性评分
        ↓ Top-k（离散、不可微）
得到少量候选位置或候选块
        ↓
Main Attention：只在候选集合上重算真正的 attention logits
        ↓ Softmax × Value
得到精确的稀疏注意力输出
```

两者都必须额外训练 Indexer，因为语言模型损失不能穿过离散 Top-k，直接告诉 Indexer 哪些候选应该入选。两篇论文都选择用主注意力分布作为教师，通过 KL loss 对齐 Indexer。

但是，“如何打分、选什么、谁共享选择结果、主分支读取什么”完全不同。

---

## 2. MSA Indexer：每个 GQA group 选 KV blocks

### 2.1 打分函数

MSA 从隐藏状态 $X$ 产生：

$$
Q^{\mathrm{idx}}
=XW_q^{\mathrm{idx}}
\in\mathbb{R}^{N\times H_{kv}\times d_{idx}},
\qquad
K^{\mathrm{idx}}
=XW_k^{\mathrm{idx}}
\in\mathbb{R}^{N\times1\times d_{idx}}.
$$

第 $r$ 个 GQA group 对 token $j$ 的 index score 是：

$$
S_{i,j}^{\mathrm{idx},(r)}
=
\frac{
Q_i^{\mathrm{idx},(r)}
(K_j^{\mathrm{idx}})^\top
}{\sqrt{d_{idx}}}.
$$

这里有一个刻意的不对称：

- Query 侧有 $H_{kv}$ 个 index heads，一组一个；
- Key 侧只有一个共享 index head。

所以 MSA 采用 $H_{qidx}=H_{kv}$，并不是数学定理，而是为了让每个 GQA group 恰好拥有一份独立选择结果。

### 2.2 从 token score 变成 block selection

将历史 token 按 $B_k=128$ 切块，在每个块内取最大 token score：

$$
M_{i,b}^{\mathrm{idx},(r)}
=
\max_{j\in\mathcal B_b,\,j\le i}
S_{i,j}^{\mathrm{idx},(r)}.
$$

再为每个 query 位置 $i$、每个 GQA group $r$ 选择 $k=16$ 个块：

$$
\mathcal I_i^{(r)}
=
\operatorname{TopK}_b
\left(M_{i,b}^{\mathrm{idx},(r)},16\right).
$$

主配置的最大注意力预算是：

$$
16\ \text{blocks}\times128\ \text{tokens/block}
=2048\ \text{tokens/group/query}.
$$

最终配方还强制包含 query 所在的 local incomplete block。

### 2.3 选择结果如何共享

MSA 是标准 GQA：$H_q=64$，$H_{kv}=4$，每组 $G=16$ 个 Query heads。

- 每个 GQA group 有自己的 16-block 清单；
- 同一组里的 16 个主 Query heads 共用这份清单；
- 不同组可以选择不同的远程块；
- 进入主分支后，各 Query head 仍用自己的 $Q$ 计算真正的 attention logits。

也就是说，MSA 共享的是“读哪些块”，不是“如何理解这些块”。

---

## 3. DSA Lightning Indexer：所有 Query heads 共选 tokens

### 3.1 打分函数

对 query token $h_t$，Lightning Indexer 产生 $H_I$ 个 index-query 向量和对应权重；对历史 token $h_s$ 产生一个 index-key 向量：

$$
q_{t,j}^{I}\in\mathbb R^{d_I},
\qquad
w_{t,j}^{I}\in\mathbb R,
\qquad
k_s^{I}\in\mathbb R^{d_I}.
$$

一个 token pair 最终只有一个标量 index score：

$$
I_{t,s}
=
\sum_{j=1}^{H_I}
w_{t,j}^{I}\,
\operatorname{ReLU}
\left(q_{t,j}^{I}\cdot k_s^{I}\right).
$$

ReLU 是吞吐导向的选择；论文还强调 Indexer head 数较少，并可以用 FP8 实现。

### 3.2 直接做 token-level Top-k

DSA 不做 block pooling，而是直接选择分数最高的 $k=2048$ 个历史 token：

$$
\mathcal S_t
=
\operatorname{TopK}_s(I_{t,s},2048).
$$

然后读取这些 token 对应的 MLA latent KV entries，再运行主注意力。

### 3.3 选择结果如何共享

$H_I$ 个 Indexer heads 会先经过加权、ReLU 和求和，坍缩成一个 $I_{t,s}$。因此每个 query token 最终只有一份 $\mathcal S_t$：

- 所有 MLA Query heads 共用同一份 2048-token 清单；
- 每个主 Query head 仍计算自己不同的 attention logits 和 softmax；
- 共享的是 selected token set，不是主注意力概率。

官方 DeepSeek-V3.2 将 DSA 实例化在 MLA 的 MQA-mode 下，使同一份 latent KV entry 能被所有 Query heads 共同读取和复用。

---

## 4. 核心差别总表

| 维度 | MSA Indexer | DSA Lightning Indexer |
|---|---|---|
| 主注意力基础 | 标准 GQA | MLA 的 MQA-mode |
| Indexer 输入 | stop-gradient hidden states | 从 hidden states 派生；稀疏训练时输入 detach |
| Query 侧 Indexer | $H_{kv}$ 个头，一 GQA group 一个 | $H_I$ 个轻量 scorer heads |
| Key 侧 Indexer | 1 个共享 $K^{idx}$ head | 1 个共享 $k_s^I$ 表示 |
| token pair 打分 | scaled dot product | weighted sum of ReLU(dot product) |
| Indexer heads 如何汇总 | 不汇总，保留 $H_{kv}$ 份 group scores | 汇总成一个标量 $I_{t,s}$ |
| Top-k 对象 | KV blocks | 单个 tokens / latent KV entries |
| 主配置预算 | $16\times128=2048$ tokens/group | 2048 tokens/query |
| 选择清单数量 | 每个 query 有 $H_{kv}$ 份 | 每个 query 只有 1 份 |
| 主 heads 如何共享 | 每个 GQA group 内共享 | 所有 Query heads 共享 |
| 被读取的 KV | 对应 GQA group 的普通 K/V blocks | MLA latent KV entry，加 decoupled RoPE key 部分 |
| 局部先验 | 强制 local incomplete block | 正文公式未描述类似的强制块规则 |
| 稀疏阶段主 attention | 每个组在所选块内精确 SDPA | 所有 heads 在同一所选 token set 上精确 MLA attention |
| Indexer 渐近复杂度 | 仍含 $O(N^2)$ token 打分 | 仍含 $O(N^2)$ token 打分 |

---

## 5. 为什么说 DSA 更“细粒度”

设需要从一个 128-token 块中读取一个真正重要的 token。

### DSA

DSA 可以只选择那一个 token：

```text
block = [无关, 无关, 重要, 无关, ...]
                       ↑
                 只读取这个 entry
```

它的 2048 个预算单位对应 2048 个独立位置，可以散落在整个长上下文中。

### MSA

MSA 的 max-pool 也能发现那个重要 token，但一旦该块入选，就要读取整个因果可见块：

```text
block = [无关, 无关, 重要, 无关, ...]
         └──────── 整块进入主注意力 ────────┘
```

因此在相同 2048-token 上限下：

- DSA 最多可以覆盖 2048 个互不相邻的 token；
- MSA 只能覆盖约 16 个不同区域，但每个区域有连续 128-token 上下文。

DSA 在**序列位置轴**更精细，也更少把预算浪费在同块无关 token 上；MSA 的连续块则更利于规整访存、批量加载和 Tensor Core 计算。

---

## 6. 但 MSA 在 head/group 轴更精细

假设一个 query 的不同语义子空间需要不同资料：

```text
GQA group 0 → 代码定义
GQA group 1 → 最近对话
GQA group 2 → 文档开头的约束
GQA group 3 → 远处的报错日志
```

MSA 可以给四个 group 四份不同 block 清单。DSA 的 Lightning Indexer 虽然内部有多个 scorer heads，但它们最终被求和成单一 $I_{t,s}$，所有 MLA Query heads 使用同一份 Top-k token set。

所以应使用二维描述：

| 选择轴 | 更精细的一方 | 原因 |
|---|---|---|
| 序列位置轴 | DSA | 单 token 选择，而不是 128-token block |
| head/group 轴 | MSA | 每个 GQA group 独立选择，而不是所有 heads 共用一份清单 |

这能避免一句“DSA 更细”造成的误解。

---

## 7. DSA 是否数学上必须使用 MQA

**不是。**“先计算一个相关性分数，再选择 token”这个 DSA 原型，本身并不要求 MQA；完全可以构造 per-head 或 per-group 的 DSA 变体。

但是 DeepSeek-V3.2 的官方实现选择 MLA MQA-mode，原因是 kernel 级效率：

1. DSA 的 token 选择非常离散；
2. 如果不同 Query head 选择不同 token，会产生大量不同 gather 列表；
3. 同一 latent entry 被多个 Query heads 共享，才能摊薄不规则 KV 读取成本；
4. MLA 可以通过矩阵吸收，在不展开 per-head K/V cache 的情况下保留 per-head 行为。

这里也不能把 MLA MQA-mode 等同于 vanilla MQA：

- vanilla MQA 直接让所有 Query heads 共用同一组普通 K/V heads；
- MLA MQA-mode 共享的是压缩 latent $c_s^{KV}$，每个 Query head 仍有自己的 absorbed query 与输出映射。

因此更准确的说法是：

> DSA 原型不被 MQA 数学约束；DeepSeek-V3.2 为了让 token-level 稀疏读取具有足够的跨 head 复用，将它实现为 MLA MQA-mode。

---

## 8. KL 监督也体现了不同的共享粒度

### MSA：per-GQA-group 教师

对第 $r$ 个 GQA group，MSA 将组内 $G$ 个主 attention heads 的概率分布平均，得到该组的教师分布：

$$
P_{i,j}^{(r)}
=
\frac{1}{G}
\sum_{\ell\in\mathcal H_r}
P_{i,j}^{(\ell)}.
$$

第 $r$ 个 Indexer Query head 只需要拟合第 $r$ 组的教师，因此能够学出 group-specific selection。

### DSA：跨全部 heads 的共享教师

DSA 将主注意力量跨所有 attention heads 求和，再沿序列轴做 L1 normalization，得到一份共享目标 $p_{t,:}$：

$$
\mathcal L_I
=
\sum_t
D_{KL}
\left(
p_{t,:}
\,\|\,
\operatorname{Softmax}(I_{t,:})
\right).
$$

这与它最终只有一份 Top-k token list 是一致的：Indexer 必须学习所有 heads 的综合需求。

### Warm-up 的一个重要差别

- **MSA**：warm-up 时主分支保持 dense，Indexer 在全序列上追踪主分支；MSA-PT/CPT 的主实验 warm-up 为 40B tokens。
- **DSA**：dense warm-up 只训练 Indexer，冻结其余全部模型参数；仅 1000 steps、约 2.1B tokens，随后进入 2048-token sparse training。

---

## 9. 对 kernel 的含义

### MSA 的硬件取舍

块选择带来连续 K/V 数据：

- 一个入选 block 可用连续加载；
- 同组 16 个 Query heads 共用该 block；
- KV-outer 调度还能跨 query positions 复用热门 K/V blocks；
- 代价是一个重要 token 会把同块其余 token 一起带进来。

### DSA 的硬件取舍

token 选择提高了信息密度，但地址更离散：

- 2048 个 entry 可能分散在整条上下文；
- 需要 gather 非连续 latent entries；
- 所有 Query heads 共享相同 entry，提供了关键的跨 head 复用；
- MLA latent cache 比展开后的 per-head K/V 小，降低了 gather 流量。

两者可以看成两种硬件友好策略：

> MSA 用“连续 block + group 内复用”换规整性；DSA 用“压缩 latent + 所有 heads 复用”支撑 token-level 稀疏性。

---

## 10. 一个统一例子

假设上下文有 128K tokens，目标预算约为 2048 tokens。

### MSA

```text
query token t
  ├─ GQA group 0 → 16 blocks → 最多 2048 tokens
  ├─ GQA group 1 → 16 blocks → 最多 2048 tokens
  ├─ GQA group 2 → 16 blocks → 最多 2048 tokens
  └─ GQA group 3 → 16 blocks → 最多 2048 tokens
```

四份列表可能重叠，也可能覆盖不同区域。这里的“2048”是每组上限，不代表四组合计一定读取 8192 个不同 token。

### DSA

```text
query token t
  └─ one Top-2048 token list
       ├─ MLA query head 0 使用
       ├─ MLA query head 1 使用
       ├─ ...
       └─ 所有 query heads 使用
```

一份列表服务所有 heads，但每个 head 在列表内部仍计算不同的主 attention logits。

---

## 最终记忆方式

```text
MSA：先按 token 打分，再 max-pool 成 block 分数；
     每个 GQA group 选自己的 16 个块。

DSA：多个 Lightning Indexer heads 给 token pair 打分后汇总；
     所有 MLA Query heads 共用同一份 2048-token 清单。
```

最简洁的二维结论：

> **DSA = token-fine、head-shared；MSA = block-coarse、group-specific。**

## 相关笔记

- [MiniMax Sparse Attention (MSA)](msa.md)
- [MSA 中的 KL 对齐与局部知识蒸馏](msa-kl-distillation.md)
- [MSA 为什么采用 KV-outer Sparse Attention Forward](msa-kv-outer-forward.md)
- [DeepSeek Sparse Attention (DSA) 算法流程](../deepseek-dsa/deepseek-sparse-attention-dsa.md)
