---
tags:
  - papers/LLM
  - topics/sparse-attention
  - topics/GPU-kernel
aliases:
  - "MSA KV-outer"
  - "MSA Sparse Attention Forward"
date: 2026
---

# MSA 为什么采用 KV-outer Sparse Attention Forward

> 关联论文：[MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)
>
> 主笔记：[MiniMax Sparse Attention (MSA)](msa.md)
>
> 论文—代码映射：[MiniMax Sparse Attention：论文算法与代码实现精确对应](MiniMax%20Sparse%20Attention：论文算法与代码实现精确对应.md)

## 核心结论

论文中的术语是 **KV-outer**，不是 KV-owner。`outer` 表示 sparse-attention kernel 把 `(KV block, KV head)` 放在外层遍历维度：

```text
Q-outer：固定 query，逐个读取该 query 选择的 KV blocks

KV-outer：固定 KV block，聚集所有选择该 block 的 queries
```

MSA 选择 KV-outer 的根本原因是：

> **多个 query 经常选择同一个 KV block。固定 KV block 后，可以让一批 queries 复用同一份 K/V tile，并将它们拼成更大的 Tensor Core MMA。**

在论文配置 $G=16,B_k=128$ 下：

$$
\text{Q-outer arithmetic intensity}\approx G=16,
$$

$$
\text{KV-outer arithmetic intensity}\approx\frac{2}{3}B_k=85.3.
$$

KV-outer 用 reverse index、query gather、partial-output buffer 和 combine kernel 的额外成本，换取：

1. 更少的重复 K/V 流量；
2. 更高的算术强度；
3. 更规整、更饱满的 Tensor Core MMA；
4. 对热门 KV blocks 的显式负载均衡。

---

## 1. KV-outer 改变的只是执行顺序

MSA 的数学定义仍然是：对每个 query $i$ 和 GQA group $r$，Indexer 给出 selected block set $\mathcal I_i^{(r)}$，Main Branch 在这些 blocks 所包含的因果可见 token 上执行精确 softmax attention。

KV-outer 不改变：

- 每个 query 选择哪些 blocks；
- selected support 中包含哪些 token；
- attention score、softmax 和 value aggregation 的数学结果。

它只改变 GPU 如何组织同一组稀疏边：

```text
原始 q2k 视角：query -> selected KV blocks

反转 k2q 视角：KV block -> queries selecting it
```

可以把 sparse-attention pattern 看成一张二部图：

```text
Query nodes ------ selected edges ------ KV-block nodes
```

- Q-outer 按左侧 query nodes 遍历边；
- KV-outer 按右侧 KV-block nodes 遍历边。

二者访问完全相同的边，只是数据复用方向不同。

---

## 2. 一个最小例子：为什么 Q-outer 会重复读取 K/V

假设 Top-k selection 得到：

```text
q0 -> block A, C
q1 -> block A, D
q2 -> block A, C
q3 -> block A, B
```

### Q-outer

```text
for query q:
    for block b in selected_blocks[q]:
        load K[b], V[b]
        compute partial attention(q, b)
```

对应访问：

```text
q0: load A, C
q1: load A, D
q2: load A, C
q3: load A, B
```

block A 被重复加载 4 次，block C 被重复加载 2 次。

在真实 MSA 中，这种复用很常见：

- attention-sink block 可能被大量 query 选择；
- local blocks 会被相邻 query 频繁选择；
- 数据相关的远程语义 block 也可能成为热点。

### KV-outer

先反转选择关系：

```text
A -> q0, q1, q2, q3
B -> q3
C -> q0, q2
D -> q1
```

然后执行：

```text
for (KV block b, KV head r):
    queries = reverse_index[b, r]
    load K[b, r], V[b, r]
    gather queries
    compute partial attention(queries, b)
```

block A 的 K/V tile 可以服务一批 gathered queries。

需要严谨地说：如果一个热门 block 对应的 query 太多，scheduler 会把它拆给多个 CTA，每个 CTA 仍需加载自己的 K/V tile。因此 KV-outer 并不保证“一个 block 在整个 GPU 上绝对只读取一次”，而是保证：

> **每次 K/V tile 加载都能在一个 query chunk 内被充分复用。**

---

## 3. 论文的 I/O 推导

设：

| 符号 | 含义 |
|---|---|
| $H_q$ | query head 数 |
| $H_{kv}$ | KV head / GQA group 数 |
| $G=H_q/H_{kv}$ | 每个 KV head 服务的 query head 数 |
| $N$ | query 和 KV 序列长度 |
| $d_h$ | attention head dimension |
| $B_k$ | 一个 KV block 包含的 token 数 |
| $k$ | 每个 query/group 选择的 block 数 |

论文第 4.2 节讨论的是 sparse prefill，假设 query length 与 KV length 相等，并按每个元素 2 bytes 估算 I/O。

### 3.1 两种顺序的 FLOPs 相同

无论先遍历 query 还是 KV block，所有 sparse edges 上的 $QK^\top$ 和 $PV$ 都必须执行：

$$
\mathrm{FLOPs}=4H_qNd_hkB_k.
\tag{13,15}
$$

所以 KV-outer 的收益不来自减少 attention FLOPs，而来自减少数据搬运和改善矩阵形状。

### 3.2 Q-outer I/O

$$
\mathrm{IO}_{Q\text{-outer}}
=
\underbrace{2\cdot2\cdot H_qNd_h}_{\mathrm{read}(Q)+\mathrm{write}(O)}
+
\underbrace{2\cdot2\cdot H_{kv}NkB_kd_h}_{\mathrm{read}(K+V)}.
\tag{14}
$$

长序列下，K/V 项占主导：

$$
4H_{kv}NkB_kd_h.
$$

它表示每个 query/group 都重新读取其 selected blocks。于是：

$$
\frac{\mathrm{FLOPs}}{\mathrm{IO}}
\approx
\frac{4H_qNd_hkB_k}{4H_{kv}NkB_kd_h}
=
\frac{H_q}{H_{kv}}
=G.
$$

论文主配置 $H_q=64,H_{kv}=4$：

$$
G=16.
$$

### 3.3 KV-outer I/O

KV-outer 需要 partial-output buffer：

$$
\begin{aligned}
\mathrm{IO}_{KV\text{-outer}}
={}&
\underbrace{2\cdot2\cdot H_{kv}Nd_h}_{\mathrm{read}(K+V)} \\
&+
\underbrace{2\cdot2\cdot H_qNkd_h}_{\mathrm{read}(Q)+\mathrm{write}(O_{buf})} \\
&+
\underbrace{2\cdot H_qN(k+1)d_h}_{\mathrm{read}(O_{buf})+\mathrm{write}(O)}.
\end{aligned}
\tag{16}
$$

最关键的变化是 K/V 流量：

$$
O(NkB_kd_h)
\quad\longrightarrow\quad
O(Nd_h).
$$

KV-outer 增加了：

- 每条 sparse edge 对应的 Q 读取；
- 每条 sparse edge 对应的 partial output 写入；
- combine 阶段的 partial output 读取。

但这些中间量按 $d_h$ 计，而 Q-outer 重复加载的 K/V 按 $B_kd_h$ 计。$B_k=128$ 时，避免整块 K/V 重读带来的收益远大于多搬运一个 query vector 或 partial vector 的成本。

忽略较小项后：

$$
\frac{\mathrm{FLOPs}}{\mathrm{IO}}
\approx
\frac{2}{3}B_k.
$$

代入 $B_k=128$：

$$
\frac{2}{3}B_k=85.3.
$$

所以：

$$
\frac{85.3}{16}\approx5.3.
$$

KV-outer 的**估计算术强度**约为 Q-outer 的 5.3 倍。它不是 5.3× wall-clock speedup；实际性能还取决于 reverse-index 构造、gather 效率、occupancy、负载均衡和 combine 开销。

---

## 4. 为什么 KV-outer 能提高 Tensor Core 利用率

MSA 主配置中：

$$
H_q=64,
\qquad
H_{kv}=4,
\qquad
G=16,
\qquad
d_h=B_k=128.
$$

固定一个 query 位置和一个 GQA group 时，只得到 $G=16$ 个 query heads：

$$
\underbrace{Q}_{16\times128}
\times
\underbrace{K^\top}_{128\times128}
\longrightarrow
\underbrace{S}_{16\times128}.
$$

score MMA 的 $M$ 维只有 16，矩阵太矮，难以充分利用 Tensor Core。

### Q-outer 为什么不容易拼 queries

不同 query 通常选择不同 block sets：

```text
q0 -> A, C
q1 -> A, D
q2 -> B, C
```

如果固定 query 处理整份 selection，无法直接把多个 query 沿序列维拼起来，因为它们的 K operands 并不相同。

### KV-outer 的 Query Concatenation

KV-outer 固定 block A 后，收集到的所有 query 都共享同一个 $K_A$：

```text
block A -> q0, q1, q5, q8, ...
```

论文把：

$$
\left\lceil\frac{128}{G}\right\rceil
=
\frac{128}{16}
=8
$$

个 query positions 拼接。每个位置贡献 16 个 query heads：

$$
8\times16=128.
$$

于是 score MMA 变为：

$$
\underbrace{Q_{cat}}_{128\times128}
\times
\underbrace{K_{block}^\top}_{128\times128}
\longrightarrow
\underbrace{S}_{128\times128}.
$$

这同时实现：

1. 同一份 K/V tile 服务多个 query positions；
2. MMA 的 $M$ 维从 16 增加到 128；
3. 连续的 128-token KV block 与 $128\times128$ tile 自然匹配；
4. TMA 可把 gathered queries 搬入 shared memory，并与后续 tile 计算形成流水。

---

## 5. Reverse Sparse Index：从 q2k 变成 k2q

Top-k selector 的自然输出是：

$$
(i,r)\longrightarrow\mathcal I_i^{(r)},
$$

即：

```text
query -> selected KV blocks
```

KV-outer kernel 需要：

```text
(KV block, KV head) -> queries selecting it
```

因此 forward 前需要构造 reverse sparse index，逻辑上类似 CSR：

```text
q2k [Hkv, Tq, topK]
        ↓
count queries per (KV head, KV block)
        ↓
prefix sum -> row pointers
        ↓
scatter query indices
        ↓
k2q CSR
```

主要数据结构可以抽象为：

- `k2q_row_ptr`：每个 `(KV head, KV block)` 的 query-list 起止位置；
- `k2q_q_indices`：压平后的 query indices；
- scheduler metadata：热点切分、partial slot 和有效 partial 数。

Reverse-index 构造是 KV-outer 的固定额外成本，也是论文实测 speedup 小于理论 FLOPs reduction 的原因之一。

---

## 6. 为什么必须 Two-phase Forward

Q-outer 可以让一个 CTA 持有 query 的所有 selected blocks，使用 FlashAttention 风格的 online softmax 递推。

KV-outer 中，一个 query 的 $k$ 个 selected blocks 可能分散在不同 CTA：

```text
CTA A: query q over block A
CTA C: query q over block C
CTA F: query q over block F
...
```

任何单个 CTA 都看不到 query 的完整 selected support，不能直接完成全局 softmax。因此论文使用 K1 + K2。

### 6.1 K1：局部归一化 partial attention

对 query/head 的第 $s$ 个 block/chunk，定义 token 集合 $\mathcal J_s$：

$$
L_s
=
\log\sum_{j\in\mathcal J_s}\exp S_j,
$$

$$
O_s
=
\sum_{j\in\mathcal J_s}
\exp(S_j-L_s)V_j.
$$

K1 将结果写入：

$$
O_{buf}[s,i,h],
\qquad
LSE_{buf}[s,i,h].
$$

这里 $O_s$ 是 block/chunk 内已经归一化的局部 attention output。

### 6.2 K2：使用 LSE 精确合并

先合并所有 partial LSE：

$$
a=\max_sL_s,
$$

$$
L
=
a+\log\sum_s\exp(L_s-a).
$$

每个 partial 在全局 softmax 中的权重为：

$$
w_s=\exp(L_s-L).
$$

最终输出：

$$
O=\sum_sw_sO_s.
$$

展开可得：

$$
\begin{aligned}
O
&=
\sum_s
\exp(L_s-L)
\sum_{j\in\mathcal J_s}\exp(S_j-L_s)V_j \\
&=
\sum_s\sum_{j\in\mathcal J_s}\exp(S_j-L)V_j.
\end{aligned}
$$

这正是所有 selected tokens 上的一次全局 softmax。因此：

> **KV-outer 的 two-phase forward 改变了工作拆分，但没有增加新的 attention 数值近似。**

---

## 7. Pre-scheduled Chunking 与热点负载均衡

KV blocks 的流行度高度倾斜：

- sink block 可能被接近所有 query 选择；
- local blocks 通常很热门；
- 大多数远程 block 只关联少量 query。

如果一个 `(KV block, KV head)` 对应一个 CTA：

```text
cold block -> CTA 很快完成
hot block  -> CTA 运行很久
```

GPU 尾部会只剩少量 hot-block CTA，产生严重长尾。

论文的 scheduler 将一个 KV tile 沿 gathered-query 维切成多个 chunk，每个 chunk 最多约：

$$
2kB_k
$$

个 queries，使热门 tile 可以分发到多个 CTA。

### 为什么预先分配 partial slot

切分后，同一 query 的不同 block/chunk partials 由不同 CTA 生成。如果直接累加到最终 output，需要 atomic update，而且 softmax output 不能简单做普通加法。

Scheduler 因此预先为每个 `(query, chunk)` 分配：

$$
s\in[0,k)
$$

的 $O_{buf}$ slot，并记录每个 query 的有效 slot count：

```text
K1: write partial to preassigned slot, no atomic accumulation
K2: read valid slots and perform LSE-weighted combine
```

这样既处理数据相关的负载不均衡，也保持结果确定且精确。

---

## 8. 完整 Forward 数据流

```text
Top-k selection: q2k
        |
        v
Build reverse index: k2q CSR
        |
        v
Scheduler:
  - inspect query count per KV tile
  - split hot tiles into chunks
  - assign O_buf slots
        |
        v
K1 KV-outer partial attention:
  for (KV block, KV head, query chunk):
      load/reuse K/V tile
      gather/concatenate queries
      compute local softmax
      write O_partial and LSE_partial
        |
        v
K2 combine:
  for each query/head:
      combine partial LSEs
      compute split weights
      merge partial outputs
        |
        v
Exact sparse-attention output
```

---

## 9. Q-outer 与 KV-outer 对比表

| 维度 | Q-outer | KV-outer |
|---|---|---|
| 外层遍历 | Query | `(KV block, KV head)` |
| 稀疏图方向 | q2k | k2q |
| 主要复用对象 | Q | K/V block |
| 不规则访问 | K/V gather | Query gather |
| 热门 KV block | 被不同 queries 重复读取 | 在 query chunk 内复用 |
| K/V I/O | $O(NkB_kd_h)$ | $O(Nd_h)$，忽略热点切分重载 |
| 额外中间量 | 较少 | $O_{buf}$ 和 $LSE_{buf}$ |
| 算术强度 | $\approx G=16$ | $\approx\frac23B_k=85.3$ |
| 单位置 MMA 的 $M$ | $G=16$ | 多位置拼接为 128 |
| Softmax 组织 | 单 CTA/online | partial + exact combine |
| 负载均衡 | 以 query 为单位 | 热门 KV row chunking |
| 前处理 | 直接使用 Top-k | 构造 reverse index + schedule |

---

## 10. 这项设计的适用边界

### 10.1 论文推导针对 sparse prefill

第 4.2 节明确讨论 query length 与 KV length 相等的 sparse prefill。Decode 时通常只有一个或少量新 query token：

- 可供拼接的 query positions 很少；
- reverse-index 形态和复用模式不同；
- latency 而非大规模 throughput 可能成为主要目标。

因此不能从论文这段推导直接断言 decode 也应使用完全相同的 KV-outer scheduler。

### 10.2 KV-outer 不是永远更优

如果出现以下情况，额外成本可能抵消复用收益：

- $B_k$ 很小，使 $\frac23B_k$ 不再明显大于 $G$；
- query 数很少，无法形成 query concatenation；
- selection 几乎没有 block reuse；
- reverse-index/scheduler 开销占比过高；
- partial buffer 的 HBM 流量成为瓶颈。

MSA 的选择与参数是协同设计的：

$$
B_k=128,
\qquad
k=16,
\qquad
G=16.
$$

较大的 $B_k$ 提供 K/V 数据复用和较高算术强度；较小的 $k$ 控制 partial-buffer 与 combine 开销；$G=16$ 则使 8 个 query positions 正好拼成 128-row MMA。

---

## 最终记忆方式

```text
Q-outer：
“我有一个 query，现在去找它的 16 个 KV blocks。”
问题：热门 K/V 被不同 query 反复搬运，单 query 的 MMA 又太矮。

KV-outer：
“我有一个 KV block，现在把所有需要它的 queries 叫过来一起算。”
收益：K/V 得到复用，queries 可以拼成大 MMA。
代价：必须反转索引、保存 partials，再做一次精确 combine。
```

因此，MSA 采用 KV-outer 的本质是：

> **把稀疏注意力中的复用中心从 query 转移到连续的 KV block，用更多调度与中间存储换取更少的数据搬运和更高的 Tensor Core 利用率。**

