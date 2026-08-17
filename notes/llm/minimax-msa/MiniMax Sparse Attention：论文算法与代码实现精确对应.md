 
> 论文：*MiniMax Sparse Attention*, arXiv:2606.13392v2。

>

> 代码基准：仓库提交 `80434d7f67877c6570ca19cac444b84bc9855dac`。

>

> 本文只把论文和该提交中的实际代码对应起来，不把 README、注释或未开源训练系统推断成已经实现的功能。

  

## 1. 先给出结论

  

论文中的 MSA 是一个可训练的两分支注意力层：

  

```text

Hidden States X

├─ Index Branch：轻量但 dense 的全上下文 QidxKidx

│ └─ block max → Top-k blocks（每个 GQA group 独立）

└─ Main Branch：只在选中 blocks 上执行精确 softmax attention

└─ output projection

  

训练时另有：KL teacher + stop-gradient + indexer warmup + forced local block

```

  

当前仓库不是完整的可训练 MSA layer，而是面向 NVIDIA SM100 的 forward/inference kernel 包。它完整覆盖的核心链路是：

  

```text

index/proxy QK score

→ block max score

→ Top-k block index

→ q2k 转 k2q CSR

→ sparse scheduler

→ K1：每个 KV block/chunk 的 partial attention

→ K2：利用 LSE 精确合并 partial outputs

```

  

以下论文组件不在仓库中：

  

- `X → Q/K/V/O` 和 `X → Qidx/Kidx` 的模型投影；

- Index Branch 输入的 `stopgrad(X)`；

- KL loss、teacher distribution 和 backward；

- indexer warmup、训练循环和模型 loss 组装；

- 论文使用的完整 H800 训练/推理 kernel；

- 论文 109B 模型、数据和权重。

  

因此，最准确的定位是：

  

> 当前仓库实现了 MSA 的 SM100 稀疏 forward/inference 执行底座及量化扩展，但没有实现论文 Algorithm 1 所描述的完整训练层。

  

---

  

## 2. 论文符号和代码张量

  

设：

  

- $T_q=\sum\texttt{qo\_segment\_lens}$：batch 内 packed query token 总数；

- $T_k=\sum\texttt{kv\_segment\_lens}$：batch 内 packed KV token 总数；

- $H_q$：query head 数；

- $H_{kv}$：KV head 数；

- $G=H_q/H_{kv}$：每个 GQA group 的 query head 数；

- $D=128$：当前 production sparse kernel 支持的 head dimension；

- $B_k=\texttt{blk\_kv}=128$：论文和 production CSR builder 的 KV block size；

- $k=\texttt{topK}$：每个 query、每个 GQA group 选中的 KV block 数。

  

| 论文符号 | 代码名称与形状 | 说明 |

|---|---|---|

| $\mathbf Q$ | `q [Tq, Hq, 128]` | packed query |

| $\mathbf K,\mathbf V$ | `[Tk,Hkv,128]` 或 `[pages,Hkv,128,128]` | dense 或 paged KV |

| $\mathbf Q^{\mathrm{idx}}$ | proxy/index Q，由调用方生成 | 仓库不包含 $W_q^{\mathrm{idx}}$ |

| $\mathbf K^{\mathrm{idx}}$ | proxy/index K，由调用方生成 | 仓库不包含 $W_k^{\mathrm{idx}}$ |

| $\mathbf M^{\mathrm{idx}}$ | `max_score [Hidx,max_k_tiles,Tq]`, FP32 | 每个 query 对每个 KV tile 的最大 score |

| $\mathcal I$ | `kv_block_indexes [Tq,H*,topK]` | query 到 KV block 的选择结果 |

| q2k | `[Hkv,Tq,topK]` | group-shared、batch-local block index |

| reverse $\mathcal I$ | `k2q_row_ptr [Hkv,total_rows+1]` | CSR row pointer |

| reverse $\mathcal I$ | `k2q_q_indices [Hkv,Tq*topK]` | 每个 KV block 对应的 query indices |

| $\mathbf O_{\mathrm{buf}}$ | `O_partial [topK,Tq,Hq,D]` | K1 输出的局部归一化结果 |

| $\operatorname{LSE}_{\mathrm{buf}}$ | `LSE_partial [topK,Tq,Hq]` | K1 输出的局部 log-sum-exp |

| $\mathbf O$ | `O_out [Tq,Hq,D]`, BF16 | K2 合并后的输出 |

| $\operatorname{LSE}$ | `LSE_out [Tq,Hq]`, FP32 | K2 合并后的全局 LSE |

  

公开 sparse forward 的输入输出约束见 [`cute/interface.py`](../python/fmha_sm100/cute/interface.py#L600-L704)，partial buffer 的实际分配见 [`cute/interface.py`](../python/fmha_sm100/cute/interface.py#L1470-L1484)。

  

---

  

## 3. Algorithm 1 逐行对应

  

论文 Algorithm 1 描述的是“一层 MSA 的训练 forward 与辅助 KL loss”。下面的“状态”严格区分完整实现、部分实现和仓库外。

  

| 行 | 论文动作 | 代码对应 | 状态 |

|---|---|---|---|

| 1 | `X → Q,K,V` | public API 直接接收 `q,k,v` | **仓库外**：没有模型 projection |

| 2 | `stopgrad(X) → Qidx,Kidx` | dense proxy/FP4 indexer 接收已生成的 index Q/K | **仓库外**：没有 projection 和 detach |

| 3 | `BlockMaxPool(Qidx,Kidx,Bk)` | dense FMHA `output_maxscore`；或 `fp4_indexer_block_scores` | **已实现/工程重命名** |

| 4 | `TopK(Midx,k)`，强制 local block | `sparse_topk_select` | **Top-k 已实现；逐 query local 不完整** |

| 5 | `TopKAttn(Q,K,V,I)` | adapter → CSR → scheduler → K1 → K2 | **已实现，forward-only** |

| 6 | `O Wo` | 无 | **仓库外** |

| 7 | `KLdiv(...)` | 只有可选 LSE 输出，没有 KL kernel/backward | **未实现** |

| 8 | 返回 `output, LKL` | 返回 `O` 和可选 LSE | **部分实现** |

  

实际 sparse Main Branch 入口为 [`sparse_atten_func`](../python/fmha_sm100/cute/interface.py#L600-L625)，高层 API adapter 为 [`sparse_fmha`](../python/fmha_sm100/sparse_fmha_adapter.py#L271-L406)。

  

---

  

## 4. 论文公式逐项映射

  

### 4.1 Equation (1)：Causal Attention 与 GQA

  

论文：

  

$$

\begin{aligned}

\mathbf{o}_t^{(h)}

&= \sum_{i\le t}\alpha_{t,i}^{(h)}\mathbf{v}_i^{(h)}, \\

\alpha_{t,i}^{(h)}

&=

\frac{

\exp\!\left(\mathbf{q}_t^{(h)}\mathbf{k}_i^{(h)\top}/\sqrt{d_h}\right)

}{

\sum_{j\le t}\exp\!\left(\mathbf{q}_t^{(h)}\mathbf{k}_j^{(h)\top}/\sqrt{d_h}\right)

}.

\end{aligned}

$$

  

代码对应：

  

- 默认 softmax scale 是 `1 / sqrt(head_dim_qk)`：[`api.py`](../python/fmha_sm100/api.py#L841-L850)；

- sparse forward 默认同样使用 `q.shape[-1] ** -0.5`：[`cute/interface.py`](../python/fmha_sm100/cute/interface.py#L706-L717)；

- `G = Hq/Hkv` 必须属于 `{1,2,4,8,16}`：[`cute/interface.py`](../python/fmha_sm100/cute/interface.py#L701-L704)；

- 当前 SM100 sparse main kernel 只支持 `D=128`：[`fwd/atten_fwd.py`](../python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py#L56-L86)；

- causal mask 使用实际 query 位置和 KV 有效长度，不会因为选中了一个 block 就允许访问未来 token：[`common/mask.py`](../python/fmha_sm100/cute/src/common/mask.py#L93-L120)。

  

实现不会显式构造 `N×N` attention matrix，而是 tiled online softmax。

  

### 4.2 Equation (2)：Index → Restricted Attention

  

论文：

  

$$

\mathcal I_i

=

\operatorname{Index}_{\phi}\!\left(\mathbf q_i,\mathbf K_{\le i}\right),

\qquad

\mathbf o_i

=

\operatorname{Attn}\!\left(

\mathbf q_i,

\mathbf K[\mathcal I_i],

\mathbf V[\mathcal I_i]

\right).

$$

  

对应的工程链路是：

  

```text

max_score / FP4 block scores

→ sparse_topk_select

→ kv_block_indexes

→ q2k

→ k2q CSR

→ sparse_atten_func

```

  

其中：

  

- `Index_phi` 的 projection/training 不在仓库；

- `I_i` 对应 `kv_block_indexes` 或内部 `q2k`；

- restricted attention 对应 CuTe-DSL K1/K2 sparse forward。

  

### 4.3 Equation (3)：GQA Group 内共享稀疏索引

  

论文规定同一 GQA group 的所有 query heads 共享选择结果：

  

$$

\mathcal I_i^{(r)}

=

\mathcal I_i^{(h)}

=

\mathcal I_i^{(h')},

\qquad

h,h'\in\mathcal H_r.

$$

  

代码用 q2k 的 `Hkv` 轴表达这一语义：

  

```text

q2k [Hkv, Tq, topK]

```

  

adapter 接受两种输入：

  

- `[Tq,Hkv,topK]`：直接 transpose；

- `[Tq,Hq,topK]`：每隔 `qhead_per_kv` 抽一个代表 head。

  

实现见 [`_convert_kv_block_indexes_to_q2k`](../python/fmha_sm100/sparse_fmha_adapter.py#L205-L222)。

  

重要边界：adapter **不会验证**同一 group 里其余 query heads 的 indices 是否相同。如果调用方传入 Hq 级 indices，组内共享契约由调用方负责。

  

### 4.4 Equation (4)：KV Block 划分

  

论文：

  

$$

\mathcal B_b

=

\left\{

(b-1)B_k+1,\ldots,\min(bB_k,N)

\right\}.

$$

  

代码中对应 `page_size/blk_kv`。production CSR builder 硬限制：

  

```text

blk_kv == 128

topK ∈ {4,8,16,32}

```

  

见 [`prepare_k2q_csr.py`](../python/fmha_sm100/cute/src/sm100/prepare_k2q_csr.py#L4-L22) 和输入校验 [`prepare_k2q_csr.py`](../python/fmha_sm100/cute/src/sm100/prepare_k2q_csr.py#L68-L94)。

  

这不是论文一般公式的完整参数化版本，而是对论文部署配置 `Bk=128` 的硬件特化。

  

### 4.5 Equation (5)：Index Branch Projection

  

论文：

  

$$

\begin{aligned}

\mathbf Q^{\mathrm{idx}}

&=\mathbf X\mathbf W_q^{\mathrm{idx}}

\in\mathbb R^{N\times H_{kv}\times d_{\mathrm{idx}}}, \\

\mathbf K^{\mathrm{idx}}

&=\mathbf X\mathbf W_k^{\mathrm{idx}}

\in\mathbb R^{N\times1\times d_{\mathrm{idx}}}.

\end{aligned}

$$

  

仓库没有：

  

- hidden states `X`；

- `Wq_idx/Wk_idx`；

- projection layer；

- RoPE/normalization 的模型级实现；

- `stopgrad(X)`。

  

仓库只消费调用方准备好的 proxy/index Q、K。若要严格匹配论文，上游应该生成：

  

```text

Qidx: [Tq,Hkv,didx]

Kidx: [Tk,1,didx]

```

  

FP4 indexer 接口允许更一般的 head ratio，因此“API 接受”不等于“严格符合论文模型结构”。

  

### 4.6 Equation (6)：Raw Index Score 与 Block Max

  

论文：

  

$$

S_{i,j}^{\mathrm{idx},(r)}

=

\frac{

\mathbf Q_i^{\mathrm{idx},(r)}

\mathbf K_j^{\mathrm{idx}\top}

}{\sqrt{d_{\mathrm{idx}}}},

$$

  

$$

M_{i,b}^{\mathrm{idx},(r)}

=

\max_{\substack{j\in\mathcal B_b\\j\le i}}

S_{i,j}^{\mathrm{idx},(r)}.

$$

  

仓库有两条实现路径。

  

#### 路径 A：Dense proxy FMHA

  

`fmha_sm100(..., output_maxscore=True)` 在 dense QK mainloop 中对每个 KV tile 取最大值：

  

- Python 输出张量分配为 `[Hidx,max_k_tiles,Tq]`，无效位置初始化为 `-inf`：[`api.py`](../python/fmha_sm100/api.py#L852-L862)；

- `output_maxscore` 会选择 max-score kernel variant：[`api.py`](../python/fmha_sm100/api.py#L874-L893)；

- CUDA mainloop 对 tile score 做归约并可只输出 score：[`sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`](../python/fmha_sm100/csrc/include/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp#L912-L957)。

  

这等价于 block max 的结果语义，但工程上复用了 dense FMHA mainloop，并不是独立命名的 `BlockMaxPool` operator。

  

#### 路径 B：FP4 Indexer

  

`fp4_indexer_block_scores` 使用 block-scaled FP4 MMA，直接得到每个 128-token page 的 FP32 max score。入口在 [`fp4_indexer_interface.py`](../python/fmha_sm100/cute/fp4_indexer_interface.py#L745-L810)。

  

FP4 是仓库的工程扩展。其输出可近似论文的实数 index score，但量化以后不保证 bit-exact 的 Top-k 排序。

  

### 4.7 Equation (7)：Top-k Block Selection

  

论文：

  

$$

\mathcal I_i^{(r)}

=

\operatorname{TopK}_{b}\!\left(

M_{i,\cdot}^{\mathrm{idx},(r)},k

\right).

$$

  

并规定 local block 永远包含在 `I` 中。

  

公开实现为 [`sparse_topk_select`](../python/fmha_sm100/api.py#L1179-L1290)：

  

```python

kv_block_indexes = sparse_topk_select(

max_score,

topk=16,

num_valid_pages=num_pages,

)

```

  

契约：

  

- 输入 `max_score [Hidx,max_k_tiles,Tq]`, FP32、contiguous；

- wrapper 当前只允许 `topk == 16`；

- 输出 `[Tq,Hidx,16]`, INT32；

- block indices 按数值升序，而不是按 score 排序；

- 无效或越界 block 输出为尾部 `-1`；

- `max_k_tiles < 12288`。

  

#### 论文 Top-k 微算法与当前代码不同

  

论文 §4.1 描述的是：

  

```text

每个 warp lane 分片扫描

→ lane-local k-element min-heap

→ register 缓存 heap root

→ k-round shuffle merge

```

  

当前代码使用：

  

```text

[H,K,Q] transpose 为 [H,Q,K]

→ 多级 bit histogram 定位 threshold bin

→ threshold candidates insertion sort

→ warp bitonic 按 block index 升序排序

```

  

设计说明见 [`sparse_topk_select.cuh`](../python/fmha_sm100/csrc/include/sparse_topk_select.cuh#L58-L89)，histogram threshold 见 [`sparse_topk_select.cuh`](../python/fmha_sm100/csrc/include/sparse_topk_select.cuh#L187-L295)，最终排序见 [`sparse_topk_select.cuh`](../python/fmha_sm100/csrc/include/sparse_topk_select.cuh#L711-L768)。

  

结论：两者的 Top-k 集合语义相同，但当前仓库不是论文 H800 heap kernel 的逐行实现。

  

#### Forced local block 的语义差异

  

论文要求对每个 query 强制加入它所在的 block：

  

$$

b_{\mathrm{local}}(i)

=

\left\lfloor\frac{i}{B_k}\right\rfloor.

$$

  

公开 selector 默认：

  

```python

force_begin_blocks = 0

force_end_blocks = 0

```

  

它能强制：

  

- 全局开头的若干 blocks，适合 sink；

- 当前有效序列全局末尾的若干 blocks，适合 decode tail/local window。

  

但在长 prefill 中，不同 query 的 local block 不同，因此全局 `force_end_blocks` 不等价于论文的 per-query local block。严格复现论文时，调用方必须逐 query 显式保证 local block 位于 `kv_block_indexes` 中。

  

### 4.8 Equation (8)：Main Branch Restricted Softmax

  

论文：

  

$$

\mathbf O_i^{(h)}

=

\operatorname{softmax}\!\left(

\frac{

\mathbf Q_i^{(h)}

\bigl(\mathbf K^{(r)}[\mathcal I_i^{(r)}]\bigr)^{\top}

}{\sqrt{d_h}}

\right)

\mathbf V^{(r)}[\mathcal I_i^{(r)}].

$$

  

代码对应 `SparseAttentionForwardSm100` K1 加 `combine` K2：

  

- K1 kernel 定义：[`fwd/atten_fwd.py`](../python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py#L56-L113)；

- K1 使用 k2q CSR/worklist 读取某个 KV block 对应的 queries；

- QK 和 PV 使用 FP32 accumulator；

- online softmax 在每个 block/chunk 内生成 `O_partial/LSE_partial`；

- K2 使用 LSE identity 恢复全部 selected blocks 上的精确 softmax；

- K2 调用：[`cute/interface.py`](../python/fmha_sm100/cute/interface.py#L1530-L1549)。

  

“精确”只针对已经选中的 token 集合。与 dense full attention 的近似误差来自 block selection，而不是 K1/K2 的 softmax 计算。

  

### 4.9 Equations (9)-(11)：KL、Teacher 与 Gradient Detach

  

在 selected token support

$\mathcal I_{i,\mathrm{tok}}^{(r)}$ 上，论文定义 Index Branch 分布：

  

$$

P_{i,j}^{\mathrm{idx},(r)}

=

\frac{

\exp\!\left(S_{i,j}^{\mathrm{idx},(r)}\right)

}{

\displaystyle\sum_{u\in\mathcal I_{i,\mathrm{tok}}^{(r)}}

\exp\!\left(S_{i,u}^{\mathrm{idx},(r)}\right)

}.

$$

  

Main Branch teacher 是同一 GQA group 内各 query head 的**概率平均**：

  

$$

P_{i,j}^{(r)}

=

\frac{1}{G}

\sum_{\ell\in\mathcal H_r}

\frac{

\exp\!\left(S_{i,j}^{(\ell)}\right)

}{

\displaystyle\sum_{u\in\mathcal I_{i,\mathrm{tok}}^{(r)}}

\exp\!\left(S_{i,u}^{(\ell)}\right)

}.

$$

  

KL loss 为：

  

$$

\mathcal L_{\mathrm{KL}}

=

\frac{1}{NH_{kv}}

\sum_{i=1}^{N}\sum_{r=1}^{H_{kv}}

D_{\mathrm{KL}}\!\left(

\operatorname{stopgrad}\!\left(P_{i,\cdot}^{(r)}\right)

\,\middle\|\,

P_{i,\cdot}^{\mathrm{idx},(r)}

\right).

$$

  

Index Branch 输入另外执行：

  

$$

\begin{aligned}

\mathbf Q^{\mathrm{idx}}

&=\operatorname{stopgrad}(\mathbf X)\mathbf W_q^{\mathrm{idx}}, \\

\mathbf K^{\mathrm{idx}}

&=\operatorname{stopgrad}(\mathbf X)\mathbf W_k^{\mathrm{idx}}.

\end{aligned}

$$

  

其含义是：

  

- `Pidx`：Index Branch 在 selected token support 上的 softmax；

- `P`：同一 GQA group 内各 Main Branch query head 的 **概率平均**；

- `KL(stopgrad(P) || Pidx)`；

- `Qidx,Kidx` 的输入为 `stopgrad(X)`。

  

当前仓库没有：

  

- `Pidx/P` 构造；

- group-average teacher；

- KL forward/backward；

- `stopgrad`；

- KL coefficient 和模型 loss 组装。

  

仓库能够返回普通 LSE 或 temperature LSE：[`cute/interface.py`](../python/fmha_sm100/cute/interface.py#L614-L676)。这些是实现 KL 可能需要的中间量，但仅有 LSE 不能视为已经实现论文的 KL 训练。

  

### 4.10 Equation (12)：复杂度

  

论文：

  

$$

F_{\mathrm{GQA}}(N)=2H_qd_hN^2,

$$

  

$$

F_{\mathrm{MSA}}(N)

=

H_{kv}d_{\mathrm{idx}}N^2

+4H_qd_hNkB_k.

$$

  

代码结构确实对应两项：

  

```text

dense/lightweight index score

+

固定 topK × blk_kv 的 sparse main attention

```

  

但仓库本身没有 `didx` projection 或完整模型，无法单独复现论文的 end-to-end FLOPs 数值。

  

### 4.11 Equations (13)-(16)：Q-outer 与 KV-outer

  

论文比较两种 loop order：

  

```text

Q-outer:

每个 query 读取它选择的 K/V blocks

K/V 重复和不规则读取多

  

KV-outer:

每个 KV block 只加载一次

gather 所有选择该 block 的 queries

```

  

当前 CuTe sparse prefill 采用 KV-outer。其直接代码实体是：

  

1. `q2k [Hkv,Tq,topK]`；

2. `build_k2q_csr` 生成 reverse index；

3. scheduler 把热门 CSR row 切成 query chunks；

4. K1 以 `(batch,kv_head,kv_block,query_chunk)` 为工作单元；

5. K2 合并每个 query 的多个 partial outputs。

  

### 4.12 Equations (17)-(19)：Index Branch Value Head Ablation

  

论文附录讨论过额外的 Index Branch value/output head，最终设计将其移除。

  

仓库没有 `Vidx/Oidx/Wo_idx`。当前 indexer 只生成 block score，这与论文最终部署设计一致，但不包含论文消融实验所使用的 value-head 训练实现。

  

---

  

## 5. 从 public API 到 GPU kernel 的完整调用链

  

### 5.1 Plan 分派

  

高层入口是 `fmha_sm100_plan`。当：

  

```text

kv_block_num > 0

且 sparse_kernel_mode == "prefill"

或 auto 模式下 max_q_len > 32

```

  

会进入 CuTe sparse prefill adapter：[`api.py`](../python/fmha_sm100/api.py#L510-L542)。

  

否则继续使用 csrc JIT 路径。短 Q sparse 会在 csrc planner 内展开成 per-token segments：[`api.py`](../python/fmha_sm100/api.py#L562-L568)。

  

### 5.2 Max-score 生成

  

典型 proxy 路径：

  

```python

proxy_plan = fmha_sm100_plan(..., output_maxscore=True)

_, max_score = fmha_sm100(

proxy_q, proxy_k_pages, proxy_v_pages, proxy_plan,

output_o=False,

output_maxscore=True,

)

```

  

这一步是轻量 index/proxy QK 的全上下文扫描，而不是 Main Branch 的 full-attention output。

  

### 5.3 Top-k

  

```python

kv_block_indexes = sparse_topk_select(

max_score.contiguous(),

16,

num_valid_pages=num_pages,

)

```

  

`num_valid_pages` 应显式传入，否则 padding tiles 可能产生不安全的 page index。wrapper 文档也将它标为 strongly recommended：[`api.py`](../python/fmha_sm100/api.py#L1195-L1209)。

  

### 5.4 Hq/Hkv indices 转为 q2k

  

adapter 将：

  

```text

kv_block_indexes [Tq,H*,topK]

```

  

转换为：

  

```text

q2k [Hkv,Tq,topK]

```

  

见 [`sparse_fmha_adapter.py`](../python/fmha_sm100/sparse_fmha_adapter.py#L205-L222)。

  

### 5.5 q2k 转 k2q CSR

  

公共格式定义：[`sparse_index_utils.py`](../python/fmha_sm100/cute/sparse_index_utils.py#L4-L25)。

  

```text

q2k [Hkv,Tq,topK]

↓

k2q_row_ptr [Hkv,total_rows+1]

k2q_q_indices [Hkv,Tq*topK]

```

  

生产 builder 的入口与校验见 [`build_k2q_csr`](../python/fmha_sm100/cute/sparse_index_utils.py#L331-L411)。

  

底层五阶段 CUDA pipeline：

  

```text

build_row_map

→ hist

→ row_prefix

→ tile_prefix_smem

→ scatter

```

  

见 [`prepare_k2q_csr.py`](../python/fmha_sm100/cute/src/sm100/prepare_k2q_csr.py#L29-L37)。

  

### 5.6 CSR 到底是什么

  

CSR 是 **Compressed Sparse Row，压缩稀疏行**。它不是一种新的 attention 算法，而是一种把稀疏关系存进连续数组的数据结构。

  

MSA 的 Top-k 最自然地产生 q2k 关系：

  

```text

query q0 → KV blocks [0, 1]

query q1 → KV blocks [0, 2]

query q2 → KV blocks [1, 2]

query q3 → KV blocks [2, 3]

```

  

这适合回答：

  

> 对于 query `q`，它选择了哪些 KV blocks？

  

但 KV-outer kernel 需要反过来回答：

  

> 对于 KV block `b`，有哪些 queries 选择了它？

  

因此代码把 q2k 反转为 k2q：

  

```text

KV block 0 → queries [0, 1]

KV block 1 → queries [0, 2]

KV block 2 → queries [1, 2, 3]

KV block 3 → queries [3]

```

  

#### 把选择关系看成稀疏矩阵

  

对于一个 KV head，可以定义二值矩阵：

  

$$

A_{b,q}

=

\begin{cases}

1, & \text{query }q\text{ 选择了 KV block }b,\\

0, & \text{否则}.

\end{cases}

$$

  

上面的例子对应：

  

$$

A=

\begin{bmatrix}

1&1&0&0\\

1&0&1&0\\

0&1&1&1\\

0&0&0&1

\end{bmatrix}.

$$

  

行是 KV block，列是 query。矩阵绝大多数位置为零，因此不值得显式存储整个 $B\times T_q$ 矩阵。

  

#### CSR 的两个核心数组

  

CSR 只保存非零元素的位置：

  

```text

k2q_row_ptr = [0, 2, 4, 7, 8]

k2q_q_indices = [0, 1, 0, 2, 1, 2, 3, 3]

```

  

其中：

  

- `k2q_row_ptr[b]` 是 block `b` 的 query 列表在 `k2q_q_indices` 中的起点；

- `k2q_row_ptr[b+1]` 是终点；

- block `b` 的 query 数量为 `row_ptr[b+1] - row_ptr[b]`；

- block `b` 的 query 列表是半开区间

`q_indices[row_ptr[b]:row_ptr[b+1]]`。

  

逐行展开：

  

| KV block `b` | CSR 切片 | 得到的 queries |

|---:|---|---|

| 0 | `q_indices[0:2]` | `[0,1]` |

| 1 | `q_indices[2:4]` | `[0,2]` |

| 2 | `q_indices[4:7]` | `[1,2,3]` |

| 3 | `q_indices[7:8]` | `[3]` |

  

所以 `row_ptr` 的长度永远是“行数加一”。最后一个元素同时等于非零关系总数 `nnz`。

  

#### 为什么 MSA 必须做这次反转

  

如果直接按 q2k 执行 Q-outer attention：

  

```text

取 query 0 → 加载 blocks 0、1

取 query 1 → 再次加载 blocks 0、2

取 query 2 → 再次加载 blocks 1、2

```

  

热门 K/V block 会被重复读取，而且不同 query 的选择很不规则。

  

转换成 k2q CSR 后可以执行 KV-outer：

  

```text

加载 KV block 0 一次

→ gather queries 0、1

→ 一起做 QK/PV

  

加载 KV block 2 一次

→ gather queries 1、2、3

→ 一起做 QK/PV

```

  

这样做的主要收益是：

  

1. 同一个 K/V block 可以被多个 queries 复用；

2. gathered queries 可以拼成更饱满的 tensor-core MMA tile；

3. CSR 只存 Top-k 产生的有效边，空间约为 $O(T_qk)$，而不是 $O(BT_q)$；

4. `row_ptr[b+1]-row_ptr[b]` 直接给出 block 热度，scheduler 可以据此切分热门 row。

  

#### 代码中的实际形状

  

概念上，每个 KV head 都有一个独立的稀疏矩阵。代码将这些 CSR 并排存放：

  

```text

q2k_indices: [Hkv, Tq, topK]

k2q_row_ptr: [Hkv, total_rows + 1]

k2q_q_indices: [Hkv, Tq * topK]

```

  

其中 `q2k_indices` 中的 block id 和 `k2q_q_indices` 中的 query id 都是 batch-local 编号；`cu_seqlens_q/k` 和内部 row map 负责把 varlen batch 映射到 packed tensors。

  

在没有 `-1` padding 时，每个 query 恰好选择 `topK` 个 blocks，因此：

  

$$

\operatorname{nnz}=T_q\times k.

$$

  

有 `-1` padding 或无效选择时，实际有效 `nnz` 可以更少。

  

#### CSR builder 的五个阶段

  

production builder 不是在 CPU 上建立 Python list，而是在 GPU 上执行：

  

1. `build_row_map`：建立 packed row 与 `(batch,block)` 的映射；

2. `hist`：统计每个 KV block 被多少 queries 选择；

3. `row_prefix`：对计数做 prefix sum，生成 `row_ptr`；

4. `tile_prefix_smem`：计算并行 scatter 的写入起点；

5. `scatter`：把 query id 写入对应 CSR row，并可同时生成调度信息。

  

因此，文档中的“CSR 流程”完整含义是：

  

```text

Top-k 给出的 query→block 邻接表

→ GPU histogram/prefix/scatter

→ block→query 的压缩邻接表

→ KV-outer sparse attention scheduler

```

  

### 5.7 Scheduler

  

一个 k2q row 的 query 数可能差异很大。scheduler 会：

  

- 选择 `target_q_per_cta`；

- 把热门 row 切成多个 query chunks；

- 给每个 `(query,kv_head)` partial 分配 split slot；

- 生成六元组 work item：

  

```text

[kv_head, row, q_begin, q_count, batch, kv_block]

```

  

相关实现位于 [`prepare_scheduler.py`](../python/fmha_sm100/cute/src/sm100/prepare_scheduler.py)。当 `return_schedule=True` 时，CSR builder 可以融合生成 `scheduler_metadata`、`qsplit_indices` 和 `split_counts`：[`prepare_k2q_csr.py`](../python/fmha_sm100/cute/src/sm100/prepare_k2q_csr.py#L139-L179)。

  

### 5.8 K1：KV-outer Partial Attention

  

K1 的一个 CTA 固定处理：

  

```text

(batch, kv_head, kv_block, query chunk)

```

  

执行过程：

  

```text

加载一次 K/V block

→ 从 CSR gather queries

→ 将 query positions × GQA heads 拼成 MMA rows

→ QK

→ causal mask

→ online softmax

→ PV

→ 写入预分配的 O_partial/LSE_partial slot

```

  

K1 主类：[`fwd/atten_fwd.py`](../python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py#L56-L113)。

  

### 5.9 Query Concatenation

  

同一个 KV head 对应 `G` 个 query heads。内核将：

  

$$

\left\lceil\frac{128}{G}\right\rceil

$$

  

个 query positions 与每个 position 的 `G` 个 heads 拼成 128 个 MMA rows。

  

例如论文配置 `Hq=64,Hkv=4,G=16`：

  

```text

8 query positions × 16 query heads = 128 MMA rows

```

  

这既保持 group-shared selection，又能提高 tensor-core 利用率。

  

### 5.10 K2：利用 LSE 精确合并

  

对 query 的各个 partial split，设 K1 输出为 `(Os,LSEs)`。K2 计算：

  

$$

\begin{aligned}

a

&=\max_s\operatorname{LSE}_s, \\

\operatorname{LSE}

&=a+\log\sum_s\exp\!\left(\operatorname{LSE}_s-a\right), \\

\mathbf O

&=\sum_s

\exp\!\left(\operatorname{LSE}_s-\operatorname{LSE}\right)

\mathbf O_s.

\end{aligned}

$$

  

对应代码：

  

- stable LSE reduction：[`fwd/combine.py`](../python/fmha_sm100/cute/src/sm100/fwd/combine.py#L767-L820)；

- weighted partial O accumulation：[`fwd/combine.py`](../python/fmha_sm100/cute/src/sm100/fwd/combine.py#L903-L948)；

- Python 调用：[`cute/interface.py`](../python/fmha_sm100/cute/interface.py#L1532-L1542)。

  

该合并是 softmax 的严格分块恒等式，不是数值近似。

  

---

  

## 6. 论文 §4 Kernel Design 与代码对应

  

### 6.1 Exp-free Top-k

  

论文观察到 softmax 保序，因此直接对 raw block scores 做 Top-k。

  

当前代码同样不在 Top-k 前计算 exp/normalization，语义完整实现；但具体选择算法由论文 heap 方案替换为 histogram 方案。

  

### 6.2 KV-outer / Reverse Sparse Index

  

论文设计在代码中完整体现为：

  

```text

q2k

→ k2q CSR

→ KV-block outer scheduler

→ gather queries

```

  

这是论文 kernel design 与当前 SM100 源码之间最直接、最忠实的结构对应。

  

### 6.3 Pre-scheduled Tile Chunking

  

论文让热门 KV block 沿 gathered-query 维度切分。当前 scheduler 预先生成 query chunks 和 split slots，避免多个 CTA 对 floating-point output 做 atomic add。

  

### 6.4 Two-phase Forward

  

论文 K1/K2 在代码中完整存在：

  

- K1：block/chunk partial attention；

- K2：LSE-aware exact combine。

  

### 6.5 Sparse KL Loss

  

论文 §4.3 的：

  

- Main/Index LSE fusion；

- KL backward；

- persistent dynamic load balancing；

  

没有作为完整训练 kernel 开源。当前 forward 可以返回 LSE/temperature LSE，但没有 KL teacher、loss 和 backward。

  

---

  

## 7. Prefill、Decode 与 C++/CuTe 两条执行栈

  

### 7.1 csrc JIT 栈

  

主要提供：

  

- dense FMHA；

- paged FMHA；

- max-score 输出；

- `sparse_topk_select`；

- 短 Q/per-token sparse variant。

  

variant 分派依据 `kv_block_indexes/max_score/out` 是否存在：[`api.py`](../python/fmha_sm100/api.py#L874-L893)。

  

### 7.2 CuTe-DSL 栈

  

主要提供：

  

- block-sparse prefill；

- q2k→k2q CSR 和 scheduler；

- K1/K2；

- BF16/FP8/NVFP4/FP4 路径；

- paged FP8 decode wrapper。

  

### 7.3 Decode 边界

  

必须区分：

  

1. csrc 短 Q 路径可以依据 `kv_block_indexes` 使用 sparse variant；

2. CuTe paged FP8 decode 当前是 dense split-KV，非空 q2k 并未完整实现。

  

因此不能笼统地说“CuTe decode 已经实现论文 sparse decode”。

  

---

  

## 8. 量化路径与论文关系

  

### 8.1 BF16/FP8 Main Branch

  

CuTe sparse forward 支持 BF16、FP8 E4M3，以及部分 BF16 Q + FP8 K/V staging 组合。QK/PV 的 MMA dtype 可以分别选择，最终输出为 BF16。

  

### 8.2 FP4 Indexer

  

仓库提供 FP4 block score indexer：

  

- MXFP4：E2M1 value + E8M0 group scale；

- NVFP4：E2M1 value + E4M3 group scale；

- block score 输出仍为 FP32。

  

这是论文公式之外的工程扩展，用更低精度近似 index score。

  

### 8.3 NVFP4 K/V Main Branch

  

仓库还能消费 packed NVFP4 K/V 和 block/global scales。该路径属于 SM100 部署扩展，不应反向解释为论文 Algorithm 1 中定义的训练数值格式。

  

---

  

## 9. 实现忠实度总表

  

| 论文组件 | 当前仓库 | 判断 |

|---|---|---|

| GQA group-shared sparse pattern | q2k 以 Hkv 为 head 轴 | 完整，但调用方需满足组内一致性 |

| `Bk=128` block sparse | CSR/main kernel 硬特化 128 | 完整实现论文部署配置 |

| raw score block max | dense max-score 或 FP4 indexer | 完整/工程替代 |

| exp-free Top-k | raw score 直接 selection | 完整 |

| heap/shuffle Top-k 微算法 | histogram/insertion/bitonic | 算法替代 |

| 每 query forced local block | 只有 global begin/end forcing | 一般 prefill 不等价 |

| Main Branch exact selected softmax | K1 + K2 | 完整 forward |

| KV-outer loop order | k2q CSR + scheduler | 完整 |

| query concatenation | `128/G` positions × `G` heads | 完整 |

| two-phase forward | partial O/LSE + combine | 完整 |

| Index projection | 无 | 仓库外 |

| stop-gradient | 无 | 缺失 |

| KL teacher/loss | 无 | 缺失 |

| indexer warmup | 无 | 缺失 |

| sparse backward | 无 | 缺失 |

| output projection | 无 | 仓库外 |

| H800 论文 kernel | 当前代码面向 SM100 | 不能逐行等同 |

| FP4/NVFP4 | 论文非核心描述 | 工程扩展 |

  

---

  

## 10. 严格复现论文时的上游检查表

  

如果要在模型代码中使用该仓库并严格匹配论文语义，应检查：

  

1. `Qidx` 形状是 `[Tq,Hkv,didx]`；

2. `Kidx` 使用一个跨 group 共享的 index key head；

3. Index Branch 输入执行 `stop_gradient(X)`；

4. index Q/K 使用与模型位置一致的 normalization/RoPE；

5. score 使用正确的 `1/sqrt(didx)` scaling；

6. 每 128 个 KV tokens 做 causal block max；

7. 每个 GQA group 独立选择，group 内 heads 共享 indices；

8. 每个 query 的 local block 必须显式加入并占用一个 slot；

9. indices 必须去重、batch-local、升序，`-1` 只能位于尾部；

10. `num_valid_pages` 必须正确，避免 padding/OOB blocks；

11. `cu_seqlens_q/k`、page table、`seqused_k` 和 causal offset 必须一致；

12. 训练时在仓库外实现 KL teacher、detach、warmup 和 backward；

13. 不应把 FP4 indexer 的结果视为论文实数 index score 的 bit-exact 实现；

14. 不应把 SM100 benchmark 直接与论文 H800 数字混用。

  

---

  

## 11. 推荐阅读代码顺序

  

如果希望从 Python 一直跟到 GPU kernel，推荐按以下顺序阅读：

  

1. [`README.md`](../README.md)：两套执行栈和公开 API；

2. [`api.py`](../python/fmha_sm100/api.py)：plan、run、max-score、Top-k 和分派；

3. [`sparse_fmha_adapter.py`](../python/fmha_sm100/sparse_fmha_adapter.py)：dense API 到 sparse backend 的桥接；

4. [`sparse_index_utils.py`](../python/fmha_sm100/cute/sparse_index_utils.py)：q2k/k2q CSR 格式；

5. [`prepare_k2q_csr.py`](../python/fmha_sm100/cute/src/sm100/prepare_k2q_csr.py)：production CSR builder；

6. [`prepare_scheduler.py`](../python/fmha_sm100/cute/src/sm100/prepare_scheduler.py)：hot-row chunking 和 split slot；

7. [`fwd/atten_fwd.py`](../python/fmha_sm100/cute/src/sm100/fwd/atten_fwd.py)：K1 KV-outer sparse attention；

8. [`fwd/combine.py`](../python/fmha_sm100/cute/src/sm100/fwd/combine.py)：K2 LSE/O combine；

9. [`sparse_topk_select.cuh`](../python/fmha_sm100/csrc/include/sparse_topk_select.cuh)：当前 Top-k 微算法；

10. [`fp4_indexer_interface.py`](../python/fmha_sm100/cute/fp4_indexer_interface.py)：FP4 block score 扩展。

  

---

  

## 12. 最终数据流总览

  

```text

模型侧（仓库外）

X

├─ Wq/Wk/Wv ───────────────→ Q, K, V

└─ stopgrad + Wqidx/Wkidx ─→ Qidx, Kidx

│

仓库 score/index 路径 ▼

dense proxy QK 或 FP4 index MMA

→ causal block max

→ max_score [Hidx,B,Tq]

→ exp-free Top-k

→ kv_block_indexes [Tq,Hkv,k]

→ 调用方补齐/保证 per-query local block

│

仓库 sparse Main Branch ▼

q2k [Hkv,Tq,k]

→ k2q CSR

→ scheduler / hot-row chunking

→ K1 KV-outer partial attention

├─ O_partial [k,Tq,Hq,D]

└─ LSE_partial [k,Tq,Hq]

→ K2 stable LSE combine

├─ O [Tq,Hq,D]

└─ LSE [Tq,Hq]

│

模型侧（仓库外） ▼

O × Wo

+ LM loss

+ lambda × sum(layer KL losses)

```

  

这条数据流是论文算法与当前仓库之间最准确的对应关系：论文定义完整可训练层，仓库实现其中的 index score/selection 接口和 SM100 sparse forward 执行部分。