# Adam 论文前置学习路线

这份路线面向第一次接触 SGD、minibatch、Momentum、AdaGrad 和 RMSProp 的读者。目标不是先学完一门优化理论课，而是在 **6–8 小时**内建立足够的共同语言，然后重新阅读 Adam 的 Algorithm 1。

核心依赖只有一条：

```text
模型、参数、loss、gradient
            │
            ▼
      Gradient Descent
            │
            ▼
 SGD 与 minibatch ───────┐
            │            │
            ▼            ▼
       Momentum       AdaGrad
                         │
                         ▼
                      RMSProp
            │            │
            └─────┬──────┘
                  ▼
                 Adam
```

Adam 不是一个适合零基础直接阅读的起点。它把 Momentum 的“平滑梯度方向”和 RMSProp 的“按坐标缩放步长”组合起来，又加入零初始化的 bias correction。前两条支路不知道时，原论文里的 `m`、`v` 和 `beta` 只会像记号堆叠。

## 先统一八个词

设训练集有 $N$ 个样本，第 $i$ 个样本产生 loss $\ell_i(\theta)$：

\[
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i(\theta)
\]

| 词 | 在这条路线中的含义 |
|---|---|
| sample | 一条训练数据 |
| parameter $\theta$ | 训练要改变的模型数值，例如 weight 和 bias |
| loss $\ell$ | 衡量当前预测有多差的标量 |
| gradient $\nabla_\theta L$ | loss 对各参数的偏导数组成的向量；负梯度给出局部下降方向 |
| learning rate $\eta$ | 一次更新沿该方向走多远 |
| step / iteration | 取一次数据、算一次梯度并更新一次参数 |
| minibatch | 一次 step 中使用的一小组样本，大小记为 $B$ |
| epoch | 整个训练集大约被使用一遍；若不丢弃尾部，每个 epoch 约有 $N/B$ 个 step |

命名上有一个陷阱：严格区分时，batch gradient descent 使用全部 $N$ 个样本，SGD 每步使用一个随机样本，minibatch SGD 每步使用 $1<B<N$ 个样本；现代框架和论文经常把 minibatch SGD 简称为 “SGD”。[Deep Learning Book §8.1.3](https://www.deeplearningbook.org/contents/optimization.html) 和[《动手学深度学习》小批量随机梯度下降](https://zh.d2l.ai/chapter_optimization/minibatch-sgd.html)都明确给出了这组区分。

## 最短主线

下面每一阶段都只在完成“过关练习”后继续。主教材使用《动手学深度学习》的官方中文在线版，它把解释、可运行 notebook 和从零实现放在同一页；不需要同时看多套教程。

### 0. 可选补丁：导数、梯度与自动微分（45–60 分钟）

**何时需要**：如果还不能解释偏导数、梯度或链式法则，先完成这一阶段；否则直接进入阶段 1。

**材料**：

- [《动手学深度学习》2.4 微积分](https://zh.d2l.ai/chapter_preliminaries/calculus.html)：只读导数、偏导数、梯度和链式法则，暂时跳过不相关细节。
- [《动手学深度学习》2.5 自动微分](https://zh.d2l.ai/chapter_preliminaries/autograd.html)：理解框架怎样记录计算图并得到 gradient。

**学习目标**：区分“函数值”和“函数对参数的梯度”；知道 backward 负责求梯度，optimizer step 才负责修改参数。

**过关练习**：对单样本线性模型

\[
\hat y=wx+b,\qquad \ell=(\hat y-y)^2
\]

手算 $\partial\ell/\partial w$ 和 $\partial\ell/\partial b$，再用 PyTorch autograd 核对。两者一致即可，不需要推一般反向传播证明。

### 1. 先看见一遍完整训练循环（60–90 分钟）

**材料**：

- [《动手学深度学习》3.1 线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html)：模型、平方 loss、解析解与 minibatch SGD 的基本关系。
- [《动手学深度学习》3.2 从零实现线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html)：从数据采样、前向、loss、backward 到参数更新的完整代码。
- 可选实现复核：[PyTorch 官方优化循环教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：核对 `zero_grad → backward → step` 在框架中的职责。

**学习目标**：能把训练循环说成：

```text
取 minibatch → forward → 聚合 loss → backward → update parameters → 清 gradient
```

**过关练习**：自己画一张上述数据流图，并在每条边标出 `data / prediction / loss / gradient / parameter`。然后指出哪些量只属于当前 step，哪些量跨 step 保留。

### 2. Gradient Descent → SGD → minibatch SGD（90 分钟）

**材料，按顺序读**：

1. [梯度下降](https://zh.d2l.ai/chapter_optimization/gd.html)：负梯度、学习率，以及狭长 loss surface 上的振荡。
2. [随机梯度下降](https://zh.d2l.ai/chapter_optimization/sgd.html)：用随机样本的梯度近似总体梯度。
3. [小批量随机梯度下降](https://zh.d2l.ai/chapter_optimization/minibatch-sgd.html)：梯度噪声与向量化效率之间的折中。

三者只改变“用哪些样本估计当前梯度”：[Deep Learning Book §8.1.3、§8.3.1](https://www.deeplearningbook.org/contents/optimization.html)说明了经验目标可以分解为逐样本 loss，并且均匀抽样的 minibatch 平均梯度可作为总体梯度的无偏估计。

\[
g_{\text{full}}=\frac1N\sum_{i=1}^{N}\nabla\ell_i(\theta),
\qquad
g_{\mathcal B}=\frac1B\sum_{i\in\mathcal B}\nabla\ell_i(\theta)
\]

**学习目标**：能解释为什么不用每一步都遍历全量数据；能同时说出较大 batch 的计算效率/低噪声优势和单步成本。

**过关练习**：用四条线性回归样本，在同一个 $\theta$ 上分别计算 $B=1,2,4$ 的梯度。回答：

- 哪个是 full gradient？
- 哪些只是 full gradient 的随机估计？
- batch size 改变后，一个 epoch 的 step 数怎样变化？

### 3. Momentum：让更新方向有记忆（60–90 分钟）

**材料**：

- [《动手学深度学习》11.6 动量法](https://zh.d2l.ai/chapter_optimization/momentum.html)：从病态二维目标、指数加权平均到从零实现。
- 可选快速复核：[Stanford CS231n — Parameter updates](https://cs231n.github.io/neural-networks-3/#update)给出了滚珠物理直觉和简短伪代码。

**学习目标**：知道 vanilla SGD 只看当前 $g_t$，Momentum 额外保存与参数同 shape 的 velocity；连续同向的梯度会积累，来回变号的方向会被平滑。

**过关练习**：任选教材中的一种明确 convention，对梯度序列

\[
g_1=(4,1),\quad g_2=(-4,1),\quad g_3=(4,1)
\]

手算三步 velocity。解释为什么第一维容易振荡，而第二维应积累速度。

不同教材/框架会把 learning rate 和 $1-\beta$ 放在不同位置，velocity 的绝对数值可能不同，但只要参数更新等价就不矛盾。[PyTorch `torch.optim.SGD` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)专门说明了其 Momentum convention 与部分文献的差异；现阶段不要混用两套公式做逐数值比较。

### 4. AdaGrad → RMSProp：让每个坐标有自己的尺度（90 分钟）

**材料，按顺序读**：

1. [《动手学深度学习》11.7 AdaGrad](https://zh.d2l.ai/chapter_optimization/adagrad.html)：累计逐元素平方梯度，并以其平方根缩放每个坐标的更新。
2. [《动手学深度学习》11.8 RMSProp](https://zh.d2l.ai/chapter_optimization/rmsprop.html)：把“从第一步开始永久累计”改为平方梯度的指数移动平均。

[AdaGrad 原始论文](https://www.jmlr.org/papers/v12/duchi11a.html)的动机之一是为稀疏但有预测力的特征自适应调整步长；[Deep Learning Book §8.5.1–§8.5.2](https://www.deeplearningbook.org/contents/optimization.html)进一步解释了 AdaGrad 的累计量只增不减，可能让深度网络训练后期的有效学习率过早变小，而 RMSProp 会逐渐遗忘久远历史。RMSProp 没有正式的原始论文，一手出处是 Geoffrey Hinton 2012 年课程的 [Lecture 6 官方课件](https://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)，其中 Lecture 6e 引入了该算法。

**学习目标**：

- AdaGrad/RMSProp 保存的不是方向，而是逐坐标的 squared-gradient scale；
- AdaGrad 使用全历史累加，RMSProp 使用带衰减的近期历史；
- 平方、开方和除法都是逐元素操作，不是向量范数。

**过关练习**：对一维序列 $g=(10,0,0,0)$，分别手算 AdaGrad 的累加器和衰减率 $\rho=0.9$ 的 RMSProp 累加器。说明为什么前者永远保留第一次大梯度，而后者会遗忘它。

### 5. 再回到 Adam（90 分钟）

**材料，按顺序读**：

1. [《动手学深度学习》11.10 Adam](https://zh.d2l.ai/chapter_optimization/adam.html)：先用熟悉的符号看组合关系和代码。
2. [Adam 原论文](https://arxiv.org/abs/1412.6980)：本轮只精读第 2 页 Algorithm 1、第 2–3 页更新规则与 bias correction；定理和证明后读。
3. 实现核对：[PyTorch `torch.optim.Adam` 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)。

此时应把 Adam 读成两条支路的合并：

| 方法 | 跨 step 保存的逐参数状态 | 它回答的问题 |
|---|---|---|
| SGD | 无 | 当前 minibatch 的下降方向是什么？ |
| Momentum | velocity / 一阶历史 | 最近的方向是否持续一致？ |
| AdaGrad | 全历史平方梯度和 | 每个坐标历来有多大？ |
| RMSProp | 平方梯度的 EMA | 每个坐标最近有多大？ |
| Adam | 一阶矩 EMA `m` + 二阶原始矩 EMA `v` | 最近往哪里走，并按最近尺度缩放多少？ |

**过关练习**：不要立即训练大模型。先为一个二维参数和三个给定梯度建立表格，每行写出 $g_t,m_t,v_t,\hat m_t,\hat v_t,\theta_t$；再写一个只接收 `parameter, gradient, state` 的 Adam `step` 函数，与 PyTorch 在 `weight_decay=0` 的前三步结果对齐。

## 进入 Adam 原论文的检查标准

下面八项中至少七项能脱离资料回答，再继续读 Adam：

- [ ] 能用一个公式区分 per-example loss 和整个训练集的 objective。
- [ ] 能区分 full-batch GD、单样本 SGD 和 minibatch SGD，并知道现代语境常把第三种简称 SGD。
- [ ] 能区分 iteration 与 epoch，并根据 $N,B$ 估算每个 epoch 的 step 数。
- [ ] 能为 $\hat y=wx+b$ 手算一次 gradient 和 parameter update。
- [ ] 能解释 minibatch 梯度为什么有噪声，以及 batch 变大对噪声、并行度和单步成本的影响。
- [ ] 能解释 Momentum 为什么需要一个与参数同 shape 的状态。
- [ ] 能解释 AdaGrad 的“永久累积”问题，以及 RMSProp 如何用 EMA 修复它。
- [ ] 一看到 Adam 的 `m` 和 `v`，能分别把它们对应到方向历史与平方梯度尺度，并知道 `v` 是二阶原始矩估计，不是中心化方差。

## 现在可以刻意跳过什么

第一轮不需要学习 Hessian、完整凸优化理论、regret bound、Nesterov Momentum、Adadelta 或 Adam 收敛反例；也先不要读 AdamW。它们都很重要，但不是理解 Adam Algorithm 1 的必要依赖。

若想用一本权威资料统一复核术语，可在完成主线后选读 [Deep Learning Book Chapter 8](https://www.deeplearningbook.org/contents/optimization.html) 的 §8.1.3、§8.3.1–§8.3.2、§8.5.1–§8.5.3。若想快速复习所有更新式，可读 [Stanford CS231n 官方课程笔记](https://cs231n.github.io/neural-networks-3/#update)。两者都应作为复核材料，不替代上面的手算与最小实现。

## 建议进度

| 天 | 内容 | 最小产出 |
|---|---|---|
| Day 1 | 阶段 0（按需）+ 阶段 1 | 一张训练循环数据流图 |
| Day 2 | 阶段 2 | $B=1,2,4$ 的梯度对比表 |
| Day 3 | 阶段 3 | 三步 Momentum 手算 |
| Day 4 | 阶段 4 | AdaGrad/RMSProp 状态对比 |
| Day 5 | 阶段 5 | 三步 Adam 手算 + 最小 `step` 实现 |

若每天只有一小时，就把它拉长到一周。判断是否完成的标准是能解释状态和更新，而不是页面是否滚到底。
