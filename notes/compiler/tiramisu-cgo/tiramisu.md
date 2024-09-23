# TIRAMISU: A Polyhedral Compiler for Expressing Fast and Portable Code

这篇笔记精读一下这篇文章，用来学习一些 compiler-topic 论文的写法。

## Abstract

首先直接说明这篇文章做了什么：

> This paper introduces TIRAMISU, a polyhedral framework designed to generate high performance code for multiple platforms including multicores, GPUs, and distributed machines.

表明 TIRAMISU 是一个 polyhedral framework，用来生成高性能代码，支持多平台，包括多核、GPU 和分布式机器。

随后，介绍 Tiramisu 主要做了什么事情，有什么特性：

> TIRAMISU introduces a scheduling language with novel commands to explicitly manage the complexities that arise when targeting these systems.

> The framework is designed for the areas of image processing, stencils, linear algebra and deep learning.

> TIRAMISU has two main features: it relies on a flexible representation based on the polyhedral model and it has a rich scheduling language allowing fine-grained control of optimizations. TIRAMISU uses a four-level intermediate representation that allows full separation between the algorithms, loop transformations, data layouts, and communication. This separation simplifies targeting multiple hardware architectures with the same algorithm.

> We evaluate TIRAMISU by writing a set of image processing, deep learning, and linear algebra benchmarks and compare them with state-of-the-art compilers and hand-tuned libraries. We show that TIRAMISU matches or outperforms existing compilers and libraries on different hardware architectures, including multicore CPUs, GPUs, and distributed machines.

上面其实都是在一段 Abstract 写完的，但是分为了四个部分：
- 介绍 Tiramisu 引入了一个调度语言。
- 介绍 Tiramisu 的适用领域。
- Tiramisu 有两个主要的特性：基于 polyhedral 模型，并且有丰富的调度语言；使用四级 IR，用于简化在多个硬件架构上运行同一个算法。
- 介绍 Tiramisu 的 benchmark 结果。

以上其实就已经把整篇文章的所有部分在一个简短的 Abstract 先介绍一遍。

## Introduction

首先介绍这篇文章产生的背景，也就是为什么要写这篇文章，解决了什么问题。在第一段中表明：

> Generating efficient code for high performance systems is becoming more and more difficult as these architectures are increasing in complexity and diversity. Obtaining the best performance requires complex code and data layout transformations, management of complex memory hierarchies, and efficient data communication and synchronization.

也就是在如今的硬件架构越来越复杂的情况下，生成高性能代码变得越来越困难，需要考虑复杂的代码和数据布局变换、内存层次结构的管理、高效的数据通信和同步。

随后，举了一个例子来解释这个问题，也就是一个 hello world 的例子，这个例子应该是贯穿全文的，为了便于理解这篇文章的 insights 在哪里，需要用这个例子说明并让读者进行理解。

> For example, consider generalized matrix multiplication (gemm), which computes C = αAB + βC and is a building block of numerous algorithms, including simulations and convolutional neural networks. Highly-tuned implementations require fusing the multiplication and addition loops, as well as applying two-level tiling, vectorization, loop unrolling, array packing [20], register blocking, and data prefetching. Furthermore, tuned implementations separate partial tiles from full tiles, since partial tiles cannot fully benefit from the same optimizations. High performance GPU implementations require even more optimizations, including coalescing memory accesses, managing data movement between global, shared, and register memory, and inserting synchronization primitives. Automatically generating such complex code is still beyond the capabilities of state-of-the-art compilers. The importance of kernels such as gemm motivates vendors to release immensely complex hand-optimized libraries for these kernels. However, for most users, obtaining this level of performance for their own code is challenging, since the effort required to explore the space of possible implementations is intractable when handcoding complicated code transformations.

以 GEMM 为例，这篇文章介绍生成一个高性能的 GEMM 需要许多层面的优化，包括 tilling，loop unrolling, array packing, register blocking, data prefetching 等，自动生成高度优化的代码对于当前 STOA 的 compiler 很困难。因此，这段主要在介绍当前 background 下面临的 challenges，便于下文去开展 insights 和 observations。

接下来文章介绍过去的工作是如何做的，也就是介绍之前 SOTA 的背景：

> Previous work using the polyhedral model has shown success in implementing complex iteration space transformations [49], [8], [44], [22], [46], [37], data locality optimizations [27], [21], and memory management optimizations [17], [43], [29], [38], [13]. Although polyhedral compilers can represent these program and data transformations, they still do not successfully select transformations that result in the best performance. Currently, these compilers do not match the performance of hand-optimized kernels for algorithms such as gemm. The blue bars in Figure 1 show the performance of stateof-the-art polyhedral compilers for gemm compared to the Intel MKL [26] and Nvidia cuBLAS [35] libraries. Fullyautomatic polyhedral compilers such as Polly [22] and Pluto [8] improve productivity, but do not obtain the desired level of performance since their search techniques consider only a subset of the necessary optimizations and rely on less accurate machine models, leading the compiler to make suboptimal decisions. Other polyhedral frameworks, such as AlphaZ [51] and CHiLL [10], eschew full automation and instead expose a scheduling language that enables users to productively explore the space of possible transformations. While these frameworks achieve better performance, their scheduling languages are not designed to target distributed systems. For example, they do not allow the user to partition computations, send data across nodes, or insert required synchronization.

尽管之前的工作取得了一些成果，**但是**在 **GEMM** 这类算子中明显不如手工优化的 kernel 性能好，例如 Intel MKL 和 Nvidia cuBLAS 库。Polly 和 Pluto 这类库的搜索技术只能做必要的优化，并且依赖于不准确的机器模型，导致编译器做出次优的决策。其他的 polyhedral 框架，例如 AlphaZ 和 CHiLL，放弃了全自动化，而是暴露了一个调度语言，让用户可以有效地探索可能的变换空间。虽然这些框架取得了更好的性能，但是它们的调度语言并没有针对分布式系统进行设计。例如，它们不允许用户对计算进行分区、跨节点发送数据或插入所需的同步。

上面这段分析了当前的 SOTA 的一些优点与缺陷，展示他们没有解决的问题。接下来一段将介绍 Tiramisu，也就是这篇 paper 的设计，解决了哪些当前 SOTA 没有解决的问题。

> In this paper, we introduce TIRAMISU, a polyhedral compiler with a scheduling language featuring novel commands for targeting multiple high performance architectures. TIRAMISU is well-suited for implementing data parallel algorithms (loop nests manipulating arrays). It takes a high level representation of the program (a pure algorithm and a set of scheduling commands), applies the necessary code transformations, and generates highly-optimized code for the target architecture. In addition to scheduling commands for loop and datalayout transformations, the TIRAMISU scheduling language introduces novel commands for explicit communication and synchronization, and for mapping buffers to different memory hierarchies. In order to simplify the implementation of the scheduling language, TIRAMISU explicitly divides the intermediate representation into four layers designed to hide the complexity and large variety of execution platforms by separating the architecture-independent algorithm from code transformations, data layout, and communication. TIRAMISU targets multicore CPUs, CUDA GPUs, distributed architectures, and FPGA. This paper presents the first three backends while Del Sozzo et al. [14] describe an FPGA backend.

首先说明，Tiramisu 可以用于多种目标硬件架构，可以将算法转化成高性能的代码。随后又再次介绍了下 Tiramisu 的架构，四级 IR。分为 communication, data layout, code transformation, algorithm。Tiramisu 能 codegen 到 CUDA, CPU, FPGA, distributed system 等架构。

> The use of a scheduling language has been shown effective for generating efficient code by multiple compilers including CHiLL, AlphaZ, and Halide [39], [40]. In comparison with Halide in particular, not only does TIRAMISU introduce novel scheduling extensions, TIRAMISU fundamentally differs in that it relies on the expressive polyhedral representation instead of the interval-based representation used by Halide. This allows TIRAMISU to naturally express non-rectangular iteration spaces, to support programs with cyclic data-flow graphs, and to apply any affine transformation (including iteration space skewing), all of which are not naturally expressible in Halide.

随后 Tiramisu 展示了和其他 SOTA 的区别，其实就是踩了一下 CHiLL, AlphaZ, Halide，写论文嘛，就是夸自己的，踩别人的。

最后总结一下这篇 paper 带来的贡献：

> This paper makes the following contributions: 

> • We introduce a polyhedral compiler with a scheduling language that features novel commands for controlling data communication, synchronization, and for mapping to different memory hierarchies. These extensions enable targeting multiple high-performance architectures including multicore CPUs, GPUs, and distributed machines. 

> • We explicitly divide the intermediate representation into four layers to simplify the implementation of the scheduling language. The four-layer IR separates the algorithm from code transformations and data-layout transformations, allowing for portability and simplifying the composition of architecture-specific lowering transformations. 

> • We evaluate TIRAMISU on a set of deep learning and linear algebra kernels and show that TIRAMISU can generate efficient code that outperforms Intel MKL by up to 2.3×. We also evaluate TIRAMISU on a set of image processing benchmarks and show that TIRAMISU matches or outperforms state-of-the-art compilers on different hardware architectures, including multicore CPUs, GPUs, and distributed machines.

分为三点：
- 提出了一个 polyhedral compiler，并且有 novel scheduling language 来控制数据通信，同步，以及映射到不同的 memory hierarchy。
- 将 IR 分为四层，将算法和 code transformation, data layout 分开，方便移植。
- 在 deep learning 和 linear algebra 上，Tiramisu 生成的代码比 Intel MKL 快 2.3 倍。
