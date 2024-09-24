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

## Related Work

按照正常的论文进度，这里应该讲 Background and Motivation，这节命名为 Related Work，我理解为背景。

分为了几个部分：
- Polyhedral compilers with automatic scheduling
- Polyhedral compilers with a scheduling language
- Non-polyhedral compilers with a scheduling language
- Other Compilers

> a) Polyhedral compilers with automatic scheduling: Polyhedral compilers such as PENCIL [4], [3], Pluto [8], Polly [22], Tensor Comprehensions [46], and PolyMage [34] are fully automatic. Some of them are designed for specific domains (such as Tensor Comprehensions and PolyMage), while Pluto, PENCIL, and Polly are more general. While fully automatic compilers provide productivity, they may not always obtain the best performance. This suboptimal performance is due to several reasons: first, these compilers do not implement some key optimizations such as array packing [20], register blocking, data prefetching, and asynchronous communication (which are all supported by TIRAMISU); second, they do not have a precise cost-model to decide which optimizations are profitable. For example, the Pluto [8] automatic scheduling algorithm (used in Pluto, PENCIL and Polly) tries to minimize the distance between producer and consumer statements while maximizing outermost parallelism, but it does not consider data layout, redundant computations, or the complexity of the control of the generated code. Instead of fully automatic scheduling, TIRAMISU relies on a set of scheduling commands, giving the user full control over scheduling.

> Polyhedral frameworks proposed by Amarasinghe et al. [1] and Bondhugula et al. [7] address the problem of automatic code generation for distributed systems. Instead of being fully automatic, TIRAMISU relies on the user to provide scheduling commands to control choices in the generated code (synchronous/asynchronous communication, the granularity of communication, buffer sizes, when to send and receive, cost of communication versus re-computation, etc.).

首先介绍了一下当前自动调度的 polyhedral compilers，随后指出了它们存在的一些问题：
- 没有实现 array packing, register blocking, data prefetching, asynchronous communication 这些优化。
- 没有精确的 cost model 来决定哪些优化是 profitable 的。

随后讲了下 Tiramisu 和它们的区别，Tiramisu 提供了 scheduling commands 来控制生成的代码。

> b) Polyhedral compilers with a scheduling language: AlphaZ [51], CHiLL [10], [24] and URUK [19] are polyhedral frameworks developed to allow users to express high-level transformations using scheduling commands. Since these frameworks are polyhedral, they can express any affine transformation. However, their scheduling languages do not target distributed architectures. In contrast, TIRAMISU features scheduling commands for partitioning computations (for distributed systems), synchronization and distribution of data across nodes. The first four columns of Table I compare between TIRAMISU and three representative polyhedral frameworks.

第二种是 Polyhedral compilers with a scheduling language。这种编译器可以表示任何仿射变换，但它们无法 codegen 到不同的硬件架构。在 Tiramisu 中则可以。

> c) Non-polyhedral compilers with a scheduling language: Halide [39] is an image processing DSL with a scheduling language that uses intervals to represent iteration spaces instead of the polyhedral model. This limits the expressiveness of Halide. For example, unlike TIRAMISU, Halide cannot naturally represent non-rectangular iteration spaces, and this is the reason why distributed Halide [15] over-approximates the amount of data to communicate (send and receive) when generating distributed code. This also makes some Halide passes overapproximate non-rectangular iteration spaces, potentially leading to less efficient code (for example, it prevents Halide from performing precise bounds inference for non-rectangular iteration spaces). The use of intervals also prevents Halide from performing many complex affine transformations, such as iteration space skewing.

> Halide does not have dependence analysis and thus relies on conservative rules to determine whether a schedule is legal. For example, Halide does not allow the fusion of two loops (using the compute_with command) if the second loop reads a value produced by the first loop. While this rule avoids illegal fusion, it prevents fusing many legal cases, which may lead to suboptimal performance. Halide also assumes the program has an acyclic dataflow graph in order to simplify checking the legality of a schedule. This prevents users from expressing many programs with cyclic dataflow. It is possible in some cases to work around the above restrictions, but such work-around methods are not general. TIRAMISU avoids overconservative constraints by relying on dependence analysis to check for the correctness of code transformations, enabling more possible schedules. Table I summarizes the comparison between TIRAMISU and Halide.

> Vocke et al. [48] extend Halide to target DSPs, and add scheduling commands such as store_in to specify in which memory hierarchy data should be stored. TVM [11] is another system that shares many similarities with Halide. It uses a modified form of the Halide IR internally. Since TVM is also a non-polyhedral compiler, the differences between Halide and TIRAMISU that are due to the use of polyhedral model also apply to TVM.

> POET [50] is a system that uses an XML-based description of code and transformation behavior to parametrize loop transformations. It uses syntactic transformations, which are less general than the polyhedral transformations used in TIRAMISU. GraphIt [52] is another compiler that has a scheduling language but that is mainly designed for the area of graph applications.

这里主要介绍了下 Halide，Halide 是使用 intervals 而不是多面体模型来表示迭代空间的，这将会限制 Halide 的表达能力。Halide 不能表示非矩阵迭代空间，因此在分布式中 Halide 会对 send/receive 的数据量进行过度估计。除此之外，Halide 没有依赖分析，不允许 fuse 两个 loop。这是由于 DAG 阻止用户表达许多程序。

> Other Compilers: Delite [9] is a generic framework for building DSL compilers. It exposes several parallel computation patterns that DSLs can use to express parallelism. NOVA [12] and Lift [42] are IRs for DSL compilers. They are functional languages that rely on a suite of higher-order functions such as map, reduce, and scan to express parallelism. TIRAMISU is complementary to these frameworks as TIRAMISU allows complex affine transformations that are easier to express in the polyhedral model.

解释了一下其他编译器的设计，主要介绍了下 Delite，NOVA，Lift。依赖于 functional language，例如 map，reduce，scan 来表达并行。Tiramisu 和这些框架是互补的，Tiramisu 允许更复杂的仿射变换，这些变换在多面体模型中更容易表达。

## The Tiramosu Embedded DSL

第三章，其实就是在讲 Design。

> TIRAMISU is a domain-specific language (DSL) embedded in C++. It provides a C++ API that allows users to write a high level, architecture-independent algorithm and a set of scheduling commands that guide code generation. Input TIRAMISU code can either be written directly by a programmer, or generated by a different DSL compiler. TIRAMISU then constructs a high level intermediate representation (IR), applies the user-specified loop and data-layout transformations, and generates optimized backend code that takes advantage of target hardware features (LLVM IR for multicores and distributed machines and LLVM IR + CUDA for GPUs).

首先介绍 Tiramisu DSL 的一个简短的流程。

接下来分为几个小节去讲 Tiramisu 的设计，分别为：

- Scope of Tiramisu
- Specifying the Algorithm
- Scgeduling Commands

> A. Scope of TIRAMISU <br> TIRAMISU is designed for expressing data parallel algorithms, especially those that operate over dense arrays using loop nests and sequences of statements. These algorithms are often found in the areas of image processing, deep learning, dense linear algebra, tensor operations and stencil computations.

第一节解释 Tiramisu 整体的 scope，主要用于在数据并行算法中。

> B. Specifying the Algorithm <br> The first part of a TIRAMISU program specifies the algorithm without specifying loop optimizations (when and where the computations occur), data layout (how data should be stored in memory), or communication. At this level there is no notion of data location; rather, values are communicated via explicit producer-consumer relationships.

> The algorithm is a pure function that has inputs, outputs, and is composed of a sequence of computations. A computation is used to represent a statement in TIRAMISU. Flow-control around computations is restricted to for loops and conditionals. While loops, early exits, and GOTOs cannot be expressed. To declare a computation, the user provides both the iteration domain of the computation and the expression to compute.

> Figure 2 shows a blur algorithm written in TIRAMISU. This algorithm declares two computations, bx and by. The first computation, bx, computes a horizontal blur of the input, while the second computation, by, computes the final blur by averaging the output of the first stage. The iterators i, j, and c in line 2 define the iteration domain of bx and by (for brevity we ignore boundary conditions). The algorithm is semantically equivalent to the following code.

> The algorithm is a pure function that has inputs, outputs, and is composed of a sequence of computations. A computation is used to represent a statement in TIRAMISU. Flow-control around computations is restricted to for loops and conditionals. While loops, early exits, and GOTOs cannot be expressed. To declare a computation, the user provides both the iteration domain of the computation and the expression to compute.

> Figure 2 shows a blur algorithm written in TIRAMISU. This algorithm declares two computations, bx and by. The first computation, bx, computes a horizontal blur of the input, while the second computation, by, computes the final blur by averaging the output of the first stage. The iterators i, j, and c in line 2 define the iteration domain of bx and by (for brevity we ignore boundary conditions). The algorithm is semantically equivalent to the following code.

这一节主要描述 Tiramisu 如何表达算法。控制流被限制在 for 循环和条件语句中。

```
for (i in 0..N-2) 
    for (j in 0..M-2) 
        for (c in 0..3) 
            bx[i][j][c] = (in[i][j][c]+in[i][j+1][c]+in[i][j+2][c])
for (i in 0..N-2) 
    for (j in 0..M-2) 
        for (c in 0..3) 
            by[i][j][c] = (bx[i][j][c]+bx[i+1][j][c]+bx[i+2][j][c])/3
```

> C. Scheduling Commands <br> TIRAMISU provides a set of high-level scheduling commands for common optimizations; Table II shows some examples. There are four types of scheduling commands:

> • Commands for loop nest transformations: these commands include common affine transformations such as loop tiling, splitting, shifting, etc. For example, applying 32×32 loop tiling to a computation C can be done by calling C.tile(i,j,32,32,i0,j0,i1,j1) where i and j are the original loop iterators and i0, j0, i1, and j1 are the names of the loop iterators after tiling.

> • Commands for mapping loop levels to hardware: examples of these include loop parallelization, vectorization, and mapping loop levels to GPU block or thread dimensions. For example, calling C.vectorize(j, 4) splits the j loop by a factor of 4 and maps the inner loop to vector lanes.

> • Commands for manipulating data: these include (1) allocating arrays; (2) setting array properties including whether the array is stored in host, device, shared, or local memory (GPU); (3) copying data (between levels of memory hierarchies or between nodes); and (4) setting array accesses. In most cases, users need only to use high level commands for data manipulation. If the high level commands are not expressive enough, the user can use the more expressive low level commands.

> • Commands for adding synchronization operations: the user can either declare a barrier or use the send and receive functions for point-to-point synchronization.

Tiramisu 提供了一些高级的调度命令，用于常见的优化。

- 嵌套循环变换/仿射变换：loop tilling, splitting, shifting 等等。
- 将循环映射到硬件上，包括循环并行化，向量化，映射到 GPU block 或 thread 维度。
- 数据操作：分配数组，设置数组属性，包括是否存储在 host，device，shared，local memory 中。
- 同步操作：声明 barrier 或使用 send 和 receive 函数进行点对点同步。

## The Tiramisu IR

这章主要解释 Tiramisu 的 IR 部分。

> The main goal of TIRAMISU’s multi-layer intermediate representation is to simplify the implementation of scheduling commands by applying them in a specific order. This section illustrates why, and describes the layers of the TIRAMISU IR.

Tiramisu 多层 IR 的目的在于简化调度命令的实现，通过应用特定的顺序。

同样分为几部分来说明：

- Rational for Multi-layer IR
- Background
- The Multi-Layer IR

> A. Rationale for a Multi-layer IR <br> In this section we provide examples showing why current intermediate representations are not adequate for TIRAMISU and why we need a multi-layer IR.

> Most current intermediate representations use memory to communicate between program statements. This creates memory-based dependencies in the program, and forces compilers to choose data layout before deciding on optimizations and mapping to hardware. Optimizing a program for different hardware architectures usually requires modifying the data layout and eliminating memory-based dependencies since they restrict optimizations [31]. Thus, any data layout specified before scheduling must be undone to allow more freedom for scheduling, and the code must be adapted to use the datalayout best-suited for the target hardware. Applying these datalayout transformations and the elimination of memory-based dependencies is challenging [23], [45], [30], [17], [33], [32], [29], [38], [13].

> Another example that demonstrates the complexity of code generation is mapping buffers to shared and local memory on GPU. The amount of data that needs to be copied to shared memory and when to perform synchronization both depend on how the code is optimized (for example, whether the code has two-level tiling or not). The same applies to deciding the amount of data to send or receive when generating distributed code. Therefore, buffer mapping to memory hierarchies, communication management, and synchronization should not occur before scheduling.

> TIRAMISU addresses these complexities in code generation by using a multi-layer IR that fully separates the architectureindependent algorithm from loop transformations, data layout and communication. The first layer representation describes the pure algorithm using producer-consumer relationships without memory locations. The second layer specifies the order of computation, along with which processor computes each value; this layer is suitable for performing a vast number of optimizations without dealing with concrete memory layouts. The third layer specifies where to store intermediate data before they are consumed. The fourth layer adds all the necessary communication and synchronization operations.

> The separation of layers defines a specific order for applying optimizations and ensures that compiler passes in a given layer need not to worry about modifying or undoing a decision made in an earlier layer. For example, the phase that specifies the order of computations and where they occur can safely assume that no data-layout transformations are required. This simple assumption allows TIRAMISU to avoid the need to rely on a large body of research that focuses on data-layout transformations to allow scheduling [23], [45], [30], [17], [33], [32], [29], [38], [13].

上面一大段都是在说明当前的多层 IR 的缺点，以及 Tiramisu 的 IR 的优点。

> B. Background

> In this section, we provide an overview of two main concepts used in the polyhedral model: integer sets and maps. These two concepts will be used in later sections to define the different IR layers.

> Integer sets represent iteration domains while maps are used to represent memory accesses and to transform iteration domains and memory accesses (apply loop nest and memory access transformations). More details and formal definitions for these concepts are provided in [47], [2], [36]. 

> An integer set is a set of integer tuples described using affine constraints. An example of a set of integer tuples is <br> {(1, 1); (2, 1); (3, 1); (1, 2); (2, 2); (3, 2)} <br> Instead of listing all the tuples as we do in the previous set, we can describe the set using affine constraints over loop iterators and symbolic constants as follows: <br> {S(i, j) : 1 ≤ i ≤ 3 ∧ 1 ≤ j ≤ 2} <br> where i and j are the dimensions of the tuples in the set.

> A map is a relation between two integer sets. For example <br> {S1(i, j) → S2(i + 2, j + 2) : 1 ≤ i ≤ 3 ∧ 1 ≤ j ≤ 2} <br> is a map between tuples in the set S1 and tuples in the set S2 (e.g. the tuple S1(i, j) maps to the tuple S2(i + 2, j + 2) ).

> All sets and maps in TIRAMISU are implemented using the Integer Set Library (ISL) [47]. We also use the ISL library notation for sets and maps throughout the paper.

这段主要是描述了一下整数点集的背景，是为了用 ISL 那套 polyhedral 工具做的背景介绍。