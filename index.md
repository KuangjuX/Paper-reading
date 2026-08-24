---
layout: home
title: Paper Reading
titleTemplate: false

hero:
  name: Paper Reading
  text: 从论文结论，走到算法与实现
  tagline: 一份面向 AI 系统工程的结构化论文知识库，覆盖 LLM 训练与推理、GPU Kernel、编译器、Agent 系统和虚拟化。
  actions:
    - theme: brand
      text: 按学习路径阅读
      link: /notes/llm/learning-path
    - theme: alt
      text: 浏览完整论文索引
      link: /README

features:
  - icon: ∑
    title: LLM 系统
    details: 从知识蒸馏、稀疏注意力到 MoE 训练与动态 Megakernel，记录公式、数据流和工程取舍。
    link: /notes/llm/minimax-msa/msa
  - icon: λ
    title: LLM × GPU Kernel
    details: 关注智能体如何理解、生成并优化高性能 Kernel，以及编译器与 Agent 的协同方式。
    link: /notes/llm-for-kernel/avo
  - icon: A
    title: Agent 系统
    details: 追踪技能表示、运行时和跨模型迁移等 Agent 基础设施问题。
    link: /notes/agent/skvm/skvm
  - icon: C
    title: 编译器
    details: 覆盖多面体编译、层次化数据流和面向专用加速器的程序变换。
    link: /notes/compiler/hida/hida
  - icon: V
    title: 虚拟化与安全
    details: 讨论用户态虚拟化、Unikernel 克隆与 GPU 程序静态验证。
    link: /notes/hypervisor/duvisor/duvisor
  - icon: 260
    title: 论文全景索引
    details: 按主题维护已读与待读论文、会议、原文链接及对应笔记入口。
    link: /README
---

## 不只是摘要

这里的笔记尽量回答三个层次的问题：论文真正解决了什么、核心机制为什么成立、以及它最终如何落到代码、Kernel 或系统执行路径上。长文中的行内公式、独立公式、表格、代码与原论文配图都会在网页中原生呈现。

## 推荐入口

如果你希望系统补齐现代 LLM 的算法背景，请从[推荐阅读路径](/notes/llm/learning-path)开始；如果你正在追踪具体实现，可以直接进入 [MiniMax Sparse Attention](/notes/llm/minimax-msa/msa)、[Event Tensor](/notes/llm/event-tensor/event-tensor) 或 [PithTrain](/notes/llm/pithtrain/pithtrain)。左侧目录会自动收录仓库中的全部笔记，顶部搜索可以检索正文与标题。
