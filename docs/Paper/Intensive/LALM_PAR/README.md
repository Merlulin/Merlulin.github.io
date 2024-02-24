# 行人属性识别的轻量级属性定位模型

> 原论文地址：https://arxiv.org/abs/2306.09822v1

## 简述

本文主要目的是轻量化网络模型

目前主流的模型轻量化方式有：低秩张量近似、剪枝、量化和知识蒸馏

> - 剪枝存在的问题：
>
>    非结构化修剪无法在传统GPU上显示运行时加速，而结构化修剪由于NN结构的变化而存在问题。
> - 量化存在的问题：
>
>   量化技术以低于浮点精度的位宽处理转换和存储权重。因此，相关的反向传播变得不可行，并且全局权重结构变得不方便维护;因此，它使得量化模型难以收敛，并且很明显的降低了精度。
> - 知识蒸馏：
> 
>   将知识从大模型转移到小模型的过程，由于大模型比小模型具有更高的知识容量，因此大模型和小模型的预测分布之间往往存在令人惊讶的大差异，即使在小模型有能力完美匹配大模型的情况下

该文章主要采用第一种方式来轻量化模型，使用了CPD技术来获得Light-Wight（LW）层，通过使用稳定的CPD-EPC算法[16]和SVD [8]减少属性定位模型（ALM）[17]的组件。

## 轻量化的相关工作

1. SVD（奇异值分解）应用于全连接层的权重矩阵，实现压缩，并且精度没有显著下降。

2. 基于低秩近似来加速卷积层的技术

3. 基于矢量量化[5]或基于张量训练分解[15]

4. 快速傅立叶变换（FFT）来加速卷积[14]和CPU代码优化[18]

