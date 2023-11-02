# 行人属性识别

## 基于解耦表征学习的行人属性识别

> Learning Disentangled Attribute Representations for Robust Pedestrian Attribute Recognition

研究现状：

    现有的行人属性识别方法，通常采用学习一个共享的行人图像特征来对多个属性进行分类。
    但是这种机制会导致在模型推理阶段的鲁棒性和置信度降低。
    现有的方法可分为如下几类：
    1. 提取一个共享的全局特征来对所有属性进行分类
        (HydraPlusNet、MsVAA、VAC、JLAC)
    2. 根据属性的空间分布将属性分成若干组，采用一组特征对同一组中的多个属性进行分类
        （RC&RA、VSGR）
    3. 试图为每个属性提取一个特定的特征

采用共享的全局特征为什么存在问题？
>[!note]
我们假设对于第j个属性，第i个样本，该样本在经过最后的分类层之后输出结果设为$logits_{i,j}$，此时第i个样本预测为第j个属性的概率为$p_{i,j}=\sigma(logits_{i,j})$，$\sigma$就是Sigmoid函数。我们默认设定预测的阈值$p_t=0.5$，则有$\hat{y}_{i, j}=\left\{\begin{array}{ll}
1, & p_{i, j}>=p_{t} \\
0, & p_{i, j}<p_{t}
\end{array},\right.$
>
>继续假定$logits_{i,j}=w^T_jf_i=|w_j|*|f_i|·cos\theta$，其中$w_j$表示第j个属性的权重向量，$f_i$则是共享的$x_i$样本特征向量。<br> 将上述式子进行整合，可以得到$\hat{y}_{i, j}=\left\{\begin{array}{ll}
1, & 0^{\circ}<=\theta<=90^{\circ} \\
0, & 90^{\circ}<\theta<180^{\circ}
\end{array} \right.$ <br>此时可以看出我们最后的预测结果只和$cos\theta$相关，而$\theta$就是特征向量$f$和属性分类权重向量$w$的夹角。<br> 因此，对于一个目标属性，一个经过良好训练的模型应该使正样本特征与对应的分类器权重之间的角度尽可能小，甚至接近0 °，这意味着高置信度的预测。