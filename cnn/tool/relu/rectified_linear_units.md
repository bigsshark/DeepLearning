# Deep Sparse Rectifier Neural Networks

线性整流单元对比tanhs在生物神经元上是更好的模型，也有更好的表现。虽然rectifier neural在0出是hard-nonlinear 和 不可微分的，但是在0处创造了更接近于自然的稀释表征。 虽然tanh可以利用额外的un-labled数据进行半监督设置，但是relu同样可以达到这样的效果而不需要非监督的预训练。 这样的结果可以看做是理解纯粹的监督学习难度的里程碑，同时缩小了有非监督学习预处理和无之间的差距。


## introduction

计算神经科学模型和机器学习神经网络模型
- 计算神经科学模型和机器学习神经网络模型目的的不同以及相关差异性
- 两者之间的桥接relu
- relu在深度神经网络中对训练的影响(3 or more)

灵长类生物的启发
- algorithm 的重要性
- 边缘检测、原始形状、向上更加复杂的形状
- 网络的前几层很像 v1和v2视觉层，并且越来越不可变在高维。

深度网络的训练
- 为什么无监督预训练有助于训练
- 原始数据导致训练失败
- 本篇paper将会从机器学习的角度给出答案

寻找tanh和slogistics的替代品
- relu
- 受限玻尔兹曼机器条件影响下的训练
- denoising auto- encoders  去噪自动编码器预训练
- l1 促进稀疏，存在无限激活的潜在数值问题
- 文本分析，图像分类在relu和tanh下的分析

relu
- relu的稀疏性
- relu的线性性在神经网络训练表现得更好
- relu激活在有无无监督预训练情况下的差距
- 桥接  神经科学/神经网络模型

文章结构
- 激发工作的神经科学和机器学习背景
- 神经元对架构的益处和缺点
- 图像分类和文本上的分析
- 结论



## background

生物神经元和神经网络神经元
- 生物神经元稀疏性的激活1-5%，神经网络神经元sigmoid为0.5，hurt gradient -based optimization
- 神经元计划函数的差异

稀疏性的好处
- 鲁棒性，去掉无关噪声
- 有效的可变表征， 不同的输入对应不同数量的激活神经元(可控的)
- 线性可分， 稀疏表示自动映射到高维空间
- 分布但是稀疏


## Deep Rectifier Networks

1 rectifier neurons
- 皮质神经元可以通过rectifier来模拟
- 单侧，抑制性。可以通过组合共享参数的整流线性和非线性来获得对称性和非对称性

1.1 Advantages
- 稀疏性除了生物上的合理也在数学上有有是。
- 非线性来自于部分激活，分布来自于非线性激活。一旦去激活神经元确定，那么输出就是输入的线性函数。
- 指数级的参数共享,线性性计算更加容易，成本更低，梯度流动良好(没有梯度消失)

1.2 potential problems
- 无线激活 ， 使用l1增加稀疏性
- 在0出的硬饱和也许会伤害反向梯度下降传播的x优化
- 对称行为的数据，需要两倍于神经元的rectifier的数量

2 unsupervised pre-training


## Experimental Study

- train: for tuning parameters 
- valid: tuning hyper-parameters
- test: for report- ing generalization performance


##  Conclusion

- 存在问题: 0点梯度，参数的不良调节
- 稀疏性神经元
- 稀疏度50-80%, 大脑是95-99%
- 在文本方面可能有强大的潜力(文本天然稀疏)
- 生物h神经和神经网络
- 预训练和为未训练之间的差距
