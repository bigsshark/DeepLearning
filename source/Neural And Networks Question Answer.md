# 什么是神经网络?
感知机 -> s型神经元 -> 大脑模型 -> 神经网络 (一种编程范式)
传统编程是我们将问题分而治之，分为多个小问题，计算机按照固定好的步骤一步步执行。神经网络编程我们不告诉计算机如何解决问题，相反神经网络编程将自己从数据中学得g解决问题的办法。
ps: 感知机是神经网络和Support Vector Machine 的基础

# 什么是深度学习?
进行神经网络编程范式的一些列强有力工具的集合，包括但不限于激活函数，前向反馈，反向传播，梯度下降，正则化，dropout， cnn，rnn ，network in network。

# 如何学习神经网络这一新的编程范式?
以原理为导向，就像学习一门编程语言，重要的是学习语法，数据结构，核心库,学完这些你可能还是只知道一小部分，但是该语言的精髓已经刻在你的心中，剩下的未知的知识只会是水到渠成


# why动手实践和理论结合?
加深理解，可能会很难，但要学会忍耐。
从最简单到不断迭代，投入情感是精通技艺的关键。面对无法理解的问题，忍耐和理解、内化才是解决问题的路径。

# 使用神经网络识别手写数字?


# 感知机、感知机实现与非门(逻辑函数),没什么卵用？y只是换了一个形式的与非门?

可以设计学习算法

# s型神经元，逻辑函数，端点?  


# 神经a网络，前馈神经网络，循环神经网络

# 神经网络到底在干嘛?

# https://www.codelast.com/原创-再谈-最速下降法梯度法steepest-descent/ 






## 改进网络的学习方法

regularization
- 饱和是什么含义
- 权重初始化对神经网络的影响
- 为何regularization 会减轻过度拟合
- 为何不对bias进行规范化
- regularization和过度拟合的关系
- 何为l1regularization
- 何为l2regularization
- 何为dropout
- 何为增加训练样本规范化
- 0处的偏导
- l1和l2的区别
- 规范化最直观的效果是啥
- drop的原理以及为啥drop能工作
- 更多的数据集可以弥补不同算法的差距


weight initialize
- 高斯分布，均值为0，标准差为1的正态分布
- 是否有比高斯分布更好的初始化模型
- https://arxiv.org/pdf/1206.5533v2.pdf 14，15权重初始化

参数选择
- 如果进行参数选择。网络数量，学习率，迭代次数，mini-batch，损失函数，参数初始化方法选择等等。

- 什么是宽泛策略？
- 学习速率如何选择
- 














-------------------------------papper-------------------------------------------------------

- dropout 论文 https://arxiv.org/pdf/1207.0580.pdf  dropout
- dropout https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf   alexNet
- https://ieeexplore.ieee.org/document/1227801/  人为扩展数据集合
- http://portal.acm.org/citation.cfm?doid=1073012.1073017 不同算法在u不同的数据集合此起彼伏
- https://dl.acm.org/citation.cfm?id=2188395 网格搜索
- http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf 贝叶斯选择最优参数
