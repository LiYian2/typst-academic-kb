// @title: 表示学习
// @description: 表示学习关注如何把原始对象 $x$ 映射到适合下游任务的向量空间 $z = f_theta(x)$。
// @order: 999

= 表示学习

表示学习关注如何把原始对象 $x$ 映射到适合下游任务的向量空间 $z = f_theta(x)$。

== 核心问题

- 表示是否保留任务相关信息？
- 表示是否抛弃噪声和无关变化？
- 表示空间的距离是否有语义意义？

== 与其他主题的关系

表示学习连接 [[nlp/transformer-attention|注意力机制]]、[[cv/vision-transformer|视觉 Transformer]] 和 [[math/linear-algebra|线性代数]]。

== 对比学习目标

一个常见目标是让正样本更接近，负样本更远：

$ cal(L)_i =
-log
frac(exp(op("sim")(z_i, z_i^+) / tau), sum_{j=1}^{N}exp(op("sim")(z_i, z_j) / tau)) $

其中 $tau$ 是温度系数。
