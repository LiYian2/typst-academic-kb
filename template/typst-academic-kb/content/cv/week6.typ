// @title: 0. 本章核心目标：从线性分类器走向视觉神经网络
// @description: 线性分类器只能学习线性决策边界，无法处理 XOR 等非线性模式；MLP 通过隐藏层和非线性激活函数提升表达能力，但对图像直接全连接会导致参数量巨大；CNN 进一步利用图像的局部性、平移共享和空间结构，大幅减少参数并学习局部视觉特征。**
// @order: 999

= 0. 本章核心目标：从线性分类器走向视觉神经网络

本章要解决的问题是：

_线性分类器只能学习线性决策边界，无法处理 XOR 等非线性模式；MLP 通过隐藏层和非线性激活函数提升表达能力，但对图像直接全连接会导致参数量巨大；CNN 进一步利用图像的局部性、平移共享和空间结构，大幅减少参数并学习局部视觉特征。_

整体进化树如下：

```text
KNN
→ 记忆训练集，预测慢，没有真正学习参数

Linear Classifier
→ 学习 W,b，预测快
→ 但只能表达线性边界

MLP
→ 引入 hidden layer
→ 通过 nonlinear activation 组合多个线性特征
→ 可表达 XOR / 非线性边界

Deep MLP
→ 多层特征抽象
→ 但训练会遇到梯度消失、激活函数选择问题

CNN
→ 针对图像结构设计
→ 局部连接 + 参数共享 + 多 filter
→ 参数更少，保留空间结构

Pooling
→ 降低空间尺寸
→ 提高表示紧凑性和一定平移鲁棒性
```

重要程度标注：
`★★★` 必考核心；`★★` 常考概念；`★` 理解即可。

= 1. 上周回顾：从 KNN 到线性分类器 `★★★`

== 1.1 KNN 的框架

KNN 的训练阶段本质上是：

```text
train(images, labels):
 memorize all data and labels
```

预测阶段：

```text
predict(test_image):
 find most similar training image
 return its label
```

它的核心问题是：训练很便宜，但预测很慢；模型没有学到显式参数，也没有抽象出数据规律。

== 1.2 线性分类器的标准机器学习框架

线性分类器不再直接记忆样本，而是学习一个带参数的函数：

$ f_W(x) $

其中：

$ x in R R^n $

表示输入样本，例如图像展平后的向量；

$ W $

表示模型参数；

$ f_W(x) $

输出每个类别的 score。

训练目标是定义损失函数：

$ L(f_W(x), y) $

然后求最优参数：

$ W^* = op("argmin")_W L(f_W(x), y) $

预测时使用：

$ f_{W^*}(x) $

核心逻辑是：

```text
定义模型 f_W(x)
→ 定义损失 L(f_W(x), y)
→ 优化参数 W*
→ 用 f_{W*}(x) 预测标签
```

= 2. 正则化：控制模型复杂度 `★★★`

课件中给出两类正则化：

$ L_2 " regularization: " R(W)=|W|_2^2=sum_k W_k^2  L_1 " regularization: " R(W)=|W|_1=sum_k |W_k| $

符号解释：

$ W $

是全部模型权重；

$ W_k $

是第 (k) 个权重参数；

$ R(W) $

是正则化项，用于惩罚复杂模型。

完整训练目标通常写成：

$ L_{"total"} = L_{"data"} + lambda R(W) $

其中：

$ L_{"data"} $

是分类损失，例如 cross entropy；

$ lambda $

是正则化强度。

直觉：

`L2` 会惩罚大权重，使模型更平滑；如果某个权重很大，那么输入特征的微小变化都会导致输出巨大变化，因此容易过拟合噪声。

在 SVM 中，最小化：

$ |W|_2^2 $

等价于最大化 margin。margin 越大，分类边界对测试样本扰动越鲁棒。

= 3. Softmax 与 Cross Entropy `★★★`

线性模型输出的是 raw scores，不是概率。假设模型对第 (i) 个样本输出：

$ s = f(x_i, W) $

其中：

$ s_j $

是第 (j) 个类别的 score。

Softmax 把 score 转换为概率：

$ q_j = frac(e^{s_j}, sum_k e^{s_k}) $

其中：

$ q_j >= 0,quad sum_j q_j = 1 $

所以它可以被解释为类别概率。

若真实标签是 one-hot 分布：

$ p = (p_1,p_2,dot(s),p_C) $

其中正确类别 (y_i) 对应：

$ p_{y_i}=1 $

其他类别为 0。

Cross Entropy loss 为：

$ L = -frac(1, N)sum_i sum_j p_{{"ij"}}log q_{{"ij"}} $

由于 $p_{{"ij"}}$ 是 one-hot，只有正确类别保留，所以：

$ L = -frac(1, N)sum_i log frac(e^{s_{y_i}}, sum_j e^{s_j}) $

符号解释：

$ N $

是训练样本数；

$ C $

是类别数；

$ s_j $

是类别 (j) 的 score；

$ s_{y_i} $

是第 (i) 个样本真实类别的 score。

直觉：
正确类别 score 越大，softmax 概率越接近 1，loss 越小；错误类别 score 越大，分母越大，loss 越大。

= 4. Gradient Descent `★★★`

梯度下降更新规则：

$ W <- W - lambda frac(partial L, partial W) $

符号解释：

$ W $

是参数；

$ L $

是损失函数；

$ frac(partial L, partial W) $

是损失对参数的梯度；

$ lambda $

是 learning rate，也叫 step size。

直觉：
梯度方向是损失上升最快的方向，因此减去梯度就是沿损失下降最快方向更新。

= 5. 为什么需要非线性模型：XOR 问题 `★★★`

XOR 的四个点可以理解为：

| 点 | 横向线性特征 | 纵向线性特征 | 标签 |
| - | ------ | ------ | --------- |
| 1 | + | + | green / + |
| 2 | - | + | red / - |
| 3 | + | - | red / - |
| 4 | - | - | green / + |

单条直线无法把绿色点和红色点分开，因此线性分类器失败。

解决方式是：先用两个线性特征分别判断点在两条线的哪一侧，然后组合它们：

$ z_1 = "sign"(w_1^->p x + b_1)  z_2 = "sign"(w_2^->p x + b_2) $

再做非线性组合：

$ y = z_1 z_2 $

当两个符号相同时，输出正；符号不同时，输出负。这正好实现 XOR 结构。

关键结论：

_多个线性函数本身不够；必须加入非线性操作，例如 threshold、activation、product 或逻辑组合。_

= 6. Single Hidden Layer MLP `★★★`

单隐藏层网络结构：

$ x in R R^n  W_1 in R R^{m times n}, quad b_1 in R R^m  h = sigma(W_1x+b_1)  w_2 in R R^m,quad b_2 in R R  o = w_2^->p h + b_2 $

符号解释：

$ x $

是输入向量；

$ n $

是输入维度；

$ m $

是 hidden layer size，是超参数；

$ W_1 $

是输入层到隐藏层的权重矩阵；

$ b_1 $

是隐藏层偏置；

$ h $

是隐藏层输出；

$ sigma $

是逐元素激活函数；

$ w_2 $

是隐藏层到输出层的权重；

$ b_2 $

是输出层偏置；

$ o $

是最终输出 score。

== 为什么必须有非线性激活？

如果没有激活函数：

$ h = W_1x+b_1  o = w_2^->p h+b_2 $

代入得到：

$ o = w_2^->p(W_1x+b_1)+b_2  o = w_2^->p W_1x + w_2^->p b_1+b_2 $

令：

$ W' = w_2^->p W_1  b' = w_2^->p b_1+b_2 $

则：

$ o = W'x+b' $

这仍然是线性模型。

所以：

_线性层叠加线性层，结果仍然是线性层。深度本身不产生非线性，激活函数才产生表达能力。_

= 7. 多分类 MLP `★★★`

对 (d) 类分类问题，输出层改为：

$ W_2 in R R^{m times d}, quad b_2 in R R^d $

更标准的写法是：

$ h = sigma(W_1x+b_1)  o = W_2^->p h + b_2  y = "softmax"(o) $

其中：

$ o in R R^d $

表示 (d) 个类别的 score；

$ y in R R^d $

表示 softmax 后的类别概率。

= 8. Multiple Hidden Layers `★★★`

多层 MLP：

$ h_1 = sigma(W_1x+b_1)  h_2 = sigma(W_2h_1+b_2)  h_3 = sigma(W_3h_2+b_3)  o = W_4h_3+b_4 $

超参数包括：

$ "number of hidden layers"  "hidden size for each layer" $

深层网络的意义是逐层构造特征：

```text
raw input
→ low-level feature
→ mid-level feature
→ high-level feature
→ class score
```

但深层网络训练会带来梯度传播问题，这引出 activation function 的选择。

= 9. Backpropagation `★★★`

课件用 step-by-step 图示展示了反向传播。核心不是记住每张图，而是理解链式法则。

假设：

$ L = L(o)  o = w_2^->p h+b_2  h = sigma(a)  a = W_1x+b_1 $

反向传播从输出层开始：

$ frac(partial L, partial w_2) =============================== frac(partial L, partial o) frac(partial o, partial w_2) $

因为：

$ o = w_2^->p h+b_2 $

所以：

$ frac(partial o, partial w_2)=h $

因此：

$ frac(partial L, partial w_2) =============================== frac(partial L, partial o)h $

对隐藏层：

$ frac(partial L, partial h) ============================= frac(partial L, partial o) frac(partial o, partial h) ============================= frac(partial L, partial o)w_2 $

对 pre-activation：

$ frac(partial L, partial a) ============================= frac(partial L, partial h) op("odot") sigma'(a) $

其中：

$ op("odot") $

表示逐元素乘法。

再对 (W_1)：

$ frac(partial L, partial W_1) =============================== frac(partial L, partial a)x^->p $

反向传播本质：

```text
forward: compute activations and loss
backward: apply chain rule layer by layer
update: W ← W − λ∂L/∂W
```

= 10. 激活函数演进 `★★★`

本章一个重要方法论演进是：

```text
Step Function
→ Sigmoid
→ tanh
→ ReLU
→ Leaky ReLU / PReLU / ELU / Maxout
```

每一步都在解决前一种激活函数的缺陷。

== 10.1 Step Function `★★`

硬阈值函数：

$ sigma(x)= cases(1,, x>0\ 0,, "otherwise") $

问题：不可导或几乎处处梯度为 0，不适合梯度下降训练。

== 10.2 Sigmoid `★★★`

$ "sigmoid"(x)=frac(1, 1+exp(-x)) $

输出范围：

(0,1)

优点：
可以解释为神经元 firing rate，也可用于二分类概率输出。

缺点 1：饱和区杀死梯度。

Sigmoid 导数：

$ sigma'(x)=sigma(x)(1-sigma(x)) $

当：

$ x -> oo $

时：

$ sigma(x)-> 1,quad sigma'(x)-> 0 $

当：

$ x -> -oo $

时：

$ sigma(x)-> 0,quad sigma'(x)-> 0 $

所以深层网络中梯度会逐层变小，导致前面层几乎无法更新。

缺点 2：输出不是 zero-centered。

Sigmoid 输出总是正数：

$ sigma(x)in(0,1) $

若一个神经元输入总为正，局部梯度也为正，那么所有权重梯度符号会受到 upstream gradient 的统一控制。结果是更新方向只能“全部增加”或“全部减少”，无法让某些权重增加、某些权重减少，从而造成 zig-zag optimization path。

== 10.3 tanh `★★★`

$ tanh(x)=frac(1-exp(-2x), 1+exp(-2x)) $

输出范围：

(-1,1)

优点：zero-centered，比 sigmoid 更适合隐藏层训练。

缺点：仍然会饱和。即当：

$ |x|->oo $

时：

$ tanh'(x)-> 0 $

所以仍然存在梯度消失问题。

== 10.4 ReLU `★★★`

$ "ReL U"(x)=max(0,x) $

分段写法：

$ "ReL U"(x)= cases(x,, x>0\ 0,, x<= 0) $

导数：

$ "ReL U"'(x)= cases(1,, x>0\ 0,, x<0) $

优点：

```text
正区间不饱和
计算非常便宜
实践中比 sigmoid/tanh 收敛更快
```

缺点：

```text
输出不是 zero-centered
可能出现 dead neuron
```

Dead ReLU 指如果某个神经元长期落在负区间：

$ x<0 $

则输出为：

0

梯度也为：

0

于是该神经元不再更新，永远不激活。课件提到实践中常用稍微正的 bias 初始化，例如：

$ b=0.01 $

来降低 ReLU 死亡风险。

== 10.5 Leaky ReLU `★★`

Leaky ReLU 解决 ReLU 负区间梯度为 0 的问题：

$ f(x)= cases(x,, x>0\ alpha x,, x<= 0) $

其中：

$ alpha>0 $

通常是小常数，例如 0.01。

优点：负区间也有梯度，因此不容易 die。

== 10.6 PReLU `★★`

Parametric ReLU 把 $alpha$ 作为可学习参数：

$ f(x)= cases(x,, x>0\ alpha x,, x<= 0) $

区别在于：

$ alpha $

通过 backpropagation 学习，而不是人工固定。

== 10.7 ELU `★★`

ELU，即 Exponential Linear Unit：

$ f(x)= cases(x,, x>0\ alpha(exp(x)-1),, x<= 0) $

优点：

```text
保留 ReLU 正区间不饱和的优点
负区间输出更接近 zero mean
负区间饱和可以增加对噪声的鲁棒性
```

缺点：

$ exp(x) $

计算更贵。

== 10.8 Maxout `★`

Maxout 不再是简单的：

$ "dot product" -> "element-wise nonlinearity" $

而是：

$ f(x)=max(w_1^->p x+b_1, w_2^->p x+b_2, dot(s), w_k^->p x+b_k) $

它可以泛化 ReLU 和 Leaky ReLU。

优点：

```text
linear regime
non-saturating
does not die
```

缺点：参数量增加，课件强调会 double number of parameters per neuron。

== 10.9 实践建议 `★★★`

课件结论：

```text
隐藏层中优先使用 ReLU
注意 learning rate
可以尝试 Leaky ReLU / Maxout / ELU 获得边际提升
不要在隐藏层使用 sigmoid 或 tanh
sigmoid/tanh 可以用于输出层控制输出范围
```

= 11. 从 MLP 到 CNN：为什么全连接不适合图像？ `★★★`

图像输入：

$ I in R R^{32times 32times 3} $

展平后：

$ x in R R^{3072} $

线性分类器参数：

$ W in R R^{10times 3072} $

如果堆叠多层 MLP：

$ 3072 -> n_1 -> n_2 -> .. -> 10 $

参数量巨大，而且破坏了图像的空间结构。

核心问题：

```text
图像具有局部结构
相邻像素高度相关
同一个视觉模式可能出现在不同位置
MLP 把图像展平后丢失空间邻接关系
全连接层参数量过大
```

CNN 的改进：

```text
局部连接：每个神经元只看局部 patch
参数共享：同一个 filter 在所有位置复用
多 filter：学习不同视觉模式
保留空间结构：输出仍是 feature map
```

= 12. Convolutional Layer `★★★`

输入图像：

$ 32times 32times 3 $

一个卷积核：

$ 5times 5times 3 $

注意：课件强调：

_Filters always extend the full depth of the input volume._

所以如果输入 depth 是 3，filter depth 也必须是 3。

参数量：

$ 5times 5times 3 = 75 $

若包含 bias：

$ 5times 5times 3+1=76 $

卷积操作是在空间位置上滑动 filter，对每个局部区域做 dot product：

$ a_{i,j} = sum_{u=1}^{F}sum_{v=1}^{F}sum_{c=1}^{C} K_{u,v,c}X_{i+u,j+v,c}+b $

符号解释：

$ X $

是输入图像或输入 volume；

$ K $

是卷积核；

$ F $

是 filter spatial size；

$ C $

是输入 depth/channel 数；

$ b $

是 bias；

$ a_{i,j} $

是输出 activation map 在位置 ((i,j)) 的值。

一个 filter 会产生一个 activation map。
如果有 6 个 filters，则产生 6 张 activation maps，并 stack 成：

$ 28times 28times 6 $

对于：

$ 32times 32times 3 $

输入，使用：

$ 5times 5times 3 $

filter，stride 1，无 padding，输出空间尺寸为：

$ 28times 28 $

因为 filter 可以放置的位置数量是：

32-5+1=28

= 13. CNN 的层级结构 `★★★`

ConvNet 是多个卷积层和激活函数交替组成：

```text
Input: 32×32×3
→ CONV, ReLU: 6 filters of 5×5×3
→ 28×28×6
→ CONV, ReLU: 10 filters of 5×5×6
→ 24×24×10
→ ...
```

第二层 filter depth 必须等于上一层输出 depth：

$ 5times 5times 6 $

如果有 10 个这样的 filters，输出 depth 是 10。

= 14. 卷积输出尺寸公式 `★★★`

无 padding 时：

$ "Output size" = frac(N-F, S)+1 $

其中：

$ N $

是输入空间尺寸；

$ F $

是 filter size；

$ S $

是 stride。

例如：

$ N=7,quad F=3 $

stride 1：

$ frac(7-3, 1)+1=5 $

输出：

$ 5times 5 $

stride 2：

$ frac(7-3, 2)+1=3 $

输出：

$ 3times 3 $

stride 3：

$ frac(7-3, 3)+1=2.33 $

不是整数，因此 filter 放不下，不能合法卷积。

= 15. Zero Padding `★★★`

为了防止空间尺寸快速缩小，CNN 常在边界补 0。

有 padding 时输出尺寸为：

$ "Output size" = frac(N+2P-F, S)+1 $

其中：

$ P $

是 padding size。

例子：

$ N=7,quad F=3,quad S=1,quad P=1 $

则：

$ frac(7+2(1)-3, 1)+1=7 $

输出仍为：

$ 7times 7 $

常见做法：当 stride 为 1，filter size 为 $F times F$，为了保持空间尺寸不变，取：

$ P=frac(F-1, 2) $

例如：

$ F=3 => P=1  F=5 => P=2  F=7 => P=3 $

注意：这要求 (F) 通常为奇数。

为什么 padding 重要？
如果不 padding，连续使用 $5times 5$ 卷积会导致：

$ 32-> 28-> 24-> .. $

空间尺寸缩小太快，不利于深层特征学习。

= 16. 卷积层参数量计算 `★★★`

例题：

输入：

$ 32times 32times 3 $

使用 10 个：

$ 5times 5 $

filters，stride 1，pad 2。

输出空间尺寸：

$ frac(32+2(2)-5, 1)+1=32 $

输出 volume：

$ 32times 32times 10 $

每个 filter 参数量：

$ 5times 5times 3+1=76 $

其中 (+1) 是 bias。

10 个 filters 总参数量：

$ 76times 10=760 $

这体现 CNN 的参数共享优势：输出有 $32times 32times 10$ 个神经元，但参数只有 760 个。

= 17. Receptive Field `★★★`

课件定义：

一个 activation map 是一张 neuron output sheet。每个 neuron：

```text
只连接输入中的一个小区域
所有空间位置共享同一个 filter 参数
```

如果 filter 是：

$ 5times 5 $

则每个输出 neuron 的 receptive field 是：

$ 5times 5 $

直觉：
receptive field 表示某个输出单元“看见”的输入区域。CNN 越深，后层 neuron 的有效 receptive field 越大，可以从局部边缘逐渐组合成更高级视觉模式。

= 18. Pooling Layer `★★★`

Pooling 的作用：

```text
减小 representation size
让特征更 manageable
对每个 activation map 独立操作
```

== Max Pooling

课件例子：

输入单个 depth slice：

$ mat(1, 2, 2, 4\ 5, 6, 7, 8\ 3, 2, 1, 0\ 1, 2, 3, 4) $

使用：

$ 2times 2 $

filter，stride 2。

四个区域分别取最大值：

左上：

$ max mat(1, 2\ 5, 6) =6 $

右上：

$ max mat(2, 4\ 7, 8) =8 $

左下：

$ max mat(3, 2\ 1, 2) =3 $

右下：

$ max mat(1, 0\ 3, 4) =4 $

输出：

$ mat(6, 8\ 3, 4) $

Pooling 不改变 depth，只缩小 height 和 width。

= 19. 算法/模型对比表 `★★★`

| 方法 | 核心思想 | 优点 | 局限 | 为什么引出下一步 |
| ----------------- | ------------------- | ----------- | -------------------------- | -------------------- |
| KNN | 记忆训练样本，找最近邻 | 简单，无训练成本 | 预测慢，不学习抽象规律 | 需要参数化模型 |
| Linear Classifier | 学习 (W,b)，输出类别 score | 预测快，可优化 | 只能线性分割 | XOR 等非线性问题失败 |
| MLP | 隐藏层 + 非线性激活 | 可表达非线性边界 | 图像全连接参数巨大 | 需要利用图像结构 |
| Deep MLP | 多层抽象特征 | 表达能力更强 | 梯度消失、训练困难 | 需要更好的激活函数 |
| CNN | 局部连接 + 参数共享 | 参数少，保留空间结构 | 需要设计 filter/stride/padding | 成为视觉任务基础结构 |
| Pooling | 下采样 feature map | 降低计算量，提高鲁棒性 | 丢失部分空间精度 | 与 CNN 组合形成经典 ConvNet |

= 20. 期末高频考点总结

最可能考计算题：

```text
softmax + cross entropy
gradient descent update
MLP 前向传播维度
为什么没有 activation 时多层线性仍是线性
卷积输出尺寸
padding 后输出尺寸
卷积层参数量
max pooling 结果
```

最可能考概念题：

```text
为什么线性分类器不能解决 XOR
为什么需要非线性激活函数
sigmoid 的两个问题：梯度消失、非 zero-centered
tanh 相比 sigmoid 的改进与不足
ReLU 的优点与 dead neuron 问题
Leaky ReLU / PReLU / ELU / Maxout 的动机
CNN 为什么比 MLP 更适合图像
filter 为什么必须覆盖 full depth
receptive field 的意义
pooling 的作用
```

最核心的一句话：

_MLP 通过非线性激活突破线性分类器的表达瓶颈；CNN 通过局部连接和参数共享突破 MLP 在图像上的参数与结构瓶颈。_
