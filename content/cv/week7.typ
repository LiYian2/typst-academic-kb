// @title: Week 7：Optimization of CNNs 学习笔记
// @description: 
// @order: 999

= Week 7：Optimization of CNNs 学习笔记

== 0. 本章核心目标：如何让 CNN “训得动、训得深、泛化好”

本章不是单纯讲 CNN 结构，而是在回答一个核心问题：

_给定一个深层 CNN，如何保证它能够被有效优化，并且在新数据上表现良好？_

可以把本章看成一棵“优化进化树”：

```text
神经网络难训练
│
├── 激活函数问题：Sigmoid 饱和、梯度消失
│ ├── ReLU / Leaky ReLU / ELU / Maxout
│ └── Batch Normalization
│
├── 架构问题：CNN 越深越难优化
│ ├── AlexNet：CNN 打败传统视觉特征
│ ├── VGG：用小卷积核堆深度
│ └── ResNet：用残差连接解决 degradation problem
│
├── 优化器问题：SGD 慢、抖、被 saddle point 卡住、梯度噪声大
│ ├── Momentum
│ ├── AdaGrad
│ ├── RMSProp
│ └── Adam
│
├── 泛化问题：训练误差低不等于测试误差低
│ ├── Early Stopping
│ ├── Ensembles
│ ├── L1/L2/Elastic Net
│ ├── Dropout
│ └── Data Augmentation
│
└── 数据不足问题
 └── Transfer Learning / Fine-tuning
```

重要程度标注：
_★★★★★ 必考核心_：BatchNorm, ResNet, Momentum/AdaGrad/RMSProp/Adam, Dropout, Data Augmentation, Transfer Learning。
_★★★★☆ 高概率考_：Sigmoid 饱和、VGG 小卷积核、degradation problem、SGD 三大问题。
_★★★☆☆ 理解即可_：AlexNet/ZFNet/SENet 作为历史脉络，AutoAugment，ensemble。

---

= 1. 从单隐藏层网络到深层 CNN：为什么优化是核心问题

== 1.1 单隐藏层网络

课件先回顾单隐藏层网络：

$ x in R R^n  W_1 in R R^(m times n), quad b_1 in R R^m  h = sigma(W_1x + b_1)  w_2 in R R^m, quad b_2 in R R  o = w_2^T h + b_2 $

其中：

- $x$：输入向量，维度为 $n$。
- $W_1$：第一层权重矩阵，把 $n$ 维输入映射到 $m$ 个隐藏单元。
- $b_1$：隐藏层偏置。
- $h$：隐藏层表示。
- $ sigma(dot) $ ：逐元素激活函数。
- $w_2$：输出层权重向量。
- $b_2$：输出层偏置。
- $o$：最终输出 scalar。

逻辑直觉：
线性变换 $W_1x+b_1$ 本身无法表示复杂非线性边界，必须引入 $sigma$。X O R 例子说明了单层线性分类器无法处理非线性可分问题，而隐藏层通过组合多个线性边界可以表达 X O R 这种模式。

---

= 2. 激活函数演进：从 Sigmoid 到非饱和激活

== 2.1 Sigmoid $
sigma(x) = frac(1, 1 + e^(-x))
$ 导数为： $
frac(d sigma(x), d x) = sigma(x)(1-sigma(x))
$ 符号解释：

- $x$：神经元输入 pre-activation。
- $sigma(x)$：激活输出，范围为 $(0,1)$。
- $frac(d sigma(x), d x)$：反向传播时局部梯度。

Sigmoid 的问题：

当 $x >> 0$ 时： $
sigma(x) approx 1, quad frac(d sigma(x), d x) approx 1(1-1)=0
$ 当 $x << 0$ 时： $
sigma(x) approx 0, quad frac(d sigma(x), d x) approx 0(1-0)=0
$ 所以 Sigmoid 在两端饱和区域梯度接近 0。反向传播时： $
frac(partial L, partial x)

frac(partial L, partial sigma)
frac(partial sigma, partial x)
$ 如果 $frac(partial sigma, partial x)approx 0$，那么 $frac(partial L, partial x)approx 0$。这就是 saturated neurons “kill gradients”。

第二个问题是 Sigmoid 输出非零中心： $
sigma(x) in (0,1)
$ 输出总为正，会导致后续层的梯度方向高度相关，优化路径 zig-zag，降低收敛效率。

重要程度：★★★★★

---

== 2.2 tanh $
tanh(x)
$ tanh 输出范围为： $
(-1,1)
$ 它比 Sigmoid 好的一点是 zero-centered，但仍然存在饱和区，因此仍可能梯度消失。

重要程度：★★★☆☆

---

== 2.3 ReL U $
upright("ReLU")(x)=max(0,x)
$ 导数： $
frac(d, d x)upright("ReLU")(x)

cases(1,, x>0 \
0,, x<0)
$ ReL U 的核心优势：正区间不饱和，梯度为 1，因此能缓解深层网络中的梯度消失。

问题：
如果一个神经元长期处于 $x<0$，梯度为 0，可能变成 dead ReL U。

重要程度：★★★★★

---

== 2.4 Leaky ReL U

课件公式： $ upright("LeakyReLU")(x)=max(0.1x,x) $
更一般写法： $ upright("LeakyReLU")(x)=max(alpha x, x) $ 其中 $alpha$ 是负半轴斜率，例如 $0.01$ 或 $0.1$。

优势：即使 $x < 0$，也有非零梯度： $ frac(d, d x) upright("LeakyReLU")(x) = cases(1,, x > 0, alpha,, x < 0) $
因此缓解 dead ReL U。

重要程度：★★★★☆

---

== 2.5 E L U

课件公式： $
upright("ELU")(x)

cases(x,, x >= 0, alpha(e^x - 1),, x < 0)
$ 其中：

- $alpha$：控制负半轴饱和值。
- $x$：pre-activation。

E L U 的直觉：
正半轴像 ReL U，负半轴平滑下降并趋向 $-alpha$。相比 ReL U，它可以产生负输出，使激活更接近 zero-centered。

重要程度：★★★☆☆

---

== 2.6 Maxout

课件公式： $
upright("Maxout")(x)

max(w_1^T x+b_1, w_2^T x+b_2)
$ 其中：

- $w_1,w_2$：两组可学习线性权重。
- $b_1,b_2$：偏置。
- $x$：输入向量。

Maxout 的直觉：
它不是固定形状的激活函数，而是让模型学习多个线性函数，并取最大值。它表达能力强，但参数量更大。

重要程度：★★★☆☆

---

= 3. Batch Normalization：从“避免饱和”到“稳定训练分布”

== 3.1 为什么需要 Normalization？

Sigmoid 或 tanh 的饱和来自输入过大或过小。课件提出解决思路：

_避免 feature values too large or too small。_

这就引出 normalization。

重要程度：★★★★★

---

== 3.2 Batch Normalization 公式

输入： $
x in RR^(N times D)
$ 其中：

- $N$：mini-batch size。
- $D$：feature/channel 维度。
- $x_(i,j)$：第 $i$ 个样本第 $j$ 个通道/特征的值。

对每个 feature/channel $j$，计算 batch mean： $ mu_j = frac(1, N) sum_(i=1)^(N) x_(i,j) $
计算 batch variance： $ sigma_j^2 = frac(1, N) sum_(i=1)^(N) (x_(i,j)-mu_j)^2 $
归一化： $ hat(x)_(i,j) = frac{x_(i,j)-mu_j}{sqrt{sigma_j^2+epsilon}} $
其中：

- $mu_j$：第 $j$ 个 feature/channel 在 batch 上的均值。
- $sigma_j^2$：第 $j$ 个 feature/channel 在 batch 上的方差。
- $epsilon$：小常数，防止除以 0。
- $hat(x)_(i,j)$：归一化后的值。

但是强制 zero mean 和 unit variance 可能过于严格，因此引入可学习 scale 和 shift： $
gamma, beta in RR^D

y_(i,j) = gamma_j hat(x)_(i,j) + beta_j
$ 其中：

- $gamma_j$：第 $j$ 个 feature 的可学习缩放参数。
- $beta_j$：第 $j$ 个 feature 的可学习平移参数。
- $y_(i,j)$：BatchNorm 输出。

如果模型希望恢复 identity mapping，可以学习： $
gamma_j = sigma_j, quad beta_j = mu_j
$ 则： $
y_(i,j) = sigma_j dot frac(x_(i,j)-mu_j, sigma_j) + mu_j = x_(i,j)
$ 忽略 $epsilon$ 时正好恢复原输入。

重要程度：★★★★★

---

== 3.3 BatchNorm 的训练时与测试时差异

训练时： $
mu_j,sigma_j^2
$ 来自当前 mini-batch。

测试时不能依赖 mini-batch，因为单个测试样本或者不同 batch 会造成输出不稳定。因此测试时使用训练过程中累积的 running mean 和 running variance：
$ mu_j^"test" = "running mean" $
$ sigma_j^2^"test" = "running variance" $
测试时 BatchNorm 变成线性算子： $
y_(i,j) = gamma_j frac(x_(i,j)-mu_j, sqrt(sigma_j^2+epsilon)) + beta_j
$ 可重写为： $
y_(i,j) = (frac(gamma_j, sqrt(sigma_j^2+epsilon))) x_(i,j) + (beta_j - frac(gamma_j mu_j, sqrt(sigma_j^2+epsilon)))
$ 所以测试时 B N 可以 fuse 到前一层 F C 或 Conv 中。

重要程度：★★★★★

---

== 3.4 BatchNorm 插入位置

课件给出的常见位置：

```text
F C / Conv
↓
BatchNorm
↓
Nonlinearity
```

即通常放在 fully connected 或 convolution layer 之后，activation function 之前。

重要程度：★★★★☆

---

= 4. C N N 架构演进：AlexNet → V G G → ResNet

== 4.1 C N N 基本卷积层

输入图像： $
32 times 32 times 3
$ 一个卷积核： $
5 times 5 times 3
$ 对一个局部 patch 做点积： $
w^T x + b
$ 其中：

- $w$：卷积核参数，展开后维度为 $5 dot 5 dot 3 = 75$。
- $x$：输入图像中一个 $5 times 5 times 3$ patch。
- $b$：bias。
- $w^T x+b$：一个空间位置的输出激活。

如果有 6 个 $5 times 5 times 3$ filters，则输出 6 张 activation maps。对于 $32 times 32$ 输入，不加 padding 且 stride = 1 时，空间尺寸变为： $
32 - 5 + 1 = 28
$ 所以输出为： $
28 times 28 times 6
$ 重要程度：★★★★★

---

== 4.2 卷积输出尺寸公式

一般情况下，输入尺寸为 $H times W$，卷积核大小为 $K$，padding 为 $P$，stride 为 $S$，输出高度为：
$ H_(upright("out")) = (H + 2P - K) / S + 1 $
输出宽度为：
$ W_(upright("out")) = (W + 2P - K) / S + 1 $
若输出通道数为 $C_("out")$，则输出 tensor 为： $ H_("out") times W_("out") times C_("out") $
重要程度：★★★★★

---

== 4.3 AlexNet：第一代 C N N 突破

AlexNet 架构：

```text
C O N V1
M A X P O O L1
N O R M1
C O N V2
M A X P O O L2
N O R M2
C O N V3
C O N V4
C O N V5
M A X P O O L3
F C6
F C7
F C8
```

意义：
AlexNet 是 I L S V R C 第一个 C N N-based winner，将 ImageNet top-5 error 从传统方法显著降低，标志着深度 C N N 在大规模视觉识别中的突破。

局限：

- 使用较大卷积核，例如 $11 times 11$。
- 网络深度有限。
- F C 层参数量大。
- 训练依赖工程技巧。

重要程度：★★★★☆

---

== 4.4 V G G：用小卷积核堆深度

V G G 的核心设计是大量使用 $3 times 3$ convolution。

课件问题：为什么用小卷积核？

三个 $3 times 3$ stride 1 conv 的 effective receptive field 等价于一个 $7 times 7$ conv。

推导：

第一层 $3 times 3$ 看到 $3$ 个像素范围。
第二层每个位置又基于第一层 $3 times 3$ 范围，因此感受野增加 $2$。
第三层再增加 $2$。

所以： $
3 + 2 + 2 = 7
$ 一般地，$L$ 个 $K times K$ stride 1 convolution 的 effective receptive field 为： $
R = 1 + L(K-1)
$ 当 $K=3,L=3$： $
R = 1 + 3(3-1)=7
$ 参数量比较：

一个 $7 times 7$ conv 参数量约为： $
49C^2
$ 三个 $3 times 3$ conv 参数量约为： $
3 times 9C^2 = 27C^2
$ 所以小卷积堆叠既减少参数，又增加非线性层数。

重要程度：★★★★★

---

== 4.5 Plain deeper network 的问题：Degradation Problem

课件展示：56-layer plain network 比 20-layer plain network 在 training error 和 test error 上都更差。

这不是 overfitting，因为 overfitting 应该表现为：

```text
training error 低，但 test error 高
```

而 degradation problem 是：

```text
training error 也更高，test error 也更高
```

所以问题不是泛化，而是优化失败：深层 plain network 理论表达能力更强，但 S G D 无法有效找到好解。

重要程度：★★★★★

---

== 4.6 ResNet：从直接拟合 H(x) 到拟合 residual F(x)

Plain layer 试图直接学习： $
H(x)
$ ResNet 改成学习 residual mapping： $
F(x) = H(x) - x
$ 所以： $
H(x) = F(x) + x
$ Residual block 输出： $
y = F(x) + x
$ 其中：

- $x$：block 输入。
- $F(x)$：由若干卷积层学习到的 residual function。
- $y$：block 输出。
- $x$ 到输出的路径是 identity shortcut。

关键直觉：
如果最优映射接近 identity，即： $
H(x)=x
$ plain network 需要学习一个完整 identity function。
ResNet 只需要让： $
F(x)=0
$ 于是： $
H(x)=F(x)+x=x
$ 学习 $F(x)=0$ 比学习复杂的 identity mapping 容易得多。

重要程度：★★★★★

---

== 4.7 ResNet 架构特征

课件总结：

```text
1. Stack residual blocks
2. 每个 residual block 有两个 3x3 conv layers
3. 周期性 double filters，并用 stride 2 downsample spatial size
4. 开头有 stem conv，例如 7x7 conv, stride 2
5. 末尾没有大 F C layers，只用 global average pooling + F C 1000
```

Global Average Pooling 的意义：

若最后 feature map 为： $H times W times C$ 则对每个 channel 做空间平均：
$ z_c = frac(1, H W) sum_(i=1)^(H) sum_(j=1)^(W) x_(i,j,c) $
