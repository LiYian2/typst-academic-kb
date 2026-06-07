// @title: Lecture 4 学习笔记：Convolutional Neural Networks, CNN
// @description: 
// @order: 999

= Lecture 4 学习笔记：Convolutional Neural Networks, CNN

== 0. 本章核心目标与进化树

本章要解决的核心问题是：_如何让神经网络有效处理具有空间/网格结构的数据，尤其是图像，同时避免全连接网络在高维图像输入上的参数爆炸、训练低效和空间结构丢失问题。_

CNN 的演进逻辑可以概括为：

```text
普通 FNN
 ↓ 问题：图像展平成向量后丢失空间结构；全连接参数量巨大
卷积层
 ↓ 通过 local receptive field + sparse connectivity + parameter sharing
特征图 feature map
 ↓ 通过 pooling 获得局部平移不变性、降采样、减少计算
经典 CNN: LeNet / AlexNet
 ↓ 问题：更深网络训练困难、过拟合、计算量大
Batch Normalization / Dropout / ReLU
 ↓ 问题：如何进一步加深、加宽、增强表达能力
VGG / GoogLeNet-Inception / ResNet
 ↓ 问题：结构越来越复杂，手工设计困难
DenseNet / ResNeXt / SENet / NASNet
 ↓ 扩展任务
UNet / Faster R-CNN / Mask R-CNN
```

重要程度标记：
`★★★★★` 必考核心；`★★★★` 高频重点；`★★★` 理解即可；`★★` 背景知识。

---

= 1. 为什么需要 CNN？★★★★★

== 1.1 FNN 的问题：图像输入不可扩展

以 CIFAR-10 为例，输入是：

$ 32 times 32 times 3 $

其中：

- $32 times 32$：图像空间尺寸；
- (3)：RGB 三个颜色通道。

如果使用普通全连接网络，图像需要被 flatten 成一个向量。对于小图像还可以接受，但如果图像大小是：

$ 200 times 200 times 3 = 120000 $

那么下一层的每一个神经元都需要 (120000) 个权重。若隐藏层有 (H) 个神经元，则参数量约为：

$ 120000H $

这会导致三个问题：

+ _参数量爆炸_：高分辨率图像无法高效训练；
+ _空间结构丢失_：flatten 后，像素之间的邻域关系被破坏；
+ _缺乏平移泛化能力_：同一个物体出现在不同位置，FNN 需要重新学习类似模式。

CNN 的核心动机就是利用图像的空间局部性：_相邻像素更相关；同一个视觉模式可能出现在任意位置。_

---

= 2. CNN 的基本定义与结构 ★★★★★

CNN 是一种特殊的 feedforward neural network，用于处理具有 grid-like topology 的数据，例如：

- 1D：语音、文本序列；
- 2D：图像；
- 3D：视频、医学影像。

CNN 的输入通常是一个三维张量：

$ I in R R^{H times W times C} $

其中：

- (H)：height，高度；
- (W)：width，宽度；
- (C)：channel/depth，通道数。

一个基本 CNN 通常由以下部分组成：

| 模块 | 作用 | 是否有参数 | 重要程度 |
| --------------------- | ----------- | ----: | ----- |
| Convolution layer | 局部特征提取 | 有 | ★★★★★ |
| Activation / ReLU | 引入非线性 | 无 | ★★★★ |
| Pooling layer | 降采样、引入局部不变性 | 无 | ★★★★★ |
| Normalization layer | 稳定训练、加速收敛 | 有/统计量 | ★★★★ |
| Fully-connected layer | 输出分类分数 | 有 | ★★★★ |

典型结构：

$ "I N P U T" -> "C O N V" -> "R E L U" -> "P O O L" -> "F C" $

以 CIFAR-10 为例：

$ [32 times 32 times 3] -> [32 times 32 times 12] -> [32 times 32 times 12] -> [16 times 16 times 12] -> [1 times 1 times 10] $

其中 (10) 表示 10 个类别的分类分数。

---

= 3. Convolution：卷积的数学定义 ★★★★★

== 3.1 连续卷积

数学中，两个函数 (f) 和 (g) 的卷积定义为：

$ (f * g)(t) = integral_{-oo}^{oo} f(tau) g(t-tau) d tau integral_{-oo}^{oo} f(t-tau) g(tau) d tau $

符号解释：

- (f)：输入信号；
- (g)：卷积核/filter/kernel；
- (t)：输出位置；
- $tau$：积分变量；
- (\*)：卷积运算。

直觉：输出位置 (t) 的值由输入函数 (f) 在邻域中的值加权求和得到，权重由 (g) 决定。

---

== 3.2 离散卷积

对于整数定义域上的离散信号：

$ (f * g)(n) = sum_{m=-oo}^{oo} f[m] g[n-m] sum_{m=-oo}^{oo} f[n-m] g[m] $

符号解释：

- (n)：输出位置；
- (m)：求和索引；
- (f[m])：输入信号在位置 (m) 的值；
- (g[n-m])：卷积核在相对位置 (n-m) 的权重。

直觉：卷积可以被看成一种 weighted smoothing。例如 GPS 信号有噪声，可以使用最近若干时刻的观测值做加权平均，近期值权重大，远期值权重小。

---

= 4. 2D Convolution 与 Cross-Correlation ★★★★★

== 4.1 真正的二维卷积

二维输入 (I) 与二维卷积核 (K) 的卷积为：

$ Z(i,j) = (I * K)(i,j) sum_m sum_n I(m,n) K(i-m,j-n) $

符号解释：

- (I)：输入图像或 feature map；
- (K)：二维卷积核；
- (Z)：输出 feature map；
- ((i,j))：输出位置；
- ((m,n))：输入位置。

这里的 (K(i-m,j-n)) 表示卷积核被翻转后参与计算。

---

== 4.2 深度学习库中的“卷积”通常是 cross-correlation

课件特别强调：很多机器学习库实现的是 cross-correlation，但习惯上仍叫 convolution：

$ Z(i,j) = (I * K)(i,j) sum_m sum_n I(i+m,j+n) K(m,n) $

区别在于：

| 操作 | 公式 | 是否翻转 kernel | 深度学习中常用？ |
| ----------------- | ------------------ | ----------: | --------: |
| Convolution | (I(m,n)K(i-m,j-n)) | 是 | 理论定义 |
| Cross-correlation | (I(i+m,j+n)K(m,n)) | 否 | 实际 CNN 实现 |

在 CNN 中，kernel 是可学习参数，是否翻转并不影响模型表达能力，所以深度学习框架通常直接用 cross-correlation。

---

= 5. CNN 的三大核心优势 ★★★★★

课件总结 CNN 的三个核心思想：

== 5.1 Sparse Connectivity：稀疏连接

普通全连接层中，每个输出神经元连接所有输入：

$ z_j = sum_i w_{{"ji"}}x_i + b_j $

而卷积层中，每个神经元只连接输入的局部区域，即 receptive field。

例如输入为：

$ 32 times 32 times 3 $

若 receptive field 是：

$ 5 times 5 $

那么每个卷积神经元只连接：

$ 5 times 5 times 3 = 75 $

个输入，而不是 $32 times 32 times 3 = 3072$ 个输入。

这大幅降低参数量和计算量。

---

== 5.2 Parameter Sharing：参数共享

同一个卷积核在整张图像上滑动，因此不同空间位置使用同一组权重。

如果一个卷积核大小是：

$ 5 times 5 times 3 $

那么它只需要 (75) 个权重，加上一个 bias。无论图像多大，这个 kernel 的参数量都不变。

直觉：检测“边缘”“角点”“纹理”的局部模式不应该依赖于图像绝对位置。猫耳朵出现在左上角或右下角，本质上仍是类似视觉模式。

---

== 5.3 Equivariance：平移等变性

课件给出的形式是：

$ f(g(x)) = g(f(x)) $

解释：

- (x)：输入；
- (g)：某种变换，例如平移；
- (f)：卷积操作或 CNN 层。

如果输入先平移，再做卷积，结果等价于先做卷积，再平移输出。

这叫 _equivariance_，不是 invariance。

区别：

| 概念 | 数学形式 | 含义 |
| ---------------- | ------------------- | ------------- |
| Equivariance 等变性 | (f(g(x)) = g(f(x))) | 输入变了，输出以相同方式变 |
| Invariance 不变性 | (f(g(x)) = f(x)) | 输入变了，输出不变 |

卷积层提供的是平移等变性；pooling 更接近平移不变性。

---

= 6. 卷积层公式 ★★★★★

课件中给出 3D tensor 输入时的卷积公式：

$ Z^{(i)}_{j,k} = sum_{l,m,n} I^{(l)}*{j+m-1,k+n-1} K^{(i,l)}*{m,n} + b^{(i)} $

符号解释：

- (I)：输入张量；
- $I^{(l)}$：输入的第 (l) 个 channel；
- (Z)：输出张量；
- $Z^{(i)}$：输出的第 (i) 个 feature map；
- ((j,k))：输出 feature map 的空间位置；
- $K^{(i,l)}$：连接输入 channel (l) 到输出 channel (i) 的卷积核；
- ((m,n))：卷积核内部位置；
- $b^{(i)}$：第 (i) 个输出 channel 的 bias；
- $sum_{l,m,n}$：对所有输入通道和 kernel 空间位置求和。

关键理解：一个输出 channel 对应一个 filter bank。若输入有 $C_{"in"}$ 个通道，输出有 $C_{"out"}$ 个通道，则卷积核整体形状是：

$ C_{"out"} times C_{"in"} times K_h times K_w $

---

== 6.1 加入 stride 的卷积公式

若 stride 为 (s)，公式变为：

$ Z^{(i)}_{j,k} = sum_{l,m,n} I^{(l)}*{(j-1)s+m,(k-1)s+n} K^{(i,l)}*{m,n} + b^{(i)} $

其中：

- (s)：stride，卷积核每次滑动的步长；
- 当 (s=1)，逐格滑动；
- 当 (s=2)，每次跳 2 格，输出尺寸更小。

---

== 6.2 输出尺寸公式

虽然课件没有显式写出，但这是考试常考公式。若输入大小为 $H times W$，kernel 大小为 $F_h times F_w$，padding 为 (P)，stride 为 (S)，则输出大小为：

$ H_{"out"} = floor.l frac(H + 2P - F_h, S) floor.r + 1  W_{"out"} = floor.l frac(W + 2P - F_w, S) floor.r + 1 $

符号解释：

- (H,W)：输入高度和宽度；
- (F_h,F_w)：卷积核高度和宽度；
- (P)：zero-padding 大小；
- (S)：stride；
- $H_{"out"},W_{"out"}$：输出 feature map 尺寸。

---

= 7. 卷积反向传播 ★★★★★

课件设损失函数为：

$ J(K,b) $

反向传播时，上一层传回梯度张量 (G)：

$ G^{(i)}_{j,k} = frac(partial J(K,b), partial Z^{(i)}_{j,k}) $

这里：

- $G^{(i)}_{j,k}$：损失对输出位置 ((j,k)) 的梯度；
- $Z^{(i)}_{j,k}$：卷积层输出；
- (J)：整体损失函数。

---

== 7.1 对 kernel 的梯度

$ frac(partial, partial K^{(i,l)}_{j,k})J(K,b) = sum_{m,n} frac(partial J, partial Z^{(i)}*{m,n}) frac(partial Z^{(i)}*{m,n}, partial K^{(i,l)}_{j,k}) $

代入卷积公式可得：

$ frac(partial, partial K^{(i,l)}_{j,k})J(K,b) = sum_{m,n} I^{(l)}*{(m-1)s+j,(n-1)s+k} G^{(i)}*{m,n} $

直觉：因为同一个 kernel 参数在不同空间位置重复使用，所以它的梯度要把所有位置的贡献累加起来。这就是 parameter sharing 在反向传播中的体现。

---

== 7.2 对输入的梯度

$ frac(partial, partial I^{(l)}_{p,q})J(K,b) = sum_i sum_{m,n} frac(partial J, partial Z^{(i)}*{m,n}) frac(partial Z^{(i)}*{m,n}, partial I^{(l)}_{p,q}) $

代入后：

$ frac(partial, partial I^{(l)}_{p,q})J(K,b) = sum_i sum_{m,n} K^{(i,l)}*{p-(m-1)s,q-(n-1)s} G^{(i)}*{m,n} $

直觉：一个输入像素可能参与多个重叠 patch 的卷积计算，所以它的梯度也来自多个输出位置的累加。

---

== 7.3 对 bias 的梯度

$ frac(partial, partial b^{(i)})J(K,b) = sum_{m,n} G^{(i)}_{m,n} $

因为同一个 bias $b^{(i)}$ 被加到第 (i) 个输出 feature map 的所有空间位置，所以梯度是这些位置梯度的总和。

---

== 7.4 参数更新

$ K^{(i,l)}_{j,k} arrow.l K^{(i,l)}_{j,k} - alpha frac(partial, partial K^{(i,l)}_{j,k}) J(K,b)  b^{(i)} arrow.l b^{(i)} - alpha frac(partial, partial b^{(i)}) J(K,b) $

其中：

- $alpha$：learning rate；
- 其余符号同上。

---

= 8. 卷积反向传播例题 ★★★★★

课件给出单通道、stride (s=1)、valid cross-correlation、无 padding、(b=0) 的例子：

$ I= mat(1, 2, 3\ 4, 5, 6\ 7, 8, 9)  K= mat(1, 0\ -1, 2)  J=sum_{m,n} Z_{m,n} $

前向传播：

$ Z_{m,n} = sum_{j=1}^{2} sum_{k=1}^{2} K_{j,k}I_{m+j-1,n+k-1} $

逐项计算：

$ Z_{1,1} = 1 dot 1 + 0 dot 2 + (-1) dot 4 + 2 dot 5 = 7  Z_{1,2} = 1 dot 2 + 0 dot 3 + (-1) dot 5 + 2 dot 6 = 9  Z_{2,1} = 1 dot 4 + 0 dot 5 + (-1) dot 7 + 2 dot 8 = 13  Z_{2,2} = 1 dot 5 + 0 dot 6 + (-1) dot 8 + 2 dot 9 = 15 $

所以：

$ Z= mat(7, 9\ 13, 15) $

因为：

$ J=sum_{m,n}Z_{m,n} $

所以：

$ G_{m,n} = frac(partial J, partial Z_{m,n}) = 1  G = mat(1, 1\ 1, 1) $

---

== 8.1 Kernel gradient

$ frac(partial J, partial K_{j,k}) = sum_{m,n}I_{m+j-1,n+k-1}G_{m,n} $

因为所有 $G_{m,n}=1$，所以：

$ frac(partial J, partial K_{1,1}) = I_{1,1}+I_{1,2}+I_{2,1}+I_{2,2} = 1+2+4+5 = 12  frac(partial J, partial K_{1,2}) = I_{1,2}+I_{1,3}+I_{2,2}+I_{2,3} = 2+3+5+6 = 16  frac(partial J, partial K_{2,1}) = I_{2,1}+I_{2,2}+I_{3,1}+I_{3,2} = 4+5+7+8 = 24  frac(partial J, partial K_{2,2}) = I_{2,2}+I_{2,3}+I_{3,2}+I_{3,3} = 5+6+8+9 = 28 $

因此：

$ frac(partial J, partial K) = mat(12, 16\ 24, 28) $

---

== 8.2 Bias gradient

$ frac(partial J, partial b) = sum_{m,n} G_{m,n} = 1+1+1+1 = 4 $

---

== 8.3 Input gradient

$ frac(partial J, partial I_{p,q}) = sum_{m,n} K_{p-m+1,q-n+1}G_{m,n} $

越界索引视为 0。

中心像素例子：

$ frac(partial J, partial I_{2,2}) = K_{2,2}+K_{2,1}+K_{1,2}+K_{1,1} = 2+(-1)+0+1 = 2 $

最终：

$ frac(partial J, partial I) = mat(1, 1, 0\ 0, 2, 2\ -1, 1, 2) $

考试重点：输入梯度不是简单复制 kernel，而是所有覆盖该输入像素的输出梯度路径之和。

---

= 9. Pooling 池化层 ★★★★★

Pooling 的作用是把某一局部区域替换成 summary statistic。

常见 pooling：

| 类型 | 定义 | 特点 |
| ---------------------------- | --------- | ---------- |
| Max pooling | 取局部最大值 | 最常用，保留最强响应 |
| Average pooling | 取局部平均值 | 更平滑 |
| L2-norm pooling | 取局部 L2 范数 | 强调能量 |
| Probability weighted pooling | 加权池化 | 较少见 |

课件例子：对矩阵做 $2 times 2$ max pooling，stride (2)：

$ mat(1, 1, 2, 4\ 5, 6, 7, 8\ 3, 2, 1, 0\ 1, 2, 3, 4) -> mat(6, 8\ 2, 4) $

四个窗口分别取最大值：

$ max mat(1, 1\ 5, 6) =6  max mat(2, 4\ 7, 8) =8  max mat(3, 2\ 1, 2) =3 $

注意：课件中输出写的是 (2)，但按标准 $2 times 2$ max pooling 应该是 (3)。这里可能是课件矩阵排版或红框标记造成的歧义。考试时按定义算：取窗口最大值。

---

== 9.1 Pooling 的作用

Pooling 的性质：

+ _对小平移不敏感_
 局部范围内物体稍微移动，max response 可能仍然相同。

+ _降低计算和显存需求_
 空间尺寸下降，下一层输入更小。

+ _改善统计效率_
 更关注是否存在某特征，而不是精确位置。

+ _处理可变尺寸输入_
 尤其是 global pooling，可以把任意空间尺寸压缩为固定维度。

---

== 9.2 Pooling 的反向传播

课件指出：pooling 的 backward pass 可以看成 forward subsampling 的反向 upsampling。

对于 max pooling：

- 前向传播记录最大值位置；
- 反向传播时，梯度只传给前向时取最大值的输入位置；
- 其他位置梯度为 0。

例如：

$ mat(1, 3\ 2, 0) -> 3 $

若上游梯度是 $delta$，则反向梯度为：

$ mat(0, delta\ 0, 0) $

---

= 10. Data Normalization 与 Batch Normalization ★★★★★

== 10.1 为什么需要 normalization？

课件举例：两个特征尺度不同：

$ x_1 in [0,1]  x_2 in [10,1000] $

Ridge regression：

$ hat(y) = w_0+w_1x_1+w_2x_2  J(w) = E[L(y,hat(y))] + lambda(w_1^2+w_2^2) $

符号解释：

- $hat(y)$：模型预测；
- (y)：真实标签；
- (w_0)：bias；
- (w_1,w_2)：特征权重；
- $L(y,hat(y)$)：损失函数；
- $lambda$：正则化强度；
- $E[dot]$：数据分布上的期望。

问题：由于 (x_2) 数值范围更大，改变 (w_2) 对预测 $hat(y)$ 影响更大；改变 (w_1) 影响较小。因此正则化项会更强地把 (w_1) 压向 0，导致模型 bias 增大。

---

== 10.2 梯度尺度不均导致训练慢

课件给出梯度：

$ frac(partial J, partial w_1) = E[ frac(partial L, partial hat(y))x_1 ] + 2lambda w_1  frac(partial J, partial w_2) = E[ frac(partial L, partial hat(y))x_2 ] + 2lambda w_2 $

因为 (x_1) 小、(x_2) 大，所以：

$ |frac(partial J, partial w_1)| << |frac(partial J, partial w_2)| $

这会使 loss contour 变得狭长，梯度下降出现 zig-zag，训练变慢。

---

== 10.3 标准数据归一化公式

设数据矩阵为：

$ X=[x_{{"ij"}}]_{N times D} $

其中：

- (N)：样本数；
- (D)：特征维度；
- $x_{{"ij"}}$：第 (i) 个样本的第 (j) 个特征。

第 (j) 个特征的均值：

$ mu_j = frac(1, N) sum_{i=1}^{N}x_{{"ij"}} $

第 (j) 个特征的方差：

$ sigma_j^2 = frac(1, N) sum_{i=1}^{N} (x_{{"ij"}}-mu_j)^2 $

归一化：

$ hat(x)_{{"ij"}} = frac(x_{{"ij"}}-mu_j, sigma_j) $

直觉：把每个特征变成均值为 0、方差为 1 的尺度，使优化更稳定。

---

= 11. Batch Normalization ★★★★★

深度模型可以看作多个模型的串联：

$ m_1 -> m_2 -> .. -> m_L $

其中：

- (m_1) 接收原始输入；
- (m_2) 接收第一层隐藏输出；
- 后续层接收前一层输出。

问题是：前面层参数改变后，后面层看到的输入分布也会改变。这被称为 covariate shift。Batch Normalization 的思想是：_不仅规范化原始输入，也规范化每一层的输入/激活。_

课件给出 BN 形式：

$ hat(x) = frac(x-mu, sigma)  y = gamma hat(x) + beta $

符号解释：

- (x)：某一层的输入或激活；
- $mu$：mini-batch 内 (x) 的均值；
- $sigma$：mini-batch 内 (x) 的标准差；
- $hat(x)$：标准化后的激活；
- $gamma$：可学习 scale 参数；
- $beta$：可学习 shift 参数；
- (y)：BN 层输出。

为什么还要 $gamma,beta$？

因为单纯标准化会限制表示能力。引入 $gamma,beta$ 后，网络可以学习是否需要恢复某些均值和尺度。

训练时：

$ mu,sigma $

来自当前 mini-batch。

测试时：

$ mu,sigma $

使用训练过程中估计的 running mean 和 running variance。

BN 的作用：

| 作用 | 解释 | 重要程度 |
| -------- | ---------------- | ----- |
| 加速训练 | 梯度尺度更稳定 | ★★★★★ |
| 降低初始化敏感性 | 不容易因激活爆炸/消失崩掉 | ★★★★ |
| 提供轻微正则化 | mini-batch 统计有噪声 | ★★★★ |
| 稳定深层网络 | 后层输入分布更稳定 | ★★★★★ |

---

= 12. CNN 发展史：从手工特征到深度 CNN ★★★★

== 12.1 早期历史

课件中的关键节点：

| 时间 | 方法/事件 | 意义 |
| ---- | ---------------------------- | --------- |
| 1962 | Hubel & Wiesel 视觉皮层研究 | 提供局部感受野启发 |
| 1980 | Fukushima Neocognitron | 类 CNN 结构 |
| 1989 | LeCun 将 BP 用于 2D convolution | 现代 CNN 起点 |
| 1996 | AT&T 支票识别系统 | CNN 实际应用 |
| 2009 | Google StreetView 检测人脸和车牌 | CNN 工业部署 |

---

== 12.2 ILSVRC 与 CNN 崛起

ILSVRC 是 ImageNet Large Scale Visual Recognition Challenge，2010–2017 举办。分类任务包含超过：

1,000,000

张训练图像和：

100,000

张测试图像。

评价指标：top-5 error rate。

---

== 12.3 非 CNN 方法的瓶颈

2010–2011 年高排名方法主要依赖手工特征和传统机器学习：

- SIFT；
- LBP；
- Fisher Vector；
- SVM；
- GIST；
- CSIFT；
- stacking；
- boosting。

结果：

| 年份 | 方法 | top-5 error |
| ----------- | ----------------------- | ----------: |
| 2010 | Descriptor Coding + SVM | 0.28 |
| 2011 | XRCE | 0.26 |
| Human level | 人类 | 0.051 |

瓶颈：手工特征表达能力有限，无法从大规模数据中端到端学习多层视觉表示。

---

= 13. AlexNet：深度学习爆发点 ★★★★★

AlexNet，2012 年，是最早具有代表性的 deep learning model 之一。

特点：

- 深度 CNN；
- 7 layers；
- ReLU；
- Dropout；
- 大规模 GPU 训练；
- 显著降低 ImageNet top-5 error。

结果：

$ 0.26 -> 0.15 $

这是 CNN 发展史上的关键转折点。

为什么 AlexNet 成功？

| 改进 | 解决的问题 |
| ------------ | ------------------------- |
| 深层卷积结构 | 学习层次化视觉特征 |
| ReLU | 缓解 sigmoid/tanh 梯度饱和，训练更快 |
| Dropout | 缓解过拟合 |
| GPU 训练 | 让大模型训练可行 |
| 大数据 ImageNet | 支持端到端特征学习 |

---

= 14. VGG：更深、更规则的 CNN ★★★★

VGG 是 ILSVRC 2014 runner-up。

特点：

- 使用 11、13、16、19 层 CNN；
- 多个小卷积核堆叠；
- 结构规整；
- 单模型 top-5 error 为 0.084；
- ensemble 后 error 为 0.073。

VGG 的方法论意义：相比设计复杂模块，VGG 证明了“简单规则结构 + 更深网络”也能显著提升性能。

---

= 15. GoogLeNet / Inception ★★★★

GoogLeNet 是 ILSVRC 2014 winner，22 层 CNN，error rate：

0.066

它的核心思想来自 Inception module。

Inception 的问题意识：不同大小的卷积核捕捉不同尺度特征：

- $1 times 1$：通道混合、降维；
- $3 times 3$：中等局部模式；
- $5 times 5$：更大感受野；
- pooling：局部统计。

所以 naive Inception 的思想是：_为什么不同时使用它们？_

```text
输入
 ├── 1×1 conv
 ├── 3×3 conv
 ├── 5×5 conv
 └── pooling
 ↓
 concatenate
```

Inception 的本质是多分支、多尺度特征提取。

---

= 16. Network in Network, NIN ★★★★

NIN 提出两个重要技术。

== 16.1 MLPConv

传统卷积层是线性卷积加非线性：

$ "linear conv" -> "activation" $

NIN 的 MLPConv 在局部 patch 上使用更复杂的 MLP：

$ "conv" -> 1 times 1 "conv" -> 1 times 1 "conv" $

直觉：普通卷积对局部 patch 的建模主要是线性投影，而 MLPConv 增强了局部非线性表达能力。

---

== 16.2 Global Average Pooling

NIN 的第二个技术是用 global average pooling 替代最后的 fully-connected layer。

如果最后一层 feature map 的 channel 数等于类别数，则每个 channel 对应一个类别。对每个 channel 做全局平均：

$ s_c = frac(1, H W) sum_{i=1}^{H} sum_{j=1}^{W} A_{c,i,j} $

其中：

- (s_c)：类别 (c) 的 score；
- $A_{c,i,j}$：第 (c) 个 channel 在位置 ((i,j)) 的激活；
- (H,W)：feature map 高度和宽度。

优点：

+ 减少参数；
+ 降低 fully-connected layer 的过拟合；
+ 对物体位置更鲁棒；
+ 增强可解释性，因为每个 channel 对应一个类别响应图。

---

= 17. ResNet：残差连接解决深层网络退化 ★★★★★

ResNet 是 2015 年的关键突破。

课件给出公式：

$ O_{i,p} = f("Conv"(O_{i,p-1})) + O_{i,p-1} $

符号解释：

- $O_{i,p-1}$：第 (p-1) 个模块/层的输入；
- $"Conv"(dot$)：卷积变换；
- $f(dot$)：非线性或卷积块函数；
- $O_{i,p}$：残差块输出；
- $+\ O_{i,p-1}$：shortcut connection / identity connection。

更标准地写：

$ y = F(x) + x $

其中：

- (x)：残差块输入；
- (F(x))：残差函数；
- (y)：残差块输出。

ResNet 解决的问题：深层网络不是简单越深越好。普通深层网络会出现 degradation problem，即训练误差也随着深度增加而变差。这不是过拟合，而是优化困难。

残差连接的直觉：如果额外层没有用，网络只需要学习：

$ F(x)=0 $

那么：

$ y=x $

这比直接学习复杂的 identity mapping 更容易。

ResNet 结果：

| 模型 | 结果 |
| ---------------------------- | ----------------: |
| ResNet ensemble, ILSVRC 2015 | top-5 error 0.036 |
| single 152-layer ResNet | top-5 error 0.045 |
| 最深可达 | 1202 layers |

ResNet 还被用于 AlphaGo。课件指出 ResNet AlphaGo 比 CNN AlphaGo 强约 600 ELO，约等价于 96% win rate。

---

= 18. DenseNet：密集连接 ★★★★

DenseNet 的思想：每一层不只连接上一层，而是收集所有前面层的信息。

课件公式：

$ O_{i,p} = "concat" ( O_{i,p-1}, f("Conv"(O_{i,p-1})) ) $

更一般地，DenseNet 第 (l) 层接收：

$ x_l = H_l([x_0,x_1,dot(s),x_{l-1}]) $

其中：

- $[x_0,x_1,dot(s),x_{l-1}]$：所有前面层输出的 concat；
- (H_l)：第 (l) 层的变换；
- (x_l)：第 (l) 层输出。

DenseNet 的动机：

+ 缓解梯度消失；
+ 强化 feature reuse；
+ 减少重复学习；
+ 让浅层细节和深层语义同时可用。

课件给出 DenseNet-161 ((k=48)) 单模型 ILSVRC error：

0.053

其中 (k) 是 growth rate，即每层新增 channel 数。

---

= 19. ResNeXt：多路径与 group convolution ★★★★

ResNeXt 是 2016 ILSVRC runner-up。

核心思想：不只是加深网络，而是引入多个并行路径。

关键词：

- cardinality；
- group convolution；
- 多路径聚合。

Group convolution 的作用是减少计算量，同时保留表达能力。相比普通卷积，group convolution 把 channel 分成若干组，每组独立卷积，然后拼接结果。

普通卷积参数量：

$ C_{"out"} C_{"in"} K_h K_w $

若分成 (G) 组，参数量约为：

$ frac(C_{"out"} C_{"in"} K_h K_w, G) $

课件给出 ResNeXt-101 单模型 ILSVRC error：

0.053

---

= 20. SENet：通道注意力 ★★★★

SENet 是最后一届 ILSVRC classification winner。

核心思想：

$ "All channels are created unequal." $

即不同 channel 的重要性不同，应当自适应分配权重。

SENet 使用 global average pooling 生成 channel-level descriptor：

$ z_c = frac(1, H W) sum_{i=1}^{H} sum_{j=1}^{W} U_{c,i,j} $

然后通过小网络生成每个 channel 的权重：

$ s = sigma(W_2delta(W_1 z)) $

最后重新标定通道：

$ tilde(U)_c = s_c U_c $

符号解释：

- (U_c)：第 (c) 个输入 channel；
- (z_c)：第 (c) 个 channel 的全局描述；
- (W_1,W_2)：全连接层参数；
- $delta$：ReLU；
- $sigma$：sigmoid；
- (s_c)：第 (c) 个 channel 的权重；
- $tilde(U)_c$：重新加权后的 channel。

课件给出 SENet-154 单模型 top-5 error：

0.038

---

= 21. NASNet / AutoML ★★★

随着 CNN 结构越来越复杂，手工设计架构成本很高。NASNet 的思想是自动搜索 CNN 架构。

课件提到 NASNet 使用 controller RNN：

```text
controller RNN → sample architecture → train/evaluate → update controller
```

目标：让模型自动发现更优的网络模块或 cell。

单个 NASNet 模型 ILSVRC top-5 error：

0.038

意义：CNN 发展从人工设计结构走向自动化结构搜索。

---

= 22. CNN 的其他任务扩展 ★★★

== 22.1 CNN for Text

CNN 不只用于图像，也可用于文本分类。文本输入可以表示为矩阵：

$ X in R R^{T times d} $

其中：

- (T)：句子长度；
- (d)：词向量维度；
- 每一行是一个 word vector；
- word vector 可以来自 word2vec，也可以由 embedding layer 训练得到。

卷积核沿文本序列滑动，可以捕捉 n-gram 局部模式。

---

== 22.2 UNet：Image-to-Image

UNet 是图像到图像任务的经典结构，常用于：

- segmentation；
- denoising；
- medical image analysis；
- image restoration。

核心结构：

```text
encoder/downsampling path
 ↓
bottleneck
 ↓
decoder/upsampling path
```

并通过 skip connection 把浅层空间细节传给解码器。

---

== 22.3 Deconvolution

Deconvolution 通常指 transposed convolution，用于上采样。它常见于：

- image generation；
- segmentation；
- decoder；
- super-resolution。

注意：严格数学意义上的 deconvolution 是反卷积，但深度学习中常把 transposed convolution 叫 deconvolution。

---

== 22.4 Faster R-CNN / Mask R-CNN

用于 object detection 和 instance segmentation。

| 模型 | 任务 |
| ------------ | ---------------------------------------- |
| Faster R-CNN | object detection |
| Mask R-CNN | object detection + instance segmentation |

Faster R-CNN 引入 region proposal network；Mask R-CNN 在检测框基础上增加 mask branch。

---

= 23. CNN 模型演进对比表 ★★★★★

| 阶段 | 代表方法 | 核心问题 | 解决方式 | 重要程度 |
| --------- | ------ | --------------- | ---------------------- | ----- |
| FNN | MLP | 参数爆炸、空间结构丢失 | 无法有效解决 | ★★★★★ |
| Early CNN | LeNet | 利用局部结构 | 卷积 + 池化 | ★★★★★ |
| AlexNet | 2012 | 手工特征性能瓶颈 | 深 CNN + ReLU + Dropout | ★★★★★ |
| VGG | 2014 | 更强表达 | 更深、更规则的小卷积核结构 | ★★★★ |
| GoogLeNet | 2014 | 多尺度特征、计算效率 | Inception 多分支 | ★★★★ |
| NIN | 2013 | 局部线性表达不足、FC 过拟合 | MLPConv + GAP | ★★★★ |
| ResNet | 2015 | 深层网络退化 | Residual shortcut | ★★★★★ |
| ResNeXt | 2016 | 表达能力与计算效率平衡 | 多路径 + group conv | ★★★★ |
| DenseNet | 2017 | 特征复用、梯度流 | dense connection | ★★★★ |
| SENet | 2017 | channel 重要性不同 | channel attention | ★★★★ |
| NASNet | AutoML | 手工设计复杂 | architecture search | ★★★ |

---

= 24. ILSVRC 性能演进 ★★★★

| 年份 | 方法 | Top-5 Error |
| ----------- | ----------------------- | ----------: |
| 2010 | Descriptor Coding + SVM | 0.28 |
| 2011 | XRCE | 0.26 |
| Human level | Human | 0.051 |
| 2012 | AlexNet | 0.15 |
| 2014 | GoogLeNet | 0.066 |
| 2014 | VGG ensemble | 0.073 |
| 2015 | ResNet ensemble | 0.036 |
| 2016 | Trimps-Soushen ensemble | 0.030 |
| 2017 | SENet variants ensemble | 0.0225 |

核心趋势：

```text
手工特征 + 传统分类器
 ↓
端到端深度 CNN
 ↓
更深网络
 ↓
多尺度模块
 ↓
残差连接
 ↓
密集连接 / 通道注意力 / 自动搜索
```

---

= 25. 期末复习重点排序

== 必须掌握 ★★★★★

+ CNN 为什么比 FNN 适合图像；
+ sparse connectivity；
+ parameter sharing；
+ equivariance vs invariance；
+ 2D convolution / cross-correlation 公式；
+ 3D tensor 卷积公式；
+ stride、padding、output size；
+ convolution backpropagation；
+ max pooling forward/backward；
+ batch normalization 公式和训练/测试差异；
+ ResNet residual connection；
+ AlexNet 为什么是转折点。

== 高频理解 ★★★★

+ VGG、GoogLeNet、Inception 的区别；
+ NIN 的 MLPConv 和 global average pooling；
+ DenseNet 与 ResNet 的区别；
+ SENet 的 channel weighting；
+ ResNeXt 的 group convolution；
+ ILSVRC 发展脉络。

== 背景知识 ★★★

+ Hubel-Wiesel；
+ Neocognitron；
+ CNN for Text；
+ UNet；
+ Faster R-CNN / Mask R-CNN；
+ NASNet / AutoML。

---

= 26. 最核心的一句话总结

CNN 的本质不是“换了一种网络层”，而是把图像的先验结构编码进模型：_局部连接利用空间局部性，参数共享利用模式可平移性，池化引入局部不变性，深层堆叠形成从边缘到语义的层次化表示。_
