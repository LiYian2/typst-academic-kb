// @title: Week 5: Machine Learning Basis 复习笔记
// @description:
// @order: 999

= Week 5: Machine Learning Basis 复习笔记

== 0. 本章核心目标：从“记忆样本”走向“学习参数”

本章要解决的问题是：给定一张图像，如何让计算机输出类别标签，例如 Cat。计算机看到的不是“猫”这个语义对象，而是一个整数张量，例如 $800 times 600 times 3$ 的 RGB 像素数组，每个像素值通常在 $[0,255]$。难点是：同一类别在像素空间中变化极大，包括视角变化、背景杂乱、光照变化、遮挡、形变、类内差异等。

本章的“进化树”可以概括为：

```text
图像分类问题
│
├── 线性代数基础：标量、向量、矩阵、范数、点积、矩阵乘法
│
├── 最近邻方法 KNN
│ ├── 直接记忆训练集
│ ├── 用距离度量比较图像
│ ├── 从 1-NN 到 K-NN：多数投票降低噪声敏感性
│ └── 问题：像素距离不可靠，测试慢，不能真正学习特征
│
├── 参数化模型：线性分类器
│ ├── 定义 score function: f(x; W,b)
│ ├── 用 loss function 衡量错误
│ └── 通过优化学习 W,b
│
├── 损失函数
│ ├── Multiclass SVM loss：强调 margin
│ ├── Softmax + Cross Entropy：输出概率解释
│ └── Regularization：控制模型复杂度，减少过拟合
│
└── 优化与自动微分
 ├── Random search 太慢
 ├── Gradient descent 沿负梯度下降
 ├── SGD 用 mini-batch 近似全数据梯度
 └── Computation graph / Autograd 用链式法则自动求梯度
```

重要程度上，期末最核心的是：KNN 的局限、训练/验证/测试集划分、线性分类器形式、SVM loss、Softmax cross-entropy、regularization、gradient descent、computation graph/backprop。

---

= 1. 线性代数基础

== 1.1 标量 Scalars 【重要】

标量是单个数，例如 $a,b,c in R R$。基本操作包括：

$ c = a + b  c = a dot b  c = sin a $

标量长度，也就是绝对值：

$ |a| = cases(a,, a > 0, -a,, "otherwise".) $

性质：

$ |a+b| <= |a| + |b|  |a dot b| = |a| dot |b| $

其中 $|a+b| <= |a|+|b|$ 是三角不等式，直觉是“先走 $a$ 再走 $b$ 的总距离不会小于合并后的净位移”。

---

== 1.2 向量 Vectors 【重要】

向量是多个标量组成的有序数组：

$ bold(x) =
mat(x_1; x_2; .. ; x_d) in RR^d $

其中 $d$ 是维度，$x_i$ 是第 $i$ 个分量。

向量逐元素加法：

$ bold(c) = bold(a) + bold(b), quad quad c_i = a_i + b_i $

向量逐元素乘法：

$ bold(c) = bold(a) dot bold(b), quad quad c_i = a_i b_i $

逐元素非线性函数：

$ bold(c) = sin bold(a), quad quad c_i = sin(a_i) $

向量的欧氏范数：

$ |bold(x)| = sqrt{sum_i x_i^2} $

这里 $|bold(x)|$ 表示向量长度。机器学习中它常用于衡量特征大小、权重大小、距离大小。

向量范数满足：

$ |bold(a)+bold(b)| <= |bold(a)|+|bold(b)| $

点积：

$ bold(a)^top bold(b) = sum_i a_i b_i $

如果两个向量正交，则：

$ bold(a)^top bold(b) = 0 $

点积的几何意义是衡量两个方向的对齐程度。如果点积大于 0，方向大致一致；小于 0，方向相反；等于 0，方向正交。

---

== 1.3 矩阵 Matrices 【重要】

矩阵是二维数组：

$ A in R R^{m times n} $

其中 $A_{{"ij"}}$ 表示第 $i$ 行、第 $j$ 列元素。

矩阵加法：

$ C = A + B, quad quad C_{{"ij"}} = A_{{"ij"}} + B_{{"ij"}} $

标量乘矩阵：

$ C = a B, quad quad C_{{"ij"}} = a B_{{"ij"}} $

逐元素非线性函数：

$ C = sin A, quad quad C_{{"ij"}} = sin(A_{{"ij"}}) $

矩阵-向量乘法：

$ bold(c) = A bold(b), quad quad c_i = sum_j A_{{"ij"}} b_j $

其中 $A in R R^{m times n}$，$bold(b) in R R^n$，所以 $bold(c) in R R^m$。

矩阵-矩阵乘法：

$ C = A B, quad quad C_{{"ik"}} = sum_j A_{{"ij"}}B_{{"jk"}} $

其中 $A in R R^{m times n}$，$B in R R^{n times p}$，所以 $C in R R^{m times p}$。

Frobenius norm：

$ |A|_{upright("Frob")} =
[
sum_{i,j} A_{{"ij"}}^2
]^{1/2} $

性质：

$ |A+B|_{upright("Frob")}
<=
|A|_{upright("Frob")} + |B|_{upright("Frob")}  |a A|_{upright("Frob")}
=

|a| dot |A|_{upright("Frob")} $

Frobenius norm 可以理解为“把矩阵拉平成向量后的欧氏长度”。

---

= 2. 图像分类问题

== 2.1 为什么图像分类不是手写规则问题？【重要】

课件强调：不像排序数字那样可以直接写一个确定规则，识别猫没有显然的 hard-coded algorithm。原因是图像分类面对的是高维、连续、变化复杂的像素空间。

典型挑战：

| 挑战 | 含义 | 为什么困难 |
| -------------------- | ------ | -------------- |
| Viewpoint variation | 相机角度改变 | 所有像素都会变化 |
| Background clutter | 背景杂乱 | 背景像素可能比物体本身更显著 |
| Illumination | 光照变化 | 同一物体颜色/亮度显著改变 |
| Occlusion | 遮挡 | 目标只露出一部分 |
| Deformation | 形变 | 猫的姿态变化大 |
| Intraclass variation | 类内差异 | 同为 cat，但外观差异巨大 |

标准机器学习流程是：

```text
收集图像和标签
→ 提取有判别力的特征
→ 使用机器学习算法训练分类器
→ 在新图像上评估分类器
```

---

= 3. K Nearest Neighbor

== 3.1 最近邻分类器 1-NN 【重要】

Nearest Neighbor 的思想是：训练阶段不真正学习参数，而是记住全部训练数据和标签；测试时，找到与 query image 最相似的训练图像，并复制它的标签。

给定训练集：

$ cal(D)_{upright("train")} = {bold(x)_i, y_i}_{i=1}^N $

其中 $bold(x)_i$ 是第 $i$ 张训练图像的特征向量，$y_i$ 是标签。对于测试样本 $bold(x)$，1-NN 预测为：

$ i^* = op("argmin")_i d(bold(x), bold(x)_i) $

$ hat(y) = y_{i^*} $

其中 $d(bold(x), bold(x)_i)$ 是距离函数。

常见 L1 距离： $ d_1(bold(x), bold(x)_i) = sum_j |x_j - x_{{"ij"}}| $。其中 $x_j$ 是测试图像第 $j$ 个像素/特征，$x_{{"ij"}}$ 是第 $i$ 个训练样本的第 $j$ 个像素/特征。

直觉：L1 距离把所有维度上的绝对差相加，衡量两张图在像素空间中相差多少。

---

== 3.2 从 1-N N 到 K-N N 【重要】

1-N N 的问题是对噪声非常敏感。如果最近的那个点恰好是异常样本，预测会错。因此 K N N 改成看最近的 $K$ 个样本，并用多数投票决定类别。

定义最近邻集合： $ cal(N)_K(bold(x)) = "the set of " K " nearest training examples to " bold(x) $
预测： $ hat(y) = op("argmax")_c sum_{i in cal(N)_K(bold(x))} bold(1)(y_i = c) $
其中 $c$ 是类别，$bold(1)(dot)$ 是指示函数。如果条件为真则为 1，否则为 0。

KNN 的变化逻辑：

| 方法 | How it works | Why it fails | How next method fixes |
| -------------------- | ------------------------ | ------------- | --------------------- |
| 1-NN | 复制最近训练样本标签 | 对噪声和异常点敏感 | KNN 用多个邻居投票 |
| KNN | 最近 $K$ 个样本多数投票 | 像素距离不表达语义，测试慢 | 引入特征提取和参数化模型 |
| Feature + classifier | 先抽取 HoG/LBP/SIFT 等特征，再分类 | 手工特征有限 | 后续深度学习自动学习特征 |

---

== 3.3 超参数选择：Train / Validation / Test 【非常重要】

KNN 有两个关键超参数：

$ K $

和距离度量：

$ d(dot, dot) $

课件强调：超参数不能直接在训练集或测试集上选。

错误做法 1：在训练集上选最优超参数。
问题：$K=1$ 在训练集上通常完美，因为每个训练点最近的就是自己，所以会过拟合。

错误做法 2：在测试集上选最优超参数。
问题：测试集被用于调参后，就不再是“未见数据”，最终 test accuracy 会被污染，无法估计真实泛化性能。

正确做法：

```text
Train set: 用于训练模型
Validation set: 用于选择超参数
Test set: 只在最后评估一次
```

Cross-validation：

把训练数据分成多个 fold，每次用其中一个 fold 做 validation，其余 fold 做 training，最后平均验证结果。课件指出 cross-validation 对小数据集有用，但在 deep learning 中不常用，因为训练成本太高。

---

== 3.4 KNN 为什么在图像分类中基本不用？【非常重要】

课件结论很明确：KNN with pixel distance never used。原因有两个。

第一，pixel distance 不具有语义信息。图像被遮挡、平移、打乱、换色后，像素距离可能巨大或很小，但语义不一定对应变化。

第二，测试时太慢。KNN 几乎没有训练成本，但测试时每个测试样本都要和大量训练样本比较。如果训练集有 $N$ 个样本，每个样本维度是 $D$，一次预测复杂度约为：

$ O(N D) $

所以 KNN 的根本局限是：它没有学到一个抽象决策规则，只是在高维像素空间做记忆检索。

---

= 4. 从 KNN 到线性分类器

== 4.1 机器学习的典型框架 【非常重要】

KNN 框架：

```text
Memorize all data and labels
→ Predict label of the most similar training image
```

线性分类器框架：

```text
Define a model f_W(x)
→ Define a loss L(f_W(x), y)
→ Solve W* minimi in g L
→ Predict using f_{W*}(x)
```

这就是本章最关键的方法论转变：从“存储数据”变成“学习参数”。

---

== 4.2 线性分类器模型 【非常重要】

给定图像 $I in R R^{H times W times C}$，例如 CIFAR-10 中常见的 $32 times 32 times 3$ 图像，可以将其 vectorize 成：

$ bold(x) in R R^{D} $

其中：

$ D = H W C $

线性分类器输出每个类别的 score：

$ bold(s) = f(bold(x); W, bold(b)) = W bold(x) + bold(b) $

其中：

$ W in RR^{C times D} bold(b) in RR^{C} bold(s) in RR^{C} C $。是类别数，例如 CIFAR-10 中 $C=10$。$s_j$ 是第 $j$ 类的分数。预测类别为：

$ hat(y) = op("argmax")_j s_j $

也可以通过特征增广去掉 bias。令：

$ tilde(bold(x))
=

mat(bold(x); 1) tilde(W) = mat(W, bold(b)) $

则：

$ bold(s)
=

tilde(W)tilde(bold(x)) $

这样 bias 被吸收到权重矩阵中，公式更简洁。

---

= 5. Multiclass SVM Loss

== 5.1 SVM loss 的核心思想 【非常重要】

SVM loss 不要求正确类别分数只比错误类别高一点，而是要求至少高出一个 margin。课件中 margin 设为 1。

对第 $i$ 个样本：

$ bold(s) = f(bold(x)_i; W) in R R^{C} $

其中 $s_j$ 是类别 $j$ 的 score，$y_i$ 是正确类别。Multiclass SVM loss：

$ L_i = sum_{j != y_i} max(0, s_j - s_{y_i} + 1) $

总损失：

$ L = frac(1, N) sum_{i=1}^{N} L_i $

符号解释：

$N$：训练样本数。
$C$：类别数。
$bold(x)*i$：第 $i$ 个输入样本。
$y_i$：第 $i$ 个样本的正确标签。
$s_j$：模型对类别 $j$ 的打分。
$s*{y_i}$：模型对正确类别的打分。
$1$：margin，表示希望正确类别至少比错误类别高 1。
$max(0, dot)$ ：hinge function，只惩罚违反 margin 的类别。

---

== 5.2 S V M loss 的逻辑直觉

对每个错误类别 $j != y_i$，希望满足： $ s_{y_i} >= s_j + 1 $
等价于： $ s_j - s_{y_i} + 1 <= 0 $
如果满足，说明正确类别已经比错误类别高出足够 margin，该类别贡献的 loss 为 0： $ max(0, s_j - s_{y_i} + 1) = 0 $
如果不满足，说明错误类别分数太高，loss 为： $ s_j - s_{y_i} + 1 $
这意味着 S V M 关心的是“相对分数差距”，不是绝对分数。

---

== 5.3 课件例子推导 【非常重要】

课件给出三个类别 cat/car/frog，三个样本的 score 表：

| 样本 | cat | car | frog | true label |
| -- | --: | --: | ---: | ---------- |
| 1 | 3.2 | 5.1 | -1.7 | cat |
| 2 | 1.3 | 4.9 | 2.0 | car |
| 3 | 2.2 | 2.5 | -3.1 | frog |

样本 1 正确类别为 cat，$s_{y}=3.2$：$ L_1 = max(0, 5.1 - 3.2 + 1) + max(0, -1.7 - 3.2 + 1) = max(0, 2.9) + max(0, -3.9) = 2.9 + 0 = 2.9 $
样本 2 正确类别为 car，$s_y=4.9$：$ L_2 = max(0, 1.3 - 4.9 + 1) + max(0, 2.0 - 4.9 + 1) = max(0, -2.6) + max(0, -1.9) = 0 $
样本 3 正确类别为 frog，$s_y=-3.1$：$ L_3 = max(0, 2.2 - (-3.1) + 1) + max(0, 2.5 - (-3.1) + 1) = max(0, 6.3) + max(0, 6.6) = 12.9 $
平均 loss：$ L = frac(2.9 + 0 + 12.9, 3) = 5.27 $
这个例子非常适合考试，因为它完整体现了 S V M loss 的计算方式。

---

= 6. Regularization

== 6.1 为什么最优 $W$ 不唯一？【非常重要】

线性分类器： $ f(bold(x); W) = W bold(x) $
S V M loss： $ L = frac(1, N) sum_{i=1}^{N} sum_{j != y_i} max(0, f(bold(x)_i;W)_j - f(bold(x)_i;W)_{y_i} + 1) $
如果存在一个 $W$ 使得 $L=0$，那么放大权重 $n W$ 也可能仍然让 loss 为 0。原因是： $ f(bold(x); n W) = n W bold(x) = n f(bold(x); W) $
代入 margin 项：
$ n W_j^top bold(x)_i - n W_{y_i}^top bold(x)_i + 1 = n ( W_j^top bold(x)_i - W_{y_i}^top bold(x)_i + frac(1, n) ) $
随着 $n$ 增大，等效 margin $frac(1, n)$ 变小，模型可能过分放大权重，导致过拟合。

---

== 6.2 正则化目标函数 【非常重要】

为了解决参数不唯一和过拟合问题，引入 regularization： $ L = frac(1, N) sum_{i=1}^{N} L_i(bold(x)_i; W, y_i) + lambda R(W) $
其中：

$L_i$ 是 data loss，要求模型预测匹配训练数据。
$R(W)$ 是 regularization term，表达对模型复杂度的偏好。
$lambda$ 是正则化强度，控制 data fitting 和 model simplicity 的 trade-off。

常见 L2 regularization： $ R(W) = |W|_2^2 = sum_k W_k^2 $
常见 L1 regularization： $ R(W) = |W|_1 = sum_k |W_k| $
课件强调正则化的两个直觉。第一是 maximum margin principle：在 S V M 中，较小的 $|W|$ 对应较大的分类间隔，通常泛化更好。第二是 controlling weights：如果某个 $W_k$ 很大，输入特征 $x_k$ 的微小变化会导致输出剧烈变化；限制权重可以让模型更平滑、更不依赖单一噪声特征。

Occam’s Razor：如果两个模型都能解释数据，选择更简单的模型。

---

= 7. Softmax Classifier 与 Cross Entropy

== 7.1 为什么需要 Softmax？【非常重要】

S V M score 不是概率，无法解释为“模型认为该类的概率是多少”。Softmax 把任意实数 score 转成非负、和为 1 的概率分布。

给定 score： $ bold(s) = f(bold(x)_i; W) $
Softmax probability： $ q_j = frac(e^{s_j}, sum_k e^{s_k}) $
其中：

$q_j$：模型预测样本属于类别 $j$ 的概率。
$s_j$：类别 $j$ 的 score。
$e^{s_j}$：指数函数保证非负。
$sum_k e^{s_k}$：归一化项，保证概率和为 1。

性质： $ q_j >= 0, sum_j q_j = 1 $
---

== 7.2 Cross Entropy Loss 【非常重要】

真实标签可以表示成 one-hot distribution： $ p_j = cases(1,, j = y_i, 0,, j != y_i.) $
交叉熵： $ H(p,q) = -sum_j p_j log q_j $
因为 $p_j$ 是 one-hot，所以只剩正确类别： $ L_i = -log q_{y_i} $
代入 Softmax： $ L_i = -log frac(e^{s_{y_i}}, sum_j e^{s_j}) $
整个数据集： $ L = -frac(1, N) sum_{i=1}^{N} log frac(e^{s_{y_i}}, sum_j e^{s_j}) $
也可以写成： $ L_i = -s_{y_i} + log sum_j e^{s_j} $
这个形式常用于推导梯度。

---

== 7.3 Cross Entropy 的取值范围与初始化检查 【重要】

当模型完全正确，并且正确类概率趋近 1： $ q_{y_i} -> 1 $
则： $ L_i = -log 1 = 0 $。所以最小值是 0。

当正确类概率趋近 0： $ q_{y_i} -> 0 $
则： $ L_i = -log 0 -> oo $
所以最大值无上界。

如果初始化时所有 score 近似相等，假设有 $C$ 个类别： $ q_j = frac(1, C) $
则： $ L_i = -log frac(1, C) = log C $
如果 $C=10$： $ L_i = log 10 approx 2.3 $
这是 debug 模型时非常重要的 sanity check。课件也特别提醒要记住这个值。

---

== 7.4 S V M vs Softmax 对比 【非常重要】

| 维度 | S V M Loss | Softmax Cross Entropy |
| ---- | --------------------------------------- | ----------------------------------------- |
| 输出解释 | score，无概率解释 | probability distribution |
| 核心目标 | 正确类分数比错误类至少高 margin | 最大化正确类概率 |
| 损失形式 | hinge loss | negative log-likelihood |
| 关注重点 | margin violation | probability calibration |
| 典型公式 | $sum_{j != y_i}max(0,s_j-s_{y_i}+1)$ | $-log(frac(e^{s_{y_i}}, sum_j e^{s_j}))$ |
| 直觉 | 分开即可，超过 margin 后不再惩罚 | 正确类概率越接近 1 越好 |

S V M 是“只要分数差够大就满意”；Softmax 是“希望正确类别概率尽可能大”。

---

= 8. Optimization: 从 Random Search 到 Gradient Descent

== 8.1 Random Search 为什么失败？【重要】

随机搜索 $W$ 的方法是：随机生成很多权重，选择 loss 最小的那个。课件中 random search 在某任务上只有 15.5% accuracy，而 S O T A 约为 99.7%，说明随机搜索在高维参数空间中极其低效。

原因是 $W$ 的维度极高，随机猜中好参数的概率几乎为零。必须利用 loss function 的局部几何信息，也就是 gradient。

---

== 8.2 Gradient Descent 【非常重要】

梯度下降核心更新公式： $ W ← W - lambda frac(partial L, partial W) $ 其中：

$W$：模型参数。
$L$：loss function。
$frac(partial L, partial W)$：loss 对参数的梯度。
$lambda$：learning rate / step size。

梯度方向是函数上升最快方向，所以负梯度方向是下降最快方向。

如果： $ L = frac(1, N) sum_{i=1}^{N} L_i $
则： $ frac(partial L, partial W) = frac(1, N) sum_{i=1}^{N} frac(partial L_i, partial W) $
所以更新为： $ W ← W - lambda frac(1, N) sum_{i=1}^{N} frac(partial L_i, partial W) $
---

== 8.3 S V M loss 的梯度 【非常重要】

对单个样本： $ L_i = sum_{j != y_i} max(0, W_j^top bold(x)_i - W_{y_i}^top bold(x)_i + 1) $
其中 $W_j$ 是权重矩阵中对应类别 $j$ 的行向量。

对某个错误类别 $j$，定义 margin violation： $ m_j = W_j^top bold(x)_i - W_{y_i}^top bold(x)_i + 1 $
如果 $ m_j <= 0 $： $ max(0, m_j) = 0 $，梯度为 0。
如果 $ m_j > 0 $：该项为 $ W_j^top bold(x)_i - W_{y_i}^top bold(x)_i + 1 $
对错误类别权重： $ frac(partial L_i, partial W_j) = bold(x)_i $
对正确类别权重： $ frac(partial L_i, partial W_{y_i}) = -bold(x)_i $
如果有多个错误类别违反 margin，正确类别的梯度会累加： $ frac(partial L_i, partial W_{y_i}) = -K_i bold(x)_i $
其中 $K_i$ 是违反 margin 的错误类别数量。

直觉：如果错误类别分数太高，就降低错误类别权重对 $bold(x)_i$ 的响应，同时提高正确类别权重对 $bold(x)_i$ 的响应。

---

= 9. Mean Squared Error 与 Linear Regression

课件也用 linear regression 解释 loss function 和 gradient descent。监督学习的核心是：找到使所有可能数据 error 最小的模型参数。但由于只能观察有限训练数据，实际优化的是 empirical error。

线性回归模型： $ hat(y) = m x + c $ 其中：

$m$：斜率。
$c$：截距。
$x$：输入。
$hat(y)$：预测值。
$y$：真实值。

Mean Squared Error： $ L = frac(1, N) sum_{i=1}^{N} (hat(y)_i - y_i)^2 = frac(1, N) sum_{i=1}^{N} (m x_i + c - y_i)^2 $
对 $m$ 的偏导： $ frac(partial L, partial m) = frac(2, N) sum_{i=1}^{N} (m x_i + c - y_i) x_i $
对 $c$ 的偏导： $ frac(partial L, partial c) = frac(2, N) sum_{i=1}^{N} (m x_i + c - y_i) $
更新： $ m ← m - lambda frac(partial L, partial m) $
$ c ← c - lambda frac(partial L, partial c) $
---

= 10. Gradient 的定义与数值梯度

== 10.1 一维导数 【重要】

一维情况下： $ frac(partial f(x), partial x) = lim_{h -> 0} frac(f(x+h)-f(x), h) $
导数表示函数在该点的瞬时变化率。

== 10.2 高维偏导 【重要】

对高维函数： $ f(a_1,..,a_i,..,a_n) $
第 $i$ 个变量的偏导为： $ frac(partial f, partial x_i) (a_1,..,a_n) = lim_{h -> 0} frac(f(a_1,..,a_i+h,..,a_n) - f(a_1,..,a_i,..,a_n), h) $
梯度是所有偏导数组成的向量： $ nabla f = mat(frac(partial f, partial x_1); frac(partial f, partial x_2); .. ; frac(partial f, partial x_n)) $
== 10.3 Numeric Gradient 的问题 【重要】

数值梯度近似： $ frac(partial f(x), partial x) approx frac(f(x+h)-f(x), h) $
问题：

第一，慢。每个参数都要单独 perturb 一次。
第二，只是近似。
第三，对 $h$ 很敏感。$h$ 太大近似不准，$h$ 太小可能有浮点误差。
第四，$h$ 本身也变成超参数。

所以实际训练深度模型时不用 numeric gradient，而是使用 analytic gradient / autograd。

---

= 11. Computation Graph 与 Reverse Accumulation

== 11.1 计算图的核心思想 【非常重要】

计算图把复杂函数拆成 primitive operations，并组织成 directed acyclic graph。Forward pass 计算输出并保存中间结果；Backward pass 按反向顺序用 chain rule 计算梯度。

课件例子： $ z = (⟨ bold(x), bold(w) ⟩ - y)^2 $
定义中间变量： $ a = ⟨ bold(x), bold(w) ⟩ $
$ b = a - y $
$ z = b^2 $
其中：

$bold(x)$：输入向量。
$bold(w)$：权重向量。
$y$：真实标签或目标值。
$a$：线性预测。
$b$：预测误差。
$z$：平方误差 loss。

---

== 11.2 反向传播推导 【非常重要】

先计算： $ frac(partial z, partial b) = 2b $ 因为： $ z = b^2 $
然后： $ frac(partial b, partial a) = 1 $
所以： $ frac(partial z, partial a) = frac(partial z, partial b) frac(partial b, partial a) = 2b $
又因为： $ a = ⟨ bold(x), bold(w) ⟩ = sum_i x_i w_i $
所以： $ frac(partial a, partial bold(w)) = bold(x) $
因此： $ frac(partial z, partial bold(w)) = frac(partial z, partial a) frac(partial a, partial bold(w)) = 2b bold(x) $
也就是： $ frac(partial z, partial bold(w)) = 2 (⟨ bold(x), bold(w) ⟩ - y) bold(x) $
这就是平方误差对线性模型权重的梯度。

课件的核心总结是：

```text
Build computation graph
→ Forward: evaluate graph and store intermediate results
→ Backward: evaluate graph in reverse order
→ Use chain rule
→ Eliminate unnecessary paths
```

---

= 12. Stochastic Gradient Descent

== 12.1 为什么需要 S G D？【非常重要】

Full-batch gradient descent： $ W ← W - lambda frac(partial L, partial W) $
其中： $ L = frac(1, N) sum_{i=1}^{N} L_i $
所以： $ W ← W - lambda frac(partial, partial W) (frac(1, N) sum_{i=1}^{N} L_i) $
当 $N$ 很大时，每次更新都遍历全部训练集，计算非常昂贵。

S G D 用 mini-batch 近似： $ W ← W - lambda frac(partial, partial W) (frac(1, B) sum_{i=1}^{B} L_i) $
其中 $B$ 是 batch size。

$B$ 的选择受显存、训练速度、梯度噪声影响。较小 batch 更新更频繁但噪声更大；较大 batch 梯度更稳定但计算和显存成本更高。

---

= 13. 本章算法演进总表

| 阶段 | 方法 | 核心公式/机制 | 解决什么 | 局限 |
| -- | -------------------- | ----------------------------------------------------- | -------------------------- | ------------------ |
| 1 | Pixel representation | $I in RR^{H times W times C}$ | 把图像转成计算机可处理形式 | 像素不等于语义 |
| 2 | 1-N N | $hat(y)=y_{op("argmin")_i d(x,x_i)}$ | 简单分类，无需训练 | 对噪声敏感，测试慢 |
| 3 | K N N | 多数投票 | 降低单点噪声影响 | pixel distance 不可靠 |
| 4 | Feature extraction | HoG/L B P/S I F T | 提升表示质量 | 手工特征有限 |
| 5 | Linear classifier | $s=W x + b$ | 学习参数化决策边界 | 需要定义 loss |
| 6 | S V M loss | $sum_{j != y_i} max(0, s_j - s_{y_i} + 1)$ | 学习 margin-based classifier | score 无概率解释 |
| 7 | Softmax C E | $-log frac(e^{s_y}, sum_j e^{s_j})$ | 概率化分类 | 仍可能过拟合 |
| 8 | Regularization | $L+lambda R(W)$ | 控制复杂度 | $lambda$ 需调参 |
| 9 | Gradient descent | $W ← W-lambda frac(partial L, partial W)$ | 高效优化参数 | 全量梯度成本高 |
| 10 | S G D | mini-batch gradient | 加速大规模训练 | 梯度有噪声 |
| 11 | Computation graph | chain rule | 自动求导 | 需要保存中间值，占内存 |

---

= 14. 期末复习重点排序

== 必须掌握

+ K N N 的训练/测试流程，以及为什么 K N N 不适合图像分类。
+ Train / validation / test 的区别，为什么不能用 test set 调超参数。
+ 线性分类器公式： $ bold(s)=W bold(x)+bold(b) $
+ Multiclass S V M loss： $ L_i = sum_{j != y_i} max(0, s_j - s_{y_i} + 1) $
+ Softmax probability： $ q_j = frac(e^{s_j}, sum_k e^{s_k}) $
+ Cross-entropy loss： $ L_i = -log q_{y_i} $
+ 正则化目标： $ L = frac(1, N) sum_i L_i + lambda R(W) $
+ 梯度下降： $ W ← W - lambda frac(partial L, partial W) $
+ Computation graph 的 forward/backward 和 chain rule。

== 很可能考计算

+ 给 score 表，计算 S V M loss。
+ 给类别数 $C$，问初始化 softmax loss： $ L=log C $
+ 推导 M S E 对 $m,c$ 的梯度。
+ 推导简单计算图： $ z = (⟨ x,w ⟩ - y)^2 $

对 $w$ 的梯度。

== 概念题高频

+ 为什么 pixel distance 不好？
+ 为什么 regularization 可以减少 overfitting？
+ SVM 和 Softmax 的区别。
+ SGD 为什么比 full gradient descent 更实用？
+ Numeric gradient 为什么不用来训练大模型？
+ Cross-validation 为什么在小数据集有用但 deep learning 中不常用？
