// @title: Week 10 学习笔记：Generative Models, 3D Reconstruction and Frontiers
// @description:
// @order: 999

= Week 10 学习笔记：Generative Models, 3D Reconstruction and Frontiers

== 0. 本章核心目标与技术进化树

本章要解决的核心问题是：

如何从“理解图像”进一步走向“生成图像”和“重建三维世界”。

传统 CV 任务主要回答：

$ "image" -> "label / mask / box" $

而生成模型与 3D 重建要回答：

$ "noise / latent / multi-view images / text" -> "new image / 3D scene / controllable world" $

整体技术演化可以理解为：

```text
传统视觉理解
Classification / Segmentation / Detection
 ↓
像素级生成
Autoencoder → VAE → GAN → Diffusion / Stable Diffusion / ControlNet
 ↓
三维场景表示
SfM → NeRF → 3D Gaussian Splatting
 ↓
生成模型与 3D 融合
DreamFusion / World Models / Video Generative Models
```

本章的内在线索不是“列举模型”，而是不断解决同一个矛盾：

如何用可训练的神经网络表示复杂真实世界，同时保持生成质量、可控性、可优化性和计算效率。

---

= 1. CV 任务回顾：从图像理解到生成建模

== 1.1 传统视觉任务谱系

课件首先回顾了四类经典 CV 任务：

| 任务 | 输入 | 输出 | 空间信息 | 实例区分 |
| --------------------- | -- | ----------------- | ---- | ------ |
| Classification | 图像 | 类别标签 | 无 | 无 |
| Semantic Segmentation | 图像 | 每个像素的类别 | 有 | 无 |
| Object Detection | 图像 | 类别 + bounding box | 有 | 有，但粗粒度 |
| Instance Segmentation | 图像 | 每个实例的 mask | 有 | 有，细粒度 |

重要程度：★★★☆☆
这部分不是本章重点，但它说明了视觉任务从“分类”逐渐走向“空间结构建模”。

== 1.2 Fully Convolutional Network：语义分割的基本思想

FCN 的核心思想是：把分类 CNN 改成全卷积结构，使其能够输出空间分辨率对应的像素级预测。

基本结构：

```text
Input image
 ↓
Downsampling: pooling / strided convolution
 ↓
Low-resolution semantic features
 ↓
Upsampling
 ↓
Pixel-wise prediction: C × H × W
```

其中：

$ "Input" in R R^{3 times H times W}  "Output" in R R^{C times H times W} $

符号含义：

$H,W$ 是图像高度和宽度；$C$ 是语义类别数；输出张量中每个位置对应一个像素的类别分布。

重要程度：★★★☆☆

== 1.3 R-CNN 到 Fast R-CNN：检测中的计算瓶颈

Slow R-CNN 的流程是：

```text
Image
 ↓
Region Proposal ≈ 2000 RoIs
 ↓
Warp each RoI to 224×224
 ↓
Each RoI independently passes ConvNet
 ↓
SVM classification + bbox regression
```

问题是非常慢，因为每张图要做约 2000 次独立 CNN forward。

Fast R-CNN 的改进：

```text
Image
 ↓
Whole-image ConvNet only once
 ↓
Crop RoI features from feature map
 ↓
Per-RoI network
 ↓
Classification + box regression
```

本质改进是：

$ "crop image then C N N"
quad -> quad
"C N N image once then crop features" $

重要程度：★★★☆☆

== 1.4 Mask R-CNN：从检测到实例分割

Mask R-CNN 在 Faster/Fast R-CNN 系列基础上增加 mask branch。

对于每个 RoI，模型同时预测：

$ "classification loss"
+
"bounding-box regression loss"
+
"mask prediction loss" $

mask "head" 输出通常是：

$ 28 times 28 $

的二值 mask。

重要程度：★★★☆☆

== 1.5 Single-stage Detectors：YOLO / SSD / RetinaNet

单阶段检测器不再显式生成 proposal，而是把图像划分为网格并直接预测 boxes 和类别。

若图像被划分为：

$ 7 times 7 $

每个 grid cell 有 $B$ 个 base boxes，类别数为 $C$，则输出为：

$ 7 times 7 times (5B + C) $

其中每个 box 预测：

d_x, d_y, d_w, d_h, "confidence"

符号含义：

$d_x,d_y$ 表示中心位置偏移；$d_w,d_h$ 表示宽高修正；confidence 表示该 box 包含目标的置信度；$C$ 是类别数。

重要程度：★★★☆☆

== 1.6 DETR：用 Transformer 做检测

DETR 的核心思想是把 object detection 转化为 set prediction。

```text
CNN backbone
 ↓
Image features + positional encoding
 ↓
Transformer encoder-decoder
 ↓
Object queries
 ↓
Set of box predictions
 ↓
Bipartite matching loss
```

它去掉了 anchor、NMS、proposal 等人工设计，用 Hungarian matching 解决预测集合与真实目标集合之间的匹配问题。

重要程度：★★★☆☆

---

= 2. Autoencoder 到 VAE：从压缩表征到生成模型

== 2.1 Autoencoder 的基本目标

Autoencoder 是无监督特征学习方法：

```text
Input x
 ↓ Encoder
Latent feature z
 ↓ Decoder
Reconstruction x_hat
```

训练目标通常是重建损失：

$ L_("A E") = |x - hat(x)|^2 $

符号含义：

$x$ 是输入数据；$hat(x)$ 是 decoder 重建结果；$z$ 是 encoder 提取的 latent representation；$|dot|^2$ 是 L2 reconstruction loss。

物理直觉：如果 $z$ 能重建 $x$，说明 $z$ 保留了输入中最有用的信息。

重要程度：★★★★☆

== 2.2 传统 Autoencoder 的问题

AE 可以做降维、特征学习、辅助预训练，但它不是一个好的生成模型。

原因是：AE 学到的是离散、不规则、可能有空洞的 latent space。也就是说：

$ x >-> z $

只是把训练样本编码到 latent space 中，但没有保证 $z$ 的分布连续、平滑、可采样。

因此，如果你随机采样一个 latent vector：

$ z ~ ? $

decoder 未必能生成真实图像，因为这个 $z$ 可能落在训练样本 manifold 之外。

传统 AE 的失败点：

```text
Good for reconstruction
Bad for generation
```

Why it fails：

$ "latent space discontinuity"
=>
"interpolation / random sampling may decode unrealistic outputs" $

重要程度：★★★★★

---

= 3. VAE：用概率 latent space 修复 AE 的不可生成问题

== 3.1 VAE 的核心思想

VAE 是 Autoencoder 的概率化版本。传统 AE 输出一个确定性 latent vector：

$ z = f_phi(x) $

而 VAE 输出一个 latent distribution：

$ q_phi(z|x) $

通常假设：

$ q_phi(z|x) = cal(N)(mu(x), Sigma(x)) $

若使用对角协方差：

$ q_phi(z|x) = cal(N)(mu(x), op("diag")(sigma^2(x))) $

符号含义：

$q_phi(z|x)$ 是 encoder 近似出来的后验分布；$mu(x)$ 是均值向量；$Sigma(x)$ 是协方差矩阵；$sigma^2(x)$ 是每个 latent dimension 的方差；$phi$ 是 encoder 参数。

生成时：

$ z ~ p(z)  hat(x) = op("Decoder")_theta(z) $

通常选择 prior：

$ p(z) = cal(N)(0, I) $

重要程度：★★★★★

== 3.2 VAE 的目标函数

课件给出的 VAE loss 是：

$ L = |x - hat(x)|^2 + D_(upright(K L))(q(z|x)|p(z)) $

完整理解是：

第一项是 reconstruction loss，要求 decoder 能从 sampled latent $z$ 重建原图。

第二项是 KL regularization，要求 encoder 输出的 latent distribution 接近标准正态分布。

符号含义：

$x$ 是输入图像；$hat(x)$ 是重建图像；$q(z|x)$ 是给定 $x$ 后 encoder 输出的 latent distribution；$p(z)$ 是预设 prior，通常是 $cal(N)(0,I)$；$D_(upright(K L))$ 是 KL divergence，用于衡量两个概率分布之间的差异。

直觉：

$ "reconstruction term"
=>
"preserve information"  "K L term"
=>
"make latent space continuous and sampleable" $

所以 VAE 同时解决两个目标：

```text
reconstruct well + latent space regularized
```

重要程度：★★★★★

== 3.3 Reparameterization Trick

问题：如果直接从

$ z ~ q_phi(z|x) $

采样，采样操作不可直接反向传播到 $mu$ 和 $sigma$。

解决方法是 reparameterization trick：

$ z = mu + sigma op(op("odot")) epsilon $

其中：

$ epsilon ~ cal(N)(0, I) $

符号含义：

$mu$ 是 encoder 输出的均值；$sigma$ 是标准差；$epsilon$ 是从标准正态分布采样的噪声；$op("odot")$ 是逐元素乘法。

关键逻辑是把随机性从网络参数中“拆出来”：

$ z ~ cal(N)$mu,sigma^2

等价写成：

$ z = mu + sigma epsilon,quad epsilon ~ cal(N)(0,1) $

这样梯度可以通过 $z$ 回传到 $mu$ 和 $sigma$。

重要程度：★★★★★

== 3.4 VAE 的 KL divergence 公式

对于对角高斯：

$ q(z|x) = cal(N)$mu, op(diag)(sigma^2$ $

prior：

$ p(z)=cal(N)(0,I) $

KL divergence 为：

$ D_(upright(K L))(q(z|x)|p(z))
=

frac(1, 2)
sum_(j=1)^{d}
(
sigma_j^2 + mu_j^2 - 1 - log sigma_j^2
) $

有些课件写法等价为：

$ D_(upright(K L))(q(z|x)|p(z))
=

-frac(1, 2)
sum_(j=1)^{d}
(
1 + log sigma_j^2 - mu_j^2 - sigma_j^2
) $

符号含义：

$d$ 是 latent dimension；$mu_j$ 是第 $j$ 个 latent dimension 的均值；$sigma_j^2$ 是第 $j$ 个 latent dimension 的方差；$log sigma_j^2$ 是 log variance。

重要直觉：

当

$ mu_j = 0,quad sigma_j^2 = 1 $

时，该维度完全匹配标准正态分布，KL 为 0。

如果 encoder 想“作弊”，把不同样本压到彼此孤立的 cluster 中，KL 项会惩罚这种行为。

重要程度：★★★★★

== 3.5 VAE 为什么容易模糊？

课件强调 VAE 生成样本容易 blurry，原因主要有两个。

第一，数据分布通常是 multi-modal。例如 MNIST 有 0 到 9 十个模式，但 VAE 用一个 unit Gaussian prior 去压缩这些模式：

$ p(z)=cal(N)(0,I) $

这会导致不同模式之间被平滑连接，采样点可能落在类别之间。

第二，pixel-wise reconstruction loss 本身有 averaging effect。如果多个可能输出都合理，L2 loss 会倾向于预测平均图像：

$ hat(x) = op("argmin")_(hat(x)) bb(E)[|x-hat(x)|^2] $

这个最优解趋向条件均值，因此容易模糊。

重要程度：★★★★★

---

= 4. GAN：用对抗训练解决 VAE 的平均化问题

== 4.1 GAN 的生成逻辑

GAN 不要求重建某个具体样本，而是学习从随机噪声到真实数据分布的映射：

$ z ~ p_z(z)  hat(x) = G(z) $

其中 $G$ 是 generator。

问题是：我们不知道哪个 $z$ 应该对应哪个真实图像，所以不能用 reconstruction loss。

因此 GAN 的目标变成：

$ G(z) ~ p_("data")(x) $

也就是说，生成样本整体看起来像真实数据分布。

重要程度：★★★★★

== 4.2 GAN 的 Minimax Objective

GAN 有两个网络：

$G$：Generator，生成 fake samples。

$D$：Discriminator，判断输入是真实样本还是生成样本。

经典目标函数是：

$ min_G max_D V(D,G)
=

bb(E)_(x ~ p_("data")(x)}
[log D(x)]
+
bb(E)_(z ~ p_z(z))
[log(1-D(G(z)))] $

符号含义：

$x$ 是真实数据；$z$ 是随机噪声；$p_("data")$ 是真实数据分布；$p_z$ 是噪声分布，常用 Gaussian；$G(z)$ 是生成样本；$D(x)$ 是 discriminator 判断 $x$ 为真实样本的概率。

训练过程：

```text
Step 1: fix G, train D
Step 2: fix D, train G to fool D
```

对 $D$ 来说，它希望：

$ D(x) -> 1  D(G(z)) -> 0 $

对 $G$ 来说，它希望：

$ D(G(z)) -> 1 $

重要程度：★★★★★

== 4.3 GAN 相比 VAE 的改进与问题

GAN 的优点是生成图像通常更锐利，因为它不是最小化 pixel-wise L2，而是通过 discriminator 学习更高级的真实感判别。

但是 GAN 的主要问题是训练不稳定：

```text
Bad convergence
Mode collapse
Discriminator too strong / too weak
No explicit likelihood
```

与 VAE 对比：

| 模型 | 优点 | 问题 |
| --------- | ------------------------- | ---------------------- |
| AE | 重建简单，特征学习有效 | latent space 不连续，不适合生成 |
| VAE | latent space 连续，可采样，有概率解释 | 容易 blurry |
| GAN | 图像锐利，视觉质量高 | 训练不稳定，mode collapse |
| Diffusion | 训练稳定，质量高，多样性强 | 推理慢，计算量大 |

重要程度：★★★★★

---

= 5. CycleGAN 与 StyleGAN

== 5.1 CycleGAN

CycleGAN 解决的是 unpaired image-to-image translation。

例如真实照片域 $X$ 和 Van Gogh 风格域 $Y$。

它使用两个 generator：

$ G: X -> Y  F: Y -> X $

和 discriminator，用于判断生成图像是否属于目标域。

Cycle consistency loss：

$ L_("cyc")(G,F)
=

bb(E)_(x ~ p_("data")(x)}
[|F(G(x))-x|*1]
+
bb(E)_(y ~ p_("data")(y))
[|G(F(y))-y|_1] $

直觉：从 $X$ 翻译到 $Y$ 后，再翻译回 $X$，应该恢复原图。

重要程度：★★★☆☆

== 5.2 StyleGAN

课件只简要提到 StyleGAN。复习时需要知道其核心思想是把 latent code 映射到 style space，通过逐层 style modulation 控制图像属性。

重要程度：★★☆☆☆

---

= 6. Diffusion Model：把复杂生成拆成逐步去噪

== 6.1 Diffusion 的核心思想

Diffusion Model 分成两个过程：

Forward diffusion process：逐步向真实数据加入噪声，破坏结构。

Reverse diffusion process：从噪声开始，逐步去噪，重建数据。

核心思想：

$ "complex one-step generation"
quad -> quad
"many small denoising steps" $

也就是把复杂映射拆成多个可管理步骤。

重要程度：★★★★★

== 6.2 Forward Diffusion Process

课件给出的 forward process 类似：

$ x_1 = sqrt(1-beta_1) x_0 + sigma_1 epsilon  x_2 = sqrt(1-beta_2) x_1 + sigma_2 epsilon $

一般形式：

$ x_t = sqrt(1-beta_t) x_(t-1) + sigma_t epsilon $

其中：

$ epsilon ~ cal(N)(0,I) $

常见 DDPM 写法是：

$ q(x_t|x_(t-1))
=
cal(N)
(
x_t;
sqrt(1-beta_t) x_(t-1),
beta_t I
) $

符号含义：

$x_0$ 是真实图像；$x_t$ 是第 $t$ 步加噪后的图像；$beta_t$ 是 noise schedule；$epsilon$ 是 Gaussian noise；$T$ 是总扩散步数。

随着 $t$ 增大：

$ x_t -> cal(N)(0,I) $

重要程度：★★★★★

== 6.3 Reverse Diffusion Process

反向过程从噪声开始：

$ x_T ~ cal(N)(0,I) $

逐步恢复：

$ x_T -> x_(T-1) -> .. -> x_0 $

课件给出的反向更新形式为：

$ x_(t-1)
=
frac(1, sqrt(1-beta_t))
(
x_t
-
frac(beta_t, sqrt(1-bar(alpha)_t))
epsilon_theta(x_t,t)
)
+
sigma_t epsilon $

其中：

$ alpha_t = 1-beta_t  bar(alpha)_t = product_(s=1)^(t) alpha_s $

符号含义：

$epsilon_theta(x_t,t)$ 是神经网络预测的噪声；$theta$ 是网络参数；$sigma_t epsilon$ 是随机项，用于增加采样多样性；$bar(alpha)_t$ 是从第 1 步到第 $t$ 步的累计保留信号比例。

直觉：

$ x_t - "predicted noise"
=>
"cleaner sample" $

重要程度：★★★★★

---

= 7. Stable Diffusion 与 Latent Diffusion Model

== 7.1 为什么不直接在 Pixel Space 生成？

直接在像素空间生成图像计算量巨大。例如高分辨率图像：

$ H times W times 3 $

维度很高，扩散模型每一步都在这个空间中去噪，会非常慢。

Latent Diffusion 的解决方案是：

```text
Image x
 ↓ Autoencoder Encoder
Latent z
 ↓ Diffusion U-Net denoising in latent space
Denoised latent z_hat
 ↓ Autoencoder Decoder
Generated image
```

核心改进：

$ "pixel-space diffusion"
quad -> quad
"latent-space diffusion" $

重要程度：★★★★★

== 7.2 Stable Diffusion 的组成

Stable Diffusion / LDM 主要包含三部分：

第一，Autoencoder，把图像压缩到 latent space 并从 latent 重建图像。

第二，Denoising U-Net，在 latent space 中逐步预测噪声并去噪。

第三，Condition encoder，例如 text encoder，用 cross-attention 把 prompt 信息注入 U-Net。

Cross-attention 的作用是让 U-Net 在每一步去噪时“看见文本条件”。

重要程度：★★★★★

---

= 8. ControlNet：从文本控制到空间控制

== 8.1 Text Prompt 的局限

文本很难精确描述空间结构，例如：

```text
角色的具体姿态
房间的透视线
物体边缘
深度关系
```

因此仅靠 text prompt 很难实现严格的空间控制。

== 8.2 ControlNet 的核心机制

ControlNet 的思想是锁住 pretrained diffusion model，同时添加一个可训练的控制分支。

它可以输入视觉条件，例如：

| 条件 | 控制内容 |
| ------------ | ------- |
| OpenPose | 人体骨架和姿态 |
| Canny Edge | 边缘和轮廓 |
| Depth Map | 三维空间布局 |
| Semantic Map | 语义区域结构 |

课件提到 Zero Convolution：

$ y_c = y + op("ControlNet")(x) $

Zero Conv 是 $1times 1$ convolution，权重和 bias 初始化为 0。

初始训练时：

$ op("ControlNet")(x)=0 $

所以：

$ y_c = y $

这保证一开始不会破坏原始 diffusion model 的能力。

同时，即使权重初始化为 0，梯度仍然可以更新，因此 ControlNet 可以逐渐学习控制信号。

重要程度：★★★★★

---

= 9. 3D Reconstruction：从多视角图像到三维场景

== 9.1 NeRF 的问题定义

NeRF 要解决的问题是：

给定一组多视角图片，学习一个连续的三维场景表示，并从新视角渲染图像。

NeRF 用神经网络表示一个 radiance field：

$ F_theta: (x,y,z,theta,phi) -> (r,g,b,sigma) $

其中：

$(x,y,z)$ 是三维空间点坐标；$ theta,phi $ 是 viewing direction；$(r,g,b)$ 是颜色；$sigma$ 是 volume density。

重要程度：★★★★★

== 9.2 NeRF 的基本流程

```text
Camera ray
 ↓
Sample points along ray
 ↓
MLP predicts color c and density sigma
 ↓
Volume rendering accumulates colors
 ↓
Compare rendered pixel with ground truth pixel
 ↓
Optimize MLP
```

核心是 differentiable volume rendering。

== 9.3 Volume Rendering 公式

对于一条 ray：

$ bold(r)(t)=bold(o)+t bold(d) $

其中：

$bold(o)$ 是 camera origin；$bold(d)$ 是 ray direction；$t$ 是沿射线的深度参数；$bold(r)(t)$ 是射线上第 $t$ 个三维点。

NeRF 预测：

$ sigma(bold(r)(t)) -> [0,oo)  bold(c)(bold(r)(t), bold(d)) -> (r,g,b) $

体渲染的连续形式为：

$ C(bold(r))
=
integral_(t_n)^(t_f)
T(t) sigma(bold(r)(t))
bold(c)(bold(r)(t), bold(d))
d t $

其中 transmittance：

$ T(t)
=
exp
(
-integral_(t_n)^(t)
sigma(bold(r)(s)) d s
) $

符号含义：

$C(bold(r))$ 是该 ray 渲染出的像素颜色；$t_n,t_f$ 是 near 和 far bounds；$T(t)$ 表示光线从相机到 $t$ 位置之前没有被遮挡的概率；$sigma$ 是该点处的密度；$bold(c)$ 是该点颜色。

直觉： $
"final color"
=

sum
"point color"
times
"point opacity"
times
"visibility before it"
$ 重要程度：★★★★★

== 9.4 NeR F 的优点与瓶颈

优点：高质量、连续场景表示、新视角合成效果好。

瓶颈：渲染慢，因为每条 ray 都要采样很多点并跑 M L P。

```text
High quality
but
painfully slow due to dense ray sampling
```

重要程度：★★★★★

---

= 10. 3D Gaussian Splatting：显式表示 + 实时渲染

== 10.1 为什么需要 3D G S？

课件对比了多种 3D 表示：

| 表示 | 优点 | 问题 |
| ----------- | ------------ | ------------------- |
| Point Cloud | 简单、易编辑 | 稀疏，视觉质量差 |
| Voxel | 规则结构 | 内存巨大 |
| Mesh | 连续表面 | 固定拓扑，优化困难 |
| S D F | 拓扑灵活 | 高频纹理困难 |
| NeR F | 高质量 | 渲染慢 |
| 3D G S | 高质量、显式、实时、可微 | 需要大量 Gaussian 和优化策略 |

3D G S 的核心优势：

```text
Real-time high-fidelity rendering
Explicit representation
Differentiable optimization
```

重要程度：★★★★★

== 10.2 3D G S Pipeline

完整流程：

```text
Multi-view images
 ↓
Structure-from-Motion
 ↓
Camera poses + sparse point cloud
 ↓
Initialize 3D Gaussians
 ↓
Optimize Gaussian parameters
 ↓
Adaptive densification / pruning
 ↓
Tile-based rasterization
 ↓
Real-time rendering
```

重要程度：★★★★★

---

= 11. SfM：3D G S 的初始化来源

== 11.1 SfM 的目标

Structure-from-Motion 的目标是从多张 2D 图像中估计： $
K, R, t
$ 以及 sparse 3D points。

其中：

$K$ 是 camera intrinsic matrix；$R$ 是 camera rotation；$t$ 是 camera translation。

最终输出：

```text
Camera poses + sparse point cloud
```

这是 3D G S 的初始 skeleton。

重要程度：★★★★☆

== 11.2 SfM 的流程

```text
Feature extraction
 ↓
Feature matching
 ↓
Geometric verification
 ↓
Image registration
 ↓
Triangulation
 ↓
Bundle adjustment
```

Feature matching 使用 S I F T 等局部特征。问题是会有大量 outliers。

Geometric verification 使用 epipolar geometry 和 R A N S A C 过滤错误匹配。

PnP 用已有 3D points 和新图像中的 2D observations 估计新相机位姿。

Triangulation 用多视角射线交会估计 3D 点。

Bundle Adjustment 联合优化所有 camera poses 和 3D points，使 reprojection error 最小。

B A 目标： $
min_({R_i,t_i},{X_j})
sum_(i,j)
|
bold(u)_("ij")
-
pi(K_i,R_i,t_i,X_j)
|^2
$ 符号含义：

$bold(u)_("ij")$ 是第 $i$ 张图中第 $j$ 个 3D 点的观测像素；$X_j$ 是第 $j$ 个 3D 点；$pi(dot)$ 是相机投影函数；$R_i,t_i,K_i$ 是第 $i$ 个相机的外参和内参。

重要程度：★★★★☆

---

= 12. 3D Gaussian Primitive

== 12.1 1D Gaussian

一维高斯分布：

$ f(x)
=

frac(1, sigma sqrt(2pi))
exp
(
-frac((x-mu)^2, 2 sigma^2)
) $

符号含义：

$mu$ 是均值，决定中心位置；$sigma^2$ 是方差，决定分布宽度；$sigma$ 是标准差。

重要程度：★★★★☆

== 12.2 3D Gaussian

三维高斯分布由 mean vector 和 covariance matrix 控制：

$ cal(G)(x)
=

exp
(
-frac(1, 2)
(x-mu)^T
Sigma^{-1}
$x-mu$
) $

其中：

$x in R R^3$ 是空间点；$mu in R R^3$ 是 Gaussian 中心；$Sigma in R R^{3 times 3}$ 是 covariance matrix，控制 ellipsoid 的大小、形状和方向。

3DGS 中每个 Gaussian primitive 包含：

mu, Sigma, alpha, "SH"

其中：

$mu$ 是位置；$Sigma$ 是形状与方向；$alpha$ 是 opacity；SH 是 spherical harmonics coefficients，用于 view-dependent color。

重要程度：★★★★★

---

= 13. 3DGS 中的颜色：Spherical Harmonics

如果每个 Gaussian 只存一个固定 RGB：

$ (r,g,b) $

那么从任何视角看颜色都一样。

但真实世界中存在高光、反射和视角相关颜色，因此颜色应该是 viewing direction 的函数：

$ bold(c) = bold(c)(bold(d)) $

Spherical Harmonics 将方向相关颜色分解为基函数加权和：

$ bold(c)(bold(d))
=
sum_(ell=0)^(L)
sum_(m=-ell)^("ell")
bold(k)_(ell m)
Y_(ell m)(bold(d)) $

符号含义：

$bold(d)$ 是 viewing direction；$Y_(ell m)$ 是 spherical harmonics basis function；$bold(k)_(ell m)$ 是每个 basis 的 RGB coefficient；$L$ 是最高阶数。

课件提到 3DGS 通常使用 degree 3，因此每个颜色通道需要：

$ (3+1)^2 = 16 $

个系数，RGB 共：

$ 16 times 3 = 48 $

个参数。

重要程度：★★★★★

---

= 14. Covariance Optimization：为什么要分解？

== 14.1 问题：直接优化协方差矩阵会失效

有效的 covariance matrix 必须满足：

$ Sigma >= 0 $

即 positive semi-definite。

如果直接用梯度下降更新：

$ Sigma_("new")
=

== Sigma_("old")

eta
nabla_Sigma L $

不保证更新后的 $Sigma_("new")$ 仍然是合法 covariance matrix。

这会导致矩阵不可逆、渲染异常或程序崩溃。

重要程度：★★★★★

== 14.2 解决：用 scale 和 rotation 分解 covariance

3DGS 使用分解形式保证合法性：

$ Sigma = R S S^T R^T $

常见写法是：

$ Sigma = R S^2 R^T $

其中：

$S$ 是 scale matrix，通常为对角矩阵；$R$ 是 rotation matrix；$Sigma$ 是最终 covariance matrix。

只要 scale 为正，$R$ 是合法旋转矩阵，则 $Sigma$ 必然是 positive semi-definite。

由于直接优化 rotation matrix 容易出现问题，例如 gimbal lock，3DGS 使用 quaternion 表示旋转。

重要程度：★★★★★

---

= 15. 3D Gaussian Optimization

3DGS 优化的参数包括：

$ mu, S, R, alpha, "S H" $

其中：

$mu$ 是 Gaussian 中心位置；$S$ 是 scale；$R$ 是 rotation；$alpha$ 是 opacity；SH 是 spherical harmonics color coefficients。

训练目标通常是让渲染图像接近真实图像：

$ L = (1-lambda) L_1 + lambda L_("D-S S I M") $

虽然课件没有详细展开这个 loss，但复习时应知道 3DGS 原论文常用 L1 + SSIM 类似组合。

重要程度：★★★★★

---

= 16. Adaptive Control：Densification / Split / Prune

3DGS 不只是优化已有 Gaussians，还会动态调整 Gaussian 数量。

== 16.1 Densification

目标是在重建不足的区域增加点。

indicator 是 position gradient：

$ |nabla_mu L| $

如果某个 Gaussian 的位置梯度很大，说明该区域当前重建不好，优化器正在努力移动它来拟合结构。

因此需要增加 Gaussian。

== 16.2 Actions

课件给出三种操作：

| 操作 | 场景 | 作用 |
| ------ | ---------------------------------------- | ----------------- |
| Clone | under-reconstruction | 补充缺失几何 |
| Split | over-reconstruction / too large Gaussian | 拆分大 Gaussian，提高细节 |
| Remove | opacity below threshold | 删除无贡献 Gaussian |

重要程度：★★★★★

---

= 17. 3DGS Rendering：Splatting + Tile-based Rasterization

== 17.1 Forward Rendering Process

3DGS 的渲染过程：

```text
3D Gaussians
 ↓
Project to 2D image plane
 ↓
Splatting: become 2D ellipses
 ↓
Divide image into tiles
 ↓
Duplicate Gaussians overlapping multiple tiles
 ↓
Sort by depth front-to-back
 ↓
Alpha blending
```

重要程度：★★★★★

== 17.2 什么是 Splatting？

Splatting 的直觉是：把一个 3D soft colored snowball 投影到 2D screen 上，形成一个椭圆状的 splat。

数学上是：

$ "3D Gaussian ellipsoid"
->
"2D projected Gaussian ellipse" $

== 17.3 Alpha Blending

最终颜色按深度顺序累积：

$ C
=

sum_(i=1)^{N}
T_i alpha_i c_i $

其中：

$ T_i = product_(j=1)^(i-1) (1-alpha_j) $

符号含义：

$C$ 是最终像素颜色；$c_i$ 是第 $i$ 个 Gaussian 的颜色；$alpha_i$ 是第 $i$ 个 Gaussian 的 opacity；$T_i$ 是前面所有 Gaussian 没有遮挡该 Gaussian 的剩余透射率。

直觉：

$ "final color"
=

sum
"current color"
times
"current opacity"
times
"light not blocked by previous objects" $

重要程度：★★★★★

---

= 18. 方法论演进总表

| 阶段 | 方法 | 解决什么 | 前一方法问题 | 新方法如何修复 |
| -------- | ---------------------- | -------------------- | ------------------------ | ----------------------------------- |
| 表征学习 | AE | 无监督学习 latent feature | 不能生成新样本 | 只能重建，不保证 latent 连续 |
| 概率生成 | VAE | 让 latent 可采样 | AE latent space 有空洞 | KL regularization 逼近 Gaussian prior |
| 对抗生成 | GAN | 提高视觉锐利度 | VAE 平均化、模糊 | discriminator 学习真实感判别 |
| 稳定高质量生成 | Diffusion | 高质量、多样生成 | GAN 训练不稳定 | 多步去噪，训练更稳定 |
| 高效图像生成 | LDM / Stable Diffusion | 降低扩散计算成本 | pixel-space diffusion 太慢 | 在 latent space 去噪 |
| 可控生成 | ControlNet | 精确控制姿态/边缘/深度 | text prompt 空间控制弱 | 加入视觉条件分支 |
| 3D 神经表示 | NeRF | 连续三维场景表示 | 传统 3D 表示有限 | MLP 表示 radiance field |
| 实时 3D 渲染 | 3DGS | 高质量实时渲染 | NeRF ray sampling 慢 | 显式 Gaussian + rasterization |

---

= 19. 期末重点排序

最高优先级，必须掌握：

VAE objective、KL divergence、reparameterization trick、VAE blurry 原因、GAN minimax objective、Diffusion forward/reverse intuition、Stable Diffusion latent-space denoising、ControlNet zero conv、NeRF radiance field 与 volume rendering、3DGS Gaussian parameters、covariance decomposition、alpha blending。

中等优先级：

FCN、Fast R-CNN、Mask R-CNN、YOLO output shape、DETR set prediction、CycleGAN cycle consistency、SfM pipeline、Bundle Adjustment。

较低优先级：

StyleGAN 细节、Video generation 具体模型、World Models 具体链接、DreamFusion 细节。课件中这些更偏前沿介绍。

---

= 20. 本章一句话总结

本章的底层逻辑是：视觉模型从“识别图像中的结构”进化到“生成图像分布”，再进一步进化到“用可微、可渲染、可优化的方式表示三维世界”；每一次方法升级，本质都在修复前一代方法在连续性、真实性、稳定性、可控性或效率上的瓶颈。
