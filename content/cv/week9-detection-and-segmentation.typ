// @title: Week 9：Detection & Segmentation 核心脉络
// @description: 本章要解决的核心目标是：让视觉模型从 image-level recognition 走向 spatial understanding。也就是说，不只是回答“图里有什么”，而是回答“在哪里”“每个像素是什么”“每个实例是谁”“模型为什么这么
// @order: 999

= Week 9：Detection & Segmentation 核心脉络

本章要解决的核心目标是：让视觉模型从 image-level recognition 走向 spatial understanding。也就是说，不只是回答“图里有什么”，而是回答“在哪里”“每个像素是什么”“每个实例是谁”“模型为什么这么判断”“能否用 prompt 直接迁移到新任务”。

整体进化树可以概括为：

```text
Image Classification
 ↓ 只预测类别，没有空间位置
Semantic Segmentation
 ↓ 给每个 pixel 分类，但不区分实例
Object Detection
 ↓ 给每个 object 输出类别 + bounding box
Instance Segmentation
 ↓ 给每个 object 输出类别 + box + mask
Visualization / Understanding
 ↓ 理解模型关注哪里：filters, saliency, CAM, Grad-CAM
Foundation Vision Models
 ↓ 用 prompt 做 zero-shot / interactive segmentation：SAM
```

= 1. Semantic Segmentation：从 patch 分类到全卷积分割

Semantic segmentation 的任务是：给图像中每一个像素分配一个语义类别。输入是图像 $I in R R^{3 times H times W}$，输出是像素级标签图：

$ Y in {1,dot(s),C}^{H times W} $

其中 $H,W$ 是图像高宽，$C$ 是语义类别数。注意它只关心“这个像素属于什么类别”，不关心同类物体之间的实例差异。例如两头 cow 都标成 cow，不区分 cow-1 和 cow-2。

重要程度：★★★★★。这是分割任务的基础定义，考试很容易问 semantic segmentation、object detection、instance segmentation 的区别。

最早的直觉方法是 sliding window：对每个像素周围取一个 patch，然后用 CNN 判断中心像素类别。形式上，对像素位置 $(i,j)$，取局部 patch：

$ P_{i,j} = I[i-r:i+r,\ j-r:j+r] $

然后预测：

$ hat(y)_{i,j} = op("argmax")_c f_theta(P_{i,j})_c $

其中 $r$ 是 patch 半径，$f_theta$ 是 CNN 分类器，$f_theta(P_{i,j})_c$ 是类别 $c$ 的 score。

这个方法的动机是：单个像素无法判断语义，必须看上下文。例如黑色像素可能是 cow，也可能是 shadow，所以要看周围区域。但它的问题很严重：相邻 patch 高度重叠，却要重复跑 CNN，计算极其浪费。也就是说，它解决了 context 问题，但没有解决 shared computation 问题。

于是自然过渡到 fully convolutional idea：不要对每个 patch 单独跑 CNN，而是对整张图一次性提取 feature map，然后对每个空间位置分类。最直接形式是：

$ F = g_theta(I) in R R^{D times H times W}  S = h_phi(F) in R R^{C times H times W}  hat(Y)_{i,j} = op("argmax")_c S_{c,i,j} $

其中 $F$ 是卷积特征图，$D$ 是通道数，$S$ 是每个像素的类别 score map。

但问题是：如果一直保持原图分辨率 $H times W$ 做卷积，计算量太大；如果像分类 CNN 一样用 pooling 或 stride 下采样，又会导致输出空间尺寸变小，不满足 pixel-wise prediction。因此 FCN 的核心折中是：encoder 先 downsample 获得高级语义，再 decoder upsample 恢复空间分辨率。

典型结构是：

```text
Input image
 ↓ convolution / pooling / strided convolution
Low-resolution semantic feature
 ↓ unpooling / transposed convolution / interpolation
High-resolution prediction map
 ↓ op("argmax") over classes
Pixel labels
```

= 2. Upsampling：从非参数放大到可学习上采样

语义分割中的核心矛盾是：深层 CNN 要下采样以扩大 receptive field、降低计算量；但 segmentation 要输出原图大小。因此必须上采样。

第一类是非学习式上采样。Nearest Neighbor upsampling 直接复制值。例如：

$ mat(1, 2 \
3, 4)
->
mat(1, 1, 2, 2 \
1, 1, 2, 2 \
3, 3, 4, 4 \
3, 3, 4, 4) $

Bed of Nails upsampling 是插零：

$ mat(1, 2 \
3, 4)
->
mat(1, 0, 2, 0 \
0, 0, 0, 0 \
3, 0, 4, 0 \
0, 0, 0, 0) $

Nearest Neighbor 简单但不可学习，Bed of Nails 通常配合后续卷积使用.

第二类是 Max Unpooling。它和 Max Pooling 成对出现。Max Pooling 下采样时不仅记录最大值，还记录最大值所在位置，即 pooling switches。上采样时把低分辨率值放回原来最大值的位置，其余填零。它的优点是能部分恢复 pooling 前的空间结构；缺点是只恢复最大激活的位置，仍然很稀疏。

第三类是 Transposed Convolution，也叫 deconvolution，但严格说不是卷积的逆，而是普通卷积线性算子的转置。普通卷积 stride $s$ 可以看作 learnable downsampling：filter 在输入上移动 $s$ 个像素，对应输出移动一个位置。Transposed convolution 则反过来：输入移动一个位置，filter 在输出上移动 $s$ 个像素；重叠区域求和。

1D 例子最容易记：

输入为 $[a,b]$，filter 为 $[x,y,z]$，stride 为 1，则输出可以理解为：

$ a[x,y,z] + b[-,x,y,z] $

重叠后得到：

$ [a x,\ a y + b x,\ a z + b y,\ b z] $

更一般地，transposed convolution 的每个输入元素都会“撒出”一个乘以该输入值的 filter pattern，多个 pattern 在输出空间重叠处相加。

重要程度：★★★★★。考试常问 transposed convolution 与 normal convolution 的区别，以及为什么它是 learnable upsampling。

= 3. U-Net：用 skip connection 修复空间细节丢失

FCN 的问题是：encoder 深层 feature 语义强，但空间细节弱；浅层 feature 分辨率高，但语义弱。U-Net 的核心思想是 encoder-decoder 加 skip connection，把 encoder 中高分辨率的浅层特征直接传给 decoder 对应层。

结构直觉：

```text
Encoder: high resolution → low resolution, local detail → semantic abstraction
Decoder: low resolution → high resolution, recover spatial prediction
Skip: concatenate encoder features into decoder
```

如果记为 encoder 第 $l$ 层特征为 $E_l$，decoder 对应层上采样特征为 $D_l$，则 U-Net 常用：

$ D_l' = upright(op("Concat"))(D_l,\ E_l) $

然后再经过卷积：

$ D_{l-1} = upright("Conv")(D_l') $

其中 $upright(op("Concat"))$ 通常在 channel dimension 上拼接。

U-Net 的意义是：它不是单纯“先压缩再恢复”，而是把 localization detail 通过 skip connection 直接送回去。因此特别适合医学图像、边界敏感的分割任务。

重要程度：★★★★☆。不一定考细节公式，但很容易考动机：为什么 FCN 需要 skip connection？为什么 U-Net 对 segmentation 有效？

= 4. Object Detection：从单目标定位到多目标检测

Object detection 的输出不是像素类别，而是对象集合：

$ {(c_i, b_i)}_{i=1}^{N} $

其中 $c_i$ 是第 $i$ 个 object 的类别，$b_i$ 是 bounding box，通常表示为：

$ b_i = (x_i, y_i, w_i, h_i) $

其中 $(x_i,y_i)$ 可以表示 box 左上角或中心点，$w_i,h_i$ 是宽高。

单目标检测可以看成 classification + localization。网络共享一个 feature vector，例如：

$ z in R R^{4096} $

分类头输出：

$ p = upright("softmax")(W_c z + b_c) $

定位头输出：

$ hat(b) = W_b z + b_b $

训练损失是 multitask loss：

$ L = L_{upright("cls")} + lambda L_{upright("loc")} $

其中：

$ L_{upright("cls")} = -log p_{y}  L_{upright("loc")} = |hat(b) - b^ast|_2^2 $ 其中 $y$ 是真实类别，$b^ast = (x^ast, y^ast, w^ast, h^ast)$ 是真实框，$lambda$ 控制分类和定位损失的权重。

多目标检测的关键困难是：每张图 object 数量不同，因此输出长度不是固定的。普通 C N N 分类器输出固定维度向量，无法直接输出可变长度集合。因此早期做法是：对图像中大量候选 crop 分别分类为 object 或 background。

问题是：所有位置、尺度、长宽比都枚举，计算量巨大。

重要程度：★★★★★。检测任务的基本输出形式、multitask loss、可变数量输出问题都很核心。

= 5. R-C N N 系列：共享计算越来越多

R-C N N 的核心流程是：

```text
Input image
 ↓ Selective Search 提出约 2000 RoIs
Warp each RoI to 224×224
 ↓ 每个 RoI 单独过 C N N
Feature vector
 ↓ S V M 分类
 ↓ bbox regressor 修正框
Detection result
```

R-C N N 的 bbox correction 通常不是直接预测最终 box，而是预测相对 proposal 的变换： $
(t_x,t_y,t_w,t_h)
$ 常见参数化是： $
t_x = frac(x^ast - x_a, w_a), quad
t_y = frac(y^ast - y_a, h_a)

t_w = log frac(w^ast, w_a), quad
t_h = log frac(h^ast, h_a)
$ 其中 $(x_a,y_a,w_a,h_a)$ 是 anchor/proposal box， $x^ast,y^ast,w^ast,h^ast$ 是 ground-truth box。预测时反变换： $
hat(x) = x_a + hat(t)_x w_a,quad
hat(y)=y_a+hat(t)_y h_a

hat(w) = w_a e^{hat(t)_w},quad
hat(h)=h_a e^{hat(t)_h}
$ R-C N N 的失败点非常明确：每张图约 2000 个 RoI，每个 RoI 都要独立过 C N N，极慢，且没有共享卷积计算。

Fast R-C N N 的改进是：先对整张图跑 backbone 得到 feature map，再把 proposal 映射到 feature map 上做 RoI Pool / Crop + Resize。流程变成：

```text
Input image
 ↓ Backbone C N N once
Conv feature map
 ↓ RoI proposals projected onto feature map
RoI pooling / crop-resize
 ↓ per-region "head"
classification + bbox regression
```

它解决了 R-C N N 的重复 C N N forward 问题。但 Fast R-C N N 仍然依赖外部 proposal 方法，例如 Selective Search，而 Selective Search 通常在 C P U 上运行，仍然慢。

Faster R-C N N 的核心创新是 Region Proposal Network，R P N。R P N 和 detector 共享 backbone feature map，直接在 feature map 每个位置预测 anchor 是否包含 object，以及对应 box correction。

如果 feature map 是： $
F in RR^{D times H' times W'}
$ 每个位置有 $K$ 个 anchors，则 R P N 输出 objectness： $
O in RR^{K times H' times W'}
$ 和 box transforms： $
T in RR^{4K times H' times W'}
$ 每个 anchor 对应： hat(t)_x,hat(t)_y,hat(t)_w,hat(t)_h 然后按 objectness score 排序，取 top proposals，例如约 300 个，再送入 Fast R-C N N detector "head"。

R-C N N 系列的进化逻辑是：

| 方法 | Proposal 来源 | C N N 计算方式 | 主要瓶颈 | 改进点 |
| ------------ | ---------------- | -------------------------- | ------------------- | --------------------- |
| R-C N N | Selective Search | 每个 RoI 单独 C N N | 约 2000 次 forward，极慢 | 使用 proposal 降低枚举空间 |
| Fast R-C N N | Selective Search | 整图一次 C N N，共享 feature | proposal 仍外部生成 | 共享卷积计算 |
| Faster R-C N N | R P N | R P N 与 detector 共享 backbone | 结构更复杂 | proposal 也由网络生成，接近端到端 |

重要程度：★★★★★。这是检测方法演进的主线，考试非常可能考 Why R-C N N slow / how Fast R-C N N fixes / why Faster R-C N N faster。

= 6. Single-Stage Detector：Y O L O / S S D / RetinaNet

Two-stage detectors 先生成 proposals，再对 proposals 分类回归。Single-stage detectors 直接在 dense grid 上预测 boxes 和 classes，不显式分 proposal stage。

Y O L O 的基本思想是：把图像划分成 $S times S$ 网格，每个 grid cell 预测 $B$ 个 bounding boxes 和类别概率。输出维度通常是： $
S times S times (5B + C)
$ 其中 $B$ 是每个 cell 的 box 数，$C$ 是类别数，每个 box 的 5 个量通常是： x,y,w,h,upright("confidence") confidence 可以理解为： $
upright("confidence") = P(upright("object")) dot upright("IoU")(hat(b), b^ast) $ 每个 cell 还预测类别概率： $
P(c mid upright("object")) $ 最终类别置信度常写成： $
P(c mid upright("object")) P(upright("object")) upright("IoU")(hat(b), b^ast) $ Y O L O 的优势是快，因为整张图只 forward 一次；缺点是 dense prediction 容易产生大量候选框，需要 N M S，且早期 Y O L O 对小物体、密集物体不如 two-stage 方法。

RetinaNet 的关键思想是 Focal Loss，用来解决 single-stage dense detector 中正负样本极度不平衡的问题。标准二分类交叉熵为： $
"CE"(p_t) = -log(p_t)
$ 其中： $
p_t =
cases(p,, y=1 \
1-p,, y=0)
$ Focal Loss 是： $
"FL"(p_t) = -alpha_t(1-p_t)^gamma log(p_t)
$ 其中 $gamma$ 是 focusing parameter，$alpha_t$ 是类别平衡系数。当样本很容易分类时，$p_t$ 接近 1，$(1-p_t)^gamma$ 很小，loss 被降低；模型会更关注 hard examples。

重要程度：★★★★☆。课件中重点展示 Y O L O 输出结构；RetinaNet 只列为 single-stage detector，但如果期末涉及检测方法对比，Focal Loss 很值得掌握。

= 7. D E T R：把检测改写为集合预测

D E T R 的核心转变是：不再使用 anchors、IoU thresholds、N M S，而是把 object detection 视为 set prediction。模型直接输出固定数量 $N$ 个 object queries 的预测： $
{$hat(c)_i,hat(b)_i$}_{i=1}^{N}
$ 其中没有 object 的 query 预测为 no-object 类。

D E T R 的训练关键是 Hungarian bipartite matching。给定真实 objects： $
{(c_j,b_j)}_{j=1}^{M}
$ 预测集合： $
{$hat(p)_i,hat(b)_i$}_{i=1}^{N}
$ 其中 $N >= M$。先找一个最优匹配 $sigma$： $
hat(sigma) = op("argmin")_{sigma in frak(S)*N}
sum*{j=1}^{M}
cal(L)_{upright("match")}((c_j,b_j),hat(p)_{sigma(j)},hat(b)_{sigma(j)}))
$ 匹配代价通常包括分类代价和 box 代价： $
cal(L)_{upright("match")}
=

-hat(p)_{sigma(j)}(c_j)
+
lambda_1 |b_j-hat(b)_{sigma(j)}|_1
+
lambda_2 cal(L){upright("GIoU")} b_j, hat(b)_{sigma(j)}
$ 完成匹配后，再对匹配上的 prediction 做分类和 box regression loss，未匹配的 query 训练为 no-object。

D E T R 的意义是：它消除了很多手工工程组件，例如 anchors、N M S、proposal generation、复杂 IoU threshold assignment。它依赖 Transformer 的 global reasoning，用 object queries 表示一组待预测对象。

重要程度：★★★★★。D E T R 是现代检测的重要范式转变，重点是 set prediction + Hungarian matching + no anchors + no N M S。

= 8. Instance Segmentation：Mask R-C N N

Instance segmentation 要输出： $ (c_i,b_i,m_i)_(i=1)^N $
其中 $m_i$ 是第 $i$ 个实例的 binary mask： $
m_i in {0,1}^{H times W}
$ 它比 object detection 多了 mask，比 semantic segmentation 多了 instance identity。

Mask R-C N N 是 Faster R-C N N 的扩展：在 RoI feature 上除了分类头和 bbox regression "head"，再加一个 mask "head"。流程是：

```text
Backbone + R P N
 ↓ proposals
RoI Align
 ↓ fixed-size RoI feature, e.g. 256×14×14
classification "head" → class scores
box "head" → 4C box coordinates
mask "head" → C × 28 × 28 masks
```

Mask "head" 对每个类别预测一个 mask： $ M in RR^{C times 28 times 28} $ 训练时只对 ground-truth 类别对应的 mask 计算 binary cross entropy：

$ L_{upright("mask")}
=

-frac(1, m^2)
sum_{u=1}^{m}
sum_{v=1}^{m}
[
y_{"uv"}log hat(y)_{"uv"}
+
(1-y_{"uv"})log(1-hat(y)_{"uv"})
]
$ 其中 $m=28$，$y_{{"uv"}}$ 是真实 mask 像素，$hat(y)_{{"uv"}}$ 是预测概率。

整体 loss 是： $
L = L_{upright("cls")} + L_{upright("box")} + L_{upright("mask")}
$ RoI Align 是 Mask R-C N N 的关键细节。它相比 RoI Pooling 去掉了粗暴量化，使用 bilinear interpolation 保留更准确的空间对齐。因为 mask prediction 对像素级对齐非常敏感，所以 RoI Align 对 instance segmentation 很重要。

重要程度：★★★★★。考试容易问：Mask R-C N N 相比 Faster R-C N N 加了什么？为什么需要 RoI Align？mask loss 怎么算？

= 9. Visualization：从看 filter 到看 saliency

模型解释部分的核心目标是：理解 C N N 到底学到了什么，以及哪些像素影响分类结果。

第一层 filter visualization 很直接。C N N 第一层卷积核形状如 AlexNet： $
64 times 3 times 11 times 11
$ 表示 64 个 filter，每个 filter 有 3 个 R G B channel，每个空间大小为 $11 times 11$。ResNet / DenseNet 第一层常见为： $
64 times 3 times 7 times 7
$ 第一层 filter 通常能可视化出边缘、方向、颜色 blob 等低级视觉模式。

Saliency Map 的思想是：看类别分数对输入像素的梯度。给定类别 $c$ 的 unnormalized score： $
S_c(I)
$ 计算： $
G = frac(partial S_c, partial I)
$ 由于 $I in RR^{3 times H times W}$，所以梯度也是： $
G in RR^{3 times H times W}
$ saliency map 通常取 R G B channel 上的绝对值最大值： $
M_{i,j} = max_{k in {R,G,B}}
|
frac(partial S_c, partial I_{k,i,j})
|
$ 直觉是：如果某个像素轻微变化会强烈影响类别 score，那么该像素对分类重要。

重要程度：★★★★☆。Saliency via backprop 的公式很可能考，尤其是“对 unnormalized class score 求梯度，而不是 softmax probability”。

= 10. C A M 与 Grad-C A M

C A M，即 Class Activation Mapping，依赖特殊结构：最后一个卷积层后接 Global Average Pooling，再接 linear classifier。

设最后卷积特征为：
$ A in RR^{H times W times K} $
其中 $A^k_{i,j}$ 是第 $k$ 个 channel 在位置 $(i,j)$ 的激活。Global Average Pooling 得到： $ F_k = frac(1, H W) sum_{i=1}^{H} sum_{j=1}^{W} A^k_{i,j} $
分类 score 为： $ S_c = sum_{k=1}^{K} w_k^c F_k $
代入 G A P： $ S_c = sum_{k=1}^{K} w_k^c frac(1, H W) sum_{i,j} A^k_{i,j} = frac(1, H W) sum_{i,j} sum_{k=1}^{K} w_k^c A^k_{i,j} $
因此类别 $c$ 的 C A M map 是： $ M_c(i,j)=sum_{k=1}^{K} w_k^c A^k_{i,j} $
其中 $w_k^c$ 是 classifier 中类别 $c$ 对 channel $k$ 的权重。直觉是：如果某个 channel 对类别 $c$ 权重大，并且在某个空间位置激活强，那么这个位置对类别 $c$ 重要。

C A M 的限制是：只能用于特定架构，通常要求最后 conv layer + G A P + linear classifier。

Grad-C A M 解决这个限制。它可以用于任意卷积层。给定某层 activation： $ A in RR^{H times W times K} $
先计算类别 score 对 activation 的梯度： $ frac(partial S_c, partial A^k_{i,j}) $
然后对空间维度做 global average pooling，得到每个 channel 的重要性权重： $
alpha_k^c = frac(1, H W) sum_{i=1}^{H} sum_{j=1}^{W} frac(partial S_c, partial A^k_{i,j})
$
最后得到 Grad-C A M：
$
M_c
=

upright("ReLU")
(
sum_{k=1}^{K}
alpha_k^c A^k
)
$
这里 ReL U 的作用是只保留对类别 $c$ 有正贡献的区域。Grad-C A M 的物理直觉是：梯度告诉我们“如果增强这个 feature channel，会不会提高类别分数”；activation 告诉我们“这个 feature 在哪里出现”。二者相乘加权后，就得到类别相关的空间热力图。

重要程度：★★★★★。C A M / Grad-C A M 的公式、区别、Grad-C A M 的四步流程都是重点。

= 11. Foundation Vision Model 与 S A M

本章最后从传统 supervised vision tasks 过渡到 promptable foundation model。N L P 中 L L M 可以通过 prompt 做 zero-shot transfer，例如 summarization、conversation、content creation。视觉任务也希望有类似模型，但困难在于：视觉标注尤其 segmentation mask 标注非常贵，数据稀缺。

S A M，即 Segment Anything，试图把 segmentation 变成 promptable task： $
"Image" + "Prompt" -> "Valid Mask"
$ prompt 可以是 point、box、text、mask。即使 prompt 有歧义，模型也要返回合理 mask。

S A M 的系统由三部分构成：

```text
Image Encoder
 ↓ heavy ViT, M A E pretraining
Prompt Encoder
 ↓ point / box / text / mask embeddings
Mask Decoder
 ↓ lightweight transformer decoder + upsampling
Output masks + predicted IoU
```

Image encoder 是 heavy ViT，并使用 M A E 预训练。M A E 的关键思想是 masked autoencoding：随机 mask 掉大量 patches，例如 75%，encoder 只处理可见的 25% patches，然后 decoder 结合 mask tokens 重建像素。视觉中 M A E 和 N L P M L M 不完全一样，因为相邻图像 patch 冗余更强，重建像素不等于预测语义 token，所以需要较强 decoder，但 decoder 仍比 encoder 轻。

Prompt encoder 的编码方式：

Point prompt：点的位置编码 + 前景/背景 learnable embedding。

Box prompt：两个角点的位置编码 + 表示角点类型的 learnable embedding。

Text prompt：C L I P text encoder。

Mask prompt：卷积处理 mask，然后与 image embedding 做 element-wise sum。

Mask decoder 使用轻量 transformer decoder，包含 prompt self-attention、prompt-to-image cross-attention、M L P、image-to-prompt cross-attention，最后用 transposed convolution 上采样得到 mask。

S A M 会预测三个 masks： $
{hat(m)_1,hat(m)_2,hat(m)_3}
$ 原因是 prompt 可能有歧义，例如一个点可能对应 whole object、part、sub-part。训练时对多个 masks 只反传最小 loss：
$
L = min_{r in {1,2,3}} L_{upright("mask")}
$
$hat(m)_r, m^ast$ 同时还有一个 IoU prediction token，用来预测每个 mask 的质量并排序：
$ hat(upright("IoU"))_r approx upright("IoU")(hat(m)_r, m^ast) $

S A M 的数据引擎分三阶段：

```text
Assisted manual stage:
120k images, 4.3M masks
 ↓
Semi-automatic stage:
180k images, 5.9M masks
 ↓
Fully automatic stage:
11M images, 1.1B masks
```

Fully automatic stage 用密集点 prompt，例如 $32 times 32$ grid，保留高置信度 mask，并用 NMS 去重。

SAM 的 zero-shot transfer 用法包括：edge detection、object proposals、instance segmentation。例如用 object detector 输出 bounding boxes，再把 boxes 作为 SAM prompt 得到 mask；也可以把 mask 再反馈给 decoder refine。

重要程度：★★★★☆。SAM 结构、promptable segmentation、三个 mask 输出、IoU prediction、model-in-the-loop data engine 是重点。

= 总结对比表

| 任务 | 输入 | 输出 | 是否有空间位置 | 是否区分实例 | 典型方法 |
| ----------------------- | -------------- | ----------------------- | -------- | ----------- | ------------------------------- |
| Classification | Image | Class label | 否 | 否 | CNN, ViT |
| Semantic Segmentation | Image | Pixel label map | 是，像素级 | 否 | FCN, U-Net |
| Object Detection | Image | Boxes + classes | 是，box 级 | 是 | R-CNN, Faster R-CNN, YOLO, DETR |
| Instance Segmentation | Image | Boxes + classes + masks | 是，mask 级 | 是 | Mask R-CNN, SAM-based pipeline |
| Saliency / CAM | Image + model | Importance map | 是 | 不一定 | Saliency, CAM, Grad-CAM |
| Promptable Segmentation | Image + prompt | Valid mask(s) | 是 | 由 prompt 决定 | SAM |

= 最核心考试记忆链

本章最重要的逻辑不是背模型名字，而是理解每一步为什么出现：

```text
Classification 不够：没有位置
→ Semantic segmentation：每个像素分类
→ Sliding window 太慢：重复计算
→ FCN：共享卷积特征
→ FCN 下采样丢分辨率：需要 upsampling
→ Transposed conv / unpooling：恢复空间尺寸
→ U-Net：skip connection 恢复细节

Detection 输出数量可变
→ crop 分类太慢
→ R-CNN 用 proposal 减少搜索
→ R-CNN 每个 RoI 跑 CNN 太慢
→ Fast R-CNN 共享整图 feature
→ proposal 仍慢
→ Faster R-CNN 用 RPN 学 proposal
→ single-stage detector 用 dense prediction 换速度
→ DETR 用 set prediction 消除 anchors 和 NMS

Segmentation + Detection
→ Mask R-CNN：Faster R-CNN + mask "head" + RoI Align

理解模型
→ filter visualization 看低层特征
→ saliency 看输入梯度
→ CAM 用 classifier weights 定位
→ Grad-CAM 用 gradients 泛化到任意层

Foundation model
→ SAM 把 segmentation 变成 promptable task
→ image encoder + prompt encoder + mask decoder
→ 多 mask 输出处理歧义
→ data engine 扩大 mask 数据
```
