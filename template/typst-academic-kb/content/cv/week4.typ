// @title: Week 04: Features & Matching 学习笔记
// @description: 本章的核心目标不是单纯“检测边缘”或“找角点”，而是建立一条完整的传统视觉匹配流水线：从图像中的局部变化出发，提取可重复、可区分的局部锚点，再用描述子进行跨图像匹配，最后用几何一致性过滤错误匹配。课件前半部分从 Sobel、Canny、Co
// @order: 999

= Week 04: Features & Matching 学习笔记

本章的核心目标不是单纯“检测边缘”或“找角点”，而是建立一条完整的传统视觉匹配流水线：从图像中的局部变化出发，提取可重复、可区分的局部锚点，再用描述子进行跨图像匹配，最后用几何一致性过滤错误匹配。课件前半部分从 Sobel、Canny、Contour 说明“如何把像素变化变成结构边界”，后半部分进入真正的 matching problem：两张图像中同一物理点在亮度、颜色、视角、尺度、旋转下会变化，因此我们需要寻找跨变化仍稳定存在的 invariants。

---

== 1. 本章进化树：从边缘检测到几何匹配

可以把本章理解成如下技术演进链：

```text
图像像素 I(x, y)
 ↓
局部亮度变化：Sobel gradient
 ↓ 解决：找边缘候选
 ↓ 问题：边缘厚、噪声敏感、无明确判定
Canny edge detection
 ↓ 解决：平滑 + 梯度 + NMS + 双阈值 + hysteresis
 ↓ 问题：边缘点不适合匹配，沿边方向不可定位
Corner / Harris
 ↓ 解决：找二维方向上都发生变化的点
 ↓ 问题：无尺度、无方向、无描述子
SIFT
 ↓ 解决：DoG 尺度空间 + 主方向对齐 + 128-D 梯度直方图描述子
 ↓ 问题：局部描述子匹配仍有假匹配
KNN + Lowe Ratio Test
 ↓ 解决：过滤最近邻不够独特的匹配
 ↓ 问题：单个匹配正确不代表全局几何一致
RANSAC + Homography
 ↓ 解决：用全局投影几何筛选 inliers
 ↓
应用：panorama stitching, localization, large-scale visual search / BoW
```

这条链条的核心逻辑是：先从图像中找到“哪里有结构”，再判断“哪些结构适合跨图像重识别”，最后用几何约束判断“这些局部相似是否共同解释同一个场景变换”。

---

= 2. Sobel：从像素到梯度

== 2.1 要解决的问题

原始图像只是灰度或颜色值。边缘检测的第一步是问：图像在哪些位置发生剧烈变化？Sobel 用卷积近似图像在 (x) 和 (y) 方向的偏导数，从而得到局部亮度变化方向和强度。课件中强调，Sobel X 检测左右方向亮度变化，即垂直边缘；Sobel Y 检测上下方向亮度变化，即水平边缘；梯度幅值把所有方向的边缘合并起来。

== 2.2 Sobel 卷积公式

设灰度图像为

$ I(x,y) $

其中：

- (I(x,y))：坐标 ((x,y)) 处的像素强度；
- (x)：水平坐标，通常向右增加；
- (y)：垂直坐标，通常向下增加。

Sobel 在两个方向上的卷积核通常写为：

$ K_x= mat(-1, 0, 1; -2, 0, 2; -1, 0, 1), quad quad K_y= mat(-1, -2, -1; 0, 0, 0; 1, 2, 1) $

于是：

$ G_x = K_x * I  G_y = K_y * I $

其中：

- \(\*\)：二维卷积操作；
- (G_x)：图像在水平方向的梯度响应，亮度从左到右变化越剧烈，(|G_x|) 越大；
- (G_y)：图像在垂直方向的梯度响应，亮度从上到下变化越剧烈，(|G_y|) 越大。

梯度幅值为：

$ |nabla I| = sqrt{G_x^2 + G_y^2} $

梯度方向为：

$ theta = op("atan2")(G_y, G_x) $

其中：

- $nabla I = (G_x, G_y)^top$：图像梯度向量；
- $|nabla I|$：边缘强度；
- $theta$：亮度上升最快的方向，也就是梯度方向；
- $op("atan2")$：带象限信息的反正切函数。

直觉上，边缘不是“线本身”，而是强度函数的陡坡。Sobel 的输出告诉我们：哪里有坡，以及坡朝哪个方向。

== 2.3 Sobel 的局限

Sobel 的问题不是完全不能找边缘，而是它只完成了“梯度估计”，没有完成“边缘决策”。课件列出三个核心痛点：边缘厚，不是单像素精度；仍然对噪声敏感；没有阈值机制，不告诉我们哪些像素最终应被认为是边缘。

因此 Sobel 的定位是：它是 Canny 的一个组件，而不是完整的边缘检测系统。

---

= 3. Canny：从梯度图到干净、连续、单像素边缘

== 3.1 Canny 为什么出现

Sobel 给出梯度，但实际边缘检测还需要解决三个问题：噪声会造成虚假梯度；真实边缘常常是几像素宽的带状区域；单阈值会在“断边”和“假边”之间两难。Canny 的贡献是把边缘检测变成一条完整 pipeline：Gaussian smoothing → gradient computation → non-maximum suppression → dual threshold → hysteresis tracking。

== 3.2 Step 1: Gaussian smoothing

噪声往往表现为高频变化，也会产生大梯度。Canny 首先用高斯滤波平滑图像：

$ I_sigma(x,y) = (G_sigma * I)(x,y) $

其中：

$ G_sigma(x,y) = frac(1, 2 pi sigma^2) op("exp")(-frac(x^2+y^2, 2 sigma^2) ) $

符号定义：

- (I)：原始图像；
- $I_sigma$：经过尺度 $sigma$ 平滑后的图像；
- $G_sigma$：二维高斯核；
- $sigma$：高斯标准差，控制平滑强度；
- (x,y)：相对卷积中心的坐标。

直觉上，$sigma$ 越大，噪声越少，但细节也越容易被抹掉。因此 Canny 的第一步已经引入了检测稳定性与细节保留之间的 trade-off。

== 3.3 Step 2: Gradient computation

在平滑图像上计算梯度：

$ G_x = frac(partial I_sigma, partial x), quad quad G_y = frac(partial I_sigma, partial y)  M(x,y)=sqrt{G_x(x,y)^2+G_y(x,y)^2}  theta(x,y)=op("atan2")(G_y(x,y),G_x(x,y)) $

其中：

- (M(x,y))：梯度幅值，即候选边缘强度；
- $ theta(x,y) $：梯度方向；
- 真正的边缘方向与梯度方向垂直。

== 3.4 Step 3: Non-Maximum Suppression

Canny 的核心创新之一是 NMS。课件中的表述是：沿梯度方向，只保留局部最大值，其余抑制，使厚边缘变成单像素边缘。

数学上，对每个像素 (p=(x,y))，沿梯度方向 $theta(p)$ 找到前后两个邻居 (p_+) 和 (p_-)。NMS 的规则为：

$ M_{"nms"}(p) = cases(M(p),, M(p) >= M(p_+) "and" M(p) >= M(p_-), 0,, "otherwise") $

其中：

- (p)：当前像素；
- (p_+,p_-)：沿梯度方向的两个相邻采样点；
- (M(p))：当前像素梯度幅值；
- $M_{"nms"}(p$)：NMS 后的边缘强度。

实际实现中，连续角度 $theta$ 通常量化为四个方向：

| 梯度角度范围 | 量化方向 | 比较邻居 |
| --------------------------- | ----------: | ----- |
| ([-22.5^circ,22.5^circ]) | (0^circ) | 左、右 |
| ([22.5^circ,67.5^circ]) | (45^circ) | 左下、右上 |
| ([67.5^circ,112.5^circ]) | (90^circ) | 上、下 |
| ([112.5^circ,157.5^circ]) | (135^circ) | 左上、右下 |

这里的直觉是：边缘是梯度幅值曲面上的 ridge。NMS 只保留 ridge 顶点。

== 3.5 Step 4: Dual threshold

单阈值的问题是两难：阈值太高，真实边缘断裂；阈值太低，噪声边缘太多。课件明确指出：单阈值无法同时保证 noise suppression 和 edge continuity，所以使用两个阈值。

设高阈值为 $T_{"high"}$，低阈值为 $T_{"low"}$，且

$ T_{"high"} > T_{"low"} $

对 NMS 后的响应 $M_{"nms"}(p$)，分类为：

$ "class"(p)= cases("strong edge",, M_{"nms"}(p)>= T_{"high"}, "weak edge",, T_{"low"} <= M_{"nms"}(p) < T_{"high"}, "non-edge",, M_{"nms"}(p) < T_{"low"}) $

其中：

- strong edge：强边缘，直接保留，是后续连接的 anchor；
- weak edge：弱边缘，是否保留取决于是否连接到 strong edge；
- non-edge：直接丢弃。

== 3.6 Step 5: Edge tracking by hysteresis

Hysteresis 的逻辑是：强边缘作为种子，弱边缘只有在与强边缘 8-connected 时才被保留。课件特别强调，强边缘是 seeds，弱边缘靠近 seeds 则生长，孤立弱边缘死亡。

形式化地，设：

$ S = {p mid M_{"nms"}(p)>= T_{"high"}}  W = {p mid T_{"low"}<= M_{"nms"}(p)<T_{"high"}} $

最终边缘集合为：

$ E = S union {p in W mid p " is connected to some " s in S " through 8-connected weak pixels"} $

其中：

- (S)：强边缘像素集合；
- (W)：弱边缘像素集合；
- (E)：最终保留的边缘集合；
- 8-connected：允许上下左右和四个对角方向连接。

Canny 的根本思想是：边缘是否真实，不只由局部强度决定，还由结构连续性决定。

---

= 4. Contour extraction：从边缘像素到边界对象

Canny 给出边缘像素，但很多任务需要的是“一个完整物体边界”。课件用 “Blind Person Touching a Wall” 类比 contour tracing：在二值图中找到边界起点，沿边界逐像素追踪，回到起点得到一条完整轮廓，然后重复直到所有边界都被找到。

可以把 contour extraction 理解为从 pixel-level edge map 到 object-level boundary representation 的转换。

设二值图为：

$ B(x,y)in{0,1} $

其中：

- (B(x,y)=1)：前景；
- (B(x,y)=0)：背景。

一个边界像素可以定义为：

$ p=(x,y) in partial Omega quad <=> quad B(p)=1 "and" exists q in cal(N)_8(p), B(q)=0 $

其中：

- $Omega$：前景区域；
- $partial Omega$：前景边界；
- $cal(N)_8(p$)：像素 (p) 的 8 邻域；
- (q)：邻域像素。

轮廓是一条有序边界序列：

$ C = (p_1,p_2,dot(s),p_n) $

满足：

$ p_{i+1}in cal(N)_8(p_i), quad quad p_n in cal(N)_8(p_1) $

即相邻边界点连通，并最终闭合。

---

= 5. Matching challenge：为什么边缘不够，为什么需要角点

== 5.1 本章后半部分的核心问题

课件从这里转向 matching：给定同一场景的两张图像，需要找对应点：

$ (x_i,y_i)<-> (x_i',y_i') $

其中：

- ((x_i,y_i))：第一张图像中的点；
- ((x_i',y_i'))：第二张图像中对应同一物理点的投影；
- (i)：第 (i) 个匹配点编号。

困难在于：像素强度、颜色会变，视角和透视几何也会变。目标是找到 invariants，即在这些变化下仍能稳定识别的 anchor。

== 5.2 为什么不能直接用边缘匹配

边缘的根本问题是 aperture problem：在一个小窗口内，边缘只提供垂直于边缘方向的变化；沿着边缘方向滑动，局部图像几乎不变。因此边缘点无法区分自己和同一条边上的邻居。课件明确指出：edge points cannot be distinguished from their neighbours。

局部窗口的变化可以粗略理解为：

$ I(x+Delta x,y+Delta y)-I(x,y) approx I_x Delta x + I_y Delta y = nabla I^top mat(Delta x; Delta y) $

如果是边缘，只有一个方向梯度大。沿边缘方向的位移 $Delta bold(x)$ 满足：

$ nabla I^top Delta bold(x) approx 0 $

因此窗口几乎不变，位置不可唯一确定。

== 5.3 好 feature 的两个硬要求

课件给出两个核心要求：repeatability 和 distinctiveness。

| 要求 | 含义 | 如果不满足会怎样 |
| --------------- | -------------- | -------------- |
| Repeatability | 不同视角、光照下仍能被检测到 | 同一物理点在另一张图里找不到 |
| Distinctiveness | 不容易和附近或其他区域混淆 | 匹配到错误位置 |

角点优于边缘，因为角点在二维方向上都有明显变化，可以提供完整的位置约束。课件总结为：flat region 有 0 个变化方向，edge 有 1 个变化方向，corner 有 2 个变化方向。

---

= 6. Harris corner：用二阶矩阵刻画“二维变化”

== 6.1 Harris 要解决的问题

Harris 的目标是数学化定义 corner。直觉上，如果一个小窗口无论往哪个方向移动，像素内容都会明显变化，那么中心点就是角点；如果只有某一个方向变化，则是边缘；如果各方向都不变，则是平坦区域。课件从 window shift 出发，用一阶 Taylor 展开推导 structure tensor。

== 6.2 Window shift energy

设窗口中心在 ((x,y))，考虑窗口平移 ((u,v)) 后的强度变化：

$ E(u,v) = sum_{(x,y)in W} w(x,y) [ I(x+u,y+v)-I(x,y) ]^2 $

符号定义：

- (E(u,v))：窗口平移 ((u,v)) 后的总强度变化；
- (W)：当前局部窗口内的像素集合；
- (w(x,y))：窗口权重，常用 Gaussian 权重，越靠近中心权重越大；
- (I(x,y))：原图像强度；
- (u,v)：窗口在水平和垂直方向上的小位移。

判别逻辑：

- 对所有方向 ((u,v))，(E(u,v)) 都小：flat；
- 某些方向大，某些方向小：edge；
- 所有方向都大：corner。

== 6.3 一阶 Taylor 展开推导

对小位移 ((u,v))，有：

$ I(x+u,y+v) approx I(x,y)+I_x(x,y)u+I_y(x,y)v $

其中：

$ I_x=frac(partial I, partial x), quad quad I_y=frac(partial I, partial y) $

所以：

$ I(x+u,y+v)-I(x,y) approx I_x u + I_y v $

代入 (E(u,v))：

$ E(u,v) approx sum_{(x,y) in W} w(x,y) (I_x u + I_y v)^2 $

展开平方：

$ E(u,v) approx sum_{(x,y) in W} w(x,y) ( I_x^2 u^2 + 2 I_x I_y u v + I_y^2 v^2 ) $

写成矩阵形式：

$ E(u,v) approx mat(u, v) M mat(u; v) $

其中 structure tensor / second-moment matrix 为：

$ M = sum_{(x,y) in W} w(x,y) mat(I_x^2, I_x I_y; I_x I_y, I_y^2) $

也可写成：

$ M= mat(sum w I_x^2, sum w I_x I_y; sum w I_x I_y, sum w I_y^2) = mat(A, C; C, B) $

其中：

- $A=sum w I_x^2$：窗口内 (x) 方向梯度能量；
- $B=sum w I_y^2$：窗口内 (y) 方向梯度能量；
- $C=sum w I_x I_y$：两个方向梯度的相关性；
- (M)：局部梯度分布的二阶统计矩阵。

物理直觉：(M) 描述局部窗口在不同方向移动时的变化强度。它不是直接保存像素值，而是保存局部梯度能量的形状。

== 6.4 特征值解释

设 (M) 的两个特征值为：

$ lambda_1,lambda_2 $

它们表示窗口沿两个主轴方向移动时的变化强度。课件中用这两个特征值分类 flat、edge、corner。

| 情况 | (lambda_1) | (lambda_2) | 解释 |
| ------ | ----------: | ----------: | --------------- |
| Flat | 小 | 小 | 任意方向移动都几乎不变 |
| Edge | 大 | 小 | 一个方向变化大，沿边方向变化小 |
| Corner | 大 | 大 | 两个方向变化都大 |

更严格地说：

$ E(u,v) approx lambda_1 alpha^2 + lambda_2 beta^2 $

其中 $(alpha,beta$) 是位移向量在 (M) 的特征向量坐标系下的表示。若两个特征值都大，则任何非零位移都会造成明显变化。

== 6.5 Harris response：避免显式求特征值

直接求 $lambda_1,lambda_2$ 需要解特征方程。Harris 用 determinant 和 trace 近似分类。课件强调：不用解特征值，直接由矩阵元素计算。

定义：

$ det(M)=lambda_1lambda_2  op("trace")(M)=lambda_1+lambda_2 $

Harris response：

$ R = det(M)-k(op("trace")(M))^2 $

展开为：

$ R = A B-C^2-k(A+B)^2 $

其中：

- (R)：Harris 角点响应；
- (k)：经验常数，通常 $k in [0.04,0.06]$；
- $ det(M) $：两个方向变化强度的乘积；
- $op("trace")(M)$：总梯度能量；
- (A,B,C)：上文定义的 structure tensor 元素。

== 6.6 为什么 (det) 和 (op("trace")) 等于特征值的积和

对 $2 times 2$ 矩阵

$ M= mat(A, C; C, B) $

特征方程为：

$ det(M-lambda I)=0 $

即：

$ det mat(A-lambda, C; C, B-lambda) =0 $

展开：

$ (A-lambda)(B-lambda)-C^2=0  lambda^2-(A+B)lambda+(A B-C^2)=0 $

另一方面，如果两个根是 $lambda_1,lambda_2$，则：

$ (lambda-lambda_1)(lambda-lambda_2) = lambda^2-(lambda_1+lambda_2)lambda+lambda_1lambda_2 $

比较系数得：

$ lambda_1+lambda_2=A+B=op("trace")(M)  lambda_1lambda_2=A B-C^2=det(M) $

这就是课件中用 Vieta’s formulas 证明 det 和 trace 的原因。

== 6.7 Harris 如何分类

$ R = lambda_1lambda_2-k(lambda_1+lambda_2)^2 $

若 flat：

$ lambda_1 approx 0, quad lambda_2 approx 0 => R approx 0 $

若 edge，例如 $lambda_1 >> lambda_2 approx 0$：

$ R approx 0 - k lambda_1^2 < 0 $

若 corner：

$ lambda_1,lambda_2 " both large" => lambda_1lambda_2 " large" => R " large positive" $

所以 Harris 的核心判据是：

$ R >> 0 => "corner"  R < 0 => "edge"  R approx 0 => "flat" $

== 6.8 Harris 后处理：NMS、threshold、minDistance

Harris 得到的是一个 response map，需要工程后处理。课件中列出三步：NMS、threshold、spatial uniformity。

=== NMS

局部邻域 $cal(N)(p$) 中只保留最大 (R)：

$ R_{"nms"}(p) = cases(R(p),, R(p)=max_{q in cal(N)(p)} R(q), 0,, "otherwise") $

=== Threshold

课件给出的阈值形式为：

$ R > R_{{"max"}}dot "qualityLevel" $

其中：

- $R_{{"max"}}=max_p R(p)$：整张图最大 Harris response；
- $"qualityLevel"$：保留角点质量比例；
- (R)：当前点 response。

OpenCV 中常见：

```python
cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
```

课件建议：

$ "qualityLevel"=0.01,quad quad "minDistance"=8"--"15 $

=== minDistance

按 (R) 从大到小排序，贪心选择角点；每选择一个角点，就删除其半径 $"minDistance"$ 内的候选点：

$ |p_i-p_j|_2 < "minDistance" => "cannot both be selected" $

这个步骤的目的不是提高单点质量，而是让角点空间分布更均匀，避免大量角点挤在同一个纹理区域。

---

= 7. Harris 到 SIFT：为什么 keypoint 还不够

== 7.1 Harris 的三个 fatal gaps

课件明确指出 Harris 有三个致命缺口：没有尺度、没有方向、没有描述子。

| Harris 缺口 | 具体症状 | SIFT 的解决方案 |
| -------------- | ------------------- | -------------------------------------- |
| No scale | 远近变化导致固定窗口看到的内容不同 | DoG scale space，keypoint 携带尺度 (sigma) |
| No orientation | 图像旋转后 patch 对不齐 | 主方向估计，描述前旋转对齐 |
| No descriptor | 只有位置，无法验证是否为同一 3D 点 | 128-D gradient histogram descriptor |

Harris 回答的是 “where is a corner?”；SIFT 回答的是 “where is a stable keypoint, at what scale and orientation, and what does its neighborhood look like?”

---

= 8. SIFT：尺度、旋转、光照鲁棒的局部特征

== 8.1 SIFT 的三大问题与三大解决方案

课件总结得非常清楚：DoG scale space 解决尺度不变性；dominant orientation alignment 解决旋转不变性；gradient orientation histogram 解决光照鲁棒性并形成 128-D descriptor。

== 8.2 Problem 1: scale invariance

同一个角点在远处和近处看起来不同。如果用固定大小窗口，远处可能看到整栋楼，近处只看到一块砖。所以 SIFT 构建尺度空间。

高斯尺度空间定义：

$ L(x,y,sigma)=G(x,y,sigma)*I(x,y) $

其中：

- $L(x,y,sigma$)：尺度 $sigma$ 下的平滑图像；
- $G(x,y,sigma$)：标准差为 $sigma$ 的高斯核；
- (I(x,y))：原图像；
- $sigma$：尺度参数。

Difference of Gaussians：

$ D(x,y,sigma) = L(x,y,k sigma)-L(x,y,sigma) $

其中：

- (D)：DoG 响应；
- (k)：相邻尺度之间的倍数；
- $ k sigma $：下一层尺度。

SIFT 在 DoG 体中寻找三维极值：

$ D(x,y,sigma) > D(x',y',sigma') quad forall (x',y',sigma')in cal(N)_{26}(x,y,sigma) $

或：

$ D(x,y,sigma) < D(x',y',sigma') quad forall (x',y',sigma')in cal(N)_{26}(x,y,sigma) $

其中：

- $cal(N)_{26}$：当前尺度同层 8 个邻居，上下尺度各 9 个邻居，总计 26 个邻居；
- 极大值或极小值都可作为候选 keypoint；
- 每个 keypoint 记录其尺度 $sigma$。

课件强调：同一个 corner 在不同距离下会在自己的 “natural scale” 被匹配。

== 8.3 Lowe’s keypoint sieve：过滤低质量点

初始 DoG 会找到很多点，课件说约 90% 是 junk，需要过滤。

=== 低对比度过滤

若 DoG 响应太小，则认为不稳定：

$ |D(bold(x))| < T_c => "discard" $

其中：

- $bold(x)=(x,y,sigma$)：尺度空间中的点；
- (T_c)：contrast threshold；
- $ D(bold(x)) $：该点 DoG 响应。

=== 边缘响应过滤

DoG 的边缘极值也可能很强，但沿边方向定位不稳定。SIFT 使用 Hessian 的主曲率比值过滤边缘点。

令 Hessian 为：

$ H= mat(D_{{"xx"}}, D_{{"xy"}}; D_{{"xy"}}, D_{{"yy"}}) $

其中：

- $D_{{"xx"}},D_{{"yy"}}$：DoG 在 (x,y) 方向的二阶导；
- $D_{{"xy"}}$：混合二阶导；
- (H)：局部二阶曲率矩阵。

设 Hessian 特征值为 $alpha,beta$，且 $alpha>= beta$。如果点是边缘，则一个曲率很大，一个曲率很小，即 $alpha/beta$ 很大。令：

$ r=frac(alpha, beta) $

Lowe 使用：

$ frac(op("Tr")(H)^2, det(H)) < frac((r+1)^2, r) $

作为保留条件。若左侧过大，则说明主曲率比例过大，是边缘响应，应丢弃。

推导如下：

$ op("Tr")(H)=alpha+beta  det(H)=alpha beta $

若 $alpha = r beta$，则：

$ frac(op("Tr")(H)^2, det(H)) = frac((r beta + beta)^2, r beta^2) frac((r+1)^2, r) $

这个逻辑与 Harris 很相似：Harris 用两个方向都大来定义 corner；SIFT 用主曲率比例过大来排除 edge-like unstable extrema。

== 8.4 Problem 2: rotation invariance

若图像旋转，patch 也旋转。SIFT 的解决方案是给 keypoint 分配主方向，再在该方向对齐后的坐标系中提取描述子。课件给出三步：计算 36-bin 梯度直方图；取峰值方向为 dominant orientation；旋转坐标系对齐该方向。

每个像素的梯度为：

$ m(x,y)=sqrt{(L(x+1,y)-L(x-1,y))^2+(L(x,y+1)-L(x,y-1))^2}  theta(x,y) = op("atan2") ( L(x,y+1)-L(x,y-1), L(x+1,y)-L(x-1,y) ) $

其中：

- (m(x,y))：尺度图像 (L) 上的梯度幅值；
- $ theta(x,y) $：梯度方向；
- (L)：keypoint 所在尺度的高斯图像。

方向直方图：

$ h(b)= sum_{(x,y)in cal(R)} w(x,y)m(x,y) dot bold(1)[theta(x,y)in "bin " b] $

其中：

- (h(b))：第 (b) 个方向 bin 的累积权重；
- $b=1,dot(s),36$，每个 bin 覆盖 $10 degree$；
- $cal(R)$：keypoint 周围圆形邻域；
- (w(x,y))：距离中心越近越大的高斯权重；
- $bold(1)[dot]$：指示函数。

主方向：

$ theta^star = op("argmax")_b h(b) $

然后将局部坐标旋转：

$ tilde(bold(x)) = R(-theta^star)(bold(x)-bold(x)_0) $

其中：

- $bold(x)_0$：keypoint 坐标；
- $R(-theta^star$)：旋转矩阵；
- $tilde(bold(x))$：主方向对齐后的局部坐标。

直觉是：不要使用图像全局的“上方”作为方向，而是使用 feature 自己的主方向作为参考系。

== 8.5 Problem 3: lighting invariance

课件指出 raw pixels 在光照变化下会彻底改变，pixel differences 也仍然敏感，而 gradient directions 更稳定。SIFT 因此使用梯度方向直方图。

SIFT descriptor 的标准构造：

+ 在尺度归一化、方向对齐后的 keypoint 周围取 $16 times 16$ 邻域；
+ 划分为 $4 times 4 = 16$ 个 cell；
+ 每个 cell 统计 8-bin 梯度方向直方图；
+ 拼接得到：

$ 16times 8 = 128 $

维向量：

$ bold(f)in R R^{128} $

L2 归一化：

$ bold(f) <- frac(bold(f), |bold(f)|_2) $

clip：

$ f_i <- min(f_i,0.2) $

再归一化：

$ bold(f) <- frac(bold(f), |bold(f)|_2) $

其中：

- $bold(f)$：SIFT descriptor；
- (f_i)：descriptor 第 (i) 维；
- clip at 0.2：减少强梯度或局部光照突变对描述子的支配；
- re-normalize：恢复单位长度。

这一步的核心是把局部 patch 从 raw appearance 转成 gradient distribution。它损失了一些精细像素信息，但换来了对亮度、对比度、微小形变更强的鲁棒性。

---

= 9. Feature matching：从描述子相似到几何一致

== 9.1 Brute-force matching

给定图像 A 的描述子集合：

$ cal(D)_A={bold(d)*i^A}*{i=1}^{N} $

图像 B 的描述子集合：

$ cal(D)_B={bold(d)*j^B}*{j=1}^{M} $

brute-force matching 对每个 $bold(d)_i^A$，寻找最近邻：

$ j^star = op("argmin")_j | bold(d)_i^A-bold(d)_j^B |_2 $

匹配为：

$ bold(d)*i^A <-> bold(d)*{j^star}^B $

复杂度：

$ O(N M) $

课件指出，brute-force 会产生许多 false matches，并且存在 scalability wall；移动 AR 等场景需要更快的检索。

== 9.2 KNN + Lowe Ratio Test

单纯最近邻的问题是：最近的不一定可靠，尤其在重复纹理中，第一近邻和第二近邻可能都很像。因此 Lowe Ratio Test 找 (k=2) 个最近邻，比较距离比值。课件给出 OpenCV 代码：

```python
if m.distance < 0.75 * n.distance:
 good.append(m)
```



数学形式：

$ frac(d_1, d_2)<tau $

或：

$ d_1 < tau d_2 $

其中：

- (d_1)：最近邻距离；
- (d_2)：第二近邻距离；
- $tau$：阈值，常用 $tau=0.75$；
- 若比值接近 0，说明第一近邻显著优于第二近邻，匹配更可靠；
- 若比值接近 1，说明第一和第二都差不多，描述子不够 distinctive。

Ratio Test 解决的是“局部描述子唯一性”问题，但它不检查多个匹配是否满足同一个几何变换。

---

= 10. RANSAC + Homography：从局部匹配到全局几何验证

== 10.1 为什么 Ratio Test 后还需要 RANSAC

课件指出，Ratio Test 只检查每个匹配本身，不检查 geometric consistency。重复纹理和局部相似 patch 仍可能留下 false matches。正确匹配必须满足同一个全局几何变换。

对于平面场景或纯相机旋转，两幅图之间可用 homography 描述。

== 10.2 Homography 公式

齐次坐标下：

$ bold(x)' ∼ H bold(x) $

其中：

$ bold(x)= mat(x; y; 1), quad quad bold(x)'= mat(x'; y'; 1)  H= mat(h_{11}, h_{12}, h_{13}; h_{21}, h_{22}, h_{23}; h_{31}, h_{32}, h_{33}) $

符号定义：

- $bold(x)$：图像 A 中点的齐次坐标；
- $bold(x)'$：图像 B 中对应点的齐次坐标；
- (H)：$3 times 3$ 单应矩阵；
- $∼$：齐次坐标下相等，即差一个非零尺度因子；
- $h_{{"ij"}}$：homography 参数。

展开为：

$ mat(tilde(x)'; tilde(y)'; tilde(w)') = H mat(x; y; 1) $

归一化：

$ x'=frac(tilde(x)', tilde(w)'), quad quad y'=frac(tilde(y)', tilde(w)') $

Homography 有 9 个元素，但整体尺度不重要，因此自由度为：

9-1=8

每对点给出两个独立方程，所以至少需要：

$ frac(8, 2)=4 $

对非共线点来求解 (H)。课件也明确写到：homography 有 8 个自由度，每个点对给 2 个方程，因此至少需要 4 对点。

== 10.3 Reprojection error

给定候选 homography (H)，匹配点 $(bold(x)_i,bold(x)_i'$) 的重投影误差为：

$ e_i = | pi(H bold(x)_i)-bold(x)_i' |_2 $

其中：

- $ pi([a,b,c]^top) = (a/c,b/c)^top $：齐次坐标转普通坐标；
- (e_i)：第 (i) 个匹配的几何误差；
- $|dot|_2$：欧氏距离。

若：

$ e_i < epsilon $

则认为该匹配是 inlier。课件示例中阈值为 5 px。

== 10.4 RANSAC 算法

RANSAC 的流程：

+ 随机采样 4 对非共线匹配点；
+ 求候选 (H)；
+ 对所有匹配计算 reprojection error；
+ 误差小于阈值的点为 inliers；
+ 重复 (N) 次；
+ 保留 inlier 数最多的 (H)；
+ 用所有 inliers 重新拟合 (H)。

形式化目标：

$ H^star = op("argmax")_H sum_i bold(1) [ | pi(H bold(x)_i)-bold(x)_i' |_2 <epsilon ] $

其中：

- $H^star$：RANSAC 选出的最佳 homography；
- $epsilon$：inlier 阈值；
- $bold(1)[dot]$：条件成立为 1，否则为 0。

如果需要估算迭代次数，常用公式是：

$ N = frac(log(1-p), log(1-w^s)) $

其中：

- (p)：希望至少一次采到全 inlier 样本的概率；
- (w)：匹配集合中 inlier 比例；
- (s)：每次采样的最小样本数，homography 中 (s=4)；
- (N)：所需 RANSAC 迭代次数。

这个公式说明：如果 inlier 比例 (w) 很低，随机采到 4 个全正确匹配的概率 $w^4$ 会很小，因此需要更多迭代。

---

= 11. Large-scale visual search：Bag of Visual Words

课件最后给出大规模视觉检索的 Bag of Visual Words。其思路是把大量局部 SIFT 描述子量化成离散 visual words，然后把每张图表示成词频直方图，用倒排索引加速检索。

== 11.1 Vocabulary building

从大量图像中提取 SIFT 描述子：

$ {bold(d)_1,bold(d)_2,dot(s),bold(d)_N}, quad quad bold(d)_i in R R^{128} $

用 K-Means 聚类成 (K) 个中心：

$ {bold(c)_1,bold(c)_2,dot(s),bold(c)_K} $

目标函数：

$ min_{{bold(c)*k}} sum*{i=1}^{N} min_{k} | bold(d)_i-bold(c)_k |_2^2 $

其中：

- $bold(c)_k$：第 (k) 个 visual word；
- (K)：词表大小，例如 10k；
- codebook：所有 visual words 的集合。

== 11.2 Encoding

每个 SIFT 描述子分配到最近的 visual word：

$ q(bold(d)_i) = op("argmin")_k | bold(d)_i-bold(c)_k |_2 $

图像表示为词频直方图：

$ bold(h) = [h_1,h_2,dot(s),h_K]^top $

其中：

$ h_k = sum_i bold(1)[q(bold(d)_i)=k] $

这样每张图像都变成一个固定长度 (K)-dim vector。

== 11.3 Retrieval

两张图像可以通过直方图相似度比较，例如余弦相似度：

$ op("sim")(bold(h)_a,bold(h)_b) = frac{bold(h)_a^top bold(h)_b} {|bold(h)_a|_2|bold(h)_b|_2} $

倒排索引的作用是：只比较共享 visual words 的图像，而不是扫描整个数据库。

---

= 12. 总结对比表

| 阶段 | 方法 | 解决什么 | 为什么前一个不够 | 核心公式/机制 | | |
| ------ | ------------------- | ------------ | -------------------- | ------------------------------------------------------------------- | -------- | -------------------- |
| 梯度估计 | Sobel | 找亮度变化 | 只能给梯度，不能给干净边缘 | (G_x=K_x*I,; G_y=K_y*I,; | nabla I | =sqrt{G_x^2+G_y^2}) |
| 干净边缘 | Canny | 单像素、连续、抗噪边缘 | Sobel 边缘厚、噪声敏感、无阈值决策 | Gaussian + NMS + dual threshold + hysteresis | | |
| 边界对象 | Contour | 把边缘像素连成轮廓 | edge map 只是散点，不是对象边界 | (C=(p_1,dot(s),p_n),; p_{i+1}incal(N)_8(p_i)) | | |
| 可匹配点 | Harris | 找二维可定位角点 | 边缘沿边方向不可定位 | (E(u,v)approx [u,v]M[u,v]^top,; R=det M-k(op(tr)M)^2) | | |
| 鲁棒局部特征 | SIFT | 尺度、旋转、光照鲁棒描述 | Harris 无尺度、无方向、无描述子 | (D=L(ksigma)-L(sigma)), 128-D histogram | | |
| 初步匹配 | BF + Ratio | 用描述子找候选对应 | 最近邻会产生大量假匹配 | (d_1/d_2\<0.75) | | |
| 几何验证 | RANSAC + Homography | 保留全局几何一致匹配 | Ratio Test 不检查几何一致性 | (bold(x)'sim H bold(x)), (e_i=|pi(Hx_i)-x_i'|_2) | | |
| 大规模检索 | BoW | 把局部特征变成图像级向量 | 暴力匹配不可扩展 | (q(d)=argmin_k|d-c_k|_2), histogram | | |

---

= 13. 最重要的底层逻辑

本章不是一堆孤立算法，而是一条逐步增强约束的链：

Sobel 只看局部梯度强度；Canny 加入局部极大值、阈值和连通性；Harris 从边缘转向二维可定位的角点；SIFT 进一步让 keypoint 携带尺度、方向和局部外观描述；Ratio Test 要求局部描述子足够独特；RANSAC 最后要求所有正确匹配服从同一个全局几何模型。

所以这章的核心思想可以压缩成一句话：

$ "Reliable matching" = "repeatable keypoints" + "distinctive descriptors" + "geometric consistency" $

前两项解决“看起来像不像”，最后一项解决“是否真的属于同一个场景变换”。
