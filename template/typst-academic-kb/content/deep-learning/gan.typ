// @title: Lecture 12 / Lecture 11: Generative Adversarial Network, GAN
// @description:
// @order: 999

= Lecture 12 / Lecture 11: Generative Adversarial Network, GAN

== 0. 本章核心目标：从“显式建模概率密度”走向“隐式生成真实样本”

本章要解决的问题是：如何学习真实数据分布 $p_{"data"}(x)$，并从中生成看起来真实的新样本。传统生成模型常常试图显式建模概率密度，例如最大化 $log p_theta(x)$。但真实图像、语音、文本等高维数据分布极其复杂，显式 likelihood 往往难以精确计算，VAE 等模型需要近似推断，因此会引入 approximation bias。

GAN 的核心转向是：不显式写出 (p_g(x)) 的密度，而是通过一个生成器 (G) 把简单噪声 $z ∼ p_z(z)$ 映射到数据空间：

$ tilde(x) = G(z;theta_g), quad quad z ∼ p_z(z) $

这里 (p_z(z)) 通常是标准高斯或均匀分布，$tilde(x)$ 是生成样本，$theta_g$ 是生成器参数。虽然我们不直接知道 (p_g(x)) 的解析密度，但 (G) 的输出隐式定义了一个生成分布 (p_g)。

本章的“进化树”可以概括为：

$ "Generative Modeling" -> cases("Explicit density models: M L E, V A E, Flow" \ "Implicit density models: G A N") -> "Adversarial learning" -> "Training issues" -> "G A N variants: Non-saturating G A N, W G A N, Unrolled G A N, C G A N, D C G A N, etc." $

---

= 1. 生成模型的基本定位

== 1.1 判别模型 vs 生成模型

判别模型学习的是：

$ p(y|x) $

也就是给定输入 (x)，预测标签 (y)。例如输入一张图片，判断它是 elephant 还是 horse。

生成模型学习的是：

$ p(x|y) $

也就是给定类别 (y)，生成符合该类别的数据 (x)。更一般地，如果不带条件，也可以学习：

$ p_{"data"}(x) $

然后从中采样生成新数据。

重要程度：★★★★★
这是 GAN 和 VAE 一类模型的出发点。考试中经常问 discriminative model 和 generative model 的区别。

== 1.2 显式生成模型与隐式生成模型

显式生成模型直接估计密度 $p_theta(x)$，通常用最大似然：

$ theta^* = op("argmax")_theta sum_{i=1}^{N}log p_theta(x_i) $

其中 (x_i) 是训练样本，(N) 是样本数，$theta$ 是模型参数。优点是目标清晰；缺点是高维复杂数据的 likelihood 很难精确计算。

隐式生成模型不直接写出 (p_g(x)) 的密度，而是定义采样过程：

$ z ∼ p_z(z), quad quad tilde(x)=G(z) $

其中 (G) 把简单分布中的样本变成复杂数据空间中的样本。GAN 属于隐式生成模型。

重要程度：★★★★★
这解释了为什么 GAN 不需要显式 likelihood，也解释了它和 VAE 的根本区别。

---

= 2. GAN 的基本结构：Generator vs Discriminator

GAN 有两个网络：

$ G(z;theta_g) $

是生成器，输入噪声 (z)，输出生成样本 $tilde(x)$。

$ D(x;theta_d) $

是判别器，输入一个样本 (x)，输出一个标量：

$ D(x) in [0,1] $

表示该样本来自真实数据分布 $p_{"data"}$ 的概率。如果 (D(x)) 越接近 1，表示判别器越认为它是真样本；如果越接近 0，表示越认为它是生成样本。

GAN 的核心博弈是：

$ G: "generate samples that fool " D  D: "distinguish real samples from fake samples" $

最终目标是：

$ p_g = p_{"data"} $

当生成分布等于真实分布时，判别器无法区分真假：

$ D(x)=frac(1, 2) $

重要程度：★★★★★
这是 GAN 的核心直觉。考试中如果问 GAN equilibrium，答案就是 $p_g=p_{"data"}$，此时 (D(x)=1/2)。

---

= 3. GAN 的原始目标函数

== 3.1 判别器目标

课件给出的判别器 cost 是：

$ J(D) = -frac(1, 2) bb(E)_{x ∼  p_{"data"}(x)} [log D(x)] ----------- frac(1, 2) bb(E)_{z ∼  p_z(z)} [log(1-D(G(z)))] $

最小化 (J(D)) 等价于最大化：

$ bb(E)_{x ∼  p_{"data"}} [log D(x)] + bb(E)_{tilde(x)∼ p_g} [log(1-D(tilde(x)))] $

其中：

$ tilde(x)=G(z), quad quad z ∼  p_z(z) $

符号解释：

$bb(E)_{x ∼ p_{"data"}}$ 表示对真实数据分布取期望。
$bb(E)_{z ∼  p_z}$ 表示对噪声分布取期望。
(D(x)) 是判别器认为 (x) 是真样本的概率。
(D(G(z))) 是判别器认为生成样本是真样本的概率。
$log D(x$) 奖励判别器把真样本判为真。
$log(1-D(G(z)))$ 奖励判别器把假样本判为假。

这个目标本质上就是二分类交叉熵：真实样本 label 为 1，生成样本 label 为 0。

重要程度：★★★★★

== 3.2 GAN minimax objective

GAN 的标准 minimax objective 是：

$ min_G max_D V(D,G) == bb(E)_{x ∼  p_{"data"}(x)} [log D(x)] + bb(E)_{z ∼  p_z(z)} [log(1-D(G(z)))] $

等价写法：

$ min_G max_D bb(E)_{x ∼  p_{"data"}} [log D(x)] + bb(E)_{tilde(x)∼ p_g} [log(1-D(tilde(x)))] $

这里 (D) 最大化该目标，因为它希望正确分类真假样本；(G) 最小化该目标，因为它希望让 (D(G(z))) 尽可能接近 1，使得 $log(1-D(G(z)))$ 变小。

重要程度：★★★★★
这是 GAN 最核心公式，必须会写、会解释。

---

= 4. Minimax Game 的理论解释

GAN 是一个 two-player zero-sum game。两个玩家是 (G) 和 (D)，目标相反。

课件中写：

$ J(D) = -frac(1, 2) bb(E)_{x ∼  p_{"data"}(x)} [log D(x)] ----------- frac(1, 2) bb(E)_{z} [log(1-D(G(z)))]  J(G)=-J(D) $

这表示最简单的零和设定中，生成器和判别器的损失互为相反数。均衡点是 (J(D)) 的 saddle point，也就是一方无法单方面改进的点。

但是这里有一个训练问题：原始 minimax 的 generator loss 可能会出现梯度消失。因为当 (D) 很强时：

$ D(G(z))approx 0 $

此时生成器从 $log(1-D(G(z)))$ 中得到的有效梯度会很弱，导致学习困难。

重要程度：★★★★☆
理论均衡很重要，但考试更可能考：为什么原始 GAN 会梯度消失，以及 non-saturating loss 怎么解决。

---

= 5. 最优判别器 ($D_G^*(x)$) 的推导

== 5.1 固定生成器 (G)，求最优判别器

对于固定的 (G)，也就是固定 (p_g)，判别器最大化：

$ V(G,D) = integral_x p_{"data"}(x) log D(x),d x + integral_z p_z(z) log(1-D(G(z))), d z $

因为 $tilde(x)=G(z$) 诱导出生成分布 (p_g(x))，所以第二项可以改写为：

$ integral_x p_g(x) log(1-D(x)),d x $

因此：

$ V(G,D) = integral_x [ p_{"data"}(x) log D(x) + p_g(x) log(1-D(x)) ]d x $

对每一个固定的 (x)，我们只需要最大化：

$ a log y + b log(1-y) $

其中：

$ a=p_{"data"}(x), quad quad b=p_g(x), quad quad y=D(x) $

求导：

$ frac(d, d y) [ a log y + b log(1-y) ] = frac(a, y) frac(b, 1-y) $

令导数为 0：

$ frac(a, y) = frac(b, 1-y)  a(1-y)=b y  a=a y+b y=(a+b) y $

所以：

$ y=frac(a, a+b) $

代回：

$ D_G^*(x) = frac(p_{"data"}(x), p_{"data"}(x)+p_g(x)) $

重要程度：★★★★★
这是 GAN 理论证明的核心。必须会推导。

== 5.2 这个公式的直觉

如果某个 (x) 在真实数据中概率大，而生成器很少生成：

$ p_{"data"}(x) >> p_g(x) $

那么：

$ D_G^*(x)approx 1 $

判别器认为它是真样本。

如果某个 (x) 主要由生成器产生，而真实数据很少出现：

$ p_g(x) >> p_{"data"}(x) $

那么：

$ D_G^*(x)approx 0 $

判别器认为它是假样本。

如果：

$ p_g(x)=p_{"data"}(x) $

那么：

$ D_G^*(x)=frac(1, 2) $

判别器完全无法区分真假。

---

= 6. GAN 的全局最优性：为什么 ($p_g=p_{"data"}$)

把最优判别器代入 GAN 目标：

$ C(G) = V(G,D^*) $
$ = [
bb(E)_{x ∼ p_{"data"}} [log D^*(x)]
+
bb(E)_{x ∼ p_g} [log(1-D^*(x))]
] $

使用：

$ D^*(x)= frac(p_{"data"}(x), p_{"data"}(x)+p_g(x)) $

以及：

$ 1-D^*(x) = frac(p_g(x), p_{"data"}(x)+p_g(x)) $

得到：

$ C(G) = bb(E)_{x ∼  p_{"data"}} [ log frac(p_{"data"}(x), p_{"data"}(x)+p_g(x)) ] + bb(E)_{x ∼  p_g} [ log frac(p_g(x), p_{"data"}(x)+p_g(x)) ] $

定义混合分布：

$ M(x)=frac(p_{"data"}(x)+p_g(x), 2) $

注意：

$ p_{"data"}(x)+p_g(x)=2M(x) $

所以：

$ log frac(p_{"data"}(x), p_{"data"}(x)+p_g(x)) == log frac{p_{"data"}(x)} {2M(x)} = log frac(p_{"data"}(x), M(x)) ------------------------------- log 2 $

同理：

$ log frac(p_g(x), p_{"data"}(x)+p_g(x)) == log frac(p_g(x), M(x)) ------------------- log 2 $

因此：

$ C(G) = -log 4 + K L( p_{"data"} | frac(p_{"data"}+p_g, 2) ) + K L( p_g | frac(p_{"data"}+p_g, 2) ) $

而 Jensen-Shannon Divergence 定义为：

$ J S(p_{"data"}|p_g) == frac(1, 2) K L(p_{"data"}|M) + frac(1, 2) K L(p_g|M) $

所以：

$ C(G) = -log 4 + 2J S(p_{"data"}|p_g) $

因为：

$ J S(p_{"data"}|p_g)>= 0 $

且当且仅当：

$ p_g=p_{"data"} $

时取 0，所以全局最优为：

$ C(G)=-log 4 $

并且：

$ p_g=p_{"data"}, quad quad D^*(x)=frac(1, 2) $

重要程度：★★★★★
这是本章最重要的理论证明之一。考试可能直接要求证明 GAN objective 等价于 minimi in g JSD。

---

= 7. GAN 的训练过程

GAN 实际训练时不会真的让 (D) 完全达到最优，而是交替优化：

先固定 (G)，更新 (D)：

$ max_D bb(E)_{x ∼  p_{"data"}} [log D(x)] + bb(E)_{tilde(x)∼ p_g} [log(1-D(tilde(x)))] $

再固定 (D)，更新 (G)：

$ min_G bb(E)_{tilde(x)∼ p_g} [log(1-D(tilde(x)))] $

训练步骤可以理解为：

$ z ∼ p_z(z) -> tilde(x)=G(z) -> D(x),D(tilde(x)) -> "update "D -> "update "G $

真实训练中通常用 Adam 或 SGD-like optimizer。每次训练同时使用两个 minibatch：一个真实样本 minibatch，一个生成样本 minibatch。

重要程度：★★★★★
GAN 训练伪代码、交替优化、两个 minibatch 是常考点。

---

= 8. GAN 的主要训练问题

== 8.1 Vanishing Gradients

原始 minimax generator loss 是：

$ J_G^{"minimax"} == bb(E)_{z ∼  p_z} [log(1-D(G(z)))] $

生成器希望最小化它。问题在于，当判别器太强时：

$ D(G(z))approx 0 $

此时 fake samples 被判别器高置信度识别为假。虽然 loss 数值可能不小，但通过 sigmoid classifier 反传到生成器的梯度容易饱和，导致生成器几乎无法更新。

解决方法之一是 non-saturating loss。

重要程度：★★★★★

== 8.2 Non-Convergence

GAN 是非凸-非凹 minimax optimization。即使理论上有 equilibrium，参数空间中的 simultaneous gradient descent 也不保证收敛。

原因包括：

$ D_{theta_d},G_{theta_g} $

都是神经网络，目标高度非凸。实际训练中，(G) 和 (D) 的更新可能互相抵消，导致 oscillation。

重要程度：★★★★☆

== 8.3 Mode Collapse

Mode collapse 指生成器只覆盖真实分布的一小部分模式。例如真实数据有 9 个模式，但 GAN 只学会生成其中 1 个模式。

形式上，真实分布 $p_{"data"}$ 可能有多个高概率区域：

$ cal(M)_1,cal(M)_2,..,cal(M)_K $

但生成分布 (p_g) 只覆盖其中少数几个：

$ p_g " covers only " cal(M)_j $

这会导致样本看起来局部真实，但多样性不足。

重要程度：★★★★★
mode collapse 是 GAN 最经典缺陷之一，考试非常可能问。

== 8.4 Mode Dropping

Mode dropping 和 mode collapse 类似，但更强调生成分布没有覆盖完整真实分布。它可能不是完全崩到一个模式，而是漏掉一些数据模式。

重要程度：★★★★☆

---

= 9. Non-Saturating GAN：解决梯度消失

原始 discriminator loss 保持不变：

$ J(D) = -frac(1, 2) bb(E)_{x ∼  p_{"data"}(x)} [log D(x)] ----------- frac(1, 2) bb(E)_{z} [log(1-D(G(z)))] $

但 generator loss 改为：

$ J(G) = -frac(1, 2) bb(E)_{z} [log D(G(z))] $

这个目标不再是最小化：

$ log(1-D(G(z))) $

而是最大化：

$ log D(G(z)) $

直觉是：原始 minimax 是让 (G) 减少 (D) 正确识别 fake 的概率；non-saturating 是让 (G) 直接增加 (D) 把 fake 判为 real 的概率。

当：

$ D(G(z))approx 0 $

时：

$ -log D(G(z)) $

非常大，梯度信号强，因此生成器仍然能学习。

重要程度：★★★★★
必须会区分 minimax loss 和 non-saturating loss。

对比表：

| 目标 | Discriminator loss | Generator loss | 主要问题/优点 |
| ----------------------- | -------------------------------------------------------- | ----------------------------------------------------- | ----------------------- |
| Original minimax GAN | (-bb(E)_{x}log D(x)-bb(E)_{z}log(1-D(G(z)))) | (bb(E)_{z}log(1-D(G(z)))) | 理论漂亮，但 (D) 太强时 (G) 梯度消失 |
| Non-saturating GAN | 同上 | (-bb(E)_{z}log D(G(z))) | 梯度更强，实践中更常用 |
| Maxi mum-likelihood game | 同上 | ($-frac(1, 2) bb(E)_z exp(sigma^{-1}(D(G(z))))$) | 试图模拟 MLE / op("KL") 目标 |

---

= 10. Maximum Likelihood Game

课件给出的 generator loss 是：

$ J(G) = -frac(1, 2) bb(E)_z exp ( sigma^{-1}(D(G(z))) ) $

其中 $sigma^{-1}$ 是 sigmoid 的 inverse，也就是 logit function：

$ sigma^{-1}(u) = log frac(u, 1-u) $

这个目标被解释为等价于最小化：

$ theta^* = op("argmin")_theta D_{op("KL")} ( p_{"data"}(x) | p_{"model"}(x;theta) ) $

即：

$ D_{op("KL")}(p_{"data"}|p_{"model"}) === integral p_{"data"}(x) log frac{p_{"data"}(x)} {p_{"model"}(x)} d x $

符号解释：

$p_{"model"}(x;theta$) 是生成模型分布。
$D_{op("KL")}(p|q$) 衡量用 (q) 近似 (p) 的信息损失。
这个 op("KL") 是 forward op("KL")，倾向于覆盖真实数据的所有模式，因此和 mode dropping 问题有关。

重要程度：★★★☆☆
课件提到，但通常不会比 non-saturating 和 WGAN 更核心。

---

= 11. WGAN：从 JSD 到 Wasserste in Distance

== 11.1 为什么 JSD 会出问题

前面证明了原始 GAN 在理想情况下等价于最小化：

$ J S(p_{"data"}|p_g) $

但在高维空间中，真实数据往往分布在低维流形上。例如图像虽然维度很高，但有效图像只占据一个低维 manifold。生成分布 (p_g) 也可能在另一个低维 manifold 上。

如果两个分布的支撑集几乎不重叠，那么 JSD 会饱和，导致梯度几乎为 0。也就是说，即使 (p_g) 离 $p_{"data"}$ 很远，JSD 也无法提供连续有效的优化方向。

重要程度：★★★★★

== 11.2 Wasserstein-1 Distance / Earth-Mover Distance

WGAN 使用 1-Wasserste in distance：

$ W(P_r,P_g) = inf_{gamma in Pi(P_r,P_g)} bb(E)_{(x,y)∼ gamma} [ |x-y| ] $

符号解释：

(P_r) 是真实分布，等同于 $p_{"data"}$。
(P_g) 是生成分布。
$Pi(P_r,P_g)$ 是所有 coupling distributions 的集合。
$gamma(x,y)$ 表示一种“运输计划”，说明如何把 (P_r) 中的质量搬运到 (P_g)。
(|x-y|) 是把质量从 (x) 移到 (y) 的代价。
$inf$ 表示在所有运输计划中找最小运输成本。

直觉：Wasserste in distance 衡量的是“把一个分布搬成另一个分布需要付出多少最小代价”。即使两个分布没有重叠，它仍然能给出平滑、可用的距离信号。

重要程度：★★★★★

== 11.3 课件中的例子

设：

$ Z ∼ U[0,1] $

真实分布：

$ p_{"data"}=(0,Z)in R R^2 $

生成分布：

$ p_g=(theta,Z) $

这表示真实样本在二维平面上的竖线 (x=0)，生成样本在竖线 $x=theta$。如果 $theta != 0$，两个分布支撑在两条平行线，几乎不重叠。JSD 容易饱和，而 Wasserste in distance 可以自然得到：

$ W(p_{"data"},p_g)=|theta| $

因此对 $theta$ 有有效梯度。

重要程度：★★★★☆

== 11.4 Kantorovich-Rubinste in Duality

直接优化 Wasserste in distance 中的 $inf_{{"gamma"}}$ 很难，因此 WGAN 使用对偶形式：

$ W(P_r,P_g) = sup_{|f|_L<= 1} bb(E)_{x ∼  P_r}[f(x)] ---------------------------- bb(E)_{x ∼  P_g}[f(x)] $

其中：

$ |f|_L<= 1 $

表示 (f) 是 1-Lipschitz function，即：

$ |f(x_1)-f(x_2)| <= |x_1-x_2| $

如果是 (K)-Lipschitz：

$ |f(x_1)-f(x_2)| <= K|x_1-x_2| $

那么目标会变成：

$ K * W(P_r,P_g) $

在 WGAN 中，(f) 用 neural network 参数化，通常称为 critic，而不是 discriminator，因为它不输出概率，而是输出一个 real-valued score。

重要程度：★★★★★

== 11.5 如何保证 Lipschitz 条件

课件提到两种方式：

$ "Gradient Clipping" $

和：

$ "Gradient Penalty" $

Gradient clipping 通过限制 critic 参数范围来近似满足 Lipschitz 条件，但可能导致 capacity 受限。Gradient penalty 则直接惩罚梯度范数偏离 1，实践中更稳定。

虽然课件没有写 WGAN-GP 公式，但常见形式是：

$ lambda bb(E)_{hat(x)} [ (|nabla hat(x)D(hat(x))|_2-1)^2 ] $

其中 $hat(x)$ 是真实样本和生成样本之间插值得到的点，$lambda$ 是 penalty coefficient。

重要程度：★★★★☆
课件只提到了 gradient clipping 和 gradient penalty，但考试可能要求解释它们为什么和 Lipschitz 约束有关。

---

= 12. Unrolled GAN：缓解 mode collapse

Unrolled GAN 的核心思想是：更新生成器时，不只看当前判别器 (D)，而是考虑判别器未来经过 (k) 步优化后的状态。

普通 GAN 中，(G) 可能 exploit 当前 (D) 的弱点，于是生成器学会某几个模式来骗过当前判别器，造成 mode collapse。

Unrolled GAN 做法是：

$ theta_D^0 = theta_D  theta_D^{i+1} = theta_D^i eta nabla_{theta_D^i} J_D(theta_D^i,theta_G), quad quad i=0,..,k-1 $

然后用第 (k) 步后的判别器参数：

$ theta_D^k $

来计算 generator objective，并对 $theta_G$ 反传：

$ J_G(theta_G,theta_D^k) $

关键是，更新 (G) 时会 backpropagate through discriminator’s (k)-step optimization。这样 (G) 不能只骗当前 (D)，而要考虑 (D) 反应后的结果，因此可以减少 mode collapse。

课件特别强调：虽然为了更新 (G) 会 unroll (k) steps of (D)，但真正保留到下一轮训练的只有 (D) 的第一次更新，而不是全部 (k) 次。这是为了避免 (D) 过拟合。

重要程度：★★★★☆

---

= 13. Conditional GAN 与 Image-to-Image Translation

== 13.1 Conditional GAN

普通 GAN 只从噪声生成样本：

$ tilde(x)=G(z) $

Conditional GAN 引入条件 (y) 或输入图像 (x)，让生成过程可控。例如：

$ tilde(y)=G(x,z) $

判别器输入 pair：

$ D(x,y) $

判断 (y) 是否是给定 (x) 的真实 translation。

重要程度：★★★★☆

== 13.2 Pix2Pix-style objective

课件中 image-to-image translation 使用 conditional GAN，并加入 L1 loss：

$ L_{op("L1")}(G) = bb(E)_{x,y ∼  p_{"data"}(x,y),z ∼  p_z(z)} [ |y-G(x,z)|_1 ] $

总目标：

$ L = L_{"G A N"}(G,D) + lambda L_{op("L1")}(G) $

符号解释：

(x) 是输入图像，例如边缘图、语义分割图、低分辨率图像。
(y) 是目标图像。
(G(x,z)) 是生成的目标域图像。
($|y-G(x,z)|_1$) 是像素级 L1 reconstruction loss。
$lambda$ 控制 GAN loss 和 L1 loss 的权重。

L1 loss 的作用是保证生成结果和 paired ground truth 对齐；GAN loss 的作用是让生成图像更真实、更符合目标域分布。

课件指出缺陷：

$ z " is useless for " G $

也就是说，在 paired image-to-image translation 中，生成器往往忽略噪声 (z)，输出 deterministic result，导致没有 stochastic outputs。

重要程度：★★★★★
CGAN + L1 是图像翻译、超分辨率、视频预测等任务的共同模板。

---

= 14. 其他 GAN variants 与应用

== 14.1 DCGAN

DCGAN 是 Deep Convolutional GAN。它把 generator 和 discriminator 改成卷积结构，更适合图像生成。Generator 常用反卷积或上采样结构，Discriminator 用 CNN 提取图像特征。

重要程度：★★★☆☆

== 14.2 InfoGAN

InfoGAN 的目标是无监督地学习 disentangled representations。它把 latent code 分成普通噪声和可解释因素，希望某些 latent variables 控制语义因素，例如数字类别、旋转角度、笔画粗细。

重要程度：★★★☆☆

== 14.3 LSGAN

LSGAN 使用 least squares loss 替代 binary cross-entropy loss。其动机是缓解梯度饱和，让 fake samples 即使在判别边界之外也有较强梯度。

重要程度：★★★☆☆

== 14.4 BEGAN

BEGAN 使用 equilibrium enforcing mechanism，并结合 Wasserstein-inspired loss，目标是稳定训练。

重要程度：★★☆☆☆

== 14.5 Progressive Growing GAN

Progressive Growing GAN 的思想是从低分辨率开始训练，例如：

$ 4times 4 -> 8times 8 -> 16times 16 -> .. -> 1024times 1024 $

逐渐增加生成器和判别器的层数，使训练更稳定，并生成高分辨率图像。

重要程度：★★★☆☆

---

= 15. GAN 的应用

GAN 的应用包括：

图像生成：从随机噪声生成 MNIST、CIFAR-10、人脸等图像。
Latent vector arithmetic：在 latent space 中做语义运算，例如：

$ z_{"smiling woman"} ------------------------ z_{"neutral woman"} + z_{"neutral man"} approx z_{"smiling man"} $

Image-to-image translation：例如边缘图到真实图像、语义图到街景。
Next video frame prediction：给定过去帧，预测未来帧。
Super-resolution：低分辨率图像生成高分辨率图像，是 conditional GAN 的特殊形式。
Unpaired image-to-image translation：例如 CycleGAN，不需要 paired data。
Semi-supervised learning：用 discriminator/classifier 的 feature matching 帮助利用少量标签。
Discrete sequential data generation：把生成器视为 policy，用 policy gradient 处理不可微采样。

重要程度：★★★☆☆
应用一般不会考特别细，但 conditional GAN、super-resolution、discrete sequence generation 可能会出概念题。

---

= 16. 离散序列生成：GAN + Policy Gradient

课件给出公式：

$ nabla J(theta) = sum_{y in Y} ( nabla G_theta(y_t|Y_{1:t-1}) ) Q_{D_phi}^{G_theta}(Y_{1:t-1},y_t) $

其中：

$G_theta$ 是生成器，也可以看作 policy。
(y_t) 是第 (t) 个 token/action。
$Y_{1:t-1}$ 是已经生成的前缀序列。
$D_phi$ 是判别器，提供 reward。
$Q_{D_phi}^{G_theta}(Y_{1:t-1},y_t$) 是 action-value function，表示在前缀 $Y_{1:t-1}$ 后选择 (y_t) 的长期回报。

课件给出的 (Q) 定义是：

$ Q_{D_phi}^{G_theta}(Y_{1:t-1},y_t) === cases(D_phi(Y_{1:T}), t=T, frac(1, N) sum_(n=1)^(N) D_phi(Y_{1:T}^(n)), t<T) $

注意课件中写了 ($t<N$)，但从语义上更自然的是 ($t<T$)：如果序列还没生成完，就用 Monte Carlo rollout 补全序列，再由判别器打分。

直觉：离散 token 采样不可微，所以不能直接把 (D) 的梯度传回 (G)。因此把 (G) 当作 policy，把 (D) 的输出当 reward，用 policy gradient 更新。

重要程度：★★★★☆
如果课程覆盖 SeqGAN，这个公式可能考。关键是理解：离散采样不可微，所以要用 policy gradient。

---

= 17. 方法演进总表

| 阶段 | 核心方法 | 解决什么问题 | 新问题 |
| ------------------ | --------------------------- | -------------------------------------- | -------------------------------- |
| 显式生成模型 | MLE, VAE | 直接建模 (p_theta(x))，目标清晰 | 高维 likelihood 难算；VAE 有近似偏差 |
| 原始 GAN | (min_G max_D V(D,G)) | 不显式建模密度，直接学习生成样本 | 训练不稳定、梯度消失、mode collapse |
| Non-saturating GAN | ($-bb(E)_z log D(G(z))$) | 缓解 (D) 太强时 (G) 梯度消失 | 仍可能 non-converge 和 mode collapse |
| Unrolled GAN | 反传穿过 (k) 步 (D) 更新 | 减少 (G) exploit 当前 (D)，缓解 mode collapse | 计算成本更高 |
| WGAN | Wasserste in distance | JSD 饱和时仍提供有意义梯度 | 需要 Lipschitz 约束 |
| WGAN-GP / clipping | gradient penalty / clipping | 近似满足 Lipschitz 条件 | clipping 可能限制 capacity |
| CGAN / Pix2Pix | 条件输入 + GAN + L1 | 可控生成、图像翻译 | 可能忽略噪声 (z)，输出缺少多样性 |
| Progressive GAN | 逐步提高分辨率 | 高分辨率图像训练更稳定 | 架构和训练流程更复杂 |

---

= 18. 期末考试重点排序

最高优先级：必须掌握

$ min_G max_D bb(E)_{x ∼  p_{"data"}}log D(x) + bb(E)_{z ∼  p_z}log(1-D(G(z)))  D_G^*(x)= frac(p_{"data"}(x), p_{"data"}(x)+p_g(x))  C(G) = -log 4 + 2J S(p_{"data"}|p_g)  p_g=p_{"data"} => D(x)=frac(1, 2)  J_G^{"non-saturating"} == -bb(E)_z log D(G(z))  W(P_r,P_g) = inf_{gamma in Pi(P_r,P_g)} bb(E)_{(x,y)∼ gamma}|x-y|  W(P_r,P_g) = sup_{|f|_L<= 1} bb(E)_{x ∼  P_r}f(x) -------------------------- bb(E)_{x ∼  P_g}f(x) $

中高优先级：需要能解释

Mode collapse、mode dropping、vanishing gradient、non-convergence 的区别。
为什么原始 GAN 理论上好，但实践训练难。
为什么 WGAN 比 JSD 更适合高维 manifold。
为什么 conditional GAN 需要把 condition 输入给 (G) 和 (D)。
为什么 Pix2Pix 需要 (L_1) loss。
为什么离散序列生成不能直接 backprop，需要 policy gradient。

较低优先级：了解即可

InfoGAN、BEGAN、Progressive GAN、latent vector arithmetic、semi-supervised GAN 的细节。

---

= 19. 一句话总结本章

GAN 的核心思想是：用一个判别器 (D) 学习真实分布和生成分布之间的 density ratio，用这个判别信号训练生成器 (G)，使隐式生成分布 (p_g) 逐渐逼近真实数据分布 $p_{"data"}$。理论上，最优判别器诱导出的目标等价于最小化 Jensen-Shannon divergence；实践中，原始 GAN 会遭遇梯度消失、非收敛和 mode collapse，因此发展出了 non-saturating loss、Unrolled GAN、WGAN、CGAN 等一系列改进方法。
