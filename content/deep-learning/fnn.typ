// @title: Lecture 3 总体进化树
// @description: x → h = f(x; θ) → P(y | h)
// @order: 999

= Lecture 3 总体进化树

```text
监督学习概率模型
P(y | x)
 ↓
深度学习视角
x → h = f(x; θ) → P(y | h)
 ↓
Feedforward Neural Network / MLP
多层仿射变换 + 非线性激活
 ↓
训练问题
min_θ J(θ)
 ↓
反向传播
高效计算所有参数梯度
 ↓
优化算法
SGD → Momentum → AdaGrad/RMSProp → Adam
 ↓
泛化控制
Weight decay + Dropout
 ↓
理论问题
Universal approximation + approximation rate + SGD convergence
```

重要程度标注：
★★★：期末核心，必须掌握公式与推导。
★★：重要概念，常用于解释或简答。
★：理解即可，通常不深考。

= 1. 从传统概率模型到 FNN：本章核心目标 ★★★

之前的监督学习通常直接学习：

$ {x_i,y_i}_{i=1}^N -> P(y mid x) $

深度学习把这个过程拆成两部分：

$ {x_i,y_i}_{i=1}^N -> h=f(x),quad P(y mid h) $

其中：

$ h=f(x) $

是神经网络学到的特征表示；而

$ P(y mid h) $

是在特征空间上的概率模型。整体仍然是：

$ P(y mid x,theta) $

其中 $theta$ 包括所有隐藏层参数和输出层概率模型参数。

逻辑直觉：传统机器学习依赖人工特征；FNN 的目标是自动学习特征变换 (h=f(x))，让后面的分类或回归模型更容易工作。

= 2. Feedforward Neural Network 基本结构 ★★★

一个 FNN 定义函数：

$ h=f(x,theta) $

训练目标是让它逼近某个目标函数：

$ f(x,theta)approx f^*(x) $

一个神经元接受输入向量 (x)，先做仿射变换：

$ z=w^->p x+b $

再经过非线性激活：

$ g(z)=g(w^->p x+b) $

符号说明：

$ x=(x_1,dot(s),x_n)^->p $

是输入向量；

$ w=(w_1,dot(s),w_n)^->p $

是权重向量；

$ b $

是偏置；

$ z $

称为 net input，即激活前输入；

$ g(dot) $

是激活函数。

如果网络有 (L) 层，则整体是函数复合：

$ h(x)=f^{(L)}(f^{(L-1)}(.. f^{(1)}(x))) $

关键点：如果所有层都是线性的，那么多层线性函数复合仍然是线性函数。因此深度本身不够，必须加入非线性激活。

= 3. 权重初始化 ★★★

初始化的目标是避免前向信号或反向梯度在层间爆炸或消失。

== Xavier initialization

$ w ~ U(-frac(1, sqrt n), frac(1, sqrt n)) $

其中 (n) 是该神经元的输入维度。通常用于 tanh。

直觉：保持每层输入输出方差大致稳定。

== He / Kaiming initialization

课件写作：

$ w ~ N[0,frac(2, sqrt n)] $

更常见标准表述是：

$ w ~ N(0,frac(2, n)) $

或标准差为：

$ sqrt{frac(2, n)} $

通常用于 ReLU，因为 ReLU 会把一部分负值截断为 0，需要更大的方差补偿。

= 4. 激活函数演进：从 Sigmoid 到 GELU ★★★

激活函数的核心问题是：既要提供非线性表达能力，又要保证梯度好传播。

== 4.1 Sigmoid：概率解释强，但隐藏层不推荐 ★★★

$ g(z)=sigma(z)=frac(1, 1+exp(-z)) $

导数为：

$ sigma'(z)=sigma(z)(1-sigma(z)) $

问题：当 $z >> 0$ 或 $z << 0$ 时，$sigma(z)$ 接近 1 或 0，导数接近 0，产生 vanishing gradient。

所以 sigmoid 不推荐作为隐藏层激活，但适合二分类输出层。

== 4.2 ReLU：解决梯度消失，但引入 dead neuron ★★★

$ g(z)=max{0,z} $

导数为：

$ g'(z)= cases(1 "if" z > 0 "," 0 "if" z < 0) $

优点：当 z > 0 时梯度恒为 1，训练更快。

缺点：如果某个神经元对所有训练样本都有 z < 0，则梯度为 0，参数无法更新，称为 dead neuron。

缓解方法：把 bias 初始化为小正数，例如：

$ b=0.1 $

== 4.3 ReLU variants：给负半轴保留梯度 ★★

统一形式：

$ g(z,alpha)=max{0,z}+alpha min{0,z} $

不同取值：

$ alpha=-1 $

是 absolute value rectification；

$ alpha=0.01 $

是 Leaky ReLU；

$ alpha $

可学习时是 Parametric ReLU。

核心改进：当 $z<0$ 时仍有非零梯度，缓解 dead neuron。

== 4.4 Tanh：零中心但仍会饱和 ★★

$ g(z)=tanh(z)=frac(exp(z)-exp(-z), exp(z)+exp(-z)) $

相比 sigmoid，tanh 输出范围是 ([-1,1])，零中心，梯度通常更大。但当 (|z|) 很大时仍然饱和。

== 4.5 Swish, Mish, GELU：现代深层网络常用 ★★

Swish：

$ g(z)=z sigma(beta z) $

其中 $beta$ 可以固定或学习。

Mish：

$ g(z)=z tanh(log(1+exp(z)))=z tanh(zeta(z)) $

其中：

$ zeta(z)=log(1+exp(z)) $

是 softplus。

GELU：

$ g(z)=z Phi(z) $

其中 $Phi(z)$ 是标准高斯分布的 CDF。

共同特点：平滑、负半轴非零梯度、非单调、上方无界，因此深层网络中通常比 ReLU 更稳定。

== 4.6 Softplus：ReLU 的光滑版本 ★

$ g(z)=zeta(z)=log(1+exp(z)) $

它是 ReLU 的平滑近似，但经验上通常不如 ReLU。

= 5. 输出层概率模型 ★★★

FNN 的前面层给出：

$ h=f(x) $

最后一层先产生 logits：

$ z=W^->p h+b $

其中 (W) 是权重矩阵，(b) 是偏置向量，$z=(z_1,z_2,dot(s))^->p$。

== 5.1 Linear-Gaussian 输出：回归 ★★★

当 (y) 是实值向量时：

$ p(y mid x)=cal N(y mid z,I) $

负对数似然为：

$ L(x,y,theta)=-log P(y mid x,theta)prop frac(1, 2)|y-z|_2^2 $

对 logit (z_k) 求导：

$ frac(partial L, partial z_k)=z_k-y_k $

直觉：预测值减真实值，就是回归误差。

== 5.2 Sigmoid 输出：二分类 ★★★

当：

$ y in {0,1} $

使用 Bernoulli：

$ P(y mid x)=upright("Bernoulli")(y mid sigma(z)) =sigma(z)^y(1-sigma(z))^{1-y} $

损失为：

$ L(x,y,theta)=-[y log sigma(z)+(1-y)log(1-sigma(z))] $

关键导数：

$ frac(partial L, partial z)=sigma(z)-y $

推导直觉：交叉熵和 sigmoid 的导数会抵消一部分复杂项，因此输出层即使 sigmoid 饱和，也只有在模型已经答对且非常自信时梯度才接近 0。

== 5.3 Softmax 输出：多分类 ★★★

当：

$ y in {1,2,dot(s),C} $

softmax 定义为：

$ P(y=k mid x)=frac(exp(z_k), sum_{j=1}^C exp(z_j)) $

交叉熵损失对 logit 的导数为：

$ frac(partial L, partial z_k)=P(y=k)-bold(1)(y=k) $

其中 $bold 1(y=k$) 是指示函数。

直觉：梯度 = 预测概率 − 真实 one-hot 标签。

= 6. 训练目标与 SGD ★★★

总体经验风险：

$ J(theta)=frac(1, N)sum_{i=1}^N L(x_i,y_i,theta) =frac(1, N)sum_{i=1}^N -log P(y_i mid x_i,theta) $

SGD minibatch 更新：

$ theta arrow.l theta - alpha frac(1, |B|) sum_{i in B} nabla L(x_i,y_i,theta) $

其中 (B) 是 minibatch，$alpha$ 是学习率。

核心问题：如何高效计算所有参数的梯度？

答案：Backpropagation。

= 7. Backpropagation 反向传播 ★★★

课件使用三层局部结构：

$ z_k=sum_j u_j W_{{"kj"}}  u_j=g(z_j)  z_j=sum_i u_i W_{{"ji"}} $

其中：

(u_i)：上一层第 (i) 个单元输出；
(z_j)：当前隐藏单元 (j) 的激活前输入；
(u_j)：隐藏单元 (j) 的激活后输出；
(z_k)：下一层或输出层单元 (k) 的激活前输入；
$W_{{"ji"}}$：从 (i) 到 (j) 的权重；
$W_{{"kj"}}$：从 (j) 到 (k) 的权重。

== 7.1 输出层梯度

$ frac(partial L, partial W_{{"kj"}}) = frac(partial L, partial z_k) frac(partial z_k, partial W_{{"kj"}}) $

因为：

$ z_k=sum_j u_j W_{{"kj"}} $

所以：

$ frac(partial z_k, partial W_{{"kj"}})=u_j $

定义 error signal：

$ delta_k=frac(partial L, partial z_k) $

因此：

$ frac(partial L, partial W_{{"kj"}})=u_j delta_k $

== 7.2 隐藏层梯度

$ frac(partial L, partial W_{{"ji"}}) = sum_k frac(partial L, partial z_k) frac(partial z_k, partial W_{{"ji"}}) $

链式展开：

$ frac(partial L, partial W_{{"ji"}}) = sum_k delta_k frac(partial z_k, partial u_j) frac(partial u_j, partial z_j) frac(partial z_j, partial W_{{"ji"}}) $

其中：

$ frac(partial z_k, partial u_j)=W_{{"kj"}}  frac(partial u_j, partial z_j)=g'(z_j)  frac(partial z_j, partial W_{{"ji"}})=u_i $

所以：

$ frac(partial L, partial W_{{"ji"}}) = u_i ( g'(z_j) sum_k W_{{"kj"}} delta_k ) $

定义：

$ delta_j= frac(partial u_j, partial z_j) sum_k W_{{"kj"}} delta_k = g'(z_j) sum_k W_{{"kj"}} delta_k $

得到：

$ frac(partial L, partial W_{{"ji"}})=u_i delta_j $

偏置梯度：

$ frac(partial L, partial b_j)=delta_j $

核心直觉：每个权重的梯度 = 输入激活 × 目标单元误差信号。

= 8. 优化算法演进 ★★★

== 8.1 SGD：只看当前梯度 ★★★

$ g arrow.l frac(1, m) nabla_theta sum_i L(f(x^{(i)};theta),y^{(i)})  theta arrow.l theta - epsilon g $

问题：梯度噪声大，路径震荡，尤其在狭长 valley 中收敛慢。

== 8.2 Momentum：引入速度，减少震荡 ★★★

$ g arrow.l frac(1, m) nabla_theta sum_i L(f(x^{(i)};theta),y^{(i)})  v arrow.l alpha v - epsilon g  theta arrow.l theta + v $

其中 (v) 是速度，$alpha$ 是 momentum 系数，常用：

$ alpha=0.5,\ 0.9,\ 0.99 $

直觉：参数像物理粒子一样在 loss landscape 中运动。过去梯度方向会累积，随机震荡会相互抵消，稳定方向会被加强。

== 8.3 AdaGrad：每个参数自适应学习率 ★★

累积平方梯度：

$ r arrow.l r + g ∘ g $

更新：

$ Delta theta arrow.l -frac(epsilon, delta+sqrt r) ∘ g  theta arrow.l theta + Delta theta $

其中 $∘$ 表示 element-wise multiplication。

优点：对稀疏特征友好，历史梯度小的参数学习率较大。

问题：(r) 单调增加，学习率会越来越小，后期可能几乎不动。

== 8.4 RMSProp：遗忘太久远历史 ★★★

$ r arrow.l rho r + (1-rho) g ∘ g  Delta theta arrow.l -frac(epsilon, sqrt{delta+r}) ∘ g  theta arrow.l theta + Delta theta $

相比 AdaGrad，RMSProp 使用指数滑动平均，不会无限累积历史梯度。

问题：它有 adaptive scaling，但没有显式 momentum，因此方向仍可能噪声较大。

== 8.5 Adam：Momentum + RMSProp ★★★

Adam 同时维护一阶矩和二阶矩：

$ s arrow.l rho_1 s + (1-rho_1) g  r arrow.l rho_2 r + (1-rho_2) g ∘ g $

bias correction：

$ hat(s)=frac(s, 1-rho_1^t)  hat(r)=frac(r, 1-rho_2^t) $

更新：

$ Delta theta arrow.l -epsilon frac(hat(s), sqrt{hat(r)}+delta)  theta arrow.l theta + Delta theta $

默认值：

$ epsilon=0.001,quad rho_1=0.9,quad rho_2=0.999,quad delta=10^{-8} $

直觉：$hat(s)$ 像 momentum，稳定方向；$hat(r)$ 像 RMSProp，自适应缩放每个维度。

= 9. Learning Rate Scheduling ★★

优化器决定“如何使用梯度”，learning rate schedule 决定“步长如何随时间变化”。

常见策略：

```text
固定学习率
 ↓
逐步衰减 step decay
 ↓
指数衰减 exponential decay
 ↓
cosine annealing / warm restart
```

Cosine annealing 的直觉：周期性降低再重启学习率，相当于模拟重新开始搜索，但使用已有较好权重作为 warm start。

= 10. Overfitting 与 Dropout ★★★

深度模型参数多，容易 overfit。常用 weight decay：

$ tilde(J)(theta)=J(theta)+lambda|theta|_2^2 $

其中 $lambda$ 控制正则化强度。

== 10.1 Bagging 思想 ★★

Bagging 训练多个模型，每个模型使用不同 bootstrap sample，预测时投票。它降低 variance，因为不同模型不太可能犯同样错误。

但对 FNN 来说，训练多个大网络太贵。

== 10.2 Dropout：廉价近似 bagging ★★★

Dropout 给输入单元和隐藏单元加二值 mask：

$ m_j ∼ upright("Bernoulli")(p) $

训练时实际输出：

$ tilde(u)_j = m_j u_j $

如果 (m_j=0)，该单元被临时移除；如果 (m_j=1)，该单元保留。

每个 minibatch 随机采样 mask，然后做一次梯度下降。只有 mask 为 1 的单元对应参数会被更新。

核心直觉：Dropout 在一个大网络中随机训练许多子网络，防止神经元之间形成过强 co-adaptation，从而减少 overfitting。

= 11. 近似理论 Appendix ★★

== 11.1 单隐藏层网络形式 ★★

$ hat(f)_m(x)=sum_{i=1}^m a_i sigma(w_i^->p x+b_i) $

其中：

(m)：隐藏单元数；
(a_i)：输出层线性组合系数；
(w_i)：第 (i) 个隐藏单元权重；
(b_i)：偏置；
$sigma$：非线性激活函数。

直觉：每个隐藏单元是一个非线性 basis function，输出是这些 basis 的线性组合。

== 11.2 Universal Approximation Theorem ★★★

若 $K subset bb(R)^d$ 是 compact set，$sigma$ 是连续非多项式激活函数，则对任意连续函数 (f) 和任意 $epsilon>0$，存在足够大的 (m)，使得：

$ sup_{x in K}|f(x) - hat(f)_m(x)| < epsilon $

结论：单隐藏层 MLP 理论上可以逼近任意连续函数。

但注意：这只说明存在性，不说明需要多少神经元，也不说明容易训练。

== 11.3 Barron class 与浅层网络近似率 ★★

Barron norm：

$ |f|_B = integral_{bb(R)^d} |omega| |hat(f)(omega)| d omega $

如果 $f in B$，则存在一个 (m) 个隐藏单元的单隐藏层网络，使得：

$ |f-hat(f)_m|_{L^2}<= q frac(C|f|_B, sqrt m) $

即：

$ O(m^{-1/2}) $

优点：维度无关。
缺点：收敛慢，而且没有利用 compositional structure。

== 11.4 深层网络近似率 ★★★

如果函数具有层级组合结构：

$ f=f_L∘ f_{L-1}∘..∘ f_1 $

每个 component function $f_ell$ 只依赖至多：

$ q<< d $

个变量，且属于 Sobolev space：

$ W^{k,oo} $

则存在 (N) 个参数的深层网络：

$ |f-f_N|_oo<= q C N^{-k} $

而一般浅层网络对：

$ f in W^{k,oo}([0,1]^d) $

的最优近似率为：

$ |f-f_N|_oo=O(N^{-k/d}) $

关键对比：浅层网络有 curse of dimensionality；深层网络可以利用组合结构，把高维函数拆成低维组合，从而显著提升近似效率。

= 12. SGD 收敛理论 Appendix ★★

== 12.1 SGD 更新与假设 ★★★

$ x_{t+1}=x_t - eta_tg_t $

其中 (g_t) 是 stochastic gradient，并满足无偏性：

$ bb E[g_t mid x_t]=nabla f(x_t) $

方差有界：

$ bb E|g_t-nabla f(x_t)|^2<= q sigma^2 $

目标函数 (L)-smooth：

$ f(y)<= q f(x)+⟨ nabla f(x),y-x⟩+frac(L, 2)|y-x|^2 $

== 12.2 One-step progress ★★★

代入：

$ x_{t+1}=x_t - eta_tg_t $

得到：

$ f(x_{t+1})<= q f(x_t)-eta_t ⟨ nabla f(x_t),g_t ⟩+frac(L eta_t^2, 2)|g_t|^2 $

取条件期望：

$ bb E(f(x_{t+1})| x_t)<= q f(x_t)-eta_t|nabla f(x_t)|^2+frac(L eta_t^2, 2) bb E|g_t|^2 $

利用：

$ bb E|g_t|^2<= q |nabla f(x_t)|^2+sigma^2 $

得到：

$ bb E(f(x_{t+1})) <= q bb E[f(x_t)] -eta_t(1-frac(L eta_t, 2)) bb E|nabla f(x_t)|^2 +frac(L eta_t^2, 2) sigma^2 $

== 12.3 Non-convex 收敛 ★★★

若：

$ eta_t=eta<= q frac(1, L) $

则：

$ bb E(f(x_{t+1})) <= q bb E[f(x_t)] -frac(eta, 2) bb E|nabla f(x_t)|^2 +frac(L eta^2, 2) sigma^2 $

最终得到：

$ frac(1, T)sum_{t=0}^{T-1}bb E|nabla f(x_t)|^2 = O(frac(1, sqrt T)) $

含义：非凸 SGD 通常不能保证到全局最优，只能保证趋近 first-order stationary point。

== 12.4 Convex 收敛 ★★

若 (f) convex 且：

$ bb E|g_t|^2<= q G^2 $

展开距离：

$ |x_{t+1} - x^*|^2 = |x_t-x^*|^2 -2eta_t chevron.l g_t, x_t - x^* chevron.r +eta_t^2|g_t|^2 $

利用 convexity：

$ f(x_t)-f^*<= q ⟨ nabla f(x_t),x_t-x^*⟩ $

得到：

$ 2eta_t bb E(f(x_t) - f^*) <= bb E |x_t - x^*|^2 - bb E |x_{t+1} - x^*|^2 + eta_t^2 G^2 $

选：

$ eta_t=frac(|x_0-x^*|, G sqrt T) $

则：

$ bb E(f(bar(x)) - f^*)<= q frac(|x_0-x^*|G, sqrt T) $

即：

$ O(frac(1, sqrt T)) $

== 12.5 Strongly convex 收敛 ★★

若 (f) 是 $mu$-strongly convex：

$ ⟨ nabla f(x_t),x_t-x^*⟩ >=q f(x_t)-f^* + frac(mu, 2)|x_t-x^*|^2 $

得到递推：

$ bb E|x_{t+1} - x^*|^2 <= q (1-mu eta_t) bb E|x_t - x^*|^2 +eta_t^2 G^2 $

当：

$ eta_t=frac(1, mu(t+1)) $

有：

$ bb E(f(bar(x)_T) - f^*)=O(frac(1, T)) $

= 13. 方法对比总表

| 模块 | 方法 | 解决什么问题 | 局限 |
| ---- | --------------- | --------------------------- | ------------------- |
| 激活函数 | Sigmoid | 概率输出自然 | 隐藏层易梯度消失 |
| 激活函数 | ReLU | 正半轴梯度稳定，训练快 | dead neuron |
| 激活函数 | Leaky/PReLU | 负半轴保留梯度 | 多一个斜率超参或参数 |
| 激活函数 | GELU/Swish/Mish | 平滑、现代深层网络优化好 | 计算略复杂 |
| 优化 | SGD | 简单、基础 | 噪声大，收敛慢 |
| 优化 | Momentum | 减少震荡，加速 | 需调 momentum |
| 优化 | AdaGrad | 参数级自适应学习率 | 学习率单调衰减 |
| 优化 | RMSProp | 遗忘久远历史 | 缺少 momentum |
| 优化 | Adam | Momentum + adaptive scaling | 有时泛化不如 SGD |
| 正则化 | Weight decay | 惩罚大权重 | 不能完全防 co-adaptation |
| 正则化 | Dropout | 近似 bagging，减少 co-adaptation | 训练噪声增加 |

= 14. 期末最可能考的核心公式

必须熟练：

$ z=w^->p x+b  h(x)=f^{(L)}(f^{(L-1)}(.. f^{(1)}(x)))  sigma(z)=frac(1, 1+exp(-z))  upright("ReLU")(z)=max{0,z}  P(y=k mid x)=frac(exp(z_k), sum_{j=1}^C exp(z_j))  frac(partial L, partial z_k)=P(y=k)-bold(1)(y=k)  frac(partial L, partial W_{{"kj"}})=u_j delta_k  delta_j=g'(z_j)sum_k W_{{"kj"}}delta_k  frac(partial L, partial W_{{"ji"}})=u_i delta_j  theta arrow.l theta - alpha frac(1, |B|) sum_{i in B} nabla L(x_i,y_i,theta)  v arrow.l alpha v-epsilon g,quad theta arrow.l theta+v  s arrow.l rho_1 s+(1-rho_1)g  r arrow.l rho_2 r+(1-rho_2)g ∘ g  Delta theta=-epsilon frac(hat(s), sqrt{hat(r)}+delta)  theta arrow.l theta+Delta theta  tilde(J)(theta)=J(theta)+lambda|theta|_2^2  sup_{x in K}|f(x) - hat(f)_m(x)| < epsilon  frac(1, T)sum_{t=0}^{T-1}bb E |nabla f(x_t)|^2=O(frac(1, sqrt T)) $

= 15. 一句话总结

本章的主线是：FNN 用多层非线性函数复合学习特征表示，用概率输出层定义监督学习目标，用反向传播高效计算梯度，用 SGD/Adam 等算法优化参数，并通过 weight decay/dropout 控制过拟合；理论上，浅层网络保证“能表示”，深层网络进一步解释“为什么更高效”。
