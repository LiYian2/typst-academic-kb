// @title: Lecture 10: Diffusion Models 逻辑主线
// @description: 本章的核心目标是回答一个问题：
// @order: 999

= Lecture 10: Diffusion Models 逻辑主线

本章的核心目标是回答一个问题：

如何从一个简单分布，例如高斯噪声，逐步生成复杂数据分布中的样本？

整章的“进化树”可以概括为：

```text
DDPM 离散扩散
 ↓ 发现：采样慢，理论上可连续化
Score-based SDE
 ↓ 发现：反向生成只需要 score ∇ log q_t(x_t)
Denoising Score Matching
 ↓ 解决：真实 marginal score 不可得
Probability Flow ODE
 ↓ 发现：随机 SDE 可转为确定性 ODE
Flow Matching
 ↓ 直接学习从噪声到数据的 vector field
Accelerated Sampling
 ↓ 解决：1000 步采样太慢
DDIM / DPM-Solver / Distillation
 ↓
Control & Classifier-Free Guidance
 ↓ 解决：生成结果如何服从文本条件
Text-guided diffusion / CFG
```

---

= 1. DDPM 回顾：离散扩散模型

重要程度：★★★★★

DDPM 的基本思想是：先定义一个固定的前向加噪过程，把真实数据逐步破坏成高斯噪声；再训练一个神经网络学习反向去噪过程，从噪声恢复数据。

== 1.1 Forward Process：数据到噪声

给定真实数据：

$ x_0 ~ q(x) $

其中：

- (x_0)：原始数据样本，例如一张真实图像；
- (q(x))：真实数据分布；
- $t = 1,dot(s),T$：扩散时间步；
- $beta_t$：第 (t) 步加入的噪声强度；
- $epsilon_t ~ cal(N)(0,I)$ ：标准高斯噪声。

前向扩散一步为：

$ x_t = sqrt(1-beta_t) x_(t-1) + sqrt(beta_t) epsilon_t $

等价地：

$ q(x_t mid x_(t-1)) = cal(N) ( x_t; sqrt(1-beta_t) x_(t-1), beta_t I ) $

整个前向链为：

$ q(x_(1:T) mid x_0) = product_(t=1)^{T} q(x_t mid x_(t-1)) $

联合分布为：

$ q(x_(0:T)) = q(x_0) q(x_(1:T) mid x_0) $

直觉：每一步都把图像缩小一点，同时加入一点高斯噪声。随着 (t) 增大，图像结构逐渐消失，最后 (x_T) 接近标准高斯噪声。

---

== 1.2 Reverse Process：噪声到数据

重要程度：★★★★★

生成时从高斯噪声开始：

$ x_T ~ p(x_T) $

通常设：

$ p(x_T)=cal(N)(x_T;0,I) $

反向一步建模为：

$ x_(t-1) = mu_theta(x_t,t)+sigma_t z, quad z ~ cal(N)(0,I) $

其中：

- $mu_theta(x_t, t)$：神经网络预测的反向均值；
- $sigma_t$：反向采样噪声标准差，常取 $sigma_t^2=beta_t$；
- (z)：采样时额外注入的高斯噪声。

概率形式：

$ p_theta(x_(t-1) mid x_t) = cal(N) ( x_(t-1); mu_theta(x_t,t), sigma_t^2 I ) $

完整生成链：

$ p_theta(x_(0:T)) = p(x_T) product_(t=1)^{T} p_theta(x_(t-1) mid x_t) $

直觉：前向过程是固定破坏数据；反向过程是学习如何一步步修复数据。

---

== 1.3 DDPM Training：预测噪声

重要程度：★★★★★

DDPM 常用的训练目标不是直接预测 (x_0)，而是预测加入到 (x_0) 中的复合噪声 $bar(epsilon)_t$。

利用重参数化：

$ x_t = sqrt(bar(alpha)_t) x_0 + sqrt(1-bar(alpha)_t) bar(epsilon)_t $

其中：

- $alpha_t = 1-beta_t$；
- $bar(alpha)_t=product_(s=1)^(t) alpha_s$；
- $bar(epsilon)_t ~ cal(N)(0,I)$：从 (x_0) 直接得到 (x_t) 的复合噪声。

训练目标：

$ L = bb(E)_(x_0 ~ q(x_0),\ t ~ U(1,T),\ bar(epsilon)_t ~ cal(N)(0,I)) [ | bar(epsilon)_t ---------------- epsilon_theta ( sqrt(bar(alpha)_t) x_0 + sqrt(1-bar(alpha)_t) bar(epsilon)_t, t ) |^2 ] $

其中：

- $epsilon_theta(x_t, t)$：神经网络预测的噪声；
- 输入是带噪图像 (x_t) 和时间步 (t)；
- 目标是让预测噪声接近真实复合噪声。

核心直觉：如果模型能准确预测噪声，就可以从 (x_t) 中减去噪声，从而逐步恢复数据。

---

= 2. 从离散扩散到连续时间 SDE

重要程度：★★★★★

DDPM 是离散时间模型。本章接下来把扩散过程看成“无限多小步”的极限，从而得到随机微分方程 SDE。

== 2.1 Forward Diffusion 的连续极限

离散前向一步：

$ x_t = sqrt(1-beta_t) x_(t-1) + sqrt(beta_t) cal(N)(0,I) $

令：

$ beta_t := beta(t) Delta t $

当 $Delta t$ 很小时：

$ sqrt(1-beta(t) Delta t) approx 1-frac(1, 2) beta(t) Delta t $

因此：

$ x_t approx x_(t-1) ------- frac(beta(t) Delta t, 2) x_(t-1) + sqrt(beta(t) Delta t) cal(N)(0,I) $

令 $Delta t-> 0$，得到前向 SDE：

$ d x_t = -frac(1, 2) beta(t) x_t d t + sqrt(beta(t)) d omega_t $

其中：

- (d x_t)：状态的无穷小变化；
- $-frac(1, 2) beta(t)$x_t d t)：drift term，确定性漂移项；
- $sqrt(beta(t)) d omega_t$：diffusion term，随机噪声项；
- $omega_t$：Wiener process，也就是布朗运动；
- $beta(t)$：连续时间噪声日程。

直觉：前向过程一方面把 (x_t) 往 0 拉，另一方面持续注入高斯噪声。最终分布接近标准高斯。

---

= 3. Reverse-Time SDE 与 Score Function

重要程度：★★★★★

前向 SDE 告诉我们如何从数据走向噪声。生成模型需要反过来，从噪声走回数据。

== 3.1 Reverse Generative SDE

前向 SDE：

$ d x_t = -frac(1, 2) beta(t) x_t d t + sqrt(beta(t)) d omega_t $

对应的反向生成 SDE：

$ d x_t = [ -frac(1, 2) beta(t) x_t ----------------------- beta(t) nabla_(x_t) log q_t(x_t) ] d t + sqrt(beta(t)) d bar(omega)_t $

其中：

- (q_t(x_t))：时间 (t) 时刻的边缘分布；
- $nabla_(x_t) log q_t(x_t$：score function；
- $d bar(omega)_t$：反向时间中的 Wiener process；
- 反向过程从 $x_T ~ cal(N)(0,I)$ 开始，模拟到 (x_0)。

最关键的是：

$ nabla_(x_t) log q_t(x_t) $

它表示当前点 (x_t) 应该往哪个方向移动，才能更接近高概率数据区域。

---

== 3.2 Score Function

重要程度：★★★★★

Score function 定义为 log-density 对输入的梯度：

$ s(x) = nabla_x log q(x) $

其中：

- (q(x))：数据分布密度；
- $log q(x$：log-likelihood；
- $nabla_x log q(x$：指向密度增长最快方向的向量场。

直觉：如果某个点在低概率区域，score 会告诉它应该往哪个方向移动，才能进入更高概率区域。

在 diffusion 中，反向生成的本质就是学习每个噪声等级下的 score：

$ s_theta(x_t,t) approx nabla_(x_t) log q_t(x_t) $

---

= 4. Reverse-Time SDE 的推导逻辑

重要程度：★★★★☆

一般前向 SDE：

$ d x_t = f(x_t,t) d t + g(t) d W_t $

其中：

- (f(x_t,t))：drift coefficient；
- (g(t))：diffusion coefficient；
- (W_t)：Wiener process。

小时间步转移：

$ x_(t+Delta t) = x_t + f(x_t,t) Delta t + g(t) sqrt(Delta t) epsilon, quad epsilon ~cal(N)(0,I) $

所以：

$ p(x'mid x) = cal(N) ( x'; x+f(x,t) Delta t, g(t)^2Delta t I ) $

反向转移由 Bayes rule 得到：

$ p(x mid x') prop p(x'mid x) q_t(x) $

对密度做一阶展开：

$ log q_t(x) approx log q_t(x') + (x-x')^->p nabla log q_t(x') $

完成 Gaussian completion 后：

$ bb(E)[x_t mid x_(t+Delta t)] = x_(t+Delta t) f(x_(t+Delta t),t) Delta t + g(t)^2nabla log q_t(x_(t+Delta t)) Delta t $

因此反向时间 SDE 为：

$ d x_t = [ f(x_t,t) -------- g(t)^2nabla_(x_t) log q_t(x_t) ] d t + g(t) d bar(W)_t $

注意这里的 (d t) 是反向时间方向的微分。带入 VP-SDE 中：

$ f(x_t,t) = -frac(1, 2) beta(t) x_t, quad g(t) = sqrt(beta(t)) $

得到：

$ d x_t = [ -frac(1, 2) beta(t) x_t ----------------------- beta(t) nabla_(x_t) log q_t(x_t) ] d t + sqrt(beta(t)) d bar(W)_t $

---

= 5. Score Matching 与 Denoising Score Matching

重要程度：★★★★★

== 5.1 直接学习 score 的问题

自然想法是直接训练：

$ min_theta bb(E)_(t ~ U(0,T)) bb(E)_(x_t ~ q_t(x_t)) [ | s_theta(x_t,t) --------------- nabla_(x_t) log q_t(x_t) |^2 ] $

问题是：

$ nabla_(x_t) log q_t(x_t) $

不可计算。因为 (q_t(x_t)) 是所有数据点扩散后的边缘混合分布，真实数据分布 (q_0(x_0)) 本身也未知。

Why it fails：目标 score 是 marginal score，需要知道整个数据分布的密度，而真实图像分布无法显式写出。

---

== 5.2 Denoising Score Matching

解决方法：不直接学习不可得的 marginal score，而是学习可得的 conditional score。

条件扩散分布：

$ q_t(x_t mid x_0) $

是高斯，因此可计算。

Denoising Score Matching 目标：

$ min_theta bb(E)_(t ~ U(0,T)) bb(E)_(x_0 ~ q_0(x_0)) bb(E)_(x_t ~ q_t(x_t mid x_0)) [ | s_theta(x_t,t) --------------- nabla_(x_t) log q_t(x_t mid x_0) |^2 ] $

关键性质：

$ s_theta(x_t,t) approx nabla_(x_t) log q_t(x_t) $

也就是说，虽然训练时用 conditional score，但期望意义下可以得到 marginal score。

---

== 5.3 DSM 与 DDPM 噪声预测的等价关系

设：

$ x_t = gamma_t x_0 + sigma_t epsilon, quad epsilon ~ cal(N)(0,I) $

其中：

- $gamma_t$：信号系数；
- $sigma_t$：噪声标准差；
- $epsilon$：标准高斯噪声。

条件分布：

$ q_t(x_t mid x_0) = cal(N)(gamma_t x_0,sigma_t^2 I) $

其 score：

$ nabla_(x_t) log q_t(x_t mid x_0) = -nabla_(x_t) frac((x_t-gamma_t x_0)^2, 2sigma_t^2) $

因此：

$ nabla_(x_t) log q_t(x_t mid x_0) = -frac(x_t-gamma_t x_0, sigma_t^2) $

又因为：

$ x_t-gamma_t x_0 = sigma_t epsilon $

所以：

$ nabla_(x_t) log q_t(x_t mid x_0) = -frac(epsilon, sigma_t) $

令神经网络 score 写成：

$ s_theta(x_t,t) := -frac(epsilon_theta(x_t,t), sigma_t) $

则 DSM loss 变成：

$ min_theta bb(E)_(t,x_0,epsilon) [ frac(1, sigma_t^2) | epsilon-epsilon ~ theta(x_t,t) |^2 ] $

这说明：DDPM 中预测噪声，本质上就是在学习 score。

---

= 6. Probability Flow ODE

重要程度：★★★★★

反向生成 SDE：

$ d x_t = -frac(1, 2) beta(t) [ x_t + 2nabla_(x_t) log q_t(x_t) ] d t + sqrt(beta(t)) d bar(omega)_t $

它在分布意义下等价于一个确定性 ODE：

$ d x_t = -frac(1, 2) beta(t) [ x_t + nabla_(x_t) log q_t(x_t) ] d t $

这个 ODE 称为 Probability Flow ODE。

重要点：

- SDE 的单条轨迹是随机的；
- ODE 的单条轨迹是确定性的；
- 但二者在每个时间 (t) 的边缘分布 (q_t(x_t)) 相同；
- 因此可以用 ODE solver 做采样。

直觉：SDE 是“带随机扰动的去噪”；Probability Flow ODE 是“确定性地沿概率流移动”。

---

= 7. Flow Matching

重要程度：★★★★★

Flow Matching 是本章第二条主线。它不从 score 出发，而是直接学习一个从简单分布流向数据分布的向量场。

== 7.1 Flow Model vs Diffusion Model

| 模型 | 动态方程 | 是否随机 | 学习对象 | 采样方式 |
| --------------- | -------------------------------------- | ---: | --------------------- | ----------- |
| Flow Model | (frac(d X_t, d t)=u_t^theta(X_t)) | 否 | vector field | 解 ODE |
| Diffusion Model | (d X_t=u_t^theta(X_t) d t+sigma_t d W_t) | 是 | drift / score / noise | 解 SDE 或 ODE |

Flow model 的生成过程：

$ X_0 ~ p_("init"), quad d X_t=u_t^theta(X_t) d t $

从 (t=0) 模拟到 (t=1)，得到：

$ X_1 ~ p_("data") $

---

== 7.2 Time-dependent Vector Field

定义：

$ u:R R^d times [0,1]-> R R^d  (x,t)>-> u_t(x) $

其中：

- (x)：当前位置；
- (t)：时间；
- (u_t(x))：在时间 (t)、位置 (x) 的瞬时速度。

轨迹满足 ODE：

$ dot(x)_t = u_t(x_t) $

也就是：

$ frac(d x_t, d t) = u_t(x_t) $

直觉：vector field 给每个点一个运动方向和速度。只要沿着这个速度场走，就能把初始分布运输到目标分布。

---

= 8. Conditional Probability Path

重要程度：★★★★★

给定一个目标数据点：

$  in  R R^d $

定义条件概率路径：

$ p_t(op("dot mid") z) $

满足：

$ p_0(op("dot mid") z)=p_("init"), quad p_1(op("dot mid") z)=delta_z $

其中：

- $p_("init")$：初始简单分布，例如标准高斯；
- $delta_z$：集中在 (z) 的 Dirac delta；
- $p_t(op("dot mid") z$：从噪声分布逐渐收缩到目标点 (z) 的路径。

课件采用高斯条件路径：

$ p_t(op("dot mid") z) = cal(N) ( alpha_t z, beta_t^2 I_d ) $

其中：

- $alpha_t$：控制均值从 0 移向 (z)；
- $beta_t$：控制方差从 1 缩小到 0；
- (I_d)：(d) 维单位矩阵。

边界条件：

$ alpha_0=0,quad beta_0=1 $

所以：

$ p_0(dot|z) = cal(N)(0,I_d) p_("init") $

以及：

$ alpha_1=1,quad beta_1=0 $

所以：

$ p_1(dot|z) = cal(N)(z,0) delta_z $

---

= 9. Conditional Vector Field

重要程度：★★★★★

对于任意条件概率路径，都存在一个等价的 vector field，使得：

$ X_0 ~ p_("init"), quad frac(d, d t) X_t=u_t(X_t mid z) => X_t ~ p_t(op("dot mid") z) $

对于高斯路径：

$ p_t(op("dot mid") z) = cal(N)(alpha_t z,beta_t^2 I_d) $

条件向量场为：

$ u_t(x mid z) = ( dot(alpha)_t -------------- frac(dot(beta)_t, beta_t) alpha_t ) z + frac(dot(beta)_t, beta_t) x $

其中：

- $dot(alpha)_t=frac(d alpha_t, d t)$；
- $dot(beta)_t=frac(d beta_t, d t)$；
- $u_t(x mid z$：在给定目标点 (z) 时，把样本 (x) 推向 (z) 的速度。

推导直觉：如果样本满足：

$ x_t=alpha_t z+beta_t epsilon $

其中 $epsilon$ 固定，则：

$ frac(d x_t, d t) = dot(alpha)_t z+dot(beta)_t epsilon $

又因为：

$ epsilon = frac(x_t-alpha_t z, beta_t) $

代入：

$ frac(d x_t, d t) = dot(alpha)_t z + dot(beta)_t frac(x_t-alpha_t z, beta_t) $

整理：

$ frac(d x_t, d t) = ( dot(alpha)_t -------------- frac(dot(beta)_t, beta_t) alpha_t ) z + frac(dot(beta)_t, beta_t) x_t $

这就是课件中的 conditional vector field。

---

= 10. Marginal Probability Path and Vector Field

重要程度：★★★★☆

真实生成不是只生成某一个固定 (z)，而是生成整个数据分布。

令：

$ z ~ p_("data") $

定义 marginal probability path：

$ p_t(x) = integral p_t(x mid z) p_("data")(z) d z $

其中：

- (p_t(x))：时间 (t) 的边缘分布；
- $p_t(x mid z$：给定目标数据点 (z) 的条件路径；
- $p_("data")(z$：真实数据分布。

对应的 marginal vector field：

$ u_t(x) = integral u_t(x mid z) frac( p_t(x mid z) p_("data")(z) , p_t(x) ) d z $

注意分式部分：

$ frac( p_t(x mid z) p_("data")(z) , p_t(x) ) $

实际上就是 posterior：

$ p(z mid x_t=x) $

所以：

$ u_t(x) = bb(E)_(z ~ p(z mid x_t=x)) [ u_t(x mid z) $

直觉：在位置 (x) 上，可能有很多目标数据点 (z) 可以解释这个中间状态。marginal vector field 是这些条件速度的后验平均。

Why it fails：真实 $p_("data")$ 不知道，积分不可计算，所以 (u_t(x)) 无法显式写出。

How it fixes：训练神经网络 $hat(u)_t(x$ 去逼近这个向量场。

---

= 11. Flow Matching Loss

重要程度：★★★★★

理想 Flow Matching loss 是：

$ L_("F M") = bb(E)_(t ~ U(0,1),\ z ~ p_("data"),\ x ~ p_t(op("dot mid") z)} [ |hat(u)_t(x) - u_t(x)|^2 ] $

问题是 (u_t(x)) 是 marginal vector field，包含不可计算积分。

于是改用 Conditional Flow Matching：

$ L_("C F M") = bb(E)_(t ~ U(0,1),\ z ~ p_("data"),\ x ~ p_t(op("dot mid") z)} [ |hat(u)_t(x) - u_t(x mid z)|^2 ] $

关键结论：最小化 $L_("C F M")$ 等价于最小化 $L_("F M")$。

直觉：虽然训练时用的是条件目标 $u_t(x mid z$，但是在期望意义下，网络最终学到的是 marginal vector field (u_t(x))。

这和 Denoising Score Matching 的逻辑非常像：

| 问题 | 不可直接训练的目标 | 可训练替代目标 | 最终学到 |
| --------------------- | --------------------- | ----------------------------- | --------------------- |
| Score-based diffusion | (nabla_x log q_t(x)) | (nabla_x log q_t(x mid x_0)) | marginal score |
| Flow Matching | (u_t(x)) | (u_t(x mid z)) | marginal vector field |

---

= 12. Gaussian CFM Loss 的具体化

重要程度：★★★★★

对高斯路径：

$ p_t(op("dot mid") z) = cal(N)(alpha_t z,beta_t^2 I_d) $

采样：

$ x=alpha_t z+beta_t epsilon, quad epsilon ~cal(N)(0,I_d) $

条件向量场：

$ u_t(x mid z) = ( dot(alpha)_t -------------- frac(dot(beta)_t, beta_t) alpha_t ) z + frac(dot(beta)_t, beta_t) x $

代入 $x=alpha_t z+beta_t epsilon$：

$ u_t(x mid z) = ( dot(alpha)_t -------------- frac(dot(beta)_t, beta_t) alpha_t ) z + frac(dot(beta)_t, beta_t) (alpha_t z+beta_t epsilon) $

展开：

$ u_t(x|z) = dot(alpha)_t z frac(dot(beta)_t, beta_t) alpha_t z + frac(dot(beta)_t, beta_t) alpha_t z + dot(beta)_t epsilon $

中间两项抵消：

$ u_t(x mid z) = dot(alpha)_t z+dot(beta)_t epsilon $

所以：

$ L_("C F M") = bb(E)_(t,z,epsilon) [ | hat(u)_t(alpha_t z+beta_t epsilon) ------------------------------------- (dot(alpha)_t z+dot(beta)_t epsilon) |^2 ] $

---

= 13. 最简单的 Optimal Transport Path

重要程度：★★★★★

课件选择：

$ alpha_t=t, quad beta_t=1-t $

于是：

$ dot(alpha)_t=1, quad dot(beta)_t=-1 $

条件路径：

$ p_t(x mid z) = cal(N) ( t z, (1-t)^2I_d ) $

采样公式：

$ x=t z+(1-t) epsilon, quad epsilon ~cal(N)(0,I_d) $

目标速度：

$ u_t(x|z) = dot(alpha)_t z+dot(beta)_t epsilon z-epsilon $

因此 CFM loss 变成：

$ L_("C F M") = bb(E)_(t ~ U(0,1),\ z ~ p_("data"),\ epsilon ~cal(N)(0,I_d)} [ | hat(u)_t(t z+(1-t) epsilon) --------------------------- (z-epsilon) |^2 ] $

训练流程：

```text
1. 采样数据 z ~ p_data
2. 采样时间 t ~ U(0,1)
3. 采样高斯噪声 epsilon ~ N(0,I)
4. 构造中间点 x = t z + (1-t) epsilon
5. 神经网络预测速度 u_hat_t(x)
6. 用目标速度 z - epsilon 做 MSE 监督
```

这是 Flow Matching 的核心公式，期末很可能考。

---

= 14. 加速采样：DDIM

重要程度：★★★★☆

DDPM 慢的根本原因：每次生成需要调用 $epsilon_theta(x_t, t)$ 共 (T) 次，典型 (T=1000)。

DDIM 的目标：少步采样，并允许确定性采样。

回忆：

$ x_t = sqrt(bar(alpha)_t) x_0 + sqrt(1-bar(alpha)_t) bar(epsilon)_t $

从当前 (x_t) 估计 (x_0)：

$ hat(x)_0 = frac( x_t-sqrt(1-bar(alpha) t) epsilon ~ theta(x_t,t) , sqrt(bar(alpha)_t) ) $

估计噪声：

$ hat(epsilon)_t = frac( x_t-sqrt(bar(alpha)_t) hat(x)_0 , sqrt(1-bar(alpha)_t) ) $

DDIM 采样：

$ x_(t-1) = sqrt{bar(alpha)_(t-1)} hat(x)*0 + sqrt{1-bar(alpha)*{t-1}-sigma_t^2} hat(epsilon)_t + sigma_t z, quad z ~cal(N)(0,I) $

其中：

- $hat(x)_0$：由当前 (x_t) 预测的干净图像；
- $hat(epsilon)_t$：预测噪声；
- $sigma_t$：控制随机性。

如果：

$ sigma_t=0 $

则 DDIM 是确定性的。

DDIM 可以只在子序列时间点采样：

$ tau_1=1<tau_2<..<tau_K=T $

更新为：

$ x_{tau_(i-1)} = sqrt{bar(alpha)*{tau*{i-1}}} hat(x)*0 + sqrt{ 1-bar(alpha)*{tau_(i-1)}-sigma_(tau_i)^2 } hat(epsilon)*{tau_i} + sigma*{tau_i} z $

核心思想：不必走完所有 (T) 个时间步，只走 $K<< T$ 个时间步。

---

= 15. DPM-Solver

重要程度：★★★★☆

DPM-Solver 的核心思想是：不要把 diffusion ODE 当成普通黑箱 ODE，而是利用它的半线性结构。

从参数化 diffusion ODE 开始：

$ frac(d x_t, d t) = h_theta(x_t,t) f(t) x_t + frac(g^2(t), 2sigma_t) epsilon_theta(x_t,t) $

其中：

- (f(t) x_t)：关于 (x_t) 的线性项；
- $frac(g^2(t), 2sigma_t) epsilon_theta(x_t,t))$：非线性神经网络项；
- $sigma_t$：噪声尺度；
- (g(t))：扩散系数。

利用 variation of constants formula，精确解：

$ x_t = e^{integral_s^t f(tau) d tau} x_s + integral_s^t ( e^{integral_tau^t f(r) d r} frac(g^2(tau), 2sigma_tau) epsilon_theta(x_tau,tau) ) d tau $

将线性项写成 diffusion coefficient：

$ e^{integral_s^t f(tau) d tau} = frac(alpha_t, alpha_s) $

得到：

$ x_t = frac(alpha_t, alpha_s) x_s + integral_s^t ( frac(alpha_t, alpha_tau) frac(g^2(tau), 2sigma_tau) epsilon_theta(x_tau,tau) ) d tau $

引入：

$ lambda = log alpha-log sigma $

得到：

$ x_t = frac(alpha_t, alpha_s) x_s alpha_t integral_(lambda_s)^{lambda_t} e^{-lambda} epsilon_theta(x_lambda,lambda) d lambda $

此时问题变成：计算一个带已知指数核 $e^{-lambda}$ 的积分。

DPM-Solver 对神经网络项做 Taylor 展开：

$ hat(epsilon)*theta(hat(x)*lambda,lambda) = sum_(n=0)^{k-1} frac( (lambda-lambda_{t_(i-1)})^n , n! ) hat(epsilon)^{(n)}*theta ( hat(x)*{t_(i-1)}, lambda_{t_(i-1)} ) + O((lambda-lambda_{t_(i-1)})^k) $

代入积分：

$ x_(t_i) = frac(alpha_(t_i), alpha_{t_(i-1)}) tilde(x)*{t*{i-1}} ------------------- alpha_(t_i) sum_(n=0)^{k-1} hat(epsilon)^{(n)}*theta ( hat(x)*{t_(i-1)}, lambda_{t_(i-1)} ) integral_{lambda_{t_(i-1)}}^{lambda_(t_i)} e^{-lambda} frac( (lambda-lambda_{t_(i-1)})^n , n! ) d lambda + O(h_i^{k+1}) $

保留 $n<= k-1$ 项，就得到 (k)-th order solver。

核心直觉：DPM-Solver 不是粗暴数值积分，而是利用 diffusion ODE 的特殊结构，把线性部分精确处理，只近似神经网络非线性部分，因此可以十几步完成高质量采样。

---

= 16. Distribution Matching Distillation

重要程度：★★★☆☆

课件只简要提到 DMD：

- Distribution Matching Distillation；
- 目标是把多步 diffusion model 蒸馏成一步或少步生成模型；
- 代表工作包括：

 * One-step diffusion with distribution matching distillation, CVPR 2024；
 * Improved distribution matching distillation for fast image synthesis, NeurIPS 2024。

期末若考，重点记住：DMD 属于 sampling acceleration / distillation 路线，目标是极大减少采样步数。

---

= 17. Stable Diffusion 与 Text Guidance

重要程度：★★★★☆

Stable Diffusion 是 2022 年发布的 text-to-image 模型。它的关键结构是：

```text
text prompt y
 ↓ text encoder
text embedding τ(y)
 ↓ cross-attention
latent diffusion model / noise predictor εθ(z_t,t,y)
 ↓ iterative denoising
generated image
```

训练目标：

$ x_t = sqrt(bar(alpha)_t) x_0 + sqrt(1-bar(alpha)_t) bar(epsilon)_t, quad bar(epsilon)_t ~ cal(N)(0,I)  min | bar(epsilon)_t ---------------- epsilon_theta(z_t,t,y) |^2 $

其中：

- (y)：文本 prompt；
- $tau(y)$：文本编码器输出的 text embedding；
- (z_t)：latent space 中的带噪变量；
- $epsilon_theta(z_t,t,y)$：带文本条件的噪声预测器。

Text guidance 的直觉：每一步去噪时，文本 embedding 都提醒模型：“当前图像应该更像 prompt 描述的内容”。

---

= 18. Classifier Guidance 与 Classifier-Free Guidance

重要程度：★★★★★

== 18.1 Vanilla Guided Sampling

给定条件 (y)，直接训练一个条件向量场：

$ u_t^theta(x mid y) $

采样流程：

```text
1. 选择 prompt y，例如 "a cat baking a cake"
2. 初始化 X_0 ~ p_init
3. 模拟 d X_t = u_t^theta(X_t | y) d t，从 t=0 到 t=1
```

问题：vanilla guidance 可能不够强，生成图像和 prompt 对齐不好。

---

== 18.2 Classifier Guidance 的直觉

目标条件分布：

$ p_t(x mid y) $

由 Bayes rule：

$ p_t(x mid y) prop p_t(x) p_t(y mid x) $

取 log：

$ log p_t(x mid y) = log p_t(x) + log p_t(y mid x) + C $

对 (x) 求梯度：

$ nabla_x log p_t(x mid y) = nabla_x log p_t(x) + nabla_x log p_t(y mid x) $

所以 guidance 可以理解为：

$ "unconditional direction" + "prompt-dependent direction" $

课件图中写作：

$ u_t^{"target"}(x mid y) = u_t^{"target"}(x) + a_t nabla_x log p_t(y mid x) $

其中：

- $u_t^{"target"}(x$：无条件生成方向；
- $a_t nabla_x log p_t(y mid x$：prompt-dependent part；
- (a_t)：与噪声日程相关的缩放系数。

进一步增强 prompt 对齐：

$ tilde(u)_t^w(x mid y) = u_t^{"target"}(x) + w a_t nabla_x log p_t(y mid x) $

其中 (w) 是 guidance scale。(w) 越大，prompt 约束越强，但过大可能导致过饱和、失真、多样性下降。

---

== 18.3 Classifier-Free Guidance

Classifier guidance 需要额外训练 classifier $p_t(y mid x$，而 CFG 不需要外部 classifier。

核心观察：

$ u_t^{"target"}(x mid y) ---------------------------- u_t^{"target"}(x) $

可以近似看作 prompt-dependent direction。

于是放大这个差值：

$ tilde(u)_t^w(x mid y) = u_t^{"target"}(x) + w ( u_t^{"target"}(x mid y) ---------------------------- u_t^{"target"}(x) ) $

等价展开：

$ tilde(u)_t^w(x mid y) = (1-w) u_t^{"target"}(x) + w u_t^{"target"}(x mid y) $

课件给出的 CFG 采样向量场为：

$ u_t^{theta,w}(x) = (1-w) u_t^theta(x mid nothing) + w u_t^theta(x mid y) $

其中：

- $u_t^theta(x mid nothing)$：空 prompt，即 unconditional prediction；
- $u_t^theta(x mid y)$：有条件 prompt prediction；
- (w)：CFG scale；
- $nothing$：空文本 token。

这就是实际生成模型中最重要的控制公式之一。

---

== 18.4 CFG Training：为什么要有 empty token

训练时需要让同一个模型同时支持：

$ u_t^theta(x mid y) $

和：

$ u_t^theta(x mid nothing) $

所以训练过程中会随机把一部分文本条件替换为空 token $nothing$。这样模型既会有条件生成，也会无条件生成。

采样时同一个 (x_t) 跑两次模型：

```text
conditional prediction: uθ(x_t | y)
unconditional prediction: uθ(x_t | ∅)
combine: (1-w) uθ(x_t|∅) + w uθ(x_t|y)
```

---

== 18.5 CFG 的本质与局限

重要程度：★★★★★

CFG 很有效，但它不是严格的数据分布建模。

当 (w=1) 时：

$ u_t^{theta,w}(x)=u_t^theta(x mid y) $

近似正常条件生成。

当 (w>1) 时：

$ u_t^{theta,w}(x) = u_t^theta(x mid y) + (w-1) [ u_t^theta(x mid y) - u_t^theta(x mid nothing) ] $

这意味着模型沿 prompt-dependent direction 走得比真实条件分布更远。

所以课件强调：

CFG is a heuristic. We do not model the data distribution anymore. We go beyond it.

也就是说，CFG 牺牲严格概率建模，换取更强的 prompt adherence 和更好的视觉效果。

---

= 19. 总结对比表

| 方法 | 核心学习对象 | 主要公式 | 解决的问题 | 局限 | | |
| -------------------- | --------------------------- | -------------------------------------------------- | --------------- | -------------------- | ------------ | ---------- |
| DDPM | 噪声 $(epsilon_theta(x_t,t))$ | $(|epsilon-epsilon_theta(x_t,t)|^2)$ | 从噪声逐步生成数据 | 采样慢 | | |
| Score-based SDE | score $(s_theta(x_t,t))$ | $(s_theta approx nabla_x log q_t(x))$ | 连续时间统一建模 | marginal score 不可直接得 | | |
| DSM | conditional score | $(nabla_x log q_t(x_t mid x_0) = -epsilon/sigma_t)$ | 让 score 可训练 | 仍需多步采样 | | |
| Probability Flow ODE | deterministic flow | $(d x_t = -frac(1,2) beta(t) [x_t + nabla log q_t(x_t)] d t)$ | 用 ODE 替代 SDE | 需要数值求解 | | |
| Flow Matching | vector field $(u_t(x))$ | $(hat(u)_t(t z+(1-t) epsilon) approx z-epsilon)$ | 直接学习噪声到数据的流 | 仍需 ODE 采样 | | |
| DDIM | 少步采样路径 | $(hat(x)_0, hat(epsilon)_t)$ 更新 | 加速 DDPM | 步数太少会降质 | | |
| DPM-Solver | 高阶 ODE solver | 指数核积分近似 | 十几步高质量采样 | 数学复杂 | | |
| CFG | 条件控制方向 | (1-w) u(x\|nothing)+ w u(x\|y) | 强化 prompt 对齐 | 不再严格建模真实分布 |

---

= 20. 期末复习优先级

最高优先级：

```text
1. DDPM forward / reverse / training loss
2. 离散扩散到 SDE 的推导
3. reverse SDE 中 score 的作用
4. DSM 如何把不可得 marginal score 转成 conditional score
5. DSM 与噪声预测的等价性
6. Probability Flow ODE
7. Flow Matching 的 conditional path、conditional vector field、CFM loss
8. CFG 公式与直觉
```

中高优先级：

```text
1. DDIM 采样公式
2. DPM-Solver 的半线性 ODE 结构
3. Stable Diffusion 中 text embedding 通过 cross-attention 注入
```

较低但需知道：

```text
1. DMD 是用于 one-step / few-step diffusion distillation
2. Runge-Kutta、Heun、exponential integrator 等属于 continuous-time solver 家族
```

核心记忆句：

扩散模型从 DDPM 的“离散噪声预测”发展到 Score-SDE 的“连续 score 学习”，再到 Probability Flow ODE 和 Flow Matching 的“确定性向量场建模”；采样加速方法解决多步生成效率问题，而 CFG 解决文本条件控制问题，但 CFG 本质是经验性地放大条件方向，并不再严格对应真实数据分布。
