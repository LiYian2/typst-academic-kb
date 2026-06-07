// @title: Lecture 9 Variational Autoencoder and Diffusion Models
// @description: Autoencoder 用确定性压缩学习表示 → VAE 把 latent space 概率化，使 decoder 可以采样生成 → VQ-VAE 把 latent 离散化 → DDPM 不再直接学习一个低维 latent，而是学习“逐步加
// @order: 999

= Lecture 9 Variational Autoencoder and Diffusion Models

_Autoencoder 用确定性压缩学习表示 → VAE 把 latent space 概率化，使 decoder 可以采样生成 → VQ-VAE 把 latent 离散化 → DDPM 不再直接学习一个低维 latent，而是学习“逐步加噪—逐步去噪”的生成过程。_ 

= 1. 本章核心目标：从“重构”走向“生成”

生成模型的目标不是直接学习判别式概率 $p(y mid x$)，而是学习数据分布本身，例如 (p(x)) 或条件分布 $p(x mid y$)。如果模型学到的分布 (P_g) 足够接近真实分布 $P_{"data"}$，就可以从 (P_g) 中采样生成新样本。课件首先区分了显式生成模型和隐式生成模型：显式模型直接做 density estimation，能写出 likelihood 并用 maximum likelihood 训练；隐式模型不显式估计密度，而更关注直接生成，例如 GAN。VAE 和 DDPM 都属于显式概率建模思路，因为它们都围绕 likelihood 或 variational bound 构造训练目标。

本章的“进化树”可以概括为：

```text
高维数据建模困难
 ↓
Autoencoder：学习低维表示，解决压缩与重构
 ↓
问题：普通 AE 只有 reconstruction，不保证 latent space 可采样
 ↓
VAE：给 latent variable 加 prior，用 encoder 近似 posterior，优化 ELBO
 ↓
问题：连续 Gaussian latent 有时表达离散结构不自然
 ↓
VQ-VAE：使用离散 embedding codebook
 ↓
问题：VAE 类模型生成质量有限，latent 压缩瓶颈明显
 ↓
DDPM：直接在数据空间构造 Markov 加噪过程，学习反向去噪生成
```

= 2. Autoencoder：从高维输入到低维表示

== 2.1 为什么需要 Autoencoder【重要度：高】

高维数据存在 curse of dimensionality：训练样本在高维空间极其稀疏，计算和存储成本高，模型难以直接学习有效结构。因此需要 dimension reduction。Autoencoder 的基本思想是学习一个映射

$ x in RR^n --> h in RR^d,quad quad d<< n, $

其中 (x) 是原始输入，(h) 是低维表示，也叫 encoding 或 representation。模型训练目标是让输出 (z) 尽量重构输入 (x)。

Autoencoder 的结构是：

$ h=f_theta(x)=a(W x + b),  z=g_{theta'}(h)=a(W' h + b'). $

其中 (W,b) 是 encoder 参数，(W',b') 是 decoder 参数，$a(dot$) 是激活函数，(h) 是隐藏表示，(z) 是重构输出。直觉上，encoder 把输入压缩成 compact representation，decoder 再从这个 representation 中恢复原始输入。

== 2.2 Autoencoder 的训练目标【重要度：高】

参数估计目标是最小化 reconstruction loss：

$ theta^*,theta'^* = op("argmin")_{theta,theta'} frac(1, n) sum_{i=1}^{n} L(x^{(i)},z^{(i)}) = op("argmin")_{theta,theta'} frac(1, n) sum_{i=1}^{n} L(x^{(i)},g_{theta'}(f_theta(x^{(i)}))). $

其中 $x^{(i)}$ 是第 (i) 个训练样本，$z^{(i)}$ 是对应重构结果，$L(dot,dot)$ 是损失函数。

如果输入是连续值，常用平方误差：

$ L(x,z)=|x-z|^2. $

如果输入和输出可视为 Bernoulli bit probability，则常用 cross-entropy：

$ L_H(x,z) = H(beta_x|beta_z) -sum_{k=1}^{d} [ x_k log z_k+(1-x_k) log(1-z_k)]. $

这里 (x_k) 是输入第 (k) 维，(z_k) 是重构输出第 (k) 维，(d) 是数据维度。这个损失本质上是在惩罚每一维 Bernoulli 分布预测错误。

== 2.3 Undercomplete 与 Overcomplete【重要度：中高】

Undercomplete autoencoder 指隐藏层维度小于输入层：

$ dim(h)<dim(x). $

这会强迫模型压缩信息，因此更可能学习到训练分布中的主要因素。Overcomplete autoencoder 指隐藏层维度大于输入层：

$ dim(h)>dim(x). $

这时模型可能直接学 identity mapping，不一定提取有意义结构。因此 overcomplete 情况通常需要额外 regularization，例如 sparsity、denoising、contractive penalty 等。课件还提到 tied weights，即限制 decoder 权重是 encoder 权重的转置：

$ W^*=W^T. $

这可以减少参数量，也让 encoder-decoder 结构更对称。

= 3. Linear Autoencoder 与 PCA 的关系

== 3.1 核心结论【重要度：高】

在线性 decoder、平方重构误差下，最优 linear autoencoder 学到的子空间等价于 PCA 主成分子空间。也就是说，autoencoder 并不是凭空发明的压缩机制；在最简单的线性情况下，它退化为经典 PCA。

课件先引用低秩近似定理。设矩阵 (A) 的 SVD 为

$ A=U Sigma V^T, $

其中 (U,V) 是正交矩阵，$Sigma$ 是奇异值对角矩阵。秩为 (k) 的最佳近似矩阵为

$ B^* = op("argmin")_{B:op("rank")(B)= k} |A-B|_F = U_{.,<= k}Sigma_{<= k,<= k}V^T_{.,<= k}. $

其中 $U_{.,<= k}$ 表示取前 (k) 个左奇异向量，$Sigma_{<= k,<= k}$ 表示前 (k) 个最大奇异值组成的对角矩阵，$V^T_{.,<= k}$ 表示对应右奇异向量部分。这个定理说明：如果只能用 rank-(k) 矩阵近似原始数据，最优做法就是保留前 (k) 个 principal directions。

Autoencoder 的平方误差满足：

$ min_{{"theta"}} sum_t frac(1, 2) (x_i^{(t)}-z_i^{(t)})^2 >= min_{W',h(X)} frac(1, 2) |X-W'h(X)|_F^2. $

这里 (X) 是数据矩阵，(W') 是 decoder 矩阵，(h(X)) 是任意 encoder 表示。最优解为：

$ W'<- U_{.,<= k}, quad quad h(X)<- Sigma_{<= k,<= k}V^T_{.,<= k}. $

进一步可以得到：

$ h(X)=U^T_{.,<= k}X. $

因此单个输入 (x) 的编码为：

$ h(x)=U^T_{.,<= k}x, $

重构为：

$ z=U_{.,<= k}h(x). $

这正是 PCA：把输入投影到前 (k) 个主成分方向，再从这些主成分方向重构。若输入经过中心化和归一化：

$ x^{(t)} <- frac(1, sqrt{T}) ( x^{(t)} ------- frac(1, T)sum_{i=1}^{T}x^{(i)} ), $

则奇异向量与 covariance matrix 的 eigenvectors 对应，奇异值与 eigenvalues 对应。

= 4. 从 AE 到 VAE：为什么普通 Autoencoder 不能直接生成？

普通 autoencoder 可以重构输入，也能学习 representation，但它没有显式规定 latent space 的概率分布。因此你不能随便从 latent space 采样一个 (z)，再期待 decoder 生成合理样本。原因是普通 AE 的 latent space 可能是破碎的、不连续的、存在大量空洞的。训练时 decoder 只见过 encoder 产生的 latent codes，没见过任意采样的 latent vectors。

VAE 的关键改进是：把 latent representation 从确定性向量变成随机变量，并给它一个 prior，例如

$ z ∼ p(z)=cal(N)(0,I). $

这样训练完成后，就可以从 prior 中采样 (z)，再用 decoder 生成 (x)。

= 5. Variational Autoencoder

== 5.1 VAE 的生成假设【重要度：高】

VAE 假设训练数据 $X={x^{(1$},x^{(2)},..,x^{(N)}}) 由潜变量 (z) 生成：

$ z ∼ p(z),  x ∼ p_theta(x mid z). $

其中 (p(z)) 通常取 Gaussian prior，例如 $cal(N)(0,I)$。Decoder network 用来建模条件分布：

$ p_theta(x mid z)=cal(N)(mu_x,sigma_x^2), $

或者对于二值数据使用 Bernoulli distribution。这里 $theta$ 是 decoder 参数，$mu_x,sigma_x^2$ 由神经网络输出。

真正想最大化的是 data likelihood：

$ p_theta(x)=integral p_theta(z)p_theta(x mid z), d z. $

但这个积分通常不可解，而且 posterior

$ p_theta(z mid x) $

也不可解。因此 VAE 引入 encoder network：

$ q_phi(z mid x), $

用它近似真实 posterior $p_theta(z|x)$.通常设：

$ q_phi(z mid x) = cal(N)(mu_z(x),sigma_z(x)). $

其中 $phi$ 是 encoder 参数，$mu_z(x$) 和 $sigma_z(x$) 是 encoder 输出。

== 5.2 ELBO 推导【重要度：极高】

VAE 的核心数学是 variational lower bound，也叫 ELBO。因为 $log p_theta(x)$ 与 (z) 无关，所以：

$ log p_theta(x) = bb(E)_{z ∼ q_phi(z mid x)} [log p_theta(x)]. $

由 Bayes rule：

$ p_theta(x)=frac(p_theta(x,z), p_theta(z mid x)). $

所以：

$ log p_theta(x) = bb(E)_{z ∼ q_phi(z mid x)} [ log frac(p_theta(x,z), p_theta(z mid x))]. $

乘除同一个 $q_phi(z|x)$：

$ log p_theta(x) = bb(E)_{z} [ log frac(p_theta(x,z), q_phi(z mid x)) frac(q_phi(z mid x), p_theta(z mid x))]. $

拆开 log：

$ log p_theta(x) = bb(E)_{z} [ log frac(p_theta(x,z), q_phi(z mid x))] + bb(E)_{z} [ log frac(q_phi(z mid x), p_theta(z mid x))]. $

第二项是 KL divergence：

$ D_{upright("KL")} ( q_phi(z mid x)|p_theta(z mid x) ) >= 0. $

因此：

$ log p_theta(x) = cal(L)(theta,phi,x) + D_{upright("KL")} ( q_phi(z mid x)|p_theta(z mid x) ) >= cal(L)(theta,phi,x). $

其中

$ cal(L)(theta,phi,x) = bb(E)_{z ∼ q_phi(z mid x)} [ log frac(p_theta(x,z), q_phi(z mid x))] $

就是 ELBO。因为 KL 非负，所以最大化 ELBO 等价于最大化 likelihood 的下界，同时也间接让 approximate posterior 靠近 true posterior。

继续展开：

$ cal(L)(theta,phi,x) = bb(E)_{z} [ log frac{p_theta(x mid z)p_theta(z)} {q_phi(z mid x)} ]. $

拆成两项：

$ cal(L)(theta,phi,x) = bb(E)_{z} [ log frac{p_theta(z)} {q_phi(z mid x)} ] + bb(E)_{z} [log p_theta(x mid z)]. $

也就是：

$ cal(L)(theta,phi,x) = D_{upright("KL")} ( q_phi(z mid x)|p_theta(z) ) + bb(E)_{z ∼ q_phi(z mid x)} [log p_theta(x mid z)]. $

这就是 VAE loss 的核心结构：

```text
ELBO = reconstruction term - regularization term
```

其中 reconstruction term 让 decoder 能从 (z) 重构 (x)，regularization term 让 encoder 输出的 posterior 不要偏离 prior (p(z))，从而保证 latent space 可采样。

== 5.3 Gaussian KL 闭式解【重要度：高】

当 prior 为标准正态：

$ p(z)=cal(N)(0,1), $

encoder posterior 为：

$ q_phi(z mid x)=cal(N)(mu_z(x;phi),sigma_z^2(x;phi)), $

则 KL 项有闭式表达。课件写作：

$ D_{upright("KL")} ( q_phi(z|x) | p(z) ) = frac(1, 2) ( 1 + log sigma_z^2 - mu_z^2 - sigma_z^2 ). $

多维情况下对 latent dimension $j=1,..,J$ 求和：

$ D_{upright("KL")} ( q_phi(z|x) | p(z) ) = frac(1, 2) sum_{j=1}^{J} ( 1 + log sigma_{"zj"}^2 - mu_{"zj"}^2 - sigma_{"zj"}^2 ). $

这里 $mu_{{"zj"}}$ 是第 (j) 个 latent 维度的均值，$sigma_{{"zj"}}^2$ 是方差，(J) 是 latent dimension。该项最大化时会倾向于让 $mu_z-> 0$、$sigma_z^2-> 1$，即让 posterior 接近 standard Gaussian prior。

== 5.4 Reparameterization Trick【重要度：极高】

VAE 训练时需要估计：

$ bb(E)_{q_phi(z mid x)} [log p_theta(x mid z)]. $

Monte Carlo 近似为：

$ bb(E)_{z}[log p_theta(x mid z)] approx frac(1, L) sum_k log p_theta(x mid z^{(k)}), quad quad z^{(k)} ∼ q_phi(z mid x). $

问题是 $z^{(k)}$ 是随机采样得到的，普通反向传播不能直接穿过 sampling node。解决方法是 reparameterization trick：

$ epsilon^{(k)} ∼ cal(N)(0,1),  z^{(k)} = mu_z(x,phi) + sigma_z(x,phi)dot epsilon^{(k)}. $

这样随机性被移到 $epsilon$，而 (z) 对 $mu_z,sigma_z$ 是可微的。换句话说，原来的随机节点

$ z ∼ cal(N)(mu_z,sigma_z^2) $

被改写成 deterministic transformation：

$ z=g_phi(x,epsilon). $

因此梯度可以从 decoder loss 反传到 encoder 参数 $phi$。

== 5.5 VAE 总损失【重要度：极高】

课件总结的 VAE objective 为：

$ cal(L)(theta,phi,x) approx frac(1, 2) sum_{j=1}^{J} ( 1+log sigma_{{"zj"}}^2-mu_{{"zj"}}^2-sigma_{{"zj"}}^2 ) + frac(1, L) sum_k log p_theta(x mid z^{(k)}), $

其中

$ z^{(k)} = mu_z(x,phi) + sigma_z(x,phi)dot epsilon^{(k)}, quad quad epsilon^{(k)} ∼ cal(N)(0,1). $

考试时要能解释两项含义：第一项是 regularization，把 latent posterior 拉向 prior；第二项是 reconstruction likelihood，让 decoder 能重构输入。训练后生成新数据时，直接采样：

$ z ∼ cal(N)(0,I), $

再输入 decoder 得到新样本。

= 6. VQ-VAE：离散 latent representation

VQ-VAE 的动机是：很多数据具有天然离散结构，例如 token、phoneme、object part、visual code。普通 VAE 使用连续 Gaussian latent，不一定适合表达这种离散因素。VQ-VAE 使用 discrete latent embeddings。Posterior 和 prior 都是 categorical distribution，采样结果是整数 index，这些 index 用来查 embedding table。

posterior categorical distribution 定义为 one-hot：

$ q(z=k mid x) = cases(1,, k=op("argmin")_j|z_e(x)-e_j|_2,\ 0,, "otherwise".) $

其中 (z_e(x)) 是 encoder 输出的连续向量，(e_j) 是 codebook 中第 (j) 个 embedding。这个公式表示：encoder 输出会被量化到最近的 codebook vector。

训练目标为：

$ L = log p(x mid z_q(x)) + | op("sg")[z_e(x)]-e|_2^2 + beta |z_e(x)-op("sg")[e]|_2^2. $

其中 $op("sg")[dot]$ 是 stop-gradient operator，前向传播时等于 identity，反向传播时梯度为 0。第二项更新 codebook embedding，让 (e) 靠近 encoder 输出；第三项是 commitment loss，让 encoder 输出承诺靠近某个 embedding，避免无限漂移；$beta$ 控制 commitment 强度。

= 7. Diffusion Model：从 latent sampling 到逐步去噪生成

== 7.1 DDPM 的核心思想【重要度：极高】

DDPM 的生成逻辑不同于 VAE。VAE 是先采样一个 latent (z)，再一次性 decoder 生成 (x)。DDPM 则定义一个固定 forward diffusion process，把真实数据逐步加噪直到变成纯 Gaussian noise；然后训练一个 neural network 学习 reverse denoising process，从噪声一步步还原数据。

forward process 定义为：

$ x_0 ∼ q(x),  x_t = sqrt{1-beta_t}x_{t-1} + sqrt{beta_t}epsilon_t, quad quad epsilon_t ∼ cal(N)(0,I). $

其中 (x_0) 是真实数据，(x_t) 是第 (t) 步加噪后的样本，$beta_t$ 是第 (t) 步噪声强度，$epsilon_t$ 是标准 Gaussian noise。每一步都会压低原始信号，同时加入噪声。典型 schedule 中 $beta_1=10^{-4}$，线性增加到 $beta_T=10^{-2}$。

对应条件分布为：

$ q(x_t mid x_{t-1}) = cal(N) ( x_t; sqrt{1-beta_t}x_{t-1}, beta_t I ). $

完整 forward Markov chain：

$ q(x_{1:T} mid x_0) = product_{t=1}^{T} q(x_t mid x_{t-1}),  q(x_{0:T}) = q(x_0)q(x_{1:T} mid x_0). $

== 7.2 Diffusion kernel 推导【重要度：极高】

定义：

$ alpha_t=1-beta_t,  bar(alpha)_t=product_(s=1)^(t)alpha_s. $

注意课件这里符号排版中 $bar(alpha)_t$ 的 product index 应理解为从 (s=1) 到 (t)。由递推可以得到：

$ x_t = sqrt{bar(alpha)_t}x_0 + sqrt{1-bar(alpha)_t}bar(epsilon)_t, quad quad bar(epsilon)_t ∼ cal(N)(0,I). $

这说明你不需要真的从 $x_0-> x_1->..-> x_t$ 一步步采样，而可以直接从 (x_0) 得到任意 (x_t)。对应 diffusion kernel 是：

$ q(x_t mid x_0) = cal(N) ( x_t; sqrt{bar(alpha)_t}x_0, (1-bar(alpha)_t)I ). $

课件有一处写成 $sqrt{1-bar(alpha)_t}I$，严格从 Gaussian covariance 角度应为 covariance $(1-bar(alpha)_t$I)，standard deviation 是 $sqrt{1-bar(alpha)_t}$。

直觉非常重要：$sqrt{bar(alpha)_t}$ 是保留的原始信号比例，$sqrt{1-bar(alpha)_t}$ 是总噪声标准差。随着 (t) 增大，$bar(alpha)_t-> 0$，所以

$ q(x_T mid x_0)approx cal(N)(0,I). $

这就是为什么 sampling 可以从纯 Gaussian noise 开始。

== 7.3 Noise schedule【重要度：中高】

Noise schedule 决定 $beta_t$ 或 $bar(alpha)_t$ 如何随时间变化。课件列出 cosine、linear、quadratic、sigmoid schedule。核心目标是让 $bar(alpha)_T approx 0$，使最终分布接近标准正态。

常见 schedule 包括：

$ beta_t = beta_1+ frac(t-1, T-1)(beta_T-beta_1) $

这是 linear schedule。

Quadratic schedule：

$ beta_t = [ sqrt{beta_1} + frac(t-1, T-1) (sqrt{beta_T}-sqrt{beta_1})]^2. $

Cosine schedule 通过函数 (f(t)) 定义：

$ bar(alpha)_t = frac(f(t), f(0)),  f(t) = cos^2 ( frac(frac{t}{T}+s, 1+s) dot frac(pi, 2) ). $

不同 schedule 会影响训练稳定性和 sampling quality。考试一般不要求背所有 schedule，但要理解：schedule 控制 signal-to-noise ratio 的衰减速度。

= 8. DDPM 的反向过程

== 8.1 Reverse process 建模【重要度：极高】

Forward process 是固定的，不需要学习。DDPM 学的是 reverse process：从

$ x_T ∼ cal(N)(0,I) $

开始，逐步采样：

$ x_{t-1} = mu_theta(x_t,t)+sigma_t z, quad quad z ∼ cal(N)(0,I). $

对应概率模型：

$ p(x_T)=cal(N)(x_T;0,I),  p_theta(x_{t-1} mid x_t) = cal(N) ( x_{t-1}; mu_theta(x_t,t), sigma_t^2I ),  p_theta(x_{0:T}) = p(x_T) product_{t=1}^{T} p_theta(x_{t-1} mid x_t). $

其中 $mu_theta(x_t,t)$ 是 neural network parameterized mean，$sigma_t$ 是 reverse step 的 standard deviation。

== 8.2 Noise predictor【重要度：极高】

DDPM 通常不直接预测 (x_0)，也不直接预测 $mu_theta$，而是训练一个 network 预测加到 (x_0) 上的 compound noise：

$ epsilon_theta(x_t,t)~= bar(epsilon)_t. $

输入是 noisy image (x_t) 和 time embedding (t)，输出是噪声估计。网络通常使用 U-Net，因为输入输出同尺寸，encoder path 提取多尺度语义，decoder path 恢复空间细节，skip connections 保留局部结构。

训练目标简化为：

$ min | bar(epsilon)_(t)-epsilon_theta(x_t,t) |^2. $

更完整写法是：

$ L = bb(E)_{x_0 ∼ q(x_0),\ t ∼ U(1,T),\ bar(epsilon)_t ∼ cal(N)(0,I)} [ | bar(epsilon)_t ---------------- epsilon_theta ( sqrt{bar(alpha)_t}x_0 + sqrt{1-bar(alpha)_t}bar(epsilon)_t, t ) |^2 ]. $

这里 $t ∼ U(1,T$) 表示训练时均匀随机采样一个时间步，$bar(epsilon)_t$ 是从 (x_0) 到 (x_t) 的 compound noise。训练过程本质是：随机取一张真实图，随机取一个时间步，加对应强度的噪声，让网络预测这份噪声。

== 8.3 为什么 noise prediction 等价于学习 reverse mean【重要度：极高】

Forward posterior 有闭式：

$ q(x_{t-1} mid x_t,x_0) = cal(N) ( x_{t-1}; tilde(mu)_t(x_t,x_0), tilde(beta)_t I ). $

其中 posterior mean 可写为：

$ tilde(mu)_t(x_t,x_0) = frac(1, sqrt{alpha_t}) ( x_t --- frac(beta_t, sqrt{1-bar(alpha)_t})epsilon ). $

课件也写作：

$ tilde(mu)_t(x_t,x_0) = frac(1, sqrt{alpha_t}) ( x_t --- frac(1-alpha_t, sqrt{1-bar(alpha)_t})bar(epsilon)_t ), $

因为

$ beta_t=1-alpha_t. $

于是模型 mean 被参数化为：

$ mu_theta(x_t,t) = frac(1, sqrt{alpha_t}) ( x_t --- frac(beta_t, sqrt{1-bar(alpha)_t}) epsilon_theta(x_t,t) ). $

或者等价写成：

$ mu_theta(x_t,t) = frac(1, sqrt{alpha_t}) ( x_t --- frac(1-alpha_t, sqrt{1-bar(alpha)_t}) epsilon_theta(x_t,t) ). $

所以，只要 $epsilon_theta(x_t,t)$ 预测对了真实 compound noise $epsilon$，那么 $mu_theta$ 就会接近真实 posterior mean $tilde(mu)_t$。因此训练 noise predictor 实际上就是在学习 reverse transition。

= 9. DDPM 的 variational bound 与 KL 分解

== 9.1 训练目标从 negative log-likelihood 开始【重要度：高】

理想目标是最小化：

$ bb(E)_{x_0 ∼ q(x_0)} [-log p_theta(x_0)]. $

直接优化困难，因此使用 variational upper bound：

$ bb(E)_{x_0 ∼ q(x_0)} [-log p_theta(x_0)] <= bb(E)_{x_0,x_{1:T} ∼ q(x_{0:T})} [ ------ log frac( p_theta(x_0,x_{1:T}) , q(x_{1:T} mid x_0) )]. $

这里 $x_{1:T}$ 是 latent variables。直觉上，这和 VAE 的 ELBO 是同一种 variational inference 逻辑：真实 likelihood 难算，于是通过引入 latent trajectory $x_{1:T}$ 构造可优化的 bound。

展开 joint distribution：

$ p_theta(x_{0:T}) = p(x_T) product_{t=1}^{T} p_theta(x_{t-1} mid x_t),  q(x_{1:T} mid x_0) = product_{t=1}^{T} q(x_t mid x_{t-1}). $

代入后得到：

$ bb(E)_q \[ -log p(x_T) ------------ sum_{t=1}^{T} log p_theta(x_{t-1} mid x_t) + sum_{t=1}^{T} log q(x_t mid x_{t-1})\]. $

通过 Bayes rule：

$ q(x_{t-1} mid x_t,x_0) = frac( q(x_t mid x_{t-1})q(x_{t-1} mid x_0) , q(x_t mid x_0) ). $

取 log：

$ log q(x_t mid x_{t-1}) = log q(x_{t-1} mid x_t,x_0) + log q(x_t mid x_0) ------------------- log q(x_{t-1} mid x_0). $

代入后会发生 telescoping sum，最终得到 KL decomposition。

== 9.2 KL decomposition【重要度：高】

最终目标可以重写为：

$ L = D_{upright("KL")} (q(x_T mid x_0)|p(x_T)) + sum_{t=2}^{T} bb(E)_q \[ D_{upright("KL")} (q(x_{t-1} mid x_t,x_0) |p_theta(x_{t-1} mid x_t))\] ------- bb(E)_q \[log p_theta(x_0 mid x_1)\]. $

三项含义：

$ L_T = D_{upright("KL")} (q(x_T mid x_0)|p(x_T)), $

用于让 final noisy distribution 匹配 Gaussian prior。

$ L_{t-1} = D_{upright("KL")} (q(x_{t-1} mid x_t,x_0) |p_theta(x_{t-1} mid x_t)), $

是核心 reverse transition matching。

$ L_0 = -bb(E)_q [log p_theta(x_0 mid x_1)], $

是 reconstruction term。

因为 $q(x_{t-1} mid x_t,x_0$) 和 $p_theta(x_{t-1} mid x_t)$ 都是 Gaussian，如果 variance 固定，KL 主要退化为 mean matching：

$ D_{upright("KL")} prop |tilde(mu) * t-mu * theta|^2. $

再将 $tilde(mu)_t$ 和 $mu_theta$ 都写成 noise 形式，就得到最终的 noise prediction MSE。

= 10. Sampling：如何真正生成图像

生成时：

$ x_T ∼ cal(N)(0,I). $

然后对 $t=T,T-1,..,1$：

$ x_{t-1} = mu_theta(x_t,t)+sigma_t z, quad quad z ∼ cal(N)(0,I). $

其中

$ mu_theta(x_t,t) = frac(1, sqrt{alpha_t}) ( x_t --- frac(1-alpha_t, sqrt{1-bar(alpha)_t}) epsilon_theta(x_t,t) ). mu_theta$ 是 denoising direction，$sigma_t z$ 是 stochastic sampling term。DDPM 的采样慢，因为每生成一张图需要调用 noise predictor (T) 次，例如 (T=1000)。

$sigma_t$ 的选择：

理论 posterior variance 为：

$ sigma_t^2 = tilde(beta)_t frac(1-bar(alpha)_{t-1}, 1-bar(alpha)_t) beta_t. $

实践中 DDPM 常简化为：

$ sigma_t^2=beta_t. $

如果

$ sigma_t>0, $

则是 stochastic sampling，即 DDPM；如果

$ sigma_t=0, $

则是 deterministic sampling，即 DDIM。

= 11. 模型对比总结

| 模型 | 核心变量 | 训练目标 | 优点 | 局限 | 重要度 |
| ----------- | ----------------------------------------- | ---------------------------------------- | ----------- | ---------------------- | --- |
| Autoencoder | deterministic hidden code (h=f_theta(x)) | reconstruction loss | 学表示、降维、预训练 | latent space 不一定可采样 | 高 |
| Linear AE | linear hidden code | squared reconstruction error | 等价 PCA，理论清楚 | 表达能力有限 | 高 |
| VAE | continuous latent (z ∼ p(z)) | ELBO = reconstruction - KL | 可采样生成，有概率解释 | 生成结果常偏模糊 | 极高 |
| VQ-VAE | discrete codebook index | reconstruction + codebook + commitment | 适合离散表示 | 量化不可微，需要 stop-gradient | 中高 |
| DDPM | noisy trajectory ($x_{1:T}$) | noise prediction MSE / variational bound | 生成质量高，训练稳定 | sampling 慢 | 极高 |

= 12. 期末最可能考的关键点

最重要的是能把 VAE 和 DDPM 的 variational thinking 连起来。VAE 引入 latent (z)，用 $q_phi(z|x)$ 近似 posterior，并最大化 ELBO；DDPM 引入 latent trajectory $x_{1:T}$，用固定 forward process 构造 tractable posterior，再训练 $p_theta(x_{t-1} mid x_t)$ 匹配它。二者都是“likelihood 难算 → 加入 variational distribution / latent variable → 优化 bound”。

VAE 必会考的公式是：

$ cal(L)(theta,phi,x) = D_{upright("KL")}(q_phi(z mid x)|p_theta(z)) + bb(E)_{q_phi(z mid x)} [log p_theta(x mid z)]. $

DDPM 必会考的公式是：

$ x_t = sqrt{bar(alpha)_t}x_0 + sqrt{1-bar(alpha)_t}epsilon,  L = bb(E)_{x_0,epsilon,t} [ |epsilon-epsilon_theta(x_t,t)|^2 ],  mu_theta(x_t,t) = frac(1, sqrt{alpha_t}) ( x_t --- frac(beta_t, sqrt{1-bar(alpha)_t}) epsilon_theta(x_t,t) ). $

理解顺序应该是：forward process 给出任意 (x_t) 的闭式采样；posterior mean 可以写成 noise 的函数；所以训练网络预测 noise 等价于学习 reverse denoising mean；最后从 $x_T ~ cal(N)(0,I)$ 逐步采样回 (x_0)。
