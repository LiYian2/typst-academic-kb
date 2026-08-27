// @title: Lecture 7 总体主线：从序列建模到通用表示学习
// @description: 本章的核心目标不是单纯介绍 Transformer，而是讲清楚一个更大的技术演进：
// @order: 999

下面是 Lecture 7: Transformer Models 的系统复习笔记。内容基于你上传的课件整理，覆盖 Transformer、BERT、GPT、ViT、CLIP 的逻辑演进与关键公式。

= Lecture 7 总体主线：从序列建模到通用表示学习

本章的核心目标不是单纯介绍 Transformer，而是讲清楚一个更大的技术演进：

传统 RNN Seq2Seq
→ Attention / Self-Attention
→ Transformer Encoder-Decoder
→ Encoder-only BERT：理解型表示学习
→ Decoder-only GPT：生成式语言模型
→ ViT：把图像变成 token 序列
→ CLIP：把图像和文本映射到同一语义空间

核心问题是：如何把不同模态的数据，例如文本序列、图像 patch、图文对，都表示成一组向量，并通过 attention 学习它们之间的关系。

= 1. Transformer 出现的动机

== 1.1 RNN Seq2Seq 的瓶颈【重要程度：★★★★★】

RNN 的基本思想是按时间步递归处理序列：

$ h_t = f(h_{t-1}, x_t) $

其中：

$ x_t $ 是第 $t$ 个输入 token 的 embedding；
$ h_t $ 是第 $t$ 步隐藏状态；
$ h_{t-1} $ 是前一步隐藏状态；
$ f $ 是 RNN/LSTM/GRU 的状态更新函数。

它的问题是：第 $t$ 步必须等待第 $t-1$ 步算完，所以训练样本内部无法并行。对于长序列，计算慢，而且长距离依赖容易衰减。

Transformer 的核心替代思路是：不要 recurrence，而是让所有 token 同时看所有 token。也就是用 self-attention 直接建模 token-token 关系。

= 2. Self-Attention：Transformer 的核心算子

== 2.1 Self-Attention 要解决什么问题【★★★★★】

给定输入 token 表示：

$ x_1, x_2, dot(s), x_n $

其中每个 $x_i$ 是一行向量，维度为：

$ x_i in R R^{1 times d_m} $

课件中原始 Transformer 设置：

$ d_m = 512 $

Self-attention 的目标是为每个 token $i$ 生成新的上下文表示 $z_i$。这个 $z_i$ 不再只表示 token 自己，而是融合了其他 token 的信息。

例如句子：

“The animal didn’t cross the street because it was too tired.”

为了理解 “it”，模型需要判断它指代 “animal” 还是 “street”。Self-attention 允许 “it” 对 “animal” 分配更高权重。

== 2.2 Query, Key, Value 的定义【★★★★★】

Transformer 引入三个可学习投影矩阵：

$ W^Q in R R^{d_m times d_k}  W^K in R R^{d_m times d_k}  W^V in R R^{d_m times d_v} $

课件中典型设置：

$ d_k = 64,quad d_v = 64 $

对于 token $i$：

$ q_i = x_i W^Q  k_i = x_i W^K  v_i = x_i W^V $

其中：

$ q_i $ 是 query，表示“当前 token 想找什么信息”；
$ k_i $ 是 key，表示“这个 token 提供的信息标签”；
$ v_i $ 是 value，表示“这个 token 真正贡献的内容”；
$ W^Q, W^K, W^V $ 是可学习参数。

直觉上，$q_i$ 和 $k_j$ 的相似度越高，说明 token $i$ 越应该关注 token $j$。

== 2.3 Scaled Dot-Product Attention【★★★★★】

先计算 token $i$ 对 token $j$ 的注意力分数：

$ s_{i,j} = q_i k_j^->p $

然后进行缩放：

$ s_{i,j} = frac(q_i k_j^->p, sqrt{d_k}) $

为什么除以 $sqrt{d_k}$？因为当 $d_k$ 很大时，点积的方差会变大，softmax 容易进入饱和区，导致梯度变小。缩放可以稳定训练。

然后 softmax 得到注意力权重：

$ alpha_{i,j}
=

frac{exp(s_{i,j})}
{sum_{j'=1}^{n} exp(s_{i,j'})} $

其中：

$ alpha_{i,j} $ 表示 token $i$ 对 token $j$ 的注意力权重；
对固定 $i$，所有 $j$ 的权重和为 1：

$ sum_{j=1}^{n} alpha_{i,j} = 1 $

最后得到新表示：

$ z_i = sum_{j=1}^{n} alpha_{i,j} v_j $

也就是说，$z_i$ 是所有 value 向量的加权平均。

= 3. Self-Attention 的矩阵形式

== 3.1 矩阵表达【★★★★★】

把所有 token 表示堆叠成矩阵：

$ X =
mat(x_1 \
x_2 \
.. \
x_n)
in R R^{n times d_m} $

则：

$ Q = X W^Q  K = X W^K  V = X W^V $

其中：

$ Q,K in R R^{n times d_k}  V in R R^{n times d_v} $

整体 attention 输出为：

$ Z = op("softmax")(frac(Q K^->p, sqrt(d_k))) V $

其中：

$ Q K^->p in R R^{n times n} $

这个矩阵的第 $(i,j)$ 个元素就是 token $i$ 对 token $j$ 的 attention score。

这是 Transformer 可以并行的关键：所有 token pair 的关系一次性通过矩阵乘法算出。

= 4. Multi-Head Attention

== 4.1 为什么需要多头注意力【★★★★★】

单头 attention 只能学习一种 token 关系。例如可能只关注主谓关系，或者只关注指代关系。多头注意力让模型在多个子空间中并行学习不同类型的关系。

引入 $h$ 组投影矩阵：

$ W_i^Q,\ W_i^K,\ W_i^V,quad i=1,dot(s),h $

每一组对应一个 attention "head"。

课件中：

$ h = 8,quad d_m=512,quad d_v=64 $

因此 $8$ 个 "head" 拼接后维度为：

$ 8 times 64 = 512 $

刚好回到 $d_m$。

== 4.2 多头公式【★★★★★】

第 $i$ 个 "head"：

$ Q_i = X W_i^Q  K_i = X W_i^K  V_i = X W_i^V $

输出：

$ Z_i = op("softmax")(frac(Q_i K_i^->p, sqrt(d_k))) V_i $

然后拼接：

$ Z = op("Concat")(Z_1,dot(s),Z_h) $

最后用输出投影矩阵：

$ Z <- Z W^O $

其中：

$W^O$ 是输出投影矩阵，用于混合不同 heads 的信息，并保持最终维度为 $d_m$。

= 5. Encoder Block

== 5.1 Encoder Block 的结构【★★★★★】

一个 Transformer encoder block 包含两个主要子层：

Self-Attention sub-layer
Feed Forward Neural Network sub-layer

每个位置上的 FFN 参数共享，但不同位置独立计算。也就是说，对于每个 token 表示 $z_i$，都会通过同一个前馈网络：

$ op("F F N")(z_i) $

典型结构可以写成：

$ op("FFN")(x) = sigma(x W_1 + b_1) W_2 + b_2 $

其中：

$W_1,W_2$ 是前馈网络权重；
$b_1,b_2$ 是偏置；
$sigma$ 是非线性激活函数，例如 ReLU 或 GELU。

Self-attention 负责 token 间通信；FFN 负责对每个 token 的表示做非线性特征变换。

== 5.2 Residual Connection 与 Layer Normalization【★★★★★】

每个子层后面都有残差连接和 LayerNorm：

$ op("LayerNorm")$x + op("Sublayer")(x$) $

其中：

$ x $ 是子层输入；
$ op("Sublayer")(x) $ 可以是 self-attention 或 FFN；
$ x + op("Sublayer")(x) $ 是残差连接。

残差连接的作用是让梯度更容易传递，从而训练深层模型。LayerNorm 的作用是稳定每层的激活分布。

= 6. 为什么 NLP 用 LayerNorm 而不是 BatchNorm

== 6.1 BatchNorm 的问题【★★★★☆】

BatchNorm 通常对 batch 维度统计均值和方差：

$ mu_B = frac(1, m)sum_{i=1}^{m} x_i  sigma_B^2 = frac(1, m)sum_{i=1}^{m}(x_i-mu_B)^2  hat(x)_i = frac(x_i-mu_B, sqrt{sigma_B^2+epsilon}) $

问题在 NLP 中尤其明显：

第一，序列长度可变，batch 内样本 padding 后统计复杂。
第二，BatchNorm 依赖 batch size，而大模型常因显存限制使用小 batch。
第三，自回归生成时，时间步逐步生成，batch statistics 会随时间变化，不稳定。
第四，经验上 LayerNorm 在 NLP Transformer 中表现更好。

== 6.2 LayerNorm 的直觉【★★★★☆】

LayerNorm 对单个样本内部的 feature 维度归一化，而不是跨 batch 归一化：

$ mu = frac(1, d)sum_{k=1}^{d} x_k  sigma^2 = frac(1, d)sum_{k=1}^{d}(x_k-mu)^2  op("L N")(x_k)
=

gamma frac(x_k-mu, sqrt{sigma^2+epsilon})+beta $

其中：

$d$ 是 hidden dimension；
$gamma,beta$ 是可学习缩放和平移参数；
$epsilon$ 防止除零。

= 7. Transformer Encoder

== 7.1 Encoder 堆叠【★★★★★】

原始 Transformer encoder 由：

$ N = 6 $

个 encoder blocks 堆叠而成。

每个 block 结构相同，但参数不同，也就是不同层之间不共享参数。

Encoder 的输入是 token embedding 加 positional encoding，输出是一组上下文 token 表示，供 decoder 或下游任务使用。

= 8. Positional Encoding

== 8.1 为什么需要位置编码【★★★★★】

Transformer 没有 recurrence，也没有 convolution，所以 self-attention 本身对顺序不敏感。换句话说，如果不加入位置编码，模型看见的是一个 set，而不是 sequence。

因此需要加入 positional encoding：

$ "input"_i = x_i + P E_i $

其中：

$ x_i $ 是 token embedding；
$ P E_i $ 是第 $i$ 个位置的位置编码。

== 8.2 正弦位置编码公式【★★★★★】

对于位置 $"pos"$，维度 $2i$ 和 $2i+1$：

$ op("PE")(italic("pos"),2i) = sin(frac(italic("pos"), 10000^{2i/d_m}))  op("PE")(italic("pos"),2i+1) = cos(frac(italic("pos"), 10000^{2i/d_m})) $

其中：

$"pos"$ 是 token 的位置；
$i$ 是维度对的索引；
$d_m$ 是 embedding 维度；
课件中 $d_m=512$，所以：

$ 0 <= i <= 255 $

不同维度使用不同频率，低维变化快，高维变化慢，从而覆盖不同尺度的位置关系。

== 8.3 为什么 Sinusoidal PE 能表达相对位置【★★★★★】

定义频率：

$ omega_i = frac(1, 10000^{2i/d_m}) $

则：

$ P E_{"pos",2i} = sin(omega_i "pos")  P E_{"pos",2i+1} = cos(omega_i "pos") $

考虑位置偏移 $k$：

$ mat(sin(omega_i("pos"+k)) \
cos(omega_i("pos"+k))) $

由三角恒等式：

$ sin(a+b)=sin a cos b+cos a sin b  cos(a+b)=cos a cos b-sin a sin b $

可得：

$ mat(sin(omega_i("pos"+k)) \
cos(omega_i("pos"+k)))
=

mat(cos(omega_i k), sin(omega_i k) \
-sin(omega_i k), cos(omega_i k))
mat(sin(omega_i "pos") \
cos(omega_i "pos")) $

记作：

$ P E_{"pos"+k} = M_k P E_{{"pos"}} $

关键点是：$M_k$ 只依赖偏移 $k$，不依赖绝对位置 $"pos"$。因此 self-attention 可以通过线性变换学习相对位置信息。

= 9. Transformer Decoder

== 9.1 Decoder 的任务【★★★★★】

Decoder 是自回归生成器。它每一步生成一个 token：

$ P(y_t mid y_1,dot(s),y_{t-1}, X) $

其中：

$X$ 是 encoder 输入序列；
$y_1,dot(s),y_{t-1}$ 是已经生成的目标 token。

Decoder 输入是已经生成的 token embedding 加 positional encoding。

== 9.2 Decoder Block 和 Encoder Block 的区别【★★★★★】

Decoder block 有三个子层：

Masked Self-Attention
Encoder-Decoder Attention
Feed Forward Network

相比 encoder block，多了 encoder-decoder attention。

== 9.3 Encoder-Decoder Attention【★★★★★】

在 encoder-decoder attention 中：

queries 来自 decoder：

$ Q_i = X_{"decoder"}W_i^Q $

keys 和 values 来自 encoder：

$ K_i = X_{"encoder"}W_i^K  V_i = X_{"encoder"}W_i^V $

输出：

$ Z_i =
op("softmax")
(
frac(Q_i K_i^->p, sqrt{d_k})
)V_i $

直觉是：decoder 当前生成位置提出 query，去 encoder 输出中检索相关源语言信息。

== 9.4 Masked Self-Attention【★★★★★】

Decoder 不能看未来 token。对于位置 $i$，只能 attend 到：

$ j <= i $

实现方式是：

$ s_{i,j} = -oo,quad j>i $

softmax 后：

$ alpha_{i,j}=0,quad j>i $

即：

$ alpha_{i,j}
=

frac(exp(s_{i,j}), sum_{j'}exp(s_{i,j'})) $

当 $s_{i,j}=-oo$ 时：

$ exp(s_{i,j})=0 $

所以未来 token 的注意力权重为 0。

== 9.5 Decoder 输出层【★★★★☆】

最后 decoder 输出经过 linear + softmax，得到词表分布：

$ P(y_t = v mid y_{<t}, X) = op("softmax")(W h_t + b)_v $

其中：

$h_t$ 是 decoder 在位置 $t$ 的最终 hidden state；
$W,b$ 是输出层参数；
$v$ 是词表中的某个 token。

训练 loss 和 RNN 语言模型类似，通常是交叉熵：

$ cal(L) = -sum_t log P(y_t^ast mid y_{<t}^ast, X) $

其中 $y_t^ast$ 是真实 token。

= 10. BERT：Transformer Encoder for Representation Learning

== 10.1 BERT 的核心定位【★★★★★】

BERT = Bidirectional Encoder Representations from Transformers。

它只使用 Transformer encoder，不使用 decoder。它的目标不是从左到右生成文本，而是学习每个 token 的双向上下文表示。

BERT 输入一个 token 序列，输出一组 contextual vectors：

$ T_1,T_2,dot(s),T_n $

每个 $T_i$ 同时融合左侧和右侧上下文。

== 10.2 BERT 模型规模【★★★★☆】

BERT BASE：

$ N=12,quad d_m=768,quad h=12 $

约 110M 参数。

BERT LARGE：

$ N=24,quad d_m=1024,quad h=16 $

约 340M 参数。

== 10.3 BERT 输入结构【★★★★★】

BERT 训练输入是句子对 $(A,B)$：

$ ["C L S"], A, ["S E P"], B, ["S E P"] $

每个 token 的最终输入 embedding 是三部分相加：

$ x_i = e_i^{"token"} + e_i^{"segment"} + e_i^{"position"} $

其中：

$e_i^{"token"}$ 是 token embedding；
$e_i^{"segment"}$ 表示属于句子 A 还是句子 B；
$e_i^{"position"}$ 是位置 embedding。

Segment embedding 有两类：

$ E_A,quad E_B $

用于区分两个句子。

= 11. BERT 的训练目标

== 11.1 MLM：Masked Language Model【★★★★★】

BERT 选择 15% 的 token 用于预测。

对于被选中的 token：

80% 替换为 `[MASK]`
10% 替换为随机 token
10% 保持不变

重点：这 15% 被选中的 token 都参与 loss，不是只有被替换成 `[MASK]` 的 80% 参与。

对于第 $i$ 个被选中 token，BERT 使用输出表示 $T_i$ 预测原始 token：

$ P(x_i mid X_{∖ i}) = op("softmax")(W T_i + b) $

MLM loss：

$ cal(L)_{"M L M"} = -sum_{i in M} log P(x_i^ast mid X_{"corrupted"}) $

其中：

$M$ 是被选中的 15% token 位置集合；
$x_i^ast$ 是原始 token；
$X_{"corrupted"}$ 是经过 mask/random/unchanged 操作后的输入。

MLM 的意义是：让 $T_i$ 依赖左右两边上下文，因此 BERT 是 bidirectional representation model。

== 11.2 为什么不用 100% `[MASK]`【★★★★☆】

因为预训练时有 `[MASK]`，但 fine-tuning 时通常没有 `[MASK]`。如果所有被选 token 都换成 `[MASK]`，会造成 pretraining-finetuning mismatch。

所以 10% random token 和 10% unchanged 是为了缓解这种分布不一致。

== 11.3 NSP：Next Sentence Prediction【★★★★☆】

BERT 还训练句子关系判断任务。

50% 情况下，句子 B 真的跟在句子 A 后面；
50% 情况下，句子 B 是随机句子。

使用 `[CLS]` 的输出向量 $C$ 预测：

$ P("IsNext" mid A, B) = op("softmax")(W C+b) $

NSP loss：

$ cal(L)_{"N S P"} = -log P(y_{"NSP"}^ast mid C) $

总 loss：

$ cal(L)_{"B E R T"} = cal(L)_{"M L M"} + cal(L)_{"N S P"} $

注意：课件强调 $C$ 不一定天然是句子内容的好 summary，它只是被训练去判断句子是否相邻。因此下游任务仍然需要 fine-tuning。

= 12. BERT 的下游任务使用方式

== 12.1 Sentence-pair Classification【★★★★★】

例如 MNLI 文本蕴含任务。输入句子对，取 `[CLS]` 表示 $C$：

$ hat(y) = op("softmax")(W C+b) $

Fine-tuning 会让 $C$ 从 “IsNext 判断表示” 转变为 “适合当前任务的句对表示”。

== 12.2 Single-sentence Classification【★★★★★】

单句分类同样用 `[CLS]`：

$ hat(y) = op("softmax")(W C+b) $

例如情感分类、主题分类。

== 12.3 Token-level Tasks【★★★★★】

例如 NER，使用每个 token 的输出：

$ hat(y)_i = op("softmax")(W T_i+b) $

其中 $T_i$ 是第 $i$ 个 token 的 contextual representation。

= 13. GPT：Transformer Decoder as Language Model

== 13.1 GPT 的定位【★★★★★】

GPT = Generative Pre-trained Transformer。

它是 autoregressive language model，只使用 Transformer decoder 架构。

BERT 是 encoder-only，适合理解；
GPT 是 decoder-only，适合生成。

== 13.2 GPT 训练目标【★★★★★】

GPT 最大化 next-token log-likelihood：

$ L =
sum_i
log P$x_i mid x_{i-k},dot(s),x_{i-1}

其中：

$x_i$ 是当前位置 token；
$k$ 是 context window size；
$x_{i-k},dot(s),x_{i-1}$ 是模型可见的历史上下文。

等价地，最小化负对数似然：

$ cal(L)_{"GPT"} = -sum_i log P(x_i mid x_{i-k},dot(s),x_{i-1}) $

GPT 使用 causal mask，保证当前位置不能看到未来 token。

== 13.3 GPT 版本演进【★★★★☆】

GPT-1：BooksCorpus，约 7000 本书。
GPT-2：WebText，约 8M documents，40GB text。
GPT-3：约 45TB text。
InstructGPT：在 GPT 基础上加入人类反馈对齐。

演进逻辑是：

语言模型预训练
→ 更大数据和模型规模
→ in-context learning
→ human feedback alignment

== 13.4 Few-shot / In-context Learning【★★★★☆】

GPT-3 的重要现象是：不需要参数 fine-tuning，只需要 prompt 中给任务描述和少量样例。

Prompt 通常包含：

Task description
Examples
New input prefix

模型通过继续生成完成任务。

= 14. InstructGPT：从“会续写”到“听指令”

== 14.1 GPT-3 的问题【★★★★★】

GPT-3 的训练目标是预测下一个 token，不是服从用户意图。因此可能出现：

不按指令回答；
hallucination；
不可解释；
有毒或偏见输出。

核心矛盾是：

$ "next-token prediction" != "helpful and safe instruction following" $

== 14.2 InstructGPT 三步训练【★★★★★】

第一步：Supervised Fine-Tuning, SFT

人工标注者对 prompt 写理想回答。然后用监督学习微调 GPT-3：

$ cal(L)_{"SFT"} = -sum_t log pi_{theta}(y_t^ast mid x, y_{ < t }^ast) $

其中：

$x$ 是 prompt；
$y_t^ast$ 是人工示范回答中的 token；
$pi_theta$ 是语言模型策略。

第二步：Reward Model, RM

给同一 prompt 生成多个回答，让人类比较偏好。训练 reward model 输出标量 reward：

$ r_phi(x,y) $

其中：

$x$ 是 prompt；
$y$ 是模型回答；
$r_phi$ 表示人类偏好评分。

第三步：PPO 强化学习优化

用 reward model 作为奖励，继续优化 SFT policy，使模型输出更符合人类偏好。

概念上：

$ max_theta bb(E)_{y ∼ pi_theta(dot mid x)} [r_phi(x,y)] $

实际 PPO 通常还包含 KL 约束，避免模型偏离原始 SFT 模型太远：

$ max_theta bb(E)[r_phi(x,y) - beta D_{"KL"}(pi_theta(dot mid x) "||" pi_{"SFT"}(dot mid x))] $

其中：

$beta$ 控制 KL 惩罚强度；
$pi_{"S F T"}$ 是监督微调后的参考策略；
$pi_theta$ 是当前要优化的策略。

= 15. ViT：Transformer Encoder Applied to Vision

== 15.1 ViT 的核心思想【★★★★★】

Self-attention 接收的是一组向量。图像原本是三维 tensor：

$ x in R R^{H times W times C} $

其中：

$H$ 是图像高度；
$W$ 是图像宽度；
$C$ 是通道数。

ViT 的思想是：把图像切成 patch，把 patch 当作 token。

== 15.2 Patch Embedding 公式【★★★★★】

将图像切成大小为：

$ P times P $

的 patch。patch 数量为：

$ N = frac(H W, P^2) $

每个 patch 展平为：

$ x_p in R R^{1 times $P^2C$} $

再乘以线性投影矩阵：

$ E in R R^{$P^2C$times D} $

得到 patch embedding：

$ z_p = x_p E $

其中：

$D$ 是 Transformer hidden dimension；
$E$ 是可学习 patch projection matrix。

== 15.3 ViT 输入序列【★★★★★】

ViT 还加入一个可学习 class token：

$ x_{"class"} $

输入序列为：

$ z_0 = [x_{"class"}; x_p^1 E; x_p^2 E; dot(s); x_p^N E] + E_{"pos"} $

其中：

$E_{"pos"}$ 是可学习位置 embedding；
课件强调：ViT 中的位置 embedding 通常是 learned，而 NLP 原始 Transformer 使用 sinusoidal positional encoding。

== 15.4 ViT Encoder 结构【★★★★☆】

ViT 使用标准 Transformer encoder：

$ z_l' = op("MSA")(op("LN")(z_{l-1})) + z_{l-1}  z_l = op("MLP")(op("LN")(z_l')) + z_l' $

其中：

MSA 是 multi-"head" self-attention；
MLP 是两层前馈网络；
LN 是 LayerNorm；
残差连接保证深层训练稳定。

分类时使用最终 class token 表示：

$ y = op("MLP Head")(z_L^0) $

其中 $z_L^0$ 是最后一层 class token。

= 16. ViT 与 CNN 的本质差异

== 16.1 Inductive Bias 差异【★★★★★】

CNN 内置图像归纳偏置：

locality：局部感受野；
2D neighborhood structure：二维邻域结构；
translation equivariance：平移等变性。

ViT 的 self-attention 是 global 的，没有强烈图像结构假设。它只通过 patch 位置 embedding 学习空间结构。

因此：

CNN 小数据更强；
ViT 大数据预训练后更强；
ViT 更依赖数据规模。

== 16.2 ViT 学到了什么【★★★★☆】

课件指出：

ViT 的 position embeddings 会恢复图像 grid structure，同一行同一列的 patch embedding 更相似。
低层同时包含 local 和 global features。
高层主要包含 global features。
ViT feature maps 通常比 ResNet feature maps 更有语义意义。
ViT 对自然分布偏移更鲁棒。

= 17. CLIP：Contrastive Language-Image Pre-training

== 17.1 CLIP 的核心问题【★★★★★】

ResNet / ViT 通常使用 labeled images 训练。
BERT / GPT 使用 text 训练。
CLIP 使用 400M image-text pairs，把图像和文本放入同一个 embedding space。

它属于 vision-language model, VLM。

核心目标是：让匹配的图文向量相似，不匹配的图文向量不相似。

== 17.2 CLIP Contrastive Loss【★★★★★】

对于一个 batch，包含 $N$ 个图文对：

$ (I_1,T_1),dot(s),(I_N,T_N) $

CLIP loss：

$ L_{"CLIP"} = -frac(1, 2N) sum_{i=1}^{N} log frac(exp(I_i dot T_i), sum_{j=1}^{N} exp(I_i dot T_j)) - frac(1, 2N) sum_{j=1}^{N} log frac(exp(I_j dot T_j), sum_{i=1}^{N} exp(I_i dot T_j)) $

其中：

$I_i$ 是第 $i$ 张图像的 embedding；
$T_i$ 是与第 $i$ 张图像匹配的文本 embedding；
$I_i dot T_j$ 是图像和文本向量的相似度，通常是 dot product 或 cosine similarity；
第一个求和是 image-to-text contrastive loss；
第二个求和是 text-to-image contrastive loss。

直觉：

对每张图像 $I_i$，正确文本 $T_i$ 应该在所有文本中概率最高；
对每段文本 $T_j$，正确图像 $I_j$ 应该在所有图像中概率最高。

== 17.3 CLIP Zero-shot Classification【★★★★★】

对于类别 $c$，构造文本 prompt：

$ T_c = "TextEncoder"$"``A photo of a {c".''}

给定图像 embedding $I$，类别概率为：

$ P(c mid I) = frac(exp(I dot T_c), sum_{c'=1}^{C} exp(I dot T_{c'})) $

其中：

$C$ 是类别数；
$T_c$ 是类别 $c$ 的文本向量；
$I$ 是图像向量。

课件中的关键解释是：$T_c$ 本质上可以看作类别 $c$ 的分类器权重向量。也就是说，CLIP 不需要重新训练分类头，而是用自然语言 prompt 生成分类器。

== 17.4 CLIP 的鲁棒性与扩展【★★★★☆】

Zero-shot CLIP 在自然分布偏移下比标准 ImageNet 模型更鲁棒。课件指出，它可以缩小 robustness gap，最多约 75%。

扩展方向：

ALIGN 使用 1.8B noisy image-text pairs；
BASIC 使用 6.6B image-text pairs 和更大模型；
还可以让 LLM 生成类别的视觉概念描述，用更丰富的 prompt 提升分类鲁棒性。

= 18. 总体方法演进表

| 阶段 | 核心结构 | 解决的问题 | 前一阶段瓶颈 | 新方法如何改进 | 重要程度 |
| -------------------- | ---------------------- | ---------------- | ------------- | ---------------------- | ----- |
| RNN Seq2Seq | recurrent hidden state | 序列到序列建模 | 无法并行，长程依赖弱 | 按时间递归建模上下文 | ★★★★☆ |
| Transformer | self-attention | 并行序列建模 | RNN 顺序依赖 | 所有 token 同时交互 | ★★★★★ |
| Multi-"head" Attention | 多组 QKV | 多关系建模 | 单头表达关系单一 | 不同 "head" 学不同关系 | ★★★★★ |
| BERT | encoder-only | 双向语言表示 | 单向 LM 不能看右上下文 | MLM 学双向上下文 | ★★★★★ |
| GPT | decoder-only | 自回归生成 | BERT 不适合生成 | causal LM 逐 token 生成 | ★★★★★ |
| InstructGPT | SFT + RM + PPO | 指令对齐 | GPT 只会续写 | 人类反馈优化偏好 | ★★★★★ |
| ViT | patch + encoder | 图像 Transformer 化 | CNN 偏置强，扩展受限 | 图像转 token，全局 attention | ★★★★☆ |
| CLIP | 图文对比学习 | 图文共享语义空间 | 分类依赖固定标签 | 文本 prompt 生成分类器 | ★★★★★ |

= 19. 期末复习抓重点

最重要的公式必须会：

Self-attention：

$ Z =
op("softmax")
(
frac(Q K^->p, sqrt{d_k})
)V $

Multi-"head" attention：

$ Z_i =
op("softmax")
(
frac(Q_i K_i^->p, sqrt{d_k})
)V_i  Z = op("Concat")(Z_1,dot(s),Z_h)W^O $

Sinusoidal PE：

$ op("PE")("pos", 2i) = sin(frac("pos", 10000^{2i/d_m})) $

$ op("PE")("pos", 2i+1) = cos(frac("pos", 10000^{2i/d_m})) $

GPT objective：

$ L =
sum_i
log P$x_imid x_{i-k},dot(s),x_{i-1}

BERT total loss：

$ cal(L)_{"BERT"} = cal(L)_{"MLM"} + cal(L)_{"NSP"} $

ViT patch number：

$ N = frac(H W, P^2) $

CLIP loss：

$ L_{"CLIP"} = -frac(1, 2N) sum_{i=1}^{N} log frac(exp(I_i dot T_i), sum_{j=1}^{N} exp(I_i dot T_j)) - frac(1, 2N) sum_{j=1}^{N} log frac(exp(I_j dot T_j), sum_{i=1}^{N} exp(I_i dot T_j)) $

= 20. 本章最核心的一句话

Transformer 的本质是：把输入表示成一组 token vectors，然后用 attention 学习 token 之间的关系；BERT、GPT、ViT、CLIP 都是在这个统一框架下，对输入形式、mask 方式、训练目标和输出任务做不同设计。
