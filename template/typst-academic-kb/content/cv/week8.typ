// @title: Week 8：Recurrent Neural Networks and Transformer
// @description:
// @order: 999

下面是基于课件 _AIAA3201-week8.pdf_ 的期末复习版学习笔记。课件主线是：从 RNN 处理序列的局限，演进到 attention，再抽象为 general attention layer，最后形成 Transformer encoder-decoder 架构。

= Week 8：Recurrent Neural Networks and Transformer

== 0. 本章核心目标与“进化树”

本章要解决的根本问题是：

_如何让神经网络有效处理可变长度的序列输入/输出，并且能够建模长距离依赖？_

传统 CNN / vanilla NN 通常假设固定大小输入和固定大小输出。例如图像分类是 image → class。但很多任务不是这样：

图像描述：image → sequence of words
视频分类：sequence of frames → action class
视频描述：sequence of frames → caption
机器翻译：source sentence → target sentence

本章技术演进树可以概括为：

```text
Vanilla NN / CNN
 ↓ 不能自然处理可变长度序列
RNN
 ↓ 用 hidden state 逐步读入序列
Seq2Seq Encoder-Decoder
 ↓ 把整个输入压缩成一个 context vector，形成信息瓶颈
RNN + Attention
 ↓ 每个 decoding step 动态选择输入中相关部分
General Attention Layer
 ↓ 抽象成 query-key-value 的通用操作
Self-Attention
 ↓ 输入之间直接两两交互，但缺少顺序信息
Positional Encoding + Masked Attention + Multi-Head Attention
 ↓
Transformer Encoder-Decoder
 ↓ 高并行、长距离依赖建模强，但 memory cost 高
Vision Transformer / Transformer-only vision-language models
```

重要程度标注：

| 模块 | 重要程度 | 期末考重点 |
| --------------------------------- | ----: | ------------------------- |
| RNN hidden state update | ★★★★★ | 必考基础 |
| Seq2Seq bottleneck | ★★★★★ | attention 的动机 |
| Attention 计算流程 | ★★★★★ | alignment、softmax、context |
| Q/K/V self-attention | ★★★★★ | Transformer 核心 |
| Positional encoding | ★★★★☆ | 为什么需要、sin/cos 直觉 |
| Masked self-attention | ★★★★☆ | autoregressive decoding |
| Multi-"head" attention | ★★★★☆ | 为什么多个 "head" |
| Transformer encoder/decoder block | ★★★★★ | 架构题常考 |
| 优化器 recap | ★★★☆☆ | 可能作为选择/简答 |

---

= 1. Recap：优化与正则化

这一部分是上一讲复习，但期末仍可能考概念选择题。

== 1.1 SGD 的三个问题

SGD 基本更新：

$ x_{t+1}=x_t-alpha nabla f(x_t) $

其中：

$ x_t $

表示第 (t) 次迭代的参数向量；

$ alpha $

是 learning rate；

$ nabla f(x_t) $

是当前参数处的梯度。

SGD 的三个典型问题：

第一，loss landscape 条件数很大。某些方向变化很快，某些方向变化很慢。SGD 会在陡峭方向震荡，在平坦方向前进很慢。

第二，遇到 local minimum 或 saddle point 时容易停滞。深度学习中 saddle point 通常比真正 bad local minimum 更常见。

第三，mini-batch gradient 有噪声。真实 empirical loss 是：

$ L(W)=frac(1, N)sum_{i=1}^{N}L_i(x_i,y_i,W) $

梯度为：

$ nabla_W L(W)=frac(1, N)sum_{i=1}^{N}nabla_W L_i(x_i,y_i,W) $

但 SGD 每次只用 mini-batch 估计这个梯度，所以方向会抖动。

---

== 1.2 Momentum：用历史梯度平滑更新

课件公式：

$ v_{t+1}=rho v_t+nabla f(x_t)  x_{t+1}=x_t-alpha v_{t+1} $

符号解释：

$ v_t $

是 velocity，即历史梯度的指数累积；

$ rho $

是 momentum coefficient，也可理解为 friction，常用 (0.9) 或 (0.99)；

$ alpha $

是 learning rate。

直觉：Momentum 不是给每个参数自适应学习率，而是让更新方向具有惯性。它会抑制来回震荡，强化持续一致的下降方向。

---

== 1.3 AdaGrad：每个参数自适应学习率

AdaGrad 累积平方梯度：

$ G_t = G_{t-1}+g_t op("odot") g_t $

更新：

$ x_{t+1}=x_t-alpha frac(g_t, sqrt{G_t}+epsilon) $

其中：

$ g_t=nabla f(x_t) $

表示当前梯度；

$ G_t $

是历史平方梯度累积；

$ op("odot") $

表示 element-wise multiplication；

$ epsilon $

是防止除零的小常数。

直觉：如果某个方向梯度长期很大，分母会变大，该方向步长变小；如果某个方向梯度较小，分母增长慢，相对步长更大。

问题：由于 (G_t) 不断累积，分母会越来越大，导致后期 step size 可能趋近于 0。

---

== 1.4 RMSProp：修复 AdaGrad 步长衰减过快

RMSProp 不再无限累积平方梯度，而是使用指数滑动平均：

$ G_t=gamma G_{t-1}+(1-gamma)g_t op("odot") g_t $

更新：

$ x_{t+1}=x_t-alpha frac(g_t, sqrt{G_t}+epsilon) $

其中：

$ gamma $

是 decay rate。

它保留 AdaGrad 的 per-parameter adaptive learning rate，但避免历史太久的梯度永久压低学习率。

---

== 1.5 Adam：Momentum + RMSProp + Bias Correction

Adam 同时维护一阶矩和二阶矩：

$ m_t=beta_1 m_{t-1}+(1-beta_1)g_t  v_t=beta_2 v_{t-1}+(1-beta_2)g_t op("odot") g_t $

因为 (m_0=0, v_0=0)，初期估计偏小，所以要 bias correction：

$ hat(m)_t=frac(m_t, 1-beta_1^t)  hat(v)_t=frac(v_t, 1-beta_2^t) $

最终更新：

$ x_{t+1}=x_t-alpha frac(hat(m)_t, sqrt{hat(v)_t}+epsilon) $

常用超参数：

$ beta_1=0.9,quad beta_2=0.999,quad alpha=10^{-3}\ "or"\ 5times 10^{-4} $

Adam 的本质：一阶矩 (m_t) 提供 momentum，二阶矩 (v_t) 提供 RMSProp/AdaGrad-style adaptive scaling。

---

= 2. RNN：为什么需要序列模型？

== 2.1 Vanilla Neural Network 的局限

普通神经网络通常是 one-to-one：

$ x >-> y $

例如 image classification。问题是它不自然支持 variable-length input/output。

序列任务有多种形式：

| 任务类型 | 输入 | 输出 | 例子 |
| ------------------- | ---- | ---- | -------------------------------- |
| one-to-one | 单个输入 | 单个输出 | image classification |
| one-to-many | 单个输入 | 序列输出 | image captioning |
| many-to-one | 序列输入 | 单个输出 | video classification |
| many-to-many | 序列输入 | 序列输出 | machine translation |
| synced many-to-many | 序列输入 | 每步输出 | frame-level video classification |

---

== 2.2 RNN 的核心思想：hidden state

RNN 的关键是有内部状态 (h_t)。每读入一个输入 (x_t)，状态都会更新：

$ h_t=f_W(h_{t-1},x_t) $

输出为：

$ y_t=g_{W_o}(h_t) $

符号解释：

$ x_t $

是第 (t) 个输入向量；

$ h_t $

是第 (t) 步 hidden state，保存历史信息；

$ f_W $

是状态转移函数，参数为 (W)；

$ g_{W_o} $

是输出函数，参数为 (W_o)；

$ y_t $

是第 (t) 步输出。

Vanilla RNN / Elman RNN 常见形式是：

$ h_t=tanh(W_{{"xh"}}x_t+W_{{"hh"}}h_{t-1}+b_h)  y_t=W_{{"hy"}}h_t+b_y $

其中：

$ W_{{"xh"}} $

把输入映射到 hidden space；

$ W_{{"hh"}} $

把上一时刻 hidden state 映射到当前 hidden state；

$ W_{{"hy"}} $

把 hidden state 映射到输出 logits；

$ b_h,b_y $

是 bias。

最重要的一点：_同一组参数 (W) 在所有 time step 共享。_

这就是 RNN 能处理任意长度序列的原因。

---

= 3. RNN 的计算图与任务类型

== 3.1 Many-to-many

输入：

$ x_1,x_2,dot(s),x_T $

状态递推：

$ h_t=f_W(h_{t-1},x_t) $

输出：

$ y_t=g_{W_o}(h_t) $

总损失通常是每个 time step 的损失求和：

$ L=sum_{t=1}^{T}L_t(y_t,hat(y)_t) $

适用于 frame-level classification、language modeling 等。

---

== 3.2 Many-to-one

只使用最终 hidden state：

$ h_T=f_W(h_{T-1},x_T)  y=g(h_T) $

适用于 sentiment classification、video classification 等。

直觉：整个输入序列的信息被压缩到最后一个 (h_T)。

---

== 3.3 One-to-many

单个输入生成序列。例如 image captioning。

图像特征 (v) 可以作为初始信息注入 RNN：

$ h_t=tanh(W_{{"xh"}}x_t+W_{{"hh"}}h_{t-1}+W_{{"ih"}}v) $

其中：

$ v $

是 image feature；

$ W_{{"ih"}} $

把图像特征映射进 hidden state。

然后每一步输出一个 token，并把上一步 token 作为下一步输入。

---

= 4. Character-level Language Model

以 vocabulary：

$ [h,e,l,o] $

训练序列：

$ "hello" $

语言模型的目标是预测下一个字符：

$ p(x_{t+1}mid x_{<= t}) $

RNN 每一步输出 logits：

$ o_t=W_{{"hy"}}h_t+b_y $

经过 softmax 得到概率：

$ p(x_{t+1}=k mid x_{<= t})=frac(exp(o_{t,k}), sum_{j=1}^{V}exp(o_{t,j})) $

其中：

$ V $

是 vocabulary size；

$ o_{t,k} $

是第 (t) 步对第 (k) 个 token 的 logit。

训练时使用 teacher forcing：输入真实前缀，预测下一个字符。

测试时使用 sampling：

```text
输入 <START>
模型输出下一个字符分布
sample 一个字符
把 sampled 字符再喂回模型
重复直到生成 <END> 或达到最大长度
```

重要直觉：RNN 不是每个 block 生成新 token，而是每个时间步通过最后输出层产生 token distribution；sample 出来的 token 再作为下一时间步输入。

---

= 5. Seq2Seq：Encoder-Decoder 架构

== 5.1 基本结构

Encoder 读取输入序列：

$ x_1,x_2,dot(s),x_T $

并生成 hidden states：

$ u_t=R N N(x_t,u_{t-1}) $

然后把整个输入压缩为一个 vector，例如：

$ h_0=f_W(z) $

其中：

$ z $

可以是 encoder 的最后状态或所有 encoder states 的组合；

$ h_0 $

是 decoder 的初始 hidden state。

Decoder 生成输出序列：

$ y_t=g_V(y_{t-1},h_{t-1},c) $

其中：

$ c $

是 context vector，早期 seq2seq 中通常就是 encoder 最终状态。

---

== 5.2 核心瓶颈：fixed-length context vector

问题是：

$ c $

必须承载整个输入序列的所有信息。

当输入很长或输出很长时，单个向量会成为信息瓶颈。模型需要把“所有未来可能需要的信息”都压缩进一个固定维度向量。

这就是 attention 的动机。

---

= 6. RNN + Attention：解决 context bottleneck

== 6.1 从固定 context 到动态 context

Vanilla encoder-decoder 使用固定 context：

$ y_t=g_V(y_{t-1},h_{t-1},c) $

Attention 改成每个时间步都有不同 context：

$ y_t=g_V(y_{t-1},h_{t-1},c_t) $

核心思想：decoder 在生成每个词时，动态选择输入中最相关的部分。

例如生成 “person” 时看图像中的人；生成 “hat” 时看帽子区域。

---

== 6.2 图像 captioning 中的 attention

CNN 提取 spatial features：

$ x in R R^{H times W times D} $

其中：

$ H,W $

是空间尺寸；

$ D $

是每个空间位置的 feature dimension；

$ z_{i,j}in R R^{D} $

是位置 ((i,j)) 的视觉特征。

对 decoder 第 (t) 步，计算 alignment score：

$ e_{t,i,j}=f_{"att"}(h_{t-1},z_{i,j}) $

其中：

$ h_{t-1} $

是 decoder 上一步 hidden state；

$ f_{"att"} $

通常是 MLP；

$ e_{t,i,j} $

表示当前生成步骤和图像位置 ((i,j)) 的匹配程度。

用 softmax 归一化：

$ a_{t,i,j}= frac{exp(e_{t,i,j})} {sum_{i'=1}^{H}sum_{j'=1}^{W}exp(e_{t,i',j'})} $

满足：

$ 0<a_{t,i,j}<1  sum_{i=1}^{H}sum_{j=1}^{W}a_{t,i,j}=1 $

context vector：

$ c_t=sum_{i=1}^{H}sum_{j=1}^{W}a_{t,i,j}z_{i,j} $

直觉：(c_t) 是所有 spatial feature 的加权平均。权重越大，说明该位置对当前生成 token 越重要。

---

== 6.3 Soft attention vs hard attention

Soft attention：

$ c_t=sum_i a_{t,i}z_i $

是连续可微的，可以 end-to-end backpropagation。

Hard attention 是离散选择某一个区域：

$ i_t ~ "Categorical"(a_t) $

然后：

$ c_t=z_{i_t} $

由于采样不可微，通常需要 reinforcement learning / policy gradient。

课件重点是 soft attention，因为它端到端可训练，不需要 attention supervision。

---

= 7. NLP 中的 Attention：机器翻译

输入序列：

$ x=x_1,x_2,dot(s),x_T $

输出序列：

$ y=y_1,y_2,dot(s),y_{T'} $

Encoder 产生每个输入位置的 hidden state：

$ z_i=R N N(x_i,u_{i-1}) $

Decoder 第 (t) 步计算对每个 source token 的 alignment：

$ e_{t,i}=f_{"att"}(h_{t-1},z_i) $

归一化：

$ a_{t,i}= frac{exp(e_{t,i})} {sum_{i'=1}^{T}exp(e_{t,i'})} $

context：

$ c_t=sum_{i=1}^{T}a_{t,i}z_i $

decoder 输出：

$ y_t=g_V(y_{t-1},h_{t-1},c_t) $

直觉：翻译时不同 target word 对应不同 source word。Attention 可以自动学 alignment，而且不需要人工标注对齐关系。

---

= 8. General Attention Layer：从任务技巧到通用神经网络层

== 8.1 把图像 attention 抽象成向量集合 attention

输入向量：

$ x={x_1,x_2,dot(s),x_N},quad x_i in R R^{D} $

query：

$ h in R R^{D} $

alignment：

$ e_i=f_{"att"}(h,x_i) $

attention weights：

$ a_i=frac(exp(e_i), sum_{j=1}^{N}exp(e_j)) $

output context：

$ c=sum_{i=1}^{N}a_i x_i $

注意：这个操作本身是 permutation invariant 的。也就是说，如果不加入位置信息，它不关心输入向量的顺序。

---

== 8.2 Dot-product attention

把 MLP alignment 换成点积：

$ e_i=h^->p x_i $

进一步使用 scaled dot prod：

$ e_i=frac(h^->p x_i, sqrt{D}) $

为什么要除以 $sqrt{D}$？

如果 (h) 和 (x_i) 的各维度近似独立且方差为 1，那么点积：

$ h^->p x_i=sum_{d=1}^{D}h_d x_{i,d} $

方差大约与 (D) 成正比。维度越大，logits 越大，softmax 越容易饱和，梯度变小。因此除以 $sqrt{D}$ 稳定数值尺度。

---

== 8.3 多个 query

如果有多个 query：

$ q_1,q_2,dot(s),q_M $

则每个 query 产生一个 output：

$ e_{i,j}=frac(q_j^->p x_i, sqrt{D})  a_{i,j}=frac(exp(e_{i,j}), sum_{i'=1}^{N}exp(e_{i',j}))  y_j=sum_{i=1}^{N}a_{i,j}x_i $

其中：

$ j $

索引 query/output position；

$ i $

索引 input position。

---

= 9. Key-Value-Query：Attention 的核心抽象

课件指出：原始 input vector 同时用于 alignment 和 weighted sum，表达能力有限。于是引入不同的线性投影：

$ k_i=x_i W_k  v_i=x_i W_v $

如果 query 也是从输入得到：

$ q_j=x_j W_q $

其中：

$ W_k $

是 key projection matrix；

$ W_v $

是 value projection matrix；

$ W_q $

是 query projection matrix。

Attention 变成：

$ e_{i,j}=frac(q_j^->p k_i, sqrt{D_k})  a_{i,j}= frac{exp(e_{i,j})} {sum_{i'=1}^{N}exp(e_{i',j})}  y_j=sum_{i=1}^{N}a_{i,j}v_i $

矩阵形式：

$ Q=X W_Q,quad K=X W_K,quad V=X W_V  "Attention"(Q,K,V)="softmax"(frac(Q K^->p, sqrt{D_k}))V $

符号解释：

$ X in R R^{N times D} $

是输入序列；

$ Q in R R^{M times D_k} $

是 queries；

$ K in R R^{N times D_k} $

是 keys；

$ V in R R^{N times D_v} $

是 values；

$ Q K^->p in R R^{M times N} $

是所有 query 和 key 的 pairwise alignment score。

直觉：

query：我现在想找什么？
key：每个输入位置有什么可被匹配的特征？
value：真正被加权读取的信息内容。

---

= 10. Self-Attention

== 10.1 定义

Self-attention 的特点是：query、key、value 都来自同一组输入 (X)。

$ Q=X W_Q  K=X W_K  V=X W_V  Y="softmax"(frac(Q K^->p, sqrt{D_k}))V $

它让每个位置 (j) 都能 attend 到所有位置 (i)。

---

== 10.2 Self-attention 的优势

RNN 的信息传递路径是 sequential：

$ x_1-> h_1-> h_2-> .. -> h_T $

远距离信息必须经过很多步。

Self-attention 中任意两个位置直接交互：

$ x_i <-> x_j $

路径长度为 1。

这使它更适合长距离依赖，并且所有 pairwise attention 可以并行计算。

---

== 10.3 Self-attention 的问题：没有顺序

Self-attention 本身是 permutation invariant / permutation equivariant。也就是说，输入顺序改变，输出也只是相应重排，模型本身不知道哪个 token 在前哪个在后。

因此语言或图像空间特征必须加入 positional encoding。

---

= 11. Positional Encoding

== 11.1 为什么需要位置编码？

输入 token embedding：

$ x_j $

加上或拼接 positional encoding：

$ p_j="p"(j) $

得到：

$ tilde(x)_j=x_j+p_j $

或课件中说的 concatenate：

$ tilde(x)_j=[x_j;p_j] $

position function：

$ "p":N N->R R^{d} $

设计目标：

第一，每个位置要有唯一编码。
第二，任意两个位置之间的距离关系应具有一致性。
第三，能泛化到比训练时更长的序列。
第四，值应有界且 deterministic。

---

== 11.2 Learned positional embedding

一种方法是学习 lookup table：

$ P in R R^{T_{{"max"}}times d} $

第 (t) 个位置编码：

$ p_t=P[t] $

优点：简单，灵活。
缺点：不能自然泛化到超过 $T_{{"max"}}$ 的长度；位置之间的结构关系需要模型自己学。

---

== 11.3 Sinusoidal positional encoding

Transformer 原论文使用固定 sin/cos 编码：

$ P E_{("p",2i)} = sin(frac("p", 10000^{2i/d_{"model"}})) quad P E_{("p",2i+1)} = cos(frac("p", 10000^{2i/d_{"model"}})) $

其中：

$ p $

是 token 位置；

$ i $

是维度 index；

$ d_{"model"} $

是 embedding dimension；

$ 2i $

表示偶数维；

$ 2i+1 $

表示奇数维。

直觉：不同维度对应不同频率的正弦/余弦函数。低维或高频维度捕捉局部位置变化，高维或低频维度捕捉长距离变化。由于 sin/cos 是确定函数，因此可以 extrapolate 到更长位置。

---

= 12. Masked Self-Attention

普通 self-attention 允许每个位置看到所有位置。但语言生成是 autoregressive：

$ p(y_1,dot(s),y_T)=product_{t=1}^{T}p(y_t mid y_{<t}) $

因此生成第 (t) 个 token 时不能看未来 token。

Masked self-attention 通过把未来位置的 alignment score 设置为：

$ -oo $

实现遮蔽：

$ e_{i,j}= cases(frac(q_j^->p k_i, sqrt{D_k}),, i<= j\ -oo,, i>j) $

softmax 后：

$ exp(-oo)=0 $

所以未来 token 的 attention weight 为 0。

这是 Transformer decoder 中最关键的机制之一。

---

= 13. Multi-Head Attention

== 13.1 定义

Multi-"head" attention 并行运行多个 attention "head"：

$ "head"_r = "Attention"(Q W_Q^{(r)}, K W_K^{(r)}, V W_V^{(r)}) $

然后 concatenate：

$ "MultiHead"(Q,K,V)="Concat"("head"_1,dot(s),"head"_H)W_O $

其中：

$ H $

是 "head" 数量；

$ W_O $

是输出投影矩阵。

---

== 13.2 为什么需要多头？

单个 attention "head" 只能在一个表示子空间里计算相关性。多个 "head" 可以学习不同关系：

一个 "head" 关注局部邻近词；
一个 "head" 关注主谓关系；
一个 "head" 关注长距离依赖；
一个 "head" 关注视觉区域之间的空间关系。

直觉：multi-"head" attention 是多个“不同视角”的 attention ensemble。

---

= 14. RNN vs Transformer

| 维度 | RNN | Transformer |
| ------ | ---------------------- | ---------------------------------- |
| 序列建模方式 | hidden state 递推 | pairwise attention |
| 并行性 | 差，必须按时间步计算 | 强，所有位置可并行 |
| 长距离依赖 | 路径长，容易衰减 | 任意位置直接交互 |
| 顺序信息 | 天然有顺序 | 需要 positional encoding |
| 内存开销 | 相对低 | attention matrix 是 (N times M)，内存高 |
| 适合任务 | 小模型、序列递推 | 大规模 NLP/CV、多模态 |
| 主要瓶颈 | sequential computation | quadratic memory |

核心结论：Transformer 用更高 memory 换取更强并行性和更短依赖路径。

---

= 15. Transformer Encoder

== 15.1 Encoder 输入

输入一组向量：

$ X=[x_1,dot(s),x_N] $

加入 positional encoding：

$ tilde(X)=X+P $

然后送入 (N) 个 encoder blocks。Vaswani et al. 原始 Transformer 中：

$ N=6,quad d_{"model"}=512 $

---

== 15.2 Encoder block 结构

一个 encoder block 包含：

```text
Input
 ↓
Add positional encoding
 ↓
Multi-"head" self-attention
 ↓
Residual connection
 ↓
LayerNorm
 ↓
MLP / Feed-forward network
 ↓
Residual connection
 ↓
LayerNorm
Output
```

数学形式：

$ Z="LayerNorm"(X+"MultiHeadSelfAttention"(X))  Y="LayerNorm"(Z+"M L P"(Z)) $

其中：

$ X $

是 block 输入；

$ Z $

是 attention sublayer 输出；

$ Y $

是 block 输出。

---

== 15.3 Encoder 中各组件的作用

Self-attention：唯一负责不同 token/vector 之间交互的模块。

MLP：对每个位置独立做非线性变换：

$ "M L P"(z_i)=W_2sigma(W_1z_i+b_1)+b_2 $

LayerNorm：对每个 token 的 feature dimension 做归一化。

BatchNorm vs LayerNorm：

| 方法 | 归一化维度 | 适合场景 |
| --------- | ------------------------------ | ------------------ |
| BatchNorm | across batch，同一 feature 在不同样本间 | CNN 常见 |
| LayerNorm | across features，同一样本内部 | RNN/Transformer 常见 |

Residual connection：缓解深层网络训练困难，使梯度更容易传播。

---

= 16. Transformer Decoder

== 16.1 Decoder 输入输出

Decoder 输入是已经生成的 target prefix：

$ y_0,y_1,dot(s),y_{t-1} $

输出是下一个 token 的 logits。

整体形式：

$ y_t=T_D(y_{0:t-1},c) $

其中：

$ c $

是 encoder 输出的 context vectors。

---

== 16.2 Decoder block 结构

一个 decoder block 包含：

```text
Input target embeddings + positional encoding
 ↓
Masked multi-"head" self-attention
 ↓
Residual + LayerNorm
 ↓
Cross-attention over encoder outputs
 ↓
Residual + LayerNorm
 ↓
MLP
 ↓
Residual + LayerNorm
Output
```

数学形式：

$ Z_1="LayerNorm"(X+"MaskedM H A"(X))  Z_2="LayerNorm"(Z_1+"CrossAttention"(Q=Z_1,K=C,V=C))  Y="LayerNorm"(Z_2+"M L P"(Z_2)) $

其中：

$ C $

是 encoder outputs；

$ Q=Z_1 $

来自 decoder 当前状态；

$ K,V=C $

来自 encoder 输出。

重点：decoder 的第二个 attention 不是 self-attention，而是 encoder-decoder attention / cross-attention。

---

= 17. Image Captioning using Transformer

图像 captioning 可以用 CNN + Transformer：

第一步，CNN 提取 spatial features：

$  in  R R^{H times W times D} $

展平成序列：

$ z_1,z_2,dot(s),z_{H W} $

第二步，Transformer encoder 编码图像特征：

$ C=T_W(z) $

第三步，Transformer decoder 根据前文 token 和图像 context 生成 caption：

$ y_t=T_D(y_{0:t-1},C) $

这对应课件中的 image captioning using transformers。

进一步问题：是否还需要 CNN？

Vision Transformer 的思想是把图像切成 patch，把每个 patch 当成一个 token：

$ "image"-> "patch sequence"-> "Transformer encoder" $

因此可能完全用 Transformer 从 pixels 到 language。

---

= 18. 本章方法论演进总结

| 阶段 | 方法 | 解决什么 | 失败点 | 下一步如何修复 |
| -- | --------------------- | ---------------------------- | ---------------------------- | ------------------------ |
| 1 | Vanilla NN/CNN | 固定输入输出任务 | 不适合 variable-length sequence | RNN 引入 hidden state |
| 2 | RNN | 序列递推建模 | 长距离依赖弱，计算不能并行 | Seq2Seq 组织输入输出 |
| 3 | Seq2Seq | sequence-to-sequence | fixed context bottleneck | Attention 每步动态读输入 |
| 4 | RNN + Attention | 缓解 bottleneck，可学习 alignment | RNN decoder 仍 sequential | General attention 抽象化 |
| 5 | Self-attention | 任意位置直接交互，可并行 | 无顺序信息 | Positional encoding |
| 6 | Masked self-attention | 支持 autoregressive generation | 单头表达力有限 | Multi-"head" attention |
| 7 | Transformer | 高并行、长依赖强 | (O(N^2)) memory 高 | 更大模型、更高效 attention、ViT 等 |

---

= 19. 期末高频考点速记

最重要公式：

$ h_t=f_W(h_{t-1},x_t)  c_t=sum_i a_{t,i}z_i  a_{t,i}=frac(exp(e_{t,i}), sum_j exp(e_{t,j}))  e_{t,i}=f_{"att"}(h_{t-1},z_i)  Q=X W_Q,quad K=X W_K,quad V=X W_V  "Attention"(Q,K,V)="softmax"(frac(Q K^->p, sqrt{D_k}))V  P E_{(p,2i)}=sin(frac(p, 10000^{2i/d_{"model"}}))  P E_{(p,2i+1)}=cos(frac(p, 10000^{2i/d_{"model"}}))  p(y_1,dot(s),y_T)=product_{t=1}^{T}p(y_t mid y_{<t}) $

最重要概念：

RNN 的参数在所有 time steps 共享。
Seq2Seq 的瓶颈是 fixed-length context vector。
Attention 的本质是根据 query 对 input values 做加权求和。
Self-attention 的 Q/K/V 来自同一输入。
Transformer 需要 positional encoding，因为 self-attention 本身不含顺序。
Decoder 必须用 mask 防止看到未来 token。
Encoder self-attention 看全部输入；decoder masked self-attention 只能看过去；decoder cross-attention 看 encoder outputs。
