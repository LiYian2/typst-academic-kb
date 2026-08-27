// @title: Lecture 8: Large Language Models 学习笔记
// @description:
// @order: 999

= Lecture 8: Large Language Models 学习笔记

== 0. 本章总目标与进化树

本章要回答的问题是：为什么现代 LLM 不再只是一个“预测下一个词”的语言模型，而可以作为通用任务求解器使用？

核心进化链如下：

```text
Count-based LM
 ↓ 解决稀疏性
Feed-forward Neural LM
 ↓ 解决固定短上下文
RNN / Deep RNN
 ↓ 缓解长期依赖困难
LSTM
 ↓ 解决串行计算、并行性差、长距离建模瓶颈
Transformer
 ↓ 通过规模化训练出现通用能力
Large Language Models
 ↓ 通过 instruction tuning 学会听指令
Instruction-tuned LLM
 ↓ 通过 RLHF / preference learning 对齐人类偏好
Aligned Chat LLM
 ↓ 接入图像、语音、工具、agent
Multimodal / Agentic LLM
```

重要程度标注：

| 模块 | 重要程度 | 考试价值 |
| ------------------------------------------------ | ----: | ---------- |
| LM 从 n-gram 到 Transformer 的演进 | ★★★★★ | 高频概念题、简答题 |
| Scaling law / emergence | ★★★★★ | 高频解释题 |
| Encoder-only / Decoder-only / Encoder-decoder 区别 | ★★★★★ | 高频选择、问答 |
| Pre-training / SFT / RLHF | ★★★★★ | 高频大题 |
| Prompting / ICL / CoT | ★★★★☆ | 高频概念与应用题 |
| LoRA | ★★★★☆ | 公式题、参数高效微调 |
| Multimodal LLM | ★★★☆☆ | 概念题 |
| Safety / hallucination / robustness | ★★★☆☆ | 开放题 |

---

= 1. The Path to LLMs：语言模型的历史演进

== 1.1 Count-based Models：计数语言模型

最早的语言模型使用 n-gram 统计条件概率：

$ P(x_1, x_2, dot(s), x_T)
= product_(t=1)^(T) P(x_t mid x_1, dot(s), x_{t-1})
approx product_(t=1)^(T) P(x_t mid x_{t-n+1}, dot(s), x_{t-1}) $

其中，$x_t$ 是第 $t$ 个 token，$T$ 是序列长度，$n$ 是 n-gram 的窗口大小。n-gram 的核心假设是 Markov approximation：当前词只依赖前面有限的 $n-1$ 个词。

它的问题是稀疏性。随着 $n$ 增大，可能的上下文组合数指数增长，很多上下文在训练集中根本没出现过，导致概率估计不稳定。课件中给出 perplexity 的下降路径：unigram 约 300，3-gram 约 150，5-gram 约 141.2，说明更长上下文能提升性能，但计数方法很快遇到稀疏性瓶颈。

Perplexity 定义为：

$ upright("PPL")
= exp(-frac(1, T) sum_(t=1)^(T) log P(x_t mid x_{"<"t})) $

或者若使用以 2 为底的 log：

$ upright("PPL")
= 2^{-frac(1, T) sum_(t=1)^(T) log_2 P(x_t mid x_{"<"t})} $

这里 $x_{"<"t} = (x_1, dot(s), x_{t-1})$ 。P P L 越低，说明模型给真实 token 分配的概率越高。直觉上，P P L 可以理解为模型在每一步“平均还在多少个候选词之间犹豫”。

== 1.2 Feed-forward Neural L M：神经语言模型

为了解决 n-gram 的稀疏性，神经语言模型引入连续词向量，把离散 token 映射到 embedding 空间： $
e_t = E[x_t]
$ 其中，$E in RR^{|V| times d}$ 是词嵌入矩阵，$|V|$ 是词表大小，$d$ 是 embedding 维度。神经 L M 用连续表示学习相似词之间的共享结构，例如 “cat” 和 “dog” 在 embedding 空间接近，因此可以缓解未见 n-gram 的问题。

典型前馈语言模型可写为： $
h_t = f(W[e_{t-n+1};dot(s);e_{t-1}] + b)
P(x_t mid x_{t-n+1:t-1})
= upright("softmax")(U h_t+c)
$ 其中 $[,;,]$ 表示向量拼接，$W,U$ 是可训练权重，$b,c$ 是偏置，$f$ 是非线性激活函数。

它解决了稀疏性，但仍然有固定窗口问题：上下文长度固定，不能自然建模长距离依赖。

== 1.3 R N N / Deep R N N：序列记忆

R N N 的关键改进是把历史信息压缩进 hidden state： $
h_t = f(W_x x_t + W_h h_{t-1} + b)
P(x_{t+1} mid x_{\<= t}) = upright("softmax")(U h_t+c)
$ 其中 $h_t$ 是第 $t$ 步的隐藏状态，理论上包含从 $x_1$ 到 $x_t$ 的历史信息。相比前馈模型，R N N 不再受固定窗口限制。

但 R N N 的问题是梯度消失/爆炸。反向传播时梯度需要连续乘很多个 Jacobian： $
frac(partial L, partial h_t)
=

frac(partial L, partial h_T)
product_(k=t+1)^(T)
frac(partial h_k, partial h_{k-1})
$ 如果这些矩阵的谱半径长期小于 1，梯度趋近 0；如果长期大于 1，梯度爆炸。因此 R N N 虽然理论上能处理长上下文，实际上难以捕捉很长距离依赖。

== 1.4 L S T M：门控记忆

L S T M 引入 cell state 和门控机制，让信息可以更稳定地跨时间传播： $
f_t = sigma(W_f [h_{t-1},x_t]+b_f)
i_t = sigma(W_i [h_{t-1},x_t]+b_i)
tilde(c)_t = tanh(W_c [h_{t-1},x_t]+b_c)
c_t = f_t op("odot") c_{t-1} + i_t op("odot") tilde(c)_t
o_t = sigma(W_o [h_{t-1},x_t]+b_o)
h_t = o_t op("odot") tanh(c_t)
$ 其中 $f_t$ 是 forget gate，决定保留多少旧记忆；$i_t$ 是 input gate，决定写入多少新信息；$o_t$ 是 output gate，决定输出多少 cell state 信息；$op("odot")$ 是逐元素乘法。

L S T M 缓解了 vanishing gradient，但仍有两个问题：第一，计算是时间步串行的，难以并行；第二，所有历史压缩到 hidden state，信息瓶颈仍然存在。

== 1.5 Transformer：并行注意力与长距离依赖

Transformer 用 self-attention 替代递归结构。核心公式： $
Q = X W_Q,quad K = X W_K,quad V = X W_V

upright("Attention")(Q,K,V)
=

upright("softmax")(frac(Q K^->p, sqrt{d_k})) V
$ 其中 $X in RR^{T times d}$ 是输入 token 表示，$Q,K,V$ 分别是 query、key、value，$d_k$ 是 key/query 的维度。$Q K^->p$ 计算 token 之间的相关性，除以 $sqrt{d_k}$ 是为了防止点积随维度增大而数值过大，softmax 后得到注意力权重。

Transformer 的关键优势是：任意两个 token 可以直接交互，路径长度为 1；训练时所有 token 可以并行计算；不再需要把整个历史压缩到单个 hidden state。

这就是课件中从 L S T M 到 Transformer 的核心转折：L S T M 解决长期依赖，但仍有瓶颈和串行计算；Transformer 用 attention 同时解决信息交互和并行训练。

---

= 2. Paradigm Shift：从任务专用模型到通用语言模型

== 2.1 Pipelined N L P

早期 N L P 通常把 language model 当作非链式模型的一部分。例如机器翻译可以拆成若干模块：词对齐、翻译概率、语言模型、解码器。这种 pipeline 的问题是各模块分离，误差传播严重，且每个任务都要设计专门系统。

== 2.2 Seq2Seq Learning

Seq2Seq 把任务统一为条件序列建模： $
P(Y=y mid X=x)
=

product_(t=1)^(T_y)
P(y_t mid y_{"<"t}, x)
$ 其中 $X=x$ 是输入序列，$Y=y$ 是输出序列，$y_t$ 是第 $t$ 个输出 token。机器翻译、摘要、问答都可以写成输入序列到输出序列的条件生成。

它解决了任务建模统一性，但仍然通常需要任务级训练数据。

== 2.3 Pretrain + Transfer

B E R T 代表的范式是：先在大规模无标注语料上预训练语言表示，再 fine-tune 到下游任务。

B E R T 的 masked language modeling objective 是： $
cal(L)_("MLM")
=

-sum_(i in M)
log P_theta(x_i mid x_{∖ M})
$ 其中 $M$ 是被 mask 的 token 位置集合，$x_i$ 是原始 token，$x_{∖ M}$ 是 mask 后的上下文。注意 loss 只对被选中预测的位置计算，而不是所有 token。

B E R T 的 fine-tuning 分类头可以写成： $
P(y mid x)=upright("softmax")(W h_("[CLS]") + b)
$ 其中 $h_("[CLS]")$ 是 [C L S] token 的最终隐藏状态，$W,b$ 是下游任务分类头参数。课件强调 B E R T 偏 representation，而 G P T 偏 generation。

== 2.4 Pretrain + Prompt

G P T-2 / G P T-3 进一步改变范式：不再为每个任务单独 fine-tune，而是把任务描述也作为文本输入，让模型学习： $
P(upright("output") mid upright("input"), upright("task"))
$ 这比单任务模型的 $
P(upright("output") mid upright("input"))
$ 更通用。任务本身可以通过自然语言 prompt 指定，例如 “Translate this French sentence into English”。

这一步的本质是：任务规范从模型参数外部的 label schema，转变为上下文中的自然语言 token。

---

= 3. Scaling Laws 与 Emergent Abilities

== 3.1 Scaling Law

Scaling law 描述模型 loss 与模型规模、数据规模、计算量之间的幂律关系。典型形式可以写为： $
L(N) = L_oo + A N^{-alpha}
$ 其中 $L(N)$ 是数据量或模型参数量为 $N$ 时的 loss，$L_oo$ 是不可约误差，$A$ 是常数，$alpha>0$ 是 scaling exponent。

取 log 后： $
log(L(N)-L_oo)
=

log A - alpha log N
$ 所以在 log-log 图上近似为直线。课件强调：loss 和 dataset size 在 log-log plot 上呈线性关系，这就是 power-law scaling。

考试直觉：如果性能随规模平滑改善，那么可以先训练小模型估计趋势，再预测大模型表现，用于选择架构、数据量和超参数。

== 3.2 Scaling 的经验结论

课件总结 Kaplan et al. 2020 的关键结论：

| 结论 | 含义 | 重要程度 |
| --------------------------------------- | --------------------- | ----: |
| Performance strongly depends on scale | 参数量、数据量、计算量越大，loss 越低 | ★★★★★ |
| Weakly depends on model shape | 在一定范围内，宽度/深度细节不如规模重要 | ★★★★☆ |
| Larger models are more sample-efficient | 大模型用较少样本也能学得更好 | ★★★★★ |
| Smooth power laws | 小模型趋势可外推到大模型 | ★★★★★ |

这里容易考的点是：scaling law 不是说架构不重要，而是说在固定 Transformer family 内，性能更强烈地受 scale 影响。

== 3.3 Emergent Abilities

Emergent abilities 指小模型没有、大模型突然出现的能力。课件例子包括 G P T-3 可以做加减法，不同能力在不同规模出现。

但 emergence 不只由参数量决定。课件指出：在一些 B I G-Bench 任务上，LaM D A 137B 和 G P T-3 175B 接近随机，而 PaL M 62B 却高于随机。这说明数据、训练方法、架构、tokenizer、优化策略都会影响 emergence。

考试表述可以写成：

```text
Emergence = scale-driven but not scale-only.
```

---

= 4. Modern L L M Architectures

== 4.1 三类 Transformer 架构

| 架构 | 代表 | 预训练目标 | 注意力模式 | 优势 | 局限 |
| --------------- | ----------- | ---------------------- | ------------------------- | -------------- | ---------------------------- |
| Encoder-only | B E R T | M L M | 双向 attention | 表示学习、分类、理解任务强 | 不适合自由生成 |
| Decoder-only | G P T / Llama | Autoregressive L M | causal attention | 生成自然、训练稳定、通用性强 | 双向理解需通过 prompt 间接实现 |
| Encoder-decoder | T5 / T0 | Masked span prediction | encoder 双向，decoder causal | 翻译、摘要、条件生成强 | 现代通用 L L M 中不如 decoder-only 主流 |

== 4.2 Decoder-only Auto-regressive L M

Decoder-only L L M 的核心训练目标是 next-token prediction： $
P_theta(x_1,dot(s),x_T)
=

product_(t=1)^(T)
P_theta(x_t mid x_{"<"t})
$ 训练 loss： $
cal(L)_("AR")
=

-sum_(t=1)^(T)
log P_theta(x_t mid x_{"<"t})
$ 其中 $x_{"<"t}$ 表示第 $t$ 个 token 之前的全部 token。由于 causal mask，位置 $t$ 只能 attend 到 $1,dot(s),t$，不能看未来 token。

Causal attention mask 可写为： $
M_{"ij"} =
cases(0,, j <= i,
-oo,, j > i)

upright("Attention")(Q,K,V)
=

upright("softmax")(
frac(Q K^->p, sqrt{d_k}) + M
) V
$ 这里 $M_{"ij"}$ 控制第 $i$ 个 token 是否能看第 $j$ 个 token。若 $j>i$，mask 为 $-oo$，softmax 后权重为 0。

== 4.3 T5 / T0 Masked Span Prediction

T5 的目标不是 mask 单个 token，而是 mask 连续 span，并由 decoder 生成被 mask 的 span： $
cal(L)_("span")
=

-sum_(t=1)^{T_y}
log P_theta(y_t mid y_{"<"t}, tilde(x))
$ 其中 $tilde(x)$ 是被 mask span 替换后的输入，$y$ 是需要恢复的 span 序列。它适合条件生成任务，例如翻译和摘要。

== 4.4 Llama Architecture：现代 Decoder-only 改造

课件列出 Llama 相比 vanilla Transformer 的关键变化：Pre-norm、R M S Norm、SwiG L U、RoP E、Grouped-query attention。

=== Pre-norm

Post-norm Transformer： $
x_{l+1} = upright("LN")(x_l + F(x_l))
$ Pre-norm Transformer： $
x_{l+1} = x_l + F upright("LN")(x_l)
$ Pre-norm 的优势是梯度更容易通过 residual path 传播，深层模型训练更稳定。

=== R M S Norm

LayerNorm： $
upright("LN")(x)
=

frac(x-mu, sqrt{sigma^2+epsilon}) op("odot") gamma + beta
$ 其中 $
mu = frac(1, d) sum_(i=1)^(d)x_i,quad
sigma^2 = frac(1, d) sum_(i=1)^(d)(x_i-mu)^2
$ R M S Norm 去掉均值中心化： $
upright("RMSNorm")(x)
=

frac(x, sqrt{frac(1, d) sum_(i=1)^(d) x_i^2 + epsilon})
op("odot") gamma
$ 它更简单、更省计算，同时保留尺度归一化效果。

=== SwiG L U

课件给出的 SwiG L U 形式是： $
upright("SwiGLU")(x)
=

upright("Swish")(x W) op("odot") x V
$ 其中 $
upright("Swish")(z)=z dot sigma(z)
$ 更完整可写为： $
upright("SwiGLU")(x)
=

x W_1 op("odot") sigma(x W_1) op("odot") (x W_2)
$ 它比 ReL U 更平滑，并且通过 gate 控制信息流。

=== RoP E

RoP E，即 Rotary Position Embedding，通过旋转 query/key 向量编码相对位置信息。二维情况下： $
R_(theta,m)
=

mat(cos(m theta), -sin(m theta),
sin(m theta), cos(m theta))

q_m' = R_(theta,m)q_m,quad k_n'=R_{theta,n}k_n
$ 内积变为： $
(q_m')^->p k_n'
=

q_m^->p R_{theta,n-m}k_n
$ 所以 attention score 自然依赖相对距离 $n-m$。这就是 RoP E 适合长上下文扩展的原因。

=== Grouped-query Attention

标准 multi-"head" attention 每个 query "head" 都有自己的 key/value "head"。Grouped-query attention 让多个 query heads 共享同一组 key/value heads，从而降低 K V cache 成本。

如果 query heads 数为 $H_q$，key/value heads 数为 $H_{"kv"}$，且 $H_{"kv"} < H_q$，则推理时 K V cache 大约减少为： $
frac(H_{"kv"}, H_q)
$ 这对长上下文推理很重要。

---

= 5. L L M Training：Pre-training → S F T → R L H F

== 5.1 Auto-regressive Pre-training

Decoder-only L L M 首先在大规模语料上做 next-token prediction： $
cal(L)_{"pretrain"}
=

-bb(E)_(x ~ cal(D))
sum_(t=1)^(T)
log P_theta(x_t mid x_{"<"t}) $ 其中 $cal(D)$ 是大规模文本语料。课件中 Llama 约使用 1T tokens，Llama 2 约使用 3T tokens。

预训练的作用不是让模型直接“听话”，而是让模型成为强大的 generalist autocomplete：它学会语言规律、世界知识、代码、推理模式和各种任务格式。

== 5.2 Standard Multi-task Learning vs Fine-tuning vs Instruction Tuning

三者区别如下：

| 方法 | 训练方式 | 优点 | 问题 |
| -------------------- | ---------------------------- | ------------ | ------------ |
| Multi-task learning | 多任务 minibatch 混合训练 | 共享表示 | 任务格式不统一 |
| Pretrain + Fine-tune | 先预训练，再单任务微调 | 单任务性能强 | 每个任务都要一个模型副本 |
| Instruction tuning | 在大量 instruction-output 数据上微调 | 学会遵循自然语言任务描述 | 仍未完全对齐人类偏好 |

Instruction tuning 的目标是激活模型的 instruction-following mode，使其泛化到新任务。课件列出数据来源包括 F L A N、Natural Instructions、Self-Instruct、Alpaca、Vicuna、WizardL M/Evol-Instruct 等。

== 5.3 S F T 与 Pre-training 的区别

课件明确强调：pre-training 的 target 是所有 token；S F T 的 loss 通常只在目标回答 token 上计算。

Pre-training： $
cal(L)_{"pretrain"}
=

-sum_(t=1)^(T) log P_theta(x_t mid x_{"<"t}) $ SFT 数据为 $(u,y)$，其中 $u$ 是 instruction + input，$y$ 是 target response。S F T loss： $
cal(L)_{"SFT"}
=

-sum_(t=1)^{T_y}
log P_theta(y_t mid u, y_{"<"t}) $ 这里 $u$ 中的 instruction token 通常不参与 loss，只作为条件上下文。直觉上，pre-training 学“怎么续写文本”，S F T 学“面对指令应该输出什么形式的答案”。

== 5.4 为什么 S F T 不够？

S F T 只能模仿训练数据中的答案格式，但不能保证输出符合人类偏好。课件指出两个核心问题：我们不知道模型 internal knowledge 里有什么；我们希望模型输出不要有害或被恶意使用。因此需要 alignment，使模型更 helpful, honest, harmless。

== 5.5 R L H F

R L H F 的典型流程：

```text
Pretrained L M
 ↓
S F T model
 ↓ 收集人类偏好比较
Reward Model
 ↓ 用 P P O / policy gradient 优化
Aligned L L M
```

Reward model 通常根据两个回答 $y^+$ 和 $y^-$ 的偏好训练： $
cal(L)_{"RM"}
=

-log sigma(r_phi(x,y^+) - r_phi(x,y^-))
$ 其中 $r_phi(x,y)$ 是 reward model 对回答 $y$ 的打分，$y^+$ 是人类更偏好的回答，$y^-$ 是较差回答。若 $r_phi(x,y^+)$ 比 $r_phi(x,y^-)$ 大，loss 变小。

R L H F 优化目标常写为： $
max_theta
bb(E)_{y ~ pi_theta(dot mid x)}
[
r_phi(x,y)
-----------

beta upright("KL")
(
pi_theta(dot mid x)
|
pi_{"ref"}
)

$ 其中 $pi_theta$ 是当前策略模型，$pi_{"ref"}$ 是参考模型，通常是 S F T model，$beta$ 控制不要偏离参考模型太远。K L 项的作用是防止 reward hacking 和语言质量崩坏。

注意：R L H F 不保证模型永远安全。课件强调 harmful outputs 不是可穷尽枚举的，并且 adversarial robustness 仍然是问题。

---

= 6. L L M Inference：Prompting, I C L, CoT

== 6.1 Prompting

Prompt 是自然语言任务说明。Prompt engineering 是寻找合适输入格式来诱导模型完成任务。一般原则是具体、明确、描述充分。

数学上，prompting 不更新参数，只改变条件上下文： $
P_theta(y mid p, x) $ 其中 $p$ 是 prompt，$x$ 是输入，$y$ 是输出。模型能力来自预训练参数 $theta$，prompt 只是激活相应行为模式。

== 6.2 In-context Learning

In-context learning 是把 demonstration examples 放进上下文，让模型从示例中推断任务： $
P_theta
(
y_(upright("test")}
mid
(x_1,y_1),dot(s),(x_k,y_k),x_(upright("test"))
)
$ 其中 $k=0$ 是 zero-shot，$k=1$ 是 one-shot，$k>1$ 是 few-shot。关键点：I C L 不做梯度更新，模型参数不变。

| 方法 | 输入 | 是否更新参数 | 优点 | 问题 |
| ----------- | ----------- | ------ | ------ | ------- |
| Zero-shot | 任务描述 + 测试输入 | 否 | 最方便 | 可能歧义 |
| One-shot | 一个示例 + 测试输入 | 否 | 接近人类教学 | 示例选择敏感 |
| Few-shot | 多个示例 + 测试输入 | 否 | 数据需求少 | 示例顺序敏感 |
| Fine-tuning | 训练集 | 是 | 性能强 | 成本高、泛化弱 |

课件特别指出：I C L 中样本顺序可能导致性能从接近 S O T A 到接近随机，且这种现象在大模型中也存在。

== 6.3 Chain-of-Thought Prompting

CoT 的核心思想是：把自然语言推理步骤作为额外输入，让模型先生成 reasoning steps，再输出答案。

形式上： $
P_theta(a mid x)
=

sum_(r)
P_theta(a, r mid x) =

sum_(r)
P_theta(r mid x) P_theta(a mid x, r) $ 其中 $x$ 是题目，$r$ 是 reasoning chain，$a$ 是最终答案。普通 prompting 直接建模 $P(a mid x $；CoT 显式引入中间变量 $r$，把复杂映射拆成“先推理、再回答”。

CoT 的收益在大模型上更明显。课件指出 CoT 能显著提升性能，而且模型越大，CoT benefit 越明显；zero-shot CoT 有时甚至超过 few-shot CoT。

---

= 7. Parameter Efficient Fine-tuning：LoRA

当没有足够数据或算力做 full fine-tuning 时，可以冻结大模型主体，只训练少量参数。课件重点给出 LoRA。

Full fine-tuning 更新整个权重矩阵：

$ W' = W + Delta W $

其中 $W in RR^{d times k}$，完整更新 $Delta W$ 也有 $d k$ 个参数。

LoRA 假设下游任务所需更新是低秩的：

$ Delta W = B A  W' = W + B A $

其中

$ B in RR^{d times r},quad
A in RR^{r times k},quad r << min(d,k) $

原本需要训练 $d k$ 个参数，现在只训练：

$ d r + r k = r(d+k) $

个参数。参数比例为：

$ frac(r (d+k), d k) $

当 $r$ 很小，例如 $r=8$ 或 $16$，训练参数量大幅下降。直觉是：大模型已经有通用能力，下游任务只需要在参数空间中做一个低维方向的调整。

---

= 8. Evaluation 与 Benchmark

LLM evaluation 难点在于任务广泛、输出开放、指标不唯一。课件列出典型任务：

| 任务 | 例子 | 评价重点 |
| ----------------- | ----------------- | ----------------- |
| Context-free QA | MMLU | 世界知识、学科知识 |
| Contextual QA | Natural Questions | 文档理解、检索 grounding |
| Code Generation | HumanEval | 代码正确性、单元测试通过率 |
| Summarization | WikiSum | 信息压缩、覆盖、忠实性 |
| Translation | FLORES | 多语言翻译质量 |
| General Benchmark | BIG-Bench | 综合能力、emergence |

考试要点：LLM evaluation 不能只看单一 benchmark，因为模型可能对某类任务强，对另一类任务弱；也不能只看 perplexity，因为 perplexity 不直接等价于 instruction following、reasoning 或 safety。

---

= 9. Multimodal LLMs

多模态 LLM 的目标是让模型不仅处理文本，还能处理图像、语音等模态。课件指出 multimodality 可以从 pre-training 引入，例如 Gemini；也可以从 instruction-tuning 引入，例如 AudioGPT、Flamingo。

== 9.1 Continuous Representations

连续表示方法把图像或音频编码成 dense embeddings，然后与文本 embedding 拼接送入 LLM：

$ Z_(upright("image")) = f_(upright("vision"))(I)  Z = [Z_(upright("image")); Z_(upright("text"))] $

其中 $f_(upright("vision"))$ 可以是 CLIP-like encoder。优点是信息丰富、性能强；缺点是计算和存储较重。

== 9.2 Discrete Units

离散表示把音频或视觉特征量化成 discrete tokens，例如通过 VQ-VAE 或 clustering：

$ z_t = op("argmin")_(j)
|h_t - c_j|^2 $

其中 $h_t$ 是连续特征，$c_j$ 是第 $j$ 个 codebook center，$z_t$ 是离散 unit id。这样多模态输入可以更像文本 token 一样被 LLM 处理。

课件强调 discrete units 的优势包括：存储更省、序列长度可减少、可以去重、支持 subword-like modeling。

---

= 10. Open Challenges

课件最后总结 LLM 的开放挑战：

```text
Capabilities:
 - More complex reasoning
 - Long context
 - New abilities

Performance:
 - Reduce hallucination
 - Improve alignment
 - Efficiently increase context length
 - Better data, training strategy, architecture

Efficiency:
 - Computational cost
 - Time and money
 - GPU / TPU / HPU architecture

Safety:
 - Reduce harm
 - Improve adversarial robustness
 - Privacy concerns

Interpretability:
 - Why do LLMs behave as they do?
```

其中期末最可能考的是 hallucination、alignment、long context efficiency、adversarial robustness 的区别。Hallucination 是事实性错误；alignment 是输出是否符合人类偏好；long context 是上下文窗口与注意力成本问题；adversarial robustness 是模型是否会被恶意 prompt 攻破。

---

= 11. 期末复习压缩版主线

这章可以用一句话概括：

```text
LLM = Transformer decoder-only architecture
 + large-scale autoregressive pre-training
 + instruction tuning
 + human preference alignment
 + prompting / ICL / CoT inference
 + scaling-driven emergent abilities
```

最核心公式必须会：

$ P_theta(x_1,dot(s),x_T) = product_(t=1)^(T) P_theta(x_t mid x_{"<"t}) $

$ cal(L)_{"pretrain"} = -sum_(t=1)^(T) log P_theta(x_t mid x_{"<"t}) $

$ cal(L)_{"SFT"} = -sum_(t=1)^(T_y) log P_theta(y_t mid u, y_{"<"t}) $

$ upright("Attention")(Q,K,V) = upright("softmax")(frac(Q K^->p, sqrt{d_k})) V $

$ upright("Attention")_{"causal"}(Q,K,V) = upright("softmax")(frac(Q K^->p, sqrt{d_k})+M) V $

$ L(N) = L_oo + A N^{-alpha}  quad  W' = W + Delta W = W + B A $

$ cal(L)_{"RM"} = -log sigma(r_phi(x,y^+) - r_phi(x,y^-)) $

$ max_theta  quad  bb(E)_{y ~ pi_theta} [ r_phi(x,y) - beta upright("KL") (pi_theta | pi_{"ref"}) ] $

最重要的概念链必须会解释：

```text
n-gram fails because of sparsity.
Neural LM fixes sparsity using embeddings.
RNN fixes fixed context but suffers vanishing gradients.
LSTM mitigates vanishing gradients but remains sequential and bottlenecked.
Transformer fixes parallelism and long-range interaction.
Scaling Transformer leads to generalization and emergent abilities.
Instruction tuning turns autocomplete into instruction following.
RLHF aligns instruction-following behavior with human preferences.
Prompting / ICL / CoT exploit capabilities at inference without parameter updates.
LoRA adapts LLMs cheaply by low-rank updates.
Multimodal LLMs extend token modeling beyond text.
```
