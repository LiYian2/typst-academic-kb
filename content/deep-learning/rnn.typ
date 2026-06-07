// @title: Lecture 5 学习笔记：Recurrent Neural Networks
// @description: 
// @order: 999

下面是 Lecture 5: Recurrent Neural Networks 的系统复习笔记，基于课件内容整理。核心主线是：

_序列建模问题 → Vanilla RNN → BPTT 与梯度问题 → LSTM/GRU 门控记忆 → 双向/深层 RNN → Encoder-Decoder → Attention → Transformer → SSM/Mamba_。

---

= Lecture 5 学习笔记：Recurrent Neural Networks

== 0. 本章核心目标

本章要解决的问题是：_如何建模长度可变、具有时间/顺序依赖的序列数据_。

普通 feedforward neural network 的输入输出维度通常固定，不能自然处理：

- 文本句子：长度可变；
- 语音信号：时间序列；
- 机器翻译：输入输出长度不同；
- POS tagging：每个输入 token 对应一个输出 tag；
- sentiment analysis：整段序列压缩成一个类别。

所以本章的技术演进可以概括为：

```text
固定长度输入输出
 ↓
RNN：用 hidden state 递归保存历史信息
 ↓
BPTT：在展开后的时间图上训练
 ↓
梯度消失/爆炸：长程依赖难学
 ↓
LSTM / GRU：用门控机制控制记忆保留与更新
 ↓
BiRNN / Deep BiRNN：引入未来上下文与更深表示
 ↓
Encoder-Decoder：处理输入输出长度不同的 seq2seq
 ↓
Attention：解除固定向量瓶颈，让 decoder 动态“看”输入
 ↓
Transformer：完全用 self-attention 替代 recurrence
 ↓
SSM / Mamba：为长上下文重新引入硬件友好的 recurrent memory
```

重要程度标注：
_★★★★★ 必考核心_；_★★★★ 高频重点_；_★★★ 理解即可_。

---

= 1. 为什么需要 RNN？【★★★★★】

RNN 的根本动机是处理 variable-length sequences。课件指出，RNN 与 feedforward network 的区别在于：RNN 的连接形成 directed cycle，因此可以使用 internal memory 处理任意长度序列。

== 1.1 序列任务类型

RNN 的灵活性来自它可以处理不同输入输出结构：

| 类型 | 形式 | 例子 | 重要程度 |
| ----------------- | ----------- | ------------------- | ----- |
| One-to-one | 单输入 → 单输出 | 普通神经网络分类 | ★★ |
| One-to-many | 单输入 → 序列输出 | Image captioning | ★★★ |
| Many-to-one | 序列输入 → 单输出 | Sentiment analysis | ★★★★★ |
| Many-to-many，同长度 | 序列输入 → 序列输出 | POS tagging | ★★★★★ |
| Many-to-many，不同长度 | 输入序列 → 输出序列 | Machine translation | ★★★★★ |

直觉上，RNN 的核心思想是：

```text
当前输出不仅依赖当前输入 x^(t)，还依赖之前积累的状态 h^(t-1)
```

也就是：

$ h^{(t)} = "function"(h^{(t-1)}, x^{(t)}) $

其中 $h^{(t)}$ 是“到第 (t) 步为止，模型对历史的压缩记忆”。

---

= 2. Vanilla RNN 结构与公式【★★★★★】

== 2.1 基本更新方程

课件中的 Vanilla RNN 更新为：

$ h^{(t)} = tanh (b + W h^{(t-1)} + U x^{(t)} )  o^{(t)} = c + V h^{(t)}  hat(y)^{(t)} = "softmax"(o^{(t)}) $

其中：

- $x^{(t)}$：第 (t) 个输入向量；
- $h^{(t)}$：第 (t) 个 hidden state；
- $h^{(t-1)}$：上一时刻 hidden state；
- $o^{(t)}$：输出层 logits；
- $hat(y)^{(t)}$：模型预测的类别概率分布；
- (U)：input-to-hidden 权重矩阵；
- (W)：hidden-to-hidden recurrent 权重矩阵；
- (V)：hidden-to-output 权重矩阵；
- (b)：hidden bias；
- (c)：output bias；
- $tanh$：非线性激活函数；
- $"softmax"$：把 logits 转成概率分布。

这里最核心的是 $W h^{(t-1)}$。它让模型把过去的信息传到当前，因此 RNN 可以处理序列。

---

== 2.2 概率解释与 Loss【★★★★★】

对于一个输入序列：

$ {x^{(1)}, x^{(2)}, .., x^{(tau)}} $

以及目标序列：

$ {y^{(1)}, y^{(2)}, .., y^{(tau)}} $

课件给出的单个训练样本 loss 是：

$ L( {x^{(1)}, .., x^{(tau)}}, {y^{(1)}, .., y^{(tau)}} ) = -sum_{t=1}^{"tau"} log P( y^{(t)} mid x^{(1)}, .., x^{(t)} ) $

其中：

- (L)：序列级负对数似然损失；
- $tau$：序列长度；
- $P(y^{(t)} mid x^{(1)}, .., x^{(t)})$：模型在看过前 (t) 个输入后，对正确标签 $y^{(t)}$ 的预测概率；
- 这个概率从 $hat(y)^{(t)}$ 中取出对应 $y^{(t)}$ 的 entry。

训练目标是最小化所有训练 pair 的总 loss，优化参数：

$ W, U, V, b, c, theta_{"em"} $

其中 $theta_{"em"}$ 是 embedding vectors。

直觉：
RNN 在每个时间步都做一次预测，因此总 loss 是所有时间步 loss 的求和。

---

= 3. BPTT：Backpropagation Through Time【★★★★★】

RNN 训练的关键是 BPTT。课件强调，RNN 需要计算：

$ nabla_W L,quad nabla_U L,quad nabla_V L,quad nabla_b L,quad nabla_c L,quad nabla_{theta_{"em"}}L $

这些梯度是在展开后的 computational graph 上通过 backpropagation through time 得到的。

== 3.1 为什么叫 Through Time？

RNN 参数在所有时间步共享。展开后形式类似：

```text
x^(1) → h^(1) → o^(1) → L^(1)
 ↓
x^(2) → h^(2) → o^(2) → L^(2)
 ↓
x^(3) → h^(3) → o^(3) → L^(3)
```

同一个 (W, U, V) 在每个时间步重复使用。因此总梯度必须把每个时间步对参数的贡献加起来。

---

== 3.2 Softmax + Cross Entropy 梯度【★★★★★】

定义：

$ hat(y)^{(t)}_i = frac{exp(o^{(t)}_i)} {sum_j exp(o^{(t)}_j)}  L^{(t)} = -log hat(y)^{(t)}_{y^{(t)}} $

则标准结果是：

$ frac(partial L^{(t)}, partial o^{(t)}_i) = hat(y)^{(t)}*i - bold(1)*{i,y^{(t)}} $

其中：

- (i)：类别 index；
- $hat(y)^{(t)}_i$：模型对类别 (i) 的预测概率；
- $bold(1)_{i,y^{(t)}}$：indicator function，如果 $i = y^{(t)}$，则为 1，否则为 0。

因此：

$ (nabla_{o^{(t)}}L)_i = hat(y)^{(t)}*i - bold(1)*{i,y^{(t)}} $

直觉：
预测概率减真实 one-hot 标签。预测过高的错误类别会被压低，正确类别会被拉高。

---

== 3.3 输出层参数梯度【★★★★★】

因为：

$ o^{(t)} = c + V h^{(t)} $

所以：

$ nabla_c L = sum_{t=1}^{T} nabla_{o^{(t)}} L  nabla_V L = sum_{t=1}^{T} (nabla_{o^{(t)}} L) (h^{(t)})^->p $

其中：

- $nabla_c L$：所有时间步 output bias 的梯度相加；
- $nabla_V L$：每个时间步的 output error 与 hidden state 做 outer product 后求和。

---

== 3.4 Hidden State 梯度递推【★★★★★】

课件中指出，$h^{(t)}$ 到总 loss 有两条路径：

```text
路径 1：h^(t) → o^(t) → L^(t)
路径 2：h^(t) → h^(t+1) → h^(t+2) → ... → future losses
```

因此：

$ nabla_{h^{(t)}} L = V^->p nabla_{o^{(t)}} L + W^->p delta^{(t+1)} $

定义 pre-activation：

$ a^{(t)} = b + W h^{(t-1)} + U x^{(t)}  h^{(t)} = tanh(a^{(t)}) $

定义：

$ delta^{(t)} equiv nabla_{a^{(t)}} L $

因为：

$ frac(partial h^{(t)}, partial a^{(t)}) = "diag"(1 - (h^{(t)})^2) $

所以：

$ delta^{(t)} = "diag"(1 - (h^{(t)})^2) nabla_{h^{(t)}} L $

合并得到：

$ delta^{(t)} = "diag"(1 - (h^{(t)})^2) [ V^->p nabla_{o^{(t)}} L + W^->p delta^{(t+1)} ] $

并且边界条件：

$ delta^{(T+1)} = 0 $

直觉：
当前 hidden state 的梯度来自当前输出误差，也来自未来时间步通过 recurrent connection 传回来的误差。

---

== 3.5 (W, U, b) 的梯度【★★★★★】

由：

$ a^{(t)} = b + W h^{(t-1)} + U x^{(t)} $

得到：

$ nabla_b L = sum_{t=1}^{T} delta^{(t)}  nabla_W L = sum_{t=1}^{T} delta^{(t)} (h^{(t-1)})^->p  nabla_U L = sum_{t=1}^{T} delta^{(t)} (x^{(t)})^->p $

其中：

- (W)：控制历史 hidden state 如何影响当前；
- (U)：控制当前输入如何写入 hidden state；
- (b)：每个时间步共享的 hidden bias；
- $delta^{(t)}$：第 (t) 步 hidden pre-activation 的误差信号。

---

= 4. Output Recurrence 与 Teacher Forcing【★★★★】

除了 hidden-to-hidden recurrence，还有一种结构是 output recurrence：上一时刻输出影响下一时刻 hidden/input。课件给出 teacher forcing 的条件似然：

$ log p(y^{(1)}, y^{(2)} mid x^{(1)}, x^{(2)}) = log p(y^{(2)} mid y^{(1)}, x^{(1)}, x^{(2)}) + log p(y^{(1)} mid x^{(1)}, x^{(2)}) $

更一般地：

$ log p(y^{(1:T)} mid x) = sum_{t=1}^{T} log p(y^{(t)} mid y^{( < t)}, x) $

Teacher forcing 的训练方式是：训练时把真实的上一时刻 token $y^{(t-1)}$ 输入给模型，而不是模型自己生成的 $hat(y)^{(t-1)}$。

优点：

- 时间步之间可以部分 decouple；
- 训练更稳定；
- 可以并行化部分计算。

缺点：

- 训练时看到真实历史，测试时看到自己生成的历史；
- 产生 exposure bias；
- 没有 hidden-to-hidden recurrence 时表达能力更弱。

课件给出的折中方案是：随机选择使用 generated values 或 actual values 作为输入，即类似 scheduled sampling。

---

= 5. 梯度消失与梯度爆炸【★★★★★】

这是 RNN 的核心瓶颈。

== 5.1 线性 RNN 情况

考虑无非线性的 RNN：

$ h^{(t)} = W h^{(t-1)} + V x^{(t)} + b $

则：

$ frac(partial h^{(t)}, partial h^{(k)}) = product_{j=k+1}^{t} frac(partial h^{(j)}, partial h^{(j-1)}) = product_{j=k+1}^{t} W^->p = (W^->p)^{t-k} $

若：

$ W^->p = V op("diag")(lambda)V^{-1} $

则：

$ (W^->p)^{t-k} = V op("diag")(lambda)^{t-k}V^{-1} $

其中：

- $lambda_i$：$W^->p$ 的第 (i) 个特征值；
- (t-k)：梯度需要跨越的时间距离。

如果：

$ |lambda_i| > 1 $

则：

$ |lambda_i|^{t-k} -> oo $

产生梯度爆炸。

如果：

$ |lambda_i| < 1 $

则：

$ |lambda_i|^{t-k} -> 0 $

产生梯度消失。

直觉：
RNN 的长程梯度本质上是在不断乘同一个矩阵。只要矩阵谱半径不接近 1，梯度就会指数级放大或缩小。

---

== 5.2 非线性 RNN 情况

考虑：

$ h^{(t)} = f( W h^{(t-1)} + V x^{(t)} + b ) $

则：

$ frac(partial h^{(t)}, partial h^{(k)}) = product_{j=k+1}^{t} frac(partial h^{(j)}, partial h^{(j-1)}) = product_{j=k+1}^{t} W^->p op("diag")(f'(h^{(j-1)})) $

其范数满足：

$ | frac(partial h^{(t)}, partial h^{(j-1)}) | <= |W^->p| dot | op("diag")(f'(h^{(j-1)})) | $

非线性激活的作用是双面的：

- 它可以 bound recurrent dynamics，因此缓解 exploding gradient；
- 但如果 (f') 很小，比如 sigmoid/tanh 饱和区，就会让 vanishing gradient 更严重。

---

= 6. 缓解梯度问题的方法【★★★★★】

== 6.1 Gradient Clipping：解决爆炸【★★★★★】

课件给出 norm clipping：

$ hat(g) = nabla_theta cal(E)  hat(g) arrow.l cases(hat(g) "if" |hat(g)| < tau "," frac(tau, |hat(g)|) hat(g) "if" |hat(g)| >= tau) $

其中：

- $hat(g)$：原始梯度；
- $theta$：模型参数；
- $cal(E)$：训练目标；
- $tau$：梯度范数阈值。

直觉：
方向不变，只限制梯度长度，防止一次更新过大导致训练崩溃。

---

== 6.2 Identity Initialization + ReLU：缓解消失【★★★★】

课件提到可以初始化 recurrent matrix 为单位矩阵：

$ W approx I $

并使用：

$ f(z) = max(0,z) $

也就是 ReLU。

直觉：

$ h^{(t)} approx h^{(t-1)} + "input update" $

如果 (W = I)，那么历史信息更容易原样传递，不会每一步都被压缩或旋转到消失。

但这不是根本解决方案，所以后续引入 LSTM/GRU。

---

= 7. LSTM：Long Short-Term Memory【★★★★★】

LSTM 的核心目标是解决 Vanilla RNN 难以学习 long-term dependency 的问题。它通过 cell state $c^{(t)}$ 建立一条较稳定的信息高速通道，并用 gates 控制信息的遗忘、写入和输出。

== 7.1 LSTM 总公式

课件公式为：

$ f^{(t)} = sigma ( W^f h^{(t-1)} + U^f x^{(t)} + b^f )  i^{(t)} = sigma ( W^i h^{(t-1)} + U^i x^{(t)} + b^i )  o^{(t)} = sigma ( W^o h^{(t-1)} + U^o x^{(t)} + b^o )  c^{(t)} = f^{(t)} ∘ c^{(t-1)} + i^{(t)} ∘ tanh ( W^c h^{(t-1)} + U^c x^{(t)} + b^c )  h^{(t)} = o^{(t)} ∘ tanh(c^{(t)}) $

其中：

- $f^{(t)}$：forget gate；
- $i^{(t)}$：input gate；
- $o^{(t)}$：output gate；
- $c^{(t)}$：cell state；
- $h^{(t)}$：hidden state；
- $∘$：Hadamard prod，即逐元素乘法；
- $sigma(dot)$：sigmoid 函数，输出在 ([0,1])；
- $W^f,W^i,W^o,W^c$：hidden-to-gate 权重；
- $U^f,U^i,U^o,U^c$：input-to-gate 权重；
- $b^f,b^i,b^o,b^c$：bias。

---

== 7.2 Forget Gate【★★★★★】

课件另一种写法：

$ f_t = sigma(W_f[h_{t-1},x_t]+b_f) $

含义：

- $f_t approx 1$：保留旧 cell state；
- $f_t approx 0$：遗忘旧 cell state。

它回答的问题是：

```text
过去的信息哪些应该留下？
```

---

== 7.3 Input Gate 与 Candidate State【★★★★★】

$ i_t = sigma(W_i[h_{t-1},x_t]+b_i)  tilde(C)*t = tanh(W_C[h*{t-1},x_t]+b_C) $

其中：

- (i_t)：决定哪些维度需要更新；
- $tilde(C)_t$：候选新记忆。

它回答的问题是：

```text
当前输入中哪些新信息应该写入记忆？
```

---

== 7.4 Cell State Update【★★★★★】

$ C_t = f_t * C_{t-1} + i_t * tilde(C)_t $

含义：

```text
新记忆 = 保留下来的旧记忆 + 写入的新候选信息
```

这是 LSTM 最关键的公式。它让梯度可以沿着 cell state 的加法路径传播，缓解长程依赖中的梯度消失。

---

== 7.5 Output Gate【★★★★】

$ o_t = sigma(W_o[h_{t-1},x_t]+b_o)  h_t = o_t * tanh(C_t) $

含义：

- (C_t)：内部记忆；
- (h_t)：暴露给外部或下一层的 hidden representation；
- (o_t)：决定哪些记忆对当前输出可见。

它回答的问题是：

```text
记忆中的哪些内容应该用于当前输出？
```

---

= 8. GRU：Gated Recurrent Unit【★★★★★】

GRU 是更简洁的门控 RNN。它没有单独的 cell state，而是直接在 hidden state 上做更新。课件公式为：

$ z^{(t)} = sigma ( W^z h^{(t-1)} + U^z x^{(t)} + b^z )  r^{(t)} = sigma ( W^r h^{(t-1)} + U^r x^{(t)} + b^r )  bar(h)^{(t)} = tanh ( W(r^{(t)} ∘ h^{(t-1)}) + U x^{(t)} + b )  h^{(t)} = z^{(t)} ∘ h^{(t-1)} + (1-z^{(t)}) ∘ bar(h)^{(t)} $

其中：

- $z^{(t)}$：update gate，控制保留多少旧 hidden state；
- $r^{(t)}$：reset gate，控制生成候选记忆时是否忽略过去；
- $bar(h)^{(t)}$：candidate hidden state；
- $h^{(t)}$：最终 hidden state。

== 8.1 Reset Gate 的直觉

如果：

$ r^{(t)} approx 0 $

则：

$ r^{(t)} ∘ h^{(t-1)} approx 0 $

模型生成 $bar(h)^{(t)}$ 时几乎不看过去，相当于“重置记忆”。

== 8.2 Update Gate 的直觉

如果：

$ z^{(t)} approx 1 $

则：

$ h^{(t)} approx h^{(t-1)} $

说明主要保留旧状态。

如果：

$ z^{(t)} approx 0 $

则：

$ h^{(t)} approx bar(h)^{(t)} $

说明主要使用新候选状态。

---

= 9. LSTM vs GRU【★★★★】

| 比较维度 | LSTM | GRU |
| ----- | ------------------------- | ------------------ |
| 状态 | hidden state + cell state | 只有 hidden state |
| 门 | forget/input/output gates | update/reset gates |
| 参数量 | 更多 | 更少 |
| 训练速度 | 通常较慢 | 通常较快 |
| 小数据表现 | 不一定占优 | 常较好 |
| 长距离依赖 | 通常更强 | 稍弱 |
| 结构复杂度 | 高 | 低 |

课件总结：GRU 在很多任务上与 LSTM 表现相近，在较少训练数据上训练更快；但 LSTM 在需要建模长距离关系的任务中通常更强。

---

= 10. Bidirectional RNN【★★★★★】

普通 RNN 只能从左到右建模：

$ x_1,x_2,..,x_t $

但很多任务中，当前 token 的标签依赖未来上下文。例如 POS tagging 中：

```text
I saw a saw.
```

第二个 saw 的词性需要看上下文。

Bidirectional RNN 使用两个方向：

$ vec(h)_t = f ( vec(W)x_t + vec(V)vec(h)_{t-1} + vec(b) )  arrow.l{h}_t = f ( arrow.l{W}x_t + arrow.l{V}arrow.l{h}_{t-1} + arrow.l{b} ) $

输出为：

$ hat(y) = g(U h_t + c)  g ( U[vec(h)_t, arrow.l{h}_t] + c ) $

其中：

- $vec(h)_t$：从左到右的 hidden state；
- $arrow.l{h}_t$：从右到左的 hidden state；
- $[vec(h)_t,arrow.l{h}_t]$：拼接两个方向的信息。

注意：
BiRNN 适合 classification、tagging、encoding，不适合严格在线的 autoregressive generation，因为它需要看到未来 token。

---

= 11. Deep Bidirectional RNN【★★★★】

Deep BiRNN 将 bidirectional RNN 堆叠多层：

$ vec(h)^{i}_t = f ( vec(W)^{i}vec(h)^{i-1}*t + vec(V)^{i}vec(h)^{i}*{t-1} + vec(b)^{i} )  arrow.l{h}^{i}_t = f ( arrow.l{W}^{i}arrow.l{h}^{i-1}*t + arrow.l{V}^{i}arrow.l{h}^{i}*{t-1} + arrow.l{b}^{i} )  hat(y) = g ( U[ vec(h)^{L}_t, arrow.l{h}^{L}_t ]+c ) $

其中：

- (i)：第 (i) 层；
- (L)：总层数；
- $vec(h)^{i-1}_t$：下层前向表示；
- $vec(h)^{i}_t$：当前层前向表示。

直觉：
底层捕捉局部模式，高层捕捉抽象语义。

---

= 12. Encoder-Decoder Seq2Seq【★★★★★】

前面的 RNN 可以处理同长度 many-to-many，但机器翻译、语音识别、问答等任务中，输入输出长度不一定相同。

因此引入 Encoder-Decoder：

```text
Encoder: x_1, ..., x_T → context vector C
Decoder: C → y_1, ..., y_S
```

课件指出，当 (C) 是 fixed-size vector 时，该结构可看成：

```text
sequence → fixed-size vector
fixed-size vector → sequence
```

但问题是：

$ C $

可能太小，无法总结长序列。

这就是 naive seq2seq 的 fixed-vector bottleneck。

---

= 13. Attention Mechanism【★★★★★】

Attention 的核心动机是解决固定 context vector (C) 的瓶颈。Bahdanau attention 的思想是：不要把整个输入压成一个固定向量，而是让 decoder 在每个输出位置动态选择输入中最相关的位置。

== 13.1 Decoder 预测公式

课件给出：

$ p(y_t mid y_1,..,y_{t-1},x) = g(y_{t-1},s_t,c_t) $

其中：

- (y_t)：当前要生成的 token；
- $y_{t-1}$：上一个 token；
- (s_t)：decoder hidden state；
- (c_t)：当前时间步的 context vector；
- (x)：输入序列；
- (g)：输出概率函数。

Decoder hidden state：

$ s_t = f(s_{t-1}, y_{t-1}, c_t) $

---

== 13.2 Context Vector

Encoder 把输入映射成 annotations：

$ (h_1,h_2,..,h_T) $

每个 (h_j) 表示输入第 (j) 个位置的上下文表示。

Attention context vector：

$ c_t = sum_{j=1}^{T} alpha_{t,j}h_j $

其中：

- (c_t)：decoder 第 (t) 步使用的上下文；
- $alpha_{t,j}$：第 (t) 个输出位置对第 (j) 个输入位置的注意力权重；
- (h_j)：encoder 第 (j) 个位置的 hidden annotation。

直觉：
(c_t) 是输入表示的加权平均。权重越大，说明当前输出越关注该输入位置。

---

== 13.3 Attention Weight

课件公式为：

$ alpha_{"tj"} = frac(exp(e_{"tj"}), sum_{k=1}^{T} exp(e_{"tk"})) $

其中：

$ e_{"tj"}=a(s_{t-1},h_j) $

- $e_{"tj"}$：alignment score；
- $a(dot)$：alignment model，通常是 feedforward neural network；
- $s_{t-1}$：decoder 上一步状态；
- (h_j)：encoder 第 (j) 个位置表示。

完整形式：

$ p(y_i mid y_1,..,y_{i-1},x) = g(y_{i-1},s_i,c_i)  s_i = f(s_{i-1},y_{i-1},c_i)  c_i = sum_{j=1}^{T_x} alpha_{"ij"} h_j  alpha_{"ij"} = frac(exp(e_{"ij"}), sum_{k=1}^{T_x} exp(e_{"ik"}))  e_{"ij"} = a(s_{i-1},h_j) $

这个机制解决了：

```text
固定 C 太小的问题
```

因为每个 decoder step 都有自己的 (c_i)，可以动态访问整个输入序列。

---

= 14. RNN 的问题与向 Transformer 演进【★★★★★】

课件总结 RNN 的主要问题：

+ 难以高效并行化；
+ 需要 backpropagation through sequence；
+ 局部信息和全局信息都要通过 hidden state 这个 bottleneck 传递。

因此出现 convolutional sequence models，比如 Neural GPU、ByteNet、ConvS2S。但卷积模型也受限于 convolution window size。

Self-attention 的优势是：

```text
任意两个位置之间的路径长度是常数
每层内部容易并行
```

这就是 Transformer 替代 RNN 的核心原因。

---

= 15. Scaled Dot-Product Attention【★★★★★】

课件公式为：

$ "Attention"(Q,K,V) = "softmax" ( frac(Q K^->p, sqrt{d_k}) ) V $

其中：

- (Q)：queries；
- (K)：keys；
- (V)：values；
- (d_k)：query/key 向量维度；
- $Q K^->p$：query 与 key 的相似度矩阵；
- $sqrt{d_k}$：缩放因子；
- softmax：把相似度转成注意力权重。

为什么要除以 $sqrt{d_k}$？

如果 (Q,K) 维度很高，dot product 的方差会变大，导致 softmax 输入过大，使 softmax 饱和，梯度变小。因此 scaling 可以稳定训练。

---

= 16. Multi-Head Attention【★★★★★】

单个 scaled dot-product attention 本质上是 weighted average。课件指出它缺少类似 convolution 中“不同相对位置使用不同线性变换”的能力。解决方案是 Multi-Head Attention。

公式：

$ "MultiHead"(Q,K,V) = "Concat"("head"_1,..,"head"_h)W^O  "head"_i = "Attention" (Q W_i^Q,K W_i^K,V W_i^V) $

其中：

- (h)："head" 数量；
- $W_i^Q$：第 (i) 个 "head" 的 query projection；
- $W_i^K$：第 (i) 个 "head" 的 key projection；
- $W_i^V$：第 (i) 个 "head" 的 value projection；
- $W^O$：输出 projection；
- $"Concat"$：把多个 "head" 拼接。

直觉：
不同 "head" 可以学习不同关系，例如：

- 语法依赖；
- 长距离 coreference；
- 局部 phrase；
- 位置关系；
- 语义相似性。

---

= 17. Transformer【★★★★】

课件总结 Transformer：

- 完全基于 attention；
- 去掉 recurrence；
- 去掉 convolution；
- Encoder：6 层 self-attention + feedforward network；
- Decoder：6 层 masked self-attention + encoder output attention + feedforward。

Transformer 相比 RNN 的优势：

| 维度 | RNN | Transformer |
| ------ | ----------------------------- | -------------------- |
| 序列依赖路径 | (O(T)) | (O(1)) per layer |
| 并行化 | 差 | 强 |
| 长程依赖 | 难 | 更容易 |
| 记忆机制 | hidden state recurrent memory | all-pairs attention |
| 时间/显存 | 通常 (O(T)) recurrent | attention 为 (O(T^2)) |

Transformer 的核心代价是：

$ O(T^2) $

因为它需要计算所有 token pair 的 attention score。

---

= 18. 从 Transformer 到 SSM / Mamba【★★★★】

课件最后讨论 attention 的瓶颈：

$ "Attention"(X) = "softmax" ( frac(Q K^->p, sqrt{d}) )V $

其时间和显存复杂度都是：

$ O(T^2) $

长上下文训练和推理昂贵。因此问题变成：

```text
能否用硬件友好的 recurrent memory 替代显式 all-pairs interaction？
```

== 18.1 RNN → SSM → Mamba

Classic RNN：

$ h_t = f(W_h h_{t-1} + W_x x_t) $

Linear RNN / Discrete SSM：

$ h_t = bar(A) h_{t-1} + bar(B) x_t  y_t = C h_t $

Mamba：

$ h_t = A(x_t)h_{t-1}+B(x_t)x_t  y_t = C(x_t)h_t $

核心区别：

| 模型 | 状态更新 |
| ----- | ---------------------------- |
| RNN | 非线性 recurrent update |
| SSM | 结构化线性状态演化 |
| Mamba | 输入依赖的 selective state update |

---

== 18.2 Selective Mechanism

传统 SSM：

$ h_t = A h_{t-1} + B x_t  y_t = C h_t $

Mamba：

$ A_t = f_A(x_t)  B_t = f_B(x_t)  C_t = f_C(x_t)  h_t = A_t h_{t-1} + B_t x_t  y_t = C_t h_t $

直觉：

- 重要 token 强烈更新 memory；
- 不重要 token 几乎不改变 memory；
- 类似 LSTM 的 gating，但放在 structured state-space evolution 中。

---

== 18.3 为什么 Mamba 可以并行化？

考虑 time-varying linear recurrence：

$ h_t = A_t h_{t-1} + b_t $

展开：

$ h_t = A_t A_{t-1} .. A_1 h_0 + sum_{k=1}^{t} (A_t A_{t-1} .. A_{k+1}) b_k $

这是 prefix product + prefix sum 结构，可以用 parallel scan 计算。

因此：

- 总 work 仍是 (O(T))；
- GPU utilization 更好；
- 避免 attention 的 $O(T^2)$ memory。

---

= 19. 总结对比表【★★★★★】

| 阶段 | 核心方法 | 解决什么问题 | 新问题 |
| ----------------- | ----------------------- | --------------------- | ------------------------------------- |
| Feedforward NN | 固定输入输出 | 普通分类/回归 | 无法处理变长序列 |
| Vanilla RNN | hidden state recurrence | 处理序列历史 | 长程依赖难学，梯度消失/爆炸 |
| BPTT | 时间展开反传 | 训练 recurrent model | 序列长时计算昂贵 |
| Gradient clipping | 限制梯度范数 | 缓解 exploding gradient | 不解决 vanishing gradient |
| LSTM | cell state + gates | 保留长期记忆 | 参数多，训练慢 |
| GRU | update/reset gates | 简化 LSTM | 长程建模略弱 |
| BiRNN | 双向上下文 | 利用过去和未来 | 不适合在线生成 |
| Encoder-Decoder | seq2seq | 输入输出长度不同 | fixed vector bottleneck |
| Attention | 动态选择输入位置 | 缓解 context bottleneck | 计算注意力仍较复杂 |
| Transformer | self-attention | 并行、短路径长程依赖 | (O(T^2)) 长上下文昂贵 |
| SSM/Mamba | selective state update | 长序列高效建模 | all-pairs interaction 不如 attention 直接 |

---

= 20. 期末复习抓重点

最应该掌握的是：

+ _Vanilla RNN 三个方程_：

$ h^{(t)}=tanh(b+W h^{(t-1)}+U x^{(t)})  o^{(t)}=c+V h^{(t)}  hat(y)^{(t)}="softmax"(o^{(t)}) $

+ _BPTT 的 hidden gradient 递推_：

$ delta^{(t)} = "diag"(1-(h^{(t)})^2) [ V^->p nabla_{o^{(t)}} L + W^->p delta^{(t+1)} ] $

+ _梯度消失/爆炸的数学原因_：

$ (W^->p)^{t-k} = V op("diag")(lambda)^{t-k} V^{-1} $

+ _LSTM cell update_：

$ C_t=f_t*C_{t-1}+i_t*tilde(C)_t $

+ _GRU final memory_：

$ h^{(t)} = z^{(t)}∘ h^{(t-1)} + (1-z^{(t)})∘ bar(h)^{(t)} $

+ _Attention context vector_：

$ c_i=sum_{j=1}^{T_x}alpha_{"ij"}h_j $

+ _Scaled dot-product attention_：

$ "Attention"(Q,K,V) = "softmax" ( frac(Q K^->p, sqrt{d_k}) )V $

+ _Mamba selective update_：

$ h_t = A(x_t) h_{t-1} + B(x_t) x_t  y_t = C(x_t) h_t $

一句话总结本章：

RNN 通过 recurrent hidden state 解决变长序列建模；LSTM/GRU 用门控解决长期记忆；Attention/Transformer 用全局交互解决 RNN 的瓶颈；Mamba 又回到 recurrent state evolution，但通过 selective scan 实现长序列高效建模。
