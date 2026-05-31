// @title: Attention Mechanism
// @description: Transformer 中注意力机制的基础笔记
// @order: 10

#set heading(numbering: "1.")

= Attention Mechanism

注意力机制把查询与键的相关性转换为对值的加权组合。它与
[[math/linear-algebra|线性代数]] 中的矩阵乘法密切相关。

== Scaled Dot-Product Attention

最常用的形式是：

$ "Attention"(Q, K, V) = op("softmax")((Q K^T) / sqrt(d_k)) V $

缩放因子 $sqrt(d_k)$ 用于控制点积幅度。

== Minimal implementation

```python
def attention(query, key, value):
    scores = query @ key.transpose(-2, -1)
    weights = scores.softmax(dim=-1)
    return weights @ value
```

== Cross reference <attention-note>

本节定义一个可引用位置。

后续可在同一篇笔记中使用 @attention-note[Cross reference] 引用这个位置。

#figure(
  table(
    columns: 2,
    [符号], [含义],
    [$Q$], [查询矩阵],
    [$K$], [键矩阵],
    [$V$], [值矩阵],
  ),
  caption: [注意力机制中的核心矩阵],
)
