// @title: Transformer 注意力
// @description: 自注意力把序列中每个 token 的表示更新为其他 token 表示的加权和。
// @order: 999

= Transformer 注意力

自注意力把序列中每个 token 的表示更新为其他 token 表示的加权和。

== Scaled Dot-Product Attention

$ op("Attention")(Q, K, V)
= op("softmax")(frac(Q K^->p, sqrt{d_k}))V $

缩放因子 $sqrt{d_k}$ 用来控制 logits 的方差，避免 softmax 过早饱和。

== 多头注意力

多头注意力把表示投影到多个子空间：

$ op("head")_i =
op("Attention")(Q W_i^Q, K W_i^K, V W_i^V) $

然后拼接所有 "head"。更多线性映射背景见 [[math/linear-algebra|线性代数]]。
