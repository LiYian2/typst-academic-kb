// @title: Vision Transformer
// @description: Vision Transformer 将图像切分为 patch，并把 patch 当作 token 输入 Transformer。
// @order: 999

= Vision Transformer

Vision Transformer 将图像切分为 patch，并把 patch 当作 token 输入 Transformer。

== Patch Embedding

设图像大小为 $H times W$，patch 大小为 $P times P$，token 数量为：

$ N = frac(H W, P^2) $

每个 patch 被展平后通过线性层映射到 $d$ 维表示。

== 与 CNN 的差异

CNN 通过局部卷积引入归纳偏置；ViT 更依赖数据规模和位置编码。相关背景见 [[deep-learning/representation-learning|表示学习]]。

== 伪代码

```python
def patch_count(height: int, width: int, patch_size: int) -> int:
 return (height * width) // (patch_size ** 2)
```
