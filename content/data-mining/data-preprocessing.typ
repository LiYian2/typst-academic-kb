// @title: 数据、度量与数据预处理
// @description: 数据类型、统计度量、相异度、清洗、变换、规约与降维的完整复习笔记。
// @order: 20

#let formula-box(body) = html.elem("div", attrs: (class: "formula-box",))[#body]

下面这份笔记严格以你上传的 Chapter 2《Data, Measurements, and Data Preprocessing》为主线整理，覆盖课件 113 页的知识结构、公式、例题、图示含义以及方法之间的演进关系。对于课件中少数前后不完全一致之处，我会明确标注，而不是自行"修正"后隐藏掉。

重要程度标记：

- ★★★★★：期末应达到"能定义、能解释、能计算、能比较"的程度
- ★★★★☆：核心概念，通常需要理解 Why / How
- ★★★☆☆：应认识并能够辨析
- ★★☆☆☆：理解用途即可
- ★☆☆☆☆：背景知识


= Chapter 2：Data, Measurements, and Data Preprocessing
<chapter-2data-measurements-and-data-preprocessing>
= 0. 全章到底在解决什么问题？★★★★★
<全章到底在解决什么问题>
本章真正的核心不是"背一堆 preprocessing 方法"，而是在回答一个更基础的问题：

#quote(block: true)[
在把现实世界的数据交给 Data Mining algorithm 之前，我们究竟应该如何表示、理解、比较、清洗、转换和压缩它，使后续算法处理的对象真正具有可计算意义？
]

整个章节可以理解成一条逐层演进的 pipeline：

```text
现实世界对象
   ↓
如何表示？
Data objects + Attributes + Data types
   ↓
数据长什么样？
Central tendency / Dispersion / Distribution / Correlation
   ↓
两个对象有多像？
Similarity / Distance / Proximity
   ↓
现实数据不干净怎么办？
Cleaning + Integration
   ↓
数据尺度/形式不适合算法怎么办？
Normalization + Discretization + Feature construction
   ↓
数据太多怎么办？
Compression + Sampling + Numerosity reduction
   ↓
维度太高怎么办？
Feature Selection / Feature Extraction
   ↓
PCA
   ↓
线性结构不够怎么办？
KPCA / SNE / t-SNE
```

因此本章的"技术进化树"可以概括为：

```text
Raw Data
│
├─ 数据语义问题
│   ├─ Data type
│   ├─ Attribute type
│   └─ Data object
│
├─ 数据理解问题
│   ├─ Mean / Median / Mode
│   ├─ Variance / Standard deviation
│   ├─ Covariance / Correlation
│   └─ Visualization
│
├─ 对象比较问题
│   ├─ Numeric → Minkowski
│   ├─ Binary → Matching / Jaccard
│   ├─ Nominal → Simple matching
│   ├─ Ordinal → Rank normalization
│   ├─ Vector → Cosine
│   └─ Distribution → KL divergence
│
├─ 数据质量问题
│   ├─ Missing
│   ├─ Noise
│   ├─ Inconsistency
│   └─ Multi-source conflicts
│
├─ 数据形式问题
│   ├─ Normalization
│   ├─ Discretization
│   └─ Concept hierarchy
│
├─ 数据规模问题
│   ├─ Compression
│   ├─ Sampling
│   ├─ Regression
│   ├─ Histogram
│   └─ Clustering
│
└─ 高维问题
    ├─ Attribute subset selection
    ├─ PCA
    │   └─ 只能线性
    └─ Nonlinear DR
        ├─ KPCA
        └─ SNE / t-SNE
```

这条链就是整章最值得记住的逻辑结构。


= 1. Data Types：首先弄清楚"数据到底是什么"★★★★★
<data-types首先弄清楚数据到底是什么>
== 1.1 四大类 Data Sets
<四大类-data-sets>
课件 pp.2--5 将数据集分成四类。

=== 1. Record Data ★★★★★
<record-data>
本质：

#quote(block: true)[
一个 object 对应一条 record，每条 record 由若干 attributes 描述。
]

包括：

+ Relational records
+ Relational tables
+ Data matrix
+ Crosstabs
+ Transaction data
+ Document data

=== Relational table
<relational-table>
典型形式：

$ X = mat(delim: "[", x_11, x_12, dots.h.c, x_(1 d); x_21, x_22, dots.h.c, x_(2 d); dots.v, dots.v, dots.down, dots.v; x_(n 1), x_(n 2), dots.h.c, x_(n d)) $

通常：

- row → data object
- column → attribute
- $n$ → objects 数量
- $d$ → attributes / dimensions 数量

=== Transaction data
<transaction-data>
例如：

#figure(
  align(center)[#table(
    columns: 2,
    align: (auto,auto,),
    table.header([TID], [Items],),
    table.hline(),
    [1], [Bread, Coke, Milk],
    [2], [Beer, Bread],
    [3], [Beer, Coke, Diaper, Milk],
  )]
  , kind: table
  )

它与普通 data matrix 不同：

#quote(block: true)[
每个 transaction 实际包含的是一个 set，而不是固定长度的普通数值向量。
]

这是后续 association rule mining 的基础。

=== Document data
<document-data>
把 document 转换成 term-frequency vector。

例如词汇表有 $d$ 个词： $upright(bold(x))_i =(x_(i 1),x_(i 2),dots.h,x_(i d))$

其中

$ x_(i j) = upright("term ") j upright(" 在 document ") i upright(" 中出现次数") $

于是自然语言 document 被变成 vector。

后面 cosine similarity 就是为这类高维 sparse vectors 服务的。


== 1.2 Graphs and Networks ★★★☆☆
<graphs-and-networks>
pp.3：

- Transportation network
- World Wide Web
- Molecular structures
- Social/information networks

核心变化：

Record Data： $upright("object") arrow.r upright("attributes")$

Graph： $upright("objects") + upright("relationships")$

因此 graph data 中不仅节点本身重要： $V = { v_1,dots.h,v_n }$

还需要考虑边： $E subset.eq V times V$

这也是后面课件指出传统 vector similarity 无法表达复杂 semantics 的原因之一。


== 1.3 Ordered Data ★★★☆☆
<ordered-data>
pp.4：

- Video：sequence of images
- Temporal data：time series
- Sequential data：transaction sequences
- Genetic sequence

和 Record Data 的核心区别：

#quote(block: true)[
顺序本身携带信息。
]

例如：

```text
A → B → C
```

和

```text
C → B → A
```

即使元素完全一样，也不一定具有同样语义。

这是 Bag-of-Words 方法的重要局限之一。


== 1.4 Spatial / Image / Multimedia Data ★★★☆☆
<spatial-image-multimedia-data>
pp.5：

- Spatial data：maps
- Image
- Video

Spatial data 的一个重要特点：

#quote(block: true)[
Spatial relationship 自身构成信息。
]

例如：

- nearby
- overlap
- inside
- north of
- adjacency

因此不能简单把它们视作 unordered attributes。


= 2. Structured Data 的四个重要特性★★★★☆
<structured-data-的四个重要特性>
课件 p.6：

== 2.1 Dimensionality
<dimensionality>
维数 $d$ 很高会出现：

#quote(block: true)[
Curse of dimensionality。
]

后面 PCA / dimensionality reduction 就是为解决这个问题。


== 2.2 Sparsity
<sparsity>
高维数据经常大量为 0。

例如 document vector： $(0,0,0,4,0,0,2,0,dots.h)$

课件特别写：

#quote(block: true)[
Only presence counts
]

对于很多 sparse binary / transaction tasks：

"出现"可能比"不出现"信息量更大。

因此后面 asymmetric binary / Jaccard 会：

#quote(block: true)[
不考虑 0--0 matching。
]


== 2.3 Resolution
<resolution>
#quote(block: true)[
Patterns depend on scale.
]

例如等待时间：

- hourly level
- daily level
- weekly level

可能显示完全不同的规律。

因此没有所谓唯一"正确"的 resolution。


== 2.4 Distribution
<distribution>
数据分析主要看：

- Centrality
- Dispersion

即： $upright("Where is the data?")$

以及

$ upright("How spread out is the data?") $


= 3. Data Objects 与 Attributes★★★★★
<data-objects-与-attributes>
== 3.1 Data Object
<data-object>
p.7：

Data set 由 data objects 组成。

一个 data object 表示一个现实 entity。

同义词：

- sample
- example
- instance
- data point
- object

例如：

Sales database：

- customer
- item
- sale

Medical database：

- patient
- treatment

University：

- student
- professor
- course

关系数据库中：

#formula-box[$ upright("row") = upright("object") $]

#formula-box[$ upright("column") = upright("attribute") $]

=== 课件提问：What is the data object of a Crosstab?
<课件提问what-is-the-data-object-of-a-crosstab>
课件只提出问题，没有给正式答案。

严格而言取决于 crosstab 的建模方式。如果作为二维 data matrix 使用，通常 row 可以视作 object；但在 OLAP-style crosstab 中，一个 cell 也可能表示一个由 row category × column category 定义的 aggregated group。

所以期末若出现，首先看题目如何定义 observation。


= 4. Attribute Types：极重要★★★★★
<attribute-types极重要>
这一部分是后面：

- distance
- normalization
- discretization
- statistical analysis

的基础。


== 4.1 Nominal Attribute ★★★★★
<nominal-attribute>
只有类别，没有顺序。

例如：

$ upright("HairColor") = { upright("black,brown,blond,...") } $

其他：

- marital status
- occupation
- ID
- zipcode

注意：

#quote(block: true)[
数字编码 ≠ numeric attribute。
]

例如：

```text
Zip code = 511400
```

虽然看起来是 number，但加减乘除没有语义。

因此它仍然是 nominal。


= 4.2 Binary Attribute★★★★★
<binary-attribute>
只有两个 states： ${ 0,1 }$

本质上：

#quote(block: true)[
Binary 是 nominal 的特殊情况。
]

但需要区分：

=== Symmetric binary
<symmetric-binary>
两个结果同等重要。

例如课件：

- gender

0 和 1 没有谁更值得关注。

=== Asymmetric binary
<asymmetric-binary>
两个 outcomes 不同等重要。

例如：

medical test： $1 = upright("positive")$

$ 0 = upright("negative") $

Convention：

#quote(block: true)[
把更重要、较稀有的事件设为 1。
]

这里非常重要，因为后面 Jaccard：

#quote(block: true)[
0--0 不计入 similarity。
]


= 4.3 Ordinal Attribute★★★★★
<ordinal-attribute>
有顺序，但相邻 levels 的距离未知。

例如：

$ upright("size") = { upright("small, medium, large") } $

或者：

- grades
- military rankings
- freshman / sophomore / junior / senior

知道：

$ upright("freshman") < upright("sophomore") < upright("junior") < upright("senior") $

但不能说：

$ upright("senior") - upright("junior") = upright("sophomore") - upright("freshman") $

除非我们人为进行 rank mapping。


= 4.4 Numeric Attribute★★★★★
<numeric-attribute>
分两种：

== Interval-scaled
<interval-scaled>
满足：

- 有顺序
- equal-sized units
- 没有 absolute zero

例如： $""^compose C,quad^compose F$

calendar dates。

摄氏： $100^compose C$

不能说：

#quote(block: true)[
是 $50^compose C$ 的两倍热。
]

因为 $0^compose C$ 不是 absence of temperature。


== Ratio-scaled
<ratio-scaled>
有 true zero。

例如：

- Kelvin temperature
- length
- count
- money

因此： $10 K = 2 times 5 K$

具有真正的比例意义。

课件给： $K = zws^compose C + 273.15$

=== 一张表记住
<一张表记住>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,right,right,right,right,),
    table.header([类型], [分类], [顺序], [差值], [比例],),
    table.hline(),
    [Nominal], [✓], [✗], [✗], [✗],
    [Ordinal], [✓], [✓], [✗], [✗],
    [Interval], [✓], [✓], [✓], [✗],
    [Ratio], [✓], [✓], [✓], [✓],
  )]
  , kind: table
  )

★★★★★ 高频辨析。


= 5. Discrete vs Continuous★★★★★
<discrete-vs-continuous>
== Discrete
<discrete>
只有：

- finite
- countably infinite

values。

例如： ${ A,B,A B,O }$

年份： $dots.h,1999,2000,2001,dots.h$

Binary 是 discrete 的特殊情况。


== Continuous
<continuous>
理论上取： $bb(R)$

中的连续值。

例如：

- temperature
- height
- weight

现实计算机只能有限 precision：

#quote(block: true)[
floating-point approximation。
]


= 6. Statistics of Data：为什么先统计？★★★★★
<statistics-of-data为什么先统计>
这一部分逻辑：

```text
Raw numbers
↓
中心在哪里？
Mean / Median / Mode
↓
散布有多大？
Variance / SD / IQR
↓
变量是否一起变化？
Covariance / Correlation
↓
数据到底长什么样？
Histogram / Boxplot / QQ / Scatter
```


= 7. Central Tendency★★★★★
<central-tendency>
= 7.1 Mean
<mean>
== Sample mean
<sample-mean>
#formula-box[$ macron(x) = 1 / n sum_(i = 1)^n x_i $]

其中：

- $x_i$：第 $i$ 个 sample
- $n$：sample size
- $macron(x)$：sample mean


== Population mean
<population-mean>
#formula-box[$ mu = 1 / N sum_(i = 1)^N x_i $]

其中：

- $N$：population size
- $mu$：population mean


== Weighted Arithmetic Mean
<weighted-arithmetic-mean>
#formula-box[$ macron(x) = frac(sum_(i = 1)^n w_i x_i, sum_(i = 1)^n w_i) $]

其中：

- $w_i$：第 $i$ 个 observation 的权重
- $x_i$：第 $i$ 个值

直觉：

#quote(block: true)[
一个 observation 被重复 $w_i$ 次。
]


== Trimmed Mean ★★★★☆
<trimmed-mean>
问题：

Mean 很容易被极端值影响。

例如： $1,2,3,4,1000$

mean： $202$

明显不能代表典型值。

因此：

#quote(block: true)[
Trimmed mean：先删除一定比例最小值和最大值，再计算 mean。
]

课件例子：

Olympic gymnastics scoring。

=== 方法演进
<方法演进>
```text
Mean
↓
Why fails?
Extreme values strongly influence mean
↓
Trimmed mean / Median
↓
降低 outlier influence
```


= 7.2 Median★★★★★
<median>
排序以后：

奇数个：

$ upright("Median") = x_((frac(n + 1, 2))) $

偶数个：

$ upright("Median") = frac(x_((n/2)) + x_((n/2 + 1)), 2) $


== Grouped data 的 median 插值公式★★★★★
<grouped-data-的-median-插值公式>
课件 p.16：

#formula-box[$ upright("Median") = L_1 + frac(n/2 - sum f_(upright("before")), f_(upright("median"))) times w i d t h $]

更规范写为：

#formula-box[$ M = L + (frac(n / 2 - F, f_m)) h $]

其中：

- $M$：estimated median
- $L$：median interval 的 lower boundary
- $n$：总 frequency
- $F$：median interval 之前的 cumulative frequency
- $f_m$：median interval frequency
- $h$：interval width

=== 推导直觉
<推导直觉>
我们要找到第：

$ n / 2 $

个 observation。

进入 median bin 之前已经有： $F$

个 observation。

因此还需要向 bin 内前进：

$ n / 2 - F $

个。

假设数据在该 interval 中均匀分布，则走过的比例：

$ frac(n/2 - F, f_m) $

乘 bin 宽度： $h$

最终：

$ M = L + frac(n/2 - F, f_m) h $


= 7.3 Mode★★★★☆
<mode>
定义：

#formula-box[$ upright("Mode") = upright("most frequently occurring value") $]

可能：

- unimodal
- bimodal
- trimodal
- multimodal

课件给经验关系：

#formula-box[$ upright("mean") - upright("mode") = 3(upright("mean") - upright("median")) $]

因此：

#formula-box[$ upright("mode") = 3 upright("median") - 2 upright("mean") $]

注意：

#quote(block: true)[
这是 empirical relation，不是普遍数学定理。
]


= 8. Symmetric vs Skewed Distribution★★★★★
<symmetric-vs-skewed-distribution>
== Symmetric
<symmetric>
通常：

$ upright("mean") = upright("median") = upright("mode") $


== Positive skew / right skew
<positive-skew-right-skew>
右侧长尾：

#formula-box[$ upright("mode") < upright("median") < upright("mean") $]

因为大值把 mean 往右拖。

Skewness： $> 0$


== Negative skew / left skew
<negative-skew-left-skew>
左侧长尾：

#formula-box[$ upright("mean") < upright("median") < upright("mode") $]

Skewness： $< 0$


= 9. Normal Distribution：68--95--99.7 Rule★★★★★
<normal-distribution689599.7-rule>
课件 p.19。

若： $X tilde.op N(mu,sigma^2)$

则约： $P(mu - sigma < X < mu + sigma)approx 68 %$

$ P(mu - 2 sigma < X < mu + 2 sigma)approx 95 % $

$ P(mu - 3 sigma < X < mu + 3 sigma)approx 99.7 % $

其中：

- $mu$：central tendency
- $sigma$：spread
- $sigma^2$：variance


= 10. Variance & Standard Deviation★★★★★
<variance-standard-deviation>
= 10.1 Sample Variance
<sample-variance>
课件 p.20：

#formula-box[$ s^2 = frac(1, n - 1) sum_(i = 1)^n(x_i - macron(x))^2 $]

标准差：

#formula-box[$ s = sqrt(s^2) $]


== Computational form★★★★★
<computational-form>
课件给：

$ s^2 = frac(1, n - 1) [sum_(i = 1)^n x_i^2 - 1 / n (sum_(i = 1)^n x_i)^2] $

=== 推导
<推导>
从：

$ sum_i(x_i - macron(x))^2 $

展开：

$ = sum_i(x_i^2 - 2 x_i macron(x) + macron(x)^2) $

$ = sum_i x_i^2 - 2 macron(x) sum_i x_i + n macron(x)^2 $

因为：

$ sum_i x_i = n macron(x) $

所以：

$ = sum_i x_i^2 - 2 n macron(x)^2 + n macron(x)^2 $

$ = sum_i x_i^2 - n macron(x)^2 $

又：

$ macron(x) = 1 / n sum_i x_i $

所以：

$ n macron(x)^2 = 1 / n (sum_i x_i)^2 $

于是：

#formula-box[$ sum_i(x_i - macron(x))^2= sum_i x_i^2 - 1 / n(sum_i x_i)^2 $]

意义：

#quote(block: true)[
只要维护 $sum x_i$、$sum x_i^2$、$n$，就可以 incremental computation。
]

这就是课件问：

#quote(block: true)[
Can you compute it incrementally and efficiently?
]

的关键答案。


= 10.2 Population variance
<population-variance>
#formula-box[$ sigma^2 = 1 / N sum_(i = 1)^N(x_i - mu)^2 $]

也可写：

#formula-box[$ sigma^2 = 1 / N sum_i x_i^2 - mu^2 $]


= 11. Random Variable 的 Variance★★★★★
<random-variable-的-variance>
课件 p.26 把样本统计扩展到 probability distribution。

若 $X$ 是 discrete：

#formula-box[$ "Var"(X)= sum_x(x - mu)^2f(x) $]

若 continuous：

#formula-box[$ "Var"(X)= integral_(- oo)^(+ oo)(x - mu)^2f(x)thin d x $]

统一：

#formula-box[$ sigma^2 = "Var"(X)= E[(X - mu)^2] $]

其中： $mu = E[X]$


== 最重要恒等式★★★★★
<最重要恒等式>
#formula-box[$ "Var"(X)= E[X^2]- E[X]^2 $]

推导： $E[(X - mu)^2]$

$ = E[X^2 - 2 mu X + mu^2] $

利用 expectation linearity：

$ = E[X^2]- 2 mu E[X]+ mu^2 $

由于： $E[X]= mu$

所以：

$ = E[X^2]- 2 mu^2 + mu^2 $

#formula-box[$ = E[X^2]- mu^2 $]


== 一个课件细节：$1/n$ vs $1/(n - 1)$
<一个课件细节1n-vs-1n-1>
p.26 同时显示：

$ s^2 = 1 / n sum_i(x_i - hat(mu))^2 $

以及：

$ s^2 = frac(1, n - 1) sum_i(x_i - hat(mu))^2 $

而 p.20 明确把 sample variance 定义为 $1/(n - 1)$。

期末建议：

#quote(block: true)[
如果题目写 sample variance 且没有其他说明，按照课件 p.20 使用 $n - 1$。
]

从统计学角度：

- $1/n$：常见于 MLE / empirical variance
- $1/(n - 1)$：unbiased sample variance

但这是对课件公式差异的解释，不是课件额外展开的内容。


= 12. Chi-Square：Categorical Correlation★★★★★
<chi-squarecategorical-correlation>
数值型数据用 covariance/correlation。

Categorical variables 不能直接算普通 Pearson correlation。

于是引出： $chi^2$

test。


= 12.1 Null Hypothesis
<null-hypothesis>
#formula-box[$ H_0 : upright("two categorical variables are independent") $]


= 12.2 Chi-square Statistic
<chi-square-statistic>
#formula-box[$ chi^2 = sum_i frac((O_i - E_i)^2, E_i) $]

更一般：

#formula-box[$ chi^2 = sum_i sum_j frac((O_(i j) - E_(i j))^2, E_(i j)) $]

其中：

- $O_(i j)$：observed count
- $E_(i j)$：expected count assuming independence

值越大：

$ \|O - E divides arrow.t arrow.r.double chi^2 arrow.t $

说明：

#quote(block: true)[
实际 joint distribution 和 independence 假设差异越大。
]


= 12.3 Expected Count★★★★★
<expected-count>
课件通过例子给出：

#formula-box[$ E_(i j) = frac((upright("row total")_i)(upright("column total")_j), N) $]

例如：

```text
                   Chess      No Chess      Total
Like SF              250          200          450
Not Like SF           50         1000         1050
Total                300         1200         1500
```

若 independent：

$ E_11 = frac(450 times 300, 1500) = 90 $

同理：

$ E_12 = frac(450 times 1200, 1500) = 360 $

$ E_21 = frac(1050 times 300, 1500) = 210 $

$ E_22 = frac(1050 times 1200, 1500) = 840 $


= 12.4 Chi-square example★★★★★
<chi-square-example>
$ chi^2 = frac((250 - 90)^2, 90) + frac((50 - 210)^2, 210) + frac((200 - 360)^2, 360) + frac((1000 - 840)^2, 840) $

课件结果：

#formula-box[$ chi^2 = 507.93 $]

非常大。

因此 science fiction preference 与 chess-playing：

#quote(block: true)[
statistically correlated。
]

课件指出：

#quote(block: true)[
可以在显著性水平 $alpha = 0.001$ 下 reject independence hypothesis（对应 confidence level 为 $0.999$）。
]


= 12.5 Degrees of Freedom★★★★★
<degrees-of-freedom>
若两个 categorical variables 分别有：

- $r$ categories
- $c$ categories

则：

#formula-box[$ d f =(r - 1)(c - 1) $]

上面的 $2 times 2$ table： $d f =(2 - 1)(2 - 1)= 1$

为什么？

因为 row sums + column sums 固定以后：

#quote(block: true)[
并非所有 cells 都可以自由改变。
]


= 12.6 极重要：
<极重要>
#formula-box[$ upright("Correlation does not imply causality") $]

课件例子：

- hospitals 数量
- car theft 数量

可能正相关。

但并不代表：

#quote(block: true)[
hospitals 导致 car theft。
]

真正 third variable： $upright("population")$

同时导致两个变量增大。


= 13. Covariance★★★★★
<covariance>
两个 random variables： $X_1,quad X_2$

的 covariance：

#formula-box[$ sigma_12 = E[(X_1 - mu_1)(X_2 - mu_2)] $]

其中： $mu_1 = E[X_1]$

$ mu_2 = E[X_2] $


== 等价公式★★★★★
<等价公式>
#formula-box[$ sigma_12 = E[X_1 X_2]- E[X_1]E[X_2] $]

推导： $E[(X_1 - mu_1)(X_2 - mu_2)]$

展开：

$ = E[X_1 X_2 - mu_2 X_1 - mu_1 X_2 + mu_1 mu_2] $

$ = E[X_1 X_2]- mu_2 E[X_1]- mu_1 E[X_2]+ mu_1 mu_2 $

== $ = E[X_1 X_2]- mu_1 mu_2 $
<ex_1x_2-mu_1mu_2>
= 13.1 Sample Covariance
<sample-covariance>
课件采用：

#formula-box[$ hat(sigma)_12 = 1 / n sum_(i = 1)^n(x_(i 1) - hat(mu)_1)(x_(i 2) - hat(mu)_2) $]

其中：

- $i$：sample index
- $x_(i 1)$：sample $i$ 的 variable 1
- $x_(i 2)$：sample $i$ 的 variable 2

Variance 是 covariance 的特殊情况：

#formula-box[$ hat(sigma)_11 = 1 / n sum_i(x_(i 1) - hat(mu)_1)^2 $]


== Interpretation
<interpretation>
若： $sigma_12 > 0$

两个变量倾向：

#quote(block: true)[
同向变化。
]

若： $sigma_12 < 0$

倾向：

#quote(block: true)[
反向变化。
]


= 13.2 Covariance = 0 是否 independent？★★★★★
<covariance-0-是否-independent>
非常重要：

#formula-box[$ X_1 perp X_2 arrow.r.double "Cov"(X_1,X_2)= 0 $]

但是反过来：

#formula-box[$ "Cov"(X_1,X_2)= 0 ⇏ X_1 perp X_2 $]

除非加入特殊条件，例如：

#quote(block: true)[
multivariate normal distribution。
]


== p.28 Example：应自己会算★★★★★
<p.28-example应自己会算>
三个 equally likely samples： $X_1 =(1,- 1,- 1)$

$ X_2 =(0,1,- 1) $

则：

$ E[X_1]= frac(1 - 1 - 1, 3) = - 1 / 3 $

$ E[X_2]= frac(0 + 1 - 1, 3) = 0 $

$ E[X_1 X_2]= frac(1(0)+(- 1)(1)+(- 1)(- 1), 3) = 0 $

因此：

$ "Cov"(X_1,X_2)= 0 - (- 1 / 3) 0 = 0 $

但： $P(X_2 = 0 divides X_1 = 1)= 1$

而：

$ P(X_2 = 0)= 1 / 3 $

不相等。

所以： $X_1,X_2$

不 independent。

这是一个非常好的考试概念题。


= 13.3 Stock covariance example
<stock-covariance-example>
数据： $(2,5),(3,8),(5,10),(4,11),(6,14)$

$ E[X_1]= frac(2 + 3 + 5 + 4 + 6, 5) = 4 $

$ E[X_2]= frac(5 + 8 + 10 + 11 + 14, 5) = 9.6 $

$ E[X_1 X_2]= frac(2 dot.op 5 + 3 dot.op 8 + 5 dot.op 10 + 4 dot.op 11 + 6 dot.op 14, 5) $

于是：

$ sigma_12 = E[X_1 X_2]- E[X_1]E[X_2] $

课件得到：

#formula-box[$ sigma_12 = 4 $]

因此： $sigma_12 > 0$

股票价格倾向一起 rise/fall。


= 14. Correlation★★★★★
<correlation>
Covariance 有一个严重问题：

#quote(block: true)[
magnitude 依赖 variables 的 units / scale。
]

于是出现 standardized covariance：

#formula-box[$ rho_12 = frac(sigma_12, sigma_1 sigma_2) $]

其中：

- $sigma_12$：covariance
- $sigma_1$：$X_1$ standard deviation
- $sigma_2$：$X_2$ standard deviation


== Sample correlation
<sample-correlation>
#formula-box[$ hat(rho)_12 = frac(sum_(i = 1)^n(x_(i 1) - hat(mu)_1)(x_(i 2) - hat(mu)_2), sqrt(sum_(i = 1)^n(x_(i 1) - hat(mu)_1)^2) sqrt(sum_(i = 1)^n(x_(i 2) - hat(mu)_2)^2)) $]

范围：

#formula-box[$ - 1 lt.eq rho lt.eq 1 $]

含义：

- $rho > 0$：positive
- $rho < 0$：negative
- $\|rho divides arrow.t$：linear relationship 越强
- $rho = 1$：perfect positive linear
- $rho = - 1$：perfect negative linear

课件对 $rho = 0$ 写：

#quote(block: true)[
independent under the same assumption as covariance discussion。
]

也就是说：

#quote(block: true)[
一般情况下只能说没有 linear correlation；加入如 multivariate normal assumption 后才可推出 independence。
]


= 15. Covariance Matrix★★★★★
<covariance-matrix>
对于二维：

$ upright(bold(X)) = mat(delim: "[", X_1; X_2) $

均值：

$ bold(mu) = mat(delim: "[", mu_1; mu_2) $

covariance matrix：

#formula-box[$ Sigma = E[(upright(bold(X)) - bold(mu))(upright(bold(X)) - bold(mu))^T] $]

二维：

#formula-box[$ Sigma = mat(delim: "[", sigma_1^2, sigma_12; sigma_21, sigma_2^2) $]

由于： $sigma_12 = sigma_21$

所以 covariance matrix：

#formula-box[$ Sigma = Sigma^T $]

是 symmetric matrix。

这会直接连接到 PCA：

#quote(block: true)[
PCA 做的就是 covariance matrix 的 eigendecomposition。
]


= 16. Graphic Statistical Descriptions★★★★☆
<graphic-statistical-descriptions>
课件列：

+ Boxplot
+ Histogram
+ Quantile plot
+ Q-Q plot
+ Scatter plot

它们分别看不同信息。


= 17. Boxplot★★★★★
<boxplot>
Quartiles： $Q_1 = 25^(t h) med p e r c e n t i l e$

$ Q_2 = upright("median") $

$ Q_3 = 75^(t h) med p e r c e n t i l e $

Interquartile range：

#formula-box[$ I Q R = Q_3 - Q_1 $]

Five-number summary：

#formula-box[$ min,Q_1,upright("median"),Q_3,max $]


== Outlier rule
<outlier-rule>
课件说：

#quote(block: true)[
higher/lower than $1.5 times I Q R$。
]

通常理解为：

#formula-box[$ x < Q_1 - 1.5 I Q R $]

或者：

#formula-box[$ x > Q_3 + 1.5 I Q R $]

即视作 potential outlier。


= 18. Histogram★★★★★
<histogram>
Histogram：

- x-axis：value intervals / bins
- y-axis：frequency

用于：

#quote(block: true)[
quantitative distribution。
]

Bar chart：

#quote(block: true)[
categorical comparison。
]


== Histogram vs Bar Chart★★★★★
<histogram-vs-bar-chart>
#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([属性], [Histogram], [Bar chart],),
    table.hline(),
    [数据], [quantitative], [categorical],
    [X-axis], [numeric intervals], [categories],
    [顺序], [不能随便改], [可以 reorder],
    [目标], [distribution], [comparison],
    [bar meaning], [area], [height],
  )]
  , kind: table
  )

课件特别强调：

#quote(block: true)[
如果 bins 宽度不同，histogram 表示 frequency 的是 area，而不是单纯 height。
]


== 课件判断题
<课件判断题>
Histogram：

- student heights
- all ER wait times
- home prices

Bar chart：

- blood types
- top diagnoses
- cities temperatures
- smokers vs non-smokers
- pre-binned age groups
- small integer visit counts


= 19. 为什么 Histogram 有时比 Boxplot 强？★★★★☆
<为什么-histogram-有时比-boxplot-强>
p.38：

两个完全不同 distributions 可能有相同： $min,Q_1,upright("median"),Q_3,max$

所以 boxplot 一样。

但 histogram 可以显示：

- multimodality
- peaks
- gaps
- shape

因此：

```text
Boxplot
→ compact robust summary
→ loses distribution shape

Histogram
→ keeps much more shape information
```


= 20. Quantile Plot★★★★☆
<quantile-plot>
排序： $x_1 lt.eq x_2 lt.eq dots.h.c lt.eq x_n$

每个： $x_i$

对应： $f_i$

表示大约： $100 f_i %$

数据满足： $x lt.eq x_i$

本质：

#quote(block: true)[
empirical CDF 的一种表现。
]

优点：

#quote(block: true)[
displays all data，因此既能看 overall behavior，也能看到 unusual values。
]


= 21. Q-Q Plot★★★★★
<q-q-plot>
把 distribution A 的 quantiles 与 distribution B 的对应 quantiles 作图：

$ Q_A(p)quad upright("vs") quad Q_B(p) $

如果 distributions 很接近：

#quote(block: true)[
points approximately lie on a straight line。
]

课件用途：

#quote(block: true)[
看两个 distributions 是否存在 shift。
]

p.40 例：

Branch 1 的 unit price 通常小于 Branch 2。

p.41 又显示：

- normal data vs normal distribution → 接近直线
- exponential data vs normal → 明显弯曲

因此 Q-Q plot 很适合判断：

#quote(block: true)[
empirical distribution 是否接近某 theoretical distribution。
]


= 22. Scatter Plot★★★★★
<scatter-plot>
每个 observation： $(x_i,y_i)$

作为平面 point。

用来检查：

- correlation
- cluster
- outlier
- nonlinear structure


== 非常重要的图示 p.43--44
<非常重要的图示-p.4344>
p.43：

数据左半部分 positive correlation，右半部分 negative correlation。

整个数据可能：

#quote(block: true)[
linear correlation 接近 0。
]

p.44 展示多个"uncorrelated"点云。

结论：

#formula-box[$ rho approx 0 eq.not upright("no structure") $]

Correlation 本质主要描述：

#quote(block: true)[
linear dependence。
]


= 23. Similarity / Distance：为什么不同类型数据需要不同距离？★★★★★
<similarity-distance为什么不同类型数据需要不同距离>
逻辑：

```text
Data mining 经常要回答：
两个 objects 是否相似？
↓
Numeric objects?
Euclidean / Minkowski
↓
Binary?
Matching / Jaccard
↓
Categorical?
Simple matching
↓
Ordinal?
Rank first
↓
Vectors?
Cosine
↓
Probability distributions?
KL divergence
```

这也是本章非常重要的一条"方法演进链"。


= 24. Similarity / Dissimilarity / Proximity★★★★★
<similarity-dissimilarity-proximity>
== Similarity
<similarity>
函数： $s(i,j)$

越大：

#quote(block: true)[
越相似。
]

常见： $0 lt.eq s(i,j)lt.eq 1$

其中：

- 0：not similar
- 1：identical / maximally similar


== Dissimilarity / Distance
<dissimilarity-distance>
$ d(i,j) $

越小：

#quote(block: true)[
越相似。
]

通常： $d(i,i)= 0$

range 可能： $[0,1]$

或者： $[0,oo)$


== Proximity
<proximity>
是 umbrella term，可以表示：

- similarity
- dissimilarity


= 25. Data Matrix vs Dissimilarity Matrix★★★★★
<data-matrix-vs-dissimilarity-matrix>
Data matrix：

$ D = mat(delim: "[", x_11, x_12, dots.h.c, x_(1 l); x_21, x_22, dots.h.c, x_(2 l); dots.v, dots.v, dots.down, dots.v; x_(n 1), x_(n 2), dots.h.c, x_(n l)) $

shape： $n times l$

其中：

- $n$：objects
- $l$：dimensions


Dissimilarity matrix： $Delta_(i j) = d(i,j)$

一般：

$ Delta = mat(delim: "[", 0, dots.h.c; d(2,1), 0, ; dots.v, dots.v, dots.down) $

如果 distance symmetric： $d(i,j)= d(j,i)$

那么只存 triangular half 即可。


= 26. Minkowski Distance★★★★★
<minkowski-distance>
Numeric data 的统一 family：

#formula-box[$ d(i,j)= (sum_(f = 1)^l \| x_(i f) - x_(j f) \|^p)^(1/p) $]

其中：

- $i,j$：两个 objects
- $f$：attribute index
- $l$：dimensions
- $p$：distance order

也称： $L_p$

norm distance。


= 27. Metric 三大性质★★★★★
<metric-三大性质>
一个 distance 若满足：

=== Positivity
<positivity>
$ d(i,j)> 0 quad(i eq.not j) $

且： $d(i,i)= 0$

=== Symmetry
<symmetry>
$ d(i,j)= d(j,i) $

=== Triangle inequality
<triangle-inequality>
#formula-box[$ d(i,j)lt.eq d(i,k)+ d(k,j) $]

则属于 metric。

注意：

#quote(block: true)[
dissimilarity 不一定必须是 metric。
]

课件例：

set differences 可以 nonmetric。


= 28. Minkowski Special Cases★★★★★
<minkowski-special-cases>
== $p = 1$：Manhattan
<p1manhattan>
#formula-box[$ d(i,j)= sum_f divides x_(i f) - x_(j f)\| $]

名称：

- Manhattan distance
- City-block distance
- $L_1$


=== Hamming Distance
<hamming-distance>
Binary vectors 上：

#quote(block: true)[
differing bits 数量。
]

例如： $101101$

与

$ 100001 $

difference positions 数量即 Hamming distance。


== $p = 2$：Euclidean
<p2euclidean>
#formula-box[$ d(i,j)= sqrt(sum_f(x_(i f) - x_(j f))^2) $]

即 $L_2$。


== $p arrow.r oo$：Chebyshev
<prightarrowinftychebyshev>
#formula-box[$ d(i,j)= max_f\|x_(i f) - x_(j f)\| $]

也叫：

- supremum distance
- $L_oo$


= 29. Minkowski example★★★★★
<minkowski-example>
数据： $x_1 =(1,2)$

$ x_2 =(3,5) $

$ x_3 =(2,0) $

$ x_4 =(4,5) $

例如：

$ L_1(x_1,x_2)=\|1 - 3\|+\|2 - 5\|= 5 $

$ L_2(x_1,x_2)= sqrt((1 - 3)^2+(2 - 5)^2) = sqrt(13) approx 3.61 $

$ L_oo(x_1,x_2)= max(2,3)= 3 $

课件完整 distance matrices：

=== $L_1$
<l_1>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,right,right,right,right,),
    table.header([], [x1], [x2], [x3], [x4],),
    table.hline(),
    [x1], [0], [], [], [],
    [x2], [5], [0], [], [],
    [x3], [3], [6], [0], [],
    [x4], [6], [1], [7], [0],
  )]
  , kind: table
  )

=== $L_2$
<l_2>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,right,right,right,right,),
    table.header([], [x1], [x2], [x3], [x4],),
    table.hline(),
    [x1], [0], [], [], [],
    [x2], [3.61], [0], [], [],
    [x3], [2.24], [5.10], [0], [],
    [x4], [4.24], [1], [5.39], [0],
  )]
  , kind: table
  )

=== $L_oo$
<l_infty>
#figure(
  align(center)[#table(
    columns: 5,
    align: (auto,right,right,right,right,),
    table.header([], [x1], [x2], [x3], [x4],),
    table.hline(),
    [x1], [0], [], [], [],
    [x2], [3], [0], [], [],
    [x3], [2], [5], [0], [],
    [x4], [3], [1], [5], [0],
  )]
  , kind: table
  )


= 30. Binary Attribute Proximity★★★★★
<binary-attribute-proximity>
考虑两个 objects $i,j$。

定义 contingency：

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,right,right,),
    table.header([], [$j = 1$], [$j = 0$],),
    table.hline(),
    [$i = 1$], [$q$], [$r$],
    [$i = 0$], [$s$], [$t$],
  )]
  , kind: table
  )

含义：

- $q$：both 1
- $r$：$i = 1,j = 0$
- $s$：$i = 0,j = 1$
- $t$：both 0


== Symmetric Binary
<symmetric-binary-1>
0 和 1 同等重要。

matches： $q + t$

mismatches： $r + s$

所以：

#formula-box[$ d(i,j)= frac(r + s, q + r + s + t) $]


== Asymmetric Binary★★★★★
<asymmetric-binary-1>
0--0 不重要。

因此忽略： $t$

得到：

#formula-box[$ d(i,j)= frac(r + s, q + r + s) $]


= 31. Jaccard Similarity★★★★★
<jaccard-similarity>
对 asymmetric binary：

#formula-box[$ s i m_(J a c c a r d)(i,j)= frac(q, q + r + s) $]

直觉：

$ upright("common presence") / upright("presence in either object") $

实际上：

$ J(A,B)= frac(\|A inter B\|, \|A union B\|) $

课件指出与后面 Pattern Discovery 中：

#quote(block: true)[
coherence
]

概念一致。


= 32. Asymmetric Binary Example★★★★★
<asymmetric-binary-example>
课件数据：

#figure(
  align(center)[#table(
    columns: 8,
    align: (auto,auto,auto,auto,auto,auto,auto,auto,),
    table.header([Name], [Gender], [Fever], [Cough], [Test1], [Test2], [Test3], [Test4],),
    table.hline(),
    [Jack], [M], [Y], [N], [P], [N], [N], [N],
    [Mary], [F], [Y], [N], [P], [N], [P], [N],
    [Jim], [M], [Y], [P], [N], [N], [N], [N],
  )]
  , kind: table
  )

Gender 为 symmetric：

#quote(block: true)[
本例不计。
]

其余： $Y,P arrow.r 1$

$ N arrow.r 0 $

结果：

$ d(upright("Jack,Mary"))= frac(0 + 1, 2 + 0 + 1) = 0.33 $

$ d(upright("Jack,Jim"))= frac(1 + 1, 1 + 1 + 1) = 0.67 $

$ d(upright("Jim,Mary"))= frac(1 + 2, 1 + 1 + 2) = 0.75 $

因此：

Jack 与 Mary 最相似。


= 33. Categorical / Nominal Proximity★★★★★
<categorical-nominal-proximity>
== Simple Matching
<simple-matching>
假设：

- $p$：attributes 总数
- $m$：matching attributes 数

则：

#formula-box[$ d(i,j)= frac(p - m, p) $]

即：

$ frac(\#m i s m a t c h e s, \#a t t r i b u t e s) $


== One-hot expansion
<one-hot-expansion>
若 Color： ${ upright("red,yellow,blue,green") }$

展开：

$ [i s\_r e d,i s\_y e l l o w,i s\_b l u e,i s\_g r e e n] $

例如： $r e d arrow.r[1,0,0,0]$

这是：

#quote(block: true)[
nominal → binary representation。
]


= 34. Ordinal Variable Distance★★★★★
<ordinal-variable-distance>
Ordinal 有 order，但没有 numerical spacing。

解决：

=== Step 1：转换为 rank
<step-1转换为-rank>
假设 variable $f$ 有： $M_f$

levels。

第 $i$ 个 object rank： $r_(i f) in { 1,dots.h,M_f }$

=== Step 2：映射到 $[0,1]$
<step-2映射到-01>
#formula-box[$ z_(i f) = frac(r_(i f) - 1, M_f - 1) $]

例如：

- freshman → 0
- sophomore → $1/3$
- junior → $2/3$
- senior → 1

于是： $d(upright("freshman,senior"))= 1$

$ d(upright("junior,senior"))= 1 / 3 $

之后把 $z$：

#quote(block: true)[
当作 interval-scaled variable。
]

=== 为什么不是 ratio-scaled？
<为什么不是-ratio-scaled>
因为： $z = 0$

只是 rank 最低点，不表示"该属性不存在"。

没有 intrinsic true zero。

所以只能解释 differences，不能解释 ratio。


= 35. Mixed-Type Attributes★★★★★
<mixed-type-attributes>
现实数据可能同时包括：

- nominal
- symmetric binary
- asymmetric binary
- numeric
- ordinal

因此每个 attribute 单独计算 dissimilarity，再 combine。

课件公式：

#formula-box[$ d(i,j)= frac(sum_(f = 1)^p w_(i j)^((f)) d_(i j)^((f)), sum_(f = 1)^p w_(i j)^((f))) $]

其中：

- $f$：attribute index
- $p$：attribute number
- $d_(i j)^((f))$：attribute $f$ 上的 distance
- $w_(i j)^((f))$：该 attribute 权重 / 是否参与计算

Binary / nominal：

$ d_(i j)^((f)) = cases(delim: "{", 0, & x_(i f) = x_(j f), 1, & x_(i f) eq.not x_(j f)) $

这条"相同即 0"的写法适用于 nominal 与 #strong[symmetric] binary。对于 asymmetric binary，两个对象同时为 0 表示共同不出现，通常不应算作 match：令该维度 $w_(i j)^((f)) = 0$，从分子和分母一并排除。

Ordinal：

先：

$ z_(i f) = frac(r_(i f) - 1, M_f - 1) $

再按 interval-scaled variable 处理。

Numeric：

课件只写：

#quote(block: true)[
Use the normalized distance
]

但该页没有显式给公式。

标准 range-normalized difference 通常写作：

$ d_(i j)^((f)) = frac(\|x_(i f) - x_(j f)\|, max_h x_(h f) - min_h x_(h f)) $

这是理解补充，不是这一页明确打印出的公式。


= 36. Cosine Similarity★★★★★
<cosine-similarity>
为什么需要 cosine？

Document term-frequency vectors 通常：

- high-dimensional
- sparse
- document length 不一致

Euclidean distance 会受 magnitude 影响。

Cosine 只比较：

#quote(block: true)[
directions / angles。
]


== Formula
<formula>
#formula-box[$ cos(d_1,d_2)= frac(d_1 dot.op d_2, parallel d_1 parallel parallel d_2 parallel) $]

其中：

$ d_1 dot.op d_2 = sum_j d_(1 j) d_(2 j) $

$ parallel d parallel = sqrt(sum_j d_j^2) $


= 36.1 Cosine Example★★★★★
<cosine-example>
$ d_1 =(5,0,3,0,2,0,0,2,0,0) $

$ d_2 =(3,0,2,0,1,1,0,1,0,1) $

dot product： $d_1 dot.op d_2$

$ = 5(3)+ 3(2)+ 2(1)+ 2(1) $

$ = 15 + 6 + 2 + 2 $

#formula-box[$ = 25 $]

课件给出的数值有一处算术错误： $parallel d_1 parallel = 6.481$

$ parallel d_2 parallel = 4 $

因此：

$ cos(d_1,d_2)= frac(25, sqrt(42) dot.op 4) $

#formula-box[$ approx 0.965 $]

高度相似。

课件特意强调：

#quote(block: true)[
vector length ≠ vector dimension。
]

$parallel d parallel$ 指 magnitude，不是 component 数量。


= 37. KL Divergence★★★★★
<kl-divergence>
前面的：

- Euclidean
- Jaccard
- cosine

主要比较 objects / vectors。

如果比较的是：

#quote(block: true)[
probability distributions
]

则引出 KL divergence。


= 37.1 Discrete KL
<discrete-kl>
#formula-box[$ D_(K L)(P parallel Q)= sum_(x in X) p(x)ln frac(p(x), q(x)) $]


= 37.2 Continuous KL
<continuous-kl>
#formula-box[$ D_(K L)(P parallel Q)= integral_(- oo)^(+ oo) p(x)ln frac(p(x), q(x)) thin d x $]

解释： $D_(K L)(P parallel Q)$

衡量：

#quote(block: true)[
用 $Q$ approximation 表示真实 $P$ 时损失的信息。
]

通常：

- $P$：true / observed distribution
- $Q$：model / approximation / theory


= 37.3 Information-theoretic interpretation★★★★★
<information-theoretic-interpretation>
KL divergence：

#quote(block: true)[
使用基于 $Q$ 的编码，而真实 samples 来自 $P$ 时，平均需要额外多少 information。
]

课件也称：

- relative entropy
- information divergence
- information gain


= 37.4 KL 不是 metric★★★★★
<kl-不是-metric>
因为：

#formula-box[$ D_(K L)(P parallel Q)eq.not D_(K L)(Q parallel P) $]

通常 asymmetric。

也不满足：

#quote(block: true)[
triangle inequality。
]

因此：

#formula-box[$ K L med d i v e r g e n c e eq.not d i s t a n c e med m e t r i c $]

虽然常用于度量 distributions difference。


= 37.5 KL Non-negativity★★★★★
<kl-non-negativity>
课件：

#formula-box[$ D_(K L)(P parallel Q)gt.eq 0 $]

而：

#formula-box[$ D_(K L)(P parallel Q)= 0 arrow.l.r.double P = Q $]


= 37.6 $p = 0$ 或 $q = 0$ 怎么办？★★★★★
<p0-或-q0-怎么办>
若： $p arrow.r 0$

则：

#formula-box[$ lim_(p arrow.r 0) p log p = 0 $]

所以 $p(x)= 0$ 项贡献为 0。


但若： $p(x)> 0$

且： $q(x)= 0$

则：

$ log frac(p(x), 0) arrow.r oo $

因此：

#formula-box[$ D_(K L)(P parallel Q)= oo $]

直觉：

#quote(block: true)[
P 说 event 可能发生，Q 却声称绝对不可能，因此 approximation catastrophically wrong。
]


= 37.7 Smoothing★★★★★
<smoothing>
现实 frequency distribution 可能只是：

#quote(block: true)[
没见过某 event，而不是 event probability 真的是 0。
]

因此加入： $epsilon.alt$

例如： $epsilon.alt = 10^(- 3)$

课件：

$ P :(a : 3/5,b : 1/5,c : 1/5) $

$ Q :(a : 5/9,b : 3/9,d : 1/9) $

supports： $S_P = { a,b,c }$

$ S_Q = { a,b,d } $

union： $S_U = { a,b,c,d }$

平滑后：

$ P' : (a : 3 / 5 - epsilon.alt / 3 , b : 1 / 5 - epsilon.alt / 3 , c : 1 / 5 - epsilon.alt / 3 , d : epsilon.alt) $

$ Q' : (a : 5 / 9 - epsilon.alt / 3 , b : 3 / 9 - epsilon.alt / 3 , c : epsilon.alt , d : 1 / 9 - epsilon.alt / 3) $

之后： $D_(K L)(P' parallel Q')$

即可有限计算。


= 38. 为什么这些 similarity measures 还不够？★★★★☆
<为什么这些-similarity-measures-还不够>
课件 p.63 是从 classical data mining 向 representation learning 的一个关键转折。

传统 vector similarity 最大问题：

#quote(block: true)[
它只看到表面 representation，不理解 semantics。
]

例如：

#quote(block: true)[
The cat bites a mouse.
]

和：

#quote(block: true)[
The mouse bites a cat.
]

Bag-of-words 几乎相同。

但语义完全不同。

又如：

- geometry
- algebra
- music
- politics

仅靠 literal overlap 不一定能知道 geometry 与 algebra 更相关。

而 graph/network 更存在：

#quote(block: true)[
structure + connection semantics。
]

所以自然演进：

```text
Hand-crafted similarity
↓
cannot model hidden semantics
↓
distributed representation
↓
representation learning
```


= 39. Data Preprocessing：为什么要做？★★★★★
<data-preprocessing为什么要做>
主要 tasks：

+ Data cleaning
+ Data integration
+ Data reduction
+ Data transformation
+ Data discretization

具体：

=== Cleaning
<cleaning>
- missing
- noise
- outliers
- inconsistency

=== Integration
<integration>
- databases
- cubes
- files

=== Reduction
<reduction>
- dimensionality reduction
- numerosity reduction
- compression

=== Transformation
<transformation>
- normalization
- discretization
- concept hierarchy


= 40. Data Quality Dimensions★★★★★
<data-quality-dimensions>
课件 p.66 六个指标：

== Accuracy
<accuracy>
数据是否正确？

== Completeness
<completeness>
是否缺失？

== Consistency
<consistency>
不同 records / sources 是否互相矛盾？

== Timeliness
<timeliness>
是否及时更新？

== Believability
<believability>
是否可信？

== Interpretability
<interpretability>
是否容易理解？

注意这说明：

#quote(block: true)[
Data quality 不是单一"accuracy"。
]

而是 multidimensional concept。


= 41. Dirty Data★★★★★
<dirty-data>
现实数据可能：

== Incomplete
<incomplete>
例如： \$Occupation=\"\"\$

== Noisy
<noisy>
例如： $S a l a r y = - 10$

== Inconsistent
<inconsistent>
例如： $A g e = 42$

但是： $B i r t h d a y = 03/07/2010$

两者可能矛盾。

也可能 schema change：

```text
1,2,3
```

变成：

```text
A,B,C
```

== Duplicate records
<duplicate-records>
同一个 entity 多条记录不同。

== Intentional errors
<intentional-errors>
例如：

#quote(block: true)[
所有人生日都写 January 1。
]

可能是：

#quote(block: true)[
disguised missing data。
]


= 42. Missing Data★★★★★
<missing-data>
可能原因：

- equipment malfunction
- conflicting data deleted
- data entry misunderstanding
- 当时认为不重要
- 没有保存 history/change

因此：

#quote(block: true)[
Missing data 不只是随机缺失。
]


= 43. Missing Data Handling：方法演进★★★★★
<missing-data-handling方法演进>
== 方法 1：Ignore tuple
<方法-1ignore-tuple>
适合：

#quote(block: true)[
classification 中 class label missing。
]

问题：

如果不同 attributes missing rates 差异很大：

#quote(block: true)[
丢整条 row 会浪费大量有效信息。
]


== 方法 2：Manual filling
<方法-2manual-filling>
优点：

#quote(block: true)[
potentially accurate。
]

问题：

#quote(block: true)[
tedious + infeasible。
]


== 方法 3：Global constant
<方法-3global-constant>
例如： $upright("\"unknown\"")$

问题：

"unknown"可能被 algorithm 当成一个新的真实 class。


== 方法 4：Attribute Mean
<方法-4attribute-mean>
$ x_(upright("missing")) arrow.l macron(x) $

问题：

会：

#quote(block: true)[
shrink variance / distort distribution。
]

课件未展开数学副作用，但直觉要知道。


== 方法 5：Class-specific Mean
<方法-5class-specific-mean>
如果 class 已知：

$ x_(upright("missing")) arrow.l macron(x)_(upright("class")) $

比 global mean 更合理，因为使用了：

#quote(block: true)[
conditional information。
]


== 方法 6：Most Probable Value
<方法-6most-probable-value>
通过：

- Bayesian formula
- decision tree

infer missing value。

演进逻辑：

```text
Delete
↓
太浪费
Global filling
↓
不利用上下文
Class mean
↓
利用 label
Probabilistic inference
↓
利用更多 dependency structure
```


= 44. Noise★★★★☆
<noise>
Noise：

#quote(block: true)[
random error / variance in measured variable。
]

原因：

- faulty instruments
- data entry
- transmission
- technology limitation
- inconsistent naming
- duplicates
- incomplete
- inconsistent data


= 45. Handling Noise★★★★★
<handling-noise>
== 1. Binning
<binning>
流程：

$ s o r t arrow.r p a r t i t i o n arrow.r s m o o t h $

smooth by：

- bin mean
- bin median
- bin boundary


== 2. Regression
<regression>
用 regression function： $hat(y) = f(x)$

代替 noisy observations。


== 3. Clustering
<clustering>
outliers 往往：

#quote(block: true)[
远离 major clusters。
]

因此 detect/remove outlier。


== 4. Semi-supervised inspection
<semi-supervised-inspection>
Computer：

#quote(block: true)[
detect suspicious values。
]

Human：

#quote(block: true)[
verify。
]

这体现：

#quote(block: true)[
自动检测效率 + domain expert judgment。
]


= 46. Data Cleaning as a Process★★★★☆
<data-cleaning-as-a-process>
不是一次 function call，而是 iterative workflow。

== Discrepancy detection
<discrepancy-detection>
使用 metadata：

- domain
- range
- dependency
- distribution

检查：

- field overloading
- uniqueness
- consecutive rule
- null rule


== Data Scrubbing
<data-scrubbing>
使用简单 domain knowledge：

- postal codes
- spell checking

detect + correct errors。


== Data Auditing
<data-auditing>
分析数据发现：

- rules
- relationships

然后寻找 violators。

例如：

- correlation
- clustering
- outlier detection


== Migration + Integration
<migration-integration>
ETL：

#formula-box[$ E x t r a c t i o n arrow.r T r a n s f o r m a t i o n arrow.r L o a d i n g $]

整个 process：

#quote(block: true)[
iterative + interactive。
]


= 47. Data Integration★★★★★
<data-integration>
定义：

#quote(block: true)[
combining multiple data sources into a coherent store。
]

为什么？

+ reduce/avoid noise
+ more complete picture
+ improve mining speed
+ improve mining quality


= 47.1 Schema Integration
<schema-integration>
例如：

$ A . c u s t\_i d equiv B . c u s t\_\# $

名字不同：

#quote(block: true)[
semantic entity 相同。
]


= 47.2 Entity Identification
<entity-identification>
例如：

$ upright("Bill Clinton") = upright("William Clinton") $

需要识别：

#quote(block: true)[
两条记录是否代表同一个现实对象。
]


= 48. Conflicts in Integration★★★★★
<conflicts-in-integration>
不同 sources 对同一个 entity 可能给不同 value。

原因：

- representation difference
- timestamp difference
- scale/unit difference

例如： $upright("metric vs British units")$


== Conflict resolution
<conflict-resolution>
可以：

- mean
- median
- mode
- max
- min
- most recent
- truth finding

Truth finding：

#quote(block: true)[
根据 source quality 估计哪个 source 更可信。
]


= 49. Redundancy in Integration★★★★★
<redundancy-in-integration>
来源：

== Same object different names
<same-object-different-names>
== Derived attributes
<derived-attributes>
例如 annual revenue 可以由 monthly revenue 得到。


== 为什么 redundancy 是问题？
<为什么-redundancy-是问题>
课件给出一组很值得理解的式子： $Y = 2 X$

若： $X_1 = X_2 = X$

那么同一个 relationship 可以写： $Y = X_1 + X_2$

也可以： $Y = 3 X_1 - X_2$

甚至： $Y = - 1291 X_1 + 1293 X_2$

因为： $- 1291 X + 1293 X = 2 X$

所以存在大量 parameter combinations。

本质：

#quote(block: true)[
attributes linearly redundant → model coefficients become non-identifiable / unstable。
]

这就是 multicollinearity 的直觉。

因此 redundancy 可通过：

- covariance
- correlation

来发现。


= 50. Data Transformation★★★★★
<data-transformation>
Transformation：

#quote(block: true)[
把一个 attribute 的原始 value set 映射到新的 replacement values。
]

方法：

+ smoothing
+ attribute construction
+ aggregation
+ normalization
+ discretization


= 51. Normalization★★★★★
<normalization>
目的：

#quote(block: true)[
不同 variables 尺度差异太大时，distance / optimization 可能被大尺度 attribute 主导。
]

例如：

```text
Age:       18–80
Income: 10000–1000000
```

Euclidean distance 中 Income 会主导。

因此 normalize。


= 51.1 Min-Max Normalization★★★★★
<min-max-normalization>
将原区间： $[m i n_A,m a x_A]$

映射到： $[n e w\_m i n_A,n e w\_m a x_A]$

公式：

#formula-box[$ v' = frac(v - m i n_A, m a x_A - m i n_A)(n e w\_m a x_A - n e w\_m i n_A)+ n e w\_m i n_A $]

=== 推导直觉
<推导直觉-1>
第一步：

$ frac(v - m i n_A, m a x_A - m i n_A) $

将位置变成： $[0,1]$

第二步 scale： $times(n e w\_m a x_A - n e w\_m i n_A)$

第三步 shift： $+ n e w\_m i n_A$


== 课件例子有一个数值不一致★★★★★
<课件例子有一个数值不一致>
文字写：

#quote(block: true)[
\$73,000
]

但公式实际使用： $73,600$

因为：

$ frac(73600 - 12000, 98000 - 12000) = 0.716 $

课件结果： $0.716$

确实对应 $73,600$，而不是 $73,000$。

如果真是： $73000$

则结果约： $0.709$

所以这是 slide 中值得注意的 typo / inconsistency。


= 51.2 Z-score Normalization★★★★★
<z-score-normalization>
#formula-box[$ v' = frac(v - mu_A, sigma_A) $]

其中：

- $mu_A$：attribute mean
- $sigma_A$：standard deviation

意义：

#quote(block: true)[
raw score 与 mean 相差多少个 standard deviations。
]

若： $v' = 1.5$

意味着：

#quote(block: true)[
高于 mean 1.5 SD。
]

课件例： $mu = 54000$

$ sigma = 16000 $

用实际 equation 中的： $v = 73600$

则：

$ v' = frac(73600 - 54000, 16000) $

#formula-box[$ = 1.225 $]


= 51.3 Decimal Scaling
<decimal-scaling>
#formula-box[$ v' = v / 10^j $]

其中 $j$ 为满足：

#formula-box[$ max(\|v'\|)< 1 $]

的最小 integer。

例如最大 absolute value 为 987： $j = 3$

因为： $987/1000 = 0.987 < 1$


= 52. Discretization★★★★★
<discretization>
定义：

#quote(block: true)[
Divide continuous attribute range into intervals and replace raw values by interval labels。
]

例如： $a g e = 27$

变： $a g e = upright("young adult")$

作用：

+ reduce data size
+ simplify analysis
+ prepare classification
+ concept hierarchy


== 三个分类轴★★★★★
<三个分类轴>
=== Supervised vs Unsupervised
<supervised-vs-unsupervised>
是否使用 labels。

=== Split vs Merge
<split-vs-merge>
Top-down：

```text
whole interval
↓
split
↓
smaller intervals
```

Bottom-up：

```text
many small intervals
↓
merge
↓
larger intervals
```

=== Recursive vs one-step
<recursive-vs-one-step>
可以反复进行。


= 53. Discretization Methods★★★★★
<discretization-methods>
课件 p.80：

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([Method], [Supervision], [Direction],),
    table.hline(),
    [Binning], [Unsupervised], [Top-down],
    [Histogram], [Unsupervised], [Top-down],
    [Clustering], [Unsupervised], [Top-down or bottom-up],
    [Decision tree], [Supervised], [Top-down],
    [Correlation / $chi^2$], [slide p.80 写 Unsupervised], [Bottom-up],
  )]
  , kind: table
  )

但是 p.84 对 Chi-merge 又明确写：

#quote(block: true)[
Supervised: use class information。
]

因为 Chi-merge 比较 neighboring intervals 的：

#quote(block: true)[
class distributions。
]

因此课件内部这里存在不一致。

从 p.84 的具体机制看：

#formula-box[$ upright("Chi-merge 是 supervised") $]

更符合其描述。

考试若老师严格按 PPT，建议记住这一处矛盾。


= 54. Equal-width Binning★★★★★
<equal-width-binning>
设：

- minimum = $A$
- maximum = $B$
- number of bins = $N$

则：

#formula-box[$ W = frac(B - A, N) $]

每个 interval width 相同。

优点：

#quote(block: true)[
straightforward。
]

问题：

=== Outliers dominate
<outliers-dominate>
若存在极端大值： $B gt.double t y p i c a l med v a l u e s$

绝大多数数据可能挤在前几个 bins。

=== Skewed data
<skewed-data>
也处理不好。


= 55. Equal-depth / Equal-frequency★★★★★
<equal-depth-equal-frequency>
每个 bin 包含大致相同： $\#s a m p l e s$

优点：

#quote(block: true)[
better scaling for skewed data。
]

问题：

#quote(block: true)[
interval widths 会不同，而且 categorical handling tricky。
]


= 56. Binning Smoothing Example★★★★★
<binning-smoothing-example>
原始排序数据： $4,8,9,15,21,21,24,25,26,28,29,34$

Equal-depth：

=== Bin 1
<bin-1>
$ 4,8,9,15 $

=== Bin 2
<bin-2>
$ 21,21,24,25 $

=== Bin 3
<bin-3>
$ 26,28,29,34 $


== Smooth by bin means
<smooth-by-bin-means>
Bin 1 mean：

$ frac(4 + 8 + 9 + 15, 4) = 9 $

变： $9,9,9,9$

Bin 2 approximately： $23,23,23,23$

Bin 3： $29,29,29,29$


== Smooth by boundaries
<smooth-by-boundaries>
Bin 1 boundaries： $4,quad 15$

每个 point 替换为 nearest boundary： $4,4,15,15$

因为 $8$ 距 $4$ 更近，而 $9$ 距 $15$ 更近。课件此处把 $9$ 也替换为 $4$，是例题笔误。

Bin 2： $21,21,25,25$

Bin 3： $26,26,26,34$


= 57. Binning → Clustering 的演进★★★★☆
<binning-clustering-的演进>
课件 p.83：

Equal-width：

#quote(block: true)[
boundaries fixed by value range。
]

Equal-depth：

#quote(block: true)[
boundaries fixed by frequency。
]

二者共同问题：

#formula-box[$ upright("inflexible") $]

它们不真正理解 natural groups。

于是：

#quote(block: true)[
K-means 等 clustering 可以根据 actual data structure 决定 partition。
]

方法演进：

```text
Equal-width
↓
fails on skew/outliers
Equal-depth
↓
better population balance, but still rigid
Clustering
↓
data-driven boundaries
```


= 58. Supervised Discretization★★★★★
<supervised-discretization>
== Decision Tree
<decision-tree>
给 class labels。

例如： $upright("cancerous vs benign")$

利用：

#quote(block: true)[
entropy
]

选择 split point。

特点：

- supervised
- top-down
- recursive


== Chi-Merge
<chi-merge>
利用： $chi^2$

比较 neighboring intervals 的 class distributions。

若两个 intervals class distribution 很相近：

$ chi^2 upright(" small") $

则 merge。

不断 merge：

#quote(block: true)[
until stopping condition。
]

特点：

#formula-box[$ s u p e r v i s e d + b o t t o m upright("-") u p $]


= 59. Concept Hierarchy★★★★☆
<concept-hierarchy>
目的：

#quote(block: true)[
从细粒度 concepts 推到粗粒度 concepts。
]

例如：

$ 27 arrow.r a d u l t $

或者：

$ s t r e e t < c i t y < s t a t e < c o u n t r y $

支持 data warehouse 的：

- drill-down
- roll-up

多粒度分析。


= 60. Nominal Concept Hierarchy
<nominal-concept-hierarchy>
四种方式：

=== Explicit total/partial ordering
<explicit-totalpartial-ordering>
$ s t r e e t < c i t y < s t a t e < c o u n t r y $

=== Explicit grouping
<explicit-grouping>
$ { U r b a n a,C h a m p a i g n,C h i c a g o } < I l l i n o i s $

=== Partial hierarchy only
<partial-hierarchy-only>
例如只知道： $s t r e e t < c i t y$

=== Automatic generation
<automatic-generation>
根据：

#quote(block: true)[
distinct value count。
]


= 61. Automatic Concept Hierarchy★★★★☆
<automatic-concept-hierarchy>
规则：

#quote(block: true)[
distinct values 越多，通常层级越低。
]

课件： $c o u n t r y : 15$

$ s t a t e : 365 $

$ c i t y : 3567 $

$ s t r e e t : 674339 $

所以：

```text
country
  ↓
state
  ↓
city
  ↓
street
```

但有 exceptions：

- weekday
- month
- quarter
- year

不能机械只看 distinct counts。


= 62. Data Compression★★★☆☆
<data-compression>
== String compression
<string-compression>
通常：

#quote(block: true)[
lossless。
]

可以完整恢复原数据。

但不 decompress：

#quote(block: true)[
操作能力有限。
]


== Audio / Video
<audio-video>
通常：

#quote(block: true)[
lossy。
]

允许 approximation。

课件指出：

#quote(block: true)[
progressive refinement。
]


== Time sequence ≠ audio
<time-sequence-audio>
Time sequence 通常：

- shorter
- slowly varying

需要不同处理。


Data reduction / dimensionality reduction：

#quote(block: true)[
也可看作 compression。
]


= 63. Sampling★★★★★
<sampling>
目标：

从完整数据： $N$

抽出： $s$

使它代表 whole dataset。

优点：

#quote(block: true)[
mining complexity 可以变成 sub-linear in $N$。
]

核心：

#formula-box[$ upright("representative subset") $]


= 63.1 Simple Random Sampling
<simple-random-sampling>
每个 item 有相同 selection probability。


== Without replacement
<without-replacement>
被选一次后：

#quote(block: true)[
remove from population。
]

不会重复。


== With replacement
<with-replacement>
被选后：

#quote(block: true)[
仍留在 population。
]

因此可能重复。

课件标了"why?"。

重要直觉：

#quote(block: true)[
有放回抽样使不同 draw 可以近似看作 independent，是 bootstrap 等统计方法的基础。
]

这部分是解释补充。


= 63.2 Stratified Sampling★★★★★
<stratified-sampling>
先 partition： $D = D_1 union D_2 union dots.h.c union D_k$

然后每个 stratum 分别 sample。

通常按比例：

$ n_h approx n N_h / N $

作用：

#quote(block: true)[
保留 minority / skewed groups。
]

为什么需要？

课件明确指出：

#quote(block: true)[
Simple random sampling 在 skewed data 上可能非常差。
]


== 一个容易忽略的点
<一个容易忽略的点>
课件：

#quote(block: true)[
Sampling may not reduce database I/Os.
]

因为 database 常：

#quote(block: true)[
page at a time。
]

即使只要一个 tuple，也可能必须读整 page。


= 64. Data Reduction★★★★★
<data-reduction>
目标：

获得： $D'$

满足： $\|D'\|lt.double divides D\|$

但：

$ A n a l y s i s(D')approx A n a l y s i s(D) $

为什么？

因为 database 可能：

#quote(block: true)[
terabytes。
]

完整 mining 时间非常长。


= 65. Parametric vs Non-parametric Reduction★★★★★
<parametric-vs-non-parametric-reduction>
== Parametric
<parametric>
假设 data obey 某 model： $D approx f_theta$

然后只存： $theta$

而不是 raw data。

典型：

- regression
- log-linear models

优点：

#quote(block: true)[
compression ratio 高。
]

问题：

#quote(block: true)[
如果 model assumption 错，information loss 可能严重。
]


== Non-parametric
<non-parametric>
不预设 fixed model。

典型：

- histogram
- clustering
- sampling

演进直觉：

```text
Parametric
→ very compact
→ but risk model misspecification

Non-parametric
→ more flexible
→ but may require more storage
```


= 66. Regression as Data Reduction★★★★☆
<regression-as-data-reduction>
Regression 的任务：

用：

- dependent variable $Y$
- independent variables $X$

建立： $Y approx f(X)$

然后 raw points 可以一定程度由：

#quote(block: true)[
model parameters
]

表示。


= 66.1 Linear Regression★★★★★
<linear-regression>
#formula-box[$ Y = w X + b $]

其中：

- $w$：slope
- $b$：intercept

目标：

#quote(block: true)[
best fit。
]

课件主要说：

#quote(block: true)[
least squares。
]

标准 least-squares objective 可写：

#formula-box[$ min_(w,b) sum_(i = 1)^n [y_i - ( w x_i + b )]^2 $]

这条 objective 是对"least squares"的数学展开。


== Why square error?
<why-square-error>
Error： $e_i = y_i - hat(y)_i$

若直接 sum：

$ sum e_i $

positive / negative 可能互相取消。

平方： $e_i^2$

保证非负，同时 stronger penalty large errors。


= 66.2 Nonlinear Regression
<nonlinear-regression>
形式： $Y = f(X\;theta)$

其中 $f$ 对 parameters 可以 nonlinear。

课件：

#quote(block: true)[
fitted by successive approximations。
]

也就是通常不能一步 closed-form solution，需要 iterative optimization。


= 66.3 Multiple Regression★★★★★
<multiple-regression>
课件：

#formula-box[$ Y = b_0 + b_1 X_1 + b_2 X_2 $]

更一般：

$ Y = b_0 + sum_(j = 1)^d b_j X_j $

用于 multidimensional feature vector。


= 67. Log-Linear Model★★★☆☆
<log-linear-model>
核心：

#quote(block: true)[
一个 function 取 logarithm 后，对 model parameters 是 linear combination。
]

一般直觉：

$ log y = b_0 + b_1 x_1 + dots.h.c + b_d x_d $

于是可以利用 linear modeling techniques。

课件用途：

#quote(block: true)[
对 discretized multidimensional attributes，通过少量 marginal combinations 估计整个 multidimensional space 中 tuples 的 probability。
]

用途：

- dimensionality reduction
- smoothing


= 68. Histogram as Reduction★★★☆☆
<histogram-as-reduction>
将原始 observations：

#quote(block: true)[
divide into buckets。
]

每个 bucket 只存：

- average
- sum
- frequency

因此从 many raw values： $x_1,dots.h,x_n$

变成少量 bucket summaries。

partition：

- equal-width
- equal-frequency


= 69. Clustering as Reduction★★★★☆
<clustering-as-reduction>
先把： $D$

partition into clusters： $C_1,dots.h,C_k$

然后不存所有 points，只存 cluster representation，例如：

- centroid
- diameter

如果 data 天然 clustered：

#quote(block: true)[
very effective。
]

如果 data "smeared"：

#quote(block: true)[
approximation 较差。
]

也可以 hierarchical clustering：

#quote(block: true)[
存到 multidimensional index tree。
]


= 70. Dimensionality Reduction：为什么需要？★★★★★
<dimensionality-reduction为什么需要>
这一部分是全章最后的大高潮。

Curse of dimensionality：

当： $d arrow.t$

会发生：

=== 1. Data sparsity ↑
<data-sparsity>
有限 $n$ 分散到越来越大的 feature space。

=== 2. Distance less meaningful
<distance-less-meaningful>
最近邻和远邻距离趋于接近。

因此：

- clustering
- outlier detection

变差。

=== 3. Number of subspaces exponential
<number-of-subspaces-exponential>
若每个 feature 可选 / 不选：

#formula-box[$ 2^d $]

possible subsets。


= 71. DR 的两个根本路线★★★★★
<dr-的两个根本路线>
== Feature Selection
<feature-selection>
从原 features： $X_1,dots.h,X_d$

中选择 subset： $X_(i_1),dots.h,X_(i_k)$

特点：

#quote(block: true)[
原始 feature semantics 保留。
]


== Feature Extraction
<feature-extraction>
构造新 features： $Z_1,dots.h,Z_k$

其中： $k lt.double d$

并且： $Z = f(X)$

PCA 就属于：

#formula-box[$ upright("feature extraction") $]


= 72. PCA★★★★★
<pca>
定义：

#quote(block: true)[
使用 orthogonal transformation，把可能 correlated 的 variables 转换成 linearly uncorrelated principal components。
]

核心目标：

$ d arrow.r k quad(k lt.double d) $

同时尽量保留：

#quote(block: true)[
variance / information。
]


= 73. PCA 的核心几何思想★★★★★
<pca-的核心几何思想>
假设二维 points 沿一条 diagonal 方向排列。

原 coordinates： $x,med y$

高度 correlated。

真正自由度可能只有：

#quote(block: true)[
沿 diagonal 的一个 direction。
]

PCA 寻找： $v_1$

使 projection variance 最大。

剩余 orthogonal direction： $v_2$

variance 很小。

于是只保留： $v_1$

完成： $2 D arrow.r 1 D$


= 74. PCA 与 Covariance Matrix★★★★★
<pca-与-covariance-matrix>
课件核心：

#formula-box[$ A v = lambda v $]

这里的 $A$ 实际是 covariance matrix： $Sigma$

因此：

#formula-box[$ Sigma v = lambda v $]

其中：

- $v$：eigenvector → principal component direction
- $lambda$：eigenvalue → variance along that direction

所以： $lambda_1 gt.eq lambda_2 gt.eq dots.h.c gt.eq lambda_d$

按 eigenvalue 从大到小排序。


= 75. 为什么 eigenvector 就是最大 variance direction？★★★★★
<为什么-eigenvector-就是最大-variance-direction>
这是 PCA 最值得会推导的一步。

假设数据已经 centered： $E[X]= 0$

选择 unit vector： $v^T v = 1$

project： $z = v^T X$

variance：

$ "Var"(z)= E[(v^T X)^2] $

$ = v^T E[X X^T]v $

centered data 有： $E[X X^T]= Sigma$

所以：

#formula-box[$ "Var"(z)= v^T Sigma v $]

目标： $max_v v^T Sigma v$

subject to： $v^T v = 1$

构造 Lagrangian：

$ L(v,lambda)= v^T Sigma v - lambda(v^T v - 1) $

对 $v$ 求导： $2 Sigma v - 2 lambda v = 0$

得到：

#formula-box[$ Sigma v = lambda v $]

而此时：

$ v^T Sigma v = v^T(lambda v) $

$ = lambda v^T v $

$ = lambda $

因此：

#quote(block: true)[
variance 最大的 direction，就是 largest eigenvalue 对应 eigenvector。
]

这就是 PCA 最核心数学逻辑。


= 76. PCA Algorithm★★★★★
<pca-algorithm>
课件 pp.103--105：

=== Step 1
<step-1>
Normalize data，使 attributes roughly comparable。

=== Step 2
<step-2>
计算 covariance matrix： $Sigma$

=== Step 3
<step-3>
求：

- eigenvalues
- eigenvectors

满足： $Sigma v_i = lambda_i v_i$

=== Step 4
<step-4>
排序：

$ lambda_1 gt.eq lambda_2 gt.eq dots.h.c $

=== Step 5
<step-5>
保留 top $k$ eigenvectors：

$ V_k =[v_1,dots.h,v_k] $

=== Step 6
<step-6>
project data：

#formula-box[$ Z =(X - mu)V_k $]

这里 $mu$ 是每列均值。只有在前一步已经把 $X$ 明确记作中心化后的数据矩阵时，才可简写为 $Z = X V_k$；中心化不是可随意省略的 convention。


== 课件关于 preprocessing 的一个理解细节
<课件关于-preprocessing-的一个理解细节>
课件写：

#quote(block: true)[
Normalize input data: Each attribute falls within the same range.
]

严格 PCA 数学中最基本的是：

#quote(block: true)[
center data。
]

即：

$ x_i arrow.l x_i - mu $

是否进一步 scale 到相同 range / standard deviation：

#quote(block: true)[
取决于应用。
]

不过考试按 PPT 应记：

#quote(block: true)[
PCA 前先 normalize，使 attributes comparable。
]


= 77. PCA 为什么可以降维？★★★★★
<pca-为什么可以降维>
原向量可以表示：

$ x = a_1 v_1 + a_2 v_2 + dots.h.c + a_d v_d $

如果： $lambda_1,dots.h,lambda_k$

很大，而： $lambda_(k + 1),dots.h,lambda_d$

很小。

则 weak components 贡献很小。

近似：

#formula-box[$ x approx sum_(j = 1)^k a_j v_j $]

因此只存： $a_1,dots.h,a_k$

即可。


= 78. PCA 局限★★★★★
<pca-局限>
课件：

+ numeric data only
+ linear method
+ each PC 是 original attributes 的 linear combination

适合：

- 主要结构近似位于低维 #strong[linear] subspace；
- 希望用最大方差方向作线性压缩/可视化。

approximately Gaussian 只是一种常见的经验情形；PCA 不要求数据 Gaussian，也不以"类别线性可分"为必要条件或目标。

问题：

若 data 在 nonlinear manifold 上：

#quote(block: true)[
PCA 只能做 linear rotation/projection。
]

于是进入 nonlinear dimensionality reduction。


= 79. Attribute Subset Selection★★★★★
<attribute-subset-selection>
另一条 DR 路线。

删除：

== Redundant attributes
<redundant-attributes>
例如：

- purchase price
- sales tax

若 sales tax 是固定比例： $t a x = c times p r i c e$

则高度 redundant。


== Irrelevant attributes
<irrelevant-attributes>
例如预测 GPA 时：

#quote(block: true)[
Student ID。
]

它几乎不携带 predictive information。


= 80. Feature Selection 为什么不能 exhaustive search？★★★★★
<feature-selection-为什么不能-exhaustive-search>
若 $d$ features：

每个 feature：

- choose
- not choose

所以：

#formula-box[$ 2^d $]

possible subsets。

当： $d = 100$

则： $2^100$

完全不可穷举。

因此需要：

#quote(block: true)[
heuristic search。
]


= 81. Feature Selection Heuristics★★★★★
<feature-selection-heuristics>
== Best single attribute
<best-single-attribute>
在 independence assumption 下：

#quote(block: true)[
用 significance test 选最强单 feature。
]

问题：

#quote(block: true)[
interaction features 可能单独都不强，但组合很强。
]


== Step-wise Forward Selection
<step-wise-forward-selection>
流程：

```text
∅
↓
选最好 feature
↓
在已有 subset 条件下继续加最好 feature
↓
repeat
```


== Step-wise Elimination
<step-wise-elimination>
从： ${ X_1,dots.h,X_d }$

开始。

不断：

#quote(block: true)[
remove worst feature。
]


== Combined Selection + Elimination
<combined-selection-elimination>
Forward + backward 交替。

比纯 forward 更好，因为：

#quote(block: true)[
一个 feature 加入后，之前 feature 的价值可能变化。
]


== Branch and Bound
<branch-and-bound>
使用：

- elimination
- backtracking

尝试找到更接近 optimal subset。


= 82. Combined Selection Example★★★★☆
<combined-selection-example>
预测 final score：

- $X_1$：StudyHours
- $X_2$：Attendance
- $X_3$：FavoriteColor

Null model： $R M S E = 14.2$

单独加入： $X_1 arrow.r 9.1$

$ X_2 arrow.r 10.8 $

$ X_3 arrow.r 14.2 $

所以先选： $X_1$

第二步加入： $X_2$

模型：

$ hat(Y) = a X_1 + b X_2 + c $

Backward check：

#quote(block: true)[
两者 significant → keep。
]

再试： $+ X_3$

没有 improvement：

#quote(block: true)[
reject。
]

最终：

#formula-box[$ { S t u d y H o u r s,A t t e n d a n c e } $]


= 83. Feature Generation★★★★☆
<feature-generation>
不是删 feature，而是：

#quote(block: true)[
构造比原 features 更有效的新 features。
]

三类：

== Attribute extraction
<attribute-extraction>
domain-specific。

== Mapping to new space
<mapping-to-new-space>
例如：

- Fourier transform
- Wavelet transform
- Manifold methods

== Attribute construction
<attribute-construction>
组合已有 features。

也包括：

#quote(block: true)[
discretization。
]


= 84. PCA → Nonlinear DR：关键方法演进★★★★★
<pca-nonlinear-dr关键方法演进>
这是本章方法论演进最明显的一段。

```text
High dimensional data
↓
PCA
↓
preserve largest linear variance
↓
Problem:
nonlinear manifold cannot be unfolded by linear projection
↓
Construct pairwise proximity P
↓
Instead of preserving coordinates,
preserve relationships
↓
KPCA / SNE
```

核心思想发生了变化：

PCA：

#quote(block: true)[
preserve variance。
]

Nonlinear DR：

#quote(block: true)[
preserve pairwise proximity / neighborhood structure。
]


= 85. Nonlinear DR 的统一框架★★★★★
<nonlinear-dr-的统一框架>
输入： $X in bb(R)^(n times d)$

构造 proximity matrix： $P in bb(R)^(n times n)$

其中： $P_(i j)$

表示 $x_i,x_j$ 的 similarity / neighborhood relationship。

然后学习： $hat(X) in bb(R)^(n times k)$

其中： $k lt.double d$

使低维空间中的： $hat(P)$

尽量保留： $P$

即：

#formula-box[$ P approx hat(P) $]


= 86. Kernel PCA★★★★★
<kernel-pca>
核心：

#quote(block: true)[
不直接在 input coordinates 做 PCA，而是通过 kernel 建模 nonlinear similarity。
]

构造：

#formula-box[$ P(i,j)= kappa(x_i,x_j) $]

其中： $kappa$

为 kernel function。

然后使用 kernel matrix $P$ 的：

#quote(block: true)[
top-$k$ eigenvectors/eigenvalues。
]


= 87. Polynomial Kernel★★★★☆
<polynomial-kernel>
课件：

#formula-box[$ kappa(x_i,x_j)=(1 + x_i^T x_j)^p $]

其中：

- $x_i^T x_j$：dot product
- $p$：polynomial degree

它隐式对应：

#quote(block: true)[
higher-order interaction features。
]


= 88. RBF Kernel★★★★★
<rbf-kernel>
课件：

#formula-box[$ kappa(x_i,x_j)= exp (- frac(parallel x_i - x_j parallel^2, 2 sigma^2)) $]

其中：

- $parallel x_i - x_j parallel$：Euclidean distance
- $sigma$：kernel bandwidth

若两点接近： $parallel x_i - x_j parallel^2 approx 0$

则： $kappa approx 1$

若远： $kappa arrow.r 0$

因此 RBF kernel 把 distance 转成 local similarity。


= 89. KPCA preserving proximity★★★★★
<kpca-preserving-proximity>
课件给 Step 2：

#formula-box[$ min sum_(i,j) (P ( i , j ) - hat(P) ( i , j ))^2 $]

也写作：

#formula-box[$ min parallel P - hat(P) parallel_(F r o)^2 $]

其中：

$ parallel A parallel_F^2 = sum_(i j) A_(i j)^2 $

即 Frobenius norm。

意思：

#quote(block: true)[
让低维 representation 重建出来的 proximity matrix 与原 kernel matrix 尽量一致。
]


= 90. Linear kernel 为什么退化到 PCA？★★★★★
<linear-kernel-为什么退化到-pca>
如果：

#formula-box[$ kappa(x_i,x_j)= x_i^T x_j $]

没有 nonlinear transformation。

此时 kernel matrix 只表达普通 linear inner products。

课件：

#formula-box[$ K P C A arrow.r s t a n d a r d med P C A $]


= 91. SNE★★★★★
<sne>
SNE：

#formula-box[$ S t o c h a s t i c med N e i g h b o r h o o d med E m b e d d i n g $]

它不再把 proximity 当普通 real-valued similarity，而是：

#quote(block: true)[
变成 neighbor probability。
]


= 91.1 High-dimensional Proximity
<high-dimensional-proximity>
课件定义：

#formula-box[$ d_(i j)^2 = frac(parallel x_i - x_j parallel^2, 2 sigma_i^2) $]

然后：

#formula-box[$ P(i,j)= frac(e^(- d_(i j)^2), sum_(l = 1,l eq.not i)^n e^(- d_(i l)^2)) $]

含义：

$ P(i,j)= P(x_j upright(" is neighbor of ") x_i) $

对于固定 $i$：

$ sum_(j eq.not i) P(i,j)= 1 $

所以 $P_i$ 是一个 probability distribution。


= 91.2 Low-dimensional proximity
<low-dimensional-proximity>
学习： $hat(x)_i$

以后，在 low-dimensional space 用类似方式计算：

$ hat(P)(i,j) $

现在目标：

#quote(block: true)[
高维 neighborhood distribution 和低维 neighborhood distribution 尽量相同。
]


= 91.3 为什么 SNE 用 KL？★★★★★
<为什么-sne-用-kl>
因为： $P_i,hat(P)_i$

都是 probability distributions。

因此自然使用： $D_(K L)$

比较。

Objective：

#formula-box[$ min_(hat(X)) sum_(i = 1)^n D_(K L)(P_i parallel hat(P)_i) $]

课件写作：

#formula-box[$ hat(x)_i = arg min_(hat(x)_i) sum_(i = 1)^n D_(K L)(P_i parallel hat(P)_i) $]

更严格地说应理解为：

#quote(block: true)[
jointly optimize all low-dimensional coordinates。
]

这正好把本章前面的 KL divergence 用到了最后的 nonlinear DR 中。

这是本章隐藏得非常漂亮的一条知识闭环：

```text
Probability-distribution similarity
        ↓
    KL divergence
        ↓
Neighborhood distribution
        ↓
       SNE
```

★★★★★ 很值得记。


= 92. PCA vs KPCA vs SNE★★★★★
<pca-vs-kpca-vs-sne>
#figure(
  align(center)[#table(
    columns: (20%, 20%, 20%, 20%, 20%),
    align: (auto,auto,auto,auto,auto,),
    table.header([方法], [保留什么], [Relationship representation], [核心 objective], [Linear?],),
    table.hline(),
    [PCA], [variance], [covariance], [max variance], [Yes],
    [KPCA], [kernel proximity], [$kappa(x_i,x_j)$], [$parallel P - hat(P) parallel_F^2$], [Nonlinear],
    [SNE], [neighborhood probability], [$P(j divides i)$], [$sum_i K L(P_i parallel hat(P)_i)$], [Nonlinear],
  )]
  , kind: table
  )

更深层的演进：

```text
PCA
坐标方差
↓
KPCA
pairwise kernel similarity
↓
SNE
pairwise similarity → probability distribution
↓
通过 KL 保 neighborhood
```


= 93. PCA 为什么不能解决课件的 nonlinear example？★★★★★
<pca-为什么不能解决课件的-nonlinear-example>
pp.113--114 图中：

红蓝数据形成类似弯曲 / moon-shaped nonlinear structures。

原数据：

#quote(block: true)[
not linearly separable。
]

PCA：

#quote(block: true)[
只能 rotate + linear project。
]

线性 transformation： $z = W^T x$

无法把 nonlinear topology 真正"展开"。

因此：

#quote(block: true)[
PCA 后依然不 linearly separable。
]


KPCA：

通过 nonlinear kernel： $kappa(x_i,x_j)$

隐式映射到高维 feature space。

在那里：

#quote(block: true)[
nonlinear boundary 可能变成 linear。
]

所以可以更好分开。


t-SNE：

通过 preservation of neighborhood：

#quote(block: true)[
直接让同类 local neighborhoods 聚集、不同结构分离。
]

课件结果也显示：

#quote(block: true)[
可以把红蓝 structures 分开。
]

课件本章没有给出 t-SNE 的 Student-$t$ low-dimensional formula，所以这里不额外把未出现在 slide 的 t-SNE 数学公式混入考试笔记。


= 94. Proximity Heatmap：应该会读图★★★★☆
<proximity-heatmap应该会读图>
p.114：

matrix 按：

```text
red samples
blue samples
```

排序。

得到四个 blocks：

```text
          red      blue
red      RR       RB
blue     BR       BB
```

Diagonal：

- RR
- BB

表示：

#quote(block: true)[
within-cluster proximity。
]

Off-diagonal：

- RB
- BR

表示：

#quote(block: true)[
between-cluster proximity。
]

好的 representation 应满足：

#formula-box[$ P_(upright("within")) gt.double P_(upright("between")) $]

课件指出：

KPCA / t-SNE：

#quote(block: true)[
比 linear PCA 更清晰地形成 diagonal blocks。
]


= 95. 本章最核心的"Why it fails → How it fixes"总表★★★★★
<本章最核心的why-it-fails-how-it-fixes总表>
#figure(
  align(center)[#table(
    columns: (25%, 25%, 25%, 25%),
    align: (auto,auto,auto,auto,),
    table.header([Old method/problem], [Why it fails], [Next method], [How it fixes],),
    table.hline(),
    [Mean], [outlier-sensitive], [Median / Trimmed mean], [减弱 extremes],
    [Five-number summary], [丢失 distribution shape], [Histogram], [显示完整 shape],
    [Covariance], [scale-dependent], [Correlation], [除以 SD 标准化],
    [Euclidean for all data], [不同 attribute semantics], [type-specific proximity], [按 nominal/binary/ordinal 处理],
    [Symmetric matching], [0--0 对 rare event 无意义], [Jaccard], [忽略 joint absence],
    [Euclidean for documents], [document length dominates], [Cosine], [比 direction 而非 magnitude],
    [Vector comparison], [不能比较 distributions], [KL divergence], [information-based distribution difference],
    [Hand-crafted similarity], [hidden semantics 丢失], [Representation learning], [学习 semantic representation],
    [Raw dirty data], [missing/noise/inconsistency], [Cleaning], [inference/smoothing/auditing],
    [Separate sources], [conflicts/redundancy], [Integration], [schema/entity/conflict resolution],
    [Different scales], [distance 被大-scale feature 主导], [Normalization], [common scale],
    [Raw continuous values], [granular, noisy, large], [Discretization], [intervals/concepts],
    [Equal-width bins], [skew/outliers], [Equal-depth], [equal population],
    [Equal-depth], [rigid boundaries], [Clustering], [data-driven partitions],
    [Unsupervised discretization], [不关心 target label], [Decision tree / Chi-Merge], [use class information],
    [Full dataset], [computationally expensive], [Sampling / reduction], [smaller representation],
    [Random sampling], [minority/skew 被漏掉], [Stratified sampling], [preserve groups],
    [Parametric reduction], [model misspecification], [Non-parametric], [fewer assumptions],
    [High-dimensional raw data], [curse of dimensionality], [DR], [smaller principal representation],
    [Exhaustive feature selection], [$2^d$], [heuristic selection], [tractable search],
    [PCA], [only linear], [KPCA], [nonlinear kernel],
    [Coordinate preservation], [nonlinear neighborhood poorly preserved], [SNE], [preserve probabilistic neighborhoods],
  )]
  , kind: table
  )


= 96. 一张表彻底区分"Reduction"相关概念★★★★★
<一张表彻底区分reduction相关概念>
#figure(
  align(center)[#table(
    columns: (33.33%, 33.33%, 33.33%),
    align: (auto,auto,auto,),
    table.header([方法], [减少什么], [是否改变 feature semantics],),
    table.hline(),
    [Sampling], [rows / objects], [否],
    [Histogram], [raw values], [是，变 aggregate],
    [Clustering reduction], [objects], [用 cluster representative],
    [Compression], [storage], [depends],
    [Feature selection], [columns], [保留],
    [PCA], [dimensions], [改变],
    [KPCA], [dimensions], [改变],
    [SNE/t-SNE], [dimensions], [改变],
    [Discretization], [number of distinct values], [改变 value representation],
  )]
  , kind: table
  )

这是考试中非常容易混的一组概念。


= 97. 必须背熟的公式清单★★★★★
<必须背熟的公式清单>
建议最后复习时至少能无提示写出这些。

=== Mean
<mean-1>
$ macron(x) = 1 / n sum_i x_i $

=== Weighted mean
<weighted-mean>
$ macron(x) = frac(sum_i w_i x_i, sum_i w_i) $

=== Grouped median
<grouped-median>
$ M = L + frac(n/2 - F, f_m) h $

=== Mean-mode-median
<mean-mode-median>
$ m e a n - m o d e = 3(m e a n - m e d i a n) $

=== Sample variance
<sample-variance-1>
$ s^2 = frac(1, n - 1) sum_i(x_i - macron(x))^2 $

=== Computational variance
<computational-variance>
$ s^2 = frac(1, n - 1) [sum_i x_i^2 - 1 / n ( sum_i x_i )^2] $

=== Random-variable variance
<random-variable-variance>
$ V a r(X)= E[(X - mu)^2]= E[X^2]- E[X]^2 $

=== Chi-square
<chi-square>
$ chi^2 = sum_(i j) frac((O_(i j) - E_(i j))^2, E_(i j)) $

=== Expected count
<expected-count-1>
$ E_(i j) = frac(r o w_i times c o l_j, N) $

=== Degrees of freedom
<degrees-of-freedom-1>
$ d f =(r - 1)(c - 1) $

=== Covariance
<covariance-1>
$ C o v(X,Y)= E[(X - mu_X)(Y - mu_Y)] $

$ = E[X Y]- E[X]E[Y] $

=== Correlation
<correlation-1>
$ rho_(X Y) = frac(C o v(X,Y), sigma_X sigma_Y) $

=== Covariance matrix
<covariance-matrix-1>
$ Sigma = E[(X - mu)(X - mu)^T] $

=== IQR
<iqr>
$ I Q R = Q_3 - Q_1 $

=== Minkowski
<minkowski>
$ d(i,j)= (sum_f divides x_(i f) - x_(j f) \|^p)^(1/p) $

=== Manhattan
<manhattan>
$ d = sum_f divides x_(i f) - x_(j f)\| $

=== Euclidean
<euclidean>
$ d = sqrt(sum_f(x_(i f) - x_(j f))^2) $

=== Chebyshev
<chebyshev>
$ d = max_f divides x_(i f) - x_(j f)\| $

=== Symmetric binary
<symmetric-binary-2>
$ d = frac(r + s, q + r + s + t) $

=== Asymmetric binary
<asymmetric-binary-2>
$ d = frac(r + s, q + r + s) $

=== Jaccard
<jaccard>
$ J = frac(q, q + r + s) $

=== Nominal simple matching
<nominal-simple-matching>
$ d = frac(p - m, p) $

=== Ordinal mapping
<ordinal-mapping>
$ z_(i f) = frac(r_(i f) - 1, M_f - 1) $

=== Mixed attributes
<mixed-attributes>
$ d(i,j)= frac(sum_f w_(i j)^((f)) d_(i j)^((f)), sum_f w_(i j)^((f))) $

=== Cosine
<cosine>
$ cos(d_1,d_2)= frac(d_1 dot.op d_2, parallel d_1 parallel parallel d_2 parallel) $

=== KL discrete
<kl-discrete>
$ D_(K L)(P parallel Q)= sum_x p(x)ln frac(p(x), q(x)) $

=== KL continuous
<kl-continuous>
$ D_(K L)(P parallel Q)= integral p(x)ln frac(p(x), q(x)) d x $

=== Min-Max
<min-max>
$ v' = frac(v - m i n_A, m a x_A - m i n_A)(n e w\_m a x_A - n e w\_m i n_A)+ n e w\_m i n_A $

=== Z-score
<z-score>
$ v' = frac(v - mu_A, sigma_A) $

=== Decimal scaling
<decimal-scaling-1>
$ v' = v / 10^j $

=== Equal-width bin
<equal-width-bin>
$ W = frac(B - A, N) $

=== Linear regression
<linear-regression-1>
$ Y = w X + b $

=== Multiple regression
<multiple-regression-1>
$ Y = b_0 + b_1 X_1 + b_2 X_2 $

=== PCA eigenproblem
<pca-eigenproblem>
$ Sigma v = lambda v $

=== PCA projection
<pca-projection>
$ Z =(X - mu)V_k $

=== Polynomial kernel
<polynomial-kernel-1>
$ kappa(x_i,x_j)=(1 + x_i^T x_j)^p $

=== RBF
<rbf>
$ kappa(x_i,x_j)= exp [- frac(parallel x_i - x_j parallel^2, 2 sigma^2)] $

=== KPCA proximity objective
<kpca-proximity-objective>
$ min parallel P - hat(P) parallel_F^2 $

=== SNE distance
<sne-distance>
$ d_(i j)^2 = frac(parallel x_i - x_j parallel^2, 2 sigma_i^2) $

=== SNE neighbor probability
<sne-neighbor-probability>
$ P(i,j)= frac(e^(- d_(i j)^2), sum_(l eq.not i) e^(- d_(i l)^2)) $

=== SNE objective
<sne-objective>
$ min sum_i D_(K L)(P_i parallel hat(P)_i) $


= 98. 期末特别容易出错的 15 个点★★★★★
<期末特别容易出错的-15-个点>
+ #strong[Numeric-looking ID 不等于 numeric attribute。]
+ #strong[Ordinal 有 order，但 interval spacing 未知。]
+ Celsius 是 interval，Kelvin 是 ratio。
+ Binary 可以 symmetric 或 asymmetric。
+ Positive skew： $ m o d e < m e d i a n < m e a n $
+ Sample variance 和 population variance denominator 不同。
+ Covariance 0 不代表 independence。
+ Correlation 不代表 causality。
+ Correlation 0 也不代表没有 nonlinear relationship。
+ Histogram 是 quantitative distribution；bar chart 是 categorical comparison。
+ Jaccard 不计算 0--0 match。
+ KL divergence asymmetric，因此不是 metric。
+ Equal-width 对 skew/outlier 很敏感。
+ PCA 做 feature extraction，不是 feature selection。
+ PCA 只能 linear；KPCA / SNE 解决 nonlinear structure。


= 99. 课件中两个值得特别记的"不一致 / 易混点"
<课件中两个值得特别记的不一致-易混点>
第一，p.78 Normalization 的文字写的是 \$73,000，但公式和最终结果 $0.716,1.225$ 实际对应 \$73,600。考试计算时以给出的实际 numeric expression 为准。

第二，p.80 把 correlation/$chi^2$ discretization 写成 "unsupervised"，但 p.84 的 Chi-Merge 明确写 "supervised: use class information"，并且其算法确实需要比较 neighboring intervals 的 class distributions。因此按方法定义和 p.84，应理解为：

#formula-box[$ upright("Chi-Merge = supervised bottom-up discretization") $]


= 100. 最后用一句话串起整个 Chapter★★★★★
<最后用一句话串起整个-chapter>
这章真正的逻辑不是：

#quote(block: true)[
"先统计，再清洗，再 PCA。"
]

而是：

#quote(block: true)[
#strong[先理解数据的语义与类型，才能选择正确的统计量与相似度；理解数据之后，再解决现实数据中的缺失、噪声、冲突和尺度问题；当数据仍然过大或过高维时，用 sampling、compression、feature selection 和 PCA 减少复杂度；当 PCA 的线性假设无法描述真实结构时，再把目标从"保留线性方差"升级为"保留 nonlinear proximity / neighborhood"，最终得到 KPCA 和 SNE。]
]

即：

#formula-box[$ R e p r e s e n t a t i o n arrow.r U n d e r s t a n d i n g arrow.r C o m p a r i s o n arrow.r C l e a n i n g arrow.r T r a n s f o r m a t i o n arrow.r R e d u c t i o n arrow.r R e p r e s e n t a t i o n med L e a r n i n g $]

这就是整个 Chapter 2 的底层架构。


= 高密度复习：Data / Preprocessing
<高密度复习data-preprocessing>
== 先判数据类型，再选距离与处理
<先判数据类型再选距离与处理>
#figure(
  align(center)[#table(
    columns: (33.33%, 33.33%, 33.33%),
    align: (auto,auto,auto,),
    table.header([类型], [合法操作], [常用相异度 / 注意],),
    table.hline(),
    [Nominal], [$=$ / $eq.not$], [simple matching；类别编码不应被误当有大小],
    [Symmetric binary], [0、1 同等重要], [match / mismatch],
    [Asymmetric binary], ["1"出现更重要], [共同 0 忽略：该维度 $w_(i j)^((f)) = 0$],
    [Ordinal], [排名 + 间距不一定相等], [先 rank-normalize 再按 numeric 处理],
    [Interval], [差有意义、零点任意], [Euclidean / standardized distance],
    [Ratio], [差与比都可解释], [可按 numeric；留意长尾/尺度],
    [Text / sparse vector], [方向常比长度重要], [cosine；$cos(d_1,d_2)= d_1 dot.op d_2/(parallel d_1 parallel parallel d_2 parallel)$],
  )]
  , kind: table
  )

== 预处理五件事
<预处理五件事>
#figure(
  align(center)[#table(
    columns: (33.33%, 33.33%, 33.33%),
    align: (auto,auto,auto,),
    table.header([目标], [常见方法], [易错点],),
    table.hline(),
    [Cleaning], [missing imputation、noise smoothing、outlier handling], [不要把异常一律当噪声删掉],
    [Integration], [schema/entity resolution、redundancy/correlation], [同一实体可能有不同 ID/单位],
    [Transformation], [normalization、aggregation、encoding、feature construction], [train/test 必须使用同一拟合参数],
    [Reduction], [sampling、aggregation、feature selection、PCA], [reduction 要保留任务相关信息],
    [Discretization], [binning、histogram、entropy-based], [boundaries 是端点；按"最近端点"计算],
  )]
  , kind: table
  )

== 高频公式
<高频公式>
$ z = frac(x - mu, sigma),#h(2em) x' = frac(x - min, max - min),#h(2em) x' = frac(x - min, max - min)(n e w_(m a x) - n e w_(m i n))+ n e w_(m i n) $

$ d_E(x,y)= sqrt(sum_j(x_j - y_j)^2),#h(2em) d_M(x,y)= sqrt((x - y)^T S^(- 1)(x - y)) $

$ chi^2 = sum frac((O - E)^2, E),#h(2em) d f =(r - 1)(c - 1),#h(2em) E_(i j) = frac((r o w_i)(c o l_j), N) $

Pearson $chi^2$ 的拒绝条件要写 #strong[显著性水平 $alpha$]；$alpha = 0.001$ 对应 confidence level $0.999$，不是 "confidence level 0.001"。

== PCA 必背
<pca-必背>
+ 对数据中心化（必要；尺度差异大时通常再 standardize）。
+ 求 covariance matrix / SVD，取最大 eigenvalues 对应的 $V_k$。
+ 投影：$Z =(X - mu)V_k$；只有 $X$ 已中心化时才可写 $X V_k$。
+ PCA 是线性、最大方差的表示法；不要求 Gaussian 或类别线性可分，也不保证保留分类边界。

== 已确认的课件陷阱
<已确认的课件陷阱>
- cosine 例中 $d_2 =(3,0,2,0,1,1,0,1,0,1)$：$parallel d_2 parallel = 4$，$cos approx 0.965$；课件的 4.12 / 0.94 是算术错误。
- binning boundaries 例首 bin 应为 $4,4,15,15$，不是 $4,4,4,15$。
- mixed-type formula 要排除 asymmetric binary 的 joint absence。
