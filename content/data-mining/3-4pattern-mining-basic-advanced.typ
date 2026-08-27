// @title: Pattern Mining: Basic and Advanced Methods
// @description: 频繁模式、关联规则、Apriori、FP-growth 与高级模式挖掘。
// @order: 35

#import "../template.typ": *

= Pattern Mining: Basic and Advanced Methods

== Overview

*Exam: ★★★★☆*

Pattern mining asks for recurring or otherwise statistically meaningful structures hidden in large data collections. The basic setting is a transaction database, where each transaction contains a set of items and the central task is to discover frequent itemsets and association rules. The main difficulty is combinatorial: a long transaction can induce exponentially many subpatterns, so useful mining systems need both compact representations and aggressive pruning.

The chapter develops a common line of ideas. First, support and confidence formalize frequency and directional implication; closed and maximal patterns compress the frequent-pattern family. Second, the downward-closure property enables scalable mining, leading from Apriori's level-wise candidate generation to vertical-format ECLAT and pattern-growth FP-growth. Third, pattern evaluation goes beyond support and confidence through lift, $chi^2$, null-invariant measures, Kulczynski, and imbalance ratio. The same principles then generalize to multilevel, multidimensional, quantitative, rare, negative, sequential, and graph patterns. Constraint-based mining makes this process user-directed by pushing constraints into either the pattern space or the data space. Finally, the course illustrates how these ideas support graph indexing, approximate graph search, copy-paste bug detection, and phrase mining.

#note[
  The two slide decks use both *absolute* minimum support (a count, such as `minsup = 2`) and *relative* minimum support (a fraction, such as `minsup = 50%`). In this chapter, $"sup"(X)$ denotes the count and $s(X)$ the fraction; the threshold is written as a count or as $sigma$ according to context.
]

== Foundations: Frequent Patterns, Rules, and Compact Representations

*Exam: ★★★★★*

=== Transaction databases, itemsets, and support

A transaction database is a collection $D = {T_1, dots, T_N}$ of transactions. Each transaction has a transaction identifier (TID) and contains a set of items; in retail data, counts or quantities may also be attached to items. An *itemset* is a nonempty set of items, and a $k$-itemset contains exactly $k$ items. For example, ${"Beer", "Nuts", "Diaper"}$ is a 3-itemset.

The absolute support of an itemset $X$ is the number of transactions containing all items in $X$; relative support is the corresponding fraction:

$
  "sup"(X) = |{T_j : X subset.eq T_j}|,
  quad
  s(X) = "sup"(X) / N.
$

For the five-transaction example in the slides,

- $"sup"({"Beer"}) = 3$ and $s({"Beer"}) = 3/5 = 0.6$;
- $"sup"({"Diaper"}) = 4$ and $s({"Diaper"}) = 4/5 = 0.8$;
- $"sup"({"Beer", "Diaper"}) = 3$;
- $"sup"({"Beer", "Eggs"}) = 1$.

An itemset $X$ is *frequent* if its support reaches the chosen threshold. With relative threshold $sigma$, the condition is $s(X) >= sigma$; with an absolute threshold, the same test is applied to $"sup"(X)$. At $sigma = 50%$ in the five-transaction database, the frequent 1-itemsets are Beer, Nuts, Diaper, and Eggs; the only frequent 2-itemset is ${"Beer", "Diaper"}$; no frequent 3-itemset exists.

Pattern discovery is broader than retail baskets: the slides treat patterns as itemsets, subsequences, or substructures that occur repeatedly or exhibit strong relationships. Such patterns support association/correlation analysis, sequential and graph mining, discriminative classification, pattern-based clustering, web-log analysis, biological sequence analysis, and mining of spatiotemporal, multimedia, time-series, and stream data.

=== From itemsets to association rules

A frequent itemset reports co-occurrence but not direction. An *association rule* $X -> Y$ is intended to express that transactions containing $X$ often also contain $Y$. The slides use two primary quantities:

$
  s(X -> Y) = s(X union Y),
  quad
  c(X -> Y) = "sup"(X union Y) / "sup"(X) = P(Y | X).
$

Here $X union Y$ is the combined itemset, so its support is the fraction of transactions containing both sides. For the rule ${"Diaper"} -> {"Beer"}$, $s = 3/5 = 0.6$ and $c = 3/4 = 0.75$. Association-rule mining is therefore: given `minsup` and `minconf`, enumerate all rules satisfying both thresholds.

#example[
  In the five-transaction database with `minsup = 50%` and `minconf = 50%`, the frequent pair ${"Beer", "Diaper"}$ produces two rules:

  - ${"Beer"} -> {"Diaper"}$ with support $60%$ and confidence $100%$;
  - ${"Diaper"} -> {"Beer"}$ with support $60%$ and confidence $75%$.

  This illustrates why frequent-itemset mining and rule mining are closely related: first discover sufficiently supported unions, then test directional rule confidence.
]

#warning[
  The basic slides say support and confidence ensure both popularity and “correlation.” Confidence alone does *not* establish statistical correlation: a rule can have high confidence merely because its consequent is common. The later pattern-evaluation slides explicitly correct this limitation using lift and other measures.
]

=== Why output compression is necessary

The pattern family can be exponentially large. In the slide example $T_1 = {a_1, dots, a_50}$ and $T_2 = {a_1, dots, a_100}$ with absolute `minsup = 1`, every nonempty subset of $T_2$ is frequent. Hence the number of frequent itemsets is

$
  sum_(k=1)^100 binom(100, k) = 2^100 - 1.
$

Even when mining is computationally possible, storing or interpreting all patterns may be impossible. Two boundary representations are therefore important.

A *closed frequent itemset* $X$ is frequent and has no proper super-itemset with exactly the same support. Closed patterns are a lossless compression with respect to support information: if a non-closed frequent pattern is omitted, its support can be recovered from an appropriate closed superpattern with the same support. In the two-transaction example above, the only closed patterns are ${a_1, dots, a_50}$ with support 2 and ${a_1, dots, a_100}$ with support 1.

A *maximal frequent itemset* $X$ is frequent and has no frequent proper super-itemset. Maximal patterns are more aggressively compressed, but they are lossy: they preserve the boundary of the frequent region, not exact support values of omitted subsets. In the same example, the unique maximal pattern is ${a_1, dots, a_100}$.

Thus the relationship is:

- closed patterns: fewer than all frequent itemsets, but exact frequency information is retained;
- maximal patterns: usually still fewer, but subset supports are lost;
- all maximal frequent itemsets are closed, but not every closed itemset is maximal.

#warning[
  The “Closed Patterns” example slide with transactions ${"Diaper", "Milk", "Bread"}$ and ${"Diaper", "Milk", "Bread", "Beer"}$ states `minsup = 2` but also lists ${"Diaper", "Milk", "Bread", "Beer"}:1$ as a closed pattern. Under the course definition, a closed pattern must first be frequent, so that support-1 itemset is *not* a closed frequent pattern when `minsup = 2`. It would be closed only if the threshold were at most 1.
]

== Scalable Frequent-Itemset Mining

*Exam: ★★★★★*

=== Downward closure: the pruning principle

The fundamental structural property is support anti-monotonicity: adding items cannot increase support. If $X subset.eq Y$, then every transaction containing $Y$ also contains $X$, so

$
  "sup"(Y) <= "sup"(X).
$

#theorem(title: "Downward-Closure / Apriori Property")[
  Every subset of a frequent itemset is frequent. Equivalently, if an itemset is infrequent, every one of its supersets is infrequent.
]

This gives a safe pruning rule: once a candidate fails minimum support, there is no reason to generate any extension containing it. Three scalable families in the slides exploit this principle in different data organizations: Apriori uses level-wise candidate generation; ECLAT uses vertical TID sets and intersections; FP-growth compresses the database into FP-trees and grows patterns recursively.

=== Apriori: level-wise candidate generation and testing

Let $F_k$ denote the set of frequent $k$-itemsets and $C_k$ the candidate $k$-itemsets. Apriori proceeds level by level:

- scan the database to obtain $F_1$;
- from $F_k$, generate candidate $(k+1)$-itemsets $C_(k+1)$ by self-join;
- prune every candidate having an infrequent $k$-subset;
- scan the database to count support of the remaining candidates and retain $F_(k+1)$;
- repeat until no candidate or frequent itemset remains.

In the slides' four-transaction example with absolute `minsup = 2`, the first scan removes $D$; the second-level candidates are all pairs among $A,B,C,E$, of which $"AC", "BC", "BE", "CE"$ are frequent; the only surviving 3-item candidate is $"BCE"$, with support 2.

Candidate generation itself has two stages. Suppose itemsets are listed in a fixed order. Two members of $F_(k-1)$ are joined when they share their first $k-2$ items and differ in the last item. Then the candidate is pruned if *any* $(k-1)$-subset is absent from $F_(k-1)$. For example, from

$
  F_3 = {"abc", "abd", "acd", "ace", "bcd"},
$

self-join can form $"abcd"$ and $"acde"$. Candidate $"acde"$ is removed because $"ade"$ is not frequent, leaving $C_4 = {"abcd"}$.

The weakness of Apriori is not correctness but cost: long patterns may require many database passes, and intermediate candidate sets can become enormous. The remaining methods try to reduce scans, candidates, or both.

=== Reducing Apriori's cost: partitioning and DHP

*Partitioning* exploits a simple theorem.

#theorem(title: "Partition Candidate Theorem")[
  If an itemset is globally frequent in a database partitioned into $D_1, dots, D_k$, then it must be frequent in at least one partition under the same relative support threshold $sigma$.
]

The proof is by contradiction. If $"sup"_i(X) < sigma |D_i|$ in every partition, then summing over partitions gives $"sup"(X) < sigma |D|$, so $X$ cannot be globally frequent. This leads to a two-scan algorithm: first partition the data so each partition fits in memory and mine local frequent patterns; then take their union as global candidates and perform one more full scan to obtain true global supports.

*Direct Hashing and Pruning (DHP)* shrinks candidate sets. While the first scan counts 1-itemsets, it hashes transaction 2-itemsets into buckets. A candidate mapped to a bucket whose bucket count is below `minsup` cannot itself be frequent, because the bucket count upper-bounds the support of every itemset mapped there. Hash collisions can inflate bucket counts, so a high bucket count does not prove frequency; a low bucket count is the useful safe pruning condition.

The slides also mention dynamic itemset counting, support lower-bound pruning, sampling, tree projection, H-miner, and hypercube/LCM-style decomposition as additional Apriori improvements or alternatives. Their common goal is to reduce scans, reduce candidate materialization, or use a data structure better matched to the search space.

=== ECLAT: vertical data and set intersections

Apriori stores transactions horizontally. ECLAT instead maps each item or itemset to its TID list:

$
  t(X) = {"TIDs of transactions containing X"}.
$

Then support counting becomes set intersection:

$
  t(X union Y) = t(X) inter t(Y),
  quad
  "sup"(X union Y) = |t(X) inter t(Y)|.
$

This supports depth-first search without repeatedly scanning the original horizontal database. TID-list relations also carry semantic information: $t(X)=t(Y)$ means $X$ and $Y$ always occur in exactly the same transactions; $t(X) subset t(Y)$ means every transaction containing $X$ also contains $Y$.

A further optimization is the *diffset*: instead of storing a child's full TID list, store what was removed relative to the parent. If $t(e)={T_10,T_20,T_30}$ and $t("ce")={T_10,T_30}$, then the diffset of $"ce"$ relative to $e$ is ${T_20}$. When intersections are large but differences are small, this representation can reduce memory and set-operation cost.

=== FP-growth: compression plus pattern growth

FP-growth avoids explicit candidate generation. Its first scan obtains frequent 1-items. Frequent items are sorted in descending support to form an `f-list`; each transaction is filtered to frequent items and reordered by that global order. A second scan inserts these ordered lists into a prefix tree. Shared prefixes share nodes and node counts accumulate, producing an *FP-tree* plus a header table linking occurrences of the same item.

In the course example with `min_support = 3`, the frequent items are $f:4, c:4, a:3, b:3, m:3, p:3$, and the order is $f-c-a-b-m-p$. Transaction 100 becomes $f,c,a,m,p$, transaction 200 becomes $f,c,a,b,m$, and so on. Repeated prefixes such as $f-c-a$ are represented once with counts.

Mining then uses divide and conquer. For a suffix item $x$, collect the prefix paths leading to $x$; these form the *conditional pattern base* or conditional database of $x$. Build its conditional FP-tree and recursively grow patterns ending in $x$. For example, the slides give

- $p$-conditional database: $"fcam":2, "cb":1$, whose frequent extension includes $c$ with support 3;
- $m$-conditional database: $"fca":2, "fcab":1$, which reduces to $"fca":3$ after local support pruning;
- $b$-conditional database: $"fca":1, f:1, c:1$, which contains no frequent local extension at threshold 3.

A single-path conditional FP-tree is a special case: all nonempty combinations of the nodes on that path can be generated in one shot, with support determined by the minimum count along the chosen path segment. The overall FP-growth recipe is therefore: recursively construct and mine conditional trees until a tree is empty or becomes a single path.

#informally[
  Apriori searches the *candidate lattice* level by level. ECLAT keeps the same pattern lattice but makes support counting cheap through TID intersections. FP-growth changes the search representation more radically: it compresses repeated transaction prefixes and recursively conditions on suffix patterns, so it never materializes the huge global candidate levels that hurt Apriori.
]

== Evaluating Pattern Interestingness

*Exam: ★★★★★*

=== Objective and subjective interestingness

Mining often returns far more valid patterns than a user can inspect. The slides separate *objective* measures, such as support, confidence, lift, and correlation statistics, from *subjective* criteria such as relevance to a user's query, unexpectedness relative to prior knowledge, freshness, and timeliness. Objective measures are reproducible from data; subjective measures depend on the task and user context.

=== Why support and confidence are insufficient

Consider the basketball/cereal contingency table from the slides: among 1000 students, 600 play basketball ($B$), 750 eat cereal ($C$), and 400 do both. The rule $B -> C$ has support $40%$ and confidence

$
  c(B -> C) = 400/600 approx 66.7%.
$

This looks strong under support-confidence thresholds, yet the overall cereal rate is $750/1000=75%$. Knowing that a student plays basketball actually *reduces* the cereal probability from $75%$ to $66.7%$. Confidence therefore needs a baseline comparison.

=== Lift and the chi-square statistic

Lift normalizes confidence by the consequent's base rate:

$
  "lift"(B,C)
  = c(B -> C) / s(C)
  = s(B union C) / (s(B)s(C)).
$

Thus lift $=1$ indicates independence, lift $>1$ positive association, and lift $<1$ negative association. In the example,

$
  "lift"(B,C) = (400/1000)/((600/1000)(750/1000)) approx 0.89,
$

while

$
  "lift"(B,C^c) = (200/1000)/((600/1000)(250/1000)) approx 1.33.
$

The $chi^2$ statistic compares observed contingency-table counts $O$ with expected counts $E$ under independence:

$
  chi^2 = sum_("cells") (O-E)^2 / E.
$

For the same table, expected counts are $450,300,150,100$ and

$
  chi^2
  = (400-450)^2/450
  + (350-300)^2/300
  + (200-150)^2/150
  + (50-100)^2/100
  approx 55.56.
$

A sufficiently large value rejects independence according to the relevant $chi^2$ reference distribution. The statistic itself is nonnegative and does not encode direction; here the negative direction is inferred because the observed $"BC"$ count, 400, is below its independence expectation, 450.

=== Null transactions and null invariance

A *null transaction* for $A$ and $B$ contains neither. In sparse transaction data there may be enormous numbers of such transactions. Lift and $chi^2$ can change dramatically when only the number of null transactions changes. The slides' example has counts $upright("AB")=100$, $upright("AB")^c=1000$, $A^c B=1000$, and $A^c B^c=100000$: intuitively $A$ and $B$ rarely co-occur relative to their one-sided occurrences, yet lift is about $8.44$ and $chi^2$ about $670$ because the huge null cell changes the marginal probabilities and independence baseline.

A measure is *null invariant* if adding or removing $A^c B^c$ transactions leaves its value unchanged. Let

$
  p = s(A union B)/s(A) = P(B|A),
  quad
  q = s(A union B)/s(B) = P(A|B).
$

Both $p$ and $q$ are null invariant. Several useful measures are built only from the co-occurrence and the two one-sided supports:

$
  "AllConf"(A,B) = s(A union B)/max(s(A),s(B)),
$

$
  "Jaccard"(A,B)
  = s(A union B)/(s(A)+s(B)-s(A union B)),
$

$
  "Cosine"(A,B)
  = s(A union B)/sqrt(s(A)s(B)),
$

$
  "Kulc"(A,B)
  = 1/2 (s(A union B)/s(A) + s(A union B)/s(B)),
$

$
  "MaxConf"(A,B)
  = max(s(A union B)/s(A), s(A union B)/s(B)).
$

AllConf, Jaccard, Cosine, Kulczynski, and MaxConf are null invariant; lift and $chi^2$ are not. The measures nevertheless respond differently to severe marginal imbalance. The slides emphasize *Kulczynski* because it averages the two directional implications symmetrically.

=== Kulczynski plus imbalance ratio

The *imbalance ratio* separates directional imbalance from average association strength:

$
  "IR"(A,B)
  = abs(s(A)-s(B))
    / (s(A)+s(B)-s(A union B)).
$

In itemset-mining notation $s(A union B)$ is the support of the combined itemset, i.e. transactions containing both $A$ and $B$; therefore the denominator equals the fraction containing at least one of the two. IR is 0 when the two marginal supports are equal and approaches 1 as they become highly imbalanced.

The D4--D6 examples in the slides are designed so that Kulc stays at $0.5$ while IR rises from $0$ to about $0.89$ and then $0.99$. They are all “neutral” in average directional implication, but the latter datasets are increasingly asymmetric. Hence Kulc and IR together distinguish *strength* from *imbalance*.

The DBLP coauthor example uses this idea to distinguish collaboration structures: advisor-advisee pairs can have high Kulc because a junior author publishes a large fraction of papers with the advisor, while the advisor has many additional papers, producing asymmetric marginal supports. The slides therefore recommend null-invariant measures in sparse settings and particularly Kulc + IR when null transactions dominate.

#warning[
  “Null invariant” does not mean “universally best.” Cosine is null invariant, yet the slides still prefer Kulc + IR in highly imbalanced sparse data because two measures with the same null-invariance property can rank asymmetric relationships differently.
]

== Beyond Flat Frequent Itemsets

*Exam: ★★★★☆*

=== Multilevel and customized-support mining

Real items often live in concept hierarchies, such as `milk -> 2% milk / skim milk`. A uniform minimum support across all hierarchy levels can suppress meaningful specific patterns because lower-level items naturally occur less often. In the slides, Milk has support $10%$, 2% Milk $6%$, and Skim Milk $2%$. A uniform $5%$ threshold keeps Milk and 2% Milk but loses Skim Milk; a reduced lower-level threshold such as $1%$ retains it.

A shared multilevel miner can use the lowest relevant threshold to pass candidates downward, then test each level with its own threshold. This creates a redundancy problem: a descendant rule may merely restate what its ancestor already predicts. If

- `milk -> wheat bread` has support $8%$, confidence $70%$;
- `2% milk -> wheat bread` has support $2%$, confidence $72%$;
- 2% milk accounts for roughly one quarter of milk sales,

then $2%$ support is almost exactly what the ancestor rule predicts, with nearly identical confidence. The descendant rule adds little information and is the natural one to prune as redundant.

More generally, different item groups can have individualized support thresholds. Expensive or rare items such as diamonds and watches may deserve thresholds around $0.05%$, whereas bread and milk may use $5%$. The mining algorithms must respect these heterogeneous thresholds without incorrectly pruning rare-but-important groups.

=== Multidimensional and quantitative associations

A single-dimensional rule repeats one predicate, e.g. `buys(X, milk) -> buys(X, bread)`. A multidimensional rule involves at least two dimensions or predicates. The slides distinguish:

- *inter-dimension* rules, with no repeated predicate, e.g. `age=18--25 and occupation=student -> buys=coke`;
- *hybrid-dimension* rules, where a predicate may repeat, e.g. `age=18--25 and buys=popcorn -> buys=coke`.

Categorical dimensions can be handled through ordinary discrete values or data-cube-style aggregation. Numerical dimensions require a representation choice. The slides list static discretization from a predefined hierarchy, dynamic discretization from the data distribution, one-dimensional clustering followed by association mining, and deviation analysis.

A quantitative rule may describe an unusual aggregate rather than item co-occurrence, for example `Gender=female -> mean wage = $7/hour` when the overall mean is $9 / upright("hour")$. The left-hand side defines a subpopulation and the right-hand side describes an extraordinary behavior of that subset. Such a rule should be accepted only when a statistical test, such as a Z-test, supports the deviation with sufficiently high confidence. Subrules refine the population further, e.g. adding `South=yes` and obtaining mean wage $6.3 / upright("hour")$.

=== Rare and negative patterns

A *rare pattern* simply has low support but may still be valuable; individualized minimum supports are one practical solution. A *negative pattern* instead concerns two individually meaningful itemsets that occur together much less often than expected.

A naive support-based definition compares

$
  s(A union B) quad "with" quad s(A)s(B).
$

This is essentially the lift idea and is not null invariant. In the needle-package example, $A$ and $B$ each occur 100 times and co-occur once. With only 200 total transactions, $s(A union B)=0.005$ while $s(A)s(B)=0.25$, suggesting strong negative correlation. If the same three relevant counts are embedded in $10^5$ transactions by adding null transactions, then $s(A union B)=10^(-5)$ but $s(A)s(B)=10^(-6)$, reversing the conclusion even though the relationship among transactions containing $A$ or $B$ has not changed.

The slides therefore define negative correlation using Kulczynski:

$
  "Kulc"(A,B)
  = 1/2 (s(A union B)/s(A) + s(A union B)/s(B)) < epsilon,
$

for frequent $A$ and $B$. In the same example, each directional conditional probability is $0.01$, so Kulc is $0.01$ regardless of the number of null transactions. A threshold $epsilon$ can therefore identify negative patterns in a null-invariant way.

=== Compressed, approximate, and redundancy-aware pattern sets

Closed and maximal patterns compress by exact set-inclusion/support relationships. The advanced slides introduce a softer pattern distance based on transaction-ID sets:

$
  "Dist"(P_1,P_2)
  = 1 - |T(P_1) inter T(P_2)| / |T(P_1) union T(P_2)|.
$

This is Jaccard distance between the supporting transaction sets. Under $delta$-clustering, a representative pattern $P$ covers patterns that it can express and whose distance to $P$ is at most $delta$. The motivation is to find a middle ground: closed patterns may preserve too many very similar patterns because they insist on exact support distinctions, while maximal patterns may throw away too much support information.

The five-pattern example in the slides illustrates this balance: all five are closed, only one is maximal, but a desired compressed output contains three representative patterns $P_2,P_3,P_4$.

A related selection problem is *redundancy-aware top-$k$ mining*: high-scoring patterns should also add new information relative to patterns already selected. The slides cite Maximal Marginal Significance (MMS) as a set-level objective balancing significance and redundancy. Unlike ordinary top-$k$, which may return several nearly equivalent patterns, redundancy-aware selection aims to cover different informative regions of pattern space.

== Constraint-Based Pattern Mining

*Exam: ★★★★★*

Constraint-based mining treats pattern discovery as an interactive query rather than “mine everything, filter later.” Users may constrain the knowledge type, data subset, dimensions/levels, interestingness thresholds, or the rule/pattern itself. The crucial systems idea is *constraint pushing*: exploit a constraint during search so that irrelevant patterns or data are never explored.

=== Pattern-space anti-monotonicity and monotonicity

A pattern constraint $c$ is *anti-monotone* if violation propagates upward: if $S$ violates $c$, every superset of $S$ also violates it. Search below $S$ can terminate immediately. Examples from the slides include

- $"sum-price"(S) <= v$, assuming prices are nonnegative;
- $"range-profit"(S) <= 15$;
- $"support"(S) >= sigma$, which is exactly the Apriori property.

A constraint is *monotone* if satisfaction propagates upward: if $S$ satisfies $c$, every superset also satisfies it. Examples include

- $"sum-price"(S) >= v$ for nonnegative prices;
- $"min-price"(S) <= v$;
- $"range-profit"(S) >= 15$.

Monotonicity is still useful because once a prefix satisfies the constraint, that particular test need not be repeated on its extensions, but it usually prunes less aggressively than anti-monotonicity.

#example[
  In the constrained Apriori slide, item prices are 1 through 5 and the pattern constraint is $"sum-price"(S) < 5$. Item 5 violates the constraint immediately and can be removed at level 1. Candidates such as ${1,5}$, ${2,5}$, and ${3,5}$ need never be generated or counted. This is constraint pushing rather than post-filtering.
]

=== Convertible constraints and why search order matters

Some constraints are neither globally monotone nor globally anti-monotone but become so if items are processed in a suitable order. The slide example is

$
  "avg-profit"(S) > 20.
$

Order items by descending profit. Along a prefix-growth search in that order, once the running average drops to 20 or below, adding only later items whose profits are no larger cannot raise it above 20. The constraint is therefore *convertible anti-monotone* for that growth order.

However, this argument does not justify ordinary Apriori subset pruning. With profits $a=40$, $g=30$, and $f=-5$,

$
  "avg"("gf")=12.5,
  quad "avg"("af")=17.5,
  quad "avg"("ag")=35,
  quad "avg"("agf") approx 21.7.
$

If Apriori discards $"af"$ and $"gf"$ solely because they violate the average constraint, it will never generate $"agf"$, even though $"agf"$ satisfies it. The conversion is safe when pattern growth respects the specified order and only extends prefixes with lower-valued items; it is not a blanket replacement for Apriori's true subset anti-monotonicity.

=== Data-space anti-monotonicity and recursive pruning

Pattern anti-monotonicity prunes candidate *patterns*. Data anti-monotonicity prunes candidate *transactions*. A constraint is data anti-monotone when, at the current projected search state, a transaction cannot support any extension satisfying the constraint; that transaction can then be removed from further processing.

For $"sum-profit"(S) >= 25$, the slide transaction $T_30={b,c,d,f,g}$ has profits $0,-20,-15,-10,20$. Even its best achievable positive sum is 20, so no subset of that transaction can satisfy the threshold; the entire transaction can be pruned for this constrained search.

Data pruning should be revisited recursively because ordinary frequency pruning can tighten what remains possible. Under constraint $"range-profit"(S) > 25$, the slides inspect the database projected on $b$. Item $a$ is locally infrequent and removed. After that removal, one projected transaction loses the only value that made a profit range above 25 possible, so the transaction itself becomes useless and can be removed. This in turn reduces support for another item. Constraint pruning and support pruning can therefore reinforce each other recursively.

=== Succinct constraints

A constraint is *succinct* if it can be enforced directly by manipulating the data before or during mining. Examples:

- to mine patterns containing item $i$, mine only the $i$-projected database;
- to mine patterns excluding $i$, delete $i$ before mining;
- $"min-price"(S) <= v$ is succinct because qualifying low-price items can be identified directly and transactions containing only higher-price items can be discarded.

By contrast, $"sum-price"(S) >= v$ is not succinct in this sense: whether a pattern reaches the sum threshold depends on how multiple items combine, so the full condition cannot be decided from individual items beforehand.

The constrained FP-growth example uses $"min-price"(S) <= 2$. After removing infrequent item 4, there is no need to start projected searches from items 3 or 5, because a valid pattern must contain at least one item with price at most 2. Only the projected spaces rooted at items 1 or 2 are relevant.

=== Combining constraints

Different constraints may require conflicting item orders. The slides propose prioritizing the constraint with greater pruning power, then reordering within projected databases to exploit another. For example, with $"avg-profit"(S) > 20$ and $"avg-price"(S) < 50$, if the profit constraint is stronger, process items first in descending profit; inside each resulting projected database, reorder appropriately for the price constraint.

The course's summary distinction is therefore:

- *pattern-space pruning*: anti-monotone, monotone, convertible, and some succinct constraints;
- *data-space pruning*: data-succinct and data-anti-monotone constraints.

The same constraint classes later reappear in sequential-pattern mining.

== Sequential Pattern Mining

*Exam: ★★★★★*

=== Sequences, subsequences, and support

A sequence is an ordered list of *elements*, and each element is an unordered set of items/events. For example, `<(ef)(ab)(df)c b>` has five ordered elements; within $(e f)$, items $e$ and $f$ are simultaneous/unordered for the basic sequence model. A sequence $alpha$ is a subsequence of sequence $beta$ if the elements of $alpha$ can be embedded into elements of $beta$ in order, with each element's itemset contained in the matched element of $beta$. The slides illustrate that `<a(bc)dc>` is a subsequence of `<a(abc)(ac)d(cf)>`.

Sequential-pattern mining finds all subsequences whose sequence-level support reaches `minsup`. Support counts *sequences* that contain the pattern, not the number of embeddings inside one sequence. With `min_sup = 2` in the four-sequence example, `<(ab)c>` is frequent.

The slides motivate this representation with ordered customer purchases, treatment workflows, disaster phases, experimental procedures, market movements, and biological DNA/protein sequences: in each case, changing event order can change the meaning of the pattern.

#warning[
  One slide states that sequential pattern mining does not consider the time at which events occur. This is correct for the *basic* formulation, which uses order only. Later slides explicitly add min-gap/max-gap, max-span, and window-size constraints, so the course itself extends the basic model with timing information.
]

The Apriori property still holds under subsequence containment: if a sequence is infrequent, every supersequence containing it is infrequent. This creates the same three algorithmic families seen for itemsets: GSP is Apriori-style, SPADE is vertical-format, and PrefixSpan is pattern-growth. CloSpan adds closed-pattern compression.

=== GSP: Apriori generalized to sequences

GSP begins with frequent singleton sequences, then repeatedly joins frequent length-$k$ sequences to obtain length-$(k+1)$ candidates, prunes candidates whose required subsequences are infrequent, and scans the database for support.

In the slides' example, eight singleton items $a$ through $h$ are considered and `min_sup = 2`; only $a,b,c,d,e,f$ survive. A length-2 sequential candidate can either place two items in separate ordered elements, such as `<ab>` and `<ba>`, or place distinct items in one unordered element, such as `<(ab)>`. With all eight singletons there would be

$
  8 times 8 + (8 times 7)/2 = 92
$

such candidates; after singleton Apriori pruning this falls to

$
  6 times 6 + (6 times 5)/2 = 51.
$

The example then performs successive scans until a single length-5 sequential pattern remains. As with Apriori, the bottleneck is candidate generation and repeated scans.

=== SPADE: vertical sequence representation

SPADE maps occurrences to vertical identifiers $("SID", "EID")$, where SID identifies a sequence and EID identifies an element position within it. Sequence extensions are tested by joining/intersecting these occurrence lists with order constraints on EIDs. For example, a sequential extension `<ab>` requires an occurrence of $a$ at an earlier EID than a matched occurrence of $b$ in the same SID, whereas an itemset extension `<(ab)>` requires compatible occurrences in the same element.

This is the sequential analogue of ECLAT: database scanning is replaced by operations on vertical occurrence lists, and search can proceed depth-first within equivalence classes.

=== PrefixSpan: projected databases and pattern growth

PrefixSpan avoids global candidate generation. First find all frequent length-1 sequences. Then, for each prefix $alpha$, build the $alpha$-projected database consisting of suffixes that remain after an occurrence of $alpha$; mine frequent local extensions and recurse.

For sequence `<a(abc)(ac)d(cf)>`, the slides show prefixes such as `<a>`, `<aa>`, and `<a(ab)>`. In a projected suffix, an underscore marks that the current suffix begins inside the same element as the last matched prefix item, so an itemset-extension remains possible.

The major strengths are that no large level-wise candidate family is generated and projected databases usually shrink with recursion. The major implementation cost is materializing many similar suffix copies. If the database fits in memory, *pseudo-projection* stores a pointer to the original sequence plus an offset rather than copying the suffix. Physical projection is still needed when data must be reorganized on disk; the slides recommend switching to pseudo-projection once a projected database becomes memory-resident.

=== Closed sequential patterns and CloSpan

A sequential pattern $s$ is closed if no proper supersequence $s'$ has the same support. Thus if

- `<abc>`: 20,
- `<abcd>`: 20,
- `<abcde>`: 15,

then `<abc>` is not closed because `<abcd>` has the same support, while the latter two are closed relative to the listed family. Closed sequential patterns reduce redundancy while preserving support information, just as closed itemsets do.

CloSpan mines closed patterns directly by comparing projected databases and applying backward subpattern/superpattern pruning. The key intuition is that a support-preserving extension can make an entire branch redundant: if extending a prefix does not reduce its supporting sequence set in the relevant way, exploring both branches may duplicate the same closed result.

#note[
  The slide states a property in the strong form “if $s$ is a subsequence of $s'$, their projected databases are equal iff the projected databases have the same size.” In practice this statement depends on CloSpan's precise projection representation and containment relation; equal cardinality by itself is not a general set-theoretic guarantee of equality. The safe course takeaway is the pruning idea: support-preserving projected extensions reveal redundancy that CloSpan exploits.
]

=== Constraint-based sequential mining and episodes

The constraint classes transfer naturally:

- anti-monotone: e.g. $"sum-price"(S) < 150$ or $"min-value"(S) > 10$;
- monotone: e.g. element count $>5$ or requiring a pattern to contain a specified itemset;
- data anti-monotone: remove a sequence from a projected database if it cannot support any extension satisfying the constraint;
- succinct: directly select sequences/items satisfying required membership conditions;
- convertible: use a value-based order inside the mining procedure when that conversion is compatible with sequence projection.

Sequential mining also has timing-specific constraints. An *order constraint* requires one event family to precede another. Min-gap/max-gap bounds the separation between matched elements. Max-span bounds the time from the first to last event in a pattern. Window size allows events close in time to be grouped into one logical element even if they were not recorded at exactly the same timestamp.

An alternative formalism is *episode mining*. A serial episode $"AB"$ imposes total order, a parallel episode $A|B$ allows either order, and regular-expression-like templates combine these forms. The slide example $(A|B) C^* (D E)$ allows $A$ and $B$ in either order, any number of $C$ events, and $D,E$ in the same window, while an additional aggregate constraint may require total price above $100$.

== Graph Patterns and Pattern-Mining Applications

*Exam: ★★★★☆*

=== Frequent subgraphs

Graph pattern mining replaces set containment by subgraph containment. Given a labeled graph database

$
  D = {G_1, G_2, dots, G_N},
$

the supporting graph set of subgraph $g$ is

$
  D_g = {G_i : g " is a subgraph of " G_i},
  quad
  "support"(g)=|D_g|/|D|.
$

A subgraph is frequent when its support reaches `minsup`. The course uses both a database of many graphs, such as chemical compounds, and the alternative setting of mining recurring subgraphs inside one large network. Applications include chemical structure discovery, gene/protein/metabolic networks, social and web networks, program execution graphs, graph classification/clustering/compression, indexing, and similarity search.

#warning[
  The graph slides sometimes write `min_sup = 2` next to a relative support such as $67%$. Here `2` is an *absolute supporting-graph count* (two of three graphs), while $67%$ is the equivalent relative support. Keep the units explicit.
]

=== Apriori-style graph mining

The same necessary downward-closure rule holds: if a graph pattern is frequent, each of its subgraphs is frequent. Apriori-style graph miners such as AGM/FSG generate larger candidates by joining frequent smaller graphs, prune candidates whose required subgraphs are infrequent, count support, and eliminate failures. Candidate growth may add one vertex or one edge; the slides report edge-growing as more efficient in the cited setting.

Graph mining adds a difficulty absent from ordinary itemsets: many syntactically different growth histories can produce isomorphic duplicate graphs. Algorithms therefore differ not only in breadth-first versus depth-first search but also in how they canonicalize graphs, eliminate duplicate candidates, store embeddings for support calculation, and decide whether to grow paths, then trees, then general graphs.

#warning[
  One slide states: “A size-$k$ subgraph is frequent *if and only if* all of its subgraphs are frequent.” The reverse implication is false in general. The correct Apriori property is one-way: if a graph is frequent, then all of its subgraphs are frequent; equivalently, one infrequent subgraph is enough to prune a candidate. All subgraphs being frequent is necessary, not sufficient, for the larger graph to be frequent.
]

=== gSpan: ordered pattern growth

gSpan uses depth-first pattern growth instead of level-wise joining. Its central problem is duplicate generation, so it gives each graph a canonical DFS-based representation and extends only according to an ordered rule, commonly described as *right-most path extension*. This turns graph enumeration into a canonical search tree and avoids repeatedly discovering the same isomorphism class through different growth histories.

#warning[
  The slide describes the right-most path as choosing the vertex with the “smallest index at each step.” In standard gSpan terminology, the right-most path is the path in the DFS spanning tree from the root to the *right-most / most recently discovered vertex* under the DFS code. The “smallest index” wording is misleading and should not be memorized as the definition.
]

The important correctness point is completeness: restricting extensions to the canonical right-most scheme still permits every graph pattern to be enumerated through its canonical DFS code.

=== Closed graph patterns and CloseGraph

A frequent graph $G$ is closed if no proper supergraph has the same support. As with itemsets and sequences, direct closed-pattern mining can dramatically reduce output while preserving exact support information for omitted non-closed patterns.

CloseGraph extends gSpan with early termination. Informally, if a frequent supergraph $G_1$ accompanies every embedding/supporting occurrence of $G$, then growing $G$ separately is redundant except for special cases handled by the algorithm: closed descendants of $G$ can be represented through the branch rooted at $G_1$. This is the graph analogue of support-preserving closure pruning.

=== Graph indexing and approximate similarity search

A graph-containment query asks for database graphs containing a query graph $Q$. Path-only indexes can be weak because different graphs may share all short path features while only one contains the full query substructure. The slides therefore motivate indexing *substructures* directly.

`gIndex` selects frequent and discriminative subgraphs. Indexing every subgraph is impossible, so support thresholds may increase with pattern size: large structures can often be represented sufficiently by smaller indexed substructures. Given already selected features $f_1,dots,f_n$, a candidate structure $x$ is valuable when it provides extra discrimination, expressed in the slides through a low conditional probability

$
  "Pr"(x | f_1, f_2, dots, f_n),
$

under the condition that the $f_i$ occur as substructures. If this probability is small, knowing the existing features does not make $x$ redundant, so $x$ is a useful additional index feature.

For approximate substructure search, sequentially comparing the query against every database graph is too costly, while indexing every possible approximate subgraph would explode. The slides instead decompose the query into features and represent graphs as binary feature vectors. If a relaxation threshold permits at most two missing features and the query has five selected features, any database graph missing more than two can be pruned before expensive graph verification.

=== Application: copy-paste bug detection with sequential patterns

CP-Miner treats copied code as repeated sequences. Source statements are tokenized: operators, constants, and keywords receive distinct token types, while identifiers of the same type are normalized so renaming does not destroy structural similarity. Token sequences are hashed to statement IDs, programs become long sequences, and block boundaries produce a sequence database.

The sequential miner is modified with a maximum gap so copied blocks can contain inserted statements. Neighboring matched segments are repeatedly composed into larger copy-paste regions. The final bug detector aligns identifiers across copies and looks for mapping conflicts. If most occurrences are consistently renamed but a small nonzero fraction remains unchanged, the unchanged token is a candidate “forget-to-change” bug.

#warning[
  In the code example, the second copied block correctly changes `total[i].adr` and `total[i].bytes` to `taken[...]`, but leaves `taken[i].more = &total[i+1]`. The slide annotation says the programmer forgot to change “id”; the concrete mismatch is actually the retained identifier `total` in the last line.
]

=== Application: phrase mining from frequent contiguous patterns

Phrase mining addresses unigram ambiguity: `United` alone can refer to multiple entities, whereas `United States` or `United Airlines` is a more specific semantic unit. The general strategy is to exploit corpus redundancy to find boundaries and phrase salience, combining frequent contiguous-pattern mining, collocation statistics, segmentation, and phrase-quality assessment.

A classical collocation measure is pointwise mutual information:

$
  "PMI"(x,y) = log(p(x,y)/(p(x)p(y))).
$

The slides also show standardized significance-style scores comparing observed phrase frequency to an independence/null expectation. A representative form is

$
  "sig"(P_1,P_2)
  approx (f(P_1 dot P_2)-mu_0(P_1,P_2))/sqrt(f(P_1 dot P_2)),
$

where $f(P_1 dot P_2)$ counts the contiguous concatenation and $mu_0$ is its expected frequency under a null model. Related tests include $t$-tests, $z$-tests, $chi^2$, likelihood ratios, and mutual information.

ToPMine's pipeline in the slides is:

- mine frequent contiguous word patterns and raw counts;
- greedily/agglomeratively merge adjacent units using a significance score;
- segment documents into phrases and recount occurrences;
- compute *rectified* phrase frequency so occurrences already absorbed into a longer phrase do not automatically inflate all subphrases;
- rank phrases using popularity, concordance, informativeness, and completeness;
- feed the resulting bag of phrases into PhraseLDA, constraining words in a phrase to share a latent topic.

The example explains why segmentation matters: if `support vector machine` is recognized as one phrase, that occurrence should not also count as independent evidence for `support`, `vector`, `support vector`, or `vector machine`. Similarly, `feature selection` can be a phrase while the boundary `selection for` is rejected when its significance is weak. The DBLP and Yelp experiments in the slides are presented as evidence that ToPMine can produce coherent multiword topics efficiently without labeled training data, including on social-media-style text.

#note[
  The slides position later phrase-mining systems as a supervision spectrum: ToPMine uses no training labels, SegPhrase uses a small labeled set, and AutoPhrase uses distant supervision such as Wikipedia.
]

== Closing Perspective

*Exam: ★★★☆☆*

Across itemsets, sequences, and graphs, the recurring architecture is the same. A pattern language defines containment and support; downward closure supplies a safe pruning direction; data layout determines whether support is counted by repeated scans, vertical intersections, or projected databases; closedness and redundancy-aware selection control output explosion; and task-specific constraints are pushed into search whenever they have monotone, anti-monotone, convertible, succinct, or data-pruning structure.

The main conceptual progression is therefore not a list of unrelated algorithms. Apriori, GSP, and Apriori-style graph mining are the same level-wise idea under different containment relations. ECLAT and SPADE are the same verticalization idea. FP-growth and PrefixSpan are the same projection-and-growth idea. Closed itemsets, CloSpan, and CloseGraph apply the same lossless compression principle to increasingly structured pattern spaces. Finally, interestingness measures and user constraints answer the question that pure frequency cannot: among all patterns that *exist*, which patterns are worth finding and reporting?
