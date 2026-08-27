#import "../template.typ": *

= Pattern Mining: Basic Concepts and Methods

== Overview

*Exam: ★★★★☆*

Pattern mining asks how to discover recurring or strongly associated structures from large data sets. A pattern may be an itemset, a subsequence, or a more general substructure; such patterns summarize intrinsic regularities that can support association and correlation analysis, sequential or structural mining, discriminative classification, pattern-based clustering, and applications such as market-basket analysis, cross-marketing, Web-log analysis, and biological-sequence analysis. The same ideas extend beyond transaction tables to spatiotemporal, multimedia, time-series, and stream data.

The chapter follows one central progression. We first formalize transactions, itemsets, support, and association rules. Because the number of frequent patterns can be exponential, we then study compressed representations. Efficient mining starts from the anti-monotonicity of support, leading to Apriori and its pruning strategies; partitioning, hashing, vertical TID-lists, and FP-growth address the remaining database-scan and candidate-generation bottlenecks. Finally, because frequent or high-confidence patterns are not necessarily interesting, we move from support and confidence to correlation-aware and null-invariant evaluation measures.

== Transactions, Frequent Itemsets, and Association Rules

*Exam: ★★★★★*

=== Transactional data and support

Let $I$ be the universe of items and let a transactional database be $D = {T_1, dots, T_N}$, where each transaction $T_i$ has a transaction identifier (TID) and contains a subset of $I$. A transaction may additionally store quantities, although the basic frequent-itemset model uses only item presence or absence. An *itemset* $X subset.eq I$ contains one or more items; if $|X| = k$, it is a $k$-itemset.

The *absolute support* or *support count* of $X$ is the number of transactions containing every item in $X$:

$
  "supp"_D(X) = |{T in D: X subset.eq T}|.
$

The *relative support* is

$
  s(X) = "supp"_D(X) / |D|,
$

which can be read as the empirical probability that a transaction contains $X$. A pattern is *frequent* when its support is at least a user-specified minimum-support threshold $sigma$. Slides use both relative thresholds, such as $sigma = 50%$, and absolute thresholds, such as a minimum support count of $1$ or $2$; the convention must therefore be checked before comparing numbers.

#example(title: "Support in the five-transaction example")[
  The lecture database is

  #table(
    columns: (0.7fr, 3fr),
    inset: 4pt,
    stroke: 0.4pt,
    [*TID*], [*Items*],
    [1], [Beer, Nuts, Diaper],
    [2], [Beer, Coffee, Diaper],
    [3], [Beer, Diaper, Eggs],
    [4], [Nuts, Eggs, Milk],
    [5], [Nuts, Coffee, Diaper, Eggs, Milk],
  )

  Hence $"supp"("Beer") = 3$, $"supp"("Diaper") = 4$, $"supp"({"Beer", "Diaper"}) = 3$, and $"supp"({"Beer", "Eggs"}) = 1$. Their relative supports are respectively $60%$, $80%$, $60%$, and $20%$.

  With $sigma = 50%$, the frequent 1-itemsets are Beer, Nuts, Diaper, and Eggs; the only frequent 2-itemset is ${"Beer", "Diaper"}$; there is no frequent itemset of size at least three.
]

=== From itemsets to association rules

A frequent itemset says that items occur together; an *association rule* adds direction, written $X -> Y$. Its support is the support of the combined itemset $X union Y$, while its *confidence* measures how often $Y$ is present among transactions that already contain $X$:

$
  c(X -> Y)
  = "supp"(X union Y) / "supp"(X)
  = s(X union Y) / s(X)
  = P(Y | X).
$

Given thresholds $"minsup"$ and $"minconf"$, association-rule mining asks for every rule whose support and confidence satisfy both thresholds. The natural decomposition is therefore: first mine frequent itemsets; then generate and test rules from those itemsets.

#example(title: "Rules from Beer and Diaper")[
  With $"minsup" = 50%$ and $"minconf" = 50%$, the frequent pair ${"Beer", "Diaper"}$ produces

  - Beer $->$ Diaper: support $60%$, confidence $3/3 = 100%$;
  - Diaper $->$ Beer: support $60%$, confidence $3/4 = 75%$.

  These are indeed all rules satisfying both thresholds in this database: any qualifying rule must have a frequent union containing at least two items, and ${"Beer", "Diaper"}$ is the only such frequent itemset.
]

#warning[
  An early slide states that support and confidence ensure both “popularity and correlation.” Support measures prevalence and confidence is a conditional co-occurrence rate, but high confidence does *not* by itself establish positive correlation. The later cereal/basketball example in the same lecture explicitly demonstrates this limitation.
]

== Pattern Explosion and Compressed Representations

*Exam: ★★★★☆*

Frequent-pattern output can itself become intractable. Consider $T_1 = {a_1, dots, a_50}$ and $T_2 = {a_1, dots, a_100}$ with minimum support count $1$. Every nonempty subset of $T_2$ occurs in at least one transaction, so the database contains

$
  sum_(k=1)^100 binom(100, k) = 2^100 - 1
$

frequent itemsets. This is too large to enumerate or store, motivating compressed descriptions of the frequent-pattern family.

=== Closed patterns

A frequent itemset $X$ is *closed* if there is no proper super-itemset $Y$ with $X subset Y$ and the same support as $X$. Closed patterns retain exact frequency information: although many non-closed itemsets are omitted, their support can be recovered from a closed superpattern having the same transaction set. In this sense, the set of frequent closed patterns is a lossless compression of frequent itemsets with respect to support.

For the two-transaction database above, there are only two closed patterns: ${a_1, dots, a_50}$ with support $2$, and ${a_1, dots, a_100}$ with support $1$. For example, ${a_2, dots, a_40}$ has support $2$ because it shares its support with the first closed pattern, while ${a_5, a_51}$ has support $1$ because it is contained only in the second transaction.

#warning[
  The “Closed Patterns” example slide sets $"minsup" = 2$ and shows ${"Diaper", "Milk", "Bread", "Beer"}:1$ immediately after the closed-pattern definition. If this line is intended as a second closed pattern, it is incorrect because support $1$ is below minimum support. The safe interpretation is that it is merely the support of a superpattern, showing that the frequent pattern ${"Diaper", "Milk", "Bread"}:2$ has no equal-support superpattern and is therefore closed.
]

=== Maximal frequent patterns

A frequent itemset $X$ is *maximal* if it has no proper super-itemset that is also frequent. In the same two-transaction example with minimum support count $1$, only ${a_1, dots, a_100}$ is maximal. Maximal patterns therefore compress more aggressively than closed patterns, but the compression is lossy: if $X$ is contained in a maximal pattern, we can infer that $X$ is frequent, but not its exact support.

#table(
  columns: (1.1fr, 2.3fr, 2.3fr),
  inset: 5pt,
  stroke: 0.4pt,
  [*Aspect*], [*Closed frequent patterns*], [*Maximal frequent patterns*],
  [Definition], [Frequent; no proper superset has the same support], [Frequent; no proper superset is frequent],
  [Compression], [Lossless for support information], [Lossy for subset supports],
  [Output size], [Smaller than all frequent itemsets, but may retain many patterns], [Usually the smallest boundary representation],
  [Best use], [Analysis and queries requiring exact frequencies], [Extreme output reduction when only frequent/not-frequent status is needed],
)

The lecture therefore favors closed patterns in applications where frequency information matters, while maximal patterns are attractive when storage reduction is the overriding goal. The slides name CHARM and CLOSET+ as later algorithms for closed-itemset mining, but do not develop their procedures in this chapter.

== Downward Closure and the Apriori Algorithm

*Exam: ★★★★★*

=== Anti-monotonicity of support

The key structural property behind scalable frequent-itemset mining is the *downward-closure* or *Apriori* property. If $X subset.eq Y$, every transaction containing $Y$ also contains $X$, so

$
  "supp"(X) >= "supp"(Y).
$

Consequently, every subset of a frequent itemset must be frequent. The contrapositive is the useful pruning rule: if any subset of a candidate is infrequent, then the candidate and all of its supersets are necessarily infrequent. This avoids exploring large regions of the itemset lattice.

#note[
  Three scalable families highlighted in the slides are: level-wise join-and-test mining (Apriori), vertical TID-list intersection (ECLAT), and frequent-pattern projection/growth (FP-growth). They exploit the same support structure but attack the computational bottleneck differently.
]

=== Level-wise candidate generation and testing

Apriori proceeds breadth-first by itemset length. Let $F_k$ be the frequent $k$-itemsets and $C_k$ the candidate $k$-itemsets.

- Scan the database to obtain $F_1$.
- While $F_k$ is nonempty, self-join compatible members of $F_k$ to form tentative $(k+1)$-itemsets.
- Prune any candidate having an infrequent $k$-subset.
- Scan the database to count the survivors and retain those meeting minimum support as $F_(k+1)$.
- Return $union_k F_k$ when no further frequent level can be produced.

If items inside each member of $F_(k-1)$ are kept in a common order, two itemsets $p$ and $q$ can be joined when their first $k-2$ items agree and their final items differ in the prescribed order. The joined candidate is admitted to $C_k$ only if *every* $(k-1)$-subset belongs to $F_(k-1)$.

#example(title: "Apriori on the lecture database")[
  For

  #table(
    columns: (0.8fr, 2.6fr),
    inset: 4pt,
    stroke: 0.4pt,
    [*TID*], [*Items*],
    [10], [A, C, D],
    [20], [B, C, E],
    [30], [A, B, C, E],
    [40], [B, E],
  )

  with minimum support count $2$, the first scan gives
  $F_1 = {A:2, B:3, C:3, E:3}$; item $D$ is discarded with support $1$.

  The six pair candidates have counts $"AB":1$, $"AC":2$, $"AE":1$, $"BC":2$, $"BE":3$, and $"CE":2$. Therefore
  $F_2 = {"AC":2, "BC":2, "BE":3, "CE":2}$.

  At the next level, the only surviving 3-item candidate is $"BCE"$: all of its 2-subsets $"BC"$, $"BE"$, and $"CE"$ are frequent. Its support is $2$, so $F_3 = {"BCE":2}$. No larger candidate survives.
]

A separate candidate-generation example makes the pruning step explicit. If
$F_3 = {"abc", "abd", "acd", "ace", "bcd"}$, self-joining produces $"abcd"$ from $"abc"$ and $"abd"$, and $"acde"$ from $"acd"$ and $"ace"$. Candidate $"acde"$ is pruned because its subset $"ade"$ is absent from $F_3$, leaving $C_4 = {"abcd"}$.

=== Why Apriori still becomes expensive

Downward closure can remove enormous parts of the search space, but Apriori may still require many full database passes and may generate huge candidate sets before discovering that most candidates are infrequent. The slides group common improvements into three directions:

- reduce database scans: partitioning and dynamic itemset counting;
- shrink candidate sets: hashing/DHP, support-bound pruning, and sampling;
- change the search data structure: tree projection, H-miner, and the decomposition method associated with LCM in the slide.

#warning[
  The slide text spells the last item “Hypecube decomposition.” This appears to be a typographical issue; because the lecture does not define or develop that term, these notes do not silently replace it with a different algorithmic concept.
]

== Scaling Frequent-Pattern Mining

*Exam: ★★★★☆*

=== Partitioning: two database scans

Partitioning reduces repeated I/O by dividing $D$ into partitions $D_1, dots, D_m$ that fit in main memory and using the same relative minimum-support threshold $sigma$ locally.

#theorem(title: "Partition pruning principle")[
  If an itemset is globally frequent in $D$, then it must be frequent in at least one partition $D_i$.
]

#proof[
  Suppose $X$ were infrequent in every partition. Then for every $i$,
  $"supp"_(D_i)(X) < sigma |D_i|$. Summing over partitions gives
  $"supp"_D(X) < sigma sum_i |D_i| = sigma |D|$, contradicting global frequency. Therefore every globally frequent itemset must appear among the union of locally frequent itemsets.
]

This yields a two-scan method. During the first scan, partition the database and mine the local frequent itemsets of each in-memory partition; their union forms a global candidate set. During the second scan, count only these candidates over the entire database and retain those whose true global support reaches the threshold.

=== Direct Hashing and Pruning (DHP)

DHP uses hashing to eliminate candidates before explicit support counting. During the first scan, while counting 1-itemsets, all 2-itemsets occurring inside each transaction are hashed into buckets. A bucket count is an upper bound on the support of any individual itemset mapped into that bucket. Therefore, if a bucket count is below minimum support, every 2-itemset hashing to it can be safely removed from $C_2$.

#informally[
  Hash collisions are one-sided for pruning. Several different itemsets may share a bucket, so a *large* bucket count does not prove that any one itemset is frequent. A *small* bucket count is decisive: no member of that bucket can possibly reach minimum support.
]

The lecture example hashes itemsets such as $"ab"$, $"ad"$, and $"ce"$ into a bucket of count $35$. If minimum support is $80$, all candidates mapped to that bucket are pruned without separate counting.

=== ECLAT and vertical data format

ECLAT changes representation rather than repeatedly scanning horizontal transactions. For each item or itemset $X$, maintain its TID-list $t(X)$, the set of transaction identifiers containing $X$. Support counting becomes set intersection:

$
  t(X union Y) = t(X) inter t(Y),
  quad "supp"(X union Y) = |t(X) inter t(Y)|.
$

The search is depth-first. Equality $t(X) = t(Y)$ means that $X$ and $Y$ occur in exactly the same transactions; inclusion $t(X) subset t(Y)$ means every transaction containing $X$ also contains $Y$. In the lecture example, $t({a,c}) = t({d}) = {T_10}$ and $t({a,c}) subset t({c,e}) = {T_10, T_30}$.

A *diffset* stores only TIDs lost when extending a pattern. For example,
$t(e) = {T_10, T_20, T_30}$ and $t({c,e}) = {T_10, T_30}$, so the diffset of $"ce"$ relative to $e$ is ${T_20}$. This can reduce memory and intersection cost when TID-lists are long but successive extensions differ only slightly.

== FP-Growth: Mining Without Candidate Generation

*Exam: ★★★★★*

Apriori spends work generating and testing candidates. FP-growth instead compresses the database into an FP-tree and recursively mines *conditional databases*. Its core strategy is divide-and-conquer: choose a frequent suffix item, collect the prefix paths leading to that item, compress those paths into a conditional FP-tree, and grow the suffix by frequent items found there.

=== Building the FP-tree

For the lecture database below, minimum support count is $3$.

#table(
  columns: (0.8fr, 4fr),
  inset: 4pt,
  stroke: 0.4pt,
  [*TID*], [*Transaction*],
  [100], [f, a, c, d, g, i, m, p],
  [200], [a, b, c, f, l, m, o],
  [300], [b, f, h, j, o, w],
  [400], [b, c, k, s, p],
  [500], [a, f, c, e, l, p, m, n],
)

The first scan keeps the frequent single items $f:4$, $c:4$, $a:3$, $b:3$, $m:3$, and $p:3$. The slide fixes the global frequency order
$f - c - a - b - m - p$; ties may be broken consistently, but every transaction must use the same order. Infrequent items are discarded, giving ordered frequent lists such as

- T100: $f,c,a,m,p$;
- T200: $f,c,a,b,m$;
- T300: $f,b$;
- T400: $c,b,p$;
- T500: $f,c,a,m,p$.

The second scan inserts these lists into a prefix tree. Shared prefixes share nodes and node counts accumulate. The final structure contains a root branch $f:4$ and another root branch $c:1$; under $f:4$, the main chain is $c:3 -> a:3$, with $a$ branching to $m:2 -> p:2$ and $b:1 -> m:1$, while $f$ also has a $b:1$ child. The root-level $c:1$ branch continues $b:1 -> p:1$. A header table stores item frequencies and links together nodes carrying the same item so that conditional paths can be collected efficiently.

=== Conditional pattern bases and recursive growth

For an item $x$, its conditional database is the multiset of prefix paths ending immediately before $x$, weighted by the count of the corresponding $x$ node. From the lecture FP-tree:

#table(
  columns: (0.7fr, 3.3fr),
  inset: 4pt,
  stroke: 0.4pt,
  [*Suffix*], [*Conditional database / prefix paths*],
  [$c$], [$f:3$],
  [$a$], [$"fc":3$],
  [$b$], [$"fca":1, f:1, c:1$],
  [$m$], [$"fca":2, "fcab":1$],
  [$p$], [$"fcam":2, "cb":1$],
)

With minimum support $3$, the conditional database of $p$ retains only $c:3$, yielding the larger pattern ${c,p}:3$. The conditional database of $b$ has no prefix item reaching support $3$, so $b$ has no frequent extension in that conditional tree. For $m$, the paths $"fca":2$ and $"fcab":1$ reduce to the shared frequent path $"fca":3$. Appending suffix $m$ produces

$m:3$, ${f,m}:3$, ${c,m}:3$, ${a,m}:3$, ${f,c,m}:3$, ${f,a,m}:3$, ${c,a,m}:3$, and ${f,c,a,m}:3$.

This illustrates the important single-path shortcut: if a conditional FP-tree consists of one path, every nonempty combination of items along that path is a frequent extension of the current suffix, so the combinations can be emitted directly instead of creating another layer of conditional trees.

The slide also depicts a more general *shared prefix path* before a branching remainder. Such a prefix can be factored out: mine the branching remainder once, then combine its results with subsets of the shared prefix. This is another way FP-growth avoids redundant recursive work.

#note[
  FP-growth still begins with support counting and therefore still relies on the same frequency threshold as Apriori. Its advantage is not a different definition of frequency, but avoiding explicit candidate generation and compressing repeated transaction prefixes.
]

== Pattern Evaluation Beyond Support and Confidence

*Exam: ★★★★★*

Pattern mining can generate many valid rules, so a second question is which patterns are worth interpreting. The slides distinguish *objective* interestingness measures, computed from data (support, confidence, correlation measures), from *subjective* criteria that depend on a user or task, such as query relevance, unexpectedness relative to prior knowledge, freshness, or timeliness.

=== Why confidence can be misleading

Let $B$ denote “plays basketball” and $C$ denote “eats cereal.” The lecture contingency table is

#table(
  columns: (1.4fr, 1fr, 1fr, 1fr),
  inset: 4pt,
  stroke: 0.4pt,
  [], [$B$], [not $B$], [Row total],
  [$C$], [400], [350], [750],
  [not $C$], [200], [50], [250],
  [Column total], [600], [400], [1000],
)

The rule $B -> C$ has support $400/1000 = 40%$ and confidence $400/600 = 66.7%$. Those values look large, yet the overall cereal rate is $s(C)=75%$. Knowing that a student plays basketball actually *reduces* the cereal probability from $75%$ to $66.7%$. Conversely, not playing basketball gives $P(C | not B)=350/400=87.5%$. Confidence therefore needs a baseline comparison.

=== Lift

Lift compares observed co-occurrence with the level expected under independence:

$
  "lift"(B,C)
  = c(B -> C) / s(C)
  = s(B union C) / (s(B) s(C)).
$

A lift of $1$ corresponds to independence, a value above $1$ to positive association, and a value below $1$ to negative association. In the example,

$
  "lift"(B,C) = (400/1000) / ((600/1000)(750/1000)) approx 0.89,
$

while

$
  "lift"(B, not C) = (200/1000) / ((600/1000)(250/1000)) approx 1.33.
$

Thus basketball and cereal are negatively associated in this table, while basketball and not eating cereal are positively associated.

=== Chi-square test of independence

For a contingency table, the expected count under independence is $E_(i,j) = r_i c_j / N$, and the Pearson statistic is

$
  chi^2 = sum_(i=1)^4 (O_i - E_i)^2 / E_i.
$

For the cereal/basketball table, the expected cell counts are $450$, $300$, $150$, and $100$, giving

$
  chi^2
  = (400-450)^2/450
  + (350-300)^2/300
  + (200-150)^2/150
  + (50-100)^2/100
  approx 55.56.
$

This is strong evidence against independence when compared with the relevant $chi^2$ reference distribution.

#warning[
  The slide says the “$chi^2$-test shows B and C are negatively correlated.” The $chi^2$ statistic itself is nonnegative and tests *whether* independence is violated; it does not encode the direction of association. The negative direction here comes from the residual $O_("BC")-E_("BC")=400-450<0$ (or equivalently from lift $<1$), not from the sign of $chi^2$.
]

== Null Invariance and Choosing an Interestingness Measure

*Exam: ★★★★★*

A *null transaction* with respect to $A$ and $B$ contains neither itemset. In very large sparse transaction data, null transactions can dominate: most baskets may contain neither milk nor coffee, and most papers may contain neither of two particular authors. A measure is *null-invariant* if adding or removing such transactions does not change its value.

The lecture constructs an extreme table with $"BC"=100$, $B(not C)=1000$, $(not B)C=1000$, and $(not B)(not C)=100000$. Although joint occurrence $"BC"$ is small in absolute terms, the huge null count changes the marginal baseline so strongly that $"lift"(B,C) approx 8.44$ and $chi^2 approx 670$; the expected $"BC"$ count under independence is only about $11.85$. The example motivates measures that ignore the null cell when null frequency is an incidental property of the data set.

=== Null-invariant measures

Define the two directional confidences

$
  p = s(A union B)/s(A) = P(B|A),
  quad
  q = s(A union B)/s(B) = P(A|B).
$

Both $p$ and $q$ are null-invariant because their numerators and denominators involve only transactions containing $A$ or $B$. Several useful symmetric measures can be expressed through them.

#table(
  columns: (1.15fr, 2.9fr, 0.85fr, 0.95fr),
  inset: 4pt,
  stroke: 0.4pt,
  [*Measure*], [*Definition*], [*Range*], [*Null-invariant?*],
  [$chi^2$], [$sum_i (O_i-E_i)^2/E_i$], [$0 <= "value" < infinity$], [No],
  [Lift], [$s(A union B)/(s(A)s(B))$], [$0 <= "value" < infinity$], [No],
  [AllConf], [$s(A union B)/max(s(A),s(B)) = min(p,q)$], [$[0,1]$], [Yes],
  [Jaccard], [$s(A union B)/(s(A)+s(B)-s(A union B))$], [$[0,1]$], [Yes],
  [Cosine], [$s(A union B)/sqrt(s(A)s(B)) = sqrt(p q)$], [$[0,1]$], [Yes],
  [Kulczynski], [$1/2 (p+q)$], [$[0,1]$], [Yes],
  [MaxConf], [$max(p,q)$], [$[0,1]$], [Yes],
)

#note[
  The slide describes the null-invariant family as “essentially min, max, mean variants of $p,q$.” This is exact for AllConf, MaxConf, and Kulczynski. Cosine is the geometric mean $sqrt(p q)$, while Jaccard is $p q/(p+q-p q)$; the slide phrase should therefore be read as intuition rather than a literal formula for every measure.
]

=== Why Kulczynski needs an imbalance companion

Null invariance alone does not make all measures equivalent. The lecture uses data sets $D_4$--$D_6$ in which the two directional implications become increasingly asymmetric. If the shared count is $1000$:

- $D_4$ has $1000$ occurrences of each one-sided case, so $p=q=0.5$;
- $D_5$ has one-sided counts $100$ and $10000$, giving approximately $p=0.09$ and $q=0.91$;
- $D_6$ has one-sided counts $10$ and $100000$, giving approximately $p=0.01$ and $q=0.99$.

Kulczynski remains $"Kulc"=(p+q)/2 approx 0.5$ in all three cases, which correctly says that the *average* directional association is neutral, but this alone hides the increasing asymmetry. The *imbalance ratio* supplies that missing information:

$
  "IR"(A,B)
  = abs(s(A)-s(B)) / (s(A)+s(B)-s(A union B)).
$

The slide reports $"IR"=0$ for $D_4$, about $0.89$ for $D_5$, and about $0.99$ for $D_6$. Thus Kulczynski plus IR distinguishes “neutral and balanced” from “neutral but highly imbalanced.”

#warning[
  The side annotation on the IR slide writes $s(A union B)=s(A)+s(B)-s(A inter B)$. This mixes two meanings of set operations. In itemset notation, $s(A union B)$ is the support of transactions containing *both* itemsets, i.e. the intersection of their transaction events. Inclusion--exclusion instead applies to the event “contains $A$ or $B$”: its probability is $s(A)+s(B)-s(A union B)$. The IR denominator above is consistent with this intended event-union normalization.
]

#example(title: "DBLP coauthor relationships")[
  The lecture uses DBLP bibliographic data to illustrate this idea. Coauthor pairs in an advisor--advisee relationship can have a high Kulczynski score even when their publication totals are very different; Jaccard may be low and cosine intermediate. The combination of a symmetric association score and imbalance information can therefore help separate close collaborators from strongly directional relationships. The slide reports DBLP as having more than 3.8 million bibliographic entries at the time represented by the course material.
]

#warning[
  The final recommendation slide groups cosine together with lift and $chi^2$ as useful when null transactions are not predominant, while an earlier table correctly marks cosine as *null-invariant*. These claims are not logically contradictory if the recommendation is read as a practical heuristic: cosine is unaffected by the null cell, but it can still behave differently from Kulczynski under strong directional imbalance. Null sensitivity is a problem for lift and $chi^2$, not for cosine.
]

A practical reading of the lecture is therefore: use support and confidence as basic prevalence/conditional-frequency filters; use lift or a statistical independence test when null counts are meaningful and not overwhelming; in sparse transactional settings where the null cell is dominated by irrelevant absence, prefer null-invariant measures. When asymmetric implications matter, Kulczynski together with IR gives both average association strength and directional imbalance.

== Chapter Synthesis

*Exam: ★★★★☆*

The chapter contains two distinct optimization problems. *Mining efficiency* asks how to enumerate all patterns meeting minimum support without traversing the full exponential itemset lattice: downward closure enables Apriori pruning, partitioning and hashing reduce I/O or candidates, ECLAT replaces scans with set intersections, and FP-growth compresses repeated prefixes and grows patterns conditionally. *Pattern evaluation* asks which of the valid outputs are meaningful: support and confidence alone are insufficient, lift and $chi^2$ compare against independence but depend on the null cell, and null-invariant measures remove that dependence. Closed and maximal patterns address a third bottleneck--the sheer number of outputs--by compressing the frequent-pattern family with different information-loss trade-offs.

#note(title: "Readings named in the lecture slides")[
  *Basic concepts and closed patterns:* Agrawal, Imielinski & Swami (SIGMOD 1993); Bayardo (SIGMOD 1998); Pasquier et al. (ICDT 1999); Han, Cheng, Xin & Yan (Data Mining and Knowledge Discovery, 2007).

  *Efficient mining:* Agrawal & Srikant (VLDB 1994); Savasere, Omiecinski & Navathe (VLDB 1995); Park, Chen & Yu (SIGMOD 1995); Toivonen (1996); Brin et al. (1997, dynamic itemset counting); Zaki et al. (1997, vertical mining); Sarawagi, Thomas & Agrawal (SIGMOD 1998); Han, Pei & Yin (SIGMOD 2000, FP-growth); Agarwal et al. (2001, tree projection); Pei et al. (2001, H-miner); Zaki & Hsiao (SDM 2002, CHARM); Wang, Han & Pei (KDD 2003, CLOSET+); Uno et al. (2004, LCM); Aggarwal, Bhuiyan & Hasan (2014 survey).

  *Pattern evaluation:* Klemettinen et al. (CIKM 1994); Brin, Motwani & Silverstein (SIGMOD 1997); Aggarwal & Yu (PODS 1998); Tan, Kumar & Srivastava (KDD 2002); Omiecinski (TKDE 2003); Wu, Chen & Han (Data Mining and Knowledge Discovery, 2010). The DBLP advisor--advisee example cites Wang et al., “Mining Advisor-Advisee Relationships from Research Publication Networks” (KDD 2010).
]
