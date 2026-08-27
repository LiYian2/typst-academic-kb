#import "../template.typ": *

= Cluster Analysis: Basic and Advanced Methods

== Overview

*Exam: ★★★★☆*

Cluster analysis groups unlabeled objects so that within-cluster similarity is high and between-cluster similarity is low. The central difficulty is that a "cluster" is not unique: the same data may admit several meaningful resolutions and several notions of structure. This chapter therefore progresses from cluster definitions to prototype-based K-means, hierarchical linkage, density-based DBSCAN, fuzzy/probabilistic/subspace/graph extensions, and finally cluster validation.

== Cluster Structure and Problem Formulation

*Exam: ★★★★☆*

A *partitional clustering* divides objects into non-overlapping subsets; a *hierarchical clustering* produces nested clusters represented by a dendrogram. Exclusive clustering assigns one cluster per point, while non-exclusive clustering allows multiple memberships. Fuzzy clustering uses $w_ij in [0,1]$ with $sum_(j=1)^k w_ij=1$. Complete clustering assigns every object; partial clustering may leave noise unassigned.

The lecture uses five cluster notions:

- *Well-separated:* every point is closer to every point in its own cluster than to any outside point.
- *Prototype-based:* assignment is determined by closeness to a representative/centroid.
- *Connectivity-based:* points are joined through chains of nearby neighbors.
- *Density-based:* dense regions are separated by low-density regions.
- *Objective/model-based:* clusters optimize a criterion or fit a parameterized statistical model.

Thus dimensionality, sparsity, attribute type and scale, distribution, autocorrelation, noise/outliers, and differences in cluster size, density, shape, and separation directly affect which method is appropriate.

== K-means and Prototype-Based Extensions

*Exam: ★★★★★*

K-means is a complete, exclusive, partitional method requiring $k$. With Euclidean distance it minimizes

$
  "SSE" = sum_(j=1)^k sum_(x in C_j) norm(x-m_j)^2,
$

where $m_j$ is the centroid of $C_j$. Starting from $k$ centers, repeatedly assign every point to its nearest center and update $m_j=1/|C_j| sum_(x in C_j)x$. Each step cannot increase SSE, but convergence is only to a local optimum/fixed point.

#warning[
The slides say SSE improves until a "local or global minima." K-means does not guarantee the global minimum.
]

Initialization matters. If $k$ equally sized true clusters each contain $n$ points, the probability that $k$ independently sampled initial centers contain exactly one point from each cluster is

$
  P=(k! n^k)/(kn)^k=k!/k^k.
$

For $k=10$, $P approx 0.00036$. K-means++ chooses later centers with probability proportional to squared distance from the nearest chosen center and has an expected $O(log k)$ approximation guarantee. Bisecting K-means repeatedly applies 2-means to split clusters and is less exposed to one global initialization.

K-means prefers compact, roughly globular clusters and can fail for differing sizes/densities, non-globular shapes, and outliers. Over-clustering into small pieces followed by merging is one workaround.

=== Fuzzy c-means

Hard K-means can use $w_ij in {0,1}$:

$
  J=sum_(j=1)^k sum_(i=1)^n w_ij norm(x_i-c_j)^2,
  quad sum_(j=1)^k w_ij=1.
$

Simply relaxing $w_ij$ to $[0,1]$ still yields hard assignments because the objective is linear in $w_ij$. Fuzzy c-means introduces $p>1$:

$
  J_p=sum_(j=1)^k sum_(i=1)^n w_ij^p norm(x_i-c_j)^2.
$

The updates are

$
  c_j=(sum_(i=1)^n w_ij^p x_i)/(sum_(i=1)^n w_ij^p),
$

$
  w_ij=1/sum_(q=1)^k
  (norm(x_i-c_j)^2/norm(x_i-c_q)^2)^(1/(p-1)).
$

Larger $p$ produces softer memberships.

#warning[
Advanced slide 8 omits the power $p$ in the centroid update; the correct update uses $w_ij^p$. Slide 7 states memberships $0.74$ and $0.36$, which violate the sum-to-one constraint. The intended second value is $0.26$, since $2.25(0.74)^2+6.25(0.26)^2 approx 1.654$.
]

=== Mixture models and EM

Mixture clustering models

$
  p(x_i)=sum_(j=1)^k pi_j p(x_i|C_j),
  quad sum_(j=1)^k pi_j=1.
$

For Gaussian mixtures, EM alternates responsibilities

$
  gamma_ij=P(C_j|x_i)
  =(pi_j p(x_i|C_j))/(sum_(q=1)^k pi_q p(x_i|C_q))
$

with parameter updates. Let $N_j=sum_i gamma_ij$:

$
  pi_j=N_j/n,
  quad mu_j=(sum_i gamma_ij x_i)/N_j,
$

$
  Sigma_j=(sum_i gamma_ij (x_i-mu_j)(x_i-mu_j)^T)/N_j.
$

EM resembles K-means in alternating assignment/update structure and initialization sensitivity, but posterior probabilities give soft membership and full covariance Gaussians can model elliptical clusters. Full covariance requires $O(d^2)$ parameters per component; EM may converge slowly and only guarantees a local optimum.

#warning[
The later slides say Gaussian EM prefers globular clusters. That is true for spherical/isotropic covariance restrictions, not general full-covariance Gaussian mixtures, which the earlier slides correctly describe as capable of elliptical clusters.
]

== Hierarchical Clustering and Linkage

*Exam: ★★★★★*

Agglomerative clustering starts from singleton clusters and repeatedly merges the closest pair; divisive clustering starts from one cluster and splits. A dendrogram records this sequence and can be cut at different heights.

For clusters $C_i,C_j$,

$
  D_"single"=min_(p in C_i,q in C_j)d(p,q),
$

$
  D_"complete"=max_(p in C_i,q in C_j)d(p,q),
$

$
  D_"avg"=1/(|C_i||C_j|)
  sum_(p in C_i) sum_(q in C_j)d(p,q).
$

Single link (MIN) follows connectivity and handles non-elliptical shapes, but noisy bridges cause chaining. Complete link (MAX) is less susceptible to chaining but favors compact clusters and can break large clusters. Group average is a compromise but remains biased toward globular structure.

Ward's method chooses the merge with smallest increase in SSE:

$
  Delta(C_i,C_j)="SSE"(C_i union C_j)-"SSE"(C_i)-"SSE"(C_j).
$

It is a hierarchical analogue of K-means. Traditional proximity-matrix implementations require $O(n^2)$ space; straightforward implementations can require $O(n^3)$ time, although better algorithms reduce this. Agglomerative merges are irreversible.

== Density-Based and Subspace Clustering

*Exam: ★★★★★*

DBSCAN uses radius $eps$ and minimum count $"MinPts"$. Define $N_eps(x)={y:d(x,y)<=eps}$. A core point has $|N_eps(x)| >= "MinPts"$, counting itself; a border point is non-core but lies near a core point; all others are noise. Connected core points form cluster backbones and border points attach to neighboring core clusters.

DBSCAN handles irregular shapes and noise and determines the number of clusters from density connectivity. It performs poorly with strongly varying densities and in high dimensions. A sorted $k$-distance plot can suggest $eps$: the knee separates dense points from sparse/noise points.

#warning[
The slides' complexity claims are implementation-dependent. Naive DBSCAN is $O(n^2)$, but indexed low-dimensional implementations can approach $O(n log n)$. Calling K-means or EM simply $O(n)$ suppresses factors for $k$, dimension, and iteration count.
]

Grid-based clustering discretizes space into cells, computes cell density, removes low-density cells, and joins adjacent dense cells. CLIQUE extends this to *subspace clustering*: clusters may exist only on subsets of attributes. It partitions candidate subspaces into equal-volume units, marks units dense above threshold $tau$, and joins contiguous dense units. An Apriori-like monotonicity rule prunes superspaces when lower-dimensional units are not dense. CLIQUE can find overlapping subspace clusters, but worst-case search is exponential and fixed density/grid parameters are difficult when cluster densities differ.

#note[
DENCLUE, Jarvis–Patrick, and Shared Nearest Neighbor (SNN) appear in the advanced outline, but the supplied slides contain no substantive instructional material for them, so they are not expanded here.
]

== Graph-Based Adaptive Clustering: Chameleon

*Exam: ★★★★☆*

Graph-based clustering represents points as vertices with proximity-weighted edges. Sparsification keeps strong/local edges, usually nearest-neighbor links, reducing computation and weakening noisy long-range connections.

Chameleon addresses the static nature of MIN and group-average merging. It first constructs a sparse $k$-NN graph, partitions it into many relatively pure, well-connected subclusters, then agglomeratively merges them using *relative interconnectivity* (RI) and *relative closeness* (RC).

If $EC(C_i,C_j)$ is total cross-edge weight and $EC(C_i),EC(C_j)$ are internal cut connectivities,

$
  "RI"(C_i,C_j)=
  EC(C_i,C_j)/((EC(C_i)+EC(C_j))/2).
$

RI near $1$ means cross-connectivity is comparable to internal connectivity. With average edge weights $bar(S)$ and sizes $m_i,m_j$,

$
  "RC"(C_i,C_j)=
  bar(S)_(EC(C_i,C_j))/
  (
    m_i/(m_i+m_j) bar(S)_(EC(C_i))
    +m_j/(m_i+m_j) bar(S)_(EC(C_j))
  ).
$

RC near $1$ means cross-cluster closeness resembles internal closeness. This normalization lets merging adapt to the scale, density, and connectivity of candidate clusters, preserving self-similarity rather than applying one absolute linkage rule.

== Clustering Evaluation

*Exam: ★★★★★*

Clustering algorithms can find apparent structure even in random data, so validation is essential. *Internal* indices use only the data and clustering; *external* indices compare cluster labels with supplied class labels.

For centroid-based clustering,

$
  "SSE"=sum_i sum_(x in C_i) norm(x-m_i)^2
$

measures cohesion and

$
  "SSB"=sum_i |C_i| norm(m_i-m)^2
$

measures separation, where $m$ is the global mean. For fixed data,

$
  "SST"="SSE"+"SSB".
$

The lecture example has total $10$: one cluster gives SSE $10$, SSB $0$; clusters ${1,2}$ and ${4,5}$ give SSE $1$, SSB $9$.

Calinski–Harabasz is

$
  "CH"=("SSB"/(k-1))/("SSE"/(n-k)),
$

with larger values preferred.

For point $i$, let $a(i)$ be average distance to its own cluster and $b(i)$ the minimum average distance to another cluster. The silhouette coefficient is

$
  s(i)=(b(i)-a(i))/max(a(i),b(i)).
$

It lies in $[-1,1]$: near $1$ is good, near $0$ indicates a boundary, and negative values suggest possible misassignment.

An ideal similarity matrix contains $1$ for same-cluster pairs and $0$ otherwise. Correlation with a proximity matrix assesses agreement between labels and pairwise geometry. Only $n(n-1)/2$ off-diagonal pairs are distinct. If proximity is a *distance*, a good clustering gives a large-magnitude negative correlation; for similarity it gives a positive one. This criterion is less suitable for some density/connectivity clusters. Reordering the proximity matrix by cluster label provides a visual block-structure check.

SSE can estimate $k$ using the elbow/change-point heuristic: choose a point after which additional clusters produce much smaller reductions in SSE.

With external classes, for cluster $C_i$ define $p_ij=n_ij/|C_i|$. Its entropy and purity are

$
  H(C_i)=-sum_j p_ij log_2 p_ij,
  quad
  "purity"(C_i)=max_j p_ij.
$

Overall scores are size-weighted:

$
  H=sum_i |C_i|/n H(C_i),
  quad
  "purity"=sum_i |C_i|/n "purity"(C_i).
$

Lower entropy and higher purity mean stronger agreement with external labels, but these labels are evaluation information, not training labels.

Finally, any validity score needs a reference scale. The lecture proposes comparing the observed index with its distribution under random data: an unusually good SSE or correlation relative to randomized data is evidence that the clustering reflects genuine structure rather than an algorithm imposing patterns on noise.

== Choosing an Algorithm

*Exam: ★★★★☆*

The final choice should jointly consider clustering type (flat versus taxonomy), cluster concept (prototype, connectivity, density, model, subspace), expected shape/size/density/separation, attribute types and scales, dimensionality, noise/outliers, data size, determinism/order dependence, scalability, and parameter burden.

K-means is simple and efficient when centroids and compact clusters are meaningful. DBSCAN is preferable for irregular shapes and noise when a global density scale is meaningful. Single-link hierarchy captures connectivity but chains through noise. Gaussian mixtures add probabilistic memberships and covariance structure at higher statistical/computational cost. CLIQUE targets subspace clusters. Chameleon is designed for complex graph structure with varying internal characteristics.

#warning[
The slides describe K-means as "really assuming spherical Gaussian distributions." This is a useful model-based interpretation of squared-Euclidean K-means, not a literal distributional assumption required to run the algorithm. Also, statements that DBSCAN "always" gives the same result need a caveat: core components are deterministic for fixed data/parameters, but a border point adjacent to multiple clusters can be assigned differently depending on implementation/traversal order.
]
