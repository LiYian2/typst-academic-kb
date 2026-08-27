#import "../template.typ": *

= Classification: Advanced Methods

== Overview

This chapter extends basic classification in three directions. First, it changes the *representation* and *model class*: feature selection removes irrelevant or redundant inputs, Bayesian networks represent dependencies that Naive Bayes ignores, support vector machines construct maximum-margin boundaries, and rule/pattern methods express discriminative structure explicitly. Second, it weakens the assumption that abundant clean labels are available, leading to semi-supervised learning, active learning, transfer learning, distant supervision, and zero-shot learning. Third, it changes the *data type*: streams, sequences, and graphs require models or similarity functions that respect temporal, sequential, or relational structure.

The unifying question is how to preserve reliable classification when the simple setting “fixed feature vector + fully labeled i.i.d. training set” is no longer adequate.

== Feature Selection and Sparse Learning

*Exam: ★★★★★*

=== Feature selection versus feature engineering

Given $p$ initial features, *feature selection* chooses a smaller subset that is useful for prediction. Irrelevant features, such as a student ID when predicting GPA, add no useful signal; redundant features, such as monthly and yearly income, may encode nearly the same information. Removing them can reduce variance, computation, and interpretive complexity.

*Feature engineering* instead constructs new representations from the original variables. The slides illustrate constructing a weekly positive-rate feature from daily positive-case, test, and hospitalization measurements. Traditionally such transformations rely on domain knowledge; modern representation learning can automate part of the process.

=== Filter, wrapper, and embedded methods

The three families differ in how tightly feature selection is coupled to the downstream classifier.

- *Filter methods* score features using a criterion independent of the final classifier. They are fast and reusable but may miss interactions that matter to a particular model.
- *Wrapper methods* repeatedly train and evaluate a classifier on candidate feature subsets. They are model-aware but computationally expensive.
- *Embedded methods* perform model fitting and feature selection in one optimization procedure. LASSO is the main example in the slides.

=== Filter methods and Fisher score

A useful feature should separate class means while keeping within-class spread small. For one feature and $c$ classes, let $n_j$ be the number of samples in class $j$, $mu_j$ its class mean, $sigma_j^2$ its within-class variance, and $mu$ the global mean. The Fisher score used in the slides is

$
  s = (sum_(j=1)^c n_j (mu_j-mu)^2) / (sum_(j=1)^c n_j sigma_j^2).
$

A large numerator indicates well-separated class means; a small denominator indicates compact classes. Other filter criteria listed in the slides include the $chi^2$ test for categorical features, information gain, and mutual information.

=== Wrapper search

With $p$ features there are $2^p-1$ nonempty subsets, so exhaustive wrapper search is exponential. Practical wrappers use greedy searches. *Forward selection* starts from the empty set and repeatedly adds the feature giving the largest performance improvement. *Backward elimination* starts from all features and repeatedly removes one. Hybrid strategies can combine both directions.

=== LASSO as an embedded method

For linear prediction, LASSO adds an $L_1$ penalty to least squares:

$
  hat(L)(bold(w)) = 1/2 sum_(i=1)^n (y_i-bold(w)^T bold(x)_i)^2 + lambda sum_(j=1)^d abs(w_j).
$

The loss favors predictive fit while the $L_1$ term shrinks coefficients and can set some exactly to zero, so the model performs feature selection during training. The slides connect this to the ideal but discontinuous $L_0$ count of selected features: $L_1$ is a convex surrogate that is much easier to optimize.

#warning[
  The slide writes the $L_1$ sum from $j=0$, which may include the intercept. Standard LASSO usually does *not* penalize the intercept; only feature coefficients are regularized unless the formulation explicitly states otherwise.
]

=== Coordinate descent and soft thresholding

Coordinate descent updates one coefficient while holding the others fixed. For coordinate $t$, define the partial residual excluding feature $t$,

$ r_i = y_i - sum_(j != t) w_j x_(ij). $

If the feature is normalized so that $sum_i x_(it)^2=1$, the unregularized one-dimensional least-squares coefficient is

$ beta_t = sum_(i=1)^n x_(it) r_i. $

The coordinate update solves

$
  min_(w_t) 1/2 (w_t-beta_t)^2 + lambda abs(w_t).
$

Its solution is soft thresholding:

- if $beta_t > lambda$, set $w_t = beta_t-lambda$;
- if $beta_t < -lambda$, set $w_t = beta_t+lambda$;
- otherwise set $w_t=0$.

This is the mechanism by which $L_1$ regularization performs sparse selection: coefficients whose unregularized evidence is too weak are collapsed exactly to zero.

#warning[
  One soft-thresholding slide labels the cases by the sign of $w_t$. The actual conditions are on the unregularized coordinate value $beta_t$: $beta_t>lambda$, $beta_t<-lambda$, or $abs(beta_t)<=lambda$.
]

#note[
  The slides summarize coordinate-descent convergence by saying convex, smooth objectives converge globally and separable nonsmooth convex terms are also acceptable. The precise statement requires regularity assumptions, but LASSO has exactly the common structure $f(bold(w))=g(bold(w))+sum_j h_j(w_j)$ with smooth convex $g$ and separable convex $h_j$, for which coordinate descent is well suited.
]

=== Beyond LASSO

Sparse penalties can encode richer structure. Elastic net combines $L_1$ and $L_2$ regularization,

$
  1/2 sum_i (y_i-bold(w)^T bold(x)_i)^2 + lambda_1 norm(bold(w))_1 + lambda_2 norm(bold(w))_2^2,
$

balancing sparsity with ridge-style shrinkage. Group LASSO uses an $L_2$ norm within each predefined group and an $L_1$-like sum across groups,

$ 1/2 sum_i (y_i-bold(w)^T bold(x)_i)^2 + sum_(g=1)^m lambda_g norm(bold(w)_g)_2, $

which encourages whole groups to enter or leave together. Fused LASSO additionally penalizes adjacent coefficient differences, for example $lambda_2 sum_(j=2)^d abs(w_j-w_(j-1))$, encouraging neighboring features to have equal or jointly zero coefficients. The slides briefly mention graph-guided variants, graphical LASSO, and matrix completion as related sparse-learning ideas.

== Bayesian Belief Networks

*Exam: ★★★★☆*

=== From Naive Bayes to dependency modeling

Naive Bayes assumes that features are conditionally independent given the class. When real variables remain dependent after conditioning on the class, this can be too restrictive. A Bayesian network represents a collection of random variables and their conditional dependencies with a directed acyclic graph (DAG).

A Bayesian network has two components:

- a DAG whose nodes are random variables and whose directed edges encode direct probabilistic dependencies;
- conditional probability tables (CPTs), or corresponding conditional distributions, specifying each node given its parents.

The joint distribution factorizes according to the graph:

$
  P(X_1,...,X_n) = product_(k=1)^n P(X_k | "Parents"(X_k)).
$

For the `Fire–Smoke–Tampering–Alarm` example, the graph has $F -> S$, $F -> A$, and $T -> A$, so

$ P(F,S,T,A) = P(F) P(T) P(S|F) P(A|F,T). $

The CPT for a node lists its conditional probability for every parent configuration.

=== Three local graph structures and conditional independence

The diagrams distinguish three elementary path patterns:

- *chain / head-to-tail*: $A -> C -> B$;
- *fork / tail-to-tail*: $A <- C -> B$;
- *collider / head-to-head*: $A -> C <- B$.

For a chain or fork, conditioning on the middle node blocks information flow between the endpoints. A collider behaves oppositely: the path is blocked when the collider is unobserved, but conditioning on the collider or one of its descendants can make the endpoints dependent. These rules are the local intuition behind d-separation.

For example, the chain $A -> C -> B$ factorizes as

$ P(A,B,C)=P(A)P(C|A)P(B|C). $

=== Learning Bayesian networks

The slides organize training by whether the structure is known and whether all variables are observed.

- *Known structure, all variables observed*: estimate only CPT entries from data.
- *Known structure, hidden variables present*: estimate parameters in the presence of latent variables.
- *Unknown structure, all variables observed*: search over DAG structures and fit their parameters.
- *Unknown structure with latent variables*: jointly identifying hidden structure and parameters is substantially harder.

#warning[
  The slides describe the known-structure/hidden-variable case as “gradient descent (greedy hill-climbing).” These are different concepts: gradient methods optimize continuous parameters, while hill climbing commonly refers to discrete structure search. For latent-variable Bayesian networks, expectation-maximization and other likelihood/Bayesian inference methods are standard parameter-learning tools. The slide should be read as conveying that hidden variables make optimization iterative and potentially locally optimal, not as a precise algorithmic equivalence.
]

#warning[
  The statement “unknown structure, all hidden variables: no good algorithms known” is too categorical and historically dated. Joint latent-structure learning remains difficult and often non-identifiable without assumptions, but many approximate and restricted methods exist. The course-level point is the sharp increase in difficulty as both structure and variables become unknown.
]

=== Plate notation

Plate notation compresses repeated random variables in probabilistic graphical models. A shaded node represents an observed variable; an unshaded node represents a hidden variable. A box (plate) surrounding a node indicates that the variable is replicated over an index range, with the number or label near the plate specifying the repetition count.

The student/course example illustrates nested repetition: course difficulty repeats across courses, student intelligence repeats across students, and grade variables repeat over student-course pairs. Plate notation represents this repeated dependency structure far more compactly than drawing every grade node separately.

== Support Vector Machines

*Exam: ★★★★★*

=== Binary classification as a geometric problem

For binary classification, let $bold(x) in bb(R)^d$ and $y in {-1,+1}$. A linear classifier uses a separating hyperplane

$ bold(w)^T bold(x)+b=0. $

There are generally many hyperplanes that separate linearly separable training data. SVM chooses the one with the largest geometric margin, seeking a boundary that stays as far as possible from the nearest training points.

The two canonical margin planes are scaled to

$ bold(w)^T bold(x)+b=1 $

and

$ bold(w)^T bold(x)+b=-1. $

Training examples lying on these planes are support vectors. The scaling to $+1$ and $-1$ is a convention made possible because multiplying $(bold(w),b)$ by a positive constant does not change the decision boundary.

#note[
  This explains the “how about 0.5?” annotation on the slide: $0.5$ could be used with a corresponding rescaling, but fixing the functional margin at $1$ removes this arbitrary scale and yields the standard optimization problem.
]

=== Hard-margin SVM

The geometric distance from a point $bold(x)$ to the hyperplane is

$ r = abs(bold(w)^T bold(x)+b)/norm(bold(w))_2. $

For a correctly classified labeled point, this can be written as $y_i(bold(w)^T bold(x)_i+b)/norm(bold(w))_2$. Maximizing the minimum distance can be rescaled into

$
  min_(bold(w),b) 1/2 norm(bold(w))_2^2
$

subject to

$ y_i(bold(w)^T bold(x)_i+b) >= 1, quad i=1,...,n. $

The distance between the two margin planes is

$ gamma = 2/norm(bold(w))_2. $

Thus minimizing the weight norm is equivalent to maximizing the margin. This is a convex quadratic program, so the optimum is global.

=== Soft-margin SVM and slack variables

Perfect linear separability is often unrealistic. Soft-margin SVM introduces slack variables $xi_i>=0$:

$
  min_(bold(w),b,bold(xi)) 1/2 norm(bold(w))_2^2 + C sum_(i=1)^n xi_i
$

subject to

$ y_i(bold(w)^T bold(x)_i+b) >= 1-xi_i. $

The interpretation is geometric:

- $xi_i=0$: the point is on or beyond the correct margin boundary;
- $0<xi_i<1$: inside the margin but still correctly classified;
- $xi_i=1$: on the decision boundary;
- $xi_i>1$: misclassified.

The parameter $C>0$ controls the trade-off. Large $C$ penalizes violations strongly and favors fitting the training data; small $C$ tolerates more violations in exchange for a wider margin and stronger regularization.

=== Nonlinear classification and the kernel trick

When the original space is not linearly separable, a nonlinear mapping $phi(bold(x))$ can embed the data into another feature space where a linear separator may be effective. Cover's theorem, as presented in the slides, motivates this idea: a complex classification pattern is more likely to become linearly separable after a nonlinear embedding into a sufficiently rich higher-dimensional space.

SVM does not need to construct $phi(bold(x))$ explicitly if the algorithm uses transformed examples only through inner products. A kernel computes

$ K(bold(x)_i,bold(x)_j) = phi(bold(x)_i)^T phi(bold(x)_j). $

Typical kernels in the slides are

$ K(bold(x)_i,bold(x)_j) = (bold(x)_i^T bold(x)_j + 1)^h $

for a polynomial kernel of degree $h$,

$ K(bold(x)_i,bold(x)_j) = exp(-norm(bold(x)_i-bold(x)_j)_2^2/(2 sigma^2)) $

for the Gaussian radial-basis kernel, and a sigmoid-style kernel of the form

$ K(bold(x)_i,bold(x)_j) = tanh(kappa bold(x)_i^T bold(x)_j - delta). $

#warning[
  The slide says that with a suitable nonlinear mapping to a sufficiently high dimension, data from two classes “can always be separated.” This needs qualifications. A rich mapping can make many finite data sets separable, but contradictory duplicate inputs with different labels cannot be separated by any deterministic feature map. The safer course-level statement is Cover's: nonlinear high-dimensional embeddings can make linear separation more likely.
]

=== Multiclass SVM and scalability

SVM is fundamentally binary, but multiclass classification can be reduced to binary problems. With $N$ classes, one-vs-rest trains $N$ classifiers; one-vs-one trains $N(N-1)/2$ pairwise classifiers.

SVM is attractive in high-dimensional spaces because the final decision is determined by support vectors rather than by every training point. However, kernel SVM training can be expensive in the *number of training objects*, both in time and memory. The slides therefore distinguish “works well in high feature dimension” from “scales to massive sample size.” They cite hierarchical clustering as one historical scaling strategy.

Applications listed in the slides include handwritten-digit recognition, object recognition, speaker identification, time-series prediction, multiclass classification, and support-vector regression. The recap emphasizes the convex max-margin formulation, kernel flexibility, and good behavior on smaller data sets, against limited native scalability to very large sample collections. As representative software, the slides list LIBSVM, SVM-light, and SVM-torch; these implementation names are ancillary to the mathematical method.

== Rule-Based and Pattern-Based Classification

*Exam: ★★★★☆*

=== IF-THEN rules

A classification rule has an antecedent and a predicted class, for example:

`IF age = youth AND student = yes THEN buys_computer = yes`.

Two basic rule-quality measures are *coverage* and *accuracy*. Coverage is the fraction of tuples satisfying the antecedent, regardless of their labels. Rule accuracy is the fraction of covered tuples whose class matches the consequent.

When several rules fire, the classifier needs conflict resolution. The slides list three possibilities: prioritize rules with more attribute tests (“size ordering”), use class-based priorities such as prevalence or misclassification cost, or arrange all rules into a quality-ranked decision list.

=== Extracting rules from a decision tree

Every root-to-leaf path in a decision tree can be written as one rule: conjunctions of attribute tests form the antecedent, and the leaf class is the consequent. For the complete deterministic tree shown in the slides, the resulting path rules are mutually exclusive and collectively exhaustive over the tree's represented input branches. This representation can be easier to inspect than a large graphical tree.

=== Sequential covering

Rules can also be induced directly instead of first constructing a tree. Sequential covering learns one rule at a time for a target class:

- start with an empty rule list;
- learn a rule that covers many examples of the target class and few examples of other classes;
- remove or mark the examples covered by that rule;
- repeat on the remaining data until no useful examples remain or rule quality falls below a threshold.

The contrast with tree induction is organizational: a tree recursively partitions the space and thereby learns many path rules together, whereas sequential covering explicitly builds a rule set one rule at a time.

=== Pattern-based classification and CBA

Pattern-based classification integrates pattern mining with supervised prediction. Higher-order patterns can act as compact discriminative features: a phrase can convey class information that individual words do not, and patterns are useful when the raw object is a graph, sequence, or semi-structured record without a natural fixed vector representation.

Classification Based on Associations (CBA) mines class association rules whose left-hand sides are conjunctions of attribute-value pairs and whose right-hand sides are class labels. The procedure in the slides is:

- mine rules with sufficiently high support and confidence;
- rank them in descending order of confidence and support;
- for a test object, apply the highest-ranked matching rule;
- if none matches, use a default class rule.

Its motivation is that multi-attribute associations can expose discriminative combinations that a greedy one-attribute-at-a-time split may not discover early.

== Learning with Limited or Indirect Supervision

*Exam: ★★★★☆*

The slides group semi-supervised learning, active learning, transfer learning, distant supervision, and zero-shot learning under “weakly supervised learning.” Their common motivation is limited conventional supervision, but they relax *different* assumptions.

#warning[
  This is a broad course taxonomy rather than a universally standard one. In much of the literature, active learning, transfer learning, and zero-shot learning are treated as distinct paradigms rather than subtypes of weak supervision. Preserve the course grouping for revision, but distinguish their actual supervision settings below.
]

=== Semi-supervised learning

Semi-supervised learning (SSL) trains from both labeled and unlabeled examples. Unlabeled data can help only when the input distribution contains structure related to the label function.

*Self-training* begins with a classifier fitted to labeled data, predicts labels for unlabeled objects, adds one or more high-confidence pseudo-labeled objects to the labeled set, retrains, and repeats. Its weakness is error reinforcement: an incorrect high-confidence pseudo-label can be fed back into later rounds.

*Co-training* uses two non-overlapping feature views. Train classifiers $f_1$ and $f_2$ on their respective views; each classifier labels confident unlabeled examples and contributes those examples to the other classifier's labeled set. The method is most plausible when the two views are individually informative and provide complementary evidence.

The slides emphasize two assumptions explaining when unlabeled data can help:

- *cluster assumption*: examples in the same high-density cluster tend to share a label, so a good decision boundary should pass through low-density regions. Semi-supervised SVM (S3VM) is shown as a max-margin example that attempts not to cut through dense unlabeled clusters;
- *manifold assumption*: nearby examples along the data manifold tend to have the same label. Graph-based SSL operationalizes this by linking nearby points and propagating labels through the graph.

=== Active learning and transductive learning

Active learning assumes that labels can be obtained from an oracle, but labeling is expensive. In pool-based active learning the learner chooses which unlabeled examples to query so that a limited labeling budget improves the classifier as much as possible. Selection strategies listed in the slides include uncertainty sampling, query-by-committee, version-space methods, and decision-theoretic criteria.

The distinction from pure SSL is that active learning *requests new ground-truth labels*, whereas SSL exploits unlabeled examples without asking for them to be labeled. Transductive learning is different again: the goal is to predict the labels of the particular unlabeled/test objects available during training rather than to learn a general inductive function for arbitrary future inputs.

=== Transfer learning and TrAdaBoost

Transfer learning extracts knowledge from one or more source tasks and applies it to a target task. The slides use sentiment analysis as an example: electronics reviews form a source domain and movie reviews a target domain.

TrAdaBoost is an instance-based transfer method. Ordinary boosting increases attention to difficult examples. TrAdaBoost treats source examples differently: if a source example is repeatedly misclassified for the target task, its weight is reduced because it may be irrelevant or misleading to the target domain. The general principle is to retain source instances that transfer well and suppress those that do not.

The central danger is *negative transfer*: source knowledge can hurt target performance when source and target are too different. The slides therefore mention transfer margins and divergence measures as ways to characterize source-target mismatch, and relate transfer learning to multitask learning and pretraining followed by fine-tuning.

=== Distant supervision

Distant supervision automatically creates large labeled data sets from heuristic or external signals. The labels are cheap and numerous but noisy. Examples in the slides include assigning tweet sentiment from emoticons and assigning tweet categories from linked web or video categories. Applications include social-media classification and relation extraction.

The key technical problem is the labeling function: it must generate enough supervision while controlling systematic noise. For example, a positive emoticon does not guarantee that the entire post expresses positive sentiment.

=== Zero-shot learning

Zero-shot learning predicts classes that were never observed as labeled training classes. The learner therefore needs external semantic information linking seen and unseen classes. The slides' animal example trains on classes such as owl, dog, and fish but may receive a cat at test time.

A semantic-attribute strategy introduces an intermediate representation. First learn classifiers for semantic attributes from seen classes; then infer those attributes for a new object; finally compare the predicted attribute pattern with external descriptions of unseen classes. Semantic attributes therefore act as a bridge that transfers knowledge from observed labels to novel labels.

In *generalized* zero-shot learning, test examples may come from either seen or unseen classes, so the model must recognize both rather than assuming every test object belongs to a novel class.

== Classification with Rich Data Types

*Exam: ★★★★☆*

=== Stream data

Stream classification receives examples sequentially rather than as one static finite table. Fraud transactions, marketing events, network monitoring, and sensor data are examples. The slides emphasize four constraints: high arrival speed, potentially unbounded length, limited opportunities to revisit old data, and *concept drift*, where the relationship between features and labels changes over time.

A chunk-based ensemble addresses these constraints by training a new classifier on the most recent chunk, processing each incoming object once for current training and model-weight updates, and reweighting historical classifiers according to how relevant they remain to recent data. The ensemble therefore acts as a memory with decay: older concepts can remain useful, but drifting concepts receive less weight.

The slides also mention the Very Fast Decision Tree (VFDT), or Hoeffding tree, and sliding windows that focus training on recent examples.

#warning[
  The slide describes Hoeffding trees as building a decision tree from a “sampled subset” of training tuples. More precisely, VFDT processes streaming examples incrementally and uses the Hoeffding bound to decide when the observed difference between candidate split statistics is large enough to choose a split with high confidence. It is not simply ordinary decision-tree training on an arbitrary sampled subset.
]

=== Sequence classification

A sequence is an ordered list $(bold(x)^1, bold(x)^2, ..., bold(x)^T)$, such as a sentence, DNA segment, or a customer's transactions over time. Sequence classification can assign one label to the whole sequence, for example sentiment, coding versus non-coding DNA, or customer value. A related setting predicts a label at each time position.

The slides present three broad strategies. The first converts a sequence into a conventional feature vector. Symbolic sequences can use $n$-gram counts; numerical sequences can first be discretized. A standard vector classifier is then trained. More recent neural approaches such as recurrent networks learn sequence representations automatically.

The second strategy defines a sequence-specific distance or similarity. $k$-NN can use a sequence distance, and nonlinear SVM can use a sequence kernel such as a string kernel.

=== Dynamic time warping

Euclidean distance compares aligned time indices directly and can fail when similar patterns occur at different speeds. Dynamic time warping (DTW) allows local stretching and compression of the time axis. For sequences $A=(A_1,...,A_m)$ and $B=(B_1,...,B_n)$ with local cost $d(A_i,B_j)$, define

$
  D(i,j) = d(A_i,B_j) + min(D(i-1,j), D(i,j-1), D(i-1,j-1)).
$

The three predecessor moves respectively skip/extend an element of one sequence or match the two current elements. The optimal warping path minimizes cumulative cost from $(1,1)$ to $(m,n)$.

#example(title: "DTW alignment from the slides")[
  For $A=[1,2,3,4,5]$ and $B=[2,2,3,4]$ with $d(A_i,B_j)=abs(A_i-B_j)$, one low-cost alignment pairs $1$ with $2$, $2$ with $2$, $3$ with $3$, $4$ with $4$, and $5$ with $4$. The final element of $B$ is therefore reused, illustrating how DTW accommodates unequal sequence lengths and timing distortions.
]

=== Graph classification

Graph data consist of nodes linked by edges. Examples include social networks, power grids, transaction networks, and biological networks. The prediction target can be a *node* label, such as webpage category, or a *graph* label, such as molecular toxicity.

One strategy is feature engineering. Node-level features may include degree, triangle counts, centrality, and PageRank; graph-level features may include graph size, diameter, and motif counts. These vectors can feed ordinary classifiers such as SVM. Modern graph neural networks automate much of this representation learning.

A second strategy defines a proximity measure directly between nodes or graphs and uses a nearest-neighbor-style classifier. The same conceptual pattern seen earlier therefore reappears: either transform a rich object into features, or define a geometry/kernel that works directly on the object type.

== Other Related Classification Techniques

*Exam: ★★★☆☆*

=== Multiclass and multilabel classification

Binary classification predicts between two classes. With $m$ classes, one-vs-all trains $m$ binary classifiers, each separating one class from the rest; all-vs-all trains a classifier for every pair, requiring $m(m-1)/2$ models. Error-correcting output codes assign each class a codeword and predict its bits with multiple binary classifiers, using code redundancy to improve robustness.

Multiclass classification should not be confused with *multilabel* classification. Multiclass usually assigns exactly one class from several alternatives, whereas a multilabel object may belong to multiple classes simultaneously.

=== Distance metric learning

Nearest-neighbor quality depends on the metric. Distance metric learning adapts the metric to the classification task instead of fixing Euclidean geometry. A Mahalanobis distance is

$
  d_M(bold(x)_i,bold(x)_j)^2 = (bold(x)_i-bold(x)_j)^T M (bold(x)_i-bold(x)_j),
$

where $M succ.curly.eq 0$ is positive semidefinite. The slide formulation seeks a metric that makes dissimilar pairs in a set $D$ far apart while constraining similar pairs in a set $S$ to stay close:

$
  max_(M succ.curly.eq 0) sum_((i,j) in D) d_M(bold(x)_i,bold(x)_j)^2
$

subject to

$ sum_((i,j) in S) d_M(bold(x)_i,bold(x)_j)^2 <= 1. $

The positive-semidefinite constraint keeps the learned quadratic form a valid nonnegative distance geometry, and the shown formulation is convex.

=== Interpretability

Interpretability concerns whether a user can understand why a classifier makes a prediction or how its decision process works. Small decision trees and linear models are naturally easier to inspect than many black-box models.

LIME (Local Interpretable Model-Agnostic Explanations) explains a particular prediction by sampling perturbed examples near the query, weighting them by locality, and fitting a simple surrogate such as a sparse linear model. The surrogate trades local fidelity to the black-box classifier against human interpretability. The slides also mention counterfactual explanations and influence functions as alternative explanation approaches.

=== Genetic algorithms

Genetic algorithms search using ideas from natural evolution. Candidate rules can be encoded as bit strings. Starting from a random population, evaluate fitness, select fitter candidates, create offspring through crossover and mutation, and repeat until an adequate fitness level is reached. In the slides, classification accuracy supplies the rule fitness. A simple crossover combines portions of two parent bit strings, while mutation flips one or more bits to maintain exploration. The slide example crosses `110010` and `101111` to obtain `110111`, then illustrates mutation by changing it to `111111`.

=== Reinforcement learning as a related feedback setting

Classification receives *instructive feedback*: the true label tells the learner what the answer should have been. Reinforcement learning instead receives *evaluative feedback*, such as a reward for an action, and must discover which actions produce high long-term value.

The multi-armed bandit is the simplest illustration. Each arm/action has an unknown reward distribution, and the learner must balance exploiting actions that appear good with exploring uncertain alternatives. The slides mention $epsilon$-greedy and upper-confidence-bound methods, with applications such as online advertising, robotics, and games. The important connection to classification is the contrast in the form of supervision, not that reinforcement learning is itself an ordinary classifier.

== Chapter Takeaways

Advanced classification changes the representation, the supervision regime, or the object type rather than merely swapping one classifier for another. Feature selection and sparse penalties control which variables matter; Bayesian networks replace Naive Bayes' blanket independence assumption with graph-structured dependencies; SVM builds convex maximum-margin classifiers and extends them nonlinearly through kernels; rule and association methods expose explicit discriminative patterns. When labels are scarce or indirect, SSL, active learning, transfer, distant supervision, and zero-shot learning each exploit a different additional information source. Streams, sequences, and graphs then require temporal adaptation, structure-aware distances, or learned representations. Across all of these methods, the recurring design principle is to encode the right notion of *relevance, dependence, similarity, or supervision* for the data at hand.
