// @title: Classification: Basic Concepts
// @description: 决策树、贝叶斯分类、KNN、逻辑回归、模型评估与集成方法。
// @order: 60

#import "../template.typ": *

= Classification: Basic Concepts

== Overview

Classification is a supervised-learning problem: labeled training examples are used to learn a model that assigns categorical labels to unseen objects. The chapter develops this idea through several complementary model families. Decision trees partition the feature space with interpretable rules; Bayesian classifiers reason from posterior probabilities; lazy learners such as $k$-nearest neighbors postpone most computation until prediction time; and logistic regression turns a linear score into a class probability. The second half of the chapter asks how such models should be evaluated and improved, leading from confusion-matrix metrics and resampling protocols to ensembles and class-imbalance techniques.

The central progression is therefore

$ "labeled data" -> "classifier" -> "evaluation on unseen data" -> "accuracy / robustness improvements". $

The methods differ in their inductive assumptions. Trees build recursive partitions, Naive Bayes assumes conditional feature independence, $k$-NN relies on local similarity, and logistic regression assumes that the log-odds are linear in the features. No single classifier is uniformly best for every data set; model choice must be tied to the structure of the data and to the evaluation criterion.

== Classification Setup and Learning Workflow

*Exam: ★★★★☆*

=== Supervised classification versus unsupervised learning

In supervised learning, every training object $bold(x)_i$ is accompanied by a class label $y_i$. A classifier is learned from these labeled examples and is then applied to new objects. In contrast, unsupervised learning such as clustering receives observations without known class labels and attempts to discover latent groups or structure in the data.

Classification predicts a *categorical* target, whereas numeric prediction models a *continuous-valued* target. Thus predicting whether a customer belongs to a class such as `positive` or `negative` is classification; predicting a house price is numeric prediction. Both are predictive tasks, but the target space and the appropriate losses differ.

=== Training, validation, testing, and deployment

A classification model may be represented as a decision tree, a set of rules, a mathematical function, or another predictive structure. The basic assumption is that each labeled training sample belongs to one predefined class. Model construction uses a *training set*. Generalization is assessed on data not used to fit the model.

A clean workflow distinguishes three roles:

- *training set*: fit model parameters or structure;
- *validation/development set*: select hyperparameters, compare variants, or refine the model;
- *test set*: estimate the final generalization performance after model selection is complete.

If the resulting performance is acceptable, the selected model is deployed to classify previously unseen data.

#warning[
  The slides sometimes use “validation test set” and say that a test set used for model refinement becomes a validation set. For rigorous evaluation, keep the roles separate: once a data set influences model selection, it is no longer an untouched final test set. The final test set should remain independent of both fitting and model selection.
]

== Decision Tree Induction

*Exam: ★★★★★*

=== Recursive partitioning

A decision tree is constructed top-down by recursively partitioning the training examples. Initially all examples are at the root. At a non-leaf node, the algorithm chooses an attribute or split using a heuristic or statistical criterion such as information gain or the Gini index, partitions the data according to that split, and recurses on the resulting subsets.

Typical stopping conditions are reached when all examples at a node have the same class, no useful attributes remain for further partitioning, or a branch receives no training examples. A leaf predicts a class, commonly the majority class among the training examples that reach that leaf.

The `Play Golf` example illustrates the structure. The root tests `Outlook`. `Overcast` immediately predicts `Yes`; the `Sunny` branch is further separated by `Windy`, and the `Rainy` branch by `Humidity`. The point of the example is not the slide order but the mechanism: each internal test reduces class uncertainty until a leaf can make a sufficiently definite prediction.

#warning[
  The slide statement “the optimal splitting is NP” is too imprecise. Evaluating the best split among a finite set of candidate splits at one node is usually tractable. The computational hardness concerns finding a globally optimal decision tree under common size/depth objectives; practical tree learners therefore use greedy local split selection.
]

=== Continuous-valued attributes

A continuous attribute can first be discretized into categorical intervals, such as age ranges. More commonly, a tree searches directly for a threshold. Sort the observed values $a_1 <= a_2 <= ... <= a_m$ and consider candidate thresholds between adjacent values,

$ p_i = (a_i + a_(i+1))/2. $

For a selected threshold $P$, the node is split into $A <= P$ and $A > P$. The threshold giving the best split criterion, for example the largest information gain, is chosen.

=== Entropy and information gain

For a discrete class variable $Y$ taking values $y_1, ..., y_m$ with $p_i = P(Y = y_i)$, entropy is

$
  H(Y) = - sum_(i=1)^m p_i log_2(p_i).
$

Entropy measures class uncertainty: it is low when one class dominates and high when the distribution is more even. Conditional entropy after observing $X$ is

$
  H(Y | X) = sum_x P(X=x) H(Y | X=x).
$

For a training set $D$ containing $m$ classes, let $p_i = |C_i, D| / |D|$. The class entropy before splitting is

$
  "Info"(D) = - sum_(i=1)^m p_i log_2(p_i).
$

If attribute $A$ partitions $D$ into $v$ subsets $D_1, ..., D_v$, the expected remaining entropy is

$
  "Info"_A(D) = sum_(j=1)^v |D_j|/|D| "Info"(D_j),
$

and the information gain is

$
  "Gain"(A) = "Info"(D) - "Info"_A(D).
$

A larger gain means that the split produces purer child nodes and therefore removes more uncertainty about the class.

#example(title: "Play Golf: choosing Outlook")[
  The training set contains $9$ `Yes` and $5$ `No` examples, so

  $ "Info"(D) = -(9/14) log_2(9/14) - (5/14) log_2(5/14) approx 0.940. $

  For `Outlook`, the class counts are `Rainy`: $(2,3)$, `Overcast`: $(4,0)$, and `Sunny`: $(3,2)$. The two mixed groups each have entropy about $0.971$, while the pure `Overcast` group has entropy $0$. Hence

  $ "Info"_"Outlook"(D) = (5/14)(0.971) + (4/14)(0) + (5/14)(0.971) approx 0.694, $

  giving

  $ "Gain"("Outlook") approx 0.940 - 0.694 = 0.246. $

  The slides report the other gains as $"Gain"("Temp") = 0.029$, $"Gain"("Humidity") = 0.151$, and $"Gain"("Windy") = 0.048$, so `Outlook` is selected at the root.
]

#warning[
  Entropy is a property of a random *variable* or distribution, not a “random number” as one slide states.
]

=== Gain ratio

Information gain is biased toward attributes with many distinct values, because a near-unique attribute such as an ID can create many small, pure partitions without providing useful generalization. Gain ratio normalizes the gain by the intrinsic information of the split:

$
  "SplitInfo"_A(D) = - sum_(j=1)^v |D_j|/|D| log_2(|D_j|/|D|),
$

$
  "GainRatio"(A) = "Gain"(A) / "SplitInfo"_A(D).
$

The slides illustrate this with `Temp`, whose partition sizes are $4,6,4$:

$ "SplitInfo"_"Temp"(D) approx 1.557, quad "GainRatio"("Temp") = 0.029/1.557 approx 0.019. $

ID3 is associated with information gain, while C4.5 refines the selection process using gain ratio.

=== Strengths and limitations

Decision trees are attractive because their decisions can be inspected as rules, they handle heterogeneous feature types, they do not require feature normalization for axis-aligned splits, and they make no parametric distributional assumption such as Gaussianity or feature independence. Many tree implementations can also handle missing values through explicit strategies.

Their weaknesses motivate later ensemble methods. Small changes in the data may change early splits and therefore the whole tree; greedy induction can miss globally better trees; deep trees can overfit; and a single axis-aligned tree may need many leaves to approximate a complicated boundary.

#note[
  Missing-value tolerance is implementation-dependent. The conceptual tree model does not automatically specify how missing attributes are routed; practical packages use strategies such as surrogate splits, learned default directions, or preprocessing.
]

== Bayesian Classification

*Exam: ★★★★★*

=== Bayes theorem and maximum a posteriori classification

For mutually exclusive and exhaustive events $A_i$, the law of total probability gives

$ P(B) = sum_i P(B | A_i) P(A_i). $

Bayes' theorem reverses a conditional probability:

$
  P(H | bold(X)) = P(bold(X) | H) P(H) / P(bold(X)) prop P(bold(X) | H) P(H).
$

Here $P(H)$ is the prior probability of a hypothesis or class, $P(bold(X)|H)$ is the likelihood of the observed evidence, and $P(H|bold(X))$ is the posterior. For classification, the denominator is the same for every candidate class, so maximum a posteriori prediction can compare $P(bold(X)|C_i)P(C_i)$ directly.

#example(title: "Cloudy morning and rain")[
  Suppose $P("Cloud")=0.40$, $P("Rain")=0.10$, and $P("Cloud"|"Rain")=0.50$. Then

  $ P("Rain"|"Cloud") = (0.50 times 0.10)/0.40 = 0.125. $

  Although half of rainy days begin cloudy, rain remains relatively unlikely because the prior probability of rain is only $10%$ and cloudy mornings are common.
]

=== Naive Bayes and the conditional-independence assumption

For $bold(X)=(X_1,...,X_d)$, the chain rule gives

$
  P(C | bold(X)) prop P(C) P(X_1|C) P(X_2|X_1,C) ... P(X_d|X_1,...,X_(d-1),C).
$

Naive Bayes replaces these potentially complicated dependencies with the assumption that features are conditionally independent given the class:

$
  P(C | bold(X)) prop P(C) product_(k=1)^d P(X_k | C).
$

This turns learning largely into estimating class priors and one-dimensional class-conditional distributions. The model is therefore extremely efficient and can be updated incrementally as new counts or sufficient statistics arrive. Its main limitation is exactly the simplifying assumption: dependencies among features such as patient profile, symptoms, and diseases are not represented. Bayesian networks, introduced in the advanced chapter, relax this restriction.

=== Categorical and continuous features

For a categorical feature, $P(X_k=v|C_i)$ is estimated from class-conditional counts. For a continuous feature, the slides use a Gaussian class-conditional model. With class-specific mean $mu_(i k)$ and standard deviation $sigma_(i k)$,

$
  p(x_k | C_i) = 1/(sqrt(2 pi) sigma_(i k)) exp(-(x_k-mu_(i k))^2/(2 sigma_(i k)^2)).
$

Under the Naive Bayes assumption, the full likelihood is the product of these feature-wise terms.

#example(title: "Naive Bayes on the Play Golf data")[
  For the single condition `Outlook = Sunny`, the data contain $3$ `Yes` and $2$ `No` examples. Since $P("Yes")=9/14$, $P("No")=5/14$, and $P("Sunny")=5/14$,

  $ P("Yes"|"Sunny") = ((3/9)(9/14))/(5/14) = 3/5 = 0.60, $

  while $P("No"|"Sunny")=2/5=0.40$.

  For the full query $bold(x)=("Rainy", "Cool", "High", "True")$, the slides multiply the four class-conditional likelihoods and the class prior. The unnormalized scores are about $0.00529$ for `Yes` and $0.02057$ for `No`; normalizing gives approximately $0.20$ versus $0.80$, so the prediction is `No`.
]

=== Zero probabilities and Laplace smoothing

Because Naive Bayes multiplies feature likelihoods, one zero estimate makes the entire class score zero. Laplace smoothing prevents this. For a categorical feature with $K$ possible values and class-conditional counts $n_v$,

$ hat(P)(X=v|C) = (n_v + 1)/(sum_(u=1)^K n_u + K). $

Thus counts $(0,990,10)$ over three categories become probabilities proportional to $(1,991,11)$ with denominator $1003$, remaining close to the empirical frequencies while avoiding exact zeros.

#note[
  Adding exactly $1$ is Laplace/add-one smoothing. Adding another positive constant is the more general additive or Lidstone form.
]

== Lazy Learning and Nearest Neighbors

*Exam: ★★★★☆*

=== Lazy versus eager learning

An eager learner constructs a global model before seeing the test object. Decision trees, Naive Bayes, and logistic regression are examples. A lazy learner stores the training examples with little preprocessing and delays most model construction until a query arrives. This shifts cost from training to prediction.

The conceptual advantage is locality: instead of committing to one global hypothesis, a lazy method can form different local approximations around different queries. Typical instance-based methods include $k$-nearest neighbors, locally weighted regression, and case-based reasoning.

=== The k-nearest neighbor rule

Represent each object as a point in a feature space. Under Euclidean distance,

$ d(bold(x), bold(z)) = sqrt(sum_(j=1)^p (x_j-z_j)^2). $

For classification, find the $k$ training examples closest to the query $bold(x)_q$ and return the most common class among them. For real-valued prediction, return the mean target value of the $k$ neighbors. With $k=1$, the feature space is partitioned into Voronoi cells, each associated with its nearest training point.

A distance-weighted variant gives closer neighbors more influence, for example

$ w_i = 1/d(bold(x)_q, bold(x)_i)^2. $

For regression the weighted prediction is a weighted average; for classification the same idea can be used in a weighted vote.

#warning[
  If a query exactly matches a stored point, $1/d^2$ is singular. In practice, exact matches are handled separately or a small stabilizing constant is added to the denominator.
]

=== Choosing k and the curse of dimensionality

The value of $k$ controls the bias-variance trade-off. Very small $k$ produces flexible, jagged boundaries: low bias but high variance and susceptibility to noise. Large $k$ averages over a wider neighborhood: lower variance but higher bias, and potentially includes points from irrelevant regions.

High dimensionality creates an additional problem: distances can become dominated by irrelevant attributes, and nearest and farthest points become less distinguishable. The slides suggest stretching/reweighting informative axes or eliminating irrelevant attributes. Feature scaling is also crucial whenever different coordinates are measured on incomparable numerical scales.

=== Case-based reasoning

Case-based reasoning extends the lazy-learning idea beyond Euclidean feature vectors. It stores rich symbolic descriptions of previous problems and solutions, retrieves similar cases, and may combine or adapt them using domain knowledge. Applications mentioned in the slides include product diagnosis in customer service and legal reasoning. Its central challenge is defining a useful similarity measure and an indexing/retrieval strategy for complex symbolic cases.

== Linear Models: From Regression to Logistic Classification

*Exam: ★★★★★*

=== Linear regression as the starting point

The slides motivate linear regression with mappings such as living area $->$ house price and college/major/GPA $->$ future income. For observations $(bold(x)_i,y_i)$ with $bold(x)_i in bb(R)^p$, linear regression models

$ hat(y)_i = bold(w)^T bold(x)_i + b. $

Least squares chooses parameters that minimize

$
  L(bold(w),b) = sum_(i=1)^n (y_i - bold(w)^T bold(x)_i - b)^2.
$

The slides then give a closed-form slope for the one-dimensional case. If $x_i$ is scalar and an intercept is included, the correct formulas are

$
  hat(w) = (sum_(i=1)^n (x_i-overline(x))(y_i-overline(y))) / (sum_(i=1)^n (x_i-overline(x))^2),
$

$
  hat(b) = overline(y) - hat(w) overline(x).
$

#warning[
  The closed-form expression shown on the linear-regression slide is inconsistent with the preceding $p$-dimensional model and appears to contain an erroneous denominator. The formula above is the correct scalar-feature least-squares solution with an intercept. For multiple features, the solution is the usual multivariate least-squares system rather than this scalar formula.
]

=== Logistic regression

Linear regression predicts an unrestricted real number, whereas binary classification needs a probability. Logistic regression applies the sigmoid function to a linear score:

$
  sigma(z) = 1/(1+exp(-z)) = exp(z)/(1+exp(z)),
$

$
  p_i = P(Y_i=1|bold(x)_i; bold(w),b) = sigma(bold(w)^T bold(x)_i+b).
$

The sigmoid maps $bb(R)$ to the open interval $(0,1)$. Its inverse is the logit,

$ log(p/(1-p)) = bold(w)^T bold(x) + b. $

Thus logistic regression is linear in *log-odds*, not in the binary label itself. A threshold such as $p=0.5$ converts the probability to a class decision; with that threshold, the decision boundary is $bold(w)^T bold(x)+b=0$. In the slide example using years of employment to predict tenure, the fitted sigmoid changes rapidly near the learned boundary around six years, illustrating how a continuous score becomes a class probability.

#warning[
  One slide says the sigmoid maps to $[0,1]$ and writes $ln(y/(1-y)) = bold(w)^T bold(x)+b$. More precisely, the sigmoid output lies in $(0,1)$, and the logit is applied to the modeled probability $p$, not directly to an observed label $y in {0,1}$.
]

=== Maximum likelihood and log-likelihood

For a binary label $y_i in {0,1}$,

$ P(Y_i=y_i|bold(x)_i) = p_i^(y_i) (1-p_i)^(1-y_i). $

Assuming independent training examples, the likelihood is

$
  L(bold(w),b) = product_(i=1)^n p_i^(y_i) (1-p_i)^(1-y_i).
$

It is usually optimized through the log-likelihood

$
  ell(bold(w),b) = sum_(i=1)^n [y_i log(p_i) + (1-y_i) log(1-p_i)].
$

If the intercept is absorbed into an augmented feature vector, this can be rewritten as

$
  ell(bold(w)) = sum_(i=1)^n [y_i bold(x)_i^T bold(w) - log(1+exp(bold(w)^T bold(x)_i))].
$

There is no ordinary closed-form maximizer, so iterative optimization is used.

=== Gradient descent versus ascent

Gradient descent minimizes an objective $F(theta)$ by moving opposite the gradient:

$ theta_(t+1) = theta_t - eta nabla F(theta_t), $

where $eta>0$ is the step size. To maximize the logistic log-likelihood, however, use *gradient ascent*. For component $w_j$,

$ partial ell / partial w_j = sum_(i=1)^n x_(i j) (y_i-p_i), $

so an ascent update is

$ w_j <- w_j + eta sum_(i=1)^n x_(i j)(y_i-p_i). $

The term $y_i-p_i$ has a simple interpretation. If $y_i=1$, an underconfident prediction $p_i<1$ moves $bold(w)$ in the direction of $bold(x)_i$; if $y_i=0$, the update moves away from $bold(x)_i$ in proportion to the predicted positive probability.

#warning[
  The slides title the update “Gradient Descent” but explicitly derive ascent on the log-likelihood. These are equivalent only after changing the sign of the objective: maximize $ell$ by gradient ascent, or minimize $-ell$ by gradient descent.
]

#warning[
  A zero gradient is only a stationary point in a general nonconvex problem; it need not be a local minimum. For standard logistic regression, the negative log-likelihood is convex, so this issue is much better behaved than the generic slide statement suggests.
]

== Model Evaluation and Selection

*Exam: ★★★★★*

=== Confusion matrix

For binary classification, choose one class as positive. The confusion matrix contains

- *TP*: actual positive, predicted positive;
- *FN*: actual positive, predicted negative;
- *FP*: actual negative, predicted positive;
- *TN*: actual negative, predicted negative.

For $m$ classes, entry $"CM"_(i,j)$ counts objects whose true class is $i$ but whose predicted class is $j$.

=== Accuracy, sensitivity, specificity, precision, and recall

Let $P="TP"+"FN"$, $N="TN"+"FP"$, and $"All"=P+N$. Then

$ "Accuracy" = ("TP"+"TN")/"All", quad "ErrorRate" = ("FP"+"FN")/"All" = 1-"Accuracy". $

Sensitivity, also called recall or true-positive rate, is

$ "Sensitivity" = "Recall" = "TP"/("TP"+"FN") = "TP"/P. $

Specificity, the true-negative rate, is

$ "Specificity" = "TN"/("TN"+"FP") = "TN"/N. $

Precision asks how many predicted positives are truly positive:

$ "Precision" = "TP"/("TP"+"FP"). $

Precision and recall often trade off as the decision threshold changes. The $F_beta$ score combines them:

$
  F_beta = (1+beta^2) "Precision" "Recall" / (beta^2 "Precision" + "Recall").
$

For $beta=1$,

$ F_1 = 2 "Precision" "Recall"/("Precision"+"Recall"). $

#warning[
  The slides say that $F_beta$ gives “$beta$ times as much weight” to recall. In the standard formula, the weighting enters as $beta^2$; $beta>1$ emphasizes recall and $beta<1$ emphasizes precision.
]

#example(title: "Why accuracy can be misleading")[
  In the cancer example, $"TP"=90$, $"FN"=210$, $"FP"=140$, and $"TN"=9560$. Hence

  $ "Sensitivity" = 90/300 = 0.30, $
  $ "Specificity" = 9560/9700 approx 0.9856, $
  $ "Accuracy" = 9650/10000 = 0.965, $
  $ "Precision" = 90/230 approx 0.3913, $
  $ F_1 approx 0.3396. $

  The classifier has $96.5%$ accuracy yet detects only $30%$ of actual cancer cases. This is the central class-imbalance lesson: a high overall accuracy can coexist with poor minority-class recognition.
]

=== Underfitting and overfitting

As model complexity increases, training error usually decreases. Test error often first decreases and then rises: the left side corresponds to underfitting, where the model is too simple to capture the signal; the right side corresponds to overfitting, where the model follows idiosyncrasies of the training sample. Model selection should therefore use validation performance rather than training error alone.

=== Holdout and cross-validation

In the holdout method, the data are randomly partitioned into independent training and test portions, for example $2/3$ and $1/3$. Repeated random sub-sampling repeats this random split several times and averages the resulting performance estimates.

In $k$-fold cross-validation, the data are partitioned into $k$ mutually exclusive folds of approximately equal size. For fold $i$, train on all folds except $D_i$ and evaluate on $D_i$; average over all folds. Leave-one-out cross-validation is the limiting case $k=n$ and is mainly practical for small data sets. Stratified cross-validation approximately preserves the class proportions in every fold, which is especially important under imbalance.

#note[
  The slides mention bootstrap evaluation but do not cover its procedure, so it is not developed here.
]

=== ROC curves and AUC

A receiver operating characteristic curve sweeps a decision threshold and plots

$ "TPR" = "TP"/P $

against

$ "FPR" = "FP"/N = "FP"/("FP"+"TN"). $

A random ranking lies near the diagonal, with area under the curve around $0.5$; a perfect ranker has AUC $1$. ROC curves therefore compare the trade-off between true-positive and false-positive rates across thresholds.

#warning[
  The slides call AUC a measure of model “accuracy.” More precisely, ROC AUC is a threshold-independent measure of ranking/discrimination: it is not the same quantity as classification accuracy at one fixed threshold.
]

== Ensembles and Class-Imbalanced Classification

*Exam: ★★★★★*

=== Why ensembles can help

An ensemble combines learned models $M_1,...,M_k$ into a stronger predictor $M^*$. Majority-vote examples in the slides show the key condition: merely adding models is not enough. Base models should be individually useful and, crucially, should make *different* errors. If all models make the same mistake, voting cannot repair it; if weak models fail on unrelated examples, aggregation can cancel some of those errors.

=== Bagging

Bagging, or bootstrap aggregation, creates diversity by resampling the training set. For $i=1,...,k$:

- draw a bootstrap sample $D_i$ from $D$ by sampling with replacement;
- train a base model $M_i$ on $D_i$.

For classification, predict by majority vote; for numeric prediction, average the base predictions. Because the models are trained independently on different bootstrap samples, they can be learned in parallel. Bagging is especially useful for unstable learners such as decision trees.

=== Boosting and AdaBoost

Boosting learns base models sequentially. Each new model places more emphasis on examples that earlier models handled poorly, and the final decision is a weighted combination of the base models.

For binary AdaBoost with labels $y_n in {-1,+1}$, initialize normalized example weights $w_n^((1))=1/N$. At round $t$, train $M_t$ and compute

$
  epsilon_t = (sum_(n=1)^N w_n^((t)) I(M_t(bold(x)_n) != y_n)) / (sum_(n=1)^N w_n^((t))).
$

For a weak learner better than chance,

$ alpha_t = 1/2 log((1-epsilon_t)/epsilon_t). $

The standard weight update is

$
  w_n^((t+1)) prop w_n^((t)) exp(-alpha_t y_n M_t(bold(x)_n)),
$

followed by normalization. Misclassified examples have $y_n M_t(bold(x)_n)=-1$ and are upweighted; correctly classified examples are downweighted. The final classifier is

$ M^*(bold(x)) = "sign"(sum_(t=1)^k alpha_t M_t(bold(x))). $

#warning[
  The AdaBoost slide uses $alpha_t=1/2 log((1-epsilon_t)/epsilon_t)$ but updates weights only by multiplying misclassified points by $exp(alpha_t)$ while leaving correct points unchanged. That update is not the standard AdaBoost update for this definition of $alpha_t$. The canonical exponential update above changes the relative misclassified/correct weight by $exp(2 alpha_t)$ and then normalizes.
]

#warning[
  A slide suggests stopping when $epsilon_t$ is below a generic threshold. Standard AdaBoost treats $epsilon_t=0$ as a perfect base learner and requires the learner to perform better than chance, typically $epsilon_t<1/2$ in the binary case. The exact stopping rule is implementation-dependent.
]

=== Gradient boosting and XGBoost

Gradient boosting also builds an additive model sequentially, but frames the process as minimizing a differentiable loss. If $f_t$ is the new weak learner,

$ hat(y)_i^((t)) = sum_(k=1)^t f_k(bold(x)_i) = hat(y)_i^((t-1)) + f_t(bold(x)_i). $

The new learner is fitted to improve the current model with respect to the loss. Trees are the usual weak learners. XGBoost is cited in the slides as a scalable implementation of this general idea.

=== Random forests

A random forest specializes bagging to decision trees and adds feature randomness. Each tree is trained on a bootstrap sample of the data. At each node, only a random subset of attributes is considered as split candidates, and the best split among that subset is selected. The extra randomization decorrelates trees, increasing the diversity that makes aggregation effective. Classification uses majority vote.

The slides distinguish two constructions: *Forest-RI* randomly selects candidate input attributes at each node; *Forest-RC* forms random linear combinations of existing attributes to create candidate features. They report random forests as comparable in accuracy to AdaBoost while being more robust to errors/outliers, and relatively insensitive to the number of candidate attributes at a split. The main practical trade-off is that forests sacrifice the interpretability of one small tree in return for greater stability and predictive performance.

The ensemble recap identifies random forests and XGBoost as strong methods for tabular data: they usually require no feature scaling, can scale to large data sets, and can accommodate missing values to some extent depending on the implementation. Both can still overfit when poorly tuned, and ensembles are less directly interpretable than a single tree.

=== Imbalanced data

In many applications the positive class is rare, as in medical screening, fraud detection, product-defect detection, accident detection, or disk failures. A classifier that predicts only the majority class can attain deceptively high accuracy: with $2%$ positives and $98%$ negatives, the always-negative rule already achieves $98%$ accuracy.

The slides group remedies into data-level and algorithm-level strategies.

At the data level:

- *oversampling* replicates or resamples minority examples;
- *undersampling* removes majority examples;
- *synthetic sampling* creates additional minority examples.

At the algorithm level:

- *threshold moving* changes the decision threshold so that the rare class is easier to predict;
- *class/cost weighting* makes errors on the important minority class, especially false negatives when they are costly, contribute more to the objective;
- *ensembles* combine multiple classifiers and can be adapted to imbalance.

Evaluation should therefore emphasize class-sensitive metrics such as sensitivity, specificity, precision, recall, $F_1$, and threshold curves rather than raw accuracy alone. ROC curves are one option emphasized by the slides.

#note[
  The important conceptual separation is between *prevalence* and *cost*. A class can be rare without every error on it having the same cost, and a decision threshold should ultimately reflect the application's error trade-offs.
]

== Chapter Takeaways

Classification learns mappings from labeled examples to categorical outputs. Decision trees greedily reduce class impurity; Naive Bayes uses posterior probability with a strong conditional-independence assumption; $k$-NN predicts from local neighborhoods; and logistic regression models linear log-odds and is trained by likelihood optimization. Evaluation must be performed on unseen data and interpreted through the confusion matrix rather than accuracy alone. Bagging, boosting, random forests, and imbalance-aware training then improve robustness by reducing instability, focusing on hard examples, diversifying models, or changing how rare classes influence learning.
