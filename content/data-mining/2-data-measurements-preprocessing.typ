// @title: Data, Measurements, and Data Preprocessing
// @description: 数据表示、统计描述、相似度、清洗、变换与降维。
// @order: 21

#import "../template.typ": *

= Data, Measurements, and Data Preprocessing

== Overview

*Exam: ★★★★☆*

Data mining begins before any mining algorithm is run. A dataset must first be represented in a form whose objects, attributes, scales, distributions, and relationships are understood; then its quality must be assessed and, when necessary, its representation must be cleaned, transformed, reduced, or re-expressed. This chapter therefore follows a natural pipeline: *represent the data* $arrow.r$ *describe its statistical structure* $arrow.r$ *define meaningful proximity* $arrow.r$ *repair and integrate imperfect sources* $arrow.r$ *transform and reduce the representation*.

The central theme is that preprocessing is not a collection of unrelated tricks. The meaning of an operation depends on the type of attribute and on what information the later analysis needs to preserve. For example, an ordinal variable should not be treated exactly like a nominal label; Euclidean distance is not appropriate for every representation; a missing value is not the same as a noisy value; and a dimensionality-reduction method can preserve linear variance while destroying nonlinear neighborhood structure.

#warning[
  The slides repeatedly use the heading “Statics of Data.” The intended term is *Statistics of Data* or *Statistical Description of Data*. This chapter uses the standard terminology.
]

== Data Objects, Dataset Structures, and Attribute Types

*Exam: ★★★★☆*

A *data object* represents an entity described by a collection of attributes. Depending on context, the same concept is also called a sample, example, instance, data point, tuple, or object. In a conventional table, rows are data objects and columns are attributes. The object represented by a row depends on the semantics of the table: a row may represent a customer, patient, transaction, document, location, or any other unit of analysis.

The slides distinguish several important dataset structures. *Record data* include relational records, numerical data matrices, transaction data, and document-term matrices. In a transaction table, an object is a transaction and the attributes are often item-presence indicators. In a document-term matrix, an object is a document and the dimensions correspond to terms, with entries such as term frequencies. *Graph and network data* represent objects and relations jointly; examples include transportation networks, the Web, molecular structures, and social or information networks. *Ordered data* include video sequences, time series, transaction sequences, and genetic sequences, where position or temporal order carries information. *Spatial and multimedia data* include maps, raster/vector spatial layers, images, and video. These structures matter because the same numerical array can require very different analysis when its dimensions encode categories, time, space, or network connectivity.

Structured data are also characterized by *dimensionality*, *sparsity*, *resolution*, and *distribution*. High dimensionality can lead to the curse of dimensionality; sparse data often contain many zero or absent entries, for which presence may be more informative than absence; patterns can depend strongly on spatial, temporal, or measurement resolution; and distributions are summarized by their central tendency and dispersion. These properties anticipate later choices of distance, preprocessing, and dimensionality reduction.

=== Attribute measurement scales

An *attribute* (also called a dimension, feature, or variable) is a data field that describes some characteristic of a data object. The measurement scale determines which comparisons and arithmetic operations are meaningful.

- *Nominal*: values are unordered names or categories, such as hair color, occupation, ZIP code, or an identifier. Equality and inequality are meaningful, but ordering and arithmetic are not.
- *Binary*: a nominal attribute with two states. For a *symmetric binary* attribute, the two states are equally important. For an *asymmetric binary* attribute, one state is the informative or rare event and is conventionally coded as 1; a positive medical test is the slide example.
- *Ordinal*: values have a meaningful order, but the magnitude of the gap between adjacent levels is not defined. Examples include small/medium/large, academic year, grades, and rankings.
- *Interval-scaled numeric*: equal differences are meaningful, but there is no true zero. Temperature in Celsius or Fahrenheit and calendar dates are standard examples; saying that 20 °C is twice 10 °C is not meaningful.
- *Ratio-scaled numeric*: equal differences and ratios are meaningful because there is an inherent zero. Kelvin temperature, length, counts, and monetary quantities are typical examples. Since $K = degree C + 273.15$, Celsius and Kelvin encode the same temperature differences but different zero points.

#warning[
  An ordinal scale should not be treated as a ratio scale merely because its ranks can be encoded numerically. Rank encodings such as 0, $1/3$, $2/3$, 1 introduce convenient interval-like spacing, but that spacing is a modeling convention rather than a measured ratio structure.
]

=== Discrete and continuous attributes

A *discrete* attribute takes values from a finite or countably infinite set. Blood type is finite, while integer-valued years form a countable set. Binary variables are a special case of discrete variables. A *continuous* attribute is conceptually real-valued, such as temperature, height, or weight; in a computer, it is necessarily represented with finite precision.

The discrete/continuous distinction is independent of the nominal/ordinal/interval/ratio distinction. For example, a count is discrete and ratio-scaled, while Celsius temperature is typically continuous and interval-scaled.

== Statistical Description of Data

*Exam: ★★★★★*

Statistical description answers two complementary questions: where is the data concentrated, and how widely is it spread? It also studies how variables co-vary and how distributions can be inspected visually. The slides emphasize that dispersion should be considered at multiple granularities. Emergency-department wait times, for example, may have different variability when summarized by hour, day, or week; rounding changes the observed spread; and subgroups may have different dispersions. Thus a single global summary can hide structure that appears at another resolution.

=== Central tendency: mean, median, and mode

For a sample $x_1, dots, x_n$, the arithmetic sample mean is

$
  overline(x) = 1/n sum_(i=1)^n x_i,
$

while the population mean for a population of size $N$ is $mu = 1/N sum_(i=1)^N x_i$. A weighted mean is

$
  overline(x)_w = (sum_(i=1)^n w_i x_i)/(sum_(i=1)^n w_i).
$

A *trimmed mean* removes a chosen fraction of extreme observations before computing the mean, reducing sensitivity to extremes; the slides use judged sports scores as the motivating example.

The *median* is the middle observation after sorting when $n$ is odd, or the average of the two middle observations when $n$ is even. For grouped data, the slides estimate the median by interpolation within the median interval:

$
  upright("median") approx L + ((n/2 - F)/f_m) w,
$

where $L$ is the lower boundary of the median interval, $F$ is the cumulative frequency before that interval, $f_m$ is its frequency, and $w$ is the interval width.

The *mode* is the most frequent value. A distribution may be unimodal, bimodal, trimodal, or more generally multimodal. The slides also give the empirical relation

$
  upright("mean") - upright("mode") approx 3 (upright("mean") - upright("median")),
$

which is a rough heuristic for moderately skewed unimodal distributions, not a general identity.

For a symmetric unimodal distribution, mean, median, and mode coincide. In a positively skewed distribution the long right tail pulls the mean rightward, typically giving $upright("mode") < upright("median") < upright("mean")$; for negative skewness the ordering is typically reversed.

#informally[
  The mean uses every magnitude and is therefore sensitive to extreme values. The median uses only ordering and is robust to extremes. The mode answers a different question: which value or region occurs most frequently. Their disagreement is therefore informative about skewness and multimodality rather than merely a nuisance.
]

=== Dispersion: variance and standard deviation

For a random variable $X$ with mean $mu = E[X]$, variance is the expected squared deviation from the mean:

$
  sigma^2 = upright("Var")(X) = E[(X-mu)^2] = E[X^2] - (E[X])^2.
$

For a discrete random variable this is $sum_x (x-mu)^2 f(x)$; for a continuous variable it is $integral_(-infinity)^infinity (x-mu)^2 f(x) dif x$. The standard deviation is $sigma = sqrt(sigma^2)$ and has the same physical units as the original variable.

For sample data, two denominators appear in the slides. The descriptive or maximum-likelihood-style second moment is

$
  s_n^2 = 1/n sum_(i=1)^n (x_i-overline(x))^2,
$

whereas the usual unbiased sample variance is

$
  s^2 = 1/(n-1) sum_(i=1)^n (x_i-overline(x))^2.
$

The latter can be written in a computational form as

$
  s^2 = 1/(n-1) (sum_(i=1)^n x_i^2 - 1/n (sum_(i=1)^n x_i)^2).
$

The population variance is $sigma^2 = 1/N sum_(i=1)^N (x_i-mu)^2$.

#warning[
  The slides mix $1/n$ and $1/(n-1)$ under the label “sample variance/covariance.” These are different conventions. For an ordinary sample used to estimate a population variance, the standard unbiased sample variance uses $n-1$. A formula with $1/n$ is a descriptive second moment or an MLE convention under particular models. Keep the convention consistent throughout a calculation.
]

For a normal distribution, the familiar empirical rule is that approximately 68% of the mass lies within $mu plus.minus sigma$, 95% within $mu plus.minus 2 sigma$, and 99.7% within $mu plus.minus 3 sigma$. Here $mu$ locates the center while $sigma$ controls spread.

=== Covariance and correlation for numerical variables

For two random variables $X_1$ and $X_2$ with means $mu_1$ and $mu_2$, covariance is

$
  sigma_12 = E[(X_1-mu_1)(X_2-mu_2)]
  = E[X_1 X_2] - E[X_1] E[X_2].
$

Positive covariance indicates that the variables tend to deviate from their means in the same direction; negative covariance indicates opposite directions. Variance is the special case $sigma_11 = upright("Var")(X_1)$.

The slides' stock example uses observations $(2,5)$, $(3,8)$, $(5,10)$, $(4,11)$, $(6,14)$. The means are $E[X_1]=4$ and $E[X_2]=9.6$, and

$
  E[X_1 X_2] - E[X_1]E[X_2] = 4,
$

so the covariance is positive.

Independence implies zero covariance whenever the relevant moments exist, because $E[X_1 X_2] = E[X_1]E[X_2]$. The converse is false in general: zero covariance means no *linear* co-movement, not independence. Under additional assumptions such as joint multivariate normality, zero covariance does imply independence.

#example[
  The slides give three equally likely pairs $(X_1,X_2)=(1,0),(-1,1),(-1,-1)$. Here $E[X_1]=-1/3$, $E[X_2]=0$, and $E[X_1X_2]=0$, hence $upright("Cov")(X_1,X_2)=0$. Yet $P(X_2=0 | X_1=1)=1$ while $P(X_2=0)=1/3$, so the variables are not independent.
]

Correlation standardizes covariance to remove the variables' scales:

$
  rho_12 = sigma_12/(sigma_1 sigma_2), quad -1 <= rho_12 <= 1.
$

For a sample, the correlation can be computed directly as

$
  hat(rho)_12 =
  (sum_(i=1)^n (x_(i 1)-overline(x)_1)(x_(i 2)-overline(x)_2)) /
  sqrt((sum_(i=1)^n (x_(i 1)-overline(x)_1)^2)(sum_(i=1)^n (x_(i 2)-overline(x)_2)^2)).
$

Values near 1 indicate strong positive linear association, values near -1 strong negative linear association, and values near 0 weak linear association. Scatter plots make this geometry visible: a narrow rising cloud has large positive correlation, a narrow falling cloud has large negative correlation, and a diffuse or nonlinear cloud can have correlation near zero.

#warning[
  $rho=0$ does *not* imply independence in general. The slides correctly attach an additional-distribution assumption to this statement; do not omit that qualification in an exam answer. Also, correlation does not imply causation: a third variable can drive both measured variables.
]

=== Covariance matrix

For a $d$-dimensional random vector $X$ with mean vector $mu$, the covariance matrix is

$
  Sigma = E[(X-mu)(X-mu)^T].
$

Its diagonal entries are variances and its off-diagonal entries are pairwise covariances. For two dimensions,

$
  Sigma = mat(sigma_1^2, sigma_12; sigma_21, sigma_2^2).
$

For real-valued variables, $sigma_12=sigma_21$, so $Sigma$ is symmetric. This matrix later becomes the central object in PCA.

=== Association for categorical variables: chi-square test

For a contingency table, the chi-square statistic compares observed counts $O_i$ with expected counts $E_i$ under an independence model:

$
  chi^2 = sum_i (O_i-E_i)^2/E_i.
$

For a cell in row $r$ and column $c$ with grand total $n$, the expected count is

$
  E_(r c) = ((upright("row total"))_r (upright("column total"))_c)/n.
$

In the slide example, 450 of 1500 people like science fiction and 300 of 1500 play chess, so under independence the expected count in the “like science fiction and play chess” cell is $450/1500 dot 300 = 90$. Using all four cells yields $chi^2 approx 507.93$. For a table with $R$ row categories and $C$ column categories, the degrees of freedom are $(R-1)(C-1)$; hence the $2 times 2$ example has one degree of freedom.

The null hypothesis is that the two categorical variables are independent. Cells with large standardized deviations between observed and expected counts contribute most to $chi^2$. The decision must be made by comparing the statistic with a chi-square distribution using the correct degrees of freedom, or equivalently by using a p-value.

#warning[
  The slide wording “the larger $chi^2$, the more likely the variables are related” is only safe after accounting for degrees of freedom and sample size. A raw $chi^2$ value is not an effect-size measure. The slide also says “reject ... at a confidence level of 0.001”; the standard statement is *p-value < 0.001* or *reject at significance level 0.001* (equivalently, confidence greater than 99.9% in the usual informal wording).
]

=== Graphical summaries

A *boxplot* summarizes a distribution through its quartiles. $Q_1$ is the 25th percentile, $Q_2$ the median, and $Q_3$ the 75th percentile; the interquartile range is $upright("IQR") = Q_3-Q_1$. A classical five-number summary is $(upright("min"), Q_1, upright("median"), Q_3, upright("max"))$. In a Tukey-style boxplot, the box spans $Q_1$ to $Q_3$, the median is drawn inside, and outliers are commonly defined relative to the fences $Q_1-1.5 upright("IQR")$ and $Q_3+1.5 upright("IQR")$.

#warning[
  One slide states both that whiskers extend to the minimum/maximum and that points beyond $1.5 upright("IQR")$ are plotted as outliers. These conventions conflict in a Tukey boxplot. If outliers are plotted individually, whiskers normally extend to the most extreme observations *within* the $1.5 upright("IQR")$ fences, not necessarily to the global minimum and maximum.
]

A *histogram* displays the distribution of quantitative data by dividing its numeric range into bins. When bin widths are unequal, bar *area* should encode frequency (or probability), so the height corresponds to frequency density. A *bar chart* instead compares categorical values; categories may be reordered, while histogram bins retain their numerical order. Thus student heights, ER waiting times, and house prices naturally call for histograms, while blood types, diagnoses, cities, and smoker/non-smoker categories call for bar charts. Small integer-valued distributions are borderline cases: a bar chart can emphasize discrete outcomes, while a histogram-like display can emphasize their distribution.

A boxplot can hide distributional shape. The slides show two different histograms with the same minimum, quartiles, median, and maximum; therefore the same boxplot can correspond to distinct modality or concentration patterns. Histograms retain more shape information at the cost of depending on binning choices.

A *quantile plot* sorts observations and pairs each $x_i$ with a cumulative fraction $f_i$, so approximately $100 f_i$ percent of the data are at or below $x_i$. Unlike a five-number summary, it displays the full ordered sample and can reveal unusual observations.

A *Q-Q plot* compares corresponding quantiles of two univariate distributions. If the two distributions are similar up to location and scale, points tend to follow an approximately straight line; systematic curvature indicates different distributional shape. The slides use one Q-Q plot to compare prices at two branches and another to compare sample data with a normal reference distribution. A branch whose corresponding quantiles lie systematically below the other tends to have lower values at those probability levels.

A *scatter plot* displays paired numerical observations as points in a plane. It provides a first look at clusters, outliers, trend direction, heteroscedasticity, and nonlinear patterns. The slide with an inverted-V pattern is especially important: the left half is positively correlated and the right half negatively correlated, but their combination may have low overall linear correlation. This is a visual reminder that “uncorrelated” does not mean “unstructured.”

== Similarity, Dissimilarity, and Proximity

*Exam: ★★★★★*

A *similarity* function assigns larger values to more alike objects, often in $[0,1]$. A *dissimilarity* or distance assigns smaller values to more alike objects, commonly with minimum 0 and range $[0,1]$ or $[0,infinity)$. *Proximity* is an umbrella term for either similarity or dissimilarity. The correct proximity definition depends on the attribute type and on the semantics of “closeness.”

A data matrix stores $n$ objects by $l$ attributes. A dissimilarity matrix instead stores pairwise distances $d(i,j)$ between the $n$ objects. For a symmetric distance, the matrix is symmetric with zero diagonal, so one triangle is sufficient. The slides' four-point numerical example shows how a two-column data matrix becomes a $4 times 4$ Euclidean dissimilarity matrix.

=== Minkowski distance for numerical data

For two $l$-dimensional objects $i=(x_(i 1),dots,x_(i l))$ and $j=(x_(j 1),dots,x_(j l))$, Minkowski distance of order $p >= 1$ is

$
  d_p(i,j) = (sum_(f=1)^l abs(x_(i f)-x_(j f))^p)^(1/p).
$

A metric satisfies non-negativity with identity of indiscernibles, symmetry, and the triangle inequality:

- $d(i,j) >= 0$ and $d(i,j)=0$ exactly when the objects coincide;
- $d(i,j)=d(j,i)$;
- $d(i,j) <= d(i,k)+d(k,j)$.

Not every useful dissimilarity is a metric.

Important special cases are:

- $p=1$: *Manhattan* or city-block distance, $d_1(i,j)=sum_f abs(x_(i f)-x_(j f))$;
- $p=2$: *Euclidean* distance, $d_2(i,j)=sqrt(sum_f (x_(i f)-x_(j f))^2)$;
- $p arrow.r infinity$: *Chebyshev* or supremum distance, $d_infinity(i,j)=max_f abs(x_(i f)-x_(j f))$.

For binary vectors, Manhattan distance equals Hamming distance: the number of bit positions at which the two vectors differ. The slide example illustrates that the same four points produce different pairwise distances under $L_1$, $L_2$, and $L_infinity$, because each norm aggregates coordinate-wise differences differently.

#informally[
  Manhattan distance adds all coordinate deviations; Euclidean distance emphasizes larger deviations through squaring; Chebyshev distance ignores all but the worst coordinate difference. There is therefore no universally “best” norm independent of the application.
]

=== Binary, nominal, ordinal, and mixed attributes

For two binary objects, define the contingency counts: $q$ = number of attributes where both are 1, $r$ = 1 for object $i$ and 0 for $j$, $s$ = 0 for $i$ and 1 for $j$, and $t$ = both 0. For symmetric binary attributes, simple mismatch distance is

$
  d(i,j) = (r+s)/(q+r+s+t).
$

For asymmetric binary attributes, joint absence $t$ is ignored:

$
  d(i,j) = (r+s)/(q+r+s).
$

The corresponding *Jaccard similarity* is

$
  upright("sim")_J(i,j) = q/(q+r+s).
$

This is appropriate when co-presence is informative but co-absence is not. The medical-record example in the slides therefore excludes gender from the asymmetric calculation and obtains $d("Jack","Mary")=1/3$, $d("Jack","Jim")=2/3$, and $d("Jim","Mary")=3/4$ from the remaining binary attributes.

For nominal attributes, a simple matching dissimilarity counts mismatches. If $m$ of $p$ attributes match,

$
  d(i,j) = (p-m)/p.
$

An alternative is one-hot encoding: a nominal variable with $M$ states is expanded to $M$ binary indicators.

For an ordinal attribute with $M_f$ ordered states, the slides map rank $r_(i f) in {1,dots,M_f}$ to

$
  z_(i f) = (r_(i f)-1)/(M_f-1),
$

then treat $z_(i f)$ as interval-scaled. Thus freshman, sophomore, junior, senior can map to $0,1/3,2/3,1$, giving distances 1 between freshman and senior and $1/3$ between junior and senior. This preserves order and imposes equal spacing without pretending that the values have a meaningful ratio zero.

For mixed-type data, the slides combine per-attribute dissimilarities through a weighted average:

$
  d(i,j) = (sum_(f=1)^p w_(i j)^((f)) d_(i j)^((f))) /
           (sum_(f=1)^p w_(i j)^((f))).
$

The indicator/weight $w_(i j)^((f))$ can exclude unavailable or inapplicable comparisons and can encode application-specific importance. Numeric attributes should first use a normalized distance so that large numerical scales do not dominate; binary and nominal attributes use type-appropriate matching; ordinal attributes are rank-normalized first.

=== Cosine similarity

Long sparse vectors, especially document-term vectors, are often compared by their orientation rather than raw Euclidean distance. For two vectors $d_1$ and $d_2$,

$
  cos(d_1,d_2) = (d_1 dot d_2)/(sqrt(d_1 dot d_1) sqrt(d_2 dot d_2)).
$

The document example uses
$d_1=(5,0,3,0,2,0,0,2,0,0)$ and
$d_2=(3,0,2,0,1,1,0,1,0,1)$. Their dot product is 25, their lengths are approximately 6.481 and 4.12, and the cosine similarity is approximately $25/(6.481 dot 4.12) approx 0.94$.

Cosine similarity is insensitive to a common positive scaling of a vector, so two documents with similar term proportions but different lengths can still be highly similar.

=== KL divergence for probability distributions

The Kullback-Leibler divergence compares probability distributions over the same variable. For discrete distributions,

$
  D_("KL")(P || Q) = sum_x p(x) ln(p(x)/q(x)),
$

and for continuous densities,

$
  D_("KL")(P || Q) = integral_(-infinity)^infinity p(x) ln(p(x)/q(x)) dif x.
$

It measures the expected information penalty incurred when a code or model based on $Q$ is used for samples generated from $P$. In applications, $P$ is often the reference or data-generating distribution and $Q$ an approximation. $D_("KL")(P||Q) >= 0$ and equals zero exactly when the two distributions agree almost everywhere.

KL divergence is *not* a metric: in general $D_("KL")(P||Q) != D_("KL")(Q||P)$ and the triangle inequality does not hold. If $p(x)=0$, the contribution is defined by the limit $p ln p arrow.r 0$. If $p(x)>0$ but $q(x)=0$, then $D_("KL")(P||Q)=infinity$ because $Q$ assigns zero probability to an event that can occur under $P$.

The slides therefore motivate smoothing when empirical frequency distributions have unseen symbols. If $P$ and $Q$ have different observed supports, a small $epsilon$ can be assigned to missing symbols and the remaining probabilities adjusted so that each distribution still sums to one.

#warning[
  Smoothing must preserve normalization. The slide gives one particular redistribution of a small $epsilon$ over the union of supports; the essential principle is to avoid zero probability where the reference distribution assigns positive mass, while renormalizing consistently.
]

=== Limits of hand-designed proximity

Simple matching, vector norms, cosine similarity, and KL divergence operate on representations whose semantics are already explicit. They do not automatically recover hidden meaning. A bag-of-words representation can make “the cat bites a mouse” and “the mouse bites a cat” nearly identical despite reversing semantic roles. Graphs and other structured objects also contain relations that are not captured by flat vectors. The slides therefore motivate richer distributed representations and representation learning as ways to learn a space in which proximity better reflects latent semantics.

== Data Quality, Cleaning, and Integration

*Exam: ★★★★☆*

Real-world data are often incomplete, noisy, inconsistent, duplicated, stale, or difficult to interpret. The slides describe data quality as multidimensional: *accuracy* asks whether values are correct; *completeness* whether required values are available; *consistency* whether sources and records agree; *timeliness* whether values are current; *believability* whether the source can be trusted; and *interpretability* whether the representation is understandable.

Preprocessing therefore includes data cleaning, integration, reduction, transformation, and discretization. Cleaning handles missingness, noise, outliers, and inconsistencies. Integration combines sources. Reduction seeks a smaller representation with nearly the same analytical value. Transformation changes the representation, including normalization and discretization.

=== Missing, noisy, inconsistent, and disguised values

Missing data can arise from equipment malfunction, deletion of inconsistent measurements, misunderstanding during data entry, fields that were not considered important at collection time, or failure to record historical changes. Common responses in the slides are:

- ignore the tuple, especially when a class label is missing and losing the tuple is acceptable;
- fill the value manually, which is usually infeasible at scale;
- use a global constant such as “unknown”;
- impute the attribute mean;
- impute a class-conditional mean;
- infer the most probable value using a probabilistic model or decision tree.

These methods make different assumptions. Mean imputation is simple but shrinks variability; a global “unknown” category changes the attribute's state space; inference-based methods can preserve structure better but can also introduce model bias.

Noise is random error or variance in a measured variable. Incorrect values may originate from faulty instruments, data-entry or transmission errors, technical limits, or inconsistent naming conventions. Other quality problems include duplicates, incomplete records, and inconsistencies such as a recorded age that conflicts with the date of birth, or a rating system that changed from numeric to alphabetic codes. Missingness can also be disguised intentionally or operationally, such as an implausible default birthday used for many people.

The slides list several noise-handling methods. *Binning* first sorts the data and partitions it, then smooths values by bin means, medians, or boundaries. *Regression* smooths by fitting a function. *Clustering* can expose observations that do not fit dense groups. *Semi-supervised inspection* flags suspicious values computationally and delegates ambiguous cases to humans.

=== Cleaning as an iterative process

Data cleaning begins with discrepancy detection using metadata such as domains, ranges, dependencies, distributions, uniqueness constraints, sequential rules, and null rules. It should also detect overloaded fields such as a single Name field containing multiple subfields. *Data scrubbing* uses simple domain knowledge, dictionaries, postal-code rules, or spell checking to detect and correct errors. *Data auditing* analyzes the data itself to discover rules and relationships whose violators may be errors or outliers, for example through correlation or clustering.

Migration and ETL tools operationalize transformations across systems: ETL means extraction, transformation, and loading. The slides stress that cleaning and integration are not one-off stages; they are iterative and interactive because correcting one discrepancy can expose another.

=== Data integration and conflict resolution

*Data integration* combines multiple databases, cubes, or files into a coherent store. It can provide a more complete picture and improve mining efficiency and quality, but it introduces schema, entity, and value-resolution problems.

*Schema integration* aligns semantically equivalent fields, such as `A.cust-id` and `B.cust-#`, and reconciles metadata. *Entity identification* determines when different records refer to the same real-world entity, even when names differ.

After entities are aligned, sources may still disagree about attribute values because of different representations, timestamps, units, scales, or source errors. The slides suggest several conflict-resolution policies: mean/median/mode/max/min aggregation, choosing the most recent value, or *truth finding* that weights sources by quality. The appropriate rule depends on semantics; for example, averaging two incompatible categorical codes is meaningless, whereas averaging repeated noisy measurements may be sensible.

Integration also introduces redundancy. The same attribute can appear under multiple names, or one field can be derivable from others, such as annual revenue from periodic values. Correlation and covariance can help detect redundant numerical attributes, but a high correlation alone does not prove semantic duplication.

#informally[
  Cleaning asks “is this value trustworthy and internally coherent?” Integration adds “does this value refer to the same entity and meaning across sources?” The two processes therefore interact: entity matching can reveal conflicts, while cleaning can improve matching.
]

== Data Transformation, Discretization, Compression, Sampling, and Reduction

*Exam: ★★★★★*

A data transformation maps values or representations to new values while retaining the information needed for later analysis. The slides group smoothing, feature construction, aggregation, normalization, and discretization under transformation, and then treat compression, sampling, and alternative representations as forms of data reduction.

=== Normalization

Normalization places numerical attributes on comparable scales. For min-max normalization of an attribute $A$ from $[upright("min")_A,upright("max")_A]$ to $[upright("new min")_A,upright("new max")_A]$,

$
  v' = (v-upright("min")_A)/(upright("max")_A-upright("min")_A)
       (upright("new max")_A-upright("new min")_A) + upright("new min")_A.
$

For the slide example with income between 12,000 and 98,000, mapping to $[0,1]$ sends 73,600 to approximately 0.716.

Z-score normalization uses the mean and standard deviation:

$
  v' = (v-mu_A)/sigma_A.
$

Thus if $mu=54000$, $sigma=16000$, and $v=73600$, then $v'=1.225$: the value lies 1.225 standard deviations above the mean.

Decimal scaling uses

$
  v' = v/10^j,
$

where $j$ is the smallest integer that makes the maximum absolute transformed value less than 1.

Normalization is especially important before distance-based methods when attributes have different units or scales; otherwise the largest-scale attribute can dominate proximity.

=== Discretization and concept hierarchies

*Discretization* divides the range of a continuous numerical attribute into intervals and replaces raw values by interval labels. It can reduce data size, simplify downstream models, and create higher-level concepts. Methods may be supervised or unsupervised, top-down splitting or bottom-up merging, and can be applied recursively.

The slides classify common methods as follows. Binning and histogram-based discretization are unsupervised top-down methods. Clustering can support unsupervised splitting or merging. Decision-tree discretization is supervised and chooses split points using class information such as entropy. ChiMerge is a supervised bottom-up method: adjacent intervals with similar class distributions produce low chi-square values and are merged recursively until a stopping condition is reached.

#warning[
  One slide labels “correlation (e.g., $chi^2$) analysis” for discretization as *unsupervised*, while the later ChiMerge slide correctly says it uses class information and is *supervised*. For ChiMerge specifically, the latter is the correct description.
]

For *equal-width* binning over attribute range $[A,B]$ with $N$ bins, the bin width is

$
  W = (B-A)/N.
$

It is simple but sensitive to outliers and often poor for skewed data because sparse tails can consume large ranges. *Equal-depth* or equal-frequency binning instead creates bins with approximately equal sample counts, adapting better to skewed densities but producing unequal numeric widths.

#example[
  The slides use sorted prices
  $4,8,9,15,21,21,24,25,26,28,29,34$.
  Equal-depth binning into three groups gives $(4,8,9,15)$, $(21,21,24,25)$, and $(26,28,29,34)$. Smoothing by bin means replaces these groups by roughly 9, 23, and 29 respectively. Smoothing by boundaries instead maps each value to the closest endpoint within its bin.
]

The visual comparison in the slides shows why clustering can outperform rigid binning: equal-width and equal-depth boundaries are constrained by the binning rule, whereas K-means can adapt boundaries to groups that are naturally separated in the data.

A *concept hierarchy* recursively replaces low-level values by higher-level concepts, enabling analysis at multiple granularities. Numeric hierarchies can be built by discretization, such as mapping exact ages to youth/adult/senior. For nominal data, hierarchies may be specified explicitly (`street < city < state < country`), defined through grouped values, partially specified and completed later, or inferred heuristically from distinct-value counts. The heuristic places attributes with more distinct values at lower levels, although semantic exceptions such as weekday/month/quarter/year show that cardinality alone is not definitive.

=== Compression and sampling

Compression reduces storage by exploiting redundancy. String compression is commonly lossless, while audio/video compression is often lossy and progressively refinable. Data reduction and dimensionality reduction can also be viewed as semantic forms of compression because they retain only the representation required for analysis.

*Sampling* selects a subset $s$ to represent a larger dataset of size $N$, potentially reducing algorithmic cost below linear dependence on $N$. The critical requirement is representativeness. Simple random sampling gives each item equal selection probability. In sampling *without replacement*, a selected item cannot be selected again; in sampling *with replacement*, each draw is made from the full population, simplifying some probabilistic analyses and bootstrap-style procedures.

When the population is skewed, simple random sampling can underrepresent rare but important groups. *Stratified sampling* partitions the population into strata and samples from each stratum, often proportionally, to preserve group representation.

#warning[
  The slides casually use “partition (or cluster)” while describing stratified sampling. A *stratum* is not the same as a cluster in cluster sampling: stratified sampling samples within every stratum, whereas cluster sampling selects clusters and then samples or observes units from selected clusters.
]

Sampling can reduce the number of records processed by a mining algorithm but may not reduce physical database I/O proportionally when storage is read in whole pages or blocks.

=== Data reduction: parametric and non-parametric representations

The goal of data reduction is a much smaller representation that yields nearly the same analytical result. Parametric methods assume a model, estimate a small parameter set, and may discard most raw data. Non-parametric methods do not commit to a fixed global model and instead use summaries such as histograms, clusters, or samples.

In regression, a response variable $Y$ is modeled from one or more predictors. Simple linear regression uses

$
  Y = w X + b,
$

and typically estimates $w$ and $b$ by least squares. Multiple linear regression generalizes this to

$
  Y = b_0 + b_1 X_1 + b_2 X_2 + dots + b_d X_d.
$

Nonlinear regression uses a nonlinear relation and is often fitted iteratively. Log-linear models make the logarithm of the modeled quantity linear in parameters, permitting linear-model machinery to summarize certain multi-dimensional relationships. In the data-reduction view, the key point is not the regression task itself but the replacement of many observations by a compact set of fitted parameters.

Histograms reduce data by storing bucket-level summaries rather than individual records. Clustering partitions data by similarity and can store only cluster representatives such as centroids and diameters; this is effective when the data genuinely form compact groups but less effective when the distribution is smeared without clear clusters.

== Dimensionality Reduction

*Exam: ★★★★★*

Dimensionality reduction addresses the growth of sparsity and combinatorial complexity as the number of features increases. In high-dimensional spaces, local density is harder to estimate, pairwise distances can become less discriminative, and the number of possible feature subsets grows exponentially. Reducing dimensionality can remove irrelevant or redundant features, suppress noise, reduce time and space cost, and enable visualization.

Two broad strategies must be distinguished. *Feature selection* keeps a subset of the original variables. *Feature extraction* creates a new lower-dimensional representation, usually by combining or transforming the original features.

=== Principal Component Analysis

PCA is a linear feature-extraction method. It applies an orthogonal transformation so that possibly correlated numerical variables are represented by linearly uncorrelated *principal components*. The components are directions of decreasing variance, and the top components provide a lower-dimensional approximation.

Let centered data vectors be $x_i in RR^d$ with covariance matrix $Sigma$. PCA solves the eigenproblem

$
  Sigma v_j = lambda_j v_j,
$

where the eigenvectors $v_j$ define orthogonal directions and the eigenvalues $lambda_j$ give the variance captured along those directions. Sorting $lambda_1 >= lambda_2 >= dots >= lambda_d$ orders components from strongest to weakest. Keeping the first $k$ eigenvectors in $V_k=(v_1,dots,v_k)$ projects a centered observation to

$
  z = V_k^T (x-mu),
$

and reconstructs the rank-$k$ approximation as $hat(x)=mu+V_k z$.

The slides' method can therefore be organized as:

- center the numerical data; when feature scales are incomparable, also scale or standardize them;
- compute the covariance matrix;
- compute its eigenvectors and eigenvalues;
- sort eigenvectors by decreasing eigenvalue;
- retain the top $k$ directions and project the data onto them.

The three-camera example supplies the geometric intuition: several measured coordinates can be highly redundant because the ball actually moves along a lower-dimensional trajectory. PCA discovers directions that summarize the dominant linear variation.

#warning[
  The slides say “normalize input data: each attribute falls within the same range.” PCA itself does not mathematically require min-max normalization. What is essential is centering; scaling/standardization is a modeling choice that becomes important when features use different units or variances.
]

#warning[
  A later slide says PCA works well when data are approximately Gaussian or form linearly separable clusters. This is too restrictive and potentially misleading. PCA does not require Gaussian data or class labels; it optimizes a *linear variance/reconstruction* criterion. Its limitation is that a linear subspace may not capture a curved manifold or neighborhood geometry.
]

=== Attribute subset selection

Attribute subset selection removes redundant or irrelevant original variables. A redundant attribute duplicates information already available elsewhere, such as a sales-tax field that is deterministically derived from purchase price under a fixed rate. An irrelevant attribute provides little useful information for the current task, such as a student ID for GPA prediction.

Exhaustively evaluating all subsets of $d$ attributes requires considering $2^d$ combinations, so the slides motivate heuristic search:

- choose the best single attribute under an independence assumption;
- *forward selection*: start from the best attribute and repeatedly add the next attribute that most improves the criterion conditional on those already selected;
- *backward elimination*: start with a larger set and repeatedly remove the weakest attribute;
- combine forward addition and backward checking;
- use branch-and-bound search when a suitable bound permits pruning and backtracking.

#example[
  In the slide's regression example, the null model has RMSE 14.2. Adding StudyHours reduces RMSE to 9.1, Attendance to 10.8, and FavoriteColor gives no improvement, so StudyHours is selected first. Attendance is then added, while FavoriteColor is rejected. Backward checks retain StudyHours and Attendance because both remain useful.
]

#warning[
  The example slide labels Attendance as `$X_6$` in one line even though it defines Attendance as `$X_2$`. This is a slide typo; the intended variable is $X_2$.
]

Feature generation is the complementary idea: rather than merely deleting attributes, construct more informative ones. The slides distinguish domain-specific extraction, mappings to new spaces such as Fourier/wavelet/manifold transforms, direct feature construction by combining existing attributes, and discretization.

=== Nonlinear dimensionality reduction: preserving proximity

When important structure is nonlinear, a linear projection can fail even if the intrinsic dimension is low. The slides express nonlinear dimensionality reduction through a common two-step idea: construct a high-dimensional proximity matrix $P$, then learn low-dimensional representations whose induced proximity $hat(P)$ is as close as possible to $P$. Different methods differ mainly in how they define proximity and how they measure preservation error.

*Kernel PCA.*


Kernel PCA first defines a kernel matrix

$
  P_(i j) = kappa(x_i,x_j).
$

Typical kernels in the slides include the polynomial kernel

$
  kappa(x_i,x_j) = (1 + x_i dot x_j)^p,
$

and the radial basis function kernel

$
  kappa(x_i,x_j) = exp(-((x_i-x_j) dot (x_i-x_j))/(2 sigma^2)).
$

After centering the kernel matrix, KPCA uses its leading eigenvectors/eigenvalues to obtain a low-dimensional representation. The slides summarize its preservation objective as making $hat(P)$ close to $P$, for example by a squared Frobenius discrepancy $sum_(i,j) (P_(i j)-hat(P)_(i j))^2$.

#warning[
  The statement “with a linear kernel, KPCA degenerates to standard PCA” is correct only with the usual centering correspondence: the data/kernel must be centered consistently. Without centering, the equivalence is not exact.
]

*Stochastic Neighborhood Embedding.*


SNE interprets proximity probabilistically. In the simplified form shown in the slides, high-dimensional neighbor probabilities are defined from squared distances,

$
  P_(i j) = exp(-d_(i j)^2) / sum_(l != i) exp(-d_(i l)^2),
$

with $d_(i j)^2$ proportional to $((x_i-x_j) dot (x_i-x_j))/(2 sigma^2)$. A low-dimensional embedding $hat(x)_i$ induces corresponding probabilities $hat(P)_(i j)$. The embedding is optimized so that each original neighborhood distribution is close to its low-dimensional counterpart, using

$
  min_(hat(x)_1,dots,hat(x)_n) sum_(i=1)^n D_("KL")(P_i || hat(P)_i).
$

Thus KPCA and SNE fit the same high-level template—construct proximity, then preserve it—but use different definitions and loss functions: kernel similarity plus spectral decomposition for KPCA, versus stochastic neighborhoods plus KL divergence for SNE.

#note[
  The slides use a simplified SNE formula with a common scale $sigma$. Standard SNE formulations may use point-dependent bandwidths selected to control neighborhood perplexity. The course-level idea here is the KL-based preservation of local neighbor probabilities.
]

=== Linear versus nonlinear geometry

The final visual examples show two intertwined classes in two dimensions. PCA cannot undo the nonlinear geometry with a linear projection, while RBF-kernel PCA and t-SNE produce embeddings in which the two groups are much more clearly separated. The accompanying proximity heatmaps have two diagonal blocks for within-class proximity and off-diagonal blocks for cross-class proximity; the nonlinear methods produce stronger within-group and weaker between-group proximity in this example.

#warning[
  The slide expands t-SNE as “t-distributional NSE.” The standard name is *t-distributed Stochastic Neighbor Embedding*.
]

#warning[
  The figures demonstrate what happens *for this dataset*; they do not imply that KPCA or t-SNE is guaranteed to make arbitrary classes linearly separable. In particular, t-SNE is primarily a neighborhood-preserving visualization method, and global distances or apparent cluster separation should not automatically be interpreted as supervised classification structure.
]

== Chapter Perspective

*Exam: ★★★★☆*

The chapter forms one preprocessing logic rather than several disconnected topics. First identify what objects and attribute scales mean. Then summarize distributions and relationships with statistics and visualizations. Choose proximity measures that match those semantics. Before mining, diagnose missingness, noise, inconsistency, conflicts, and redundancy; clean and integrate accordingly. Transform values when scale or granularity is unsuitable, and reduce data size when full resolution is unnecessary. Finally, reduce feature dimensionality either by selecting original attributes or by extracting a lower-dimensional representation such as PCA, KPCA, or SNE.

A recurring exam principle is to ask *what information an operation preserves*. Mean/variance preserve only coarse moments; a boxplot preserves five-number-summary structure but not modality; normalization changes scale but not ordering; discretization replaces exact values by interval identity; sampling preserves information only to the extent that the sample is representative; PCA preserves dominant linear variance; nonlinear methods instead attempt to preserve a chosen notion of proximity. Correct preprocessing is therefore inseparable from the downstream question one wants the data to answer.
