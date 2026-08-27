// @title: LLMs in Data Mining
// @description: 大语言模型、文档处理、检索以及 LLM 辅助的数据挖掘方法。
// @order: 100

#import "../template.typ": *

= LLMs in Data Mining

== Overview

*Exam: ★★★★☆*

This chapter studies how large language models (LLMs) enter a data-mining workflow at three levels. First, an LLM is a probabilistic language model controlled through prompts and decoding. Second, unstructured documents must be normalized, enriched with metadata, and chunked before reliable retrieval. Third, LLMs can augment classical data-mining tasks with semantic knowledge, pseudo-constraints, generated features, and executable operations. The progression is

$ "language modeling" -> "prompted LLM" -> "retrieval-ready data" -> "LLM-assisted data mining". $

The central idea is not that an LLM replaces every conventional algorithm. In several examples, the LLM contributes semantic interpretation, domain knowledge, natural-language reasoning, or code generation, while retrieval, clustering, validation, and execution components retain their own roles.

== From Language Modeling to Large Language Models

*Exam: ★★★★☆*

=== The language-modeling problem

A language model assigns probability to a token sequence and, equivalently, a conditional distribution to the next token given its preceding context. For $x^(1), ..., x^(N)$, the chain rule gives

$
  p(x^(1), ..., x^(N)) = product_(i=1)^N p(x^(i) | x^(1), ..., x^(i-1)).
$

Thus a model can rank sentences by plausibility and generate autoregressively: after a context such as ``the students opened their'', it assigns probabilities to continuations such as ``books'' or ``laptops''. This also explains its role as a judge of grammaticality, semantic plausibility, and stylistic consistency.

#informally[
A generative LLM is a very large next-token predictor. Multi-turn dialogue uses the previous dialogue plus the current message as context, predicts one next token, appends it, and repeats until stopping.
]

Language modeling is useful whenever several candidate sequences compete. The slides illustrate machine translation, spelling correction, and speech recognition; the same capability supports summarization, question answering, OCR-related processing, and other language tasks. Unified language modeling goes further by expressing many tasks---QA, sentiment analysis, summarization, natural-language inference, sentence completion, commonsense reasoning, and others---through textual inputs and outputs rather than task-specific interfaces.

A model may also appear to store factual knowledge because training encodes many statistical associations. This role is less reliable than grammatical or distributional scoring: fluent generation does not guarantee factual correctness.

=== What makes an LM ``large''

The lecture characterizes an LLM as a language model with many parameters, trained on large amounts of data for a long time. Transformer architectures make large-scale parallel computation practical, enabling both larger corpora and larger models. The historical figures emphasize rapidly increasing model scale and training cost and show multiple model families rather than one canonical LLM.

#warning[
The model catalogue is a historical snapshot and contains factual/terminological issues. It labels PaLM 2 as ``open-source'', although its weights were not released as an open-source model, and it gives Llama 3.1 a June 23, 2024 release date rather than its July 2024 release. Treat the table as an illustrative 2024-era landscape, not an authoritative current registry.
]

The lecture illustrates code generation, mathematical reasoning, conversational assistance, long-context summarization, and search augmented by external information. These motivate the rest of the chapter: useful behavior depends not only on pretraining, but on task specification, supplied context, and output checking.

#warning[
The slides repeatedly describe generation as choosing the next ``word''. Modern LLMs generally operate on *tokens*, which may be words, word pieces, punctuation, or other units. ``Next-token prediction'' is the technically correct formulation.
]

== Prompt Engineering and Controlled Generation

*Exam: ★★★★★*

Prompt engineering specifies the task, context, constraints, and desired output form so that a general-purpose model behaves like a task-specific component. The lecture organizes good prompting around two principles: make instructions clear and specific, and structure multi-step tasks so that necessary intermediate work occurs before the final answer.

=== Clear task specification

Clear does not mean short. A robust prompt should separate instructions from data, state the expected output representation, and specify what to do when prerequisites fail. Four recurring tactics are:

- use delimiters such as triple quotes, backticks, dashes, angle brackets, or XML-style tags;
- request structured output such as JSON or HTML for downstream consumption;
- ask the model to check task conditions or assumptions before execution;
- use few-shot prompting by providing successful input-output examples.

Delimiters clarify which text is data and which text is the governing instruction, especially when supplied content contains instruction-like strings. They should not, however, be regarded as a complete security mechanism against prompt injection.

=== Decomposition and iterative development

For multi-stage tasks, the prompt can explicitly specify ordered steps. The lecture also recommends having the model work out a problem before judging a student's proposed solution; otherwise it may accept a superficially plausible error.

Prompt design is therefore iterative: write a clear specification, test it, analyze failures, refine ambiguity or task decomposition, and evaluate the revision on a *batch* of examples. Batch evaluation matters because fixing one example can regress on others.

#warning[
``Give the model time to think'' is best interpreted operationally as *requesting task decomposition and intermediate checks*, not as literal wall-clock thinking time.
]

=== Hallucination and retrieval grounding

A central limitation is hallucination: plausible, fluent statements may be false. The lecture's mitigation strategy is retrieval grounding: first find relevant information, then answer using it. This separates evidence acquisition from generation and anticipates the retrieval pipeline later in the chapter.

=== Prompt-level capabilities

The slides group common prompted operations into four families. *Summarizing* compresses input while preserving requested aspects. *Inferring* extracts information such as sentiment, semantics, entities, or topics. *Transforming* changes representation while preserving intended content, such as translation or code conversion. *Expanding* turns a compact specification into a longer artifact. These are task formulations, not separate model architectures.

=== Temperature

Temperature controls the sharpness of the sampling distribution. For logits $z_i$, the temperature-scaled probability is

$
  p_i = exp(z_i / T) / sum_j exp(z_j / T),
$

where $T > 0$. Lower $T$ concentrates probability on high-logit tokens and makes sampling more predictable; higher $T$ flattens the distribution and increases variability. Low temperature therefore suits repeatability-oriented tasks, while higher temperature can encourage diversity.

#warning[
At exactly $T = 0$, the displayed formula is undefined. APIs exposing ``temperature = 0'' normally implement deterministic or near-greedy decoding rather than literally substituting zero into this softmax expression.
]

== Preparing Unstructured Documents for LLM Applications

*Exam: ★★★★★*

Raw documents are heterogeneous: PDFs, HTML pages, presentations, JSON, CSV, and other formats encode structure differently. An LLM application therefore needs a preprocessing layer that converts source-specific representations into a common reusable structure. The lecture distinguishes document content for keyword/similarity search, document elements such as titles, narrative text, list items, tables, and images, and element metadata such as filename, file type, page number, and section.

=== Normalization and serialization

Preprocessing is difficult because formats use different cues for the same semantic element. A title may be represented visually in a PDF, structurally in HTML, or through a slide placeholder in PowerPoint. Extraction procedures differ by format, and useful metadata often requires understanding hierarchy rather than merely extracting characters.

Normalization converts source-specific structures into common typed elements. Downstream logic can then filter repeated headers/footers, combine elements into sections, and try alternative chunking strategies without repeating expensive source parsing. Serialization makes normalized results reusable. The slides favor JSON because it is widely understood, language-independent, common in HTTP interfaces, and convertible to JSONL for streaming. The HTML and PowerPoint examples illustrate the same abstraction: partition source content into typed elements and attach metadata in a unified representation.

=== Metadata and chunking

Metadata adds context beyond an element's main text. Source metadata includes URL, filename, or file type; structural metadata includes element type, hierarchy, page number, or section membership. It supports filtering, provenance, structural reconstruction, and retrieval.

Chunking divides documents into retrieval units suitable for vector databases and prompts. Roughly equal-sized chunks are simple but may split related content. Element-aware chunking first identifies atomic elements and then combines them, for example keeping material under one section heading together. The resulting chunks are more coherent because boundaries follow document structure rather than arbitrary character counts.

#informally[
Normalization asks, ``What are the meaningful document elements, independent of file format?'' Chunking asks, ``Which groups of those elements should become retrieval units?'' Separating the stages lets one normalized document support many chunking experiments.
]

=== LLM-assisted preprocessing pipeline

The lecture connects normalized documents to a vector database. A user issues a preprocessing query; retrieval selects relevant normalized content; that context is combined with the query; and the LLM generates the requested response:

$ "raw documents" -> "normalized elements" -> "vector database" -> "retrieved context + query" -> "LLM response". $

This also operationalizes the earlier hallucination mitigation principle: generation is grounded in retrieved content rather than relying only on pretrained memory.

== LLM-Enhanced Core Data-Mining Tasks

*Exam: ★★★★☆*

The lecture uses pattern mining, classification, clustering, and anomaly detection to show different ways an LLM can complement classical data mining. The common theme is that the LLM contributes semantic judgments or knowledge that are difficult to encode with purely numerical rules.

=== Pattern mining as collaborative abstraction

The pattern-mining example uses an eight-step collaborative process: (1) identify initial examples, (2) extract recurring solutions, (3) define the problems addressed by those solutions, (4) distill problem-solution pairs into named patterns, (5) identify system affordances, (6) relate patterns to those affordances, (7) refine patterns by adding limits and removing overlap, and (8) consolidate them into a reusable collection.

#example[
For online-learning engagement, observed cases include short videos, frequent quizzes, progress badges, and discussion prompts. These are abstracted into patterns such as ``Short Content Units'' and ``Frequent Small Quizzes'', linked to platform affordances such as video hosting or quiz tools, then documented with a problem, solution, rationale, applicability conditions, and examples. Here the mining target is a semantic design pattern rather than merely a frequent itemset.
]

=== Text classification through progressive reasoning

The cited classification approach uses a *progressive reasoning strategy*: before predicting a class such as sentiment, the prompt asks the LLM to identify relatively superficial clues---keywords, tone, or semantic cues---and then use them to determine the label. The lecture reports that this approach outperforms vanilla zero-shot prompting and zero-shot chain-of-thought in its shown example.

The methodological point is decomposition. Direct zero-shot classification asks for $x -> y$ immediately, whereas progressive prompting inserts an evidence stage $x -> c -> y$, where $c$ denotes task-relevant clues. This can ground the final decision more explicitly in the input.

=== Few-shot clustering with LLM-generated constraints

The clustering example preserves a conventional algorithm but changes the source of supervision. An LLM generates pairwise constraints; pairwise-constrained K-means consumes these pseudo-oracle constraints to form clusters. Such constraints express whether two examples should belong to the same cluster or to different clusters. The LLM supplies semantic pairwise judgments, while the clustering algorithm combines them with geometric structure; the method is not simply ``ask the LLM for cluster labels''.

#warning[
The slide title says ``few-short clustering''. The cited paper and intended term are *few-shot clustering*.
]

=== LLM-assisted anomaly detection

The lecture distinguishes two roles. In *zero-shot detection*, pretrained LLM knowledge is used for anomaly detection without task-specific training. In *data augmentation*, the LLM generates synthetic examples and category descriptions that enrich an anomaly-detection model. The first uses the LLM as detector; the second uses it to improve data or semantic context for another detector.

== LLMs for Tabular Data

*Exam: ★★★★★*

Tabular data exposes a spectrum of LLM integration. A table can be serialized so prediction becomes language modeling; an LLM can contribute domain knowledge by generating features evaluated by a conventional model; or it can generate executable code for spreadsheet operations. These approaches differ in where numerical computation and validation occur.

=== LIFT: prediction through a language interface

LIFT (Language-Interfaced Fine-Tuning) converts non-language supervised-learning tasks into text-to-text tasks. Non-linguistic inputs---including tabular values, image features, or molecular structures in the slides---are serialized into natural-language representations. Targets become answers or fill-in-the-blank completions, allowing a unified text-generation objective.

If $s(x)$ is the serialized input and $t(y)$ the textual target, the slide's verbal objective can be formalized as

$
  max_theta sum_(x,y) log p_theta(t(y) | s(x)).
$

Here $theta$ denotes model parameters. Ordinary supervised prediction has been recast as conditional language modeling.

#note[
The likelihood equation is a compact formalization of the slide's verbal training objective; the slide itself does not provide an explicit formula.
]

=== CAAFE: domain knowledge for feature engineering

CAAFE uses an LLM as a context-aware feature engineer rather than the final predictor. The prompt contains dataset context, feature names, data types, and a small set of samples. The LLM proposes new features and Python code, an interpreter executes the code, a predictive model evaluates the resulting features on validation data, and useful features are retained for the next iteration.

$ "dataset context" -> "LLM feature code" -> "execution" -> "validation" -> "retain or reject". $

The validation loop is crucial: semantic domain knowledge generates candidates, but empirical downstream performance decides whether they survive. The lecture reports improvements across the shown classifiers and better results than context-agnostic automated feature-engineering baselines such as DFS and AutoFeat.

=== TableLLM: reasoning versus operation execution

TableLLM distinguishes two regimes. *Document-embedded tables* are relatively small and use a reading-comprehension style: given relevant text, a table, and a question, the model returns a textual answer. *Spreadsheet-embedded tables* may contain hundreds or thousands of rows and require querying, updating, merging, sorting, and chart generation; here the model generates code and obtains the result through execution.

The training-data construction mirrors this distinction. For text-driven reasoning, question-answer pairs from WikiTQ, FeTaQA, and TAT-QA are expanded into textual reasoning chains. For spreadsheet operations, data from WikiTQ, TAT-QA, FeTaQA, and GitTables are expanded into ten operation types covering Query, Update, Merge, and Chart tasks. The slides describe consistency filtering: multiple textual candidates are generated and the one most consistent with the reference is selected; for code, many Pandas solutions are generated and a majority/consistency criterion is used to choose a reference-consistent solution.

Training then uses two interfaces:

- *Text-driven:* input = related text + table + instruction; output = textual answer.
- *Code-driven:* input = table header + first few rows + operation instruction; output = Pandas code.

This separation addresses a practical scaling issue. Small tables can often be reasoned over directly in context, but large spreadsheets are better handled by generating operations that a deterministic execution engine applies to the full table. The evaluation table on the final TableLLM slide reports strong average performance for TableLLM across both document-embedded and spreadsheet-embedded benchmarks while using one inference per example.

#warning[
Slide 66 labels two different stages as ``③'', and slide 67 likewise labels both training branches as ``④''. These are parallel text-driven and code-driven branches, not four sequential steps with duplicated numbering.
]

== Synthesis

*Exam: ★★★★☆*

Across the chapter, LLM-enhanced data mining follows a recurring architecture: translate heterogeneous data or tasks into an interface the model can understand, use prompting or fine-tuning to obtain semantic judgments or executable proposals, and retain conventional components for retrieval, structure, execution, or validation. The important distinctions are therefore where the LLM enters the pipeline and what evidence checks its output.

For exam purposes, be able to explain the autoregressive factorization and temperature equation; the two prompting principles and their tactics; why normalization, metadata, and chunking are separate preprocessing stages; retrieval grounding as a response to hallucination; and the specific role of the LLM in each data-mining example. In particular, distinguish direct prediction (LIFT), semantic candidate generation followed by validation (CAAFE), pseudo-constraint generation followed by clustering, and code generation followed by table execution (TableLLM).
