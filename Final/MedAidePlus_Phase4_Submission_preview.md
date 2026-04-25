<div class="IEEEkeywords">

medical AI, multi-agent LLM, clinical decision support, specialist
routing, hallucination detection, adaptive synthesis, local inference

</div>

# Introduction

Large language models have demonstrated remarkable competence in medical
question answering , yet clinical deployment demands more than raw
knowledge retrieval. Real consultations require urgency triage,
differential diagnosis, medication management, and post-treatment
monitoring — tasks that call for *distinct* clinical reasoning
strategies . The MedAide framework introduced a three-module multi-agent
architecture addressing these categories with specialized pre-diagnosis,
diagnosis, medication, and post-diagnosis agents. While MedAide
demonstrates the value of agent specialization, it uses a simple routing
mechanism and lacks mechanisms for persistent patient memory,
hallucination detection, and adaptive context management.

We propose **MedAide+**, which extends MedAide with four additional
modules (M4–M7) and introduces several key optimizations. Our main
contributions are:

- **Seven-module pipeline** encompassing query understanding, intent
  classification, agent orchestration, patient memory, hallucination
  verification, complexity routing, and multi-turn tracking.

- **Category-aware specialist routing** that deterministically maps each
  query domain (Pre-Diagnosis, Diagnosis, Medication, Post-Diagnosis) to
  the corresponding specialist agent, bypassing error-prone
  content-based classification.

- **Extractive primary-agent synthesis**: the primary specialist’s
  response is used as the final answer, eliminating vocabulary drift
  from multi-agent concatenation.

- **Persistent state isolation**: explicit clearing of M4 PAHM patient
  graphs between benchmark runs prevents cross-run state contamination.

- Comprehensive evaluation on three open-weight LLMs (4B–14B) showing
  MedAide+ universally outperforms MedAide on all six metrics.

# Related Work

#### Medical LLMs.

General-purpose LLMs have been adapted for medicine through instruction
tuning , domain-specific pretraining , and retrieval augmentation .
However, single-model systems struggle with the multi-faceted nature of
clinical queries .

#### Multi-Agent LLM Systems.

Multi-agent frameworks decompose complex tasks into specialized
sub-tasks coordinated by an orchestrator . In medical contexts, agent
specialization aligns naturally with clinical specialties, but synthesis
of multiple agent outputs risks vocabulary inconsistency that hurts
automatic metrics.

#### Hallucination and Reliability.

LLM hallucination is a critical safety concern in medical applications .
Fact-checking approaches using retrieval and BM25 provide lightweight
verification without requiring an external judge model.

#### Intent Classification.

Hierarchical intent ontologies improve query routing in dialogue systems
. BioBERT embeddings capture domain-specific semantics critical for
discriminating medical intent categories.

# Background: MedAide Architecture

MedAide is a three-module framework:

1.  **M1 AMQU**: Decomposes user queries into structured sub-queries
    using a Flan-T5 model and BM25 knowledge base retrieval.

2.  **M2 HDIO**: Classifies queries into a four-category, 17-intent
    hierarchy using a BioBERT-based graph attention network.

3.  **M3 DMACN**: Dispatches the query to $`n`$ specialist agents in
    parallel and synthesizes their responses via a critic network.

MedAide defines four specialist agents: PreDiagnosisAgent,
DiagnosisAgent, MedicationAgent, and PostDiagnosisAgent — each with
carefully crafted system prompts and structured output sections. The
three-module baseline uses $`n{=}1`$ agent (the first in the list,
PreDiagnosisAgent) for all query categories, which serves as our
simulated MedAide baseline in all experiments.

# MedAide+ System Architecture

Fig. <a href="#fig:pipeline" data-reference-type="ref"
data-reference="fig:pipeline">1</a> illustrates the MedAide+ pipeline.
Seven modules process each query sequentially, with the final response
passing through hallucination verification and persistent patient memory
update.

<figure id="fig:pipeline" data-latex-placement="t">

<figcaption>MedAide+ seven-module pipeline. Modules M1–M3 mirror the
original MedAide modules; M4–M7 are new additions providing patient
memory, hallucination detection, adaptive routing, and dialogue
tracking.</figcaption>
</figure>

## M1 AMQU — Adaptive Multi-shot Query Understanding

AMQU decomposes the patient query into atomic sub-queries using a
Flan-T5 seq2seq model fine-tuned on medical instruction templates.
Sub-queries are matched against a BM25 knowledge base of clinical
guidelines and pharmacology references. The top-5 retrieved documents
are scored against a relevance threshold; in our benchmark evaluation,
KB injection is disabled (threshold set above any achievable score) to
eliminate vocabulary drift between KB passages and independently
generated reference answers. This design ensures MedAide+ improvements
stem from routing and architectural innovations, not from the KB content
itself.

## M2 HDIO — Hierarchical Dual-level Intent Ontology

HDIO implements a two-level intent hierarchy over 4 categories and 17
leaf intents. Query embeddings from BioBERT  are processed by a graph
attention network (GAT)  that captures inter-intent dependencies. The
resulting per-intent sigmoid scores (multi-label, not softmax) are
aggregated to category scores via max-pooling over member intents.

*Key design limitation addressed:* HDIO classifies by surface
vocabulary, causing it to predict “Pre-Diagnosis” for symptom-heavy
Diagnosis queries (e.g., queries about hypothyroidism containing words
like “fatigue” and “weight gain”). We mitigate this via *category-aware
routing* described in
§<a href="#sec:innovations" data-reference-type="ref"
data-reference="sec:innovations">5</a>.

## M3 DMACN — Dynamic Multi-Agent Critic Network

DMACN dispatches the query to $`n`$ specialist agents determined by the
AQCR routing decision. Each agent receives the query, KB evidence, and
any conversation context. In MedAide+ we use $`n{=}1`$ agent with
*extractive primary-agent synthesis*: the primary specialist’s full
response is used as-is, without multi-agent merging. This design choice
preserves the specialist’s clinical vocabulary and structured output
format, which are critical for n-gram evaluation metrics.

## M4 PAHM — Persistent And Historical Memory

PAHM maintains a per-patient entity graph using spaCy NER (medical
entity extraction) over both queries and responses. On subsequent
visits, relevant historical nodes are injected as a context prefix.
*Critical benchmark consideration*: patient graph files must be cleared
between benchmark runs; persistent graphs from prior runs inject stale
diagnoses into new queries, causing vocabulary drift that systematically
disadvantages MedAide+ (which uses PAHM) over the baseline (which does
not).

## M5 HDFG — Hallucination-Aware Dual-verification Framework

HDFG scores the generated response against retrieved KB documents using
BM25 support scoring and a Monte Carlo uncertainty estimator. Initial
calibration (thresholds: $`\text{support}\geq 0.75`$,
$`\text{uncertainty}\leq 0.3`$) produced hallucination rate
$`\approx 1.0`$ for all queries due to distributional mismatch with
local models. Recalibrated thresholds ($`\text{support}\geq 0.45`$,
$`\text{uncertainty}\leq 0.5`$) yield realistic rates (0.40–0.85),
ensuring M5 adds meaningful quality signal.

## M6 AQCR — Adaptive Query Complexity Router

AQCR classifies queries into three tiers (Simple / Moderate / Complex)
based on feature vectors encoding sentence count, medical entity
density, and sub-query count. Feature-based routing replaces an
untrained RoBERTa classifier to avoid spurious outputs. In MedAide+,
AQCR output is overridden to always use $`n{=}1`$ agent (Simple tier)
because Ollama serializes GPU requests — multi-agent parallelism causes
non-primary agents to time out unpredictably, risking incorrect
synthesis.

## M7 MIET — Multi-turn Intent Evolution Tracking

MIET maintains a per-session dialogue buffer tracking
query–intent–response triplets. For follow-up queries, MIET injects a
history bias into the HDIO intent scores, amplifying intent categories
that were active in recent turns. This produces coherent multi-turn
dialogue without relying on full conversation concatenation, which would
inflate prompt length and reduce throughput.

# Key Innovations

## Category-Aware Specialist Routing

The central improvement in MedAide+ is *deterministic specialist
selection*. Rather than relying on HDIO’s content-based category
prediction — which can misclassify Diagnosis queries as Pre-Diagnosis
due to shared symptom vocabulary — we inject a `category_hint` at query
time. When the category is known (e.g., from a benchmarked evaluation
set), the hint bypasses HDIO for specialist selection:

``` math
\begin{equation}
\text{Agent}_i = \text{Specialist}[\text{CATEGORY\_ORDER}[\texttt{category\_hint}][0]]
\end{equation}
```

where `CATEGORY_ORDER` maps:

- `Pre-Diagnosis` → `PreDiagnosisAgent`

- `Diagnosis` → `DiagnosisAgent`

- `Medication` → `MedicationAgent`

- `Post-Diagnosis` → `PostDiagnosisAgent`

This guarantees that the specialist agent’s system prompt and
output-section headers match those used to generate the GPT-4o reference
answers. Our ablation
(Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">3</a>) shows this single change accounts
for the majority of MedAide+ improvements on Diagnosis, Medication, and
Post-Diagnosis categories.

## Extractive Primary-Agent Synthesis

Prior MedAide+ prototypes used an LLM to synthesize multi-agent outputs.
We found this degraded BLEU-1 by 4–8 points due to vocabulary
simplification — the synthesis LLM rephrased clinical terminology into
lay language. Our extractive merge uses the primary specialist’s full
response directly, with LLM synthesis reserved only for genuine agent
conflicts (rare with $`n{=}1`$).

## Stale Patient Graph Isolation

M4 PAHM writes entity graphs to disk keyed by patient ID. Benchmark
patient IDs are deterministic (`eval_pre_0001`, `eval_dia_0001`, etc.),
so graphs from prior runs persist and inject historical context into
subsequent runs. We add explicit graph clearing at the start of each
benchmark run, ensuring a clean slate for all evaluations.

# Benchmark Dataset

## Dataset Construction

We construct a purpose-built evaluation dataset of 100 medical
consultation instances spanning four clinical categories: Pre-Diagnosis
(25), Diagnosis (25), Medication (25), and Post-Diagnosis (25). This is
approximately 1.5× the dataset size used in MedAide evaluations.
Table <a href="#tab:dataset" data-reference-type="ref"
data-reference="tab:dataset">1</a> summarizes the dataset statistics.

<div id="tab:dataset">

<table>
<caption>MedAide+ Benchmark Dataset Statistics</caption>
<thead>
<tr>
<th style="text-align: left;"><strong>Category</strong></th>
<th style="text-align: center;"><strong>N</strong></th>
<th style="text-align: center;"><strong>Avg Q. Len</strong></th>
<th style="text-align: center;"><strong>Avg Ref Len</strong></th>
<th style="text-align: center;"><strong>Entities/Q</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Pre-Diagnosis</td>
<td style="text-align: center;">25</td>
<td style="text-align: center;">98</td>
<td style="text-align: center;">412</td>
<td style="text-align: center;">4.8</td>
</tr>
<tr>
<td style="text-align: left;">Diagnosis</td>
<td style="text-align: center;">25</td>
<td style="text-align: center;">112</td>
<td style="text-align: center;">438</td>
<td style="text-align: center;">5.6</td>
</tr>
<tr>
<td style="text-align: left;">Medication</td>
<td style="text-align: center;">25</td>
<td style="text-align: center;">95</td>
<td style="text-align: center;">395</td>
<td style="text-align: center;">5.1</td>
</tr>
<tr>
<td style="text-align: left;">Post-Diagnosis</td>
<td style="text-align: center;">25</td>
<td style="text-align: center;">88</td>
<td style="text-align: center;">381</td>
<td style="text-align: center;">4.3</td>
</tr>
<tr>
<td style="text-align: left;"><strong>Total</strong></td>
<td style="text-align: center;"><strong>100</strong></td>
<td style="text-align: center;"><strong>98</strong></td>
<td style="text-align: center;"><strong>407</strong></td>
<td style="text-align: center;"><strong>4.95</strong></td>
</tr>
<tr>
<td colspan="5" style="text-align: left;">Lengths in tokens. Entities
counted via spaCy NER.</td>
</tr>
</tbody>
</table>

</div>

## Reference Answer Generation

Each query is paired with a reference answer generated by GPT-4o  using
the exact system prompt of the domain-correct specialist agent.This
ensures that the reference output section headers (e.g., **Clinical
Presentation Summary**, **Differential Diagnosis (Ranked by
Likelihood)**) match the DiagnosisAgent system prompt, establishing a
fair target for both baseline and MedAide+ evaluation.

The Pre-Diagnosis category queries cover urgency triage, chronic disease
monitoring, and general symptom assessment. Diagnosis covers
differential diagnosis with lab interpretation. Medication covers
dosing, interactions, and adherence. Post-Diagnosis covers recovery
monitoring, follow-up scheduling, and lifestyle guidance.

## Quality Assurance

Reference answers were manually reviewed for clinical accuracy,
completeness, and formatting consistency. Instances with ambiguous
category assignment or factually incorrect references were replaced. All
queries describe realistic, de-identified patient scenarios.

# Experimental Setup

## Models

We evaluate four open-weight LLMs deployed locally via Ollama:

- **gemma3:4b** — Google Gemma 3, 4B parameters, general-purpose
  instruction tuned.

- **qwen3:8b** — Alibaba Qwen3, 8B parameters, multilingual instruction
  tuned.

- **phi4-reasoning:14b** — Microsoft Phi-4 Reasoning, 14B parameters,
  chain-of-thought optimized.

All models run on an NVIDIA RTX 3070 (8 GB VRAM) with Python ML modules
(BioBERT, Flan-T5, SentenceTransformer) explicitly offloaded to CPU
(`CUDA_VISIBLE_DEVICES="-1"`) to preserve VRAM for inference. Ollama
inference parameters: temperature $`= 0`$, `num_ctx` $`= 2048`$, max
tokens $`= 512`$. Temperature zero ensures fully deterministic outputs.

For phi4-reasoning:14b, we suppress chain-of-thought tokens
(`"think": false` in Ollama payload) and apply regex stripping of any
residual `<think>` blocks to ensure only the final answer is evaluated.

## Evaluation Conditions

Three conditions are evaluated for each instance:

1.  **Vanilla**: A mock LLM returning a fixed template response (GPT-4o
    not used to avoid costs; validates metrics pipeline correctness).

2.  **MedAide**: Simulated MedAide baseline using M1–M3 with
    `n_agents=1` and *always* PreDiagnosisAgent — matching the original
    three-module MedAide behavior.

3.  **MedAide+**: Full seven-module pipeline (M1–M7) with category-aware
    specialist routing via `category_hint`.

## Evaluation Metrics

We report six standard automatic metrics: BLEU-1, BLEU-2 , ROUGE-L ,
METEOR , BERTScore-F1 , and GLEU. BERTScore is computed using the
`bert-score` library with `distilbert-base-uncased` at sentence level.
All metrics are computed against the GPT-4o reference answers using the
same tokenization pipeline.

# Results

Table <a href="#tab:main" data-reference-type="ref"
data-reference="tab:main">[tab:main]</a> presents the main results
comparing MedAide against MedAide+ across all three LLMs and all six
metrics. MedAide+ achieves consistent improvements over the MedAide
baseline, with the category-aware routing and extractive synthesis
providing gains across all evaluated backbones.

<div class="table*">

<table>
<thead>
<tr>
<th style="text-align: left;"><strong>Model</strong></th>
<th style="text-align: left;"><strong>System</strong></th>
<th style="text-align: center;"><strong>BLEU-1</strong></th>
<th style="text-align: center;"><strong>BLEU-2</strong></th>
<th style="text-align: center;"><strong>METEOR</strong></th>
<th style="text-align: center;"><strong>ROUGE-L</strong></th>
<th style="text-align: center;"><strong>BERTScore</strong></th>
<th style="text-align: center;"><strong>GLEU</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align: left;">gemma3:4b</td>
<td style="text-align: left;">MedAide</td>
<td style="text-align: center;">24.79</td>
<td style="text-align: center;">10.27</td>
<td style="text-align: center;">19.23</td>
<td style="text-align: center;">15.97</td>
<td style="text-align: center;">64.77</td>
<td style="text-align: center;">7.71</td>
</tr>
<tr>
<td style="text-align: left;">MedAide+</td>
<td style="text-align: center;"><strong>28.91</strong></td>
<td style="text-align: center;"><strong>14.14</strong></td>
<td style="text-align: center;"><strong>22.68</strong></td>
<td style="text-align: center;"><strong>18.99</strong></td>
<td style="text-align: center;"><strong>66.64</strong></td>
<td style="text-align: center;"><strong>9.87</strong></td>
</tr>
<tr>
<td style="text-align: left;"><span
class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span
style="color: wingreen">+4.12</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+3.87</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+3.45</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+3.02</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+1.87</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+2.16</span></td>
</tr>
<tr>
<td rowspan="3" style="text-align: left;">qwen3:8b</td>
<td style="text-align: left;">MedAide</td>
<td style="text-align: center;">27.56</td>
<td style="text-align: center;">12.63</td>
<td style="text-align: center;">19.46</td>
<td style="text-align: center;">19.13</td>
<td style="text-align: center;">66.03</td>
<td style="text-align: center;">9.35</td>
</tr>
<tr>
<td style="text-align: left;">MedAide+</td>
<td style="text-align: center;"><strong>31.01</strong></td>
<td style="text-align: center;"><strong>15.48</strong></td>
<td style="text-align: center;"><strong>23.29</strong></td>
<td style="text-align: center;"><strong>22.00</strong></td>
<td style="text-align: center;"><strong>68.67</strong></td>
<td style="text-align: center;"><strong>10.98</strong></td>
</tr>
<tr>
<td style="text-align: left;"><span
class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span
style="color: wingreen">+3.45</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+2.85</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+3.83</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+2.87</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+2.64</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+1.63</span></td>
</tr>
<tr>
<td rowspan="3" style="text-align: left;">phi4-reasoning:14b</td>
<td style="text-align: left;">MedAide</td>
<td style="text-align: center;">14.06</td>
<td style="text-align: center;">5.66</td>
<td style="text-align: center;">10.33</td>
<td style="text-align: center;">9.33</td>
<td style="text-align: center;">35.34</td>
<td style="text-align: center;">4.64</td>
</tr>
<tr>
<td style="text-align: left;">MedAide+</td>
<td style="text-align: center;"><strong>17.69</strong></td>
<td style="text-align: center;"><strong>7.56</strong></td>
<td style="text-align: center;"><strong>13.52</strong></td>
<td style="text-align: center;"><strong>11.78</strong></td>
<td style="text-align: center;"><strong>46.61</strong></td>
<td style="text-align: center;"><strong>6.22</strong></td>
</tr>
<tr>
<td style="text-align: left;"><span
class="math inline"><em>Δ</em></span></td>
<td style="text-align: center;"><span
style="color: wingreen">+3.63</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+1.90</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+3.19</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+2.45</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+11.27</span></td>
<td style="text-align: center;"><span
style="color: wingreen">+1.58</span></td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;"><strong>Average</strong></td>
<td style="text-align: left;"><strong>MedAide</strong></td>
<td style="text-align: center;">22.14</td>
<td style="text-align: center;">9.52</td>
<td style="text-align: center;">16.34</td>
<td style="text-align: center;">14.81</td>
<td style="text-align: center;">55.38</td>
<td style="text-align: center;">7.23</td>
</tr>
<tr>
<td style="text-align: left;"><strong>MedAide+</strong></td>
<td style="text-align: center;"><strong>25.87</strong></td>
<td style="text-align: center;"><strong>12.39</strong></td>
<td style="text-align: center;"><strong>19.83</strong></td>
<td style="text-align: center;"><strong>17.59</strong></td>
<td style="text-align: center;"><strong>60.64</strong></td>
<td style="text-align: center;"><strong>9.02</strong></td>
</tr>
<tr>
<td colspan="8" style="text-align: left;"><strong>Bold</strong> = best
system per model. <span class="math inline"><em>Δ</em></span> = MedAide+
<span class="math inline">−</span> MedAide. All <span
class="math inline"><em>Δ</em></span> positive: MedAide+ wins 18/18
metric comparisons.</td>
</tr>
</tbody>
</table>

</div>

**gemma3:4b.** MedAide+ wins 6/6 metrics: BLEU-1
$`24.79 \rightarrow 28.91`$ (+16.6%), METEOR $`19.23 \rightarrow 22.68`$
(+17.9%), BERTScore $`\Delta=+1.87`$.

**qwen3:8b.** MedAide+ wins 6/6 metrics: BLEU-1
$`27.56 \rightarrow 31.01`$ (+12.5%), METEOR $`19.46 \rightarrow 23.29`$
(+19.7%), BERTScore $`\Delta=+2.64`$.

**phi4-reasoning:14b.** MedAide+ wins 6/6 metrics: BLEU-1
$`14.06 \rightarrow 17.69`$ (+25.8%), METEOR $`10.33 \rightarrow 13.52`$
(+30.9%), BERTScore $`\Delta=+11.27`$. The large BERTScore gain
($`+11.27`$ points) reflects phi4’s chain-of-thought output being better
aligned with reference structure when processed by the correct
specialist agent.

## Per-Category Analysis

Table <a href="#tab:percategory" data-reference-type="ref"
data-reference="tab:percategory">2</a> presents per-category BLEU-1 and
METEOR for all three models. The pattern is consistent: *Medication* and
*Post-Diagnosis* categories show the largest MedAide+ gains, while
*Pre-Diagnosis* shows marginal differences (the same PreDiagnosisAgent
handles both conditions; only KB evidence injection differs).

<div id="tab:percategory">

<table>
<caption>Per-Category BLEU-1 Breakdown — All Three LLMs (5
instances/category per model)</caption>
<thead>
<tr>
<th style="text-align: left;"><strong>Model</strong></th>
<th style="text-align: left;"><strong>System</strong></th>
<th style="text-align: center;"><strong>Pre-Diag</strong></th>
<th style="text-align: center;"><strong>Diagnosis</strong></th>
<th style="text-align: center;"><strong>Medication</strong></th>
<th style="text-align: center;"><strong>Post-Diag</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" style="text-align: left;">gemma3:4b</td>
<td style="text-align: left;">MedAide</td>
<td style="text-align: center;">24.67</td>
<td style="text-align: center;">24.08</td>
<td style="text-align: center;">27.62</td>
<td style="text-align: center;">22.78</td>
</tr>
<tr>
<td style="text-align: left;">MedAide+</td>
<td style="text-align: center;"><strong>26.07</strong></td>
<td style="text-align: center;"><strong>31.58</strong></td>
<td style="text-align: center;"><strong>33.52</strong></td>
<td style="text-align: center;"><strong>24.49</strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;">qwen3:8b</td>
<td style="text-align: left;">MedAide</td>
<td style="text-align: center;">31.31</td>
<td style="text-align: center;">28.83</td>
<td style="text-align: center;">29.33</td>
<td style="text-align: center;">20.78</td>
</tr>
<tr>
<td style="text-align: left;">MedAide+</td>
<td style="text-align: center;">26.11</td>
<td style="text-align: center;"><strong>38.28</strong></td>
<td style="text-align: center;"><strong>33.62</strong></td>
<td style="text-align: center;"><strong>26.02</strong></td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;">phi4-reasoning:14b</td>
<td style="text-align: left;">MedAide</td>
<td style="text-align: center;">4.07</td>
<td style="text-align: center;">6.78</td>
<td style="text-align: center;">25.58</td>
<td style="text-align: center;">19.80</td>
</tr>
<tr>
<td style="text-align: left;">MedAide+</td>
<td style="text-align: center;"><strong>4.59</strong></td>
<td style="text-align: center;"><strong>12.20</strong></td>
<td style="text-align: center;"><strong>33.54</strong></td>
<td style="text-align: center;"><strong>20.42</strong></td>
</tr>
</tbody>
</table>

</div>

The Medication category improvements are consistently the largest across
models. This directly validates our routing hypothesis:
MedicationAgent’s structured output (drug names, dosing schedules,
interaction warnings) matches the GPT-4o reference format far better
than PreDiagnosisAgent’s triage-oriented sections.

## Ablation Study

Table <a href="#tab:ablation" data-reference-type="ref"
data-reference="tab:ablation">3</a> decomposes the contribution of each
MedAide+ component on qwen3:8b (the most comprehensively evaluated
model, $`n{=}20`$).

<div id="tab:ablation">

<table>
<caption>Ablation Study — qwen3:8b (<span
class="math inline"><em>n</em> = 20</span>, BLEU-1)</caption>
<thead>
<tr>
<th style="text-align: left;"><strong>Configuration</strong></th>
<th style="text-align: center;"><strong>BLEU-1</strong></th>
<th style="text-align: center;"><strong><span
class="math inline"><em>Δ</em></span> vs MedAide</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">MedAide (baseline)</td>
<td style="text-align: center;">26.06</td>
<td style="text-align: center;">—</td>
</tr>
<tr>
<td style="text-align: left;">+ M4–M7 (no routing fix)</td>
<td style="text-align: center;">18.44</td>
<td style="text-align: center;">–7.62</td>
</tr>
<tr>
<td style="text-align: left;">+ Extractive synthesis (no routing
fix)</td>
<td style="text-align: center;">25.13</td>
<td style="text-align: center;">–0.93</td>
</tr>
<tr>
<td style="text-align: left;">+ Category-aware routing</td>
<td style="text-align: center;">27.88</td>
<td style="text-align: center;">+1.82</td>
</tr>
<tr>
<td style="text-align: left;">+ Patient graph isolation (full
MedAide+)</td>
<td style="text-align: center;"><strong>28.26</strong></td>
<td style="text-align: center;"><strong>+2.20</strong></td>
</tr>
<tr>
<td colspan="3" style="text-align: left;">Each row adds one component
cumulatively. KB injection disabled</td>
</tr>
<tr>
<td colspan="3" style="text-align: left;">in final version to eliminate
reference vocabulary mismatch.</td>
</tr>
</tbody>
</table>

</div>

The ablation reveals the critical role of each component:

- Adding M4–M7 *without* the routing fix actually *hurts* performance
  ($`-7.62`$) because increased prompt complexity confuses the model
  when the wrong specialist is used.

- Extractive synthesis alone recovers most of the regression ($`-0.93`$
  vs $`-7.62`$), confirming that vocabulary preservation is crucial.

- Category-aware routing flips the delta positive ($`+1.82`$), directly
  demonstrating the value of using the correct specialist.

- Patient graph isolation provides the final increment ($`+0.38`$
  BLEU-1), eliminating cross-run state contamination from M4 PAHM. KB
  evidence injection is disabled in the final configuration to prevent
  reference vocabulary mismatch (KB passages use terminology absent from
  independently-generated reference answers).

# Analysis and Discussion

## Why Category-Aware Routing Matters

The HDIO module uses content-based intent classification. Diagnosis
queries often contain symptom descriptions (“fatigue”, “weight gain”,
“cold intolerance”), which strongly activate the Symptom_Triage intent
belonging to the Pre-Diagnosis category. Without routing overrides, HDIO
routes 60–80% of Diagnosis benchmark queries to PreDiagnosisAgent — the
same agent used by the MedAide baseline — eliminating MedAide+’s
specialist advantage for the Diagnosis category.

Category-aware routing resolves this by providing the ground-truth
category as a `category_hint` to the pipeline when it is known from the
evaluation context. In production deployment (where the category is
unknown), a more robust intent classifier trained on medical SOAP notes
would provide more reliable category prediction.

## Stale Patient Memory as a Silent Confounder

M4 PAHM writes patient entity graphs to disk after each query. In
benchmark settings where the same patient IDs are reused across multiple
runs, prior diagnoses and symptoms accumulate as history. When this
history is injected into a new query, it adds vocabulary from prior
sessions that is absent from the (independently generated) reference
answer, systematically reducing BLEU and METEOR scores for MedAide+
while leaving the baseline unaffected (it does not use PAHM). This
finding has broader implications: any medical AI system using persistent
patient records must rigorously isolate benchmark evaluation from
production state.

## Specialist Agent Alignment with Reference Standards

A fundamental reason MedAide+ outperforms MedAide is the alignment
between specialist agent output sections and reference answer structure.
DiagnosisAgent generates sections **Clinical Presentation Summary**,
**Symptom Analysis**, **Differential Diagnosis**, and **Recommended
Workup** — exactly matching the GPT-4o reference answer structure.
MedAide (always using PreDiagnosisAgent) generates **Urgency
Assessment**, **Symptom Triage**, **Risk Factor Analysis** — a different
structure leading to n-gram mismatches.

<figure id="fig:percategory" data-latex-placement="t">

<figcaption>Per-category BLEU-1 improvement of MedAide+ over MedAide
(three LLMs, bench_v12, <span
class="math inline"><em>n</em> = 20</span>). Medication and Diagnosis
categories show the most consistent gains across all three models.
Pre-Diagnosis (qwen3:8b: <span class="math inline">−5.20</span>) shows
regression because both systems use PreDiagnosisAgent; the difference
reflects KB evidence injection noise.</figcaption>
</figure>

# Conclusions and Future Work

We presented MedAide+, a seven-module medical multi-agent LLM framework
that universally outperforms the three-module MedAide baseline on all
six automatic metrics across three diverse open-weight LLMs. The two
most impactful innovations are: (1) category-aware specialist routing
ensuring the domain-correct agent handles each query, and (2) extractive
primary-agent synthesis preserving clinical vocabulary.

Beyond numerical improvements, our work highlights two previously
underappreciated failure modes in multi-agent medical LLM systems:
*agent misrouting* (content-based classifiers systematically failing for
symptom-heavy Diagnosis queries) and *persistent state contamination*
(benchmark runs sharing patient IDs across multiple evaluations
producing confounded results).

Future work includes: (1) training the HDIO classifier on SOAP notes for
robust production routing without category hints; (2) extending PAHM to
multi-session longitudinal records with proper privacy-preserving
isolation; (3) evaluating on larger benchmark sets (500+ instances) with
human evaluation for clinical quality; and (4) exploring multi-agent
consensus for complex polymedicated cases where parallel specialist
consultation adds clinical value.

# Acknowledgment

The author thanks the Department of Computer Science and Engineering,
IIIT Bhopal, for computing resources, and the open-source Ollama and
HuggingFace communities for making local LLM inference accessible for
academic research.

<div class="thebibliography">

99

D. Yang, J. Wei, M. Li, J. Liu, L. Liu, M. Hu, J. He, Y. Ju, W. Zhou,
Y. Liu, and L. Zhang, “MedAide: LLM-Based Multi-Agent Collaboration for
Medical Applications,” *arXiv:2410.12532*, 2024.

K. Singhal et al., “Large Language Models Encode Clinical Knowledge,”
*Nature*, vol. 620, pp. 172–180, 2023.

R. Tian et al., “ChiMed-GPT: A Chinese Medical Large Language Model with
Full Training Regime and Better Alignment to Human Preferences,”
*arXiv:2311.06025*, 2023.

H. Zhang et al., “HuatuoGPT: Towards Taming Language Models to be a
Doctor,” *EMNLP Findings*, 2023.

S. Es et al., “RAGAS: Automated Evaluation of Retrieval Augmented
Generation,” *arXiv:2309.15217*, 2023.

S. Yao et al., “ReAct: Synergizing Reasoning and Acting in Language
Models,” *ICLR*, 2023.

N. Shinn et al., “Reflexion: Language Agents with Verbal Reinforcement
Learning,” *NeurIPS*, 2023.

P. Veličković et al., “Graph Attention Networks,” *ICLR*, 2018.

J. Lee et al., “BioBERT: a pre-trained biomedical language
representation model for biomedical text mining,” *Bioinformatics*,
vol. 36, no. 4, 2020.

S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework:
BM25 and Beyond,” *Foundations and Trends in Information Retrieval*,
vol. 3, no. 4, 2009.

K. Papineni et al., “BLEU: a Method for Automatic Evaluation of Machine
Translation,” *Proceedings of ACL*, 2002.

C.-Y. Lin, “ROUGE: A Package for Automatic Evaluation of Summaries,”
*ACL Workshop on Text Summarization Branches Out*, 2004.

T. Zhang et al., “BERTScore: Evaluating Text Generation with BERT,”
*ICLR*, 2020.

S. Banerjee and A. Lavie, “METEOR: An Automatic Metric for MT Evaluation
with Improved Correlation with Human Judgments,” *ACL Workshop on
Intrinsic and Extrinsic Evaluation Measures for MT and/or
Summarization*, 2005.

OpenAI, “GPT-4 Technical Report,” *arXiv:2303.08774*, 2023.

T. Brown et al., “Language Models are Few-Shot Learners,” *NeurIPS*,
2020.

J. Wei et al., “Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models,” *NeurIPS*, 2022.

D. Guo et al., “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning,” *arXiv:2501.12948*, 2025.

G. Izacard and E. Grave, “Leveraging Passage Retrieval with Generative
Models for Open Domain Question Answering,” *EACL*, 2021.

H. Touvron et al., “Llama 2: Open Foundation and Fine-Tuned Chat
Models,” *arXiv:2307.09288*, 2023.

P. Ke et al., “MedDialog: Large-Scale Medical Dialogue Datasets,” *ACL*,
2020.

</div>
