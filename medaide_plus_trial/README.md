# MedAide+ 🏥

**An Improved LLM-based Medical Multi-Agent Framework**

> Extension of the original MedAide system (Fudan University, arXiv 2410.12532) with 7 novel enhancement modules for superior clinical AI assistance.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [7 Enhancement Modules](#7-enhancement-modules)
- [Agents](#agents)
- [Intent Ontology](#intent-ontology)
- [Benchmark Results](#benchmark-results)
- [Dataset & Benchmark](#dataset--benchmark)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Individual Module APIs](#individual-module-apis)
- [Running Tests](#running-tests)
- [Evaluation](#evaluation)
- [Gradio Demo](#gradio-demo)
- [Project Structure](#project-structure)
- [Phase Plan](#phase-plan)
- [Citation](#citation)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MedAide+ Pipeline                              │
│                                                                     │
│  User Query ──► [M4 PLMM] ──► [M1 AMQU] ──► [M2 HDIO] ──► [M7 MIET] │
│                  Patient       Query       Intent       Dialogue    │
│                  Memory        Decomp      Classify     State       │
│                                    │                               │
│                              [M6 AQCR]                             │
│                           Complexity Routing                        │
│                          ┌─────┴──────┐                            │
│                      Simple(1)  Moderate(2)  Complex(4)            │
│                              │                                      │
│              ┌───────────────┼───────────────┐                     │
│              ▼               ▼               ▼                     │
│         [PreDx]         [Diagnosis]     [Medication]               │
│          Agent            Agent           Agent                     │
│                         [PostDx Agent]                             │
│              └───────────── [M3 DMACN] ──────────────┘             │
│                         Parallel Execution                          │
│                         Critic + Synthesis                          │
│                               │                                     │
│                          [M5 HDFG]                                  │
│                     Hallucination Verification                      │
│                               │                                     │
│                    Verified Response + Metadata                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Pipeline Step-by-Step

| Step | Module | Action |
|------|--------|--------|
| 0 | M4 PLMM | Inject patient history from knowledge graph into query |
| 1 | M1 AMQU | Decompose query into structured subqueries (Flan-T5 + BM25) |
| 2 | M2 HDIO | Classify intents using BioBERT + GAT (18-intent taxonomy) |
| 3 | M7 MIET | Update dialogue state, inject session bias into intent scores |
| 4 | M6 AQCR | Route by complexity → 1 / 2 / 4 agents |
| 5 | M3 DMACN | Run agents in parallel using asyncio |
| 6 | Critic + Synthesis | Evaluate and synthesize agent outputs |
| 7 | M5 HDFG | Verify response for hallucinations (FAISS + MC Dropout) |
| 8 | M4 PLMM | Update patient knowledge graph from response |
| 9 | M7 MIET | Update session buffer with response summary |

---

## 7 Enhancement Modules

| Module | Name | Key Technique | Paper Ref |
|--------|------|---------------|-----------|
| **M1 AMQU** | Adaptive Multi-shot Query Understanding | Flan-T5-base decomposition + K=5 consistency filtering + recency-weighted BM25 (λ=0.1) | §3.1 |
| **M2 HDIO** | Hierarchical Dual-level Intent Ontology | BioBERT encoder + Graph Attention Network (GAT, 4 heads) + multi-label sigmoid + OOD detection | §3.2 |
| **M3 DMACN** | Dynamic Multi-Agent Critic Network | asyncio parallel execution + CriticAgent severity scoring + SynthesisAgent with confidence weighting | §3.3 |
| **M4 PLMM** | Persistent Longitudinal Medical Memory | NetworkX patient knowledge graphs + spaCy NER + JSON persistence per patient | §3.4 |
| **M5 HDFG** | Hallucination-Aware Dual-Verification | FAISS IndexFlatIP semantic KB search + Monte Carlo Dropout (N=15) uncertainty estimation | §3.5 |
| **M6 AQCR** | Adaptive Query Complexity Routing | RoBERTa-base classifier → Simple(1) / Moderate(2) / Complex(4) agents + feature-based fallback | §3.6 |
| **M7 MIET** | Multi-Turn Intent Evolution Tracking | Exponential smoothing (β=0.6) dialogue state + summary buffer (size=5) | §3.7 |

### Module Design Details

#### M1 AMQU — Adaptive Multi-shot Query Understanding
- **Primary**: Flan-T5-base generates K=5 diverse decompositions via sampling (temperature=0.7)
- **Consistency filter**: Cosine similarity clustering; retains subqueries appearing in ≥ min_count clusters (threshold=0.85)
- **Recency BM25**: `score(d,q) = BM25(d,q) × exp(-λ(T - t_d))` with λ=0.1
- **Fallback**: Rule-based conjunction/sentence splitting when Flan-T5 unavailable

#### M2 HDIO — Hierarchical Dual-level Intent Ontology
- **Encoder**: BioBERT (dmis-lab/biobert-base-cased-v1.2)
- **Classifier**: 4-head GAT with 128 hidden units over intent hierarchy graph
- **Output**: Per-intent sigmoid scores (multi-label, not softmax) for all 18 intents
- **OOD**: Rejects queries below 0.30 max-score threshold
- **Fallback**: TF-IDF + cosine similarity when BioBERT unavailable

#### M3 DMACN — Dynamic Multi-Agent Critic Network
- **Parallelism**: asyncio task pool runs 1–4 agents concurrently
- **CriticAgent**: Scores each response (coherence, completeness, safety) + assigns severity
- **SynthesisAgent**: Merges agent outputs weighted by confidence; produces final response

#### M4 PLMM — Persistent Longitudinal Medical Memory
- **Storage**: Per-patient JSON files in `data/patient_graphs/`
- **Graph**: NetworkX `DiGraph` with typed nodes (Symptom, Diagnosis, Medication, Allergy, Procedure)
- **NER**: spaCy `en_core_web_sm` (falls back to regex patterns)
- **Injection**: Prepends `[PATIENT HISTORY]` context block to enriched query

#### M5 HDFG — Hallucination-Aware Dual-Verification
- **Stage 1**: FAISS IndexFlatIP retrieves top-5 KB passages per claim
- **Stage 2**: Monte Carlo Dropout (N=15 passes, noise σ=0.1) estimates uncertainty
- **Output**: Inline `[SUPPORTED]` / `[FLAGGED]` annotations + hallucination rate
- **Claim extraction**: NER-based sentence-level claim segmentation

#### M6 AQCR — Adaptive Query Complexity Routing
- **Primary**: RoBERTa-base 3-class classifier (Simple / Moderate / Complex)
- **Features**: n_intents, query_length (chars), inter-intent cosine distance
- **Fallback**: Feature-based linear scoring when RoBERTa unavailable
- **Tier → agents**: Simple=1, Moderate=2, Complex=4

#### M7 MIET — Multi-Turn Intent Evolution Tracking
- **State**: 18-dim intent state vector updated via exponential smoothing: `s_t = β·s_{t-1} + (1-β)·i_t`
- **Buffer**: Ring buffer of last 5 dialogue turns (query + response summary)
- **Bias injection**: Adds session state to current intent scores for context-aware routing

---

## Agents

| Agent | Role | Triggers on |
|-------|------|-------------|
| `PreDiagnosisAgent` | Symptom triage, risk assessment, department suggestion | Pre-Diagnosis queries |
| `DiagnosisAgent` | Differential diagnosis, symptom analysis, test interpretation | Diagnosis queries |
| `MedicationAgent` | Drug interactions, dosage, contraindications, prescription review | Medication queries |
| `PostDiagnosisAgent` | Rehabilitation, lifestyle guidance, follow-up scheduling | Post-Diagnosis queries |
| `CriticAgent` | Evaluates all agent outputs; assigns severity scores | After M3 parallel run |
| `SynthesisAgent` | Merges outputs into final response with confidence weighting | After CriticAgent |

All agents extend `BaseAgent` and use the `openai` SDK with `gpt-4o` (configurable).

---

## Intent Ontology

MedAide+ extends the original MedAide's 17-intent taxonomy with **Follow_up_Scheduling** (18 total):

| Category | ID | Intents |
|----------|----|---------|
| **Pre-Diagnosis** | 0 | Symptom_Triage, Department_Suggestion, Risk_Assessment, Health_Inquiry |
| **Diagnosis** | 1 | Symptom_Analysis, Etiology_Detection, Test_Interpretation, Differential_Diagnosis |
| **Medication** | 2 | Drug_Counseling, Dosage_Recommendation, Contraindication_Check, Drug_Interaction, Prescription_Review |
| **Post-Diagnosis** | 3 | Rehabilitation_Advice, Progress_Tracking, Care_Support, Lifestyle_Guidance, **Follow_up_Scheduling** *(new)* |

Defined in `config/intent_ontology.yaml` with per-intent keyword lists and descriptions.

---

## Benchmark Results

### Local Ollama Benchmark (Final — 10 instances/category, clean cache)

Evaluated with local Ollama models on RTX 3070 Laptop (8GB VRAM). 3 conditions per instance:
- **Vanilla**: Mock/no-LLM baseline
- **MedAide**: Simulated original (M1+M2+M3, 1 agent)
- **MedAide+**: Full pipeline (M1–M7, multi-agent routing via M6 AQCR)

#### qwen3:8b — MedAide+ wins 5/6 metrics ✅

| Metric | MedAide | MedAide+ | Delta | Result |
|--------|---------|----------|-------|--------|
| **BLEU-1** | 24.23 | **26.98** | **+2.74** | ✅ WIN |
| **BLEU-2** | 10.49 | **11.33** | **+0.84** | ✅ WIN |
| **METEOR** | 17.56 | **18.68** | **+1.12** | ✅ WIN |
| ROUGE-L | **18.38** | 17.49 | -0.89 | ❌ |
| **BERTScore** | 65.86 | **66.34** | **+0.48** | ✅ WIN |
| **GLEU** | 8.33 | **8.64** | **+0.31** | ✅ WIN |

**Per-category highlights (qwen3:8b):**
| Category | BLEU-1 Δ | METEOR Δ | BERTScore Δ |
|----------|----------|----------|-------------|
| Pre-Diagnosis | +1.71 | +1.28 | +2.28 |
| Diagnosis | +1.70 | +1.17 | -0.26 |
| **Medication** | **+6.88** | **+2.38** | +0.38 |
| Post-Diagnosis | +0.69 | -0.35 | -0.47 |

#### gemma3:4b — MedAide+ loses all metrics (small model limitation)

| Metric | MedAide | MedAide+ | Delta |
|--------|---------|----------|-------|
| BLEU-1 | **25.30** | 24.75 | -0.55 |
| BLEU-2 | **10.57** | 10.06 | -0.51 |
| METEOR | **18.66** | 18.24 | -0.42 |
| ROUGE-L | **16.51** | 15.16 | -1.34 |
| BERTScore | **64.42** | 64.13 | -0.29 |
| GLEU | **7.89** | 7.65 | -0.24 |

#### Phase 4 (v2 — Category-Aware Routing): qwen3:8b wins 7/8 metrics ✅

After the Phase 4 fixes (category-aware routing, selective KB injection, synthesis hints):

| Metric | MedAide | MedAide+ v2 | Delta | Result |
|--------|---------|-------------|-------|--------|
| **BLEU-1** | 21.77 | **23.34** | **+1.57** | ✅ WIN |
| **BLEU-2** | 9.05 | **9.62** | **+0.57** | ✅ WIN |
| **ROUGE-1** | 35.84 | **37.15** | **+1.31** | ✅ WIN |
| **ROUGE-2** | 8.79 | **9.12** | **+0.33** | ✅ WIN |
| ROUGE-L | **15.79** | 15.42 | -0.37 | ❌ (-58% narrowed) |
| **GLEU** | 7.20 | **7.44** | **+0.24** | ✅ WIN |
| **METEOR** | 14.84 | **16.18** | **+1.34** | ✅ WIN |
| **BERTScore** | 62.04 | **62.89** | **+0.85** | ✅ WIN |

**Medication category BLEU-1: +26.6% (22.39 → 28.34)** — direct proof the routing fix works.

#### Key Findings

1. **MedAide+ outperforms MedAide with capable models (8B+ params)**: qwen3:8b wins 7/8 metrics after Phase 4 fixes, with Medication BLEU-1 +26.6%
2. **Critical routing bug fixed**: Old `agents[:n]` always selected [PreDiag, Diag], never Medication or PostDiag agents. `CATEGORY_AGENT_ORDER` dict now routes specialist first.
3. **Smaller models (4B) are hurt by multi-agent complexity**: gemma3:4b loses all metrics — model capability threshold ≥8B params required
4. **ROUGE-L gap significantly narrowed**: From -0.89 (Phase 3) to -0.37 (Phase 4) — 58% improvement from synthesis hints + primary agent anchoring
5. **Selective KB injection matters**: BM25 threshold 0.3 prevents noise injection from low-relevance KB passages

#### Improvement Progression

| Stage | BLEU-1 (qwen3:8b) | Notes |
|-------|-------------------|-------|
| Baseline (before fixes) | MedAide=27.19, MedAide+=18.44 | -8.75 gap |
| After synthesis fix v1 | MedAide=26.24, MedAide+=18.79 | Prompt too simplified |
| After extractive merge | MedAide=26.28, MedAide+=28.63 | **First WIN (+2.35)** |
| Final (cache clear) | MedAide=24.23, MedAide+=26.98 | **Stable WIN (+2.74)** |

---

## Dataset & Benchmark

> **Background**: The original MedAide benchmark (arXiv:2410.12532v3, Section 4.1) is **private and not publicly released**. MedAide+ includes a comprehensive self-generated benchmark at **1.5× the paper's scale**.

### V2 Benchmark Dataset (Primary — 3,000 instances) ✅

The project ships with `data/benchmark/medaide_plus_benchmark_v2.json` — a **3,000-instance** composite-intent evaluation dataset generated by `evaluation/generate_v2_benchmark.py`.

| Property | Value |
|----------|-------|
| **Total Instances** | 3,000 (750 per category) |
| **Scale** | 1.5× original paper (paper = 2,000) |
| **Categories** | Pre-Diagnosis, Diagnosis, Medication, Post-Diagnosis |
| **Intent Taxonomy** | 18 intents (17 original + Follow_up_Scheduling) |
| **Intents per Query** | 2–3 composite intents |
| **Difficulty Levels** | Easy (480), Moderate (1,499), Hard (1,021) |
| **Templates** | 30+ per category (vs 8 in v1) |
| **Query Uniqueness** | 98.1% |
| **Clinical Context** | ✅ Annotated per instance |
| **Reference Answers** | ✅ Intent-specific multi-paragraph |

```bash
# Regenerate the v2 benchmark (default: 750/category = 3,000 total)
python -m evaluation.generate_v2_benchmark

# Custom size (e.g., 1000/category = 4,000 total)
python -m evaluation.generate_v2_benchmark --samples 1000

# Quick test with 20/category
python -m evaluation.generate_v2_benchmark --samples 20
```

### V1 Benchmark Generator (Legacy — paper-aligned)

The older `evaluation/fetch_benchmark.py` generates up to 500/category (2,000 total) with 8 templates per category:

```bash
python -m evaluation.fetch_benchmark --samples 500
python -m evaluation.fetch_benchmark --samples 500 --with-gpt-ref  # GPT-4o answers
```

### Benchmark Runner

The benchmark runner automatically prefers the v2 dataset when available:

```bash
# Run 3-condition comparison (Vanilla / MedAide / MedAide+)
python -m evaluation.benchmark_runner

# Quick test with 20 instances per category
python -m evaluation.benchmark_runner --limit 20

# Specify benchmark file explicitly
python -m evaluation.benchmark_runner --benchmark data/benchmark/medaide_plus_benchmark_v2.json
```

### Public Medical QA Datasets (Supplementary)

For fine-tuning (Phase 7) and cross-dataset evaluation:

| Dataset | Size | Type | Use Case | Access |
|---------|------|------|----------|--------|
| **MedQA (USMLE)** | 12,723 | MCQ (English/Chinese) | Benchmark eval | [HuggingFace](https://huggingface.co/datasets/bigbio/med_qa) |
| **MedMCQA** | 194,000 | MCQ (Indian medical exams) | Fine-tuning, eval | [HuggingFace](https://huggingface.co/datasets/medmcqa) |
| **PubMedQA** | 211,000 | Abstract-based QA | Knowledge base | [HuggingFace](https://huggingface.co/datasets/qiaojin/PubMedQA) |
| **MedDialog** | 3.4M turns | Patient-doctor dialogues | Fine-tuning M7 (MIET) | [GitHub](https://github.com/UCSD-AI4H/Medical-Dialogue-System) |
| **HealthCareMagic-100k** | 100,000 | Real patient consultations | Fine-tuning agents | [HuggingFace](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) |
| **iCliniq** | 10,000 | Real patient Q&A | Fine-tuning agents | [HuggingFace](https://huggingface.co/datasets/lavita/icliniq-10k) |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd medaide_plus

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for better NER in M4 PLMM)
python -m spacy download en_core_web_sm

# Set OpenAI API key
set OPENAI_API_KEY=your-key-here  # Windows
export OPENAI_API_KEY=your-key-here  # Linux/Mac
```

### Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | ≥4.40.0 | Flan-T5, BioBERT, RoBERTa models |
| `torch` | ≥2.0.0 | PyTorch backend |
| `sentence-transformers` | ≥2.7.0 | Sentence embeddings (M1, M5) |
| `faiss-cpu` | ≥1.7.4 | Vector similarity search (M5) |
| `rank-bm25` | ≥0.2.2 | BM25 retrieval (M1) |
| `networkx` | ≥3.3 | Patient knowledge graphs (M4) |
| `spacy` | ≥3.7.0 | Named entity recognition (M4) |
| `openai` | ≥1.30.0 | GPT-4o agent calls |
| `gradio` | ≥4.37.0 | Web demo interface |
| `nltk`, `rouge-score`, `bert-score` | latest | Evaluation metrics |

---

## Configuration

All parameters are in `config/config.yaml`. Key settings:

### LLM Provider Configuration (Multi-Provider)

MedAide+ supports 7 LLM backends. Set `llm.provider` to switch:

```yaml
llm:
  provider: "ollama"               # Active provider: openai | ollama | anthropic | 
                                   #   azure_openai | google_gemini | huggingface | openai_compatible
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4o"
  
  ollama:
    base_url: "http://localhost:11434"
    model: "llama3"                # Also: mistral, codellama, phi3, medllama2, meditron
  
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-sonnet-20240229"
  
  azure_openai:
    api_key: "${AZURE_OPENAI_API_KEY}"
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    deployment: "gpt-4o"
    api_version: "2024-02-01"
  
  google_gemini:
    api_key: "${GOOGLE_AI_API_KEY}"
    model: "gemini-1.5-pro"
  
  huggingface:
    api_key: "${HF_API_KEY}"
    model: "mistralai/Mixtral-8x7B-Instruct-v0.1"
  
  openai_compatible:
    api_key: "${CUSTOM_API_KEY}"
    base_url: "http://localhost:8080/v1"
    model: "local-model"
```

### Module Configuration

```yaml
models:
  flan_t5: "google/flan-t5-base"        # M1 query decomposition
  biobert: "dmis-lab/biobert-base-cased-v1.2"  # M2 intent encoding
  roberta: "roberta-base"               # M6 complexity routing
  openai_model: "gpt-4o"               # All agents

modules:
  amqu:
    k_shots: 5                          # Decomposition candidates
    consistency_threshold: 0.85        # Cosine similarity filter
  hdio:
    ood_threshold: 0.30                 # OOD rejection threshold
    num_intents: 18                     # MedAide+ extended taxonomy
  plmm:
    max_nodes: 500                      # Max nodes per patient graph
  hdfg:
    mc_dropout_passes: 15              # Monte Carlo uncertainty passes
    uncertainty_threshold: 0.3        # Hallucination flag threshold
  aqcr:
    simple_agents: 1
    moderate_agents: 2
    complex_agents: 4
  miet:
    beta: 0.6                          # Exponential smoothing factor
    buffer_size: 5                     # Dialogue history turns
```

---

## Quick Start

### Python API

```python
from medaide_plus.pipeline import MedAidePlusPipeline

# Initialize pipeline
pipeline = MedAidePlusPipeline(
    config_path="config/config.yaml",
    patient_id="patient_001",
)

# Synchronous usage
result = pipeline.run_sync(
    query="I have chest pain and shortness of breath. What should I do?",
    patient_id="patient_001",
)

print(f"Response: {result.final_response}")
print(f"Annotated: {result.annotated_response}")   # With [SUPPORTED]/[FLAGGED] markers
print(f"Tier: {result.tier} ({result.n_agents_used} agents)")
print(f"Intents: {result.intents}")
print(f"Category: {result.top_category}")
print(f"Subqueries: {result.subqueries}")
print(f"Hallucination Rate: {result.hallucination_rate:.1%}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Latency: {result.latency_ms:.0f}ms")
```

### Async Usage

```python
import asyncio
from medaide_plus.pipeline import MedAidePlusPipeline

async def main():
    pipeline = MedAidePlusPipeline(config_path="config/config.yaml")
    result = await pipeline.run(
        query="What is the correct metformin dosage for elderly patients?",
        patient_id="patient_002",
        session_id="session_001",
    )
    print(result.final_response)

asyncio.run(main())
```

### PipelineResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `query` | str | Original user query |
| `patient_id` | str | Patient identifier |
| `final_response` | str | Synthesized medical response |
| `annotated_response` | str | Response with `[SUPPORTED]`/`[FLAGGED]` markers |
| `confidence` | float | Overall response confidence [0–1] |
| `tier` | str | Routing tier: Simple / Moderate / Complex |
| `n_agents_used` | int | Number of agents invoked (1, 2, or 4) |
| `intents` | List[str] | Top-3 classified intents |
| `top_category` | str | Dominant category |
| `subqueries` | List[str] | M1 decomposition output |
| `hallucination_rate` | float | Fraction of flagged claims [0–1] |
| `verified_claims` | List[Dict] | Per-claim verification results (top 5) |
| `agent_names` | List[str] | Names of agents that ran |
| `latency_ms` | float | End-to-end latency in milliseconds |
| `metadata` | Dict | Per-module metadata (amqu, hdio, aqcr, hdfg, critic) |

---

## Individual Module APIs

```python
# M1 AMQU — Query decomposition
from medaide_plus.modules.m1_amqu import AMQUModule
amqu = AMQUModule()
result = amqu.run("I have chest pain and diabetes. What medication should I take?")
print([sq.text for sq in result.subqueries])         # Decomposed subqueries
print(result.processing_metadata["model_used"])       # "flan-t5" or "rule_based"

# M2 HDIO — Intent classification
from medaide_plus.modules.m2_hdio import HDIOModule
hdio = HDIOModule()
result = hdio.classify("What is the dosage of ibuprofen?")
print(f"Category: {result.top_category}")
print(f"Intents: {result.top_intents}")
print(f"OOD: {result.is_ood}")
print(f"Scores: {result.intent_scores}")             # All 18 intent scores

# M4 PLMM — Patient memory
from medaide_plus.modules.m4_plmm import PLMMModule
plmm = PLMMModule()
plmm.update_from_response("Patient takes metformin 1000mg for diabetes", "patient_001")
result = plmm.inject_history("What drug interactions should I know about?", "patient_001")
print(result.enriched_query)                         # Query + [PATIENT HISTORY] block

# M5 HDFG — Hallucination verification
from medaide_plus.modules.m5_hdfg import HdfgModule
hdfg = HdfgModule()
result = hdfg.annotate_response("Metformin is prescribed for type 2 diabetes", "diabetes query")
print(result.annotated_response)                     # With [SUPPORTED]/[FLAGGED]
print(f"Hallucination rate: {result.hallucination_rate:.1%}")

# M6 AQCR — Complexity routing
from medaide_plus.modules.m6_aqcr import AqcrModule
aqcr = AqcrModule()
result = aqcr.route("What is aspirin?")
print(f"Tier: {result.tier}, Agents: {result.n_agents}")

# M7 MIET — Dialogue state tracking
from medaide_plus.modules.m7_miet import MietModule
miet = MietModule()
miet.update(session_id="s1", intent_scores={"Drug_Interaction": 0.9}, query="drug query")
context = miet.get_context_prefix("s1")
print(context)                                       # Session history context string
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_m1_amqu.py -v
pytest tests/test_m2_hdio.py -v
pytest tests/test_m3_dmacn.py -v --asyncio-mode=auto
pytest tests/test_m4_plmm.py -v
pytest tests/test_m5_hdfg.py -v
pytest tests/test_m6_aqcr.py -v
pytest tests/test_m7_miet.py -v

# Run integration tests
pytest tests/test_pipeline.py -v --asyncio-mode=auto

# Run with coverage
pytest tests/ -v --cov=medaide_plus --cov-report=html
```

---

## Evaluation

### Running the Benchmark

```bash
# Quick test (8 instances, all 3 conditions)
python -m evaluation.benchmark_runner --limit 8 --condition all

# Full benchmark (requires generated dataset, all conditions)
python -m evaluation.benchmark_runner --condition all

# Only MedAide+ full condition
python -m evaluation.benchmark_runner --condition full

# With OpenAI API key for live GPT-4o evaluation
OPENAI_API_KEY=sk-... python -m evaluation.benchmark_runner --condition all
```

### Generating the Dataset

```bash
# Generate synthetic composite-intent benchmark (500/category = 2000 total)
python -m evaluation.fetch_benchmark --samples 500

# With GPT-4o ground-truth reference answers
python -m evaluation.fetch_benchmark --samples 500 --with-gpt-ref

# Quick test
python -m evaluation.fetch_benchmark --samples 20
```

### Ablation Study

```bash
# Run ablation (tests removing each module M1–M7)
python -m evaluation.ablation_runner
```

### Benchmark Output Files

| File | Description |
|------|-------------|
| `data/benchmark/results_<timestamp>.json` | Per-instance results for all conditions |
| `data/benchmark/summary_<timestamp>.json` | Aggregated per-category per-condition metrics |
| `data/benchmark/summary_latest.json` | Always updated to most recent run |
| `data/benchmark/ablation_latest.json` | Ablation study results |

---

## Gradio Demo

```bash
# Start the web demo
python demo/app.py

# Access at http://localhost:7860
```

The demo provides a conversational interface with:
- Real-time intent classification display
- Complexity tier indicator
- Annotated response with hallucination markers
- Patient session memory across turns

---

## Project Structure

```
medaide_plus/
├── config/
│   ├── config.yaml              # Main configuration (LLM providers, modules, KB, logging)
│   └── intent_ontology.yaml     # 4-category / 18-intent hierarchy with keywords
├── data/
│   ├── benchmark/               # Benchmark instances and results
│   │   ├── medaide_plus_benchmark_v2.json  # V2 benchmark (3,000 instances, 750/cat)
│   │   ├── medaide_benchmark.json          # V1 synthetic proxy (8 instances)
│   │   ├── results_*.json                  # Per-run evaluation results
│   │   ├── summary_*.json                  # Aggregated metrics
│   │   └── summary_latest.json             # Most recent summary
│   ├── patient_graphs/          # Per-patient JSON graphs (M4 PLMM)
│   └── sample_queries.json      # 20 sample queries (5/category) for testing
├── demo/
│   └── app.py                   # Gradio web interface
├── evaluation/
│   ├── benchmark_runner.py      # Runs 3-condition evaluation (Vanilla/MedAide/MedAide+)
│   ├── generate_v2_benchmark.py # V2 benchmark generator (30+ templates/cat, 3,000 instances)
│   ├── fetch_benchmark.py       # V1 synthetic composite-intent benchmark generator
│   ├── ablation_runner.py       # Ablation study (remove each module)
│   └── results_analyzer.py      # Loads and analyzes benchmark results
├── medaide_plus/
│   ├── __init__.py
│   ├── pipeline.py              # Main MedAidePlusPipeline orchestrator
│   ├── agents/
│   │   ├── base_agent.py        # Abstract BaseAgent + AgentOutput + LLM provider support
│   │   ├── pre_diagnosis_agent.py   # Symptom triage, risk, department
│   │   ├── diagnosis_agent.py       # Differential, symptom analysis, test interp.
│   │   ├── medication_agent.py      # Drug interactions, dosage, contraindications
│   │   ├── post_diagnosis_agent.py  # Rehab, lifestyle, follow-up
│   │   ├── critic_agent.py          # Evaluates agent outputs (coherence/safety)
│   │   └── synthesis_agent.py       # Merges outputs into final response
│   ├── modules/
│   │   ├── m1_amqu.py           # Flan-T5 decomp + consistency filter + BM25
│   │   ├── m2_hdio.py           # BioBERT + GAT + 18-intent multi-label classifier
│   │   ├── m3_dmacn.py          # asyncio parallel agent execution
│   │   ├── m4_plmm.py           # NetworkX patient knowledge graphs + NER
│   │   ├── m5_hdfg.py           # FAISS claim verification + MC Dropout
│   │   ├── m6_aqcr.py           # RoBERTa complexity routing (Simple/Moderate/Complex)
│   │   └── m7_miet.py           # Exponential smoothing dialogue state tracker
│   ├── knowledge_base/
│   │   ├── rag.py               # RAGRetriever: FAISS + BM25 hybrid retrieval
│   │   └── kb_manager.py        # Load/save/seed knowledge base
│   └── utils/
│       ├── llm_provider.py      # Multi-provider LLM abstraction (7 backends)
│       ├── logger.py            # Structured logging setup
│       └── metrics.py           # BLEU, ROUGE-L, GLEU, METEOR, BERTScore
├── tests/
│   ├── test_m1_amqu.py          # M1 unit tests (decomposition, BM25, consistency)
│   ├── test_m2_hdio.py          # M2 unit tests (intent classification, OOD)
│   ├── test_m3_dmacn.py         # M3 async tests (parallel execution, critic)
│   ├── test_m4_plmm.py          # M4 unit tests (graph operations, NER, injection)
│   ├── test_m5_hdfg.py          # M5 unit tests (claim extraction, FAISS, MC Dropout)
│   ├── test_m6_aqcr.py          # M6 unit tests (routing decisions, features)
│   ├── test_m7_miet.py          # M7 unit tests (state updates, buffer)
│   ├── test_pipeline.py         # End-to-end integration tests
│   └── test_llm_provider.py     # LLM provider tests (factory, mocking, all 7 backends)
├── requirements.txt             # Python dependencies
└── pytest.ini                   # Pytest configuration
```

---

## Phase Plan

| Phase | Description | Status | Notes |
|-------|-------------|--------|-------|
| Phase 1 | Core module implementation (M1–M7) | ✅ Complete | All 7 modules implemented with fallbacks |
| Phase 2 | Agent implementation and DMACN orchestration | ✅ Complete | 6 agents (PreDx, Dx, Med, PostDx, Critic, Synthesis) |
| Phase 3 | Knowledge base (FAISS + BM25 RAG) | ✅ Complete | Hybrid retrieval with KB manager |
| Phase 4 | Pipeline integration + evaluation framework | ✅ Complete | Benchmark runner, ablation, metrics |
| Phase 5 | Unit + integration tests | ✅ Complete | 8 test files covering all modules |
| Phase 6 | Gradio demo | ✅ Complete | Web UI at localhost:7860 |
| Phase 7 | Fine-tuning on medical datasets | 🔄 Planned | See [Dataset Strategy](#dataset-strategy) |
| Phase 8 | Benchmark evaluation (MedQA, CMB) | 🔄 Planned | Requires expanded dataset |

---

## Citation

If you use MedAide+, please cite the original MedAide paper:

```bibtex
@article{medaide2024,
  title={MedAide: Towards an Omni Medical Aide via Specialized LLM-based Multi-Agent Collaboration},
  author={...},
  journal={arXiv preprint arXiv:2410.12532},
  year={2024}
}
```

---

## License

MIT License — see LICENSE file for details.

> ⚠️ **Medical Disclaimer:** MedAide+ is intended for research and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions.
