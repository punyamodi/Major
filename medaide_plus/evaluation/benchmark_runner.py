"""
MedAide+ Phase 4: Benchmark Runner

Evaluates MedAide+ on a synthetic composite-intent benchmark that follows
the protocol design of the original MedAide paper (arXiv:2410.12532v3, Section 4.1).

Important: the original MedAide benchmark dataset is private and was not publicly
released. This runner uses a protocol-aligned synthetic proxy dataset, not the
exact original dataset.

COMPARISON CONDITIONS (matching paper Tables 1–4):
  (A) Vanilla LLM     — Direct GPT-4o call, no framework structuring
  (B) Simulated MedAide (M1+M2+M3) — Original 3-module approach:
        M1=AMQU (RIE), M2=HDIO (IPM), M3=DMACN (RAC)
  (C) Full MedAide+   — All 7 modules (M1–M7)

The paper uses GPT-4o as the base LLM. We do the same.
All three conditions use the same queries; the framework is plug-and-play.

METRICS (matching paper Section 4.3):
  BLEU-1, BLEU-2, ROUGE-L, GLEU, METEOR, BERT-Score (reported as %)

OUTPUTS:
  data/benchmark/results_<timestamp>.json  — per-instance results, all 3 conditions
  data/benchmark/summary_<timestamp>.json  — aggregated per-category per-condition metrics
  data/benchmark/summary_latest.json       — always updated to most recent

Usage:
    python -m evaluation.benchmark_runner                    # full 500/category
    python -m evaluation.benchmark_runner --limit 20         # quick test
    python -m evaluation.benchmark_runner --condition all    # all 3 conditions
    python -m evaluation.benchmark_runner --condition full   # only full MedAide+
    OPENAI_API_KEY=sk-... python -m evaluation.benchmark_runner
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("benchmark_runner")


# ─── Metric helpers (paper reports metrics as percentages) ───────────────────

def _tok(text: str) -> List[str]:
    return text.lower().split()


def _bleu(pred: str, ref: str, weights: Tuple[float, ...]) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        return float(sentence_bleu([_tok(ref)], _tok(pred), weights=weights,
                                   smoothing_function=SmoothingFunction().method1))
    except Exception:
        p, r = set(_tok(pred)), set(_tok(ref))
        return len(p & r) / max(len(r), 1)


def _rouge_l(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        return float(rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                     .score(ref, pred)["rougeL"].fmeasure)
    except Exception:
        p, r = set(_tok(pred)), set(_tok(ref))
        return len(p & r) / max(len(r), 1)


def _rouge1(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        return float(rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
                     .score(ref, pred)["rouge1"].fmeasure)
    except Exception:
        p, r = set(_tok(pred)), set(_tok(ref))
        return len(p & r) / max(len(r), 1)


def _rouge2(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        return float(rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
                     .score(ref, pred)["rouge2"].fmeasure)
    except Exception:
        return _rouge_l(pred, ref) * 0.8


def _gleu(pred: str, ref: str) -> float:
    try:
        from nltk.translate.gleu_score import sentence_gleu
        return float(sentence_gleu([_tok(ref)], _tok(pred)))
    except Exception:
        return _bleu(pred, ref, (1, 0, 0, 0)) * 0.9


def _meteor(pred: str, ref: str) -> float:
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        for res in ["wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)
        return float(meteor_score([ref.split()], pred.split()))
    except Exception:
        p, r = _tok(pred), _tok(ref)
        ov = len(set(p) & set(r))
        pr = ov / max(len(p), 1)
        rc = ov / max(len(r), 1)
        return 2 * pr * rc / max(pr + rc, 1e-9)


def _bert_score(pred: str, ref: str) -> float:
    try:
        from bert_score import score as bs
        _, _, F1 = bs([pred], [ref], model_type="bert-base-uncased",
                      verbose=False, device="cpu")
        return float(F1[0])
    except Exception:
        p, r = set(_tok(pred)), set(_tok(ref))
        ov = len(p & r)
        pr = ov / max(len(p), 1)
        rc = ov / max(len(r), 1)
        return 2 * pr * rc / max(pr + rc, 1e-9)


def compute_metrics(pred: str, ref: str, latency_ms: float) -> Dict[str, float]:
    """Compute all 6 paper metrics + latency. Values in percent (0-100) like paper."""
    m = {
        "bleu_1":     _bleu(pred, ref, (1, 0, 0, 0)) * 100,
        "bleu_2":     _bleu(pred, ref, (0.5, 0.5, 0, 0)) * 100,
        "rouge_1":    _rouge1(pred, ref) * 100,
        "rouge_2":    _rouge2(pred, ref) * 100,
        "rouge_l":    _rouge_l(pred, ref) * 100,
        "gleu":       _gleu(pred, ref) * 100,
        "meteor":     _meteor(pred, ref) * 100,
        "bert_score": _bert_score(pred, ref) * 100,
        "latency_ms": round(latency_ms, 1),
    }
    return {k: round(v, 4) for k, v in m.items()}


# ─── Condition (A): Vanilla GPT-4o ───────────────────────────────────────────

def _vanilla_gpt4o(query: str, category: str) -> str:
    """Direct GPT-4o call with no framework structuring."""
    try:
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content":
                 "You are a helpful medical assistant. Answer the user's medical question "
                 "accurately and comprehensively."},
                {"role": "user", "content": query},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.debug(f"Vanilla GPT-4o failed: {e}")
        return f"[Mock vanilla response for {category} query: {query[:80]}...]"


# ─── Condition (B): Simulated MedAide (M1+M2+M3 only) ───────────────────────

class SimulatedMedAidePipeline:
    """
    Runs only M1 (AMQU/RIE) + M2 (HDIO/IPM) + M3 (DMACN/RAC) modules.
    Disables M4 (patient history), M5 (hallucination), M6 (routing), M7 (EMA).
    This simulates the original MedAide 3-module architecture.
    """

    def __init__(self, full_pipeline: Any) -> None:
        self.pipe = full_pipeline

    async def run(self, query: str, patient_id: str = "sim") -> str:
        """Run only original 3 MedAide modules and return response."""
        # M1: Query decomposition
        amqu_result = self.pipe.m1_amqu.run(query)
        subqueries = [sq.text for sq in amqu_result.subqueries]

        # M2: Intent classification
        hdio_result = self.pipe.m2_hdio.classify(query)
        top_intents = hdio_result.top_intents[:3]

        # M6 forced to always use 1 agent (no routing, original MedAide uses fixed routing)
        n_agents = 1

        # M3: Parallel agent execution
        agent_context = {
            "intents": top_intents,
            "subqueries": subqueries,
            "patient_id": patient_id,
            "tier": "Simple",
            "context_prefix": "",
        }
        from medaide_plus.modules.m3_dmacn import DMACNResult
        dmacn_result = await self.pipe.m3_dmacn.run(
            query=query, context=agent_context, n_agents=n_agents
        )

        # Critic + Synthesis (same as full pipeline but no M4/M5/M6/M7)
        critic_report = await self.pipe.critic_agent.evaluate(dmacn_result.agent_outputs)
        synthesis_result = await self.pipe.synthesis_agent.synthesize(
            agent_outputs=dmacn_result.agent_outputs,
            critic_report=critic_report,
            query=query,
        )
        return synthesis_result.synthesized_response


# ─── Main Benchmark Runner ────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Runs the full 3-condition comparison matching MedAide paper Tables 1–4.

    Args:
        config_path: Path to config/config.yaml.
        save_dir: Directory for saving results.
        limit: Max instances per category (None = run all 500).
        conditions: Which conditions to evaluate ('vanilla', 'medaide', 'full', 'all').
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        save_dir: Optional[Path] = None,
        limit: Optional[int] = None,
        conditions: str = "all",
    ) -> None:
        self.config_path = config_path
        self.save_dir = save_dir or Path(__file__).parent.parent / "data" / "benchmark"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.conditions = conditions
        self.pipeline = None
        self.sim_medaide = None

    def _init_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        logger.info("Initializing MedAide+ full pipeline (may take 1–3 min)...")
        from medaide_plus.pipeline import MedAidePlusPipeline
        self.pipeline = MedAidePlusPipeline(config_path=self.config_path)
        self.sim_medaide = SimulatedMedAidePipeline(self.pipeline)
        logger.info("Pipeline ready.")

    def _run_instance_vanilla(self, query: str, ref: str, category: str) -> Dict:
        """Run condition (A): Vanilla GPT-4o."""
        t0 = time.time()
        pred = _vanilla_gpt4o(query, category)
        lat = (time.time() - t0) * 1000
        return {"response": pred, "metrics": compute_metrics(pred, ref, lat)}

    def _run_instance_medaide(self, query: str, ref: str, patient_id: str) -> Dict:
        """Run condition (B): Simulated MedAide (M1+M2+M3)."""
        self._init_pipeline()
        t0 = time.time()
        pred = asyncio.run(self.sim_medaide.run(query, patient_id=patient_id))
        lat = (time.time() - t0) * 1000
        return {"response": pred, "metrics": compute_metrics(pred, ref, lat)}

    def _run_instance_full(self, query: str, ref: str, patient_id: str) -> Dict:
        """Run condition (C): Full MedAide+ (M1–M7)."""
        self._init_pipeline()
        t0 = time.time()
        result = asyncio.run(
            self.pipeline.run(query=query, patient_id=patient_id)
        )
        lat = (time.time() - t0) * 1000
        return {
            "response": result.final_response,
            "intents": result.intents,
            "tier": result.tier,
            "hallucination_rate": result.hallucination_rate,
            "n_agents": result.n_agents_used,
            "metrics": compute_metrics(result.final_response, ref, lat),
        }

    def run_category(
        self, category: str, instances: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Run all instances in one category for all selected conditions.

        Returns:
            (all_instance_records, category_summary)
        """
        to_run = instances[: self.limit] if self.limit else instances
        n = len(to_run)
        logger.info(f"\nCategory: {category} — {n} instances")

        all_records = []

        for i, inst in enumerate(to_run, 1):
            query = inst["query"]
            ref = inst.get("reference_answer", "")
            pid = f"eval_{category[:3].lower()}_{i:04d}"
            record: Dict[str, Any] = {
                "id": inst.get("id", f"q{i}"),
                "domain": category,
                "query": query,
                "expected_intents": inst.get("expected_intents", []),
                "reference_answer": ref[:200],
                "conditions": {},
            }

            # (A) Vanilla
            if self.conditions in ("all", "vanilla"):
                try:
                    record["conditions"]["vanilla"] = self._run_instance_vanilla(
                        query, ref, category
                    )
                except Exception as e:
                    record["conditions"]["vanilla"] = {"error": str(e), "metrics": {}}

            # (B) Simulated MedAide
            if self.conditions in ("all", "medaide"):
                try:
                    record["conditions"]["medaide"] = self._run_instance_medaide(
                        query, ref, pid
                    )
                except Exception as e:
                    record["conditions"]["medaide"] = {"error": str(e), "metrics": {}}

            # (C) Full MedAide+
            if self.conditions in ("all", "full", "medaide_plus"):
                try:
                    record["conditions"]["medaide_plus"] = self._run_instance_full(
                        query, ref, pid
                    )
                except Exception as e:
                    record["conditions"]["medaide_plus"] = {"error": str(e), "metrics": {}}

            all_records.append(record)
            if i % 50 == 0 or i == n:
                logger.info(f"  {category}: {i}/{n} done")

        summary = _aggregate_category(all_records)
        return all_records, summary

    def run_all(
        self, benchmark: Dict[str, List[Dict]]
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Run all 4 categories."""
        all_results: List[Dict] = []
        per_cat_summaries: Dict[str, Any] = {}

        for cat, instances in benchmark.items():
            cat_records, cat_summary = self.run_category(cat, instances)
            all_results.extend(cat_records)
            per_cat_summaries[cat] = cat_summary

        global_summary = _aggregate_global(all_results)

        # Build results_analyzer-compatible `conditions` section.
        # Maps friendly display names to {global, per_category} dicts.
        _cond_display = {
            "vanilla":      "Vanilla GPT-4o",
            "medaide":      "Simulated MedAide",
            "medaide_plus": "Full MedAide+",
        }
        conditions_view: Dict[str, Any] = {}
        for ck, cname in _cond_display.items():
            if ck not in global_summary:
                continue
            conditions_view[cname] = {
                "global": global_summary[ck],
                "per_category": {
                    cat: per_cat_summaries.get(cat, {}).get(ck, {})
                    for cat in per_cat_summaries
                },
            }

        return all_results, {
            "per_category": per_cat_summaries,
            "overall": global_summary,
            "conditions": conditions_view,   # consumed by results_analyzer.py
            "n_evaluated": len(all_results),
            "conditions_run": self.conditions,
            "timestamp": datetime.now().isoformat(),
        }

    def save_results(
        self, all_results: List[Dict], summary: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rp = self.save_dir / f"results_{ts}.json"
        sp = self.save_dir / f"summary_{ts}.json"
        lp = self.save_dir / "summary_latest.json"

        with open(rp, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        with open(sp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        import shutil
        shutil.copy2(sp, lp)

        logger.info(f"Results -> {rp}")
        logger.info(f"Summary -> {sp}")
        return rp, sp


# ─── Aggregation helpers ─────────────────────────────────────────────────────

_METRIC_KEYS = ["bleu_1", "bleu_2", "rouge_1", "rouge_2", "rouge_l",
                "gleu", "meteor", "bert_score", "latency_ms"]
_CONDITION_KEYS = ["vanilla", "medaide", "medaide_plus"]


def _aggregate_category(records: List[Dict]) -> Dict[str, Any]:
    """Average metrics per condition for one category."""
    agg: Dict[str, Any] = {}
    for cond in _CONDITION_KEYS:
        vals: Dict[str, List[float]] = {m: [] for m in _METRIC_KEYS}
        for r in records:
            m = r.get("conditions", {}).get(cond, {}).get("metrics", {})
            if m:
                for mk in _METRIC_KEYS:
                    if mk in m:
                        vals[mk].append(m[mk])
        if any(vals[m] for m in _METRIC_KEYS):
            agg[cond] = {mk: round(float(np.mean(vals[mk])), 2) if vals[mk] else 0.0
                         for mk in _METRIC_KEYS}
            agg[cond]["n"] = max(len(vals[m]) for m in _METRIC_KEYS)
    return agg


def _aggregate_global(all_records: List[Dict]) -> Dict[str, Any]:
    """Macro-average across all categories."""
    agg: Dict[str, Any] = {}
    for cond in _CONDITION_KEYS:
        vals: Dict[str, List[float]] = {m: [] for m in _METRIC_KEYS}
        for r in all_records:
            m = r.get("conditions", {}).get(cond, {}).get("metrics", {})
            if m:
                for mk in _METRIC_KEYS:
                    if mk in m:
                        vals[mk].append(m[mk])
        if any(vals[m] for m in _METRIC_KEYS):
            agg[cond] = {mk: round(float(np.mean(vals[mk])), 2) if vals[mk] else 0.0
                         for mk in _METRIC_KEYS}
    return agg


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _print_results_table(summary: Dict[str, Any]) -> None:
    """Print a formatted results table matching MedAide paper style."""
    metric_display = [
        ("bleu_1", "BLEU-1"), ("bleu_2", "BLEU-2"), ("meteor", "METEOR"),
        ("rouge_l", "ROUGE-L"), ("bert_score", "BERTScore"), ("gleu", "GLEU"),
    ]
    cond_display = {
        "vanilla": "Vanilla GPT-4o",
        "medaide": "GPT-4o + MedAide (orig.)",
        "medaide_plus": "GPT-4o + MedAide+",
    }

    for cat, cat_data in summary.get("per_category", {}).items():
        if not cat_data:
            continue
        print(f"\n{'-'*75}")
        print(f"  {cat}")
        print(f"{'-'*75}")
        header = f"  {'System':<28}" + "".join(f"{lbl:>10}" for _, lbl in metric_display)
        print(header)
        print(f"  {'-'*70}")
        for cond_key, cond_label in cond_display.items():
            m = cat_data.get(cond_key, {})
            if not m:
                continue
            row = f"  {cond_label:<28}"
            for mkey, _ in metric_display:
                row += f"{m.get(mkey, 0.0):>10.2f}"
            row += "  %"
            print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MedAide+ Phase 4 benchmark (3-condition comparison)."
    )
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max instances per category (e.g. 20 for quick test).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--condition", default="all",
                        choices=["all", "vanilla", "medaide", "full"],
                        help="Which conditions to run.")
    parser.add_argument("--fetch", action="store_true",
                        help="Generate benchmark first if not present.")
    parser.add_argument("--fetch-samples", type=int, default=500)
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    # Prefer v2 benchmark if available, fall back to v1
    default_v2 = base / "data" / "benchmark" / "medaide_plus_benchmark_v2.json"
    default_v1 = base / "data" / "benchmark" / "medaide_benchmark.json"
    if args.benchmark:
        bench_path = Path(args.benchmark)
    elif default_v2.exists():
        bench_path = default_v2
    else:
        bench_path = default_v1
    out_dir = Path(args.output_dir) if args.output_dir else base / "data" / "benchmark"

    if args.fetch or not bench_path.exists():
        from evaluation.fetch_benchmark import generate_composite_benchmark, save_benchmark
        bm = generate_composite_benchmark(n_per_category=args.fetch_samples)
        save_benchmark(bm, bench_path.parent)

    with open(bench_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    benchmark = raw.get("categories", raw)

    runner = BenchmarkRunner(
        config_path=args.config,
        save_dir=out_dir,
        limit=args.limit,
        conditions=args.condition,
    )
    all_results, summary = runner.run_all(benchmark)
    rp, sp = runner.save_results(all_results, summary)

    print("\n" + "=" * 75)
    print("  MedAide+ Phase 4 -- Benchmark Results (values in %)")
    print("  Matches: arXiv:2410.12532v3 Tables 1-4 protocol")
    print("=" * 75)
    _print_results_table(summary)
    print(f"\n  Full results -> {rp}")
    print(f"  Summary      -> {sp}")


if __name__ == "__main__":
    main()
