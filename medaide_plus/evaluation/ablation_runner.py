"""
MedAide+ Phase 4: Ablation Study Runner

Matches MedAide paper Table 5 (arXiv:2410.12532v3, Section 4.6):
  - Ablation tests the contribution of each module
  - Original paper ablates: RIE (M1), IPM (M2), RAC (M3)
  - MedAide+ ablates all 7 modules: M1–M7 plus the original 3

ABLATION CONDITIONS:
  Baseline         — Full MedAide+ (M1–M7 all active)
  w/o RIE (M1)     — No query decomposition (matches original paper's w/o RIE)
  w/o IPM (M2)     — No intent classification (matches paper's w/o IPM)
  w/o RAC (M3)     — No parallel agent collaboration (matches paper's w/o RAC)
  w/o PLMM (M4)    — No patient history injection (MedAide+ addition)
  w/o HDFG (M5)    — No hallucination detection (MedAide+ addition)
  w/o AQCR (M6)    — No complexity routing (MedAide+ addition)
  w/o MIET (M7)    — No multi-turn EMA tracking (MedAide+ addition)

METRICS: ROUGE-L and GLEU (as in paper Table 5), reported as %

Usage:
    python -m evaluation.ablation_runner
    python -m evaluation.ablation_runner --limit 20   # quick test
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
logger = logging.getLogger("ablation_runner")


# ─── Metric helpers ──────────────────────────────────────────────────────────

def _tok(text: str) -> List[str]:
    return text.lower().split()


def _rouge_l_pct(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        return float(rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                     .score(ref, pred)["rougeL"].fmeasure) * 100
    except Exception:
        p, r = set(_tok(pred)), set(_tok(ref))
        return len(p & r) / max(len(r), 1) * 100


def _gleu_pct(pred: str, ref: str) -> float:
    try:
        from nltk.translate.gleu_score import sentence_gleu
        return float(sentence_gleu([_tok(ref)], _tok(pred))) * 100
    except Exception:
        p, r = set(_tok(pred)), set(_tok(ref))
        return len(p & r) / max(len(r), 1) * 90


# ─── Ablation patches ────────────────────────────────────────────────────────

class AblationCondition:
    name: str
    description: str
    paper_equivalent: str  # name used in original paper Table 5

    def apply(self, pipeline: Any) -> None:
        pass  # Baseline: all modules active

    def restore(self, pipeline: Any) -> None:
        pass


class NoM1_RIE(AblationCondition):
    """w/o RIE (M1 AMQU) — matches paper 'w/o RIE'."""
    name = "w/o RIE (M1)"
    description = "No query decomposition; raw query passed as single subquery"
    paper_equivalent = "w/o RIE"

    def apply(self, pipeline: Any) -> None:
        from medaide_plus.modules.m1_amqu import AMQUResult, SubQuery
        self._orig = pipeline.m1_amqu.run

        def passthrough(query: str):
            return AMQUResult(
                original_query=query,
                subqueries=[SubQuery(text=query)],
                processing_metadata={"ablation": "no_m1"},
            )
        pipeline.m1_amqu.run = passthrough

    def restore(self, pipeline: Any) -> None:
        if hasattr(self, "_orig"):
            pipeline.m1_amqu.run = self._orig


class NoM2_IPM(AblationCondition):
    """w/o IPM (M2 HDIO) — matches paper 'w/o IPM'."""
    name = "w/o IPM (M2)"
    description = "No intent classification; uniform distribution over 18 intents"
    paper_equivalent = "w/o IPM"

    def apply(self, pipeline: Any) -> None:
        from medaide_plus.modules.m2_hdio import HDIOResult, ALL_INTENTS
        self._orig = pipeline.m2_hdio.classify

        def uniform(query: str):
            n = len(ALL_INTENTS)
            return HDIOResult(
                intent_scores={i: 1.0 / n for i in ALL_INTENTS},
                top_intents=ALL_INTENTS[:3],
                top_category="Diagnosis",
                is_ood=False,
                confidence=1.0 / n,
                metadata={"ablation": "no_m2"},
            )
        pipeline.m2_hdio.classify = uniform

    def restore(self, pipeline: Any) -> None:
        if hasattr(self, "_orig"):
            pipeline.m2_hdio.classify = self._orig


class NoM3_RAC(AblationCondition):
    """w/o RAC (M3 DMACN) — matches paper 'w/o RAC' (no agent collaboration)."""
    name = "w/o RAC (M3)"
    description = "No parallel agent collaboration; single agent sequential response"
    paper_equivalent = "w/o RAC"

    def apply(self, pipeline: Any) -> None:
        self._orig = pipeline.m3_dmacn.run

        async def single_run(query, context=None, n_agents=1):
            # Disable collaboration by forcing a single active agent while
            # preserving the DMACNResult contract expected by the pipeline.
            result = await self._orig(query=query, context=context or {}, n_agents=1)
            result.metadata = {**result.metadata, "ablation": "no_m3"}
            return result
        pipeline.m3_dmacn.run = single_run

    def restore(self, pipeline: Any) -> None:
        if hasattr(self, "_orig"):
            pipeline.m3_dmacn.run = self._orig


class NoM4_PLMM(AblationCondition):
    """w/o PLMM (M4) — no patient longitudinal memory."""
    name = "w/o PLMM (M4)"
    description = "No patient history injection (stateless per query)"
    paper_equivalent = "N/A (MedAide+ extension)"

    def apply(self, pipeline: Any) -> None:
        from medaide_plus.modules.m4_plmm import PLMMResult
        self._oi = pipeline.m4_plmm.inject_history
        self._ou = pipeline.m4_plmm.update_from_response

        def passthrough(query, patient_id):
            return PLMMResult(enriched_query=query, patient_id=patient_id,
                              relevant_history=[], n_history_nodes=0,
                              metadata={"ablation": "no_m4"})

        pipeline.m4_plmm.inject_history = passthrough
        pipeline.m4_plmm.update_from_response = lambda *a, **k: None

    def restore(self, pipeline: Any) -> None:
        if hasattr(self, "_oi"):
            pipeline.m4_plmm.inject_history = self._oi
            pipeline.m4_plmm.update_from_response = self._ou


class NoM5_HDFG(AblationCondition):
    """w/o HDFG (M5) — no hallucination detection/filtering."""
    name = "w/o HDFG (M5)"
    description = "No hallucination detection; raw synthesis output returned"
    paper_equivalent = "N/A (MedAide+ extension)"

    def apply(self, pipeline: Any) -> None:
        from medaide_plus.modules.m5_hdfg import HdfgResult
        self._orig = pipeline.m5_hdfg.annotate_response

        def passthrough(response, query=""):
            return HdfgResult(original_response=response, annotated_response=response,
                              verified_claims=[], hallucination_rate=0.0,
                              overall_confidence=1.0,
                              metadata={"ablation": "no_m5"})
        pipeline.m5_hdfg.annotate_response = passthrough

    def restore(self, pipeline: Any) -> None:
        if hasattr(self, "_orig"):
            pipeline.m5_hdfg.annotate_response = self._orig


class NoM6_AQCR(AblationCondition):
    """w/o AQCR (M6) — no complexity-based routing (always 1 agent)."""
    name = "w/o AQCR (M6)"
    description = "No adaptive complexity routing; always use 1 agent"
    paper_equivalent = "N/A (MedAide+ extension)"

    def apply(self, pipeline: Any) -> None:
        from medaide_plus.modules.m6_aqcr import AQCRResult
        self._orig = pipeline.m6_aqcr.route

        def single_route(query, intents, subqueries=None):
            return AQCRResult(tier="Simple", n_agents=1, confidence=1.0,
                              routing_method="ablation",
                              features={},
                              tier_probabilities={
                                  "Simple": 1.0,
                                  "Moderate": 0.0,
                                  "Complex": 0.0,
                              })
        pipeline.m6_aqcr.route = single_route

    def restore(self, pipeline: Any) -> None:
        if hasattr(self, "_orig"):
            pipeline.m6_aqcr.route = self._orig


class NoM7_MIET(AblationCondition):
    """w/o MIET (M7) — no multi-turn EMA state tracking."""
    name = "w/o MIET (M7)"
    description = "No multi-turn intent evolution tracking (stateless)"
    paper_equivalent = "N/A (MedAide+ extension)"

    def apply(self, pipeline: Any) -> None:
        self._u = pipeline.m7_miet.update
        self._i = pipeline.m7_miet.inject_state
        self._p = pipeline.m7_miet.get_context_prefix

        pipeline.m7_miet.update = lambda *a, **k: None
        pipeline.m7_miet.inject_state = lambda sid, scores: scores
        pipeline.m7_miet.get_context_prefix = lambda sid: ""

    def restore(self, pipeline: Any) -> None:
        if hasattr(self, "_u"):
            pipeline.m7_miet.update = self._u
            pipeline.m7_miet.inject_state = self._i
            pipeline.m7_miet.get_context_prefix = self._p


# Registry: ordered to match paper Table 5 structure
ABLATION_CONDITIONS: List[AblationCondition] = [
    AblationCondition(),      # Baseline (no patching)
    NoM1_RIE(),
    NoM2_IPM(),
    NoM3_RAC(),
    NoM4_PLMM(),
    NoM5_HDFG(),
    NoM6_AQCR(),
    NoM7_MIET(),
]
# Set baseline name/description
ABLATION_CONDITIONS[0].name = "Full MedAide+ (Baseline)"
ABLATION_CONDITIONS[0].description = "All 7 modules active"
ABLATION_CONDITIONS[0].paper_equivalent = "Full Model"

CONDITION_KEY_MAP = {
    "baseline": 0, "m1": 1, "m2": 2, "m3": 3, "m4": 4, "m5": 5, "m6": 6, "m7": 7
}


# ─── Ablation Runner ─────────────────────────────────────────────────────────

class AblationRunner:
    """
    Runs ablation experiments matching MedAide paper Table 5.

    Args:
        config_path: Path to config.yaml.
        save_dir: Output directory.
        limit: Max instances per category per condition.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        save_dir: Optional[Path] = None,
        limit: Optional[int] = None,
    ) -> None:
        self.config_path = config_path
        self.save_dir = save_dir or Path(__file__).parent.parent / "data" / "benchmark"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.pipeline = None

    def _init_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        logger.info("Initializing MedAide+ pipeline for ablation study...")
        from medaide_plus.pipeline import MedAidePlusPipeline
        self.pipeline = MedAidePlusPipeline(config_path=self.config_path)
        logger.info("Pipeline ready.")

    def _run_condition(
        self,
        cond: AblationCondition,
        benchmark: Dict[str, List[Dict]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Run one ablation condition across all categories.
        Returns per-category {"rouge_l": ..., "gleu": ..., "n": ...}.
        """
        self._init_pipeline()
        cond.apply(self.pipeline)
        cat_metrics: Dict[str, Dict] = {}

        try:
            for cat, instances in benchmark.items():
                to_run = instances[: self.limit] if self.limit else instances
                rouge_vals, gleu_vals = [], []
                for inst in to_run:
                    q = inst["query"]
                    ref = inst.get("reference_answer", "")
                    try:
                        result = asyncio.run(
                            self.pipeline.run(query=q, patient_id="ablation_eval")
                        )
                        pred = result.final_response
                        rouge_vals.append(_rouge_l_pct(pred, ref))
                        gleu_vals.append(_gleu_pct(pred, ref))
                    except Exception as e:
                        logger.debug(f"  Ablation instance error: {e}")
                        rouge_vals.append(0.0)
                        gleu_vals.append(0.0)

                cat_metrics[cat] = {
                    "rouge_l": round(float(np.mean(rouge_vals)), 2) if rouge_vals else 0.0,
                    "gleu":    round(float(np.mean(gleu_vals)), 2) if gleu_vals else 0.0,
                    "n": len(rouge_vals),
                }
        finally:
            cond.restore(self.pipeline)

        return cat_metrics

    def run_all(
        self,
        benchmark: Dict[str, List[Dict]],
        condition_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run selected ablation conditions and return results."""
        if condition_keys:
            selected = [ABLATION_CONDITIONS[CONDITION_KEY_MAP[k]]
                        for k in condition_keys if k in CONDITION_KEY_MAP]
        else:
            selected = ABLATION_CONDITIONS

        results: Dict[str, Any] = {}
        cats = list(benchmark.keys())

        for cond in selected:
            logger.info(f"Ablation: {cond.name}")
            cat_metrics = self._run_condition(cond, benchmark)

            all_rouge = [v["rouge_l"] for v in cat_metrics.values()]
            all_gleu  = [v["gleu"]    for v in cat_metrics.values()]

            results[cond.name] = {
                "key": cond.paper_equivalent,
                "description": cond.description,
                "per_category": cat_metrics,
                "macro_rouge_l": round(float(np.mean(all_rouge)), 2),
                "macro_gleu":    round(float(np.mean(all_gleu)), 2),
            }
            logger.info(
                f"  -> macro ROUGE-L={results[cond.name]['macro_rouge_l']:.2f}%  "
                f"GLEU={results[cond.name]['macro_gleu']:.2f}%"
            )

        return results

    def save_results(self, results: Dict[str, Any]) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.save_dir / f"ablation_{ts}.json"
        lat = self.save_dir / "ablation_latest.json"

        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        import shutil
        shutil.copy2(out, lat)

        logger.info(f"Ablation results -> {out}")
        return out


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _print_ablation_table(results: Dict[str, Any]) -> None:
    """Print Table 5 style ablation table."""
    cats = ["Pre-Diagnosis", "Diagnosis", "Medication", "Post-Diagnosis"]
    print("\n" + "=" * 90)
    print("  Table 5 -- Ablation Study (ROUGE-L / GLEU in %)")
    print("  Matches: arXiv:2410.12532v3 Section 4.6")
    print("=" * 90)

    col_w = 14
    header = f"  {'Condition':<28}"
    for cat in cats:
        header += f"  {cat[:6]+'/R-L':>{col_w}}"
    for cat in cats:
        header += f"  {cat[:6]+'/GLEU':>{col_w}}"
    header += f"  {'MacroR-L':>{col_w}}  {'MacroGLEU':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cond_name, data in results.items():
        pc = data.get("per_category", {})
        row = f"  {cond_name:<28}"
        for cat in cats:
            row += f"  {pc.get(cat, {}).get('rouge_l', 0.0):>{col_w}.2f}"
        for cat in cats:
            row += f"  {pc.get(cat, {}).get('gleu', 0.0):>{col_w}.2f}"
        row += f"  {data.get('macro_rouge_l', 0.0):>{col_w}.2f}"
        row += f"  {data.get('macro_gleu', 0.0):>{col_w}.2f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MedAide+ ablation study (matches paper Table 5)."
    )
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="+",
                        choices=list(CONDITION_KEY_MAP.keys()), default=None,
                        help="Ablation conditions to run (default: all).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    bench_path = (Path(args.benchmark) if args.benchmark
                  else base / "data" / "benchmark" / "medaide_benchmark.json")
    out_dir = Path(args.output_dir) if args.output_dir else base / "data" / "benchmark"

    if not bench_path.exists():
        logger.error(f"Benchmark not found: {bench_path}\nRun: python -m evaluation.fetch_benchmark")
        sys.exit(1)

    with open(bench_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    benchmark = raw.get("categories", raw)

    runner = AblationRunner(config_path=args.config, save_dir=out_dir, limit=args.limit)
    results = runner.run_all(benchmark, condition_keys=args.conditions)
    out_path = runner.save_results(results)
    _print_ablation_table(results)
    print(f"\n  Ablation results -> {out_path}")


if __name__ == "__main__":
    main()
