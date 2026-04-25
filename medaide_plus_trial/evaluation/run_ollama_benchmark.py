"""
MedAide+ Ollama Multi-Model Benchmark Orchestrator
====================================================
Runs the full 3-condition benchmark (Vanilla / MedAide / MedAide+) for every
specified Ollama model and saves per-model result files.

Usage
-----
    python -m evaluation.run_ollama_benchmark                   # defaults
    python -m evaluation.run_ollama_benchmark --limit 20        # quick smoke test
    python -m evaluation.run_ollama_benchmark --models qwen3:8b gemma3:4b

Output files (in data/benchmark/ollama_results/)
-------------------------------------------------
    <model_slug>_results_<ts>.json   — per-instance results
    <model_slug>_summary_<ts>.json   — aggregated metric table
    comparison_<ts>.json             — side-by-side model comparison
"""

from __future__ import annotations

import os
# Force all Python ML models (BioBERT, SentenceTransformer, Flan-T5) onto CPU.
# Without this, torch auto-detects CUDA and loads them on the RTX 3070, consuming
# ~2-3 GB VRAM and evicting the Ollama LLM model to system RAM, causing CPU-only
# inference at ~2–3 tok/s instead of the expected ~15 tok/s on GPU.
# Ollama runs as a SEPARATE process with its own CUDA context and is NOT affected.
# Use "-1" (not ""), which is the universally recognized value for "no CUDA devices"
# on Windows; empty string is ambiguous and may be ignored by the CUDA driver.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Windows consoles can default to a non-UTF8 encoding (e.g., cp1252).
# Ensure benchmark status prints never crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── helpers ──────────────────────────────────────────────────────────────────

def _slug(model: str) -> str:
    """Convert 'qwen3:8b' → 'qwen3_8b' for file names."""
    return model.replace(":", "_").replace("/", "_")


def _model_running(model: str) -> bool:
    """Return True when the model can respond to a test prompt."""
    try:
        import httpx
        resp = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'ok'."}],
                "stream": False,
                "options": {"num_predict": 5},
            },
            timeout=30,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _ollama_list() -> List[str]:
    """Return list of locally available Ollama model names.

    Primary: `ollama list` subprocess.
    Fallback: Ollama REST API `/api/tags`.
    """
    # 1) CLI subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=20
        )
        lines = (result.stdout or "").strip().splitlines()
        models: List[str] = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        if models:
            return models
    except Exception:
        pass

    # 2) REST API fallback
    try:
        import httpx

        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json() or {}
        models = [m.get("name", "") for m in (data.get("models") or [])]
        return [m for m in models if m]
    except Exception:
        return []


def _write_ollama_config(model: str, config_path: Path) -> Path:
    """
    Write a temporary config.yaml that forces the Ollama provider with the
    given model name. Returns path to the temp config.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["llm"]["provider"] = "ollama"
    cfg["llm"]["ollama"]["model"] = model
    cfg["llm"]["ollama"]["base_url"] = "http://localhost:11434"
    cfg["llm"]["ollama"]["max_tokens"] = 512  # Balanced: sufficient detail + not excessive
    cfg["llm"]["ollama"]["temperature"] = 0.0  # Deterministic: same agent + same prompt → same output → clean delta
    cfg["llm"]["ollama"]["num_ctx"] = 2048  # Keep KV-cache small → full GPU load

    tmp = config_path.parent / f"_tmp_ollama_{_slug(model)}.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return tmp


# ── benchmark runner wrapper ──────────────────────────────────────────────────

def run_benchmark_for_model(
    model: str,
    limit: int,
    condition: str,
    out_dir: Path,
    config_path: Path,
    benchmark_path: Optional[Path],
) -> Dict[str, Any]:
    """Run benchmark_runner for one Ollama model and return the summary dict."""
    slug = _slug(model)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_out_dir = out_dir / slug
    model_out_dir.mkdir(parents=True, exist_ok=True)

    tmp_cfg = _write_ollama_config(model, config_path)

    try:
        # Import here so env vars from tmp_cfg are read at call time
        os.environ["MEDAIDE_CONFIG"] = str(tmp_cfg)

        # Build CLI args for benchmark_runner
        cmd = [
            sys.executable, "-m", "evaluation.benchmark_runner",
            "--config", str(tmp_cfg),
            "--output-dir", str(model_out_dir),
            "--condition", condition,
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        if benchmark_path:
            cmd += ["--benchmark", str(benchmark_path)]

        print(f"\n{'='*70}")
        print(f"  Running benchmark: {model}  (limit={limit}, condition={condition})")
        print(f"{'='*70}")

        start = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(ROOT))
        elapsed = time.time() - start

        # Read back the latest summary file written by benchmark_runner
        summary_files = sorted(model_out_dir.glob("summary_*.json"), reverse=True)
        results_files = sorted(model_out_dir.glob("results_*.json"), reverse=True)

        summary: Dict[str, Any] = {}
        if summary_files:
            with open(summary_files[0], "r", encoding="utf-8") as f:
                summary = json.load(f)

        # Attach metadata
        meta = {
            "model": model,
            "provider": "ollama",
            "elapsed_seconds": round(elapsed, 1),
            "limit": limit,
            "condition": condition,
            "timestamp": ts,
            "summary_file": str(summary_files[0]) if summary_files else None,
            "results_file": str(results_files[0]) if results_files else None,
            "exit_code": result.returncode,
        }
        summary["_meta"] = meta
        return summary

    finally:
        if tmp_cfg.exists():
            tmp_cfg.unlink()


# ── comparison table ──────────────────────────────────────────────────────────

def _print_comparison(all_summaries: Dict[str, Any]) -> None:
    """Print a side-by-side comparison table for all models."""
    metrics = ["BLEU-1", "BLEU-2", "ROUGE-L", "GLEU", "METEOR", "BERTScore"]
    conditions_order = ["vanilla", "medaide", "full"]
    condition_labels = {"vanilla": "Vanilla", "medaide": "MedAide", "full": "MedAide+"}

    print("\n" + "=" * 90)
    print("  MULTI-MODEL COMPARISON — MedAide+ Benchmark (values in %)")
    print("=" * 90)

    models = list(all_summaries.keys())
    if not models:
        print("  (no results)")
        return

    header = f"  {'Metric':<14}" + "".join(
        f"{m[:18]:>22}" for m in models
    )
    print(header)
    print("  " + "-" * (14 + 22 * len(models)))

    for cond in conditions_order:
        label = condition_labels.get(cond, cond)
        print(f"\n  [{label}]")
        for metric in metrics:
            row = f"    {metric:<12}"
            for model in models:
                val = all_summaries[model].get(cond, {}).get(metric.lower().replace("-", "_").replace(" ", "_"), None)
                if val is None:
                    # Try alternate key formats
                    for k in [metric, metric.lower(), metric.upper(),
                               metric.replace("-", "_"), "bert_score"]:
                        if k in all_summaries[model].get(cond, {}):
                            val = all_summaries[model][cond][k]
                            break
                row += f"{(val if val is not None else 'N/A'):>22}"
            print(row)

    print("\n" + "=" * 90)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MedAide+ Ollama multi-model benchmark."
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["qwen3:8b", "gemma3:4b"],
        help="Ollama model tags to benchmark (default: qwen3:8b gemma3:4b)",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Instances per category (default: 20 for speed; use 750 for full run)",
    )
    parser.add_argument(
        "--condition", default="all",
        choices=["all", "vanilla", "medaide", "full"],
        help="Which evaluation conditions to run (default: all)",
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Path to benchmark JSON (auto-detects v2 if omitted)",
    )
    parser.add_argument(
        "--config", type=str,
        default=str(ROOT / "config" / "config.yaml"),
        help="Base config.yaml path",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(ROOT / "data" / "benchmark" / "ollama_results"),
        help="Directory for results",
    )
    parser.add_argument(
        "--skip-check", action="store_true",
        help="Skip Ollama availability check (for scripting)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve benchmark path
    if args.benchmark:
        bench_path = Path(args.benchmark)
    else:
        v2 = ROOT / "data" / "benchmark" / "medaide_plus_benchmark_v2.json"
        v1 = ROOT / "data" / "benchmark" / "medaide_benchmark.json"
        bench_path = v2 if v2.exists() else v1

    print(f"\nBenchmark: {bench_path.name}")
    print(f"Models:    {', '.join(args.models)}")
    print(f"Limit:     {args.limit} instances/category")
    print(f"Output:    {out_dir}")

    # Check which models are available locally
    if not args.skip_check:
        available = _ollama_list()
        pretty = ", ".join(available) if available else "(none)"
        print(f"\nLocal Ollama models: {pretty}")
        for model in args.models:
            base = model.split(":")[0]
            found = any(m.startswith(model) or m.startswith(base) for m in available)
            if not found:
                print(f"  [WARN] '{model}' not found locally - run: ollama pull {model}")

    # Sanity-ping Ollama daemon
    try:
        import httpx
        r = httpx.get("http://localhost:11434/", timeout=5)
        print(f"\nOllama daemon: OK (status {r.status_code})")
    except Exception as e:
        print(f"\n[WARN] Ollama daemon not reachable: {e}")
        if not args.skip_check:
            print("Start Ollama and re-run, or pass --skip-check.")
            sys.exit(1)

    # Clear stale patient graphs so M4 PLMM starts fresh.
    # Stale graphs from prior runs inject medical history context into MedAide+
    # responses, causing vocabulary drift from the reference and lower BLEU/METEOR.
    graphs_dir = ROOT / "data" / "patient_graphs"
    if graphs_dir.exists():
        stale = list(graphs_dir.glob("*.json"))
        if stale:
            for g in stale:
                g.unlink()
            print(f"\n  Cleared {len(stale)} stale patient graph(s) in {graphs_dir}")
        else:
            print(f"\n  Patient graphs dir clean ({graphs_dir})")

    # Run each model
    all_summaries: Dict[str, Any] = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in args.models:
        print(f"\n\n{'#'*70}")
        print(f"#  MODEL: {model}")
        print(f"{'#'*70}")

        # Wait for model to be ready (it might be loading)
        if not args.skip_check:
            print(f"  Pinging {model}...", end="", flush=True)
            ready = False
            for _ in range(5):
                if _model_running(model):
                    ready = True
                    break
                print(".", end="", flush=True)
                time.sleep(3)
            print(" OK" if ready else " TIMEOUT (will try anyway)")

        summary = run_benchmark_for_model(
            model=model,
            limit=args.limit,
            condition=args.condition,
            out_dir=out_dir,
            config_path=config_path,
            benchmark_path=bench_path,
        )
        all_summaries[model] = summary

    # Print comparison
    _print_comparison(all_summaries)

    # Save combined comparison
    comparison_path = out_dir / f"comparison_{ts}.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": args.models,
                "limit": args.limit,
                "condition": args.condition,
                "timestamp": ts,
                "results": all_summaries,
            },
            f,
            indent=2,
            default=str,
        )

    # Also keep latest comparison link
    latest_path = out_dir / "comparison_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": args.models,
                "limit": args.limit,
                "condition": args.condition,
                "timestamp": ts,
                "results": all_summaries,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n  Comparison saved → {comparison_path}")
    print(f"  Latest link     → {latest_path}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
