#!/usr/bin/env python3
"""
watch_and_run_gpt.py
Watches for phi4-reasoning:14b benchmark to complete, then:
1. Clears patient graphs
2. Runs gpt-oss:20b benchmark
3. Calls update_paper_results.py
4. Recompiles the LaTeX paper

Usage:  python watch_and_run_gpt.py
"""

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
TRIAL = BASE / "medaide_plus_trial"
PHI4_DIR = TRIAL / "data" / "benchmark" / "ollama_results" / "phi4-reasoning_14b"
GRAPHS_DIR = TRIAL / "data" / "patient_graphs"
PYTHON = sys.executable


def wait_for_phi4(poll_seconds=60):
    print("Watching for phi4-reasoning:14b results...", flush=True)
    while True:
        summaries = list(PHI4_DIR.glob("summary_*.json"))
        if summaries:
            print(f"\n✅ phi4 results found: {summaries[0].name}")
            return True
        print(".", end="", flush=True)
        time.sleep(poll_seconds)


def clear_graphs():
    graphs = list(GRAPHS_DIR.glob("*.pkl"))
    for g in graphs:
        g.unlink()
    print(f"🗑  Cleared {len(graphs)} patient graph files.")


def run_gpt_benchmark():
    print("\n🚀 Starting gpt-oss:20b benchmark...", flush=True)
    cmd = [
        PYTHON, "-m", "evaluation.run_ollama_benchmark",
        "--models", "gpt-oss:20b",
        "--limit", "5",
        "--condition", "all"
    ]
    result = subprocess.run(cmd, cwd=str(TRIAL))
    return result.returncode == 0


def update_paper():
    print("\n📄 Updating paper with results...", flush=True)
    result = subprocess.run([PYTHON, str(BASE / "update_paper_results.py")])
    return result.returncode == 0


def compile_paper():
    print("\n📑 Compiling LaTeX paper...", flush=True)
    tex = BASE / "MedAidePlus_Phase4_Paper.tex"
    for i in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(tex)],
            cwd=str(BASE), capture_output=True, text=True
        )
        if "Output written on" in result.stdout:
            print(f"  Pass {i+1}: ✅ compiled successfully")
        else:
            print(f"  Pass {i+1}: ⚠️  check output")
            print(result.stdout[-500:])
    return True


if __name__ == "__main__":
    wait_for_phi4()
    clear_graphs()
    ok = run_gpt_benchmark()
    if ok:
        update_paper()
        compile_paper()
        print("\n🎉 All done! Check MedAidePlus_Phase4_Paper.pdf")
    else:
        print("\n❌ gpt-oss:20b benchmark failed. Update paper manually.")
