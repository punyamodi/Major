"""
MedAide+ Phase 4: Results Analyzer

Generates tables and figures matching MedAide paper (arXiv:2410.12532v3):
  - Table 1: Global comparison (Vanilla vs Simulated MedAide vs Full MedAide+)
  - Table 2: Per-category BLEU-1/2 comparison
  - Table 3: Per-category ROUGE-1/2/L comparison
  - Table 4: Per-category GLEU/METEOR/BERT-Score comparison
  - Table 5: Ablation study (ROUGE-L + GLEU, per module removed)

PAPER REFERENCE NUMBERS (GPT-4o + MedAide from paper, Table 2–4):
  These are the published MedAide metrics for GPT-4o as base LLM.
  BLEU-1: 41.37%, BLEU-2: 33.15%,
  ROUGE-1: 44.82%, ROUGE-2: 38.21%, ROUGE-L: 42.06%,
  GLEU: 33.74%, METEOR: 40.18%, BERT-Score: 88.65%
  (Reported for GPT-4o + MedAide, all categories averaged.)

Usage:
    python -m evaluation.results_analyzer
    python -m evaluation.results_analyzer --summary-only
    python -m evaluation.results_analyzer --tables-only
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("results_analyzer")

# ─── Published MedAide baselines (paper Table 2–4, GPT-4o + MedAide) ─────────

PAPER_MEDAIDE_GLOBAL = {
    "bleu_1":    41.37,
    "bleu_2":    33.15,
    "rouge_1":   44.82,
    "rouge_2":   38.21,
    "rouge_l":   42.06,
    "gleu":      33.74,
    "meteor":    40.18,
    "bert_score": 88.65,
}

# NOTE:
# The paper-level numbers above are global metrics. We do not fabricate
# per-category paper values; those are marked as N/A in category-level tables.

CATEGORIES = ["Pre-Diagnosis", "Diagnosis", "Medication", "Post-Diagnosis"]

METRIC_DISPLAY = {
    "bleu_1":     "BLEU-1",
    "bleu_2":     "BLEU-2",
    "rouge_1":    "ROUGE-1",
    "rouge_2":    "ROUGE-2",
    "rouge_l":    "ROUGE-L",
    "gleu":       "GLEU",
    "meteor":     "METEOR",
    "bert_score": "BERT-Score",
}

CONDITIONS_ORDER = ["Vanilla GPT-4o", "Simulated MedAide", "Full MedAide+"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_latest(path: Path) -> Optional[Dict]:
    if not path.exists():
        logger.warning(f"Not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe(d: Dict, *keys, default=0.0):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d if isinstance(d, (int, float)) else default


def _bold_max(values: List[float]) -> List[str]:
    """Return formatted strings, bold (stars) the maximum."""
    mx = max(values) if values else 0
    return [f"**{v:.2f}**" if v == mx else f"{v:.2f}" for v in values]


# ─── Table Printers ──────────────────────────────────────────────────────────

def print_table1_global(summary: Dict, paper_ref: Dict = PAPER_MEDAIDE_GLOBAL) -> None:
    """Table 1: Global (macro-average) across all categories, all metrics."""
    print("\n" + "=" * 96)
    print("  Table 1 -- Global Macro-Average Metric Comparison (all categories, all metrics)")
    print("  ref: arXiv:2410.12532v3 Table 2/3/4")
    print("=" * 96)

    conditions = summary.get("conditions", {})
    metrics = list(METRIC_DISPLAY.keys())

    # Header
    hdr = f"  {'Metric':<18}"
    for cond in CONDITIONS_ORDER:
        hdr += f"  {cond:>18}"
    hdr += f"  {'MedAide (Paper)':>18}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for m in metrics:
        vals = []
        for cond in CONDITIONS_ORDER:
            v = _safe(conditions, cond, "global", m)
            vals.append(v)
        formatted = _bold_max(vals)
        row = f"  {METRIC_DISPLAY[m]:<18}"
        for fv in formatted:
            row += f"  {fv:>18}"
        row += f"  {paper_ref.get(m, 0.0):>18.2f}"
        print(row)
    print()


def print_table2_bleu(summary: Dict) -> None:
    """Table 2: BLEU-1 and BLEU-2 per category."""
    print("\n" + "=" * 96)
    print("  Table 2 -- BLEU-1 / BLEU-2 Per Category")
    print("=" * 96)
    conditions = summary.get("conditions", {})
    metrics = [("bleu_1", "BLEU-1"), ("bleu_2", "BLEU-2")]
    _per_category_table(conditions, metrics)


def print_table3_rouge(summary: Dict) -> None:
    """Table 3: ROUGE-1/2/L per category."""
    print("\n" + "=" * 96)
    print("  Table 3 -- ROUGE-1 / ROUGE-2 / ROUGE-L Per Category")
    print("=" * 96)
    conditions = summary.get("conditions", {})
    metrics = [("rouge_1", "ROUGE-1"), ("rouge_2", "ROUGE-2"), ("rouge_l", "ROUGE-L")]
    _per_category_table(conditions, metrics)


def print_table4_other(summary: Dict) -> None:
    """Table 4: GLEU / METEOR / BERT-Score per category."""
    print("\n" + "=" * 96)
    print("  Table 4 -- GLEU / METEOR / BERT-Score Per Category")
    print("=" * 96)
    conditions = summary.get("conditions", {})
    metrics = [("gleu", "GLEU"), ("meteor", "METEOR"), ("bert_score", "BERT-Score")]
    _per_category_table(conditions, metrics)


def _per_category_table(conditions: Dict, metrics: List[Tuple[str, str]]) -> None:
    col_w = 14
    for metric_key, metric_label in metrics:
        print(f"\n  {metric_label}")
        hdr = f"  {'Category':<22}"
        for cond in CONDITIONS_ORDER:
            hdr += f"  {cond:>{col_w}}"
        hdr += f"  {'Paper(MedAide)':>{col_w}}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for cat in CATEGORIES:
            vals = [_safe(conditions, cond, "per_category", cat, metric_key)
                    for cond in CONDITIONS_ORDER]
            formatted = _bold_max(vals)
            row = f"  {cat:<22}"
            for fv in formatted:
                row += f"  {fv:>{col_w}}"
            row += f"  {'N/A':>{col_w}}"
            print(row)
    print()


def print_table5_ablation(ablation: Dict) -> None:
    """Table 5: Ablation study ROUGE-L / GLEU per condition per category."""
    print("\n" + "=" * 96)
    print("  Table 5 -- Ablation Study: Impact of Each Module on ROUGE-L and GLEU (%)")
    print("  Matches arXiv:2410.12532v3 Table 5 format (extended to M1-M7)")
    print("=" * 96)

    col_w = 10
    # Two-row header: condition | ROUGE-L x4 | GLEU x4 | macro
    hdr1 = f"  {'Condition':<28}"
    for _ in CATEGORIES:
        hdr1 += " " + " " * col_w
    hdr1 += " " + " " * col_w
    hdr2 = f"  {'':28}  "
    for cat in CATEGORIES:
        hdr2 += f"  {cat[:10]:>{col_w}}"
    hdr2 += f"  {'Macro':>{col_w}}"
    hdr2 += "   "
    for cat in CATEGORIES:
        hdr2 += f"  {cat[:10]:>{col_w}}"
    hdr2 += f"  {'Macro':>{col_w}}"
    print("  " + " " * 28 + "    ROUGE-L (%)  " + " " * (len(CATEGORIES) * (col_w+2) - 16)
          + "   GLEU (%)")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))

    for cond_name, data in ablation.items():
        pc = data.get("per_category", {})
        row = f"  {cond_name:<28}"
        for cat in CATEGORIES:
            row += f"  {pc.get(cat, {}).get('rouge_l', 0.0):>{col_w}.2f}"
        row += f"  {data.get('macro_rouge_l', 0.0):>{col_w}.2f}"
        row += "   "
        for cat in CATEGORIES:
            row += f"  {pc.get(cat, {}).get('gleu', 0.0):>{col_w}.2f}"
        row += f"  {data.get('macro_gleu', 0.0):>{col_w}.2f}"
        print(row)
    print()


# ─── CSV Export ──────────────────────────────────────────────────────────────

def export_csvs(summary: Dict, ablation: Optional[Dict], out_dir: Path) -> None:
    try:
        import csv
    except ImportError:
        logger.warning("csv module unavailable - skipping CSV export")
        return

    conditions = summary.get("conditions", {})
    metrics = list(METRIC_DISPLAY.keys())

    # Table 1 CSV
    with open(out_dir / "table1_global.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric"] + CONDITIONS_ORDER + ["MedAide (Paper)"])
        for m in metrics:
            row = [METRIC_DISPLAY[m]]
            for cond in CONDITIONS_ORDER:
                row.append(f"{_safe(conditions, cond, 'global', m):.2f}")
            row.append(f"{PAPER_MEDAIDE_GLOBAL.get(m, 0.0):.2f}")
            w.writerow(row)
        logger.info(f"  -> {out_dir / 'table1_global.csv'}")

    # Tables 2–4: per-category metric groups
    for label, metric_keys in [
        ("table2_bleu", ["bleu_1", "bleu_2"]),
        ("table3_rouge", ["rouge_1", "rouge_2", "rouge_l"]),
        ("table4_other", ["gleu", "meteor", "bert_score"]),
    ]:
        with open(out_dir / f"{label}.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["Metric", "Category"] + CONDITIONS_ORDER + ["Paper (MedAide)"]
            w.writerow(header)
            for m in metric_keys:
                for cat in CATEGORIES:
                    row = [METRIC_DISPLAY[m], cat]
                    for cond in CONDITIONS_ORDER:
                        row.append(f"{_safe(conditions, cond, 'per_category', cat, m):.2f}")
                    row.append("N/A")
                    w.writerow(row)
        logger.info(f"  -> {out_dir / label}.csv")

    # Table 5: ablation
    if ablation:
        with open(out_dir / "table5_ablation.csv", "w", newline="") as f:
            w = csv.writer(f)
            header = ["Condition"] + [f"ROUGE-L_{c}" for c in CATEGORIES] + ["ROUGE-L_Macro"] \
                   + [f"GLEU_{c}" for c in CATEGORIES] + ["GLEU_Macro"]
            w.writerow(header)
            for cname, data in ablation.items():
                pc = data.get("per_category", {})
                row = [cname]
                row += [f"{pc.get(c, {}).get('rouge_l', 0.0):.2f}" for c in CATEGORIES]
                row += [f"{data.get('macro_rouge_l', 0.0):.2f}"]
                row += [f"{pc.get(c, {}).get('gleu', 0.0):.2f}" for c in CATEGORIES]
                row += [f"{data.get('macro_gleu', 0.0):.2f}"]
                w.writerow(row)
        logger.info(f"  -> {out_dir / 'table5_ablation.csv'}")


# ─── Matplotlib Figures ──────────────────────────────────────────────────────

def generate_figures(summary: Dict, ablation: Optional[Dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib unavailable - skipping figure generation")
        return

    conditions = summary.get("conditions", {})

    # Figure 1: Global bar chart - key metrics comparison
    metrics_to_plot = ["bleu_1", "bleu_2", "rouge_1", "rouge_l", "gleu", "meteor", "bert_score"]
    x = np.arange(len(metrics_to_plot))
    width = 0.22
    colors = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000"]
    labels = CONDITIONS_ORDER + ["MedAide (Paper)"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (cond, color) in enumerate(zip(CONDITIONS_ORDER, colors[:3])):
        vals = [_safe(conditions, cond, "global", m) for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width, label=cond, color=color, alpha=0.85)
    paper_vals = [PAPER_MEDAIDE_GLOBAL.get(m, 0) for m in metrics_to_plot]
    ax.bar(x + 3 * width, paper_vals, width, label="MedAide (Paper)", color=colors[3], alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([METRIC_DISPLAY[m] for m in metrics_to_plot], fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Figure 1 -- Global Metric Comparison: Vanilla vs MedAide Sim vs MedAide+ vs Paper",
                 fontsize=11, pad=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = out_dir / "figure1_global_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    logger.info(f"  -> {p1}")

    # Figure 2: Ablation bar chart (ROUGE-L per condition, macro)
    if ablation:
        cond_names = list(ablation.keys())
        rl = [ablation[c].get("macro_rouge_l", 0.0) for c in cond_names]
        gleu = [ablation[c].get("macro_gleu", 0.0) for c in cond_names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, vals, title, color in [
            (axes[0], rl, "ROUGE-L (%)", "#5b9bd5"),
            (axes[1], gleu, "GLEU (%)", "#ed7d31"),
        ]:
            bars = ax.bar(range(len(cond_names)), vals, color=color, alpha=0.8, edgecolor="white")
            ax.set_xticks(range(len(cond_names)))
            ax.set_xticklabels(cond_names, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(f"Figure 2 -- Ablation: Macro {title}", fontsize=10)
            ax.set_ylim(0, max(vals) * 1.2 if vals else 1)
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        p2 = out_dir / "figure2_ablation.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        logger.info(f"  -> {p2}")

    # Figure 3: Radar chart (Full MedAide+ vs MedAide Paper)
    _generate_radar(conditions, out_dir)


def _generate_radar(conditions: Dict, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    metrics_r = ["bleu_1", "bleu_2", "rouge_1", "rouge_l", "gleu", "meteor"]
    labels_r = [METRIC_DISPLAY[m] for m in metrics_r]
    N = len(metrics_r)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    cond_data = {
        "Vanilla GPT-4o":   [_safe(conditions, "Vanilla GPT-4o",   "global", m) for m in metrics_r],
        "Simulated MedAide":[_safe(conditions, "Simulated MedAide", "global", m) for m in metrics_r],
        "Full MedAide+":    [_safe(conditions, "Full MedAide+",     "global", m) for m in metrics_r],
        "MedAide (Paper)":  [PAPER_MEDAIDE_GLOBAL.get(m, 0) for m in metrics_r],
    }
    palette = ["#5b9bd5", "#ed7d31", "#70ad47", "#ffc000"]

    for (name, vals), color in zip(cond_data.items(), palette):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, vals_plot, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, fontsize=10)
    ax.set_title("Figure 3 -- Radar: MedAide+ vs Baselines (global metrics, %)",
                 size=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)

    fig.tight_layout()
    p3 = out_dir / "figure3_radar.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  -> {p3}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MedAide+ Phase 4 results and generate tables + figures."
    )
    parser.add_argument("--summary", type=str, default=None,
                        help="Path to summary_latest.json (default: data/benchmark/summary_latest.json)")
    parser.add_argument("--ablation", type=str, default=None,
                        help="Path to ablation_latest.json (default: data/benchmark/ablation_latest.json)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only Table 1 (global summary).")
    parser.add_argument("--tables-only", action="store_true",
                        help="Print tables, skip figure generation.")
    parser.add_argument("--no-csv", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).parent.parent / "data" / "benchmark"
    summary_path  = Path(args.summary)  if args.summary  else base / "summary_latest.json"
    ablation_path = Path(args.ablation) if args.ablation else base / "ablation_latest.json"
    out_dir = Path(args.output_dir) if args.output_dir else base

    summary  = _load_latest(summary_path)
    ablation = _load_latest(ablation_path) if ablation_path.exists() else None

    if summary is None:
        logger.error(
            "No summary file found. Run benchmark_runner.py first:\n"
            "  python -m evaluation.benchmark_runner --fetch --fetch-samples 5 --limit 5"
        )
        sys.exit(1)

    print_table1_global(summary)

    if not args.summary_only:
        print_table2_bleu(summary)
        print_table3_rouge(summary)
        print_table4_other(summary)
        if ablation:
            print_table5_ablation(ablation)
        else:
            logger.info("No ablation file; skipping Table 5. Run ablation_runner.py first.")

    if not args.no_csv:
        logger.info("\nExporting CSVs...")
        export_csvs(summary, ablation, out_dir)

    if not args.tables_only:
        logger.info("Generating figures...")
        generate_figures(summary, ablation, out_dir)

    print("\nAnalysis complete.")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
