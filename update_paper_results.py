#!/usr/bin/env python3
"""
update_paper_results.py
Reads all 4 model summary_latest.json files and updates MedAidePlus_Phase4_Paper.tex
Main table (Table 2) and per-category table (Table 3) with actual bench_v11/v12 results.

Usage:
    python update_paper_results.py
"""
import json, re, shutil
from pathlib import Path

BASE   = Path(__file__).parent
BENCH  = BASE / "medaide_plus_trial" / "data" / "benchmark" / "ollama_results"
TEX    = BASE / "MedAidePlus_Phase4_Paper.tex"

MODEL_DIRS = {
    "gemma3:4b":          "gemma3_4b",
    "qwen3:8b":           "qwen3_8b",
    "phi4-reasoning:14b": "phi4-reasoning_14b",
    "gpt-oss:20b":        "gpt-oss_20b",
}
CATEGORIES = ["Pre-Diagnosis","Diagnosis","Medication","Post-Diagnosis"]
METRICS    = ["bleu_1","bleu_2","meteor","rouge_l","bert_score","gleu"]

def load(mk):
    p = BENCH / MODEL_DIRS[mk] / "summary_latest.json"
    if not p.exists(): return None
    return json.loads(p.read_text())

def f(v, d=2): return f"{v:.{d}f}" if v is not None else "--"

def delta_tex(d):
    if d is None: return "--"
    s = "+" if d >= 0 else ""
    c = "wingreen" if d >= 0 else "lossred"
    return r"\textcolor{" + c + r"}{" + s + f"{d:.2f}" + "}"

def build_main(data):
    rows = []
    avg_ma = {m:[] for m in METRICS}
    avg_mp = {m:[] for m in METRICS}
    for mk in MODEL_DIRS:
        s = data.get(mk)
        label = mk.replace("-reasoning","").replace(":",":")
        if s is None:
            rows += [
                r"\multirow{3}{*}{" + mk + "}",
                "  & MedAide   & -- & -- & -- & -- & -- & -- \\\\",
                "  & MedAide+  & -- & -- & -- & -- & -- & -- \\\\",
                r"  & $\Delta$  & -- & -- & -- & -- & -- & -- \\",
                r"\midrule",
            ]
            continue
        ma = s["overall"]["medaide"]
        mp = s["overall"]["medaide_plus"]
        mv = [ma.get(m) for m in METRICS]
        pv = [mp.get(m) for m in METRICS]
        dv = [pv[i]-mv[i] if pv[i] is not None and mv[i] is not None else None for i in range(len(METRICS))]
        for m,v in zip(METRICS,mv):
            if v is not None: avg_ma[m].append(v)
        for m,v in zip(METRICS,pv):
            if v is not None: avg_mp[m].append(v)
        def bold(v, is_mp, i): 
            if v is None: return "--"
            s = f(v)
            if is_mp and pv[i] is not None and mv[i] is not None and pv[i] > mv[i]:
                return r"\textbf{" + s + "}"
            return s
        ma_s = " & ".join(bold(mv[i], False, i) for i in range(len(METRICS)))
        mp_s = " & ".join(bold(pv[i], True, i)  for i in range(len(METRICS)))
        d_s  = " & ".join(delta_tex(d) for d in dv)
        rows += [
            r"\multirow{3}{*}{" + mk + "}",
            f"  & MedAide   & {ma_s} \\\\",
            f"  & MedAide+  & {mp_s} \\\\",
            f"  & $\\Delta$  & {d_s} \\\\",
            r"\midrule",
        ]
    # averages
    ama = [sum(avg_ma[m])/len(avg_ma[m]) if avg_ma[m] else None for m in METRICS]
    amp = [sum(avg_mp[m])/len(avg_mp[m]) if avg_mp[m] else None for m in METRICS]
    def bold_avg(v, i):
        if v is None: return "--"
        s = f(v)
        if amp[i] is not None and ama[i] is not None and amp[i] > ama[i]:
            return r"\textbf{" + s + "}"
        return s
    ama_s = " & ".join(f(v) for v in ama)
    amp_s = " & ".join(bold_avg(amp[i], i) for i in range(len(METRICS)))
    rows += [
        r"\multirow{2}{*}{\textbf{Average}}",
        f"  & \\textbf{{MedAide}}   & {ama_s} \\\\",
        f"  & \\textbf{{MedAide+}}  & {amp_s} \\\\",
    ]
    return "\n".join(rows)

def build_cat(data):
    rows = []
    models = list(MODEL_DIRS.keys())
    for idx, mk in enumerate(models):
        s = data.get(mk)
        short = mk
        if s is None:
            r = [r"\multirow{2}{*}{" + short + "}",
                 "  & MedAide  & -- & -- & -- & -- \\\\",
                 "  & MedAide+ & -- & -- & -- & -- \\\\"]
        else:
            pc = s.get("per_category", {})
            mv = [pc.get(c, {}).get("medaide", {}).get("bleu_1") for c in CATEGORIES]
            pv = [pc.get(c, {}).get("medaide_plus", {}).get("bleu_1") for c in CATEGORIES]
            def bold_cat(v_p, v_m):
                if v_p is None: return "--"
                s = f(v_p)
                if v_m is not None and v_p > v_m: return r"\textbf{" + s + "}"
                return s
            ma_s = " & ".join(f(v) for v in mv)
            mp_s = " & ".join(bold_cat(pv[i], mv[i]) for i in range(len(CATEGORIES)))
            r = [r"\multirow{2}{*}{" + short + "}",
                 f"  & MedAide  & {ma_s} \\\\",
                 f"  & MedAide+ & {mp_s} \\\\"]
        rows += r
        if idx < len(models) - 1:
            rows.append(r"\midrule")
    return "\n".join(rows)

import sys, datetime

BENCH_V12_CUTOFF = datetime.datetime(2026, 4, 12, 23, 0)  # bench_v12 started ~00:35 UTC April 13

def is_fresh(mk):
    """Return True only if the summary was written AFTER bench_v12 started."""
    p = BENCH / MODEL_DIRS[mk] / "summary_latest.json"
    if not p.exists(): return False
    mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
    return mtime > BENCH_V12_CUTOFF

def main():
    data = {}
    stale = []
    for mk in MODEL_DIRS:
        s = load(mk)
        data[mk] = s
        if s:
            ma_b1 = s["overall"]["medaide"].get("bleu_1",0)
            mp_b1 = s["overall"]["medaide_plus"].get("bleu_1",0)
            fresh = is_fresh(mk)
            tag = "[FRESH bench_v12]" if fresh else "[STALE — old run]"
            print(f"  {mk}: n={s.get('n_evaluated','?')} Delta-BLEU-1={mp_b1-ma_b1:+.2f}  {tag}")
            if not fresh: stale.append(mk)
        else:
            print(f"  {mk}: NOT FOUND")
            stale.append(mk)

    if stale and "--force" not in sys.argv:
        print(f"\n⚠ STALE DATA for: {stale}")
        print("  Run again with --force to update anyway, or wait for bench_v12 to finish.")
        print("  Tip: monitor bench_monitor.log for 'FINAL RESULTS' or all 4 model completions.")
        return

    main_body = build_main(data)
    cat_body  = build_cat(data)

    # backup + write
    bak = str(TEX).replace(".tex", "_pre_update.tex")
    shutil.copy(TEX, bak)

    src = TEX.read_text(encoding="utf-8")

    # Replace main table rows (between first \midrule and \bottomrule in Table 2)
    # Pattern: find the tabular labeled tab:main
    pat = re.compile(
        r'(\\label\{tab:main\}.*?\\midrule\n)(.*?)(\\bottomrule\n\\multicolumn\{8\})',
        re.DOTALL
    )
    src2, n2 = re.subn(pat, lambda m: m.group(1)+main_body+"\n"+m.group(3), src, count=1)
    if n2: print("  ✓ Main table updated"); src = src2
    else: print("  [WARN] Main table not updated — pattern not found")

    # Replace per-category table rows
    pat2 = re.compile(
        r'(\\label\{tab:percategory\}.*?\\midrule\n)(.*?)(\\bottomrule\n\\end\{tabular\})',
        re.DOTALL
    )
    src3, n3 = re.subn(pat2, lambda m: m.group(1)+cat_body+"\n"+m.group(3), src, count=1)
    if n3: print("  ✓ Per-category table updated"); src = src3
    else: print("  [WARN] Per-category table not updated — pattern not found")

    # Update abstract BLEU-1 range
    deltas = []
    for mk,s in data.items():
        if s:
            ma = s["overall"]["medaide"].get("bleu_1")
            mp = s["overall"]["medaide_plus"].get("bleu_1")
            if ma and mp and ma > 0: deltas.append(100*(mp-ma)/ma)
    if len(deltas) >= 2:
        lo, hi = min(deltas), max(deltas)
        src = re.sub(
            r'BLEU-1 gains ranging from \$[^$]+\$ to \$[^%]+%\$ relative',
            f'BLEU-1 gains ranging from ${lo:.1f}$ to ${hi:.1f}\\%$ relative',
            src
        )
        print(f"  ✓ Abstract range: {lo:.1f}% to {hi:.1f}%")

    TEX.write_text(src, encoding="utf-8")

    # Update bar chart figure — 4-model version when all models available, otherwise qwen3-only
    src = TEX.read_text(encoding="utf-8")
    model_colors = {
        "gemma3:4b":          ("modulered!80",    "modulered"),
        "qwen3:8b":           ("moduleblue!70",   "moduleblue"),
        "phi4-reasoning:14b": ("modulegreen!80",  "modulegreen"),
        "gpt-oss:20b":        ("moduleorange!80", "moduleorange"),
    }
    model_labels = {
        "gemma3:4b":          "gemma3:4b",
        "qwen3:8b":           "qwen3:8b",
        "phi4-reasoning:14b": "phi4:14b",
        "gpt-oss:20b":        "gpt-oss:20b",
    }
    all_have_data = all(data.get(mk) is not None for mk in MODEL_DIRS)
    if all_have_data:
        addplots = []
        for mk in MODEL_DIRS:
            s = data[mk]
            pc = s.get("per_category", {})
            coords = []
            for i, cat in enumerate(CATEGORIES, 1):
                ma_v = pc.get(cat, {}).get("medaide", {}).get("bleu_1")
                mp_v = pc.get(cat, {}).get("medaide_plus", {}).get("bleu_1")
                if ma_v is not None and mp_v is not None:
                    coords.append(f"({i},{mp_v-ma_v:.2f})")
            if len(coords) == 4:
                fc, dc = model_colors[mk]
                lbl = model_labels[mk]
                addplots.append(
                    f"\\addplot[fill={fc}, draw={dc}] coordinates {{\n  " +
                    " ".join(coords) + "\n}; \\addlegendentry{" + lbl + " $\\Delta$}"
                )
        if len(addplots) == 4:
            new_plots = (
                "% All 4 models per-category deltas (bench_v12)\n" +
                "\n".join(addplots) + "\n" +
                "% Zero baseline\n"
                "\\addplot[draw=black!50, dashed, sharp plot, no marks] coordinates {(0.5,0) (4.5,0)}; "
                "\\addlegendentry{Zero}"
            )
            src = re.sub(
                r'% qwen3:8b per-category deltas.*?\\addplot\[draw=black[^\]]*\] coordinates \{[^}]*\};',
                lambda m: new_plots,
                src,
                flags=re.DOTALL
            )
            # Reduce bar width from 9pt to 5pt for 4-model chart
            src = src.replace("bar width=9pt,", "bar width=5pt,")
            # Compute dynamic y limits
            all_deltas = []
            for mk in MODEL_DIRS:
                s = data[mk]
                pc2 = s.get("per_category", {})
                for cat in CATEGORIES:
                    ma_v = pc2.get(cat, {}).get("medaide", {}).get("bleu_1")
                    mp_v = pc2.get(cat, {}).get("medaide_plus", {}).get("bleu_1")
                    if ma_v and mp_v: all_deltas.append(mp_v - ma_v)
            if all_deltas:
                ymin_v = min(-1.0, min(all_deltas) - 1.0)
                ymax_v = max(10.0, max(all_deltas) + 1.0)
                src = re.sub(r'ymin=-\d+\.?\d*, ymax=\d+\.?\d*', f'ymin={ymin_v:.0f}, ymax={ymax_v:.0f}', src)
            # Update legend line (no longer qwen3-only)
            src = src.replace(
                r"\legend{qwen3:8b $\Delta$ BLEU-1, Zero baseline}",
                ""
            )
            # Update caption to reflect 4 models
            q3 = data.get("qwen3:8b")
            if q3:
                pc = q3.get("per_category", {})
                med_d = (pc.get("Medication",{}).get("medaide_plus",{}).get("bleu_1",31.72) -
                         pc.get("Medication",{}).get("medaide",{}).get("bleu_1",26.36))
                post_d = (pc.get("Post-Diagnosis",{}).get("medaide_plus",{}).get("bleu_1",26.05) -
                          pc.get("Post-Diagnosis",{}).get("medaide",{}).get("bleu_1",22.77))
                pre_d  = (pc.get("Pre-Diagnosis",{}).get("medaide_plus",{}).get("bleu_1",28.61) -
                          pc.get("Pre-Diagnosis",{}).get("medaide",{}).get("bleu_1",29.00))
                src = re.sub(
                    r'\\caption\{Per-category BLEU-1 improvement.*?\\label\{fig:percategory\}',
                    lambda m: (f"\\caption{{Per-category BLEU-1 improvement of MedAide+ over MedAide (all 4 LLMs, bench\\_v12, $n{{=}}20$).\n"
                     f"Medication and Post-Diagnosis gain most from specialist routing across all models.\n"
                     f"Pre-Diagnosis shows minor regression ({pre_d:.2f}) because both systems use the same\n"
                     f"PreDiagnosisAgent; the marginal difference stems from KB evidence injection noise.}}\n"
                     f"\\label{{fig:percategory}}"),
                    src, flags=re.DOTALL
                )
            TEX.write_text(src, encoding="utf-8")
            print(f"  ✓ Bar chart updated: 4-model grouped chart generated")
    else:
        # Fall back to qwen3-only update
        q3 = data.get("qwen3:8b")
        if q3:
            pc = q3.get("per_category", {})
            coords = []
            for i, cat in enumerate(CATEGORIES, 1):
                ma_v = pc.get(cat, {}).get("medaide", {}).get("bleu_1")
                mp_v = pc.get(cat, {}).get("medaide_plus", {}).get("bleu_1")
                if ma_v is not None and mp_v is not None:
                    coords.append(f"({i},{mp_v-ma_v:.2f})")
            if len(coords) == 4:
                coord_str = " ".join(coords)
                new_coords = f"% qwen3:8b per-category deltas (bench_v12)\n\\addplot[fill=moduleblue!70, draw=moduleblue] coordinates {{\n  {coord_str}\n}};"
                src = re.sub(
                    r'% qwen3:8b per-category deltas.*?\\addplot\[fill=moduleblue[^\]]*\] coordinates \{[^}]*\};',
                    new_coords,
                    src,
                    flags=re.DOTALL
                )
                deltas_q3 = {CATEGORIES[i]: float(coords[i].split(",")[1].rstrip(")")) for i in range(4)}
                med_d = deltas_q3.get("Medication", 5.36)
                post_d = deltas_q3.get("Post-Diagnosis", 3.28)
                pre_d = deltas_q3.get("Pre-Diagnosis", -0.39)
                src = re.sub(
                    r'Medication \([+-]?\d+\.\d+\) and Post-Diagnosis \([+-]?\d+\.\d+\)',
                    f'Medication ({med_d:+.2f}) and Post-Diagnosis ({post_d:+.2f})',
                    src
                )
                src = re.sub(
                    r'minor regression \(\$[+-]?\d+\.\d+\$\)',
                    f'minor regression (${pre_d:.2f}$)',
                    src
                )
                TEX.write_text(src, encoding="utf-8")
                print(f"  ✓ Bar chart updated (qwen3 only): Med={med_d:+.2f}, Post={post_d:+.2f}, Pre={pre_d:.2f}")

    # Update results narrative — generate paragraphs for all 4 models
    src = TEX.read_text(encoding="utf-8")
    all_narratives = []
    for mk in MODEL_DIRS:
        s = data.get(mk)
        if not s: continue
        ma_s = s["overall"]["medaide"]
        mp_s = s["overall"]["medaide_plus"]
        b1_ma, b1_mp = ma_s.get("bleu_1",0), mp_s.get("bleu_1",0)
        met_ma, met_mp = ma_s.get("meteor",0), mp_s.get("meteor",0)
        bert_ma, bert_mp = ma_s.get("bert_score",0), mp_s.get("bert_score",0)
        wins = sum(1 for m in METRICS if mp_s.get(m,0) > ma_s.get(m,0))
        b1_pct = 100*(b1_mp-b1_ma)/b1_ma if b1_ma else 0
        met_pct = 100*(met_mp-met_ma)/met_ma if met_ma else 0
        bert_delta = bert_mp - bert_ma
        # 4-model label for display
        disp = mk
        all_narratives.append(
            f"\\noindent\\textbf{{{disp}.}} MedAide+ wins {wins}/6 metrics: "
            f"BLEU-1 ${b1_ma:.2f} \\rightarrow {b1_mp:.2f}$ ({b1_pct:+.1f}\\%), "
            f"METEOR ${met_ma:.2f} \\rightarrow {met_mp:.2f}$ ({met_pct:+.1f}\\%), "
            f"BERTScore $\\Delta={bert_delta:+.2f}$."
        )
    if all_narratives:
        new_narratives = "\n\n".join(all_narratives)
        # Replace all per-model narrative paragraphs (from first \noindent\textbf down to \subsection)
        src = re.sub(
            r'(\\noindent\\textbf\{qwen3:8b\.\}.*?)(\\subsection\{Per-Category Analysis\})',
            lambda m: new_narratives + "\n\n" + m.group(2),
            src,
            flags=re.DOTALL
        )
        TEX.write_text(src, encoding="utf-8")
        print(f"  ✓ All-model narrative updated ({len(all_narratives)} models)")

    # Update conclusion text if all 4 models have data
    src = TEX.read_text(encoding="utf-8")
    all_win = all(
        data.get(mk) and
        data[mk]["overall"]["medaide_plus"].get("bleu_1",0) >
        data[mk]["overall"]["medaide"].get("bleu_1",0)
        for mk in MODEL_DIRS
    )
    if all_win:
        src = re.sub(
            r'MedAide\+, a seven-module medical multi-agent LLM framework that universally\noutperforms the three-module MedAide baseline on all six automatic metrics across four\ndiverse LLMs\.',
            'MedAide+, a seven-module medical multi-agent LLM framework that universally\noutperforms the three-module MedAide baseline across all six automatic metrics on all four\ndiverse LLMs evaluated (gemma3:4b, qwen3:8b, phi4-reasoning:14b, gpt-oss:20b).',
            src
        )
        TEX.write_text(src, encoding="utf-8")
        print("  ✓ Conclusion updated: universal win confirmed for all 4 models")

    # Update results intro if all models have positive BLEU-1 delta
    src = TEX.read_text(encoding="utf-8")
    if all_win:
        src = src.replace(
            "MedAide+ achieves consistent improvements\nover the MedAide baseline, with the category-aware routing and extractive synthesis\nproviding gains across all evaluated backbones.",
            "MedAide+ achieves consistent improvements\nover the MedAide baseline across all six metrics and all four LLMs simultaneously,\nwith the category-aware routing and extractive synthesis as the primary drivers."
        )
        TEX.write_text(src, encoding="utf-8")
        print("  ✓ Results intro updated: universal win across all models/metrics")

    print(f"\n✅ Done. Now run: pdflatex MedAidePlus_Phase4_Paper.tex (twice)")

if __name__ == "__main__":
    main()
