"""Day 6-7: figures and results table from the evaluation CSVs.

Reads (missing files are skipped):
    results/eval_final.csv        main sweep           (python3 scripts/04_evaluate.py --ckpt ... --full --scene-seed 9000 --episodes ... --tag final)
    results/eval_abl_<name>.csv   ablations (accuracy 1.0, K=9)   names: h1 h2 h3 h4 oldcost prior exec5
Writes PNGs to docs/figures/ and docs/results_table.md.

Usage:
    python3 scripts/06_make_figures.py
    python3 scripts/06_make_figures.py --tag mixed_final     # figures for the mixed-skill scene
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = ROOT / "docs" / "figures"
COL = {"hier": "#d62728", "oracle": "#2ca02c", "direct": "#1f77b4"}
NAME = {"hier": "Hierarchical planner (CAVIN-lite)", "oracle": "Oracle executor (same BCI session)",
        "direct": "Direct low-level BCI control"}
ABL = [("h1", "horizon 1"), ("h2", "horizon 2"), ("h3", "horizon 3 (default)"), ("h4", "horizon 4"),
       ("oldcost", "no cost shaping"), ("prior", "prior sampling only (no CEM)"), ("exec5", "replan every 5 steps")]


def read(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("K", "blocked", "allow_grasp", "episode", "success", "steps", "n_commands"):
            r[k] = int(float(r[k]))
        r["accuracy"] = float(r["accuracy"])
        r["committed_correct"] = float(r["committed_correct"])
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, min(p, max(0.0, c - h)), max(p, min(1.0, c + h))   # clipped so that lo <= p <= hi despite rounding


def wolpaw_bits(N, P):
    """Wolpaw information transfer per selection (bits) for N equiprobable classes at accuracy P.
    Standard approximation (assumes errors spread uniformly over the other classes)."""
    P = min(max(P, 1.0 / N), 1.0)
    if P >= 1.0 - 1e-12:
        return float(np.log2(N))
    return float(np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))


def bits_per_task(method, acc, K, commands):
    """User input needed per task = commands issued x bits per command (direct control uses 5 command classes)."""
    return commands * wolpaw_bits(5 if method == "direct" else K, acc)


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def rate(rows):
    return wilson(sum(r["success"] for r in rows), len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final", help="reads results/eval_<tag>.csv; non-default tags get a figure suffix and skip ablations")
    args = ap.parse_args()
    sfx = "" if args.tag == "final" else f"_{args.tag}"
    FIG.mkdir(parents=True, exist_ok=True)
    main_rows = read(ROOT / "results" / f"eval_{args.tag}.csv")
    lines = []
    if main_rows:
        Kmax = max(r["K"] for r in main_rows)
        accs = sorted({r["accuracy"] for r in main_rows})
        episodes = len({r["episode"] for r in main_rows})

        # 1) success vs decoder accuracy
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for a, blocked in zip(ax, (0, 1)):
            for m in ("hier", "oracle", "direct"):
                pts = []
                for acc in accs:
                    sub = sel(main_rows, method=m, accuracy=acc, blocked=blocked)
                    if m != "direct":
                        sub = [r for r in sub if r["K"] == Kmax]
                    if sub:
                        pts.append((acc,) + rate(sub))
                if pts:
                    x, p, lo, hi = zip(*pts)
                    a.errorbar(x, p, yerr=[np.array(p) - np.array(lo), np.array(hi) - np.array(p)], marker="o", capsize=3,
                               color=COL[m], label=NAME[m])
            a.set_title("blocked scenes" if blocked else "unblocked scenes")
            a.set_xlabel("simulated decoder accuracy")
            a.set_ylim(-0.02, 1.05)
            a.grid(alpha=0.3)
        ax[0].set_ylabel("task success rate (95% Wilson CI)")
        ax[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Success vs. decoder accuracy (K={Kmax} commands, {episodes} episodes per point)")
        fig.tight_layout()
        fig.savefig(FIG / f"success_vs_accuracy{sfx}.png", dpi=150)
        plt.close(fig)

        # 2) commands per task vs accuracy
        fig, a = plt.subplots(figsize=(6, 4))
        for m in ("hier", "oracle", "direct"):
            ys = []
            for acc in accs:
                sub = sel(main_rows, method=m, accuracy=acc)
                if m != "direct":
                    sub = [r for r in sub if r["K"] == Kmax]
                if sub:
                    ys.append((acc, np.mean([r["n_commands"] for r in sub])))
            if ys:
                a.plot(*zip(*ys), marker="o", color=COL[m], label=NAME[m])
        a.set_yscale("log")
        a.set_xlabel("simulated decoder accuracy")
        a.set_ylabel("BCI commands per task (log)")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"commands_vs_accuracy{sfx}.png", dpi=150)
        plt.close(fig)

        # 2b) user effort: bits of intent the user must transmit per task vs success
        fig, a = plt.subplots(figsize=(6.5, 4.5))
        for m in ("hier", "direct"):
            pts = []
            for acc in accs:
                sub = sel(main_rows, method=m, accuracy=acc)
                if m != "direct":
                    sub = [r for r in sub if r["K"] == Kmax]
                if sub:
                    cmds = np.mean([r["n_commands"] for r in sub])
                    pts.append((bits_per_task(m, acc, Kmax, cmds), np.mean([r["success"] for r in sub]), acc))
            if pts:
                x, y, ac = zip(*pts)
                a.plot(x, y, marker="o", color=COL[m], label=NAME[m])
                for xi, yi, ai in zip(x, y, ac):
                    a.annotate(f"{ai:.1f}", (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7, color=COL[m])
        a.set_xscale("log")
        a.set_xlabel("user input per task (bits, Wolpaw estimate; labels = decoder accuracy)")
        a.set_ylabel("task success rate")
        a.set_ylim(-0.02, 1.05)
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG / f"effort_vs_success{sfx}.png", dpi=150)
        plt.close(fig)

        # 3) success vs vocabulary size (hier / oracle) at the accuracy closest to 0.8
        Ks = sorted({r["K"] for r in main_rows if r["K"] > 0})
        if len(Ks) > 1:
            acc0 = min(accs, key=lambda x: abs(x - 0.8))
            fig, a = plt.subplots(figsize=(6, 4))
            for m in ("hier", "oracle"):
                pts = [(K,) + rate(sel(main_rows, method=m, accuracy=acc0, K=K)) for K in Ks if sel(main_rows, method=m, accuracy=acc0, K=K)]
                if pts:
                    x, p, lo, hi = zip(*pts)
                    a.errorbar(x, p, yerr=[np.array(p) - np.array(lo), np.array(hi) - np.array(p)], marker="o", capsize=3,
                               color=COL[m], label=NAME[m])
            a.set_xlabel("command vocabulary size K (= number of distinct tasks)")
            a.set_ylabel("task success rate")
            a.set_title(f"Decoder accuracy {acc0:.1f}")
            a.set_ylim(-0.02, 1.05)
            a.set_xticks(Ks)
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG / f"success_vs_vocab{sfx}.png", dpi=150)
            plt.close(fig)

        # results table
        lines += [f"### Main results (K={Kmax}, {episodes} episodes per cell)", "",
                  "| method | accuracy | scene | success (95% CI) | BCI commands/task | user input (bits/task) | steps |", "|---|---|---|---|---|---|---|"]
        for blocked in (0, 1):
            for m in ("hier", "oracle", "direct"):
                for acc in accs:
                    sub = sel(main_rows, method=m, accuracy=acc, blocked=blocked)
                    if m != "direct":
                        sub = [r for r in sub if r["K"] == Kmax]
                    if sub:
                        p, lo, hi = rate(sub)
                        cm = np.mean([r["n_commands"] for r in sub])
                        lines.append(f"| {m} | {acc:.1f} | {'blocked' if blocked else 'unblocked'} | {p:.2f} ({lo:.2f}-{hi:.2f}) | "
                                     f"{cm:.1f} | {bits_per_task(m, acc, Kmax, cm):.1f} | {np.mean([r['steps'] for r in sub]):.0f} |")
        lines.append("")

    # 4) ablations
    abl = []
    for key, label in ABL:
        rows = read(ROOT / "results" / f"eval_abl_{key}.csv") if sfx == "" else None
        if rows:
            abl.append((label, rows))
    if abl:
        fig, a = plt.subplots(figsize=(9, 4))
        w = 0.38
        xs = np.arange(len(abl))
        for off, blocked, name, c in ((-w / 2, 0, "unblocked", "#9ecae1"), (w / 2, 1, "blocked", "#3182bd")):
            ps, los, his = [], [], []
            for _, rows in abl:
                p, lo, hi = rate(sel(rows, blocked=blocked))
                ps.append(p); los.append(p - lo); his.append(hi - p)
            a.bar(xs + off, ps, w, yerr=[los, his], capsize=3, color=c, label=name)
        a.set_xticks(xs)
        a.set_xticklabels([l for l, _ in abl], rotation=20, ha="right", fontsize=8)
        a.set_ylabel("task success rate (correct goal)")
        a.set_ylim(0, 1.05)
        a.legend(fontsize=8)
        a.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIG / f"ablations{sfx}.png", dpi=150)
        plt.close(fig)
        lines += ["### Ablations (decoder accuracy 1.0, K=9)", "", "| variant | unblocked | blocked |", "|---|---|---|"]
        for label, rows in abl:
            u, b = rate(sel(rows, blocked=0)), rate(sel(rows, blocked=1))
            lines.append(f"| {label} | {u[0]:.2f} ({u[1]:.2f}-{u[2]:.2f}) | {b[0]:.2f} ({b[1]:.2f}-{b[2]:.2f}) |")
        lines.append("")

    if lines:
        (ROOT / "docs" / f"results_table{sfx}.md").write_text("\n".join(lines))
        print("\n".join(lines))
        print("figures written to docs/figures/")
    else:
        print("no result CSVs found (expected results/eval_final.csv and/or results/eval_abl_*.csv)")


if __name__ == "__main__":
    main()
