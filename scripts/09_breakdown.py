"""Per-skill breakdown of an evaluation CSV, without re-running anything.

The evaluation draws the intended command for episode `ep` from a deterministic RNG, so the target object
(and therefore light/heavy) can be reconstructed from the CSV. Command k selects object k % n_objects.

Usage (mixed-skill scene):
    BCIHP_CONFIG=configs/mixed.yaml python3 scripts/09_breakdown.py --csv results/eval_mixed_final.csv --scene-seed 12000
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from bcihp.utils import load_config


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/eval_mixed_final.csv")
    ap.add_argument("--scene-seed", type=int, default=12000)
    args = ap.parse_args()

    cfg = load_config()
    n_obj, n_heavy = cfg["env"]["n_objects"], cfg["env"].get("n_heavy", 0)
    with open(ROOT / args.csv) as f:
        rows = list(csv.DictReader(f))
    kmax = max(int(r["K"]) for r in rows)
    groups = {}
    for r in rows:
        kk = int(r["K"]) if r["method"] != "direct" else kmax
        intended = int(np.random.default_rng(10_000 * kk + (args.scene_seed - 5000) + int(r["episode"])).integers(kk))
        heavy = (intended % n_obj) >= n_obj - n_heavy
        key = (r["method"], float(r["accuracy"]), int(r["blocked"]), "heavy" if heavy else "light")
        groups.setdefault(key, []).append(int(r["success"]))

    print(f"{'method':7} {'acc':>4} {'scene':>9} | {'light (grasp)':>22} | {'heavy (push)':>22}")
    for m in ("hier", "oracle", "direct"):
        for blocked in (0, 1):
            for acc in sorted({k[1] for k in groups}, reverse=True):
                cells = []
                for typ in ("light", "heavy"):
                    v = groups.get((m, acc, blocked, typ), [])
                    if v:
                        p, lo, hi = wilson(sum(v), len(v))
                        cells.append(f"{p:.2f} ({lo:.2f}-{hi:.2f}) n={len(v):3d}")
                    else:
                        cells.append("-")
                if any(c != "-" for c in cells):
                    print(f"{m:7} {acc:4.1f} {'blocked' if blocked else 'unblocked':>9} | {cells[0]:>22} | {cells[1]:>22}")
