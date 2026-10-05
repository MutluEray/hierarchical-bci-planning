"""Data aggregation round: add the learned planner's own experience to the training data.

Why: the model was trained on oracle and random-exploration data, but the planner visits other states (discs pushed
into corners, objects left held, awkward contact poses), where the model's predictions are over-optimistic. This script
runs the current planner on random scenes/tasks, optionally lets the scripted oracle take over after 10-80 steps
(so the data contain expert recovery from the learner's own states), and appends the resulting windows to the dataset.
Re-train on the merged data with 03_train.py afterwards. It can be repeated (use the new checkpoint and merged data).

Usage (default scene):
    python3 scripts/11_aggregate.py --ckpt checkpoints/cavin_lite_b05.pt --out data/transitions_agg.npz
    python3 scripts/03_train.py --data data/transitions_agg.npz --beta-c 0.5 --beta-z 0.1 --tag agg
Usage (mixed-skill scene):
    BCIHP_CONFIG=configs/mixed.yaml python3 scripts/11_aggregate.py --ckpt checkpoints/cavin_lite_mixed.pt --out data/transitions_mixed_agg.npz
    BCIHP_CONFIG=configs/mixed.yaml python3 scripts/03_train.py --data data/transitions_mixed_agg.npz --beta-c 0.5 --beta-z 0.1 --tag mixed_agg
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcihp import datagen
from bcihp.baselines import ScriptedSymbolicPlanner
from bcihp.model import load_model
from bcihp.planner import HierarchicalPlanner, TorchLatentModel
from bcihp.utils import load_config, make_env, set_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=None, help="existing dataset to extend (default: cfg.data.out_path)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40000, help="number of new on-policy windows")
    ap.add_argument("--handover", type=float, default=0.5, help="fraction of tasks where the oracle takes over after 10-80 steps")
    ap.add_argument("--noise", type=float, default=0.1, help="Gaussian noise on the planner's xy actions")
    ap.add_argument("--seed", type=int, default=777, help="collection seed (use a new one for every round)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    env = make_env(cfg)
    net, _ = load_model(str(ROOT / args.ckpt), args.device)
    planner = HierarchicalPlanner(TorchLatentModel(net, args.device), cfg, env.zone_centers, env.n_slots)
    oracle = ScriptedSymbolicPlanner(mode="auto", seed=args.seed)

    t0 = time.time()
    new, stats = datagen.collect_onpolicy(env, planner, oracle, args.n, stride=cfg["data"]["stride"],
                                          handover_fraction=args.handover, noise=args.noise, seed=args.seed)
    base = datagen.load(str(ROOT / (args.base or cfg["data"]["out_path"])))
    merged = datagen.merge(base, new)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    datagen.save(merged, str(out))
    pure = max(stats["tasks_pure"], 1)
    print(f"planner on random tasks (noise {args.noise}, no handover): {stats['success_pure']}/{stats['tasks_pure']} "
          f"= {stats['success_pure'] / pure:.2f} success | handover tasks: {stats['tasks_handover']}")
    print("new data:   ", datagen.summarize(new))
    print("merged data:", datagen.summarize(merged))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
