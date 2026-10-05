"""Day 2: generate the self-supervised macro-transition dataset.

Usage:
    python3 scripts/02_collect_data.py              # n_transitions from configs/default.yaml
    python3 scripts/02_collect_data.py --n 2000     # quick test
    python3 scripts/02_collect_data.py --no-grasp --out data/transitions_push.npz   # push-only dataset
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcihp import datagen
from bcihp.utils import load_config, make_env, set_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="number of windows (default: cfg.data.n_transitions)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-grasp", action="store_true", help="push-only world (use with --out data/transitions_push.npz)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    dc = cfg["data"]
    if args.no_grasp:
        cfg["env"]["allow_grasp"] = False
    env = make_env(cfg)
    n = args.n or dc["n_transitions"]
    out = Path(args.out or ROOT / dc["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = datagen.collect(env, n_transitions=n, stride=dc["stride"], expert_fraction=dc["expert_fraction"],
                           noise=dc["action_noise"], seed=cfg["seed"])
    datagen.save(data, str(out))
    print(datagen.summarize(data))
    print(f"state_dim={int(data['state_dim'])}  action shape per window={data['actions'].shape[1:]}")
    print(f"saved {out}  ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
