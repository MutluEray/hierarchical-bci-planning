"""Day 2: direct low-level BCI control baseline (cursor-style) vs decoder accuracy.

Prints success rate and mean number of BCI commands per task as decoder accuracy drops.
This is the baseline the hierarchical planner has to beat on the same noisy channel.

Usage:
    python3 scripts/02b_test_direct_bci.py
    python3 scripts/02b_test_direct_bci.py --episodes 100 --video
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from bcihp.baselines import DirectBCIController
from bcihp.commands import IntentChannel
from bcihp.utils import load_config, make_env, set_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--video", action="store_true", help="save a clip at accuracy 0.8")
    ap.add_argument("--render", default="auto", choices=["auto", "mujoco", "topdown"])
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    env = make_env(cfg)
    dc = cfg["direct_bci"]

    print(f"Direct BCI control ({dc['mode']} skill, {dc['steps_per_command']} micro-steps per command), "
          f"blocked scenes, budget {env.horizon} micro-steps")
    for acc in (1.0, 0.9, 0.8, 0.7, 0.6):
        ch = IntentChannel(DirectBCIController.vocab_size, acc, neighbor_bias=cfg["commands"]["neighbor_bias"],
                           seed=cfg["seed"])
        ctl = DirectBCIController(ch, mode=dc["mode"], steps_per_command=dc["steps_per_command"],
                                  grip_steps=dc["grip_steps"], seed=cfg["seed"])
        res = []
        for ep in range(args.episodes):
            env.reset(blocked=True, seed=2000 + ep)
            res.append(ctl.run_episode(env))
        print(f"decoder accuracy {acc:.1f} | success {np.mean([r['success'] for r in res]):.2f} | "
              f"commands/task {np.mean([r['n_commands'] for r in res]):.1f} | "
              f"decoder errors/task {np.mean([r['n_errors'] for r in res]):.1f} | "
              f"steps {np.mean([r['steps'] for r in res]):.0f}")

    if args.video:
        import imageio.v2 as imageio
        ch = IntentChannel(DirectBCIController.vocab_size, 0.8, neighbor_bias=cfg["commands"]["neighbor_bias"],
                           seed=cfg["seed"])
        ctl = DirectBCIController(ch, mode=dc["mode"], steps_per_command=dc["steps_per_command"],
                                  grip_steps=dc["grip_steps"], seed=cfg["seed"])
        env.reset(blocked=True, seed=1003)
        frames = []
        orig = env.render
        env.render = lambda: orig(args.render)
        r = ctl.run_episode(env, frames=frames)
        out = ROOT / "results" / "videos"
        out.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out / "direct_bci_acc0.8.mp4", frames, fps=20)
        print(f"saved results/videos/direct_bci_acc0.8.mp4 (success={r['success']}, commands={r['n_commands']})")
