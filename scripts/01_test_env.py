"""Day 1 smoke test: run the scripted oracle (push / grasp / mixed) on random scenes and save demo clips.

Usage:
    python3 scripts/01_test_env.py                  # 30 episodes per cell + two demo videos
    python3 scripts/01_test_env.py --no-video       # skip rendering
    python3 scripts/01_test_env.py --render topdown # force the matplotlib renderer
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from bcihp.baselines import ScriptedSymbolicPlanner
from bcihp.env import SceneConfig, TabletopEnv
from bcihp.utils import load_config, set_seed


def make_env(cfg: dict) -> TabletopEnv:
    keys = SceneConfig.__dataclass_fields__
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["env"].items() if k in keys}
    return TabletopEnv(SceneConfig(**kwargs), seed=cfg["seed"])


def save_clip(env, expert, mode, seed, name, render, out_dir):
    import imageio.v2 as imageio
    env.reset(blocked=True, seed=seed)
    frames = []
    orig = env.render
    env.render = lambda: orig(render)
    res = expert.run_episode(env, frames=frames, mode=mode)
    env.render = orig
    path = out_dir / f"{name}.mp4"
    try:
        imageio.mimsave(path, frames, fps=20)
    except Exception as e:  # noqa: BLE001
        path = out_dir / f"{name}.gif"
        print(f"mp4 writer unavailable ({e}); writing gif instead")
        imageio.mimsave(path, frames, fps=20)
    imageio.imwrite(out_dir / f"{name}_last_frame.png", frames[-1])
    print(f"saved {path} ({len(frames)} frames, success={res['success']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--render", default="auto", choices=["auto", "mujoco", "topdown"])
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    env = make_env(cfg)

    for mode in ("push", "grasp", "mixed"):
        expert = ScriptedSymbolicPlanner(mode=mode, seed=cfg["seed"])
        for blocked in (False, True):
            res = []
            for ep in range(args.episodes):
                env.reset(blocked=blocked, seed=1000 + ep)
                res.append(expert.run_episode(env))
            print(f"mode={mode:5} blocked={blocked!s:5} | oracle success {np.mean([r['success'] for r in res]):.2f} | "
                  f"mean steps {np.mean([r['steps'] for r in res]):.0f} / {env.horizon}")

    if not args.no_video:
        out_dir = ROOT / "results" / "videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        expert = ScriptedSymbolicPlanner(seed=cfg["seed"])
        save_clip(env, expert, "push", 1003, "demo_push_blocked", args.render, out_dir)
        save_clip(env, expert, "grasp", 1003, "demo_grasp_blocked", args.render, out_dir)
