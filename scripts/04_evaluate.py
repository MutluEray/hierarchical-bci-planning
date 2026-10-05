"""Days 5-6: evaluate hierarchical planning vs baselines under a noisy BCI intent channel.

Usage (first look, ~minutes):
    python3 scripts/04_evaluate.py --ckpt checkpoints/cavin_lite_b05.pt
    python3 scripts/04_evaluate.py --ckpt checkpoints/cavin_lite_b05.pt --episodes 40 --acc 0.6 0.8 1.0 --tag b05
    python3 scripts/04_evaluate.py --methods oracle direct --episodes 40      # no model needed
    python3 scripts/04_evaluate.py --ckpt ... --horizon 1                     # ablation: single effect step
    python3 scripts/04_evaluate.py --ckpt ... --no-grasp                      # push-only environment
    python3 scripts/04_evaluate.py --ckpt ... --video                         # save a rollout clip
    python3 scripts/04_evaluate.py --ckpt ... --full                          # full sweep from the config
    python3 scripts/04_evaluate.py --ckpt ... --scene-seed 9000               # fresh scenes for final numbers
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from bcihp import eval as ev
from bcihp.commands import IntentChannel, command_to_goal
from bcihp.utils import load_config, make_env, set_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cavin_lite_b05.pt")
    ap.add_argument("--methods", nargs="+", default=["hier", "oracle", "direct"], choices=["hier", "oracle", "direct"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--acc", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    ap.add_argument("--K", type=int, nargs="+", default=[9])
    ap.add_argument("--scenes", default="both", choices=["both", "blocked", "unblocked"])
    ap.add_argument("--horizon", type=int, default=None, help="override planner.high_level_horizon (ablation)")
    ap.add_argument("--no-grasp", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--scene-seed", type=int, default=5000, help="first scene seed; use a fresh value (e.g. 9000) for final results")
    ap.add_argument("--samples", type=int, default=None, help="planner.cem_samples")
    ap.add_argument("--iters", type=int, default=None, help="planner.cem_iters (1 = prior sampling only, no CEM)")
    ap.add_argument("--exec-steps", type=int, default=None, help="planner.exec_steps (MPC replanning period)")
    ap.add_argument("--approach-weight", type=float, default=None)
    ap.add_argument("--release-weight", type=float, default=None)
    ap.add_argument("--held-weight", type=float, default=None)
    ap.add_argument("--approach-heavy", type=float, default=None, help="approach weight for ungraspable targets only (0 removes the pull to the disc centre)")
    ap.add_argument("--push-weight", type=float, default=None, help="push-geometry cost for ungraspable targets (0 = off)")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    if args.horizon is not None:
        cfg["planner"]["high_level_horizon"] = args.horizon
    for key, val in (("cem_samples", args.samples), ("cem_iters", args.iters), ("exec_steps", args.exec_steps),
                     ("approach_weight", args.approach_weight), ("release_weight", args.release_weight),
                     ("held_weight", args.held_weight), ("push_weight", args.push_weight),
                     ("approach_weight_heavy", args.approach_heavy)):
        if val is not None:
            cfg["planner"][key] = val
    print("planner config:", {k: cfg["planner"].get(k) for k in ("cem_samples", "cem_iters", "cem_elites", "high_level_horizon", "exec_steps",
                                                                  "approach_weight", "approach_weight_heavy", "held_weight", "release_weight", "push_weight")})
    accs, Ks, episodes = args.acc, args.K, args.episodes
    if args.full:
        accs, Ks, episodes = cfg["eval"]["accuracies"], cfg["eval"]["vocab_sizes"], cfg["eval"]["episodes_per_cell"]
    scenes = {"both": [False, True], "blocked": [True], "unblocked": [False]}[args.scenes]

    model = None
    if "hier" in args.methods:
        from bcihp.model import load_model
        from bcihp.planner import TorchLatentModel
        net, ck = load_model(str(ROOT / args.ckpt), args.device)
        model = TorchLatentModel(net, args.device)
        print(f"loaded {args.ckpt} (betas {ck.get('betas')})")

    sfx = f"_{args.tag}" if args.tag else ""
    out_csv = str(ROOT / "results" / f"eval{sfx}.csv")
    rows = ev.sweep(cfg, args.methods, accs, Ks, scenes, episodes, out_csv, planner_model=model,
                    allow_grasp=not args.no_grasp, horizon=args.horizon, scene_seed=args.scene_seed)
    print(ev.summarize(rows))
    print(f"saved {out_csv}")

    if args.video and model is not None:
        import imageio.v2 as imageio
        from bcihp.planner import HierarchicalPlanner
        cfg["env"]["allow_grasp"] = not args.no_grasp
        env = make_env(cfg)
        planner = HierarchicalPlanner(model, cfg, env.zone_centers, env.n_slots, horizon=args.horizon)
        intended = 4
        o, z = command_to_goal(intended, env.cfg.n_objects, env.cfg.n_zones)
        env.reset(target_obj=o, goal_zone=z, blocked=True, seed=5003)
        ch = IntentChannel(max(Ks), 0.8, cfg["commands"]["neighbor_bias"], seed=cfg["seed"])
        frames = []
        r = ev.run_hier_episode(env, planner, ch, intended, cfg["session"], frames)
        out = ROOT / "results" / "videos"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"hier{sfx}.mp4"
        imageio.mimsave(path, frames, fps=20)
        print(f"saved {path} | success={r['success']} commands={r['n_commands']} committed_correct={r['committed_correct']}")
