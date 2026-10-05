"""Day 5-6: why does the learned planner fail?  Failure-mode breakdown + effect-realisation check.

Runs the hierarchical planner with the CORRECT goal (no decoding noise, no confirmation phase) so that
only planning quality is measured.

Results are split by target type when the scene has heavy (push-only) objects.
Pass A (normal MPC, exec_steps from the config): success rate and a breakdown of failed episodes:
    held at end        : the target was still held by the gripper when time ran out (never released)
    never moved        : the target moved < 2 cm
    others disturbed   : target missed the zone and some other object moved > 5 cm
    missed zone        : target moved, nothing else disturbed, but ended outside the zone (median distance shown)
Pass B (full 20-step chunks): compares what the model PREDICTED the chosen effect would do with what the
    generated actions ACTUALLY did (ee / target / other-object position error, and predicted vs achieved
    progress of the target toward the zone). Large gaps = the action generator does not realise the effects
    chosen by the meta-dynamics (hierarchical inconsistency).

Usage:
    python3 scripts/05_diagnose.py --ckpt checkpoints/cavin_lite_b05.pt
    python3 scripts/05_diagnose.py --ckpt checkpoints/cavin_lite_b05.pt --episodes 60 --scene blocked
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from bcihp.commands import command_to_goal
from bcihp.planner import HierarchicalPlanner, TorchLatentModel
from bcihp.utils import load_config, make_env, set_seed


def run_episode(env, planner, obj, zone, n_exec, log=None):
    done = env.t >= env.horizon
    while not done:
        s0 = env.get_state()
        out = planner.plan(s0, [(obj, zone, 1.0)])
        acts = out["actions"][:n_exec]
        pred = planner.model.predict_next(s0[None], out["c1"][None])[0] if log is not None else None
        full = len(acts) >= planner.model_H
        for a in acts:
            _, done = env.step(a)
            if done:
                break
        if log is not None and full and not done:
            ee0, p0, _, _ = env.unpack_state(s0)
            ee1, p1, _, _ = env.unpack_state(env.get_state())
            eep, pp, act, _ = env.unpack_state(pred)
            zc = env.zone_centers[zone]
            others = [j for j in range(env.n_slots) if j != obj and env.active[j]]
            log.append(dict(
                ee_err=np.linalg.norm(eep - ee1), tgt_err=np.linalg.norm(pp[obj] - p1[obj]),
                oth_err=max([np.linalg.norm(pp[j] - p1[j]) for j in others], default=0.0),
                pred_prog=np.linalg.norm(p0[obj] - zc) - np.linalg.norm(pp[obj] - zc),
                act_prog=np.linalg.norm(p0[obj] - zc) - np.linalg.norm(p1[obj] - zc)))
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cavin_lite_b05.pt")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--scene", default="both", choices=["both", "blocked", "unblocked"])
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--samples", type=int, default=None, help="planner.cem_samples")
    ap.add_argument("--iters", type=int, default=None, help="planner.cem_iters (1 = prior sampling only, no CEM)")
    ap.add_argument("--exec-steps", type=int, default=None, help="planner.exec_steps (MPC replanning period)")
    ap.add_argument("--approach-weight", type=float, default=None)
    ap.add_argument("--release-weight", type=float, default=None)
    ap.add_argument("--held-weight", type=float, default=None)
    ap.add_argument("--approach-heavy", type=float, default=None, help="approach weight for ungraspable targets only (0 removes the pull to the disc centre)")
    ap.add_argument("--push-weight", type=float, default=None, help="push-geometry cost for ungraspable targets (0 = off)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    for key, val in (("cem_samples", args.samples), ("cem_iters", args.iters), ("exec_steps", args.exec_steps),
                     ("approach_weight", args.approach_weight), ("release_weight", args.release_weight),
                     ("held_weight", args.held_weight), ("push_weight", args.push_weight),
                     ("approach_weight_heavy", args.approach_heavy)):
        if val is not None:
            cfg["planner"][key] = val
    print("planner config:", {k: cfg["planner"].get(k) for k in ("cem_samples", "cem_iters", "cem_elites", "high_level_horizon", "exec_steps",
                                                                  "approach_weight", "approach_weight_heavy", "held_weight", "release_weight", "push_weight")})
    from bcihp.model import load_model
    net, ck = load_model(str(ROOT / args.ckpt), args.device)
    model = TorchLatentModel(net, args.device)
    env = make_env(cfg)
    planner = HierarchicalPlanner(model, cfg, env.zone_centers, env.n_slots, horizon=args.horizon)
    planner.model_H = net.H
    scenes = {"both": [False, True], "blocked": [True], "unblocked": [False]}[args.scene]
    K = cfg["commands"]["vocab_size"]

    for blocked in scenes:
        types = ["light", "heavy"] if env.heavy.any() else ["light"]
        blank = lambda: {"success": 0, "held at end": 0, "never moved": 0, "others disturbed": 0, "missed zone": 0}
        cats, counts, miss_d, steps = {t: blank() for t in types}, {t: 0 for t in types}, {t: [] for t in types}, {t: [] for t in types}
        for ep in range(args.episodes):
            o, z = command_to_goal(int(np.random.default_rng(10_000 * K + ep).integers(K)), env.cfg.n_objects, env.cfg.n_zones)
            env.reset(target_obj=o, goal_zone=z, blocked=blocked, seed=5000 + ep)
            typ = "heavy" if env.heavy[o] else "light"
            counts[typ] += 1
            p_init = env.obj_positions().copy()
            planner.reseed(1000 + ep)
            run_episode(env, planner, o, z, planner.exec_steps)
            p_end = env.obj_positions()
            steps[typ].append(env.t)
            c = cats[typ]
            if env.task_success(o, z):
                c["success"] += 1
                continue
            others = [j for j in range(env.n_slots) if j != o and env.active[j]]
            if env.holding == o:
                c["held at end"] += 1
            elif np.linalg.norm(p_end[o] - p_init[o]) < 0.02:
                c["never moved"] += 1
            elif any(np.linalg.norm(p_end[j] - p_init[j]) > 0.05 for j in others):
                c["others disturbed"] += 1
            else:
                c["missed zone"] += 1
                miss_d[typ].append(np.linalg.norm(p_end[o] - env.zone_centers[z]))
        print(f"\n=== {'blocked' if blocked else 'unblocked'} scenes, {args.episodes} episodes, correct goal, exec_steps={planner.exec_steps} ===")
        for typ in types:
            n = max(counts[typ], 1)
            label = {"light": "light targets (grasp)", "heavy": "heavy targets (push)"}[typ] if len(types) > 1 else "all targets"
            print(f"  -- {label}: n={counts[typ]}, mean steps {np.mean(steps[typ]) if steps[typ] else 0:.0f} / {env.horizon}")
            for k, v in cats[typ].items():
                print(f"     {k:18s} {v:3d}  ({v / n:.0%})")
            if miss_d[typ]:
                print(f"     missed-zone distance: median {np.median(miss_d[typ]) * 100:.1f} cm (success needs < 3.5 cm)")

        # Pass B: effect realisation with full chunks, split by target type
        saved = planner.exec_steps
        planner.exec_steps = net.H
        log, succ = [], {t: 0 for t in types}
        for ep in range(args.episodes):
            o, z = command_to_goal(int(np.random.default_rng(10_000 * K + ep).integers(K)), env.cfg.n_objects, env.cfg.n_zones)
            env.reset(target_obj=o, goal_zone=z, blocked=blocked, seed=5000 + ep)
            typ = "heavy" if env.heavy[o] else "light"
            planner.reseed(1000 + ep)
            start = len(log)
            run_episode(env, planner, o, z, net.H, log)
            for r in log[start:]:
                r["typ"] = typ
            succ[typ] += int(env.task_success(o, z))
        planner.exec_steps = saved
        for typ in types:
            sub = [r for r in log if r["typ"] == typ]
            if not sub:
                continue
            g = lambda k: np.array([r[k] for r in sub])
            big = g("pred_prog") > 0.02
            print(f"  [full 20-step chunks, {typ}] success {succ[typ] / max(counts[typ], 1):.2f} | {len(sub)} planned chunks | prediction vs reality "
                  f"(median cm): ee {np.median(g('ee_err')) * 100:.1f}, target {np.median(g('tgt_err')) * 100:.1f}, others {np.median(g('oth_err')) * 100:.1f}")
            if big.any():
                realised = g("act_prog")[big] > 0.01
                print(f"     chunks predicted to advance the target >2 cm: {big.mean():.0%} | of those, actually advanced >1 cm: {realised.mean():.0%} "
                      f"| achieved {g('act_prog')[big].mean() * 100:.1f} cm vs predicted {g('pred_prog')[big].mean() * 100:.1f} cm")
