"""Evaluation harness (DAY 5-6).

Methods
  hier    : committed high-level command -> CAVIN-lite hierarchical planner (learned)
  oracle  : same BCI session, but executed by the scripted oracle (upper bound: isolates decoding errors)
  direct  : cursor-style low-level BCI control with a noisy 5-command decoder (needs weld-grasp)

BCI session (hier / oracle): the user re-issues one intended command; the system commits when the Bayesian
belief reaches cfg.session.commit_threshold; waiting costs `cmd_interval` micro-steps per decoded command and
counts against the same episode budget as everything else. Reported: success, #BCI commands, steps,
and whether the committed intent was correct.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np

from .baselines import DirectBCIController, ScriptedSymbolicPlanner
from .commands import IntentChannel, IntentSession, command_to_goal
from .utils import make_env


def _idle(env, n: int, frames=None) -> None:
    for _ in range(n):
        if env.t >= env.horizon:
            return
        env.step(np.array([0.0, 0.0, -1.0]))
        if frames is not None:
            frames.append(env.render())


def _confirm(env, session: IntentSession, interval: int, frames=None) -> int:
    """Issue commands (each followed by `interval` idle micro-steps) until the session commits."""
    while session.committed is None and env.t < env.horizon:
        session.issue()
        _idle(env, interval, frames)
    return session.committed if session.committed is not None else session.map_command()


def run_hier_episode(env, planner, channel, intended: int, scfg: dict, frames=None) -> dict:
    n_obj, n_zone = env.cfg.n_objects, env.cfg.n_zones
    true_obj, true_zone = command_to_goal(intended, n_obj, n_zone)
    env.target_obj, env.goal_zone = true_obj, true_zone
    session = IntentSession(channel, intended, scfg["commit_threshold"], scfg["max_commands"])
    if frames is not None:
        frames.append(env.render())
    cmd = _confirm(env, session, scfg["cmd_interval"], frames)
    obj, zone = command_to_goal(cmd, n_obj, n_zone)
    done = env.t >= env.horizon
    while not done:
        out = planner.plan(env.get_state(), [(obj, zone, 1.0)])
        for a in out["actions"]:
            _, done = env.step(a)
            if frames is not None:
                frames.append(env.render())
            if done or env.t >= env.horizon:
                done = True
                break
    return dict(success=env.task_success(true_obj, true_zone), steps=env.t, n_commands=session.n_commands,
                committed_correct=int(cmd == intended))


def run_oracle_episode(env, oracle: ScriptedSymbolicPlanner, channel, intended: int, scfg: dict, frames=None) -> dict:
    n_obj, n_zone = env.cfg.n_objects, env.cfg.n_zones
    true_obj, true_zone = command_to_goal(intended, n_obj, n_zone)
    env.target_obj, env.goal_zone = true_obj, true_zone
    session = IntentSession(channel, intended, scfg["commit_threshold"], scfg["max_commands"])
    cmd = _confirm(env, session, scfg["cmd_interval"], frames)
    obj, zone = command_to_goal(cmd, n_obj, n_zone)
    oracle.run_episode(env, obj, zone, mode="auto", frames=frames)
    return dict(success=env.task_success(true_obj, true_zone), steps=env.t, n_commands=session.n_commands,
                committed_correct=int(cmd == intended))


def run_direct_episode(env, ctl: DirectBCIController, intended: int) -> dict:
    true_obj, true_zone = command_to_goal(intended, env.cfg.n_objects, env.cfg.n_zones)
    r = ctl.run_episode(env, true_obj, true_zone)
    return dict(success=r["success"], steps=r["steps"], n_commands=r["n_commands"], committed_correct=float("nan"))


def sweep(cfg: dict, methods, accuracies, vocab_sizes, scenes, episodes: int, out_csv: str, planner_model=None,
          allow_grasp: bool = True, horizon: int | None = None, progress: bool = True,
          scene_seed: int = 5000) -> list:
    cfg = copy.deepcopy(cfg)
    cfg["env"]["allow_grasp"] = allow_grasp
    env = make_env(cfg)
    scfg, nb = cfg["session"], cfg["commands"]["neighbor_bias"]
    if "direct" in methods and not allow_grasp:
        print("note: direct control needs weld-grasp; skipping it in push-only mode")
        methods = [m for m in methods if m != "direct"]
    planner = None
    if "hier" in methods:
        from .planner import HierarchicalPlanner
        planner = HierarchicalPlanner(planner_model, cfg, env.zone_centers, env.n_slots, horizon=horizon)
    oracle = ScriptedSymbolicPlanner(mode="mixed", seed=cfg["seed"])

    jobs = []
    for method in methods:
        Ks = [0] if method == "direct" else vocab_sizes      # direct control does not depend on the vocabulary
        for K in Ks:
            for acc in accuracies:
                for blocked in scenes:
                    jobs.append((method, K, acc, blocked))
    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(jobs) * episodes, desc="evaluating", unit="ep")
        except ImportError:
            pass

    rows = []
    for method, K, acc, blocked in jobs:
        for ep in range(episodes):
            kk = K if method != "direct" else max(vocab_sizes)
            intended = int(np.random.default_rng(10_000 * kk + (scene_seed - 5000) + ep).integers(kk))
            o, z = command_to_goal(intended, env.cfg.n_objects, env.cfg.n_zones)
            env.reset(target_obj=o, goal_zone=z, blocked=blocked, seed=scene_seed + ep)
            seed = cfg["seed"] + ep
            if method == "direct":
                ch = IntentChannel(DirectBCIController.vocab_size, acc, nb, seed=seed)
                ctl = DirectBCIController(ch, mode=cfg["direct_bci"]["mode"], steps_per_command=cfg["direct_bci"]["steps_per_command"],
                                          grip_steps=cfg["direct_bci"]["grip_steps"], seed=seed)
                r = run_direct_episode(env, ctl, intended)
            else:
                ch = IntentChannel(K, acc, nb, seed=seed)
                if method == "hier":
                    planner.reseed(1000 + ep)
                    r = run_hier_episode(env, planner, ch, intended, scfg)
                else:
                    r = run_oracle_episode(env, oracle, ch, intended, scfg)
            rows.append(dict(method=method, K=K, accuracy=acc, blocked=int(blocked), allow_grasp=int(allow_grasp),
                             episode=ep, success=int(r["success"]), steps=r["steps"], n_commands=r["n_commands"],
                             committed_correct=r["committed_correct"]))
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def summarize(rows: list) -> str:
    keys = sorted({(r["method"], r["K"], r["accuracy"], r["blocked"]) for r in rows},
                  key=lambda k: (k[3], k[0], k[1], -k[2]))
    lines = [f"{'method':7} {'K':>3} {'acc':>5} {'scene':>9} | {'success':>7} {'cmds/task':>9} {'steps':>6} {'commit ok':>9} | n"]
    for k in keys:
        sub = [r for r in rows if (r["method"], r["K"], r["accuracy"], r["blocked"]) == k]
        cc = [r["committed_correct"] for r in sub if r["committed_correct"] == r["committed_correct"]]
        lines.append(f"{k[0]:7} {k[1] if k[1] else '-':>3} {k[2]:5.2f} {'blocked' if k[3] else 'unblocked':>9} | "
                     f"{np.mean([r['success'] for r in sub]):7.2f} {np.mean([r['n_commands'] for r in sub]):9.1f} "
                     f"{np.mean([r['steps'] for r in sub]):6.0f} {(f'{np.mean(cc):.2f}' if cc else '-'):>9} | {len(sub)}")
    return "\n".join(lines)
