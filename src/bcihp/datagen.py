"""Self-supervised data generation (DAY 2).

Collects task-agnostic macro-transitions (s_t, a_{t:t+H}, s_{t+H}) with H = cfg.env.micro_steps_per_macro.
No command / goal labels are stored: as in CAVIN, the model learns effects and motions from interaction
only; commands enter at planning time.

Two data sources per episode (mixed by `expert_fraction`):
  * expert      : the scripted oracle, chaining several random (target, zone) tasks until the horizon;
                  gives purposeful multi-step behaviour (clear blocker -> push target, grasp -> carry -> place).
  * exploration : random primitives (random pushes, random grasp-and-carry placements, random wandering)
                  executed back to back; gives broad coverage of effects the oracle never produces.
Each full episode is cut into overlapping windows every `stride` micro-steps.

Saved arrays (npz):
  states (N, sd), actions (N, H, 3), next_states (N, sd), episode (N,), kind (N,) 0=expert 1=explore,
  obj_disp (N,)  total object displacement inside the window (m)  -> used to weight/diagnose effects,
  used_grip (N,) 1 if the window contains a held object, plus scalar metadata.
"""
from __future__ import annotations

import numpy as np

from .baselines import ScriptedSymbolicPlanner
from .env import EE_LIM, TabletopEnv


# ------------------------------------------------------------------------------ recording
def _record(env: TabletopEnv, fn):
    """Run fn() while recording (state_t, executed action_t). Returns (states (T+1, sd), actions (T, 3))."""
    S, A = [env.get_state()], []
    orig = env.step

    def rec(action):
        a = np.asarray(action, dtype=float).ravel()
        mv = a[:2]
        n = np.linalg.norm(mv)
        if n > 1.0:
            mv = mv / n
        if a.size >= 3:
            grip = 1.0 if a[2] > 0 else -1.0
        else:
            grip = 1.0 if (env.holding >= 0 and not env._releasing) else -1.0
        out = orig(action)
        A.append([mv[0], mv[1], grip])
        S.append(out[0])
        return out

    env.step = rec
    try:
        fn()
    finally:
        del env.step
    return np.array(S, dtype=np.float32), np.array(A, dtype=np.float32)


# ------------------------------------------------------------------------------ episode types
def _expert_episode(env: TabletopEnv, planner: ScriptedSymbolicPlanner, rng, noise: float, max_tasks: int = 6):
    for _ in range(max_tasks):
        if env.t >= env.horizon:
            break
        env.target_obj = int(rng.integers(env.cfg.n_objects))
        env.goal_zone = int(rng.integers(env.cfg.n_zones))
        planner.run_episode(env, mode="mixed", max_replans=4)


def _random_primitives(env: TabletopEnv, planner: ScriptedSymbolicPlanner, rng):
    act = np.flatnonzero(env.active)
    kind = rng.choice(["push", "carry", "wander"], p=[0.4, 0.3, 0.3])
    pos = env.obj_positions()
    if kind == "push":
        i = int(rng.choice(act))
        ang = rng.uniform(0, 2 * np.pi)
        dest = np.clip(pos[i] + rng.uniform(0.06, 0.25) * np.array([np.cos(ang), np.sin(ang)]), -0.265, 0.265)
        return planner._push_waypoints(env, i, dest)
    if kind == "carry" and env.cfg.allow_grasp:
        i = int(rng.choice(act))
        if rng.random() < 0.5:
            dest = env.zone_centers[int(rng.integers(env.cfg.n_zones))]
        else:
            dest = rng.uniform(-0.25, 0.25, 2)
        return planner.carry_primitives(env, i, dest)
    n_pts = int(rng.integers(1, 4))
    return [("move", rng.uniform(-EE_LIM + 0.01, EE_LIM - 0.01, 2)) for _ in range(n_pts)]


def _explore_episode(env: TabletopEnv, planner: ScriptedSymbolicPlanner, rng, noise: float):
    while env.t < env.horizon:
        prims = _random_primitives(env, planner, rng)
        planner.execute(env, prims, max_steps=env.horizon - env.t, stop_on_done=False, noise=noise)


# ------------------------------------------------------------------------------ windows
def _windows(S: np.ndarray, A: np.ndarray, H: int, stride: int, n_slots: int):
    out = []
    T = len(A)
    for t in range(0, T - H + 1, stride):
        s, s2 = S[t], S[t + H]
        p0 = s[2:2 + 2 * n_slots].reshape(n_slots, 2)
        p1 = s2[2:2 + 2 * n_slots].reshape(n_slots, 2)
        act0 = s[2 + 2 * n_slots:2 + 3 * n_slots] > 0.5
        act1 = s2[2 + 2 * n_slots:2 + 3 * n_slots] > 0.5
        both = act0 & act1
        disp = float(np.linalg.norm(p1[both] - p0[both], axis=1).sum())
        held = float(S[t:t + H + 1, 2 + 3 * n_slots:2 + 4 * n_slots].max() > 0.5)
        out.append((s, A[t:t + H], s2, disp, held))
    return out


# ------------------------------------------------------------------------------ main entry
def collect(env: TabletopEnv, n_transitions: int, stride: int = 10, expert_fraction: float = 0.4,
            noise: float = 0.05, seed: int = 0, progress: bool = True) -> dict:
    """Collect ~n_transitions windows. Returns a dict of arrays (see module docstring)."""
    H = env.cfg.micro_steps_per_macro
    rng = np.random.default_rng(seed)
    planner = ScriptedSymbolicPlanner(mode="mixed", seed=seed)
    cols = {k: [] for k in ("states", "actions", "next_states", "episode", "kind", "obj_disp", "used_grip")}
    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_transitions, desc="collecting", unit="win")
        except ImportError:
            pass
    n, ep = 0, 0
    while n < n_transitions:
        env.reset(seed=int(rng.integers(1 << 30)))
        kind = 0 if rng.random() < expert_fraction else 1
        if kind == 0:
            S, A = _record(env, lambda: _expert_episode(env, planner, rng, noise))
        else:
            S, A = _record(env, lambda: _explore_episode(env, planner, rng, noise))
        for s, a, s2, disp, held in _windows(S, A, H, stride, env.n_slots):
            cols["states"].append(s); cols["actions"].append(a); cols["next_states"].append(s2)
            cols["episode"].append(ep); cols["kind"].append(kind)
            cols["obj_disp"].append(disp); cols["used_grip"].append(held)
            n += 1
            if pbar is not None:
                pbar.update(1)
        ep += 1
    if pbar is not None:
        pbar.close()
    data = {k: np.array(v) for k, v in cols.items()}
    for k in ("states", "actions", "next_states", "obj_disp"):
        data[k] = data[k].astype(np.float32)
    data["episode"] = data["episode"].astype(np.int32)
    data["kind"] = data["kind"].astype(np.int8)
    data["used_grip"] = data["used_grip"].astype(np.int8)
    for k in ("states", "actions", "next_states", "episode", "kind", "obj_disp", "used_grip"):
        data[k] = data[k][:n_transitions]
    data["H"] = np.array(H)
    data["stride"] = np.array(stride)
    data["n_slots"] = np.array(env.n_slots)
    data["n_objects"] = np.array(env.cfg.n_objects)
    data["state_dim"] = np.array(env.state_dim)
    return data


def collect_onpolicy(env: TabletopEnv, planner, oracle: ScriptedSymbolicPlanner, n_transitions: int, stride: int = 10,
                     handover_fraction: float = 0.5, noise: float = 0.1, seed: int = 0, progress: bool = True):
    """Windows from the LEARNED planner's own behaviour (kind = 2).

    Each episode chains up to 4 random tasks (random object and zone, correct goal, no decoding noise). For each task
      * pure rollout (prob 1 - handover_fraction): the planner acts, with Gaussian noise on the xy actions, until success or
        the episode budget ends; these tasks give an unbiased estimate of the planner's own success rate;
      * handover (prob handover_fraction): the planner acts for 10-80 steps, then the scripted oracle finishes the task, so
        the data contain expert recovery from states the learner actually reaches.
    Windows are cut every `stride` steps, exactly like `collect`. Returns (data dict, stats dict).
    """
    H = env.cfg.micro_steps_per_macro
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("states", "actions", "next_states", "episode", "kind", "obj_disp", "used_grip")}
    stats = {"tasks_pure": 0, "success_pure": 0, "tasks_handover": 0}
    pbar = None
    if progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_transitions, desc="on-policy", unit="win")
        except ImportError:
            pass

    def release(env_):
        for _ in range(10):
            if env_.holding < 0:
                break
            env_.step(np.array([0.0, 0.0, -1.0]))

    n, ep = 0, 0
    while n < n_transitions:
        env.reset(target_obj=int(rng.integers(env.cfg.n_objects)), goal_zone=int(rng.integers(env.cfg.n_zones)),
                  seed=int(rng.integers(1 << 30)))
        planner.reseed(int(rng.integers(1 << 30)))

        def run():
            for _ in range(4):
                if env.t >= env.horizon:
                    break
                release(env)
                o, z = env.target_obj, env.goal_zone
                handover = bool(rng.random() < handover_fraction)
                budget = int(rng.integers(10, 80)) if handover else env.horizon
                t0 = env.t
                while env.t < env.horizon and env.t - t0 < budget and not env.task_success(o, z):
                    out = planner.plan(env.get_state(), [(o, z, 1.0)])
                    for a in out["actions"]:
                        a = np.array(a, dtype=float)
                        a[:2] += rng.normal(0.0, noise, 2)
                        env.step(a)
                        if env.t >= env.horizon or env.t - t0 >= budget:
                            break
                ok = env.task_success(o, z)
                if handover:
                    stats["tasks_handover"] += 1
                    if not ok and env.t < env.horizon:
                        oracle.run_episode(env, o, z, mode="auto", max_replans=4)
                else:
                    stats["tasks_pure"] += 1
                    stats["success_pure"] += int(ok)
                env.set_task(int(rng.integers(env.cfg.n_objects)), int(rng.integers(env.cfg.n_zones)))

        S, A = _record(env, run)
        for s_, a_, s2_, disp, held in _windows(S, A, H, stride, env.n_slots):
            cols["states"].append(s_); cols["actions"].append(a_); cols["next_states"].append(s2_)
            cols["episode"].append(ep); cols["kind"].append(2)
            cols["obj_disp"].append(disp); cols["used_grip"].append(held)
            n += 1
            if pbar is not None:
                pbar.update(1)
        ep += 1
    if pbar is not None:
        pbar.close()
    data = {k: np.array(v)[:n_transitions] for k, v in cols.items()}
    for k in ("states", "actions", "next_states", "obj_disp"):
        data[k] = data[k].astype(np.float32)
    data["episode"] = data["episode"].astype(np.int32)
    data["kind"] = data["kind"].astype(np.int8)
    data["used_grip"] = data["used_grip"].astype(np.int8)
    data["H"] = np.array(H)
    data["stride"] = np.array(stride)
    data["n_slots"] = np.array(env.n_slots)
    data["n_objects"] = np.array(env.cfg.n_objects)
    data["state_dim"] = np.array(env.state_dim)
    return data, stats


def merge(a: dict, b: dict) -> dict:
    """Concatenate two datasets from the same environment configuration (episode ids of b are offset)."""
    assert int(a["state_dim"]) == int(b["state_dim"]) and int(a["H"]) == int(b["H"]), "datasets come from different configurations"
    out = {}
    offset = int(a["episode"].max()) + 1
    for k in ("states", "actions", "next_states", "kind", "obj_disp", "used_grip"):
        out[k] = np.concatenate([a[k], b[k]])
    out["episode"] = np.concatenate([a["episode"], b["episode"] + offset]).astype(np.int32)
    for k in ("H", "stride", "n_slots", "n_objects", "state_dim"):
        out[k] = a[k]
    return out


def summarize(data: dict) -> str:
    n = len(data["states"])
    moved = float((data["obj_disp"] > 0.01).mean())
    kind = data["kind"]
    names = {0: "expert", 1: "explore", 2: "on-policy"}
    mix = " / ".join(f"{names[k]} {float((kind == k).mean()):.0%}" for k in (0, 1, 2) if (kind == k).any())
    return (f"{n} windows from {len(np.unique(data['episode']))} episodes | H={int(data['H'])} | {mix} | "
            f"windows with object motion >1cm: {moved:.0%} | with a held object: {float(data['used_grip'].mean()):.0%} | "
            f"median object displacement when moved: {float(np.median(data['obj_disp'][data['obj_disp'] > 0.01])):.3f} m")


def save(data: dict, path: str) -> None:
    np.savez_compressed(path, **data)


def load(path: str) -> dict:
    return dict(np.load(path))


def split_by_episode(data: dict, val_fraction: float = 0.1, seed: int = 0):
    """Train/val split that keeps all windows of an episode together."""
    eps = np.unique(data["episode"])
    rng = np.random.default_rng(seed)
    val_eps = set(rng.choice(eps, size=max(1, int(len(eps) * val_fraction)), replace=False).tolist())
    mask = np.array([e in val_eps for e in data["episode"]])
    per_row = {k: v for k, v in data.items() if hasattr(v, "shape") and v.shape[:1] == mask.shape}
    tr = {k: v[~mask] for k, v in per_row.items()}
    va = {k: v[mask] for k, v in per_row.items()}
    for d in (tr, va):
        for k in ("H", "stride", "n_slots", "n_objects", "state_dim"):
            d[k] = data[k]
    return tr, va
