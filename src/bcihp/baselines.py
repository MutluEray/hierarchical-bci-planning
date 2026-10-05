"""Baselines (Days 1-2 and 5).

1. ScriptedSymbolicPlanner : oracle expert with two skills: PUSH (clear blockers sideways, push target to the
                              zone) and GRASP (weld-grasp the target, carry it over clutter, release in zone).
                              Needed for the Day 1 demo, data generation, and as the upper bound.
2. DirectBCIController     : 4 directional low-level commands for the end effector (TODO, Day 2).
3. FlatCEMPlanner          : CEM directly in raw action space (TODO, Day 5).
"""
from __future__ import annotations

import numpy as np

from .commands import IntentChannel
from .env import (CORRIDOR, EE_LIM, GRASP_R, R_EE, R_OBJ, TabletopEnv)

CLEAR = R_EE + R_OBJ + 0.008          # path clearance when travelling without pushing
BEHIND = R_EE + R_OBJ + 0.018         # standoff behind an object before pushing it
PUSH_THROUGH = R_EE + R_OBJ - 0.005   # ee ends this far behind the push destination
GRASP_STAND = 0.060                   # ee standoff from the object centre before closing the gripper
WP_TOL = 0.003


def _seg_dist(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - a)), 0.0
    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab))), t


def _line_free(a, b, obstacles, clear):
    return all(_seg_dist(o, a, b)[0] >= clear for o in obstacles)


def safe_path(a, b, obstacles, clear=CLEAR, res=0.01):
    """Waypoints (excluding a, including b) from a to b around circular obstacles.

    Grid A* (8-connected) over the ee workspace with obstacles inflated by `clear`, followed by
    line-of-sight shortcutting. Falls back to the straight line if no path exists.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if _line_free(a, b, obstacles, clear):
        return [b]
    n = int(round(2 * EE_LIM / res)) + 1
    xs = np.linspace(-EE_LIM, EE_LIM, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    blocked = np.zeros((n, n), dtype=bool)
    for o in obstacles:
        blocked |= (gx - o[0]) ** 2 + (gy - o[1]) ** 2 < clear ** 2

    def cell(p):
        return (int(np.clip(round((p[0] + EE_LIM) / res), 0, n - 1)),
                int(np.clip(round((p[1] + EE_LIM) / res), 0, n - 1)))

    sc, gc = cell(a), cell(b)
    for c in (sc, gc):  # endpoints may sit inside an inflated region (e.g. just after a push)
        blocked[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4] &= (
            (gx[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4] - xs[c[0]]) ** 2
            + (gy[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4] - xs[c[1]]) ** 2 > 0.03 ** 2)
    import heapq
    h = lambda c: np.hypot(c[0] - gc[0], c[1] - gc[1])
    openq, came, g = [(h(sc), sc)], {sc: None}, {sc: 0.0}
    while openq:
        _, cur = heapq.heappop(openq)
        if cur == gc:
            break
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < n and 0 <= nb[1] < n) or blocked[nb]:
                    continue
                ng = g[cur] + np.hypot(dx, dy)
                if ng < g.get(nb, 1e18):
                    g[nb], came[nb] = ng, cur
                    heapq.heappush(openq, (ng + h(nb), nb))
    if gc not in came:
        return [b]
    path, c = [], gc
    while c is not None:
        path.append(np.array([xs[c[0]], xs[c[1]]]))
        c = came[c]
    path = path[::-1]
    path[0], path[-1] = a, b
    # line-of-sight shortcutting (the final waypoint is always kept)
    out, i = [], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(path[i], path[j], obstacles, clear * 0.9):
            j -= 1
        out.append(path[j])
        i = j
    return out


class ScriptedSymbolicPlanner:
    """Closed-loop oracle. Plans a list of primitives, executes them, and re-plans after each sub-task.

    Primitives: ("move", xy) | ("grasp",) | ("release",).
    mode: "push"   -> only pushing (clear blockers sideways, push target),
          "grasp"  -> only weld-grasp pick-and-place (carry over clutter),
          "mixed"  -> pick a skill at random per episode (used for diverse data generation),
          "auto"   -> grasp whenever the target can be grasped, push otherwise (heavy objects).
    Heavy objects (env.heavy) cannot be grasped, so every mode pushes them.
    """

    def __init__(self, mode: str = "mixed", seed: int = 0):
        assert mode in ("push", "grasp", "mixed", "auto")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def choose_skill(env: TabletopEnv, target_obj: int, mode: str, rng) -> str:
        if not env.cfg.allow_grasp or env.heavy[target_obj] or mode == "push":
            return "push"
        if mode == "mixed":
            return "grasp" if rng.random() < 0.5 else "push"
        return "grasp"                                       # "grasp" or "auto" on a graspable target

    # ---- push skill -----------------------------------------------------------------------------
    def _push_waypoints(self, env: TabletopEnv, obj_idx: int, dest: np.ndarray):
        pos = env.obj_positions()
        p = pos[obj_idx]
        d = dest - p
        dist = np.linalg.norm(d)
        if dist < 1e-6:
            return []
        u = d / dist
        behind = p - u * BEHIND
        obstacles = [pos[j] for j in range(env.n_slots) if env.active[j] and j != obj_idx]
        wps = safe_path(env.ee.copy(), np.clip(behind, -EE_LIM, EE_LIM), obstacles + [p])
        wps.append(np.clip(dest - u * PUSH_THROUGH, -EE_LIM, EE_LIM))
        return [("move", w) for w in wps]

    def _clear_destination(self, env: TabletopEnv, blocker: int, target_obj: int, goal_zone: int):
        pos = env.obj_positions()
        a, g = pos[target_obj], env.zone_centers[goal_zone]
        u = (g - a) / np.linalg.norm(g - a)
        nrm = np.array([-u[1], u[0]])
        p = pos[blocker]
        lat = (p - a) @ nrm
        need = CORRIDOR + 0.03
        best = None
        for s in (+1, -1):
            dest = p + s * nrm * max(need - s * lat, 0.04)
            behind = p - s * nrm * BEHIND
            penalty = 0.0
            if np.any(np.abs(dest) > 0.275):
                penalty += 10.0
            if np.any(np.abs(behind) > EE_LIM):
                penalty += 10.0
            cost = penalty + np.linalg.norm(dest - p)
            if best is None or cost < best[0]:
                best = (cost, dest)
        return np.clip(best[1], -0.275, 0.275)

    def _plan_push(self, env, target_obj, goal_zone):
        bl = env.blockers(target_obj, goal_zone)
        if bl:
            dest = self._clear_destination(env, bl[0], target_obj, goal_zone)
            return self._push_waypoints(env, bl[0], dest)
        return self._push_waypoints(env, target_obj, env.zone_centers[goal_zone])

    # ---- grasp skill ----------------------------------------------------------------------------
    @staticmethod
    def _pick_stand(env: TabletopEnv, target_obj: int, pos: np.ndarray) -> np.ndarray:
        """Standoff point around the target from which the TARGET is the closest object to the gripper
        (the gripper welds the nearest object), preferring the point closest to the current ee."""
        p = pos[target_obj]
        others = [pos[j] for j in range(env.n_slots) if env.active[j] and j != target_obj]
        best, best_score = None, -1e9
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            c = p + GRASP_STAND * np.array([np.cos(ang), np.sin(ang)])
            if np.any(np.abs(c) > EE_LIM):
                continue
            margin = min([np.linalg.norm(c - o) for o in others], default=1.0) - GRASP_STAND
            if margin < 0.015:
                score = -100.0 + margin                 # target not clearly nearest: heavily penalise
            else:
                score = -np.linalg.norm(c - env.ee)     # otherwise prefer the closest reachable point
            if score > best_score:
                best, best_score = c, score
        return best if best is not None else np.clip(p + np.array([0.0, -GRASP_STAND]), -EE_LIM, EE_LIM)

    def carry_primitives(self, env: TabletopEnv, obj_idx: int, dest: np.ndarray):
        """Primitives to weld-grasp object `obj_idx`, carry it to `dest` (ee position), and release."""
        if env.holding >= 0 and env.holding != obj_idx:
            return [("release",)]                      # drop whatever we hold by mistake
        prims = []
        if env.holding != obj_idx:
            pos = env.obj_positions()
            stand = self._pick_stand(env, obj_idx, pos)
            obstacles = [pos[j] for j in range(env.n_slots) if env.active[j] and j != obj_idx] + [pos[obj_idx]]
            prims += [("move", w) for w in safe_path(env.ee.copy(), stand, obstacles)]
            prims.append(("grasp",))
        prims += [("move", np.asarray(dest, float).copy()), ("release",)]
        return prims

    def _plan_grasp(self, env, target_obj, goal_zone):
        return self.carry_primitives(env, target_obj, env.zone_centers[goal_zone])

    def plan(self, env: TabletopEnv, target_obj: int, goal_zone: int, skill: str = "push"):
        """Primitive list for the next sub-task."""
        if skill == "grasp" and env.cfg.allow_grasp:
            return self._plan_grasp(env, target_obj, goal_zone)
        return self._plan_push(env, target_obj, goal_zone)

    # ---- execution ------------------------------------------------------------------------------
    @staticmethod
    def _grip(env) -> float:
        return 1.0 if (env.holding >= 0 and not env._releasing) else -1.0

    def execute(self, env: TabletopEnv, prims, max_steps: int | None = None, stop_on_done: bool = True,
                noise: float = 0.0, frames: list | None = None, render_every: int = 1) -> dict:
        """Execute a primitive list with env.step. Returns dict(done, steps)."""
        st = {"done": False, "n": 0}

        def stop() -> bool:
            return (stop_on_done and st["done"]) or (max_steps is not None and st["n"] >= max_steps)

        def do(action):
            a = np.array(action, float)
            if noise > 0:
                a[:2] += self.rng.normal(0.0, noise, 2)
            _, d = env.step(a)
            st["done"], st["n"] = d, st["n"] + 1
            if frames is not None and env.t % render_every == 0:
                frames.append(env.render())

        for prim in prims:
            if prim[0] == "move":
                for _ in range(120):
                    err = prim[1] - env.ee
                    if stop() or np.linalg.norm(err) < WP_TOL:
                        break
                    do([*(err / env.cfg.max_delta), self._grip(env)])
            elif prim[0] == "grasp":
                for _ in range(6):
                    if stop():
                        break
                    do([0.0, 0.0, 1.0])
                    if env.holding >= 0:
                        break
                for _ in range(4):                     # let the object slide into the gripper
                    if stop():
                        break
                    do([0.0, 0.0, 1.0])
            else:  # release
                for _ in range(25):
                    if stop():
                        break
                    do([0.0, 0.0, -1.0])
                    if env.holding < 0:
                        break
            if stop():
                break
        return dict(done=st["done"], steps=st["n"])

    def run_episode(self, env: TabletopEnv, target_obj: int | None = None, goal_zone: int | None = None,
                    frames: list | None = None, render_every: int = 1, max_replans: int = 8,
                    mode: str | None = None) -> dict:
        """Run the oracle on the env's current scene. Appends RGB frames to `frames` if given."""
        target_obj = env.target_obj if target_obj is None else target_obj
        goal_zone = env.goal_zone if goal_zone is None else goal_zone
        env.target_obj, env.goal_zone = target_obj, goal_zone
        mode = self.choose_skill(env, target_obj, mode or self.mode, self.rng)
        done = env.task_success(target_obj, goal_zone)
        n_replans = 0
        if frames is not None:
            frames.append(env.render())
        while not done and n_replans < max_replans and env.t < env.horizon:
            n_replans += 1
            prims = self.plan(env, target_obj, goal_zone, mode)
            done = self.execute(env, prims, frames=frames, render_every=render_every)["done"]
        return dict(success=env.task_success(target_obj, goal_zone), steps=env.t, replans=n_replans, skill=mode)


class DirectBCIController:
    """Baseline: cursor-style low-level BCI control (what a typical MI-BCI offers today).

    Vocabulary (5 commands): LEFT, UP, RIGHT, DOWN, GRIP (toggle close/open). Each decoded command
    drives the end effector for `steps_per_command` micro-steps (GRIP takes `grip_steps`).

    The simulated user KNOWS how to do the task: at every command it intends the cardinal direction toward
    the next oracle waypoint (or GRIP at grasp/release points). The only handicaps are the coarse command
    vocabulary and the noisy decoder (IntentChannel confusion matrix), so this is an optimistic baseline
    for direct control. The comparison metric is success and the number of BCI commands per task.
    """

    LEFT, UP, RIGHT, DOWN, GRIP = range(5)
    DIRS = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    vocab_size = 5

    def __init__(self, channel: IntentChannel | None = None, mode: str = "grasp", steps_per_command: int = 2,
                 grip_steps: int = 8, seed: int = 0):
        self.channel = channel or IntentChannel(self.vocab_size, 1.0, seed=seed)
        assert self.channel.K == self.vocab_size
        self.planner = ScriptedSymbolicPlanner(mode="push", seed=seed)
        self.mode = mode
        self.steps_per_command = steps_per_command
        self.grip_steps = grip_steps
        self.max_stall = 25
        self._plan, self._idx, self._stall = None, 0, 0
        self.rng = np.random.default_rng(seed)

    def intended_command(self, env: TabletopEnv, target_obj: int, goal_zone: int, skill: str) -> int:
        """The simulated user commits to an oracle plan and follows it waypoint by waypoint; it re-plans only
        when the plan is exhausted or no progress was made for `max_stall` commands."""
        half = self.steps_per_command * env.cfg.max_delta / 2 + 1e-3   # per-axis arrival tolerance
        for _ in range(4):
            if self._plan is None or self._idx >= len(self._plan) or self._stall > self.max_stall:
                self._plan = self.planner.plan(env, target_obj, goal_zone, skill)
                self._idx, self._stall = 0, 0
                if not self._plan:
                    return self.GRIP
            prim = self._plan[self._idx]
            if prim[0] == "move":
                err = prim[1] - env.ee
                if np.max(np.abs(err)) <= half:        # waypoint reached
                    self._idx, self._stall = self._idx + 1, 0
                    continue
                self._stall += 1
                if abs(err[0]) >= abs(err[1]):
                    return self.RIGHT if err[0] > 0 else self.LEFT
                return self.UP if err[1] > 0 else self.DOWN
            if prim[0] == "grasp":
                if env.holding == target_obj:
                    self._idx, self._stall = self._idx + 1, 0
                    continue
                self._stall += 1
                if env.holding >= 0:                       # grabbed the wrong object: open the gripper
                    return self.GRIP
                # Only close once the TARGET is the nearest object and within reach; otherwise creep toward it.
                pos = env.obj_positions()
                d = np.linalg.norm(pos - env.ee, axis=1)
                d[~env.active] = np.inf
                if int(np.argmin(d)) == target_obj and d[target_obj] < GRASP_R - 0.012:
                    return self.GRIP
                err = pos[target_obj] - env.ee
                if abs(err[0]) >= abs(err[1]):
                    return self.RIGHT if err[0] > 0 else self.LEFT
                return self.UP if err[1] > 0 else self.DOWN
            # release
            if env.holding < 0:
                self._idx, self._stall = self._idx + 1, 0
                continue
            self._stall += 1
            return self.GRIP
        return self.GRIP

    def _execute(self, env: TabletopEnv, cmd: int, frames, render_every) -> bool:
        done = False
        if cmd == self.GRIP:
            g = 1.0 if env.holding < 0 else -1.0
            for _ in range(self.grip_steps):
                _, done = env.step(np.array([0.0, 0.0, g]))
                if frames is not None and env.t % render_every == 0:
                    frames.append(env.render())
                if done:
                    break
        else:
            for _ in range(self.steps_per_command):
                g = 1.0 if (env.holding >= 0 and not env._releasing) else -1.0
                _, done = env.step(np.array([*self.DIRS[cmd], g]))
                if frames is not None and env.t % render_every == 0:
                    frames.append(env.render())
                if done:
                    break
        return done

    def run_episode(self, env: TabletopEnv, target_obj: int | None = None, goal_zone: int | None = None,
                    frames: list | None = None, render_every: int = 1) -> dict:
        target_obj = env.target_obj if target_obj is None else target_obj
        goal_zone = env.goal_zone if goal_zone is None else goal_zone
        env.target_obj, env.goal_zone = target_obj, goal_zone
        skill = ScriptedSymbolicPlanner.choose_skill(env, target_obj, self.mode, self.rng)
        done = env.task_success(target_obj, goal_zone)
        n_cmd = n_err = 0
        self._plan, self._idx, self._stall = None, 0, 0
        if frames is not None:
            frames.append(env.render())
        while not done and env.t < env.horizon:
            c = self.intended_command(env, target_obj, goal_zone, skill)
            d = self.channel.decode(c)
            n_cmd += 1
            n_err += int(d != c)
            done = self._execute(env, d, frames, render_every)
        return dict(success=env.task_success(target_obj, goal_zone), steps=env.t, n_commands=n_cmd,
                    n_errors=n_err, skill=skill)


class FlatCEMPlanner:
    """TODO (Day 5): CEM over raw action sequences using a learned one-step model."""

    def plan(self, state, goal_costs):
        raise NotImplementedError
