"""Two-level CEM / MPC planner in the learned latent space (DAY 5).

Level 1 (effects): sample sequences of T effect codes (c_1..c_T), roll them through the meta-dynamics model
  s_t = f(s_{t-1}, c_t), score the predicted states against the committed goal, and refine with CEM.
  Iteration 0 draws c_t sequentially from the state-conditioned prior p(c | s_{t-1}) ("learning to sample");
  later iterations refit a Gaussian over the elite sequences.
Level 2 (motions): decode the first effect code of the best sequence into a low-level action chunk with the
  action generator, using the prior-mean motion code z. The first `exec_steps` actions are executed, then the
  planner re-plans from the new state (MPC).

Goal cost for a predicted state sequence (goals = [(target_obj, zone, weight), ...]):
  sum_g w_g * ( d_T + 0.25 * mean_t d_t + approach_w * |ee_T - obj_T| + held_w * held_T
                + release_w * held_T * [d_T < 0.06] )
  with d_t = dist(obj_g, zone_g). The approach and release terms are hand-designed shaping (they give the
  approach / release effects a gradient); say so in the write-up. Set approach_w = release_w = 0 and
  held_w = 0.05 to recover the unshaped cost used in the first evaluation.
  For UNGRASPABLE targets (typed scenes, graspable flag = 0) two extra options exist:
    approach_weight_heavy : approach weight used instead of approach_weight (default: same; 0 removes the pull
                            toward the disc centre, which is wrong when the disc must be pushed)
    push_weight           : + push_w * |ee_T - behind_T| * [d_T > 0.05], with behind = obj_T - 0.068 * unit(zone - obj_T),
                            i.e. reward the gripper for being behind the disc on the line to the zone (hand-designed
                            push geometry; default 0 = off)

The planner core is numpy only; `TorchLatentModel` adapts a trained CavinLite (torch) to numpy arrays.
"""
from __future__ import annotations

import numpy as np


class TorchLatentModel:
    """numpy <-> torch adapter around a trained CavinLite."""

    def __init__(self, model, device: str = "cpu"):
        import torch
        self.torch, self.model, self.device = torch, model.eval(), device
        self.effect_dim = model.cd

    def _t(self, x):
        return self.torch.as_tensor(np.asarray(x), dtype=self.torch.float32, device=self.device)

    def seed(self, s: int) -> None:
        self.torch.manual_seed(int(s))

    def sample_effects(self, s: np.ndarray, n: int) -> np.ndarray:           # (B, sd) -> (B, n, cd)
        return self.model.sample_effects(self._t(s), n).cpu().numpy()

    def predict_next(self, s: np.ndarray, c: np.ndarray) -> np.ndarray:       # (B, sd), (B, cd) -> (B, sd)
        return self.model.predict_next(self._t(s), self._t(c)).cpu().numpy()

    def generate_actions(self, s: np.ndarray, c: np.ndarray) -> np.ndarray:   # (1, sd), (1, cd) -> (H, ad)
        return self.model.generate_actions(self._t(s), self._t(c), deterministic=True)[0].cpu().numpy()


class HierarchicalPlanner:
    def __init__(self, model, cfg: dict, zone_centers: np.ndarray, n_slots: int, seed: int = 0,
                 horizon: int | None = None):
        pc = cfg["planner"]
        self.model = model
        self.N, self.E, self.iters = pc["cem_samples"], pc["cem_elites"], pc["cem_iters"]
        self.T = horizon or pc["high_level_horizon"]
        self.exec_steps = pc.get("exec_steps", 10)
        self.approach_w = pc.get("approach_weight", 0.25)
        self.held_w = pc.get("held_weight", 0.05)
        self.release_w = pc.get("release_weight", 0.2)
        self.approach_w_heavy = pc.get("approach_weight_heavy", None)
        self.push_w = pc.get("push_weight", 0.0)
        self.behind_d = 0.068
        self.zone = np.asarray(zone_centers, dtype=np.float32)
        self.n_slots = n_slots
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        if hasattr(self.model, "seed"):
            self.model.seed(seed)

    # ------------------------------------------------------------------ cost / rollout
    def _cost(self, S: np.ndarray, goals, s0: np.ndarray | None = None) -> np.ndarray:
        """S: (N, T, sd) predicted raw states -> (N,) cost. s0 (1, sd) is the current state (for the graspable flag)."""
        n = self.n_slots
        typed = s0 is not None and s0.shape[1] > 2 + 4 * n
        total = np.zeros(S.shape[0], dtype=np.float64)
        for obj, zone, w in goals:
            graspable = bool(s0[0, 2 + 4 * n + obj] > 0.5) if typed else True
            pos = S[:, :, 2 + 2 * obj: 4 + 2 * obj]
            d = np.linalg.norm(pos - self.zone[zone], axis=-1)               # (N, T) object -> zone
            da = np.linalg.norm(S[:, :, 0:2] - pos, axis=-1)                 # (N, T) end effector -> object
            held = np.clip(S[:, :, 2 + 3 * n + obj], 0.0, 1.0)
            dT, hT = d[:, -1], held[:, -1]
            aw = self.approach_w if (graspable or self.approach_w_heavy is None) else self.approach_w_heavy
            c = dT + 0.25 * d.mean(1) + aw * da[:, -1] + self.held_w * hT + self.release_w * hT * (dT < 0.06)
            if (not graspable) and self.push_w > 0:
                to_zone = self.zone[zone] - pos[:, -1]
                u = to_zone / (np.linalg.norm(to_zone, axis=-1, keepdims=True) + 1e-6)
                behind = pos[:, -1] - self.behind_d * u
                c = c + self.push_w * np.linalg.norm(S[:, -1, 0:2] - behind, axis=-1) * (dT > 0.05)
            total += w * c
        return total

    def _rollout(self, s0: np.ndarray, C: np.ndarray) -> np.ndarray:
        s = np.repeat(s0, C.shape[0], axis=0)
        out = []
        for t in range(C.shape[1]):
            s = self.model.predict_next(s, C[:, t])
            out.append(s)
        return np.stack(out, 1)

    # ------------------------------------------------------------------ planning
    def plan(self, state: np.ndarray, goals) -> dict:
        s0 = np.asarray(state, dtype=np.float32)[None]
        N, T, E = self.N, self.T, self.E
        s = np.repeat(s0, N, axis=0)
        C, S = [], []
        for _ in range(T):                                    # iteration 0: sequential prior sampling
            c = self.model.sample_effects(s, 1)[:, 0]
            s = self.model.predict_next(s, c)
            C.append(c)
            S.append(s)
        C, S = np.stack(C, 1), np.stack(S, 1)
        cost = self._cost(S, goals, s0)
        i = int(cost.argmin())
        best_cost, best_C = float(cost[i]), C[i].copy()
        for _ in range(self.iters - 1):                       # CEM refinement in effect-code space
            el = C[np.argsort(cost)[:E]]
            mu, sd = el.mean(0), np.maximum(el.std(0), 0.05)
            Cn = (mu + sd * self.rng.standard_normal((N, T, C.shape[-1]))).astype(np.float32)
            Cn[:E] = el
            C = Cn
            cost = self._cost(self._rollout(s0, C), goals, s0)
            i = int(cost.argmin())
            if cost[i] < best_cost:
                best_cost, best_C = float(cost[i]), C[i].copy()
        a = self.model.generate_actions(s0, best_C[0][None])  # low level: decode first effect into actions
        pred = self._rollout(s0, best_C[None])[0]              # (T, sd) predicted states along the chosen effect sequence
        return dict(actions=np.asarray(a)[: self.exec_steps], cost=best_cost, c1=best_C[0], pred_states=pred)
