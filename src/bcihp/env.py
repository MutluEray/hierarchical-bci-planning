"""Tabletop pushing + weld-grasp environment in MuJoCo (DAY 1).

Design (see docs/PLAN.md):
  * Planar world: every movable object is a cylinder on slide joints (x, y), plus a kinematically
    controlled z joint used only to draw a lifted/carried object. No tipping or table-contact physics.
  * The pusher is a mocap cylinder moved by end-effector xy deltas. A 2-link arm drawn above the table
    follows it via analytic IK. The arm is COSMETIC (visual only, no collisions).
  * Two manipulation primitives:
      - PUSH: pusher contacts objects (real contact dynamics).
      - WELD-GRASP ("magic grasp"): closing the gripper next to an object rigidly attaches it to the end
        effector (kinematic weld, object slides into the gripper and is lifted). While carrying, the
        gripper and carried object do not collide with anything (carried over clutter). Opening lowers and
        releases it; a release that would overlap another object or leave the workspace is refused.
        This is an abstraction of grasping, not a contact-rich grasp model; state this in the write-up.
  * Slots: `n_objects` regular objects followed by `n_obstacles_max` obstacle slots. Inactive obstacle
    slots are parked off the table.
  * Action: (dx, dy, grip) in [-1, 1]. grip > 0 = close/hold, grip <= 0 = open/release. A 2-vector action
    keeps the current grip state. cfg.allow_grasp=False ignores grip (push-only ablation).
  * State (task-agnostic, metres): [ee_xy (2), obj_xy for all slots (2n), active mask (n), held mask (n)].
    Inactive slots report xy = 0 with mask = 0. The task (target, zone) is NOT in the state; it is
    provided at planning time through the command.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise ImportError("mujoco is required: pip install mujoco") from e

# ----------------------------------------------------------------------------- geometry constants
R_OBJ = 0.03             # object radius
R_EE = 0.02              # pusher radius
OBJ_H = 0.04
WS_HALF = 0.31           # inner half-size of the walled workspace
EE_LIM = 0.285           # end-effector bounds (+-)
EE_HOME = np.array([0.0, -0.27])
ZONE_Y = 0.20
ZONE_X_SPAN = 0.18
ZONE_VIS_R = 0.05
ZONE_SUCCESS_R = 0.035   # object centre must be within this distance of the zone centre
SPAWN_X = 0.17
SPAWN_Y = (-0.20, 0.0)
CORRIDOR = 0.07          # half-width of the "straight path" corridor used to define blocking
ARM_BASE = np.array([0.0, -0.55])
ARM_L = 0.45
ARM_Z = 0.115
GRASP_R = 0.07           # gripper can weld an object whose centre is within this distance
LIFT = 0.07              # carry height (m)
LIFT_RATE = 0.012        # lift/lower speed per micro-step (m)
HOLD_DECAY = 0.85        # per-substep decay of the grasp offset (object slides into the gripper)
GHOST_R = R_EE + R_OBJ + 0.006   # pusher stays non-colliding with a just-released object until clear of it
OBJ_COLORS = [(0.85, 0.25, 0.25, 1), (0.25, 0.55, 0.85, 1), (0.95, 0.75, 0.2, 1), (0.6, 0.4, 0.8, 1),
              (0.95, 0.55, 0.15, 1), (0.55, 0.35, 0.2, 1)]
HEAVY_H = 0.08           # heavy objects are taller and cannot be grasped (they must be pushed)
OBSTACLE_COLOR = (0.35, 0.35, 0.38, 1)
ZONE_COLORS = [(0.4, 0.8, 0.45, 1), (0.4, 0.8, 0.8, 1), (0.9, 0.6, 0.8, 1), (0.8, 0.8, 0.4, 1)]


@dataclass
class SceneConfig:
    n_objects: int = 3
    n_obstacles_max: int = 2
    n_zones: int = 3
    max_macro_steps: int = 10
    micro_steps_per_macro: int = 20
    max_delta: float = 0.012         # max end-effector displacement per micro-step (m)
    substeps: int = 10               # physics substeps per micro-step (dt = 0.002 s)
    render_size: tuple = (640, 480)  # (width, height)
    allow_grasp: bool = True         # False = push-only ablation (grip action ignored)
    n_heavy: int = 0                 # the last n_heavy regular objects are "heavy": taller, cannot be grasped, push only


def _quat_z_to_dir(d: np.ndarray) -> np.ndarray:
    """Quaternion rotating +z onto a horizontal unit vector d = (dx, dy, 0)."""
    s = np.sqrt(0.5)
    return np.array([s, -d[1] * s, d[0] * s, 0.0])


class TabletopEnv:
    def __init__(self, cfg: SceneConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.n_slots = cfg.n_objects + cfg.n_obstacles_max
        self.heavy = np.zeros(self.n_slots, dtype=bool)
        if cfg.n_heavy > 0:
            self.heavy[cfg.n_objects - cfg.n_heavy: cfg.n_objects] = True
        self._typed = cfg.n_heavy > 0
        crowded = self.n_slots > 5
        self._sx = 0.20 if crowded else SPAWN_X
        self._sy = (-0.24, 0.04) if crowded else SPAWN_Y
        self.horizon = cfg.max_macro_steps * cfg.micro_steps_per_macro
        self.zone_centers = self._make_zones()
        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self._index_model()
        self.active = np.zeros(self.n_slots, dtype=bool)
        self.ee = EE_HOME.copy()
        self.target_obj = 0
        self.goal_zone = 0
        self.t = 0
        self.holding = -1            # slot index currently welded to the gripper, or -1
        self._releasing = False
        self._lift = 0.0
        self._off = np.zeros(2)
        self._ghost_obj = -1
        self._renderer = None
        self._cam = None
        self.reset()

    # ------------------------------------------------------------------ model construction
    def _make_zones(self) -> np.ndarray:
        n = self.cfg.n_zones
        xs = np.array([0.0]) if n == 1 else np.linspace(-ZONE_X_SPAN, ZONE_X_SPAN, n)
        return np.stack([xs, np.full(n, ZONE_Y)], axis=1)

    def _build_xml(self) -> str:
        W, H = self.cfg.render_size
        parts = []
        geom_common = 'solref="0.006 1" friction="0.4 0.005 0.0001"'
        for i in range(self.n_slots):
            rgba = OBJ_COLORS[i % len(OBJ_COLORS)] if i < self.cfg.n_objects else OBSTACLE_COLOR
            h = HEAVY_H if self.heavy[i] else OBJ_H
            mass_attr = ' mass="0.1131"' if self.heavy[i] else ""          # same pushing dynamics as light objects
            cap = (f'<geom type="cylinder" size="{R_OBJ * 0.97} 0.0015" pos="0 0 {h}" contype="0" conaffinity="0" '
                   f'mass="0" rgba="0.12 0.12 0.12 1"/>') if self.heavy[i] else ""
            parts.append(
                f'<body name="obj{i}" pos="{10 + i} 0 0">'
                f'<joint name="o{i}x" type="slide" axis="1 0 0" damping="6"/>'
                f'<joint name="o{i}y" type="slide" axis="0 1 0" damping="6"/>'
                f'<joint name="o{i}z" type="slide" axis="0 0 1" damping="50"/>'
                f'<geom name="og{i}" type="cylinder" size="{R_OBJ} {h / 2}" pos="0 0 {h / 2}"{mass_attr} '
                f'rgba="{" ".join(map(str, rgba))}" {geom_common}/>{cap}'
                f'<geom name="halo{i}" type="cylinder" size="{R_OBJ + 0.012} 0.0006" pos="0 0 0.0006" '
                f'contype="0" conaffinity="0" mass="0" rgba="1 1 1 0"/></body>')
        zones = []
        for k, (zx, zy) in enumerate(self.zone_centers):
            rgba = ZONE_COLORS[k % len(ZONE_COLORS)]
            zones.append(f'<geom type="cylinder" size="{ZONE_VIS_R} 0.0008" pos="{zx} {zy} 0.0008" '
                         f'contype="0" conaffinity="0" rgba="{" ".join(map(str, rgba))}"/>'
                         f'<geom name="zring{k}" type="cylinder" size="{ZONE_VIS_R + 0.012} 0.0004" pos="{zx} {zy} 0.0004" '
                         f'contype="0" conaffinity="0" mass="0" rgba="1 1 1 0"/>')
        wl = WS_HALF + 0.01
        walls = (
            f'<geom type="box" size="{wl + 0.01} 0.01 0.03" pos="0 {wl} 0.03" rgba="0.5 0.45 0.4 1"/>'
            f'<geom type="box" size="{wl + 0.01} 0.01 0.03" pos="0 {-wl} 0.03" rgba="0.5 0.45 0.4 1"/>'
            f'<geom type="box" size="0.01 {wl + 0.01} 0.03" pos="{wl} 0 0.03" rgba="0.5 0.45 0.4 1"/>'
            f'<geom type="box" size="0.01 {wl + 0.01} 0.03" pos="{-wl} 0 0.03" rgba="0.5 0.45 0.4 1"/>')
        link_hl = ARM_L / 2 - 0.012
        arm = "".join(
            f'<body name="arm{k}" mocap="true" pos="0 0 {ARM_Z}"><geom type="capsule" size="0.012 {link_hl}" '
            f'contype="0" conaffinity="0" rgba="0.75 0.75 0.8 1"/></body>' for k in (1, 2))
        return f"""
<mujoco model="tabletop">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <visual><global offwidth="{W}" offheight="{H}"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.92 0.94 0.98" rgb2="0.7 0.74 0.82" width="64" height="64"/>
  </asset>
  <worldbody>
    <light pos="0 -0.3 1.8" dir="0 0.15 -1" diffuse="0.9 0.9 0.9"/>
    <geom type="plane" size="2 2 0.1" pos="0 0 -0.02" rgba="0.82 0.84 0.88 1" contype="0" conaffinity="0"/>
    <geom type="box" size="{WS_HALF + 0.02} {WS_HALF + 0.02} 0.01" pos="0 0 -0.01" rgba="0.87 0.82 0.72 1"
          contype="0" conaffinity="0"/>
    {"".join(zones)}
    {walls}
    <geom type="cylinder" size="0.05 {ARM_Z / 2}" pos="{ARM_BASE[0]} {ARM_BASE[1]} {ARM_Z / 2}"
          rgba="0.55 0.55 0.6 1" contype="0" conaffinity="0"/>
    {arm}
    <body name="pusher" mocap="true" pos="0 0 0.05">
      <geom name="pg" type="cylinder" size="{R_EE} 0.05" rgba="0.2 0.2 0.25 1" solref="0.006 1" friction="0.4 0.005 0.0001"/>
    </body>
    {"".join(parts)}
  </worldbody>
</mujoco>"""

    def _index_model(self) -> None:
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        S = range(self.n_slots)
        self._qx = np.array([m.jnt_qposadr[jid(f"o{i}x")] for i in S])
        self._qy = np.array([m.jnt_qposadr[jid(f"o{i}y")] for i in S])
        self._qz = np.array([m.jnt_qposadr[jid(f"o{i}z")] for i in S])
        self._dx = np.array([m.jnt_dofadr[jid(f"o{i}x")] for i in S])
        self._dy = np.array([m.jnt_dofadr[jid(f"o{i}y")] for i in S])
        self._dz = np.array([m.jnt_dofadr[jid(f"o{i}z")] for i in S])
        self._body_pos0 = np.array([m.body_pos[bid(f"obj{i}")][:2] for i in S])
        self._pusher_mid = m.body_mocapid[bid("pusher")]
        self._arm_mid = [m.body_mocapid[bid("arm1")], m.body_mocapid[bid("arm2")]]
        self._g_obj = [gid(f"og{i}") for i in S]
        self._g_pusher = gid("pg")
        self._g_halo = [gid(f"halo{i}") for i in S]
        self._g_zring = [gid(f"zring{k}") for k in range(self.cfg.n_zones)]

    # ------------------------------------------------------------------ state access
    @property
    def state_dim(self) -> int:
        return 2 + 4 * self.n_slots + (self.n_slots if self._typed else 0)

    @property
    def action_dim(self) -> int:
        return 3   # (dx, dy, grip)

    def obj_positions(self) -> np.ndarray:
        """(n_slots, 2) world xy of every slot (parked slots are off-table)."""
        return np.stack([self.data.qpos[self._qx] + self._body_pos0[:, 0],
                         self.data.qpos[self._qy] + self._body_pos0[:, 1]], axis=1)

    def held_mask(self) -> np.ndarray:
        m = np.zeros(self.n_slots)
        if self.holding >= 0:
            m[self.holding] = 1.0
        return m

    def get_state(self) -> np.ndarray:
        pos = self.obj_positions() * self.active[:, None]
        parts = [self.ee, pos.reshape(-1), self.active.astype(float), self.held_mask()]
        if self._typed:
            parts.append((~self.heavy).astype(float))        # graspable mask
        return np.concatenate(parts).astype(np.float32)

    def unpack_state(self, s: np.ndarray):
        """Return (ee_xy, obj_xy (n_slots, 2), active mask, held mask) from a state vector."""
        n = self.n_slots
        return s[:2], s[2:2 + 2 * n].reshape(n, 2), s[2 + 2 * n:2 + 3 * n], s[2 + 3 * n:2 + 4 * n]

    # ------------------------------------------------------------------ task geometry
    def blockers(self, target_obj: int, goal_zone: int) -> list[int]:
        """Active slots (other than the target) inside the straight-path corridor, nearest-to-target first."""
        pos = self.obj_positions()
        a, g = pos[target_obj], self.zone_centers[goal_zone]
        d = g - a
        L = np.linalg.norm(d)
        u = d / max(L, 1e-9)
        out = []
        for j in range(self.n_slots):
            if j == target_obj or not self.active[j]:
                continue
            p = pos[j] - a
            t = p @ u
            lat = abs(p[0] * u[1] - p[1] * u[0])
            if 0.0 < t < L and lat < CORRIDOR:
                out.append((t, j))
        return [j for _, j in sorted(out)]

    def blocked(self, target_obj: int, goal_zone: int) -> bool:
        return len(self.blockers(target_obj, goal_zone)) > 0

    def task_success(self, target_obj: int, goal_zone: int) -> bool:
        """Target centre inside the zone and not currently held."""
        if self.holding == target_obj:
            return False
        p = self.obj_positions()[target_obj]
        return bool(np.linalg.norm(p - self.zone_centers[goal_zone]) < ZONE_SUCCESS_R)

    # ------------------------------------------------------------------ reset
    def reset(self, target_obj=None, goal_zone=None, blocked=None, seed=None) -> np.ndarray:
        """Randomise the scene.

        blocked: True -> 1..n_obstacles_max obstacles placed on the straight path target->zone;
                 False -> none on the path; None -> 50/50.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg, rng = self.cfg, self.rng
        for _ in range(200):
            tgt = int(rng.integers(cfg.n_objects)) if target_obj is None else int(target_obj)
            zone = int(rng.integers(cfg.n_zones)) if goal_zone is None else int(goal_zone)
            want_blocked = bool(rng.random() < 0.5) if blocked is None else bool(blocked)
            n_block = int(rng.integers(1, cfg.n_obstacles_max + 1)) if (want_blocked and cfg.n_obstacles_max > 0) else 0
            placed = self._sample_scene(tgt, zone, n_block)
            if placed is not None:
                break
        else:
            raise RuntimeError("Could not sample a valid scene; relax constraints.")
        pos, active = placed
        self.active = active
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.n_slots):
            p = pos[i] if active[i] else np.array([10.0 + i, 0.0])
            self.data.qpos[self._qx[i]] = p[0] - self._body_pos0[i, 0]
            self.data.qpos[self._qy[i]] = p[1] - self._body_pos0[i, 1]
        for g in self._g_obj:
            self.model.geom_contype[g] = 1
            self.model.geom_conaffinity[g] = 1
        self.holding, self._releasing, self._lift, self._ghost_obj = -1, False, 0.0, -1
        self._off = np.zeros(2)
        self.ee = EE_HOME.copy()
        self._set_ee_mocap(self.ee)
        self._update_pusher_collision()
        mujoco.mj_forward(self.model, self.data)
        self._update_arm()
        self.target_obj, self.goal_zone, self.t = tgt, zone, 0
        self._set_highlight()
        return self.get_state()

    def _set_highlight(self) -> None:
        """Visual only: white halo under the commanded object, ring around the goal zone in that object's colour."""
        rgba = self.model.geom_rgba
        for g in self._g_halo + self._g_zring:
            rgba[g, 3] = 0.0
        rgba[self._g_halo[self.target_obj]] = [1.0, 1.0, 1.0, 0.95]
        c = OBJ_COLORS[self.target_obj % len(OBJ_COLORS)]
        rgba[self._g_zring[self.goal_zone]] = [c[0], c[1], c[2], 1.0]

    def set_task(self, target_obj: int, goal_zone: int) -> None:
        self.target_obj, self.goal_zone = int(target_obj), int(goal_zone)
        self._set_highlight()

    def _sample_scene(self, tgt: int, zone: int, n_block: int):
        cfg, rng = self.cfg, self.rng
        g = self.zone_centers[zone]
        pos = np.zeros((self.n_slots, 2))
        active = np.zeros(self.n_slots, dtype=bool)
        sep = 2 * R_OBJ + 0.015

        def ok_free(p, placed_idx):
            if np.any(np.abs(p) > self._sx + 0.05):
                return False
            return all(np.linalg.norm(p - pos[j]) >= sep for j in placed_idx)

        for _ in range(100):  # target
            p = np.array([rng.uniform(-self._sx, self._sx), rng.uniform(*self._sy)])
            if np.linalg.norm(p - g) >= 0.2 and np.linalg.norm(p - EE_HOME) > 0.1:
                break
        else:
            return None
        pos[tgt], active[tgt] = p, True
        placed = [tgt]
        u = (g - p) / np.linalg.norm(g - p)
        nrm = np.array([-u[1], u[0]])
        L = np.linalg.norm(g - p)

        def in_corridor(q):
            r = q - p
            t = r @ u
            return 0.0 < t < L and abs(r @ nrm) < CORRIDOR + 0.02

        # blockers (obstacle slots, placed on the path)
        for b in range(n_block):
            idx = cfg.n_objects + b
            for _ in range(100):
                q = p + u * rng.uniform(0.35, 0.65) * L + nrm * rng.uniform(-0.02, 0.02)
                if ok_free(q, placed) and np.linalg.norm(q - g) > ZONE_VIS_R + R_OBJ + 0.02:
                    pos[idx], active[idx] = q, True
                    placed.append(idx)
                    break
            else:
                return None
        # distractors (other regular objects, kept off the path)
        for idx in range(cfg.n_objects):
            if idx == tgt:
                continue
            for _ in range(200):
                q = np.array([rng.uniform(-self._sx, self._sx), rng.uniform(*self._sy)])
                if ok_free(q, placed) and not in_corridor(q) and np.linalg.norm(q - EE_HOME) > 0.1:
                    pos[idx], active[idx] = q, True
                    placed.append(idx)
                    break
            else:
                return None
        return pos, active

    # ------------------------------------------------------------------ grasp logic
    def _set_obj_collision(self, i: int, on: bool) -> None:
        self.model.geom_contype[self._g_obj[i]] = 1 if on else 0
        self.model.geom_conaffinity[self._g_obj[i]] = 1 if on else 0

    def _update_pusher_collision(self) -> None:
        ghost = self.holding >= 0
        if not ghost and self._ghost_obj >= 0:
            if np.linalg.norm(self.obj_positions()[self._ghost_obj] - self.ee) < GHOST_R:
                ghost = True
            else:
                self._ghost_obj = -1
        v = 0 if ghost else 1
        self.model.geom_contype[self._g_pusher] = v
        self.model.geom_conaffinity[self._g_pusher] = v

    def _handle_grip(self, want_closed: bool) -> None:
        if not self.cfg.allow_grasp:
            return
        if want_closed:
            if self.holding < 0:
                pos = self.obj_positions()
                d = np.linalg.norm(pos - self.ee, axis=1)
                d[~self.active] = np.inf
                d[self.heavy] = np.inf                      # heavy objects cannot be lifted
                i = int(np.argmin(d))
                if d[i] < GRASP_R:
                    self.holding, self._releasing = i, False
                    self._off = pos[i] - self.ee
                    self._set_obj_collision(i, False)
                    self._ghost_obj = -1
            elif self._releasing:
                self._releasing = False
        elif self.holding >= 0 and not self._releasing:
            self._releasing = True

    def _try_finalize_release(self) -> None:
        i = self.holding
        pos = self.obj_positions()
        p = pos[i]
        others = [j for j in range(self.n_slots) if j != i and self.active[j]]
        clear = all(np.linalg.norm(p - pos[j]) >= 2 * R_OBJ + 0.004 for j in others)
        if clear and np.all(np.abs(p) <= WS_HALF - R_OBJ):
            self._set_obj_collision(i, True)
            self.holding, self._releasing, self._ghost_obj = -1, False, i
        else:
            self._releasing = False   # refused: placement invalid, keep holding

    def _apply_kinematics(self, ee_xy: np.ndarray) -> None:
        d = self.data
        if self.holding >= 0:
            i = self.holding
            self._off *= HOLD_DECAY
            p = ee_xy + self._off
            d.qpos[self._qx[i]] = p[0] - self._body_pos0[i, 0]
            d.qpos[self._qy[i]] = p[1] - self._body_pos0[i, 1]
            d.qvel[self._dx[i]] = 0.0
            d.qvel[self._dy[i]] = 0.0
        z = np.zeros(self.n_slots)
        if self.holding >= 0:
            z[self.holding] = self._lift
        d.qpos[self._qz] = z
        d.qvel[self._dz] = 0.0

    # ------------------------------------------------------------------ dynamics
    def _set_ee_mocap(self, xy: np.ndarray) -> None:
        self.data.mocap_pos[self._pusher_mid] = [xy[0], xy[1], 0.05 + self._lift]

    def step(self, action: np.ndarray):
        """action: (dx, dy[, grip]); xy in [-1, 1] scaled by cfg.max_delta (vector norm clipped to 1).
        grip > 0 closes/holds, grip <= 0 opens/releases; a 2-vector keeps the current grip state.
        Returns (state, done). done = task success or horizon reached."""
        a = np.asarray(action, dtype=float).ravel()
        mv = a[:2]
        n = np.linalg.norm(mv)
        if n > 1.0:
            mv = mv / n
        want_closed = (a[2] > 0.0) if a.size >= 3 else (self.holding >= 0 and not self._releasing)
        self._handle_grip(want_closed)
        target_lift = LIFT if (self.holding >= 0 and not self._releasing) else 0.0
        self._lift += float(np.clip(target_lift - self._lift, -LIFT_RATE, LIFT_RATE))
        if self.holding >= 0 and self._releasing and self._lift <= 1e-9:
            self._try_finalize_release()
        new = np.clip(self.ee + mv * self.cfg.max_delta, -EE_LIM, EE_LIM)
        for k in range(self.cfg.substeps):
            ee_k = self.ee + (new - self.ee) * (k + 1) / self.cfg.substeps
            self._set_ee_mocap(ee_k)
            self._apply_kinematics(ee_k)
            mujoco.mj_step(self.model, self.data)
            self._apply_kinematics(ee_k)
        self.ee = new
        self._update_pusher_collision()
        self._update_arm()
        self.t += 1
        done = self.task_success(self.target_obj, self.goal_zone) or self.t >= self.horizon
        return self.get_state(), done

    def _update_arm(self) -> None:
        """Cosmetic 2-link planar IK: base -> elbow -> ee (links drawn at height ARM_Z + lift)."""
        d = self.ee - ARM_BASE
        r = float(np.clip(np.linalg.norm(d), 1e-3, 2 * ARM_L - 1e-3))
        dn = d / np.linalg.norm(d)
        a = r / 2
        h = np.sqrt(max(ARM_L ** 2 - a ** 2, 0.0))
        elbow = ARM_BASE + dn * a + np.array([dn[1], -dn[0]]) * h   # elbow bends toward +x
        z = ARM_Z + self._lift
        for mid, (p0, p1) in zip(self._arm_mid, [(ARM_BASE, elbow), (elbow, self.ee)]):
            seg = p1 - p0
            ln = max(np.linalg.norm(seg), 1e-6)
            u = seg / ln
            mid_pt = (p0 + p1) / 2
            self.data.mocap_pos[mid] = [mid_pt[0], mid_pt[1], z]
            self.data.mocap_quat[mid] = _quat_z_to_dir(np.array([u[0], u[1], 0.0]))

    # ------------------------------------------------------------------ rendering
    def render(self, mode: str = "auto") -> np.ndarray:
        """RGB frame (H, W, 3) uint8.

        mode: "mujoco"  -> 3D offscreen render (needs OpenGL),
              "topdown" -> matplotlib top-down schematic (works anywhere),
              "auto"    -> try mujoco once, fall back to topdown if it fails.
        """
        if mode == "auto":
            mode = getattr(self, "_render_mode", "mujoco")
            if mode == "mujoco":
                try:
                    out = self._render_mujoco()
                    self._render_mode = "mujoco"
                    return out
                except Exception as e:  # noqa: BLE001
                    print(f"[env.render] MuJoCo offscreen rendering unavailable ({type(e).__name__}); "
                          "falling back to top-down renderer.")
                    self._render_mode = "topdown"
            return self._render_topdown()
        return self._render_mujoco() if mode == "mujoco" else self._render_topdown()

    def _render_mujoco(self) -> np.ndarray:
        if self._renderer is None:
            W, H = self.cfg.render_size
            self._renderer = mujoco.Renderer(self.model, height=H, width=W)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.0, -0.1, 0.0]
            cam.distance = 1.25
            cam.azimuth = 90.0
            cam.elevation = -62.0
            self._cam = cam
        self._renderer.update_scene(self.data, camera=self._cam)
        return self._renderer.render().copy()

    def _render_topdown(self) -> np.ndarray:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        W, H = self.cfg.render_size
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        lim = WS_HALF + 0.03
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((-WS_HALF, -WS_HALF), 2 * WS_HALF, 2 * WS_HALF, fc="#ded4bd", ec="#7a6f60", lw=3))
        for k, (zx, zy) in enumerate(self.zone_centers):
            ax.add_patch(Circle((zx, zy), ZONE_VIS_R, fc=ZONE_COLORS[k % len(ZONE_COLORS)][:3], ec="k",
                                lw=2 if k == self.goal_zone else 0.5, alpha=0.8))
        zx, zy = self.zone_centers[self.goal_zone]
        ax.add_patch(Circle((zx, zy), ZONE_VIS_R + 0.012, fc="none", ec=OBJ_COLORS[self.target_obj % len(OBJ_COLORS)][:3], lw=3.5))
        pos = self.obj_positions()
        for i in range(self.n_slots):
            if not self.active[i]:
                continue
            col = OBJ_COLORS[i % len(OBJ_COLORS)][:3] if i < self.cfg.n_objects else OBSTACLE_COLOR[:3]
            held = i == self.holding
            if held:  # soft drop shadow to suggest it is lifted
                ax.add_patch(Circle(pos[i] + np.array([0.012, -0.012]), R_OBJ, fc="k", alpha=0.25, ec="none"))
            ax.add_patch(Circle(pos[i], R_OBJ * (1.12 if held else 1.0), fc=col,
                                ec="#00a0c0" if held else ("#111" if self.heavy[i] else "k"),
                                lw=3 if (held or i == self.target_obj or self.heavy[i]) else 0.8, zorder=4 if held else 2))
            if self.heavy[i]:
                ax.text(pos[i][0], pos[i][1], "H", ha="center", va="center", fontsize=9, color="w", weight="bold", zorder=5)
        d = self.ee - ARM_BASE
        r = float(np.clip(np.linalg.norm(d), 1e-3, 2 * ARM_L - 1e-3))
        dn = d / np.linalg.norm(d)
        elbow = ARM_BASE + dn * r / 2 + np.array([dn[1], -dn[0]]) * np.sqrt(max(ARM_L ** 2 - (r / 2) ** 2, 0.0))
        ax.plot([ARM_BASE[0], elbow[0], self.ee[0]], [ARM_BASE[1], elbow[1], self.ee[1]], color="#888", lw=4,
                solid_capstyle="round", zorder=1)
        ax.add_patch(Circle(self.ee, R_EE, fc="#00a0c0" if self.holding >= 0 else "#333", ec="k", zorder=5))
        fig.subplots_adjust(0, 0, 1, 1)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return frame

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
