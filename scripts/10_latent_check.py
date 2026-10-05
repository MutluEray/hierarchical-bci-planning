"""Latent-structure check: how much of the MOTION information lives in z rather than in the effect code c?

At planning time the planner optimises effect codes c and decodes actions with the PRIOR-MEAN motion code z.
If z carries most of the information about how an effect is achieved (e.g. which way to go around a disc to push
it), the prior-mean z loses it and the generated actions do not realise the planned effect.
This script measures, on validation windows of the saved checkpoint, the action reconstruction error when z comes from
  posterior  : q_g(z | s, c, a)       (sees the true actions; the training-time value)
  prior mean : mean of p(z | s, c)    (what the planner uses)
  zero       : z = 0
and compares both with the 'no information' level (error of the subset's mean action sequence).
Subsets: grasp-carry windows, push windows (object moved, nothing held), windows without object motion.

Usage:
    BCIHP_CONFIG=configs/mixed.yaml python3 scripts/10_latent_check.py --ckpt checkpoints/cavin_lite_mixed.pt
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from bcihp import datagen
from bcihp.model import load_model
from bcihp.utils import load_config, set_seed


@torch.no_grad()
def action_errors(net, s, a, s2):
    sn, s2n = net.norm(s), net.norm(s2)
    mu_c, lv_c = net._gauss(net.q_h(torch.cat([sn, s2n], -1)))
    c = net._rsample(mu_c, lv_c)
    mu_qz, lv_qz = net._gauss(net.q_g(torch.cat([sn, c, a.reshape(len(a), -1)], -1)))
    z_post = net._rsample(mu_qz, lv_qz)
    mu_pz, _ = net._gauss(net.p_z(torch.cat([sn, c], -1)))

    def err(z):
        pred = torch.tanh(net.action_gen(torch.cat([sn, c, z], -1))).view(-1, net.H, net.ad)
        return ((pred - a) ** 2).sum((-1, -2))

    return err(z_post), err(mu_pz), err(torch.zeros_like(z_post))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cavin_lite_mixed.pt")
    ap.add_argument("--data", default=None)
    ap.add_argument("--max-n", type=int, default=20000)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    data = datagen.load(str(ROOT / (args.data or cfg["data"]["out_path"])))
    _, va = datagen.split_by_episode(data, cfg["train"].get("val_fraction", 0.1), seed=cfg["seed"])
    net, ck = load_model(str(ROOT / args.ckpt), "cpu")
    n = min(args.max_n, len(va["states"]))
    t = lambda k: torch.as_tensor(va[k][:n], dtype=torch.float32)
    s, a, s2 = t("states"), t("actions"), t("next_states")
    disp, grip = va["obj_disp"][:n], va["used_grip"][:n]
    subsets = {"grasp-carry (object held, moved)": (grip == 1) & (disp > 0.01),
               "push (object moved, nothing held)": (grip == 0) & (disp > 0.01),
               "no object motion": disp <= 0.01}
    print(f"checkpoint {args.ckpt} | betas {ck.get('betas')} | {n} validation windows")
    print(f"{'subset':36} {'n':>6} | {'posterior z':>11} {'prior-mean z':>13} {'zero z':>8} | {'no-information level':>20} | share of information lost with prior-mean z")
    ep, em, ez = action_errors(net, s, a, s2)
    for name, m in subsets.items():
        if m.sum() < 20:
            continue
        mm = torch.as_tensor(m)
        base = float(((a[mm] - a[mm].mean(0, keepdim=True)) ** 2).sum((-1, -2)).mean())
        post, pm, zz = float(ep[mm].mean()), float(em[mm].mean()), float(ez[mm].mean())
        lost = (pm - post) / max(base - post, 1e-9)
        print(f"{name:36} {int(m.sum()):6d} | {post:11.3f} {pm:13.3f} {zz:8.3f} | {base:20.3f} | {lost:6.0%}")
