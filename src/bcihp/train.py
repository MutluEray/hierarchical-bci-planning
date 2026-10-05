"""Training loop for CAVIN-lite (DAY 3-4).

Diagnostics logged every epoch (validation split, episodes kept together):
  rec_s_post     state reconstruction using the POSTERIOR effect code (upper bound of what c can explain)
  rec_s_prior    state prediction using ONE effect code sampled from the PRIOR p(c|s) (what planning gets)
  rec_s_nochange error of the trivial predictor s'' = s (the baseline any useful model must beat)
  cover_min      best-of-32 prior samples (min over samples of the state error): does the prior COVER the true effect?
  *_moved        same, restricted to windows where some object moved >1 cm (where the hard cases are)
  kl_c / kl_z    total KL (nats); active_c / active_z = number of latent dims with KL > 0.05 (collapse check)
Reading guide: rec_s_post << rec_s_nochange means c carries effect information; cover_min close to rec_s_post
means sampling from the prior can reproduce real effects; active_c == 0 means posterior collapse.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from .datagen import split_by_episode
from .model import CavinLite

ACTIVE_KL = 0.05


def _tensors(d: dict, device: str, max_n: int | None = None):
    n = len(d["states"]) if max_n is None else min(max_n, len(d["states"]))
    f = lambda k: torch.as_tensor(d[k][:n], dtype=torch.float32, device=device)
    return f("states"), f("actions"), f("next_states"), f("obj_disp")


@torch.no_grad()
def evaluate(model: CavinLite, T, betas=(0.0, 0.0), n_cover: int = 32, max_n: int = 4000) -> dict:
    model.eval()
    s, a, s2, disp = (x[:max_n] for x in T)
    _, st = model(s, a, s2, beta_c=betas[0], beta_z=betas[1])
    sn, s2n = model.norm(s), model.norm(s2)
    nochange = ((sn - s2n) ** 2).sum(-1)                                   # (B,)
    c = model.sample_effects(s, n_cover)                                  # (B, n, cd)
    pred = model.norm(model.predict_next(s, c))                           # (B, n, sd)
    err = ((pred - s2n[:, None]) ** 2).sum(-1)                            # (B, n)
    moved = disp > 0.01
    out = dict(
        rec_s_post=float(st["rec_s"]), rec_a=float(st["rec_a"]),
        kl_c=float(st["kl_c"]), kl_z=float(st["kl_z"]),
        active_c=int((st["kl_c_dim"] > ACTIVE_KL).sum()), active_z=int((st["kl_z_dim"] > ACTIVE_KL).sum()),
        rec_s_prior=float(err[:, 0].mean()), rec_s_nochange=float(nochange.mean()),
        cover_min=float(err.min(1).values.mean()),
    )
    if moved.any():
        out.update(rec_s_prior_moved=float(err[moved, 0].mean()), rec_s_nochange_moved=float(nochange[moved].mean()),
                   cover_min_moved=float(err[moved].min(1).values.mean()))
    else:
        out.update(rec_s_prior_moved=float("nan"), rec_s_nochange_moved=float("nan"), cover_min_moved=float("nan"))
    out["_kl_c_dim"] = st["kl_c_dim"].cpu().numpy()
    out["_kl_z_dim"] = st["kl_z_dim"].cpu().numpy()
    return out


def plot_training(history: list, kl_c_dim, kl_z_dim, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    g = lambda k: [h.get(k, np.nan) for h in history]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ep, g("rec_s_post"), label="posterior c")
    ax[0, 0].plot(ep, g("rec_s_prior"), label="prior c (1 sample)")
    ax[0, 0].plot(ep, g("cover_min"), label="best of 32 prior samples")
    ax[0, 0].plot(ep, g("rec_s_nochange"), "k--", label="no-change baseline")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_title("Validation state error (normalised)"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(ep, g("train_rec_a"), label="train"); ax[0, 1].plot(ep, g("rec_a"), label="val")
    ax[0, 1].set_title("Action reconstruction error"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(ep, g("kl_c"), label="KL effect c"); ax[1, 0].plot(ep, g("kl_z"), label="KL motion z")
    ax[1, 0].set_title("Validation KL (nats)"); ax[1, 0].legend(fontsize=8)
    w = 0.4
    ax[1, 1].bar(np.arange(len(kl_c_dim)) - w / 2, kl_c_dim, w, label="c dims")
    ax[1, 1].bar(np.arange(len(kl_z_dim)) + w / 2, kl_z_dim, w, label="z dims")
    ax[1, 1].axhline(ACTIVE_KL, color="r", ls=":", lw=1)
    ax[1, 1].set_title("Per-dimension KL at the end (red = 'active' threshold)"); ax[1, 1].legend(fontsize=8)
    for a_ in ax.ravel():
        a_.set_xlabel("epoch")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train(data: dict, cfg: dict, device: str = "cpu", epochs: int | None = None, beta_c: float | None = None,
          beta_z: float | None = None, out_path: str = "checkpoints/cavin_lite.pt", log_dir: str = "results",
          max_n: int | None = None, verbose: bool = True, tag: str = ""):
    mc, tc = cfg["model"], cfg["train"]
    sfx = f"_{tag}" if tag else ""
    epochs = epochs or tc["epochs"]
    beta_c = mc["beta_effect"] if beta_c is None else beta_c
    beta_z = mc["beta_motion"] if beta_z is None else beta_z
    tr, va = split_by_episode(data, tc.get("val_fraction", 0.1), seed=cfg["seed"])
    Ttr, Tva = _tensors(tr, device, max_n), _tensors(va, device)
    n = len(Ttr[0])
    bs = min(tc["batch_size"], n)

    model = CavinLite(int(data["n_slots"]), int(data["H"]), data["actions"].shape[-1],
                      mc["effect_dim"], mc["motion_dim"], mc["hidden"], int(data["state_dim"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=tc["lr"] * 0.05)
    anneal = max(1, tc["kl_anneal_epochs"])
    if verbose:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"train windows {n} | val windows {len(Tva[0])} | params {n_par / 1e6:.2f}M | device {device} | "
              f"beta_c={beta_c} beta_z={beta_z} | epochs {epochs}")

    history, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        k = min(1.0, (ep + 1) / anneal)
        bc, bz = beta_c * k, beta_z * k
        perm = torch.randperm(n, device=device)
        run = {"loss": 0.0, "rec_s": 0.0, "rec_a": 0.0}
        nb = 0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            loss, st = model(Ttr[0][idx], Ttr[1][idx], Ttr[2][idx], beta_c=bc, beta_z=bz,
                             free_bits=mc.get("free_bits", 0.02), state_weight=mc.get("state_weight", 10.0),
                             action_weight=mc.get("action_weight", 1.0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
            opt.step()
            run["loss"] += float(st["loss"]); run["rec_s"] += float(st["rec_s"]); run["rec_a"] += float(st["rec_a"])
            nb += 1
        sched.step()
        ev = evaluate(model, Tva, betas=(beta_c, beta_z))
        kl_c_dim, kl_z_dim = ev.pop("_kl_c_dim"), ev.pop("_kl_z_dim")
        row = dict(epoch=ep + 1, beta_scale=k, train_loss=run["loss"] / nb, train_rec_s=run["rec_s"] / nb,
                   train_rec_a=run["rec_a"] / nb, **ev)
        history.append({kk: (float(v) if not isinstance(v, (int, float)) else v) for kk, v in row.items()})
        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == epochs):
            print(f"ep {ep + 1:3d}/{epochs} | train loss {row['train_loss']:7.3f} | val rec_s post/prior/nochg "
                  f"{ev['rec_s_post']:.4f}/{ev['rec_s_prior']:.4f}/{ev['rec_s_nochange']:.4f} | "
                  f"cover@32 {ev['cover_min']:.4f} | rec_a {ev['rec_a']:.3f} | KL c/z {ev['kl_c']:.2f}/{ev['kl_z']:.2f} | "
                  f"active c/z {ev['active_c']}/{ev['active_z']} | {time.time() - t0:.0f}s")

    final = evaluate(model, Tva, betas=(beta_c, beta_z), n_cover=256, max_n=2000)
    final.pop("_kl_c_dim"); final.pop("_kl_z_dim")
    final = {k: float(v) for k, v in final.items()}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(),
                    meta=dict(n_slots=int(data["n_slots"]), H=int(data["H"]), action_dim=int(data["actions"].shape[-1]),
                              effect_dim=mc["effect_dim"], motion_dim=mc["motion_dim"], hidden=mc["hidden"],
                              state_dim=int(data["state_dim"])),
                    history=history, final=final, cfg=cfg, betas=(beta_c, beta_z)), out_path)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"train_log{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    plot_training(history, kl_c_dim, kl_z_dim, str(log_dir / "plots" / f"training_curves{sfx}.png"))
    if verbose:
        last = history[-1]
        print(f"saved {out_path} | log {log_dir / f'train_log{sfx}.csv'} | plot {log_dir / 'plots' / f'training_curves{sfx}.png'}")
        print(f"moved-window state error: nochange {last['rec_s_nochange_moved']:.4f} | prior(1) "
              f"{last['rec_s_prior_moved']:.4f} | best-of-32 {last['cover_min_moved']:.4f} | "
              f"best-of-256 {final['cover_min_moved']:.4f} | posterior {last['rec_s_post']:.4f}")
    return model, history
