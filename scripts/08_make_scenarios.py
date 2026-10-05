"""Scenario demo: multi-task BCI sessions with a live HUD, saved as high-quality videos + a web page.

Each session = one persistent scene (obstacles on the paths) and a sequence of high-level commands, e.g. 3 tasks.
For every task: noisy decoded commands -> Bayesian confirmation -> committed goal -> hierarchical planner.
The HUD shows the task tally (completed/attempted), phase, the planner's predicted subgoals for the target,
the belief over intents, commands issued and user input in bits, and the measured number of commands direct
low-level control needs at the same decoder accuracy (from results/eval_final.csv if present).
Everything shown is the true outcome of the run, failures included.

Usage:
    python3 scripts/08_make_scenarios.py --ckpt checkpoints/cavin_lite_b05.pt
    python3 scripts/08_make_scenarios.py --ckpt checkpoints/cavin_lite_b05.pt --sessions 6 --tasks 4 --accuracy 0.8
Writes site/<name>/session<k>.mp4 and site/<name>.html (--name, default 'scenarios').
"""
import argparse
import csv
import json
import sys
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from bcihp.commands import IntentChannel, IntentSession, command_to_goal
from bcihp.env import (OBJ_COLORS, OBSTACLE_COLOR, R_EE, R_OBJ, WS_HALF, ZONE_COLORS, ZONE_VIS_R)
from bcihp.planner import HierarchicalPlanner, TorchLatentModel
from bcihp.utils import load_config, make_env, set_seed

OBJ_NAMES = ["red", "blue", "yellow", "purple", "orange", "brown"]
ZONE_NAMES = ["green", "teal", "pink", "olive"]
PANEL_W, PANEL_H = 576, 720


def wolpaw_bits(N, P):
    P = min(max(P, 1.0 / N), 1.0)
    if P >= 1.0 - 1e-12:
        return float(np.log2(N))
    return float(np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1)))


def fit(img, H, W):
    out = np.full((H, W, 3), 250, dtype=np.uint8)
    h, w = min(H, img.shape[0]), min(W, img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


def short(k, n_obj, n_zone):
    o, z = command_to_goal(k, n_obj, n_zone)
    return f"{OBJ_NAMES[o % len(OBJ_NAMES)][:2].capitalize()}\u2192{ZONE_NAMES[z % 4][0].upper()}"


def draw_map(ax, env, pred_xy):
    lim = WS_HALF + 0.03
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((-WS_HALF, -WS_HALF), 2 * WS_HALF, 2 * WS_HALF, fc="#ded4bd", ec="#7a6f60", lw=2))
    tcol = OBJ_COLORS[env.target_obj % len(OBJ_COLORS)][:3]
    for k, (zx, zy) in enumerate(env.zone_centers):
        ax.add_patch(Circle((zx, zy), ZONE_VIS_R, fc=ZONE_COLORS[k % len(ZONE_COLORS)][:3], ec="k", lw=0.5, alpha=0.85))
    zx, zy = env.zone_centers[env.goal_zone]
    ax.add_patch(Circle((zx, zy), ZONE_VIS_R + 0.012, fc="none", ec=tcol, lw=3))
    pos = env.obj_positions()
    for i in range(env.n_slots):
        if not env.active[i]:
            continue
        col = OBJ_COLORS[i % len(OBJ_COLORS)][:3] if i < env.cfg.n_objects else OBSTACLE_COLOR[:3]
        held = i == env.holding
        ax.add_patch(Circle(pos[i], R_OBJ, fc=col, ec="#00a0c0" if held else ("k" if i != env.target_obj else "w"),
                            lw=2.5 if (held or i == env.target_obj) else 0.8, zorder=3))
    if pred_xy is not None and len(pred_xy):
        pts = np.vstack([pos[env.target_obj][None], pred_xy])
        ax.plot(pts[:, 0], pts[:, 1], ls=":", color="k", lw=1.2, zorder=4)
        for j, p in enumerate(pred_xy):
            ax.add_patch(Circle(p, R_OBJ * 0.8, fc="none", ec=tcol, lw=1.8, ls="--", alpha=0.45 + 0.5 * (j + 1) / len(pred_xy), zorder=4))
            ax.text(p[0], p[1], str(j + 1), ha="center", va="center", fontsize=7, zorder=5)
    ax.add_patch(Circle(env.ee, R_EE, fc="#00a0c0" if env.holding >= 0 else "#333", ec="k", zorder=6))


def hud(env, ctx):
    fig = plt.figure(figsize=(PANEL_W / 100, PANEL_H / 100), dpi=100)
    fig.patch.set_facecolor("#fafafa")
    fig.text(0.04, 0.975, ctx["title"], fontsize=12.5, weight="bold", va="top")
    fig.text(0.04, 0.935, ctx["command"], fontsize=10, va="top")
    ax = fig.add_axes([0.02, 0.46, 0.60, 0.43])
    draw_map(ax, env, ctx.get("pred"))
    ax.set_title("planner's view (dashed = predicted subgoals)", fontsize=7.5, pad=2)
    K = len(ctx["belief"])
    ax2 = fig.add_axes([0.72, 0.47, 0.25, 0.39])
    cols = ["#bbbbbb"] * K
    if ctx.get("committed") is not None:
        cols[ctx["committed"]] = "#d62728"
    ax2.barh(range(K), ctx["belief"], color=cols, edgecolor=["k" if k == ctx["intended"] else "none" for k in range(K)], lw=1.2)
    ax2.set_yticks(range(K)); ax2.set_yticklabels(ctx["labels"], fontsize=6 if K > 9 else 7); ax2.invert_yaxis()
    ax2.set_xlim(0, 1); ax2.tick_params(axis="x", labelsize=6)
    ax2.set_title("belief over intents\n(outlined = true intent)", fontsize=7.5)
    y = 0.415
    fig.text(0.04, y, f"Tasks completed: {ctx['done']} / {ctx['attempted']}", fontsize=15, weight="bold", va="top"); y -= 0.055
    marks = ["\u2713" if st else "\u2717" for st in ctx["results"]]
    if len(marks) < ctx["n_tasks"]:
        marks += ["\u25b6"] + ["\u25cb"] * (ctx["n_tasks"] - len(marks) - 1)
    fig.text(0.04, y, "  ".join(marks), fontsize=14, va="top"); y -= 0.06
    if ctx.get("banner"):
        good = ctx.get("banner_color") or ("#1a7f37" if ctx["banner"].startswith("\u2713") else "#b42318")
        fig.text(0.04, y, ctx["banner"], fontsize=12.5, weight="bold", color=good, va="top")
    y -= 0.055
    lines = ctx.get("summary") or [f"Phase: {ctx['phase']}",
                                   f"BCI commands this task: {ctx['n_cmd']}   (decoder accuracy {ctx['acc']:.1f})",
                                   f"Session input so far: {ctx['total_cmd']} commands \u2248 {ctx['bits']:.0f} bits"]
    if not ctx.get("summary"):
        if ctx.get("direct") is not None:
            lines.append(f"Direct cursor control needs \u2248 {ctx['direct']:.0f} commands per task")
        lines.append(f"Step {env.t} / {env.horizon}")
    for ln in lines:
        fig.text(0.04, y, ln, fontsize=9.5, va="top"); y -= 0.038
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return fit(img, PANEL_H, PANEL_W)


class Recorder:
    def __init__(self, env, writer, fps):
        self.env, self.writer, self.fps = env, writer, fps

    def frame(self, ctx, repeat=1):
        left = fit(self.env.render(), PANEL_H, 960)
        f = np.concatenate([left, hud(self.env, ctx)], axis=1)
        for _ in range(repeat):
            self.writer.append_data(f)


def phase_of(env):
    if env.task_success(env.target_obj, env.goal_zone):
        return "done"
    if env._releasing:
        return "releasing"
    if env.holding >= 0:
        return "carrying"
    return "approaching / pushing"


def run_task(env, planner, ch, k, scfg, rec, base_ctx):
    n_obj, n_zone = env.cfg.n_objects, env.cfg.n_zones
    o, z = command_to_goal(k, n_obj, n_zone)
    env.set_task(o, z)
    env.t = 0
    session = IntentSession(ch, k, scfg["commit_threshold"], scfg["max_commands"])
    kind = ""
    if env.cfg.n_heavy > 0:
        kind = " (heavy: must be pushed)" if env.heavy[o] else " (light: can be picked up)"
    ctx = dict(base_ctx, intended=k, belief=session.b.copy(), committed=None, n_cmd=0, phase="waiting for intent", pred=None, banner=None,
               command=f"Fetch the {OBJ_NAMES[o % len(OBJ_NAMES)]} disc \u2192 {ZONE_NAMES[z % 4]} zone{kind}")
    rec.frame(ctx)
    while session.committed is None and env.t < env.horizon:
        session.issue()
        ctx.update(belief=session.b.copy(), n_cmd=session.n_commands, committed=session.committed,
                   phase="decoding command" if session.committed is None else "intent confirmed")
        for _ in range(scfg["cmd_interval"]):
            if env.t >= env.horizon:
                break
            env.step(np.array([0.0, 0.0, -1.0]))
            rec.frame(ctx)
    cmd = session.committed if session.committed is not None else session.map_command()
    gobj, gzone = command_to_goal(cmd, n_obj, n_zone)
    ctx.update(committed=cmd)
    done = env.t >= env.horizon
    while not done:
        out = planner.plan(env.get_state(), [(gobj, gzone, 1.0)])
        ps = out["pred_states"][:, 2 + 2 * gobj: 4 + 2 * gobj]
        ctx["pred"] = ps
        for a in out["actions"]:
            _, done = env.step(a)
            ctx["phase"] = phase_of(env)
            rec.frame(ctx)
            if done or env.t >= env.horizon:
                done = True
                break
    ok = env.task_success(o, z)
    wrong = cmd != k
    res = list(base_ctx["results"]) + [bool(ok)]
    ctx.update(pred=None, phase="done" if ok else "failed", results=res, done=int(sum(res)), attempted=len(res),
               banner="\u2713 task completed" if ok else ("\u2717 wrong goal was confirmed" if wrong else "\u2717 timed out"))
    rec.frame(ctx, repeat=rec.fps)                              # hold the result for one second
    for _ in range(8):                                          # release anything still held, outside the task budget
        if env.holding < 0:
            break
        env.step(np.array([0.0, 0.0, -1.0]))
        rec.frame(ctx)
    return dict(success=bool(ok), n_cmd=session.n_commands, correct=not wrong, steps=int(env.t), ctx=ctx)


def make_tasks(env, rng, n_tasks):
    """Command indices for one session. With heavy objects, light and heavy targets alternate."""
    n_obj, n_zone = env.cfg.n_objects, env.cfg.n_zones
    light = [i for i in range(n_obj) if not env.heavy[i]]
    heavy = [i for i in range(n_obj) if env.heavy[i]]
    rng.shuffle(light)
    rng.shuffle(heavy)
    order = [x for pair in zip_longest(light, heavy) for x in pair if x is not None] if heavy else list(rng.permutation(n_obj))
    zones = rng.permutation(n_zone).tolist()
    tasks = []
    for i in range(n_tasks):
        o = int(order[i % len(order)])
        z = int(zones[i % n_zone] if i < n_zone else zones[(i + 1) % n_zone])
        tasks.append(o + n_obj * z)
    return tasks


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scenario demo: multi-task BCI sessions</title><style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.45}
.note{background:#fff6e0;border-left:4px solid #e0a000;padding:.7rem 1rem;margin:1rem 0;font-size:.92rem}
.row{display:flex;gap:.5rem;flex-wrap:wrap;margin:1rem 0}button{padding:.5rem .9rem;border:1px solid #bbb;border-radius:6px;background:#f7f7f7;cursor:pointer}
button.active{background:#1f77b4;color:#fff;border-color:#1f77b4}video{width:100%;border:1px solid #ccc;border-radius:6px;background:#000}
#cap{margin:.6rem 0}.ok{color:#1a7f37}.bad{color:#b42318}</style></head><body>
<h1>Multi-task sessions under a sparse, noisy intent channel</h1>
<p>__SUMMARY__ <a href="index.html">Single-command clips</a>.</p>
<div class="note"><b>Honest framing.</b> The BCI signal is <b>simulated</b> (discrete commands, confusion matrix, decoder accuracy __ACC__), not EEG;
the arm is cosmetic and grasping is a kinematic weld. Each session is one fixed-seed run; all outcomes are shown, including failures.
The panel shows the planner's predicted subgoals (dashed circles), the belief over intents, and the user input needed compared with direct control.</div>
<div class="row" id="btns"></div><video id="v" controls autoplay muted loop playsinline></video><div id="cap"></div>
<script>const S=__SESSIONS__;const b=document.getElementById('btns'),v=document.getElementById('v'),c=document.getElementById('cap');
function show(i){const s=S[i];v.src=s.file;v.play();[...b.children].forEach((x,j)=>x.classList.toggle('active',j===i));
c.innerHTML='<b>Session '+(i+1)+'</b> &mdash; '+s.done+'/'+s.n+' tasks completed &middot; '+s.cmds+' BCI commands in total &middot; '+s.results.map(r=>r?'<span class="ok">\u2713</span>':'<span class="bad">\u2717</span>').join(' ');}
S.forEach((s,i)=>{const x=document.createElement('button');x.textContent='Session '+(i+1)+' ('+s.done+'/'+s.n+')';x.onclick=()=>show(i);b.appendChild(x);});show(0);</script></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cavin_lite_b05.pt")
    ap.add_argument("--accuracy", type=float, default=0.8)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--scene-seed", type=int, default=8000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--name", default="scenarios", help="output name: site/<name>/session<k>.mp4 and site/<name>.html")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import imageio.v2 as imageio
    from bcihp.model import load_model

    cfg = load_config()
    set_seed(cfg["seed"])
    cfg["env"]["render_size"] = [960, 720]
    net, _ = load_model(str(ROOT / args.ckpt), args.device)
    env = make_env(cfg)
    planner = HierarchicalPlanner(TorchLatentModel(net, args.device), cfg, env.zone_centers, env.n_slots)
    K = cfg["commands"]["vocab_size"]
    n_obj, n_zone = env.cfg.n_objects, env.cfg.n_zones
    labels = [short(k, n_obj, n_zone) for k in range(K)]

    direct = None
    csv_path = ROOT / "results" / "eval_final.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            rows = [r for r in csv.DictReader(f) if r["method"] == "direct" and abs(float(r["accuracy"]) - args.accuracy) < 1e-6]
        if rows:
            direct = float(np.mean([float(r["n_commands"]) for r in rows]))

    out_dir = ROOT / "site" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    bits_per_cmd = wolpaw_bits(K, args.accuracy)
    meta = []
    for s in range(args.sessions):
        rng = np.random.default_rng(args.scene_seed + s)
        tasks = make_tasks(env, rng, args.tasks)
        env.reset(target_obj=tasks[0] % n_obj, goal_zone=(tasks[0] // n_obj) % n_zone, blocked=True, seed=args.scene_seed + s)
        path = out_dir / f"session{s + 1}.mp4"
        results, total_cmd, last_ctx = [], 0, None
        writer = imageio.get_writer(path, fps=args.fps, codec="libx264", quality=8, macro_block_size=16)
        rec = Recorder(env, writer, args.fps)
        for ti, k in enumerate(tasks):
            base = dict(title=f"Session {s + 1} \u00b7 task {ti + 1} of {len(tasks)}", labels=labels, acc=args.accuracy, n_tasks=len(tasks),
                        results=list(results), done=int(sum(results)), attempted=len(results), total_cmd=total_cmd,
                        bits=total_cmd * bits_per_cmd, direct=direct)
            planner.reseed(1000 + 10 * s + ti)
            ch = IntentChannel(K, args.accuracy, cfg["commands"]["neighbor_bias"], seed=cfg["seed"] + 10 * s + ti)
            r = run_task(env, planner, ch, k, cfg["session"], rec, base)
            results.append(r["success"])
            total_cmd += r["n_cmd"]
            last_ctx = r["ctx"]
            print(f"session {s + 1} task {ti + 1}: command {k} | success={r['success']} commands={r['n_cmd']} committed_correct={r['correct']}")
        n_done = int(sum(results))
        summary = [f"BCI input: {total_cmd} commands (about {total_cmd * bits_per_cmd:.0f} bits)"]
        if direct is not None:
            summary.append(f"Direct cursor control would need about {direct * len(results):.0f} commands")
            summary.append("for the same number of tasks (measured average).")
        end = dict(last_ctx, title=f"Session {s + 1} complete", command=f"{n_done} of {len(results)} tasks completed", pred=None,
                   banner=f"{n_done} / {len(results)} tasks completed", banner_color="#1a7f37" if n_done == len(results) else "#b26a00",
                   phase="session complete", summary=summary, results=results, done=n_done, attempted=len(results))
        rec.frame(end, repeat=2 * args.fps)                      # end card: hold for two seconds
        writer.close()
        meta.append(dict(file=f"{args.name}/session{s + 1}.mp4", done=int(sum(results)), n=len(results), cmds=int(total_cmd), results=results))
        print(f"session {s + 1}: {sum(results)}/{len(results)} tasks completed -> {path}")

    tot_done, tot_n = sum(m["done"] for m in meta), sum(m["n"] for m in meta)
    summary = f"{tot_done} of {tot_n} tasks completed across {len(meta)} sessions at decoder accuracy {args.accuracy:.1f}."
    (ROOT / "site" / f"{args.name}.html").write_text(PAGE.replace("__SESSIONS__", json.dumps(meta)).replace("__ACC__", f"{args.accuracy:.1f}")
                                                   .replace("__SUMMARY__", summary))
    print(summary, f"| wrote site/{args.name}.html")
