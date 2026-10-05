"""Day 7: rollout clips for every high-level command, a filmstrip figure, and the static demo page.

For each of the K commands: reset a fixed scene (obstacles on the path), run the BCI confirmation session
(noisy decoder) and the hierarchical planner, and save results/videos/... clips into site/clips/. The clips are
generated from fixed seeds and the page shows the TRUE outcome of each (success or failure); nothing is
cherry-picked. Writes: site/clips/cmd<k>.mp4, site/index.html, docs/figures/filmstrip.png.

Usage:
    python3 scripts/07_make_demo.py --ckpt checkpoints/cavin_lite_b05.pt
    python3 scripts/07_make_demo.py --ckpt checkpoints/cavin_lite_agg.pt --name v2      # keeps the existing clips
"""
import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from bcihp import eval as ev
from bcihp.commands import IntentChannel, command_to_goal
from bcihp.planner import HierarchicalPlanner, TorchLatentModel
from bcihp.utils import load_config, make_env, set_seed

OBJ_NAMES = ["red", "blue", "yellow", "purple", "orange", "brown"]
ZONE_NAMES = ["green", "teal", "pink", "olive"]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hierarchical planning under a sparse BCI intent channel: demo</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.45}
 .note{background:#fff6e0;border-left:4px solid #e0a000;padding:.7rem 1rem;margin:1rem 0;font-size:.92rem}
 .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin:1rem 0}
 button{padding:.6rem;border:1px solid #bbb;border-radius:6px;background:#f7f7f7;cursor:pointer;font-size:.9rem}
 button.active{background:#1f77b4;color:#fff;border-color:#1f77b4}
 video{width:100%;border:1px solid #ccc;border-radius:6px;background:#000}
 #cap{margin:.6rem 0;font-size:.95rem}.ok{color:#1a7f37}.bad{color:#b42318}
</style></head><body>
<h1>One sparse command, a full manipulation sequence</h1>
<p>A simulated BCI user issues a high-level command through a noisy decoder. A learned hierarchical latent planner
(CAVIN-lite) turns the committed command into the multi-step arm motion. Pick a command:</p>
<div class="note"><b>Honest framing.</b> The BCI signal is <b>simulated</b>: a discrete command vocabulary with a
confusion matrix (decoder accuracy __ACC__), not real EEG. The arm is cosmetic and grasping is a kinematic weld.
Each clip is one fixed-seed run and its true outcome is shown, including failures. In the clips the commanded object
has a white halo under it and the goal zone has a ring in that object's colour; object and zone colours are independent
because a command selects both.</div>
<div class="grid" id="btns"></div>
<video id="v" controls autoplay muted loop playsinline></video>
<div id="cap"></div>
<script>
const clips = __CLIPS__;
const btns = document.getElementById('btns'), v = document.getElementById('v'), cap = document.getElementById('cap');
function show(i){
  const c = clips[i]; v.src = c.file; v.play();
  [...btns.children].forEach((b,j)=>b.classList.toggle('active', j===i));
  cap.innerHTML = '<b>' + c.label + '</b> &mdash; ' + (c.success ? '<span class="ok">task succeeded</span>' : '<span class="bad">task failed</span>')
    + ' &middot; ' + c.commands + ' BCI command(s) &middot; ' + c.steps + ' simulation steps'
    + ' &middot; command decoded ' + (c.correct ? 'correctly' : 'incorrectly (planner pursued the wrong goal)');
}
clips.forEach((c,i)=>{const b=document.createElement('button'); b.textContent=c.label; b.onclick=()=>show(i); btns.appendChild(b);});
show(0);
</script></body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/cavin_lite_b05.pt")
    ap.add_argument("--accuracy", type=float, default=0.8)
    ap.add_argument("--scene-seed", type=int, default=7000)
    ap.add_argument("--blocked", type=int, default=1, choices=[0, 1])
    ap.add_argument("--name", default="index", help="'index' writes site/index.html + site/clips; any other name writes site/<name>.html + site/<name>_clips")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import imageio.v2 as imageio
    from bcihp.model import load_model

    cfg = load_config()
    set_seed(cfg["seed"])
    net, _ = load_model(str(ROOT / args.ckpt), args.device)
    env = make_env(cfg)
    planner = HierarchicalPlanner(TorchLatentModel(net, args.device), cfg, env.zone_centers, env.n_slots)
    K = cfg["commands"]["vocab_size"]
    clips_name = "clips" if args.name == "index" else f"{args.name}_clips"
    clips_dir = ROOT / "site" / clips_name
    clips_dir.mkdir(parents=True, exist_ok=True)

    meta, strip_frames = [], None
    for k in range(K):
        o, z = command_to_goal(k, env.cfg.n_objects, env.cfg.n_zones)
        env.reset(target_obj=o, goal_zone=z, blocked=bool(args.blocked), seed=args.scene_seed + k)
        ch = IntentChannel(K, args.accuracy, cfg["commands"]["neighbor_bias"], seed=cfg["seed"] + k)
        planner.reseed(1000 + k)
        frames = []
        r = ev.run_hier_episode(env, planner, ch, k, cfg["session"], frames)
        path = clips_dir / f"cmd{k}.mp4"
        imageio.mimsave(path, frames, fps=20)
        label = f"Fetch the {OBJ_NAMES[o % len(OBJ_NAMES)]} object \u2192 {ZONE_NAMES[z % 4]} zone"
        meta.append(dict(k=k, label=label, file=f"{clips_name}/cmd{k}.mp4", success=bool(r["success"]), commands=int(r["n_commands"]),
                         steps=int(r["steps"]), correct=bool(r["committed_correct"])))
        print(f"cmd {k}: {label} | success={r['success']} commands={r['n_commands']} steps={r['steps']}")
        if strip_frames is None and r["success"] and len(frames) > 12:
            strip_frames = frames
    if strip_frames is None:
        strip_frames = frames

    (ROOT / "site" / f"{args.name}.html").write_text(PAGE.replace("__CLIPS__", json.dumps(meta)).replace("__ACC__", f"{args.accuracy:.1f}"))
    idx = np.linspace(0, len(strip_frames) - 1, 8).astype(int)
    tiles = [strip_frames[i] for i in idx]
    rows = [np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)]
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    imageio.imwrite(ROOT / "docs" / "figures" / ("filmstrip.png" if args.name == "index" else f"filmstrip_{args.name}.png"), np.concatenate(rows, 0))
    n_ok = sum(m["success"] for m in meta)
    print(f"{n_ok}/{K} demo clips succeeded | wrote site/{args.name}.html")
