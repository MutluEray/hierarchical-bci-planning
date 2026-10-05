"""Days 3-4: train CAVIN-lite on data/transitions.npz.

Usage:
    python3 scripts/03_train.py --smoke            # 3 epochs on 5k windows (~seconds): checks the pipeline
    python3 scripts/03_train.py                    # full training from configs/default.yaml
    python3 scripts/03_train.py --beta-c 0.5 --tag b05   # KL-weight sweep; outputs get a suffix, nothing is overwritten
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcihp import datagen
from bcihp.train import train
from bcihp.utils import load_config, set_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="cpu is usually faster than mps for these small MLPs")
    ap.add_argument("--beta-c", type=float, default=None)
    ap.add_argument("--beta-z", type=float, default=None)
    ap.add_argument("--tag", default="", help="suffix for checkpoint / log / plot names (for sweeps)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    data = datagen.load(str(ROOT / (args.data or cfg["data"]["out_path"])))
    epochs, max_n = args.epochs, None
    out = args.out or (f"checkpoints/cavin_lite_{args.tag}.pt" if args.tag else "checkpoints/cavin_lite.pt")
    if args.smoke:
        epochs, max_n, out = 3, 5000, "checkpoints/smoke.pt"
    t0 = time.time()
    train(data, cfg, device=args.device, epochs=epochs, beta_c=args.beta_c, beta_z=args.beta_z,
          out_path=str(ROOT / out), log_dir=str(ROOT / "results"), max_n=max_n, tag=args.tag)
    print(f"done in {time.time() - t0:.0f}s")
