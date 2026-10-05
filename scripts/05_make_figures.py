"""Day 6: plots + rollout videos from results/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bcihp.utils import load_config

if __name__ == "__main__":
    cfg = load_config()
    print("TODO (Day 6): read results/results.csv, write plots to results/plots and videos to results/videos")
