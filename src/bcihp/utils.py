"""Shared helpers: config loading, seeding, paths."""
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    """Load a YAML config. Default: configs/default.yaml, or the file named by the BCIHP_CONFIG environment variable."""
    import os
    path = path or os.environ.get("BCIHP_CONFIG", "configs/default.yaml")
    with open(ROOT / path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_env(cfg: dict, seed=None):
    """Build a TabletopEnv from the config dict (keys not in SceneConfig are ignored)."""
    from .env import SceneConfig, TabletopEnv
    keys = SceneConfig.__dataclass_fields__
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg["env"].items() if k in keys}
    return TabletopEnv(SceneConfig(**kwargs), seed=cfg["seed"] if seed is None else seed)
