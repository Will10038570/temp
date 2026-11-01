from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent  # Q1/


def load_paths(config=SRC_DIR / "configs" / "paths.yaml"):
    """讀取 paths.yaml，把相對路徑轉成以 Q1/ 為基準的絕對路徑。"""
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {k: (PROJECT_ROOT / v).resolve() for k, v in cfg.items()}
