from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent  # Q1/
CONFIG_DIR = SRC_DIR / "configs"


def load_paths(config=CONFIG_DIR / "paths.yaml"):
    """讀取 paths.yaml（三個流程共用的路徑），把相對路徑轉成以 Q1/ 為基準的絕對路徑。"""
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {k: (PROJECT_ROOT / v).resolve() for k, v in cfg.items()}


def load_config(name: str) -> dict:
    """讀取 configs/{name}.yaml（pipeline / eval / quantization），原樣回傳 dict。"""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
