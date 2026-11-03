"""trt_quant.yaml 載入 helper，供所有量化工具共用。"""
from pathlib import Path

import yaml

_SRC_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _SRC_DIR.parent
_DEFAULT_CFG = _SRC_DIR / "configs" / "trt_quant.yaml"


def load_config(path: Path = _DEFAULT_CFG) -> dict:
    """讀取 trt_quant.yaml，把 paths 區段的相對路徑轉成以專案根目錄為基準的絕對路徑。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: (_PROJECT_ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg
