"""quantization.yaml 載入 helper，供所有量化工具共用。"""
from q1.paths import load_config as _load, load_paths


def load_config() -> dict:
    """回傳 quantization.yaml 的內容，並把 paths.yaml 的共用路徑（絕對路徑）放在 cfg["paths"]。"""
    cfg = _load("quantization")
    cfg["paths"] = load_paths()
    return cfg
