"""量化結果對照報告產生器。

讀取每個精度的 profiling.json + summary.json + quant_coverage_*.json，
輸出 Markdown 表格到 configs/quantization.yaml 的 compare.out 路徑。

使用範例（在 src/ 下執行）：
  python tools/compare_quant.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._config import load_config  # noqa: E402


# ── 資料讀取 ──────────────────────────────────────────────────────────────────

def _load_profiling(dist_dir: Path) -> dict | None:
    path = dist_dir / "profiling.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_summary(eval_dir: Path) -> dict | None:
    path = eval_dir / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_coverage(cov_dir: Path, prec: str) -> tuple[dict | None, dict | None]:
    onnx_path = cov_dir / f"quant_coverage_{prec}_onnx.json"
    trt_path  = cov_dir / f"quant_coverage_{prec}_trt.json"
    onnx = json.load(open(onnx_path, encoding="utf-8")) if onnx_path.exists() else None
    trt  = json.load(open(trt_path,  encoding="utf-8")) if trt_path.exists()  else None
    return onnx, trt


def _r(v, fmt=".3f"):
    """安全格式化，None 或 nan 輸出 '-'。"""
    if v is None:
        return "-"
    try:
        import math
        if math.isnan(v):
            return "-"
        return format(v, fmt)
    except (TypeError, ValueError):
        return "-"


# ── 指標計算 ──────────────────────────────────────────────────────────────────

def _metrics(label: str, dist_dir: Path, eval_dir: Path,
             cov_dir: Path) -> dict:
    row: dict = {"label": label}

    prof = _load_profiling(dist_dir)
    if prof and "yolo" in prof:
        wall_s = prof["yolo"]["wall_s"]
        n      = prof["meta"]["frames_used"]
        row["fps"]         = n / wall_s if wall_s > 0 else None
        row["lat_mean_ms"] = wall_s * 1000 / n if n > 0 else None
        row["vram_peak_mb"] = prof["yolo"]["vram_peak_mb"]
    else:
        row["fps"] = row["lat_mean_ms"] = row["vram_peak_mb"] = None

    smry = _load_summary(eval_dir)
    if smry:
        row["precision_det"] = smry.get("precision")
        row["recall_det"]    = smry.get("recall")
        de = smry.get("distance_error", {})
        row["mae_m"]  = de.get("mae_m")
        row["p90_m"]  = de.get("p90_abs_m")
    else:
        row["precision_det"] = row["recall_det"] = row["mae_m"] = row["p90_m"] = None

    if label in ("int8", "fp8"):
        onnx_cov, trt_cov = _load_coverage(cov_dir, label)
        if onnx_cov:
            total = len(onnx_cov)
            q = sum(1 for v in onnx_cov.values() if v["weight_quantized"])
            row["quant_coverage"] = f"{q}/{total} ({q/total*100:.0f}%)" if total else "-"
        else:
            row["quant_coverage"] = "-"
        if trt_cov and onnx_cov:
            fallback = sum(
                1 for name, info in trt_cov.items()
                if label not in info["precision"].lower()
            )
            row["trt_fallback"] = str(fallback)
        else:
            row["trt_fallback"] = "-"
    else:
        row["quant_coverage"] = "N/A"
        row["trt_fallback"]   = "N/A"

    return row


# ── Markdown 渲染 ─────────────────────────────────────────────────────────────

_COLS = [
    ("Precision",       "label",         "s"),
    ("FPS",             "fps",           ".1f"),
    ("Lat mean (ms)",   "lat_mean_ms",   ".1f"),
    ("VRAM peak (MB)",  "vram_peak_mb",  ".1f"),
    ("Precision (det)", "precision_det", ".3f"),
    ("Recall",          "recall_det",    ".3f"),
    ("MAE (m)",         "mae_m",         ".3f"),
    ("P90 Err (m)",     "p90_m",         ".3f"),
    ("Quant Coverage",  "quant_coverage","s"),
    ("TRT Fallback 層", "trt_fallback",  "s"),
]


def render_markdown(rows: list[dict]) -> str:
    headers = [c[0] for c in _COLS]
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep)     + " |",
    ]
    for row in rows:
        cells = []
        for _, key, fmt in _COLS:
            v = row.get(key)
            if fmt == "s":
                cells.append(str(v) if v is not None else "-")
            else:
                cells.append(_r(v, fmt))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    labels: list[str] = cfg["compare"]["labels"]
    outdir_base = cfg["paths"]["outputs"]
    cov_dir = outdir_base / "quant_coverage"
    out_path = Path(cfg["compare"]["out"])
    if not out_path.is_absolute():
        out_path = (cfg["paths"]["outputs"].parent / cfg["compare"]["out"]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label in labels:
        dist_dir = outdir_base / f"distance_{label}"
        eval_dir = outdir_base / f"eval_{label}"
        if not dist_dir.exists() and not eval_dir.exists():
            print(f"[compare] {label}: 找不到輸出資料夾，跳過")
            continue
        row = _metrics(label, dist_dir, eval_dir, cov_dir)
        rows.append(row)
        print(f"[compare] {label}: FPS={_r(row['fps'], '.1f')}  "
              f"MAE={_r(row['mae_m'])} m  "
              f"Coverage={row['quant_coverage']}")

    if not rows:
        print("[compare] 沒有可用的結果，請先執行 run_pipeline.py 與 run_eval.py")
        return

    md = render_markdown(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# TensorRT 量化精度 × 速度對照表\n\n")
        f.write(md)

    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
