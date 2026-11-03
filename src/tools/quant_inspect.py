"""量化覆蓋率檢查工具。

輸出：
  {outdir}/quant_coverage_onnx.json  每個 Conv/Gemm op 的 Q/DQ 覆蓋狀態與 scale
  {outdir}/quant_coverage_trt.json   TRT engine 每層實際推論精度（含 fallback 偵測）

使用範例（在 src/ 下執行）：
  python tools/quant_inspect.py int8
  python tools/quant_inspect.py fp8
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._config import load_config  # noqa: E402


# ── ONNX 覆蓋率 ───────────────────────────────────────────────────────────────

def inspect_onnx(qdq_onnx: Path) -> dict:
    """掃描 ONNX graph 中每個 Conv/Gemm 的 Q/DQ 覆蓋狀態。

    回傳 {op_name: {type, weight_quantized, act_quantized, weight_scale, act_scale}}
    """
    import onnx
    import onnx_graphsurgeon as gs

    graph = gs.import_onnx(onnx.load(str(qdq_onnx)))

    # 建立 tensor name → producer op 的 lookup
    producer: dict[str, gs.Node] = {}
    for node in graph.nodes:
        for out in node.outputs:
            producer[out.name] = node

    def _is_qdq(tensor) -> bool:
        return (tensor.name in producer and
                producer[tensor.name].op == "QuantizeLinear")

    def _get_scale(tensor) -> float | None:
        if tensor.name not in producer:
            return None
        q_node = producer[tensor.name]
        if q_node.op != "QuantizeLinear":
            return None
        # scale 是 QuantizeLinear 的第二個輸入（常數）
        scale_input = q_node.inputs[1]
        if hasattr(scale_input, "values"):
            v = scale_input.values
            return float(v.flat[0]) if hasattr(v, "flat") else float(v)
        return None

    result = {}
    for node in graph.nodes:
        if node.op not in ("Conv", "Gemm", "MatMul"):
            continue
        act_t = node.inputs[0] if node.inputs else None
        w_t   = node.inputs[1] if len(node.inputs) > 1 else None
        result[node.name] = {
            "type":              node.op,
            "act_quantized":     _is_qdq(act_t) if act_t else False,
            "weight_quantized":  _is_qdq(w_t)  if w_t  else False,
            "act_scale":         _get_scale(act_t) if act_t else None,
            "weight_scale":      _get_scale(w_t)   if w_t  else None,
        }
    return result


# ── TRT engine 覆蓋率 ─────────────────────────────────────────────────────────

def inspect_engine(engine_path: Path) -> dict:
    """用 trtexec --exportLayerInfo 取每層實際推論精度。

    回傳 {layer_name: {precision, tactic}}
    """
    tmp = engine_path.with_suffix(".layers.json")
    cmd = ["trtexec",
           f"--loadEngine={engine_path}",
           f"--exportLayerInfo={tmp}",
           "--noDataTransfers"]
    print(f"[trtexec] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    with open(tmp, encoding="utf-8") as f:
        layers = json.load(f)

    result = {}
    for layer in layers:
        result[layer["Name"]] = {
            "precision": layer.get("Precision", layer.get("ParameterType", "unknown")),
            "tactic":    layer.get("TacticName", layer.get("Tactic", "")),
        }
    return result


# ── 比對 fallback ─────────────────────────────────────────────────────────────

def _print_summary(onnx_cov: dict, trt_cov: dict, precision: str) -> None:
    total = len(onnx_cov)
    q_weight = sum(1 for v in onnx_cov.values() if v["weight_quantized"])
    q_act    = sum(1 for v in onnx_cov.values() if v["act_quantized"])
    print(f"\nONNX Q/DQ 覆蓋率（{precision}）：")
    print(f"  Conv/Gemm 總數：{total}")
    print(f"  weight 量化：{q_weight} / {total}  ({q_weight/total*100:.1f}%)")
    print(f"  activation 量化：{q_act} / {total}  ({q_act/total*100:.1f}%)")

    tgt_prec = precision  # "int8" | "fp8"
    fallback = [name for name, info in trt_cov.items()
                if tgt_prec not in info["precision"].lower()
                and "fp32" not in info["precision"].lower()]
    non_target = [name for name, info in trt_cov.items()
                  if tgt_prec not in info["precision"].lower()]
    print(f"\nTRT engine 精度分佈（{precision}）：")
    from collections import Counter
    prec_dist = Counter(v["precision"] for v in trt_cov.values())
    for prec, cnt in sorted(prec_dist.items()):
        print(f"  {prec:10s}: {cnt} 層")
    print(f"  非目標精度層數：{len(non_target)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("precision", choices=["int8", "fp8"], help="要檢查的精度")
    args = p.parse_args()

    cfg = load_config()
    engine_dir = cfg["paths"]["outdir"] / "engines"
    outdir = cfg["paths"]["outdir"] / "quant_coverage"
    outdir.mkdir(parents=True, exist_ok=True)

    prec = args.precision
    qdq_onnx   = engine_dir / f"{prec}_qdq.onnx"
    engine_path = engine_dir / f"{prec}.engine"

    print(f"[inspect] ONNX: {qdq_onnx}")
    onnx_cov = inspect_onnx(qdq_onnx)
    onnx_out = outdir / f"quant_coverage_{prec}_onnx.json"
    with open(onnx_out, "w", encoding="utf-8") as f:
        json.dump(onnx_cov, f, ensure_ascii=False, indent=2)
    print(f"[inspect] ONNX coverage → {onnx_out}")

    print(f"[inspect] TRT engine: {engine_path}")
    trt_cov = inspect_engine(engine_path)
    trt_out = outdir / f"quant_coverage_{prec}_trt.json"
    with open(trt_out, "w", encoding="utf-8") as f:
        json.dump(trt_cov, f, ensure_ascii=False, indent=2)
    print(f"[inspect] TRT coverage → {trt_out}")

    _print_summary(onnx_cov, trt_cov, prec)


if __name__ == "__main__":
    main()
