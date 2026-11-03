"""TensorRT engine 導出工具。

讀取 trt_quant.yaml，支援兩個子命令：

  export      導出一或多種精度的 TRT engine
  sensitivity 對 INT8 或 FP8 做逐層 sensitivity analysis

使用範例（在 src/ 下執行）：
  python tools/export_trt.py export
  python tools/export_trt.py export --precision int8
  python tools/export_trt.py sensitivity

精度選擇與 skip_layers 皆在 configs/trt_quant.yaml 中設定。
"""
import copy
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._config import load_config  # noqa: E402

# ── 共用前處理 ────────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR uint8 → RGB float32 (1, 3, imgsz, imgsz), [0, 1]，以 letterbox 縮放。"""
    import cv2
    h, w = img.shape[:2]
    scale = imgsz / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    canvas[:nh, :nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]  # (1, 3, H, W)


def _dummy_input(imgsz: int):
    """回傳 YOLO 用的 dummy float32 tensor，shape (1, 3, imgsz, imgsz)。"""
    import torch
    return torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32).cuda()


def _to_tensor(out):
    """YOLO model eval mode 回傳值可能是 tensor 或 (tensor, ...) tuple，統一取主要輸出。"""
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def _build_forward_loop(calib_csv: Path, coda_root: Path, cam: str, imgsz: int):
    """回傳供 ModelOpt calibration 用的 forward_loop callable。"""
    import torch
    from q1.data.coda import CodaSequence

    frames: list[tuple[int, int]] = []
    with open(calib_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frames.append((int(row["seq"]), int(row["frame"])))

    def forward_loop(model):
        for seq_id, frame in frames:
            img = CodaSequence(coda_root, seq_id, cam).read_image(frame)
            tensor = torch.from_numpy(_letterbox(img, imgsz)).cuda()
            model(tensor)

    return forward_loop


def _load_subset(calib_csv: Path, coda_root: Path, cam: str,
                 imgsz: int, n: int) -> list:
    """回傳前 n 幀的 preprocessed tensor list，供 sensitivity analysis 用。"""
    import torch
    from q1.data.coda import CodaSequence

    tensors = []
    with open(calib_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows[:n]:
        img = CodaSequence(coda_root, int(row["seq"]), cam).read_image(int(row["frame"]))
        tensors.append(torch.from_numpy(_letterbox(img, imgsz)).cuda())
    return tensors


# ── Config 建構 ───────────────────────────────────────────────────────────────

def _build_int8_config(calib_algo: str, percentile: float, skip_layers: list[str]) -> dict:
    import nvidia_modelopt.torch.quantization as mtq

    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)

    # TODO: 以 nvidia-modelopt 官方文件確認 algorithm 字串值：
    #   "entropy"    → INT8_DEFAULT_CFG 預設即 entropy-based，不需覆寫
    #   "percentile" → {"type": "percentile", "percentile_list": [percentile]}
    #   "mse"        → "mse"（需確認 ModelOpt 版本是否支援）
    if calib_algo == "percentile":
        config["algorithm"] = {"type": "percentile", "percentile_list": [percentile]}
    elif calib_algo == "mse":
        config["algorithm"] = "mse"
    # "entropy" → 不改 algorithm，INT8_DEFAULT_CFG 預設即 entropy-based

    for layer in skip_layers:
        config["quant_cfg"][f"*{layer}*"] = {"enable": False}

    return config


def _build_fp8_config(skip_layers: list[str]) -> dict:
    import nvidia_modelopt.torch.quantization as mtq

    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for layer in skip_layers:
        config["quant_cfg"][f"*{layer}*"] = {"enable": False}
    return config


# ── TRT engine build（via trtexec）───────────────────────────────────────────

def _build_engine(qdq_onnx: Path, precision_flag: str, engine_path: Path) -> None:
    cmd = ["trtexec", f"--onnx={qdq_onnx}", precision_flag,
           f"--saveEngine={engine_path}"]
    print(f"[trtexec] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[trtexec] engine → {engine_path}")


# ── Q/DQ 導出（INT8）────────────────────────────────────────────────────────

def _export_qdq_int8(weights: Path, calib_csv: Path, coda_root: Path,
                     cam: str, imgsz: int, opset: int,
                     calib_algo: str, percentile: float,
                     skip_layers: list[str], outdir: Path) -> Path:
    import torch
    import nvidia_modelopt.torch.quantization as mtq
    from ultralytics import YOLO

    print("[int8] ModelOpt PTQ 校正中…")
    yolo = YOLO(str(weights))
    model = yolo.model.eval().cuda()

    config = _build_int8_config(calib_algo, percentile, skip_layers)
    forward_loop = _build_forward_loop(calib_csv, coda_root, cam, imgsz)
    mtq.quantize(model, config, forward_loop)

    qdq_path = outdir / "int8_qdq.onnx"
    torch.onnx.export(
        model, _dummy_input(imgsz), str(qdq_path),
        opset_version=opset,
        input_names=["images"], output_names=["output0"],
    )
    print(f"[int8] Q/DQ ONNX → {qdq_path}")
    return qdq_path


# ── Q/DQ 導出（FP8）─────────────────────────────────────────────────────────

def _check_trt_fp8() -> None:
    import tensorrt as trt
    major, minor, _ = (int(x) for x in trt.__version__.split(".")[:3])
    if (major, minor) < (10, 0):
        raise RuntimeError(
            f"FP8 需要 TensorRT >= 10.0，目前版本為 {trt.__version__}"
        )


def _export_qdq_fp8(weights: Path, calib_csv: Path, coda_root: Path,
                    cam: str, imgsz: int, opset: int,
                    skip_layers: list[str], outdir: Path) -> Path:
    import torch
    import nvidia_modelopt.torch.quantization as mtq
    from ultralytics import YOLO

    _check_trt_fp8()
    print("[fp8] ModelOpt FP8 PTQ 校正中…")
    yolo = YOLO(str(weights))
    model = yolo.model.eval().cuda()

    config = _build_fp8_config(skip_layers)
    forward_loop = _build_forward_loop(calib_csv, coda_root, cam, imgsz)
    mtq.quantize(model, config, forward_loop)

    qdq_path = outdir / "fp8_qdq.onnx"
    torch.onnx.export(
        model, _dummy_input(imgsz), str(qdq_path),
        opset_version=opset,
        input_names=["images"], output_names=["output0"],
    )
    print(f"[fp8] Q/DQ ONNX → {qdq_path}")
    return qdq_path


# ── 主導出函式 ────────────────────────────────────────────────────────────────

def export(cfg: dict, precision: str | None = None) -> None:
    """導出指定精度（None = 依 config precisions list 全部導出）。"""
    from ultralytics import YOLO

    paths = cfg["paths"]
    exp = cfg["export"]
    weights = paths["weights"]
    outdir = paths["outdir"] / "engines"
    outdir.mkdir(parents=True, exist_ok=True)

    targets = [precision] if precision else exp["precisions"]

    for prec in targets:
        print(f"\n{'─'*60}\n[export] precision = {prec}")

        if prec == "fp32":
            print("[fp32] ultralytics TRT 直接導出…")
            raw = YOLO(str(weights)).export(
                format="engine", imgsz=exp["imgsz"], device=exp["device"])
            shutil.copy(raw, outdir / "fp32.engine")
            print(f"[fp32] → {outdir / 'fp32.engine'}")

        elif prec == "fp16":
            print("[fp16] ultralytics TRT 直接導出（half=True）…")
            raw = YOLO(str(weights)).export(
                format="engine", imgsz=exp["imgsz"], half=True, device=exp["device"])
            shutil.copy(raw, outdir / "fp16.engine")
            print(f"[fp16] → {outdir / 'fp16.engine'}")

        elif prec == "int8":
            calib_csv = paths["outdir"] / "calib_frames.csv"
            qdq = _export_qdq_int8(
                weights, calib_csv, paths["coda_root"],
                cfg["calibration"]["cam"], exp["imgsz"], exp["onnx_opset"],
                cfg["int8"]["calib_algo"], cfg["int8"]["percentile"],
                cfg["int8"]["skip_layers"], outdir,
            )
            _build_engine(qdq, "--int8", outdir / "int8.engine")

        elif prec == "fp8":
            calib_csv = paths["outdir"] / "calib_frames.csv"
            qdq = _export_qdq_fp8(
                weights, calib_csv, paths["coda_root"],
                cfg["calibration"]["cam"], exp["imgsz"], exp["onnx_opset"],
                cfg["fp8"]["skip_layers"], outdir,
            )
            _build_engine(qdq, "--fp8", outdir / "fp8.engine")

        else:
            print(f"[export] 未知精度 {prec!r}，跳過")


# ── Sensitivity Analysis ──────────────────────────────────────────────────────

def sensitivity(cfg: dict) -> None:
    """逐層 disable 量化，計算 output MSE vs FP32 baseline，輸出 ranking JSON。"""
    import copy
    import torch
    import torch.nn.functional as F
    import nvidia_modelopt.torch.quantization as mtq
    from ultralytics import YOLO

    paths = cfg["paths"]
    exp = cfg["export"]
    sens_cfg = cfg["sensitivity"]
    precision = sens_cfg["precision"]
    n_subset = sens_cfg["n_subset"]
    calib_csv = paths["outdir"] / "calib_frames.csv"
    outdir = paths["outdir"] / "sensitivity"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[sensitivity] precision={precision}, n_subset={n_subset}")

    # FP32 baseline model（不量化）
    fp32_model = YOLO(str(paths["weights"])).model.eval().cuda()

    # 量化 model
    q_model = YOLO(str(paths["weights"])).model.eval().cuda()
    if precision == "int8":
        q_cfg_fn = _build_int8_config
        config = q_cfg_fn(
            cfg["int8"]["calib_algo"], cfg["int8"]["percentile"], []
        )
    else:
        _check_trt_fp8()
        config = _build_fp8_config([])

    forward_loop = _build_forward_loop(
        calib_csv, paths["coda_root"], cfg["calibration"]["cam"], exp["imgsz"]
    )
    mtq.quantize(q_model, config, forward_loop)

    # 校正子集
    subset = _load_subset(
        calib_csv, paths["coda_root"], cfg["calibration"]["cam"],
        exp["imgsz"], n_subset,
    )

    # FP32 baseline outputs
    with torch.no_grad():
        fp32_outs = [fp32_model(x) for x in subset]

    # 逐層分析
    mod_map = {name: mod for name, mod in q_model.named_modules()
               if hasattr(mod, "weight_quantizer") or hasattr(mod, "input_quantizer")}
    quantized_layers = list(mod_map)
    print(f"[sensitivity] 可量化層數：{len(quantized_layers)}")

    ranking: list[dict] = []
    for i, layer_name in enumerate(quantized_layers):
        mod = mod_map[layer_name]
        for attr in ("input_quantizer", "weight_quantizer"):
            if hasattr(mod, attr):
                getattr(mod, attr).disable()
        with torch.no_grad():
            mses = [
                F.mse_loss(_to_tensor(q_model(x)), _to_tensor(ref).detach()).item()
                for x, ref in zip(subset, fp32_outs)
            ]
        for attr in ("input_quantizer", "weight_quantizer"):
            if hasattr(mod, attr):
                getattr(mod, attr).enable()
        mse = float(np.mean(mses))
        ranking.append({"layer": layer_name, "mse": mse})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(quantized_layers)} 層完成…")

    ranking.sort(key=lambda x: x["mse"], reverse=True)

    out_path = outdir / f"{precision}_sensitivity_ranking.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)

    print(f"\n[sensitivity] Top 10 最敏感層：")
    for entry in ranking[:10]:
        print(f"  {entry['layer']:60s}  MSE={entry['mse']:.6f}")
    print(f"\n→ {out_path}")
    print("請將需要跳過的層名填入 configs/trt_quant.yaml 的 skip_layers，再重跑 export。")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    exp_p = sub.add_parser("export", help="導出 TRT engine")
    exp_p.add_argument("--precision", choices=["fp32", "fp16", "int8", "fp8"],
                       help="指定單一精度（預設依 config precisions list 全部導出）")

    sub.add_parser("sensitivity", help="Sensitivity analysis")

    args = p.parse_args()
    cfg = load_config()

    if args.cmd == "export":
        export(cfg, getattr(args, "precision", None))
    else:
        sensitivity(cfg)


if __name__ == "__main__":
    main()
