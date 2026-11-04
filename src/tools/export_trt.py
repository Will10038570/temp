"""TensorRT engine 導出工具。

讀取 quantization.yaml，支援兩個子命令：

  export      導出一或多種精度的 TRT engine
  sensitivity 對 INT8 或 FP8 做逐層 sensitivity analysis

使用範例（在 src/ 下執行）：
  python tools/export_trt.py export
  python tools/export_trt.py export --precision int8
  python tools/export_trt.py sensitivity

精度選擇與 skip_layers 皆在 configs/quantization.yaml 中設定。

導出路徑：
  fp32 / fp16  ultralytics 直接導出 TRT engine
  int8 / fp8   ultralytics FP32 ONNX → modelopt.onnx.quantization（ONNX 層 PTQ，插入 Q/DQ）
               → TensorRT Python API build engine（不需要 trtexec）
sensitivity 仍用 PyTorch 端 fake-quant（modelopt.torch.quantization）做逐層 MSE 比較。
"""
import copy
import csv
import json
import shutil
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


# ── Config 建構（PyTorch 端，僅 sensitivity analysis 使用）──────────────────────

def _build_int8_config(skip_layers: list[str]) -> dict:
    """PyTorch fake-quant 設定（mtq.quantize 用）。

    只用於 sensitivity analysis：PyTorch 端量化模型在 torch 2.14 + Windows 下無法 trace 匯出 ONNX
    （access violation），但 eager forward 正常，足以做逐層 MSE 比較。
    nvidia-modelopt 0.48 PyTorch 端頂層 algorithm 沒有 entropy，固定用 INT8_DEFAULT_CFG 預設的 "max"。
    """
    import modelopt.torch.quantization as mtq

    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    _append_skip_layers(config, skip_layers)
    return config


def _append_skip_layers(config: dict, skip_layers: list[str]) -> None:
    """將 skip_layers 加入 quant_cfg 以停用量化。

    nvidia-modelopt 0.48 的 quant_cfg 是 list[{"quantizer_name", "enable"/"cfg"}]（後面的規則覆蓋前面的），
    舊版是 dict[pattern → cfg]；兩種格式都處理。
    """
    quant_cfg = config["quant_cfg"]
    for layer in skip_layers:
        pattern = f"*{layer}*"
        if isinstance(quant_cfg, list):
            quant_cfg.append({"quantizer_name": pattern, "enable": False})
        else:
            quant_cfg[pattern] = {"enable": False}


def _build_fp8_config(skip_layers: list[str]) -> dict:
    import modelopt.torch.quantization as mtq

    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    _append_skip_layers(config, skip_layers)
    return config


# ── ONNX 層 PTQ（INT8 / FP8）─────────────────────────────────────────────────
#
# 流程：ultralytics 導出 FP32 ONNX → modelopt.onnx.quantization 插入 Q/DQ（用 CODa 校正幀）
#       → TensorRT Python API build engine（前面加 ultralytics metadata header，YOLO() 才能載入）
# 不走 PyTorch 端 mtq.quantize + torch.onnx.export：量化後模型在 torch 2.14 + Windows 下 trace 會 access violation。

def _export_fp32_onnx(weights: Path, imgsz: int, opset: int, outdir: Path) -> Path:
    """ultralytics 導出靜態 shape 的 FP32 ONNX（Detect head 會自動切到 export 模式）。"""
    from ultralytics import YOLO

    raw = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=opset,
                                     simplify=True, dynamic=False)
    dst = outdir / "fp32.onnx"
    shutil.copy(raw, dst)
    print(f"[onnx] FP32 ONNX → {dst}")
    return dst


def _module_to_onnx_regex(layer: str) -> str:
    """PyTorch 模組名 → ONNX 節點名 regex（torch.onnx scope 命名規則）。

    例：model.23.cv3.0.0.0.conv → ^/model\\.23/cv3\\.0/cv3\\.0\\.0/cv3\\.0\\.0\\.0/conv/
    規則：名稱後第一個數字併入該名稱（model.23），之後每個數字各自形成一層 scope（cv3.0/cv3.0.0）。
    若字串以 "/" 開頭，視為使用者直接給的 ONNX 節點 regex，原樣回傳。
    """
    import re
    if layer.startswith("/"):
        return layer
    scopes: list[str] = []
    prev_digit = False
    for part in layer.split("."):
        if part.isdigit() and scopes:
            if prev_digit:                      # 數字接數字 → 新一層 scope
                scopes.append(f"{scopes[-1]}.{part}")
            else:                               # 名稱接數字 → 併入該名稱
                scopes[-1] = f"{scopes[-1]}.{part}"
            prev_digit = True
        else:
            scopes.append(part)
            prev_digit = False
    return "^/" + "/".join(re.escape(s) for s in scopes) + "/"


class _CodaCalibReader:
    """onnxruntime CalibrationDataReader：逐幀讀 CODa 校正影像（不一次載入全部，避免 OOM）。"""

    def __init__(self, calib_csv: Path, coda_root: Path, cam: str, imgsz: int,
                 input_name: str = "images"):
        from q1.data.coda import CodaSequence
        with open(calib_csv, encoding="utf-8") as f:
            self._frames = [(int(r["seq"]), int(r["frame"])) for r in csv.DictReader(f)]
        if not self._frames:
            raise ValueError(f"校正幀清單是空的：{calib_csv}")
        self._it = iter(self._frames)
        self._seq_cls = CodaSequence
        self._coda_root, self._cam, self._imgsz = coda_root, cam, imgsz
        self._input_name = input_name
        self._first = None
        self.n = len(self._frames)

    def _load(self, seq_id: int, frame: int) -> dict:
        img = self._seq_cls(self._coda_root, seq_id, self._cam).read_image(frame)
        return {self._input_name: np.ascontiguousarray(_letterbox(img, self._imgsz))}

    def get_next(self):
        nxt = next(self._it, None)
        return None if nxt is None else self._load(*nxt)

    def get_first(self):
        """modelopt 用來試跑 session：回傳第一幀但不推進 iterator（同 modelopt CalibrationDataProvider）。"""
        if self._first is None:
            self._first = self._load(*self._frames[0])
        return self._first

    def rewind(self):
        self._it = iter(self._frames)


def _make_calib_reader(calib_csv, coda_root, cam, imgsz):
    """包成真正的 onnxruntime CalibrationDataReader 子類別（modelopt 會做 isinstance 檢查）。"""
    from onnxruntime.quantization import CalibrationDataReader

    class Reader(_CodaCalibReader, CalibrationDataReader):
        pass

    return Reader(calib_csv, coda_root, cam, imgsz)


def _quantize_onnx(fp32_onnx: Path, mode: str, calib_csv: Path, coda_root: Path,
                   cam: str, imgsz: int, calib_algo: str, skip_layers: list[str],
                   nodes_to_exclude: list[str], calibration_eps: list[str],
                   outdir: Path) -> Path:
    """modelopt.onnx.quantization：FP32 ONNX → Q/DQ ONNX（mode = "int8" | "fp8"）。"""
    from modelopt.onnx.quantization import quantize as onnx_quantize

    # nvidia-modelopt 0.48 ONNX 端 int8/fp8 的 calibration_method 只有 entropy（預設）/ max
    if calib_algo not in ("entropy", "max"):
        raise ValueError(f"calib_algo={calib_algo!r} 不受支援，ONNX PTQ 只接受 'entropy' | 'max'")

    exclude = [_module_to_onnx_regex(l) for l in skip_layers] + list(nodes_to_exclude)
    reader = _make_calib_reader(calib_csv, coda_root, cam, imgsz)
    out = outdir / f"{mode}_qdq.onnx"
    print(f"[{mode}] ModelOpt ONNX PTQ：calib={calib_algo}，{reader.n} 幀，"
          f"EP={calibration_eps}，排除 {len(exclude)} 條規則")
    for e in exclude:
        print(f"  exclude: {e}")

    onnx_quantize(
        str(fp32_onnx),
        quantize_mode=mode,
        calibration_data_reader=reader,
        calibration_method=calib_algo,
        calibration_eps=calibration_eps,
        # 保留 FP32 圖（modelopt 預設 fp16 會把圖轉 FP16 + Cast），理由見 _build_engine docstring
        high_precision_dtype="fp32",
        nodes_to_exclude=exclude or None,
        output_path=str(out),
    )
    print(f"[{mode}] Q/DQ ONNX → {out}")
    return out


def _ultralytics_metadata(onnx_path: Path) -> dict:
    """取 ultralytics 寫在 ONNX metadata_props 的 metadata（字串值；AutoBackend 載入時會自行轉型）。"""
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)
    return {p.key: p.value for p in m.metadata_props}


def _build_engine(qdq_onnx: Path, precision: str, engine_path: Path, metadata: dict) -> None:
    """TensorRT Python API 將 Q/DQ ONNX build 成 engine（explicit quantization，不需 calibrator）。

    檔頭格式與 ultralytics 導出的 .engine 相同（4-byte little-endian 長度 + metadata JSON + engine bytes），
    q1.detection.yolo 才能用 YOLO("xxx.engine") 載入並取得 class names 等資訊。

    只開 INT8 / FP8 flag，不開 FP16：在 TensorRT 10.16 + RTX 5070 Laptop 上實測，只要開 FP16 flag
    （或用 STRONGLY_TYPED 跑 FP16 圖），build 多數會失敗於 MyelinCheckException
    （"Could not find any implementation for node {ForeignNode[...]}"），根本原因未查明；
    FP32 圖 + 只開 INT8 flag 則 5/5 次成功，且推論速度與 FP16 圖版本相同（實測 19.5 vs 19.6 ms/img）。
    """
    import json as _json
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10：explicit batch 為預設
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(qdq_onnx)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT 解析 ONNX 失敗：{qdq_onnx}\n{errs}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8 if precision == "int8" else trt.BuilderFlag.FP8)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    print(f"[trt] building {precision} engine（TensorRT {trt.__version__}）…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build 失敗，請看上方 TensorRT log")

    meta = _json.dumps(metadata)
    with open(engine_path, "wb") as f:
        f.write(len(meta).to_bytes(4, byteorder="little", signed=True))
        f.write(meta.encode())
        f.write(serialized)
    print(f"[trt] engine → {engine_path}")


# ── FP8 前置檢查 ──────────────────────────────────────────────────────────────

def _check_trt_fp8() -> None:
    import tensorrt as trt
    major, minor, _ = (int(x) for x in trt.__version__.split(".")[:3])
    if (major, minor) < (10, 0):
        raise RuntimeError(
            f"FP8 需要 TensorRT >= 10.0，目前版本為 {trt.__version__}"
        )


# ── 主導出函式 ────────────────────────────────────────────────────────────────

def export(cfg: dict, precision: str | None = None) -> None:
    """導出指定精度（None = 依 config precisions list 全部導出）。"""
    from ultralytics import YOLO

    paths = cfg["paths"]
    exp = cfg["export"]
    weights = paths["weights"]
    outdir = paths["outputs"] / "engines"
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

        elif prec in ("int8", "fp8"):
            if prec == "fp8":
                _check_trt_fp8()
            qcfg = cfg[prec]
            fp32_onnx = _export_fp32_onnx(weights, exp["imgsz"], exp["onnx_opset"], outdir)
            qdq = _quantize_onnx(
                fp32_onnx, prec, paths["outputs"] / "calib_frames.csv", paths["coda_root"],
                cfg["calibration"]["cam"], exp["imgsz"],
                qcfg["calib_algo"], qcfg["skip_layers"], qcfg["nodes_to_exclude"],
                cfg["calibration"]["eps"], outdir,
            )
            _build_engine(qdq, prec, outdir / f"{prec}.engine", _ultralytics_metadata(fp32_onnx))

        else:
            print(f"[export] 未知精度 {prec!r}，跳過")


# ── Sensitivity Analysis ──────────────────────────────────────────────────────

def sensitivity(cfg: dict) -> None:
    """逐層 disable 量化，計算 output MSE vs FP32 baseline，輸出 ranking JSON。"""
    import copy
    import torch
    import torch.nn.functional as F
    import modelopt.torch.quantization as mtq
    from ultralytics import YOLO

    paths = cfg["paths"]
    exp = cfg["export"]
    sens_cfg = cfg["sensitivity"]
    precision = sens_cfg["precision"]
    n_subset = sens_cfg["n_subset"]
    calib_csv = paths["outputs"] / "calib_frames.csv"
    outdir = paths["outputs"] / "sensitivity"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[sensitivity] precision={precision}, n_subset={n_subset}")

    # FP32 baseline model（不量化）
    fp32_model = YOLO(str(paths["weights"])).model.eval().cuda()

    # 量化 model
    q_model = YOLO(str(paths["weights"])).model.eval().cuda()
    if precision == "int8":
        config = _build_int8_config([])
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
    print("請將需要跳過的層名填入 configs/quantization.yaml 的 skip_layers，再重跑 export。")


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
