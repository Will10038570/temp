# TensorRT 量化測試

## 概覽

將 `yolo11s.pt` 導出為四種 TRT engine，接入現有推論 pipeline，量測各精度的速度與精度，產出對照報告。

```
yolo11s.pt
  ├─ FP32 ─ ultralytics TRT 直接導出 ──────────────────────▶ fp32.engine
  ├─ FP16 ─ ultralytics TRT 直接導出（half=True）──────────▶ fp16.engine
  ├─ INT8 ─ ModelOpt PTQ (Entropy) ─▶ int8_qdq.onnx ──────▶ int8.engine
  └─ FP8  ─ ModelOpt PTQ (E4M3)    ─▶ fp8_qdq.onnx  ──────▶ fp8.engine
```

**所有可調參數集中於 `src/configs/trt_quant.yaml`，不需修改任何 Python 程式碼。**

---

## 前置需求

| 套件 | 用途 |
|---|---|
| `ultralytics` | YOLO 推論與 FP32/FP16 engine 導出 |
| `nvidia-modelopt` | INT8 / FP8 PTQ，Q/DQ 節點插入 |
| `onnx`, `onnx-graphsurgeon` | ONNX graph 驗證與 Q/DQ 覆蓋率檢查 |
| TensorRT ≥ 10.0 + `trtexec` | Engine build（FP8 需 TRT ≥ 10.0） |
| CUDA 12.8+（建議） | RTX 5070 / Blackwell 支援 |

---

## 設定檔：`src/configs/trt_quant.yaml`

所有決策點說明：

```yaml
paths:
  weights:     "models/pretrained/yolo11s.pt"   # 原始模型
  kept_frames: "data/coda/raw/CODa_tiny/kept_frames.csv"
  coda_root:   "data/coda/raw/CODa_tiny"
  outdir:      "outputs"

calibration:
  n:    300      # INT8/FP8 PTQ 校正幀數（建議 200–500）
  seed: 42       # 隨機種子
  cam:  "cam0"

export:
  imgsz:      1280          # 需與 run_pipeline.py --imgsz 一致
  onnx_opset: 17
  device:     "0"           # GPU device id
  precisions: [fp32, fp16, int8, fp8]   # 移除不需要的精度即可

int8:
  calib_algo: "entropy"     # entropy | percentile | mse
  percentile: 99.9          # calib_algo == "percentile" 時生效
  skip_layers: []           # ← Sensitivity analysis 後填入，不改 code

fp8:
  skip_layers: []           # ← Sensitivity analysis 後填入，不改 code

sensitivity:
  n_subset:  50             # output MSE 計算用的子集幀數
  precision: "int8"         # int8 | fp8

pipeline:
  cam:        "cam0"
  conf:       0.25
  iou_nms:    0.70
  num_frames: null          # null = 全部；整數 = 限制幀數（除錯用）

evaluation:
  iou_thr:       0.5
  min_gt_points: 5

compare:
  labels: [fp32, fp16, int8, fp8]
  out:    "outputs/comparison_table.md"
```

---

## 執行流程

所有指令在 `src/` 下執行。

### 1. 抽取校正幀

```bash
python tools/calib_sample.py
```

從 `kept_frames.csv` 跨 seq 分層抽取校正集，輸出 `outputs/calib_frames.csv`。  
抽樣數量與 seed 由 `calibration.n` / `calibration.seed` 控制。

---

### 2. 導出 TRT engine

**一次導出所有精度（依 `export.precisions`）：**
```bash
python tools/export_trt.py export
```

**只導出指定精度：**
```bash
python tools/export_trt.py export --precision fp16
python tools/export_trt.py export --precision int8
```

INT8 / FP8 會先輸出 `int8_qdq.onnx` / `fp8_qdq.onnx`，再由 `trtexec` build engine。  
所有 engine 輸出到 `outputs/engines/`。

> **注意：** INT8 / FP8 導出前須先完成步驟 1。  
> **注意：** Engine 與 GPU 型號綁定，需在 RTX 5070 上 build。

---

### 3. Sensitivity Analysis（選做，INT8 / FP8 精度調優用）

```bash
python tools/export_trt.py sensitivity
```

分析精度由 `sensitivity.precision` 設定（`int8` 或 `fp8`）。  
逐層 disable 量化，計算 output MSE vs FP32 baseline（共 `sensitivity.n_subset` 幀）。

輸出：`outputs/sensitivity/{int8|fp8}_sensitivity_ranking.json`

```json
[
  { "layer": "model.2.cv1.conv", "mse": 0.00412 },
  { "layer": "model.0.conv",     "mse": 0.00387 },
  ...
]
```

**閱讀 ranking，決定要跳過的層後，編輯 `trt_quant.yaml`：**
```yaml
int8:
  skip_layers: ["model.0.conv", "model.22.cv3"]
```

重新執行步驟 2 的 int8 / fp8 導出，skip_layers 即生效。

---

### 4. Q/DQ 覆蓋率檢查（選做）

```bash
python tools/quant_inspect.py int8
python tools/quant_inspect.py fp8
```

輸出：
- `outputs/quant_coverage/quant_coverage_{int8|fp8}_onnx.json`：每個 Conv/Gemm 的 Q/DQ 狀態與 scale
- `outputs/quant_coverage/quant_coverage_{int8|fp8}_trt.json`：TRT engine 每層實際精度

Terminal 會印出覆蓋率與 TRT fallback 層數。

---

### 5. 跑推論 Pipeline（每個精度各一次）

所有參數（weights 路徑、outdir、cam、conf、imgsz、DistanceConfig）皆從 `trt_quant.yaml` 讀取，不需帶任何 argparse 參數。

```bash
python run_pipeline.py --precision fp32
python run_pipeline.py --precision fp16
python run_pipeline.py --precision int8
python run_pipeline.py --precision fp8
```

輸出至 `outputs/distance_{precision}/`。

---

### 6. 跑 Evaluation（每個精度各一次）

同樣從 `trt_quant.yaml` 讀取所有參數（iou_thr、min_gt_points、cam 等）。

```bash
python run_eval.py --precision fp32
python run_eval.py --precision fp16
python run_eval.py --precision int8
python run_eval.py --precision fp8
```

輸出至 `outputs/eval_{precision}/`。

---

### 7. 產出比較報告

```bash
python tools/compare_quant.py
```

輸出 `outputs/comparison_table.md`。

---

## 輸出目錄結構

```
outputs/
├── calib_frames.csv               # 校正幀清單
├── engines/
│   ├── fp32.engine
│   ├── fp16.engine
│   ├── int8.engine
│   ├── fp8.engine
│   ├── int8_qdq.onnx              # INT8 Q/DQ ONNX（中間產物）
│   └── fp8_qdq.onnx               # FP8 Q/DQ ONNX（中間產物）
├── sensitivity/
│   ├── int8_sensitivity_ranking.json
│   └── fp8_sensitivity_ranking.json
├── quant_coverage/
│   ├── quant_coverage_int8_onnx.json
│   ├── quant_coverage_int8_trt.json
│   ├── quant_coverage_fp8_onnx.json
│   └── quant_coverage_fp8_trt.json
├── distance_{fp32|fp16|int8|fp8}/
│   ├── distances.csv
│   ├── detections.csv
│   ├── frames.csv
│   └── profiling.json
├── eval_{fp32|fp16|int8|fp8}/
│   ├── summary.json
│   ├── matches.csv
│   └── unmatched.csv
└── comparison_table.md            # 最終對照報告
```

---

## 對照報告欄位說明

`comparison_table.md` 的欄位：

| 欄位 | 來源 | 說明 |
|---|---|---|
| FPS | `profiling.json` yolo.wall_ms.mean | 1000 / mean wall_ms |
| Lat P95 (ms) | `profiling.json` yolo.wall_ms.p95 | YOLO 推論 P95 延遲 |
| Precision (det) | `summary.json` | TP / (TP + FP) |
| Recall | `summary.json` | TP / (TP + FN) |
| MAE (m) | `summary.json` distance_error.mae_m | 距離估計平均絕對誤差 |
| P90 Err (m) | `summary.json` distance_error.p90_abs_m | 距離誤差 90th percentile |
| Quant Coverage | `quant_coverage_*_onnx.json` | weight 量化層數 / 總層數 |
| TRT Fallback 層 | `quant_coverage_*_trt.json` | TRT 實際非目標精度的層數 |

---

## 量化策略說明

| 精度 | Weights | Activations | 工具 |
|---|---|---|---|
| FP32 | 不量化 | 不量化 | ultralytics |
| FP16 | 不量化（FP16 cast）| 不量化 | ultralytics |
| INT8 | per-channel symmetric, zero_point=0 | per-tensor, Entropy 校正 | ModelOpt + trtexec |
| FP8 | per-tensor E4M3, 無 zero_point | per-tensor E4M3 | ModelOpt + trtexec |

**Skip layers** 決策流程：
1. 跑 `export_trt.py sensitivity` 取得 MSE ranking
2. 人工決定門檻（MSE 較高的層為量化敏感層）
3. 填入 `trt_quant.yaml` 的 `skip_layers`，重新 export

---

## 已知 TODO

| 項目 | 位置 | 說明 |
|---|---|---|
| ModelOpt `algorithm` 字串值 | `export_trt.py` `_build_int8_config` | `"entropy"` 對應 ModelOpt 的確切值，以及 `"mse"` 是否被支援，需以 nvidia-modelopt 官方文件或 `print(mtq.INT8_DEFAULT_CFG)` 確認後移除 TODO 注釋 |
| `torch.onnx.export` 相容性 | `export_trt.py` `_export_qdq_int8/fp8` | ultralytics YOLO11 的 `model.model` 直接 trace 有時需要額外處理（`dynamic_axes`、`do_constant_folding`），需在 RTX 5070 上實際跑 INT8/FP8 export 時確認 |
