# YOLO 量化測試方法

## 測試矩陣

三個版本各自跑完整 pipeline，互相比較：

| 版本 | 指令 |
|---|---|
| FP32 | `run_pipeline.py --outdir outputs/distance_fp32` |
| FP16 | `run_pipeline.py --half --outdir outputs/distance_fp16` |
| TRT INT8 | `run_pipeline.py --backend trt_int8 --outdir outputs/distance_int8` |

所有版本使用同一份 `kept_frames.csv`。

---

## Calibration Set（INT8 專屬）

- 從**訓練資料**另外挑選約 500 張，不能用測試集
- 覆蓋各類別比例（car / pedestrian / cyclist）接近真實分布
- 覆蓋不同距離與場景
- 確定後鎖定，不再調整

---

## Warmup

前 30 幀不計入統計，確保 TRT INT8 kernel autotuning 穩定。

---

## 量測指標

每個版本執行 `run_eval.py` 後取得：

| 指標 | 來源 |
|---|---|
| precision / recall | `eval/summary.json` |
| distance MAE (m) | `eval/summary.json` |
| distance P90 (m) | `eval/summary.json` |
| YOLO wall_ms mean / p95 | `distance/profiling.json` |
| YOLO gpu_peak_mb | `distance/profiling.json`（`max_memory_allocated`） |

---

## 執行順序

```
# 各版本跑 pipeline
run_pipeline.py --outdir outputs/distance_fp32
run_pipeline.py --half --outdir outputs/distance_fp16
run_pipeline.py --backend trt_int8 --outdir outputs/distance_int8

# 各版本跑 eval
run_eval.py --dist-dir outputs/distance_fp32  --outdir outputs/eval_fp32
run_eval.py --dist-dir outputs/distance_fp16  --outdir outputs/eval_fp16
run_eval.py --dist-dir outputs/distance_int8  --outdir outputs/eval_int8
```

---

## 輸出目錄

```
outputs/
  distance_fp32/      distances.csv, detections.csv, timing.json, profiling.json
  distance_fp16/
  distance_int8/
  eval_fp32/          summary.json, matches.csv, unmatched.csv
  eval_fp16/
  eval_int8/
  quant_report.json   ← 三版本比較匯總
```

---

## 比較表

| | FP32 | FP16 | INT8 |
|---|---|---|---|
| precision | — | | |
| recall | — | | |
| distance MAE (m) | — | | |
| distance P90 (m) | — | | |
| YOLO wall_ms p95 (ms) | — | | |
| YOLO gpu_peak_mb (MB) | — | | |

---

## 待實作

1. TRT INT8 detector（`--backend trt_int8` 接入 `yolo.py`）
2. `profiling.json` 整合進 `run_pipeline.py`（依 `profiling.md` 規劃）
