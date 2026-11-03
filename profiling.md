# Distance Pipeline 效能量測指南

## 四個量測階段

| Stage | 內容 | 程式碼位置 | 主要資源 |
|---|---|---|---|
| 1. Data I/O | `read_image` + `read_lidar` | `run_pipeline.py:88-90` | 磁碟 |
| 2. YOLO | `yolo()` 推論 | `run_pipeline.py:91-92` | GPU |
| 3. prepare_points | 去近點、去地面、投影、裁切 | `pipeline.py:50-51` | CPU |
| 4. clustering | `select_object_points` × N bboxes | `pipeline.py:52-54` | CPU |

---

## 四個量測指標

| 指標 | 工具 | 量法 | 層級 |
|---|---|---|---|
| Wall time | `time.perf_counter` | 每段前後打點，取差值（ms） | process |
| CPU time | `psutil.Process.cpu_times` | 每段前後取 `user + sys` 差值（ms） | process |
| Peak GPU memory | `torch.cuda.max_memory_allocated` | 每段前呼叫 `reset_peak_memory_stats`，結束後讀值（MB） | process |
| RAM | `psutil.Process.memory_info().rss` | delta（前後相減，MB） | process |

### 為何 GPU memory 用 peak

PyTorch 提供 `reset_peak_memory_stats()` 可在每段開始前清掉峰值紀錄（不釋放記憶體，只重設水位線），所以 `max_memory_allocated()` 量到的就是該段自己的最高用量，不需要手動算 delta。

### 為何 RAM 用 delta

`psutil.rss` 沒有 reset 機制，它是當下整個 process 的記憶體快照，無法清掉基準值，只能靠前後相減隔離單一 stage 的用量。

### 為何不另外量 GPU time

目前 pipeline 為 single thread，CPU 一定要等 `yolo()` 回傳結果才能繼續，因此 wall time 已包含 GPU 執行時間，不需要額外用 `torch.cuda.Event` 打點。

---

## profiling.json 格式

每次執行 `run_pipeline.py` 會在 `--outdir`（預設 `outputs/distance`）產生 `profiling.json`。
檔案只包含 summary（由 `profiler.py:compute_summary` 產生，`per_frame` 原始資料不寫入檔案）：

```json
{
  "io":      {"wall_ms": {"mean": 12.3, "p95": 18.1}, "cpu_ms": {"mean": 2.1, "p95": 3.0}, "gpu_peak_mb": {"mean": 0.0, "p95": 0.0}, "ram_delta_mb": {"mean": 45.2, "p95": 60.0}},
  "yolo":    {"wall_ms": {"mean": 38.1, "p95": 45.2}, "cpu_ms": {"mean": 4.5, "p95": 5.1}, "gpu_peak_mb": {"mean": 512.3, "p95": 520.1}, "ram_delta_mb": {"mean": 1.2, "p95": 2.0}},
  "prepare": {"wall_ms": {"mean": 5.2,  "p95": 7.3},  "cpu_ms": {"mean": 5.1, "p95": 6.8}, "gpu_peak_mb": {"mean": 0.0, "p95": 0.0}, "ram_delta_mb": {"mean": 8.4, "p95": 10.1}},
  "cluster": {"wall_ms": {"mean": 3.8,  "p95": 6.1},  "cpu_ms": {"mean": 3.7, "p95": 5.9}, "gpu_peak_mb": {"mean": 0.0, "p95": 0.0}, "ram_delta_mb": {"mean": 2.1, "p95": 3.5}}
}
```

### 欄位說明

| 欄位 | 說明 |
|---|---|
| `wall_ms` | 實際經過時間（含 I/O wait、scheduling） |
| `cpu_ms` | CPU 實際計算時間（user + sys） |
| `gpu_peak_mb` | 該段的 GPU 記憶體峰值（process 層級） |
| `ram_delta_mb` | 該段造成的 RAM 增量（process 層級） |

---

## TensorRT 量化對比

換成 TensorRT engine 後，使用同一套量測方法，但需注意：

1. **engine 初始化不納入量測**：TRT engine 建立時有大量 GPU 記憶體分配，應在初始化完成後再開始量測
2. **warmup 必須先跑**：第一次推論會做 kernel 選擇，速度偏慢，建議 warmup 至少 3 次再開始記錄

```python
# engine 初始化（不量這段）
engine = build_trt_engine(...)

# warmup（不量這段）
for _ in range(3):
    engine(warmup_img)

# 正式量測（與原本 YOLO 量法相同）
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
preds = engine(img)
wall_ms = (time.perf_counter() - t0) * 1000
gpu_peak_mb = torch.cuda.max_memory_allocated() / 1e6
```

對比時直接比較兩份 `profiling.json` 的 `summary.yolo` 欄位即可。

---

## 相依套件

```
psutil
torch  (已存在)
```

安裝：
```bash
pip install psutil
```
