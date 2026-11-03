# 視覺化工具說明

所有指令在 `src/` 目錄下執行。

---

## 工具總覽

| 工具 | 用途 | 輸出 |
|---|---|---|
| `vis_coda_lidar2img.py` | 點雲投影到影像，檢查 LiDAR↔相機標定 | `.jpg` |
| `vis_coda_bbox3d.py` | GT 3D bbox 投影到影像 | `.jpg` |
| `vis_coda_pointcloud.py` | 3D 點雲互動檢視（Open3D） | 互動視窗 / `.jpg` |
| `vis_frame_points.py` | 前處理後的 FramePoints 3D 檢視（Open3D） | 互動視窗 / `.jpg` |
| `vis_cluster_selection.py` | 每個 YOLO 框的選群結果（2D 疊圖） | `.jpg` + `clusters.csv` |
| `vis_cluster_3d.py` | 每個 YOLO 框的選群結果（3D，Open3D） | 互動視窗 / `.jpg` |
| `vis_eval_errors.py` | TP / FP / FN 錯誤分析圖 | `.jpg` + `index.csv` |

---

## 建議使用順序

```
1. vis_coda_lidar2img      ← 確認標定沒問題
2. vis_coda_bbox3d         ← 確認 GT 標註
3. vis_frame_points        ← 確認前處理（去地面、裁切）
4. vis_cluster_selection   ← 確認選群結果（2D）
5. vis_cluster_3d          ← 深入看某個 frame（3D）
6. vis_eval_errors         ← 跑完 run_eval.py 後看 FP/FN
```

---

## 1. vis_coda_lidar2img.py

把 os1 點雲投影到 RGB 影像，點依相機深度上色（近藍遠紅），用來確認 LiDAR↔相機標定是否準確。
加 `--yolo` 可同時跑 YOLO 並把偵測框疊在圖上，輸出 `detections.csv` 供後續工具使用。

```bash
# seq 0 前 10 幀
python tools/vis_coda_lidar2img.py --seq 0

# 指定 frame，背景換成 Canny 邊緣圖（方便看對齊）
python tools/vis_coda_lidar2img.py --seq 0 --frame 683 --show --edges

# 每隔 50 幀取一幀，疊上 GT bbox
python tools/vis_coda_lidar2img.py --seq 0 --stride 50 --num 20 --bbox

# 從 CSV 清單畫（需有 seq, frame 欄位）
python tools/vis_coda_lidar2img.py --list ../outputs/distance/frames.csv

# 全部 frame 跑 YOLO，輸出 detections.csv
python tools/vis_coda_lidar2img.py --all-seqs --all --yolo
```

**主要參數：**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--seq` | — | 序列編號，可給多個（`--seq 0 1 2`） |
| `--all-seqs` | — | 處理全部序列 |
| `--list` | — | CSV 清單（需有 seq, frame 欄位），覆蓋 `--seq` |
| `--cam` | `cam0` | `cam0` 或 `cam1` |
| `--frame` | — | 指定 frame（可給多個） |
| `--num` | `10` | 不指定 `--frame` 時最多畫幾幀 |
| `--stride` | `1` | 每隔幾幀取一幀 |
| `--all` | — | 畫出所有有點雲的 frame |
| `--edges` | — | 背景換成 Canny 邊緣圖 |
| `--bbox` | — | 疊上 GT 3D bbox |
| `--show` | — | 用視窗顯示（任意鍵下一張，q 離開） |
| `--min-depth` | `0.5` | 相機深度小於此值的點不畫（m） |
| `--max-depth` | `80.0` | 相機深度大於此值的點不畫（m） |
| `--color-max` | `30.0` | 深度上色上限（m） |
| `--radius` | `1` | 點的半徑（像素） |
| `--dim` | `0.6` | 背景亮度倍率（0~1） |
| `--outdir` | `outputs/vis_lidar2img` | 加 `--yolo` 時預設 `outputs/vis_lidar2img_yolo` |
| `--yolo` | — | 跑 YOLO 並疊框，另輸出 `detections.csv` |
| `--weights` | `models/pretrained/yolo11s.pt` | YOLO 權重 |
| `--imgsz` | `1280` | YOLO 輸入解析度 |
| `--conf` | `0.25` | 信心門檻 |
| `--half` | — | FP16 推論 |

**輸出：** `outputs/vis_lidar2img/{cam}/{seq}/{frame}.jpg`
加 `--yolo`：另輸出 `detections.csv`、`timing.json`（YOLO 推論統計）

---

## 2. vis_coda_bbox3d.py

把 CODa GT 3D bbox 的 8 個角點與 12 條邊投影到影像，用來確認標注位置是否合理。
只畫 person / bicycle / car 三類（CODa 原始類別自動歸類）。

```bash
# seq 0 前 10 個有標注的 frame
python tools/vis_coda_bbox3d.py --seq 0

# 指定 frame，用視窗顯示
python tools/vis_coda_bbox3d.py --seq 0 --frame 683 --show

# 所有有標注的 frame
python tools/vis_coda_bbox3d.py --seq 0 --all

# 只看 person，不畫完全遮擋的物件
python tools/vis_coda_bbox3d.py --seq 0 --classes person --hide-full-occluded
```

**主要參數：**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--seq` | （必填） | 序列編號 |
| `--cam` | `cam0` | `cam0` 或 `cam1` |
| `--frame` | — | 指定 frame（可給多個） |
| `--num` | `10` | 前幾個有標注的 frame |
| `--all` | — | 所有有標注的 frame |
| `--classes` | 全部三類 | 只畫指定類別（`person bicycle car`） |
| `--hide-full-occluded` | — | 不畫 `isOccluded=Full` 的物件 |
| `--show` | — | 用視窗顯示 |
| `--outdir` | `outputs/vis_bbox3d` | — |

**輸出：** `outputs/vis_bbox3d/{cam}/{seq}/{frame}.jpg`

---

## 3. vis_coda_pointcloud.py

用 Open3D 互動檢視一個 sequence 的原始點雲，點依距離上色（近藍遠紅），可疊上 GT 3D bbox。

```bash
# seq 0，互動模式
python tools/vis_coda_pointcloud.py --seq 0

# 從 frame 956 開始，只看 person / bicycle
python tools/vis_coda_pointcloud.py --seq 20 --frame 956 --classes person bicycle

# 不開視窗，每幀截圖存檔
python tools/vis_coda_pointcloud.py --seq 0 --save --num 20
```

**主要參數：**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--seq` | （必填） | 序列編號 |
| `--frame` | — | 從這個 frame 開始 |
| `--annotated-only` | — | 只瀏覽有標注的 frame |
| `--no-bbox` | — | 一開始不畫 3D bbox（可按 B 切換） |
| `--classes` | 全部三類 | 只畫這些類別的 bbox |
| `--max-range` | `50.0` | 距離上色上限（m） |
| `--min-range` | `0.5` | 小於此距離的點不畫（m） |
| `--point-size` | `1.5` | 點的大小 |
| `--zoom` | `0.03` | 初始視角縮放，越小越近 |
| `--save` | — | 不互動，每幀存截圖 |
| `--num` | — | `--save` 時最多存幾幀 |
| `--outdir` | `outputs/vis_pointcloud` | — |

**互動按鍵：** N/→ 下一幀、P/← 上一幀、B 顯示/隱藏 bbox、Q 離開

---

## 4. vis_frame_points.py

用 Open3D 檢視點雲前處理（`prepare_points`）後的 FramePoints，顯示哪些點被去掉、去掉的原因。
可疊上地面平面、相機視野、GT bbox，用來確認去近點、去地面、影像裁切的參數是否合理。

```bash
# seq 0
python tools/vis_frame_points.py --seq 0

# 調整地面門檻
python tools/vis_frame_points.py --seq 20 --frame 956 --ground-thr 0.2

# 不開視窗，存截圖
python tools/vis_frame_points.py --seq 0 --save --num 20
```

**點的顏色：**
- 保留的點：依相機距離（或離地高度，按 C 切換）近藍遠紅
- 地面點：棕色
- 影像外 / 相機後方的點：暗灰

**主要參數：**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--seq` | （必填） | 序列編號 |
| `--cam` | `cam0` | — |
| `--frame` | — | 從這個 frame 開始 |
| `--annotated-only` | — | 只瀏覽有標注的 frame |
| `--min-range` | `0.5` | 去近點門檻（m） |
| `--min-depth` | `0.5` | 去相機後方點的門檻（m） |
| `--ground-thr` | `0.15` | 離地高度小於此值視為地面（m） |
| `--hide-removed` | — | 一開始不畫被丟掉的點（可按 R 切換） |
| `--no-bbox` | — | 一開始不畫 3D bbox（可按 B 切換） |
| `--classes` | 全部三類 | 只畫這些類別的 bbox |
| `--max-range` | `50.0` | 距離上色上限（m） |
| `--frustum-depth` | `30.0` | 相機視野畫到多深（m） |
| `--point-size` | `2.0` | — |
| `--zoom` | `0.03` | 初始視角縮放 |
| `--save` | — | 不互動，存截圖 |
| `--num` | — | `--save` 時最多幾幀 |
| `--outdir` | `outputs/vis_frame_points` | — |

**互動按鍵：** N/→ P/← 換幀、R 顯示/隱藏被丟掉的點、C 切換上色模式、G 地面平面、F 相機視野、B bbox、Q 離開

---

## 5. vis_cluster_selection.py

讀現有的 `detections.csv`，對每個 YOLO 框執行完整的 `estimate_frame`，把選群過程疊在影像上。
用來確認哪些框選到正確物件、哪些失敗，以及調整 DistanceConfig 參數。

```bash
# 隨機抽 30 幀（預設）
python tools/vis_cluster_selection.py

# 所有有偵測結果的 frame
python tools/vis_cluster_selection.py --all

# 指定 seq 和 frame，同時用視窗顯示
python tools/vis_cluster_selection.py --seq 0 --frame 481 --show

# 調整選群參數
python tools/vis_cluster_selection.py --num 50 --classes person car --w-cov 0.3 --w-center 0.7

# 另存每個偵測框的放大裁切圖
python tools/vis_cluster_selection.py --num 30 --crops
```

**主要參數：**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--dets` | `outputs/vis_lidar2img_yolo/detections.csv` | detections.csv 路徑 |
| `--seq` | — | 只看這個序列 |
| `--frame` | — | 指定 frame（需搭配 `--seq`） |
| `--num` | `30` | 隨機抽幾幀 |
| `--all` | — | 所有有偵測結果的 frame |
| `--seed` | `0` | 隨機種子 |
| `--classes` | 全部三類 | 只處理這些類別 |
| `--conf` | `0.25` | 信心門檻 |
| `--crops` | — | 另存各框的放大裁切圖 |
| `--show` | — | 用視窗顯示 |
| `--dim` | `0.5` | 背景亮度倍率 |
| `--radius` | `2` | 點的半徑（像素） |
| `--outdir` | `outputs/vis_cluster` | — |
| DistanceConfig 參數 | 各自預設 | `--shrink`, `--eps-min`, `--eps-k`, `--min-points`, `--min-ratio`, `--w-center`, `--w-cov`, `--min-cluster` |

**輸出：** `outputs/vis_cluster/{seq}/{frame}.jpg`、`clusters.csv`
加 `--crops`：另輸出 `outputs/vis_cluster/crops/{seq}_{frame}_{det}_{class}.jpg`

---

## 6. vis_cluster_3d.py

讀現有的 `detections.csv`，用 Open3D 互動檢視選群結果：選中的點紅色、其餘藍色，GT bbox 綠色。
終端機會列出每個框的類別、conf、status、點數、選中點落在 GT box 內的比例（`in_gt`）、距離。

```bash
# seq 0，互動模式
python tools/vis_cluster_3d.py --seq 0

# 從指定 frame 開始，只看有標注的
python tools/vis_cluster_3d.py --seq 0 --frame 481 --annotated-only

# 調整選群參數
python tools/vis_cluster_3d.py --seq 0 --w-cov 0.3 --w-center 0.7

# 不開視窗，每幀截圖存檔
python tools/vis_cluster_3d.py --seq 0 --save --num 20
```

**主要參數：**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--seq` | （必填） | 序列編號 |
| `--dets` | `outputs/vis_lidar2img_yolo/detections.csv` | detections.csv 路徑 |
| `--cam` | `cam0` | — |
| `--frame` | — | 從這個 frame 開始 |
| `--annotated-only` | — | 只瀏覽有 GT 標注的 frame |
| `--classes` | 全部三類 | 只處理這些類別 |
| `--conf` | `0.25` | 信心門檻 |
| `--no-bbox` | — | 一開始不畫 GT bbox（可按 B 切換） |
| `--no-label` | — | 一開始不畫 YOLO 標籤（可按 L 切換） |
| `--label-size` | `0.35` | 標籤字高（m） |
| `--point-size` | `2.0` | — |
| `--zoom` | `0.03` | 初始視角縮放 |
| `--save` | — | 不互動，每幀存截圖 |
| `--num` | — | `--save` 時最多幾幀 |
| `--outdir` | `outputs/vis_cluster_3d` | — |
| DistanceConfig 參數 | 各自預設 | 同 vis_cluster_selection |

**互動按鍵：** N/→ P/← 換幀、B 顯示/隱藏 bbox、L 顯示/隱藏標籤、Q 離開

---

## 7. vis_eval_errors.py

讀 `run_eval.py` 的輸出，把 TP（綠）、FP（紅）、FN（藍）畫在影像上，用來分析誤判原因。
自動挑選 FP 最多、FN 最多、三種都有的 frame 各約 1/3，每個 sequence 限量以分散場景。

**前置條件：** 需先跑完 `run_pipeline.py` 和 `run_eval.py`。

```bash
# 預設：從 outputs/eval 讀結果，抽 100 幀
python tools/vis_eval_errors.py

# 指定不同的 pipeline 輸出與 eval 輸出
python tools/vis_eval_errors.py --dist-dir ../outputs/distance_fp16 --eval-dir ../outputs/eval_fp16

# 調整 IoU 門檻、near 標記門檻
python tools/vis_eval_errors.py --n 100 --iou 0.5 --near 0.3 --outdir ../outputs/vis_eval
```

**圖例：**
- **綠框（粗）**：TP 的 YOLO 框；**綠框（細）**：配對到的 GT 投影框；標 IoU
- **紅框**：FP；標類別、score、同類 GT 的最大 IoU；若框到非目標類別的物件，標出該類別
- **藍框**：FN（GT 投影框）；標類別、最大 IoU、遮擋程度、距離；8 個角點投影畫成小點
- **`near` 標記**：best IoU 在 `[--near, 門檻)` 之間，表示框到了但不夠齊，降低門檻就會變 TP

**主要參數：**

| 參數 | 預設 | 說明 |
|---|---|---|
| `--dist-dir` | `outputs/distance` | `run_pipeline.py` 的輸出資料夾 |
| `--eval-dir` | `outputs/eval` | `run_eval.py` 的輸出資料夾 |
| `--root` | `paths.yaml` 的 `coda_root` | CODa 根目錄 |
| `--cam` | `cam0` | — |
| `--min-depth` | `0.5` | 投影 GT bbox 時的相機深度門檻（m） |
| `--iou` | eval summary 的門檻 | 配對 IoU 門檻，不給時從 `summary.json` 讀 |
| `--near` | `0.3` | best IoU >= 此值時標 near |
| `--n` | `100` | 最多挑幾幀 |
| `--max-per-seq` | `4` | 每類每個 sequence 最多幾張 |
| `--seed` | `0` | 隨機種子 |
| `--outdir` | `outputs/vis_eval` | — |

**輸出：** `outputs/vis_eval/{index}_{tag}_s{seq}_f{frame}.jpg`、`index.csv`

---

## 常用 `--dets` 路徑對照

`vis_cluster_selection.py` 和 `vis_cluster_3d.py` 預設讀 `vis_lidar2img_yolo/detections.csv`（用 `vis_coda_lidar2img.py --yolo` 產出）。
若要用 `run_pipeline.py` 的偵測結果改用：

```bash
python tools/vis_cluster_selection.py --dets ../outputs/distance/detections.csv
python tools/vis_cluster_3d.py --seq 0 --dets ../outputs/distance/detections.csv
```
