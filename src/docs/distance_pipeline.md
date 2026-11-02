# 距離估計流程

> **維護規則：修改流程時必須同步更新這份文件。**
> 範圍包括 `q1/pipeline.py`、`q1/distance/`、`q1/detection/`、`run_pipeline.py`，以及 `DistanceConfig` 的參數或預設值、選群規則、輸出欄位或狀態的定義。
> 文件和程式碼不一致時，以程式碼為準，但應該在同一個 commit 裡把文件修正。

RGB 影像經 YOLO 得到 2D 偵測框，從框內的 LiDAR 點中選出屬於物件的點，再用這些點估計物件的距離。

## 1. 程式結構

依賴方向是單向的：`run_pipeline.py`、`tools/`、`vis/` 可以 import `q1/`，`q1/` 不 import 它們。

| 位置 | 角色 |
|---|---|
| `q1/data/` | CODa 讀取（`coda.py`）、LiDAR → 影像投影（`projection.py`）、CODa ↔ COCO 類別對應（`classes.py`） |
| `q1/detection/yolo.py` | `YoloDetector`（推論 + 計時）、讀寫 `detections.csv` |
| `q1/distance/ground.py` | RANSAC 擬合地面平面 |
| `q1/distance/points.py` | 整幀點雲前處理 → `FramePoints` |
| `q1/distance/cluster.py` | 單一偵測框的分群、選群、往外長回、算距離 |
| `q1/pipeline.py` | `DistanceConfig`（所有參數）、`estimate_frame()`（一幀的完整流程）、輸出欄位 |
| `run_pipeline.py` | 正式流程的 CLI 入口 |
| `q1/evaluation.py` | GT 投影、greedy 配對、GT 距離、誤差統計 |
| `run_eval.py` | 量化驗證的 CLI 入口，接在 `run_pipeline.py` 之後 |
| `vis/draw.py` | 繪圖函式（不屬於流程） |
| `tools/` | 一次性腳本和視覺化（不屬於流程） |

## 2. 流程總覽

```
RGB 影像 ──YoloDetector──▶ 2D 框（只保留 TARGET_CLASSES）──┐
                                                          ├─▶ 每個框：select_object_points ─▶ 距離
LiDAR 點雲 ──prepare_points（每幀一次）──▶ FramePoints ──┘
             去近點 → 去地面 → 投影 → 只留影像內的點
```

進入點是 `q1.pipeline.estimate_frame(points, calib, preds, cfg, classes)`，回傳 `(FramePoints, [(pred, result), ...])`。

## 3. 各步驟

### 3.1 2D 偵測（`q1/detection/yolo.py`）

- 預設使用 ultralytics YOLO（`yolo11s.pt`），`imgsz=1280`，`conf=0.25`。
- 只處理 `TARGET_CLASSES` = person、bicycle、car（`q1/data/classes.py`）。YOLO 推論時只保留 COCO 的 person、bicycle、car 三類；bus、truck、motorcycle 等**不併入 car**，和其他 COCO 類別一樣不輸出。CODa GT 也只保留 Pedestrian→person、Bike→bicycle、Car→car 三類；Bus、Pickup/Delivery Truck、Service/Utility Vehicle、Motorcycle、Scooter 等一律忽略（YOLO 分不出這麼多種車輛，不合併）。
- 也可以用 `--dets` 讀現有的 `detections.csv`，不重跑 YOLO。

### 3.2 整幀點雲前處理（`q1/distance/points.py`、`ground.py`）

每幀只做一次，所有偵測框共用。

1. **去近點**：離 LiDAR 小於 `min_range`（0.5 m）的點丟掉。沒有回波的點會落在原點附近。
2. **去地面**：用 RANSAC 擬合平面，離地高度不超過 `ground_thr`（0.15 m）的點丟掉。
   - 候選點：os1 座標中 z 介於 −1.5 到 −0.3 m、水平距離小於 30 m 的點。os1 離地約 0.86 m。
   - 100 次迭代，inlier 門檻 0.1 m，平面法向量與垂直方向的夾角需小於 15°，最後用所有 inlier 做最小平方修正。
   - 找不到合適的平面時，退回水平面：z = 候選點高度的第 10 百分位數。
   - 必須在分群之前去地面，否則地面點會把物件和背景連成同一群。
3. **投影**：用 `calib_os1_to_cam` 的 `projection_matrix` 投影到 `2d_rect` 影像，只保留影像內、且相機深度大於 `min_depth`（0.5 m）的點。
4. 輸出 `FramePoints`，包含 `uv`（像素座標）、`xyz`（os1 座標，分群用）、`xyz_cam`（相機座標，算距離用）、`plane`。

### 3.3 單一框選出物件點（`q1/distance/cluster.py` 的 `select_object_points`）

1. **取點**：`in_box` 是完整 bbox 內的點；`core` 是縮小到 `shrink`（80%）後 core 框內的點。
   縮小只用來**決定選哪一群**，不是用來丟點；第 5 步會把完整 bbox 內屬於物件的點長回來。
2. **Euclidean clustering**（只對 core 點做）：距離小於 eps 的點相連，取連通分量；點數少於 `min_points`（3）的群視為雜訊。
   - eps 隨距離自適應：`eps = max(eps_min, eps_k × d × 0.35°)`，其中 d 是 core 點相機座標距離的中位數。預設 `eps_min = 0.3 m`、`eps_k = 2`。
   - 0.35° 是 OS1-128 的水平和垂直角解析度。遠處點變稀疏，固定 eps 會把同一個物件切成好幾群。
3. **候選群**：點數至少是最大群的 `min_ratio`（0.3）倍。
4. **打分選群**：`score = w_center × center + w_cov × cov`，取分數最高的候選群。預設兩個權重都是 0.5。
   - `center`：群投影位置（中位數）離 core 框中心的正規化距離，換算成 0–1 的分數，中心為 1、角落為 0。
   - `cov`：群的 2D 外框在 core 框寬度和高度方向的覆蓋比例相乘。真正的物件通常會撐滿框，前景遮擋物和背景通常不會。
   - 選中群的點數少於 `min_cluster`（5）時，結果標為 `invalid`，不往下做。
5. **往外長回**：從選中的群出發，在完整 bbox 內以同一個 eps 做 region growing。只能長到 core 外的點，core 內屬於其他群的點視為阻擋。
6. **距離**：
   - `dist`：最終點在相機座標的歐氏距離中位數。這是主要輸出。
   - `depth`：相機 z 的中位數。
   - 注意：LiDAR 只打得到物件表面，所以這是**表面距離**，比物件中心略近。

### 3.4 結果狀態

| status | 意義 |
|---|---|
| `valid` | 成功：選中群的點數至少 `min_cluster`（5），已往外長回並算出距離 |
| `invalid` | 失敗，`dist_m`、`depth_m` 為空。可能原因：core 內的點少於 `min_points`（物件太遠、太小，或被地面濾除）；沒有任何群的點數達到 `min_points`；或選中群的點數少於 `min_cluster`。可用 `n_core`、`n_clusters`、`n_candidates` 欄位判斷是哪一種 |

## 4. 參數（`q1/pipeline.py` 的 `DistanceConfig`）

`run_pipeline.py` 和 `tools/vis_cluster_selection.py` 都會從 `DistanceConfig` 自動產生 CLI 參數（底線換成連字號，例如 `--w-cov`）。

| 參數 | 預設 | 說明 |
|---|---|---|
| `min_range` | 0.5 | 離 LiDAR 小於此距離（m）的點丟掉 |
| `min_depth` | 0.5 | 相機深度小於此值（m）的點丟掉 |
| `ground_thr` | 0.15 | 離地高度不超過此值（m）視為地面 |
| `shrink` | 0.8 | core 框的縮小比例 |
| `eps_min` | 0.3 | 分群 eps 的下限（m） |
| `eps_k` | 2.0 | eps 隨距離放大的倍率 |
| `min_points` | 3 | 一個群最少要有幾個點 |
| `min_ratio` | 0.3 | 候選群點數至少是最大群的幾倍 |
| `w_center` | 0.5 | 選群分數中「靠近中心」的權重 |
| `w_cov` | 0.5 | 選群分數中「2D 覆蓋率」的權重 |
| `min_cluster` | 5 | 選中群（往外長回之前）的點數低於此值時標為 `invalid` |

## 5. 執行（在 `src/` 下）

frame 來源：預設是 `{coda_root}/kept_frames.csv`，由 `tools/filter_coda_frames.py --apply` 產生，只留下 RGB（`2d_rect/cam0`）、LiDAR、3D bbox 三者齊全，且含目標類別的 frame。目前 CODa_tiny 有 908 個 frame，分布在 21 個 seq。LiDAR 優先讀 `3d_raw/os1`，沒有時改讀 `3d_comp/os1`；tiny split 只有 `3d_comp`，每幀 131072 點（128 線 × 1024）。

```bash
python run_pipeline.py                           # kept_frames.csv 的所有 frame，跑 YOLO
python run_pipeline.py --dets ../outputs/vis_lidar2img_yolo/detections.csv   # 用現有的偵測結果
python run_pipeline.py --seq 0 --num 20 --w-cov 0.3 --w-center 0.7
```

輸出在 `outputs/distance/`：

- `distances.csv`：每個偵測框一列，欄位見 `RESULT_FIELDS`。
  - `seq, frame, det, class, score, x1..y2`：偵測框資訊。
  - `status`：見 3.4。
  - `n_box, n_core, n_clusters, n_candidates, n_final`：各階段的點數或群數。
  - `eps`：實際使用的分群 eps。
  - `dist_m, depth_m`：距離。
  - `iou, cov_w, cov_h`：最終點的 2D 外框與 YOLO 框的 IoU，以及寬、高方向的覆蓋率。
- `detections.csv`：YOLO 原始結果（只在跑 YOLO 時產生）。
- `frames.csv`：這次處理的所有 frame（包含沒有偵測框的），`run_eval.py` 用它來算 FN。
- `timing.json`：YOLO 與距離估計的耗時，以及這次執行使用的資料根目錄、相機和 `DistanceConfig`。

## 6. 檢查結果

```bash
python tools/vis_cluster_selection.py --all             # 所有 seq 的所有 frame → outputs/vis_cluster/{seq}/{frame}.jpg
python tools/vis_cluster_selection.py --num 30 --crops   # 隨機抽 30 個 frame，另存裁切圖
```

這支工具直接呼叫 `estimate_frame`，所以圖上看到的就是正式流程的結果。輸出在 `outputs/vis_cluster/`，圖例見 `vis/draw.py` 的 `draw_cluster_selection`。

3D 檢視（可轉視角）：`python tools/vis_cluster_3d.py --seq 0 --annotated-only`。同樣呼叫 `estimate_frame`，畫前處理後的 FramePoints：選中的點畫紅色、其餘藍色，3D bbox GT 畫綠色（只畫影像範圍內的），每個偵測框上方標 YOLO 類別與 conf；終端機列出每個偵測框選中點落在 GT box 內的比例（`in_gt`）。

- `iou` 只能當**檢查訊號，不能當驗證指標**。點本來就是從 YOLO 框裡取出來的，而且選群時已經用過覆蓋率；如果選到填滿整個框的背景，IoU 也會很高。
- 30 m 以上的小物件只有 2–3 條掃描線，IoU 低屬於正常現象。
- 定量驗證見下方「量化驗證」。

### 量化驗證（`run_eval.py`、`q1/evaluation.py`）

接在 `run_pipeline.py` 之後執行，讀 `--dist-dir`（預設 `outputs/distance`）的輸出，和 CODa 3D bbox GT 比較。

1. **GT**：只用歸類到 `TARGET_CLASSES` 的 GT（`classes.gt_boxes`）。8 個角點投影到影像取外接矩形，並裁到影像範圍內；有角點在相機後方（深度小於 `min_depth`），或投影完全在影像外的 GT 不列入。
2. **配對**：依類別的 greedy 配對。YOLO 框依 score 由高到低排序，每個框取**同類別**、IoU 最大、尚未配對、且 IoU ≥ `--iou`（預設 0.5）的 GT；跨類別的 IoU 視為 −1，不會配對。沒配對到的 YOLO 框記為 FP（包含位置對但類別錯的）、GT 記為 FN，只記錄編號和數量。`unmatched.csv` 的 `best_iou` 用的是不分類別的原始 IoU。因為依類別配對，`class_match` 和 `class_accuracy_on_TP` 恆為 1。
3. **GT 距離**：用 `timing.json` 記錄的 `DistanceConfig` 重建 FramePoints，和 pipeline 是同一份點（去近點、去地面、只留影像內）。取落在 GT 3D box 內的點，算相機座標歐氏距離的中位數，定義和 `dist_m` 相同。
4. **誤差**：`err_m = dist_m − gt_dist_m`，有正負號；負值表示估得比較近。只有 pipeline 的 status 是 `valid`、且 GT box 內至少有 `--min-gt-points`（5）個點的配對才算誤差。其餘配對的 `eval` 欄位分別標為 `pred_invalid` 或 `gt_few_points`。

```bash
python run_eval.py                     # → outputs/eval/
python run_eval.py --iou 0.3 --outdir ../outputs/eval_iou03
```

輸出：`matches.csv`（每個配對一列）、`unmatched.csv`（FP／FN，含與對方的最大 IoU）、`summary.json`（TP／FP／FN、類別正確率、距離誤差統計，包含整體和各類別）。

注意：投影後的 3D box 外接矩形通常比 YOLO 框大，所以 IoU 0.5 會讓不少其實正確的配對變成 FP／FN。可以看 `unmatched.csv` 的 `best_iou` 分布來判斷。

## 7. 已知限制

- 距離是物件表面距離，不是物件中心，和 3D GT 中心比較時會有系統性的偏差。
- 只用單幀，沒有做時間上的平滑或追蹤。
- 沒有做 z-buffer 去遮擋。LiDAR 與相機之間的視差，可能讓被遮住的背景點投影進框內。
- 重疊的偵測框會各自獨立選群，同一批點可能同時被兩個框使用。
- 距離估計每幀平均約 220 ms（中位數 203 ms、P95 366 ms，908 幀實測），主要花在分群和 RANSAC，尚未優化。
