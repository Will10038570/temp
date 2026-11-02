"""距離估計的量化驗證：YOLO 偵測框與 CODa 3D bbox GT 配對，再比較兩邊的點雲距離。

  1. GT 3D bbox 的 8 個角點投影到影像取外接矩形（投影不到影像內的 GT 不列入）
  2. 依類別 greedy 配對：pred 依 score 由高到低，取同類別 IoU 最大、尚未配對且 >= 門檻的 GT；類別不符的算 FP
  3. 配對成功的，GT 距離 = FramePoints 中落在 GT 3D box 內的點，在相機座標的歐氏距離中位數
     （和 select_object_points 的 dist 定義相同，點也來自同一份去近點、去地面、只留影像內的 FramePoints）

流程說明文件：docs/distance_pipeline.md 第 6 節
"""
import numpy as np

from q1.data.projection import bbox_corners, in_box3d, project_bbox
from q1.distance.cluster import box_iou


def project_gt(boxes, calib, min_depth):
    """GT 3D bbox list（已經過 classes.gt_boxes 歸類）→ 影像內的 GT。

    回傳 list of {"box", "corners", "bbox2d"}。有角點在相機後方、或投影完全在影像外的 GT 不列入（見 project_bbox）。
    """
    out = []
    for b in boxes:
        proj = project_bbox(b, calib, min_depth)
        if proj is not None:
            out.append({"box": b, "corners": bbox_corners(b), "bbox2d": proj[1]})
    return out


def greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5, pred_classes=None, gt_classes=None):
    """依類別的 greedy 配對：pred 依 score 由高到低，取同類別 IoU 最大、尚未配對且 >= 門檻的 GT。

    pred_classes / gt_classes：長度與 pred_boxes / gt_boxes 相同的類別名稱 list；
    提供時跨類別的 IoU 設為 -1，類別不同的配對不成立（pred 算 FP，GT 繼續等待）。

    回傳 (matches, fp, fn, iou)：
      matches  [(pred index, gt index, IoU), ...]
      fp / fn  沒配對到的 pred / gt index
      iou      len(pred) x len(gt) 的原始 IoU 矩陣（未遮掩，供 unmatched best_iou 使用）
    """
    iou = np.array([[box_iou(p, g) for g in gt_boxes] for p in pred_boxes],
                   dtype=float).reshape(len(pred_boxes), len(gt_boxes))
    # 類別遮掩：跨類別設為 -1，讓配對邏輯自然跳過
    eff = iou.copy()
    if pred_classes is not None and gt_classes is not None:
        for pi, pc in enumerate(pred_classes):
            for gi, gc in enumerate(gt_classes):
                if pc != gc:
                    eff[pi, gi] = -1.0
    used = np.zeros(len(gt_boxes), bool)
    matches = []
    if len(gt_boxes):
        for pi in np.argsort(-np.asarray(pred_scores, dtype=float), kind="stable"):
            cand = np.where(used, -1.0, eff[pi])
            gi = int(np.argmax(cand))
            if cand[gi] >= iou_thr:
                used[gi] = True
                matches.append((int(pi), gi, float(cand[gi])))
    matched = {m[0] for m in matches}
    fp = [i for i in range(len(pred_boxes)) if i not in matched]
    fn = [i for i in range(len(gt_boxes)) if not used[i]]
    return matches, fp, fn, iou


def gt_distance(fp, corners):
    """FramePoints 中落在 GT 3D box 內的點 → (點數, 相機座標歐氏距離中位數 m；沒有點時為 NaN)。"""
    m = in_box3d(fp.xyz, corners)
    if not m.any():
        return 0, float("nan")
    return int(m.sum()), float(np.median(np.linalg.norm(fp.xyz_cam[m], axis=1)))


def error_stats(err):
    """有號誤差（估計 − GT）陣列 → 統計值。"""
    err = np.asarray(err, dtype=float)
    if not len(err):
        return {"n": 0}
    a = np.abs(err)
    return {"n": int(len(err)), "mean_signed_m": float(err.mean()), "median_signed_m": float(np.median(err)),
            "mae_m": float(a.mean()), "median_abs_m": float(np.median(a)), "p90_abs_m": float(np.percentile(a, 90)),
            "max_abs_m": float(a.max())}
