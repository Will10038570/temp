"""距離估計主流程：RGB + LiDAR + 2D 偵測框 → 每個偵測框的物件點與距離。

  影像 ──YOLO──▶ 2D 框 ─┐
                        ├─▶ select_object_points（每個框）─▶ 距離
  點雲 ──prepare_points─┘

流程說明文件：docs/distance_pipeline.md（修改流程、參數或輸出欄位時必須同步更新）
"""
from dataclasses import asdict, dataclass

from q1.data.classes import TARGET_CLASSES
from q1.distance.cluster import select_object_points
from q1.distance.points import prepare_points

# 要估距離的類別：person、bicycle、car（見 q1/data/classes.py）


@dataclass
class DistanceConfig:
    # 點雲前處理
    min_range: float = 0.5
    min_depth: float = 0.5
    ground_thr: float = 0.15
    # 選群
    shrink: float = 0.8
    eps_min: float = 0.3
    eps_k: float = 2.0
    min_points: int = 3
    min_ratio: float = 0.3
    w_center: float = 0.5
    w_cov: float = 0.5
    min_final: int = 5

    def point_kwargs(self):
        return {k: getattr(self, k) for k in ("min_range", "min_depth", "ground_thr")}

    def select_kwargs(self):
        skip = {"min_range", "min_depth", "ground_thr"}
        return {k: v for k, v in asdict(self).items() if k not in skip}


def estimate_frame(points, calib, preds, cfg=DistanceConfig(), classes=TARGET_CLASSES):
    """一幀的完整流程。

    points：N x 4 原始點雲；preds：YOLO 預測 list（bbox / category / score）
    回傳 (FramePoints, [(pred, result), ...])，result 見 select_object_points。
    不在 classes 內的預測直接略過。
    """
    fp = prepare_points(points, calib, **cfg.point_kwargs())
    results = [(p, select_object_points(fp.uv, fp.xyz, fp.xyz_cam, p["bbox"], **cfg.select_kwargs()))
               for p in preds if classes is None or p["category"] in classes]
    return fp, results


RESULT_FIELDS = ["seq", "frame", "det", "class", "score", "x1", "y1", "x2", "y2", "status",
                 "n_box", "n_core", "n_clusters", "n_candidates", "n_final", "eps",
                 "dist_m", "depth_m", "iou", "cov_w", "cov_h"]


def result_row(seq_id, frame, j, pred, res):
    """一個偵測框的結果 → 一列（對應 RESULT_FIELDS）。"""
    return [seq_id, frame, j, pred["category"], round(pred["score"], 3), *(round(v, 1) for v in pred["bbox"]),
            res["status"], int(res["in_box"].sum()), int(res["core"].sum()), len(res["clusters"]),
            sum(c["candidate"] for c in res["clusters"]), int(res["final"].sum()), round(res["eps"], 3),
            round(res["dist"], 2), round(res["depth"], 2), round(res["iou"], 3),
            round(res["cov_w"], 3), round(res["cov_h"], 3)]
