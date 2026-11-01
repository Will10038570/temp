"""從單一 2D 偵測框內的 LiDAR 點找出屬於物件的點，並估計距離。

輸入的點需已去地面、已投影（見 points.py）。每個偵測框：
  1. 取完整 bbox 內的點；再取縮小後（預設 80%）core 區域內的點
  2. 對 core 點做 Euclidean clustering（eps 依距離自適應）
  3. 過濾小群（< 最大群點數 * min_ratio），剩下的是候選群
  4. 候選群用「投影離 core 框中心的距離 + 2D 覆蓋率」打分，選分數最高者
  5. 從選中的群往完整 bbox 的外圈長回來（只長到 core 外的點，不吃 core 內的其他群）
  6. 距離 = 最終點在相機座標的歐氏距離中位數
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# OS1-128：水平 1024 點 / 360°、垂直 45° / 128 線，兩者都約 0.35°
OS1_ANGLE_RES = np.radians(0.35)


def adaptive_eps(depth, eps_min=0.3, k=2.0, angle_res=OS1_ANGLE_RES):
    """eps = max(eps_min, k * 該距離的點間距)。遠處點稀疏，固定 eps 會把同一物件切碎。"""
    return max(eps_min, k * depth * angle_res)


def euclidean_cluster(xyz, eps, min_points=3):
    """距離 < eps 的點相連，取連通分量。點數 < min_points 的群標成 -1。"""
    n = len(xyz)
    if n == 0:
        return np.zeros(0, dtype=int)
    pairs = cKDTree(xyz).query_pairs(eps, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    ids, counts = np.unique(labels, return_counts=True)
    small = ids[counts < min_points]
    labels[np.isin(labels, small)] = -1
    # 重新編號成 0..K-1，依點數由多到少
    keep = ids[counts >= min_points]
    keep = keep[np.argsort(-counts[counts >= min_points])]
    remap = {old: new for new, old in enumerate(keep)}
    return np.array([remap.get(l, -1) for l in labels])


def shrink_box(box, ratio):
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2 * ratio, (y2 - y1) / 2 * ratio
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh])


def in_box(uv, box):
    return (uv[:, 0] >= box[0]) & (uv[:, 0] <= box[2]) & (uv[:, 1] >= box[1]) & (uv[:, 1] <= box[3])


def extent(uv):
    return np.array([*uv.min(axis=0), *uv.max(axis=0)])


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def coverage(ext, box):
    """群的 2D 外框在 box 寬、高方向各覆蓋多少比例，回傳 (cov_w, cov_h)。"""
    w = max(0.0, min(ext[2], box[2]) - max(ext[0], box[0])) / max(box[2] - box[0], 1e-6)
    h = max(0.0, min(ext[3], box[3]) - max(ext[1], box[1])) / max(box[3] - box[1], 1e-6)
    return w, h


def score_clusters(uv, labels, box, min_ratio=0.3, w_center=0.5, w_cov=0.5):
    """對每個群算分數。回傳 list of dict（依點數排序），candidate=True 的才參與選擇。

    center：群投影的中位數位置離 box 中心的正規化距離，轉成 0~1 分（中心 1、角落 0）
    cov   ：群 2D 外框在 box 寬 × 高的覆蓋比例（0~1）
    """
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    half = np.array([(box[2] - box[0]) / 2, (box[3] - box[1]) / 2])
    ids = [i for i in np.unique(labels) if i >= 0]
    if not ids:
        return []
    n_max = max((labels == i).sum() for i in ids)
    out = []
    for i in ids:
        m = labels == i
        med = np.median(uv[m], axis=0)
        center_dist = np.linalg.norm((med - c) / np.maximum(half, 1e-6)) / np.sqrt(2)
        center = 1 - min(center_dist, 1.0)
        ext = extent(uv[m])
        cw, ch = coverage(ext, box)
        out.append({
            "id": int(i), "n": int(m.sum()), "center": float(center), "cov_w": cw, "cov_h": ch,
            "cov": cw * ch, "candidate": m.sum() >= min_ratio * n_max,
            "score": float(w_center * center + w_cov * cw * ch), "med_uv": med, "extent": ext,
        })
    return out


def grow(xyz, seed_mask, blocked_mask, eps):
    """從 seed 點往外長：只走 eps 內相連、且不在 blocked 的點。回傳最終 mask。"""
    allowed = seed_mask | ~blocked_mask
    idx = np.flatnonzero(allowed)
    sub = euclidean_cluster(xyz[idx], eps, min_points=1)
    seed_labels = np.unique(sub[seed_mask[idx]])
    out = np.zeros(len(xyz), dtype=bool)
    out[idx[np.isin(sub, seed_labels)]] = True
    return out


def select_object_points(uv, xyz, xyz_cam, box, *, shrink=0.8, eps_min=0.3, eps_k=2.0, min_points=3,
                         min_ratio=0.3, w_center=0.5, w_cov=0.5, min_final=5):
    """單一偵測框的完整流程。uv / xyz / xyz_cam 是已去地面、已在影像內的點（FramePoints 的欄位）。

    回傳 dict：
      status   ok / no_points / no_cluster / few_points
      in_box   完整 bbox 內的點 mask（對應輸入的點）
      core     core 區域內的點 mask
      labels   core 點的群編號（長度 = core.sum()，-1 為雜訊）
      clusters score_clusters 的結果
      chosen   選中的群 id（None 表示沒選到）
      final    最終物件點 mask（對應輸入的點）
      eps, dist（相機座標歐氏距離中位數 m）, depth（相機 z 中位數 m）, extent, iou, cov_w, cov_h
    """
    box = np.asarray(box, dtype=float)
    core_box = shrink_box(box, shrink)
    full = in_box(uv, box)
    core = full & in_box(uv, core_box)
    res = {"status": "no_points", "in_box": full, "core": core, "labels": np.zeros(0, int),
           "clusters": [], "chosen": None, "final": np.zeros(len(uv), bool), "core_box": core_box,
           "eps": np.nan, "dist": np.nan, "depth": np.nan, "extent": None, "iou": np.nan,
           "cov_w": np.nan, "cov_h": np.nan}
    if core.sum() < min_points:
        return res
    rng_core = np.linalg.norm(xyz_cam[core], axis=1)
    eps = adaptive_eps(np.median(rng_core), eps_min, eps_k)
    labels = euclidean_cluster(xyz[core], eps, min_points)
    clusters = score_clusters(uv[core], labels, core_box, min_ratio, w_center, w_cov)
    res.update(eps=eps, labels=labels, clusters=clusters)
    cands = [c for c in clusters if c["candidate"]]
    if not cands:
        res["status"] = "no_cluster"
        return res
    best = max(cands, key=lambda c: c["score"])
    res["chosen"] = best["id"]

    core_idx = np.flatnonzero(core)
    seed = np.zeros(len(uv), bool)
    seed[core_idx[labels == best["id"]]] = True
    # 只在完整 bbox 內長；core 內不屬於選中群的點視為擋住
    sub = np.flatnonzero(full)
    final_sub = grow(xyz[sub], seed[sub], core[sub] & ~seed[sub], eps)
    final = np.zeros(len(uv), bool)
    final[sub[final_sub]] = True

    ext = extent(uv[final])
    cw, ch = coverage(ext, box)
    res.update(final=final, extent=ext, iou=box_iou(ext, box), cov_w=cw, cov_h=ch,
               dist=float(np.median(np.linalg.norm(xyz_cam[final], axis=1))),
               depth=float(np.median(xyz_cam[final][:, 2])),
               status="ok" if final.sum() >= min_final else "few_points")
    return res
