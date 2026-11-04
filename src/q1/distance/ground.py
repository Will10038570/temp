"""LiDAR 地面平面估計（os1 座標）。"""
import numpy as np


def fit_ground(xyz, z_range=(-1.5, -0.3), max_range=30.0, iters=100, inlier_thr=0.1,
               max_tilt_deg=15.0, sample=5000, seed=0):
    """RANSAC 擬合地面平面，回傳 (normal, d)，平面為 normal·p + d = 0，normal 朝上。

    只用 LiDAR 座標下高度在 z_range、水平距離 < max_range 的點當候選（os1 離地約 0.86 m）。
    候選點太少或找不到夠平的平面時，退回水平面 z = 候選點高度的 10 百分位。
    """
    r = np.linalg.norm(xyz[:, :2], axis=1)
    cand = xyz[(xyz[:, 2] > z_range[0]) & (xyz[:, 2] < z_range[1]) & (r < max_range)]
    if len(cand) < 50:
        z0 = np.percentile(xyz[:, 2], 10) if len(xyz) else z_range[0]
        return np.array([0.0, 0.0, 1.0]), -z0
    rng = np.random.default_rng(seed)
    if len(cand) > sample:
        cand = cand[rng.choice(len(cand), sample, replace=False)]
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best, best_n = None, 0
    for _ in range(iters):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        if n[2] < 0:
            n = -n
        if n[2] < cos_tilt:
            continue
        inl = np.abs(cand @ n - n @ p[0]) < inlier_thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    if best is None:
        z0 = np.percentile(cand[:, 2], 10)
        return np.array([0.0, 0.0, 1.0]), -z0
    # 用所有 inlier 做最小平方修正
    pts = cand[best]
    c = pts.mean(axis=0)
    n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]  # 只需 Vt；預設 full_matrices=True 會多算 MxM 的 U，慢且耗記憶體
    if n[2] < 0:
        n = -n
    return n, -n @ c


def height_above(xyz, plane):
    n, d = plane
    return xyz @ n + d
