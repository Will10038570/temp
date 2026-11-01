"""3D bbox → 影像投影。

角點計算與 coda-devkit helpers/geometry.py 的 get_3dbbox_corners 相同。
devkit 的 helpers 會 import ROS，Windows 上無法直接使用，所以在這裡重寫。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# 角點順序：0-3 為底面、4-7 為頂面，與 devkit 一致
_X_SIGN = np.array([-1, -1, 1, 1] * 2)
_Y_SIGN = np.array([1, -1, -1, 1] * 2)
_Z_SIGN = np.array([-1] * 4 + [1] * 4)

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def bbox_corners(box):
    """CODa 3D bbox dict → 8x3 角點（os1 座標，單位 m）。"""
    half = np.stack([_X_SIGN * box["l"], _Y_SIGN * box["w"], _Z_SIGN * box["h"]], axis=1) / 2
    if "r" in box:
        rot = R.from_euler("xyz", [box["r"], box["p"], box["y"]]).as_matrix()
    else:
        rot = R.from_quat([box["qx"], box["qy"], box["qz"], box["qw"]]).as_matrix()
    return half @ rot.T + np.array([box["cX"], box["cY"], box["cZ"]])


def in_box3d(xyz, corners):
    """點是否在 8 角點定義的 3D box 內（可旋轉）。用角點 0 的三條邊當作 box 的三個軸。"""
    c = corners.mean(axis=0)
    axes = [corners[b] - corners[a] for a, b in BOX_EDGES if a == 0 or b == 0]
    d = xyz - c
    inside = np.ones(len(xyz), bool)
    for e in axes:
        inside &= np.abs(d @ e) <= np.dot(e, e) / 2 + 1e-9
    return inside


def project_points(pts, calib):
    """os1 座標的 Nx3 點 → (Nx2 像素座標, N 個相機座標深度 z，單位 m)。

    用 calib_os1_to_cam 的 projection_matrix（os1 → 校正後像素），對應 2d_rect 影像。
    devkit vis_annos_rgb.py 改用原始 K 與畸變係數，套在 2d_rect 上會偏移。
    """
    T = calib["T_lidar_to_cam"]
    depth = pts @ T[:3, :3].T[:, 2] + T[2, 3]
    uvw = np.hstack([pts, np.ones((len(pts), 1))]) @ calib["P_lidar_to_rect"].T
    return uvw[:, :2] / uvw[:, 2:3], depth


def project_bbox(box, calib, min_depth=0.1):
    """回傳 (8x2 角點像素, 2D 框 [x1, y1, x2, y2]，已裁到影像範圍內)。

    任一角點在相機後方（深度 < min_depth）或框完全在影像外時回傳 None：
    角點在相機後方時，投影會翻轉，算出的 2D 框不正確。
    """
    uv, depth = project_points(bbox_corners(box), calib)
    if np.any(depth < min_depth):
        return None
    w, h = calib["width"], calib["height"]
    x1, y1 = np.clip(uv.min(axis=0), 0, [w - 1, h - 1])
    x2, y2 = np.clip(uv.max(axis=0), 0, [w - 1, h - 1])
    if x2 <= x1 or y2 <= y1:
        return None
    return uv, np.array([x1, y1, x2, y2])
