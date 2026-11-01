"""整幀點雲前處理：去近點 → 去地面 → 投影到影像 → 只留影像內的點。每幀做一次，所有偵測框共用。"""
from dataclasses import dataclass

import numpy as np

from q1.data.projection import project_points
from q1.distance.ground import fit_ground, height_above


@dataclass
class FramePoints:
    uv: np.ndarray       # Nx2 像素座標（2d_rect 影像）
    xyz: np.ndarray      # Nx3 os1 座標（m），分群用
    xyz_cam: np.ndarray  # Nx3 相機座標（m），算距離用
    plane: tuple         # 地面平面 (normal, d)，os1 座標


def prepare_points(points, calib, min_range=0.5, min_depth=0.5, ground_thr=0.15):
    """points：N x 4（x, y, z, intensity）原始點雲。

    min_range  離 LiDAR 小於此距離（m）的點丟掉（沒有回波的點落在原點附近）
    min_depth  相機深度小於此值（m）的點丟掉（在相機後方或太近）
    ground_thr 離地高度 <= 此值（m）視為地面
    """
    xyz = points[:, :3].astype(np.float64)
    xyz = xyz[np.linalg.norm(xyz, axis=1) >= min_range]
    plane = fit_ground(xyz)
    xyz = xyz[height_above(xyz, plane) > ground_thr]
    uv, depth = project_points(xyz, calib)
    w, h = calib["width"], calib["height"]
    keep = (depth > min_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    T = calib["T_lidar_to_cam"]
    xyz = xyz[keep]
    return FramePoints(uv=uv[keep], xyz=xyz, xyz_cam=xyz @ T[:3, :3].T + T[:3, 3], plane=plane)
