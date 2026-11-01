"""CODa 讀取工具。

檔名規則與格式依官方 coda-devkit（docs/DATA_REPORT.md）：
  2d_rect/{cam}/{seq}/2d_rect_{cam}_{seq}_{frame}.png
  3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_{frame}.json
  3d_raw/os1/{seq}/3d_raw_os1_{seq}_{frame}.bin     float32, N x 4 (x, y, z, intensity)
  3d_comp/os1/{seq}/3d_comp_os1_{seq}_{frame}.bin   同上，做過 ego 運動補償（tiny split 只有這個）
  calibrations/{seq}/calib_os1_to_{cam}.yaml, calib_{cam}_intrinsics.yaml
"""
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def yaw_rates(root, seq):
    """回傳 seq 每個 frame 的 yaw rate（°/s，繞車體 z 軸，左轉為正），index 即 frame 編號。

    poses/dense/{seq}.txt 第 i 行對應 frame i，格式 ts x y z qw qx qy qz。
    用前後一幀的姿態差（中央差分）計算；首尾幀用單邊差分。
    部分序列的 pose 有 NaN 或範數明顯不為 1 的四元數，用到這些 pose 的 frame 回傳 NaN。
    """
    from scipy.spatial.transform import Rotation as R
    poses = np.loadtxt(Path(root) / "poses" / "dense" / f"{seq}.txt", ndmin=2)
    quat = poses[:, [5, 6, 7, 4]]
    norm = np.linalg.norm(quat, axis=1)
    valid = np.isfinite(poses).all(axis=1) & (np.abs(norm - 1) < 0.1)
    quat[~valid] = [0, 0, 0, 1]
    rot = R.from_quat(quat)
    idx = np.arange(len(poses))
    a, b = np.clip(idx - 1, 0, len(poses) - 1), np.clip(idx + 1, 0, len(poses) - 1)
    dt = poses[b, 0] - poses[a, 0]
    dyaw = (rot[a].inv() * rot[b]).as_rotvec()[:, 2]
    rate = np.degrees(dyaw / np.where(dt > 0, dt, np.inf))
    rate[~(valid[a] & valid[b])] = np.nan
    return rate


def _yaml_matrix(node):
    return np.array(node["data"], dtype=np.float64).reshape(node["rows"], node["cols"])


class CodaSequence:
    def __init__(self, root, seq, cam="cam0"):
        self.root = Path(root)
        self.seq = str(seq)
        self.cam = cam

    def image_path(self, frame):
        return self.root / "2d_rect" / self.cam / self.seq / f"2d_rect_{self.cam}_{self.seq}_{frame}.png"

    def bbox_path(self, frame):
        return self.root / "3d_bbox" / "os1" / self.seq / f"3d_bbox_os1_{self.seq}_{frame}.json"

    def lidar_dir(self):
        """優先用 3d_raw，沒有時改用 3d_comp。回傳 (資料夾, 模態名稱)。"""
        for mod in ("3d_raw", "3d_comp"):
            d = self.root / mod / "os1" / self.seq
            if d.is_dir():
                return d, mod
        raise FileNotFoundError(f"{self.root} 沒有 seq {self.seq} 的 3d_raw/os1 或 3d_comp/os1")

    def lidar_path(self, frame):
        d, mod = self.lidar_dir()
        return d / f"{mod}_os1_{self.seq}_{frame}.bin"

    def lidar_frames(self):
        """有點雲的 frame，依編號排序。"""
        d, _ = self.lidar_dir()
        return sorted(int(p.stem.rsplit("_", 1)[1]) for p in d.glob("*.bin"))

    def annotated_frames(self):
        """有 3D bbox 標註、且有對應 RGB 影像的 frame，依編號排序。"""
        frames = []
        for p in (self.root / "3d_bbox" / "os1" / self.seq).glob("*.json"):
            frame = int(p.stem.rsplit("_", 1)[1])
            if self.image_path(frame).exists():
                frames.append(frame)
        return sorted(frames)

    def read_image(self, frame):
        # cv2.imread 在 Windows 讀不了含中文的路徑，改用 imdecode
        data = np.fromfile(self.image_path(frame), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.image_path(frame))
        return img

    def read_bboxes(self, frame):
        """沒有標註的 frame 回傳空 list。"""
        if not self.bbox_path(frame).exists():
            return []
        with open(self.bbox_path(frame), encoding="utf-8") as f:
            return json.load(f)["3dbbox"]

    def read_lidar(self, frame):
        return np.fromfile(self.lidar_path(frame), dtype=np.float32).reshape(-1, 4)

    def read_calib(self):
        """回傳 dict：
        T_lidar_to_cam  4x4，os1 → 相機座標
        P_lidar_to_rect 3x4，os1 → 校正後（rectified）影像像素
        K, dist         原始（未校正）影像的內參與畸變係數
        width, height
        """
        calib_dir = self.root / "calibrations" / self.seq
        with open(calib_dir / f"calib_os1_to_{self.cam}.yaml", encoding="utf-8") as f:
            ext = yaml.safe_load(f)
        with open(calib_dir / f"calib_{self.cam}_intrinsics.yaml", encoding="utf-8") as f:
            intr = yaml.safe_load(f)
        return {
            "T_lidar_to_cam": _yaml_matrix(ext["extrinsic_matrix"]),
            "P_lidar_to_rect": _yaml_matrix(ext["projection_matrix"]),
            "K": _yaml_matrix(intr["camera_matrix"]),
            "dist": np.array(intr["distortion_coefficients"]["data"], dtype=np.float64),
            "width": intr["image_width"],
            "height": intr["image_height"],
        }
