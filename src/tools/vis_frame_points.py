"""用 Open3D 逐幀檢視點雲前處理（q1/distance/points.py prepare_points）的結果 FramePoints。

保留下來的點（FramePoints.xyz）依相機距離上色；被丟掉的點依原因用不同灰階／顏色畫，
可疊上地面平面、相機視野（frustum）與 3D bbox 標註，用來檢查去近點、去地面、影像範圍裁切是否合理。

範例（在 src/ 下執行）：
  python tools/vis_frame_points.py --seq 0
  python tools/vis_frame_points.py --seq 20 --frame 956 --ground-thr 0.2
  python tools/vis_frame_points.py --seq 0 --save --num 20   # 不開互動視窗，每幀存成截圖

點的顏色：
  保留的點（FramePoints）  依相機座標距離（或離地高度，按 C 切換），近藍遠紅
  被當成地面丟掉的點        棕色
  不在影像內／相機後方的點  暗灰
  （離 LiDAR < min_range 的點不畫）

互動按鍵：
  N / →  下一幀        P / ←  上一幀
  R      顯示／隱藏被丟掉的點
  C      切換保留點上色：相機距離 / 離地高度
  G      顯示／隱藏地面平面
  F      顯示／隱藏相機視野
  B      顯示／隱藏 3D bbox
  Q      離開（Open3D 內建）
  滑鼠左鍵旋轉、右鍵或 Ctrl+左鍵平移、滾輪縮放
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.data.classes import TARGET_CLASSES, gt_boxes  # noqa: E402
from q1.data.coda import CodaSequence  # noqa: E402
from q1.data.projection import BOX_EDGES, bbox_corners, project_points  # noqa: E402
from q1.distance.ground import height_above  # noqa: E402
from q1.distance.points import prepare_points  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from q1.pipeline import DistanceConfig  # noqa: E402
from vis.draw import class_color, colormap  # noqa: E402

KEY_RIGHT, KEY_LEFT = 262, 263  # GLFW key codes
GROUND_RGB = (0.45, 0.30, 0.15)
OUT_RGB = (0.25, 0.25, 0.25)
COLOR_MODES = ("dist", "height")


def split_points(points, calib, fp, min_range, min_depth, ground_thr):
    """把原始點分成 near / ground / out（影像外或相機後方）/ kept，回傳各自的 os1 xyz。

    判斷方式與 prepare_points 相同，地面平面直接用 fp.plane（不重新擬合）。
    kept 的點數應等於 len(fp.xyz)。
    """
    xyz = points[:, :3].astype(np.float64)
    near = np.linalg.norm(xyz, axis=1) < min_range
    rest = xyz[~near]
    ground = height_above(rest, fp.plane) <= ground_thr
    above = rest[~ground]
    uv, depth = project_points(above, calib)
    w, h = calib["width"], calib["height"]
    inside = (depth > min_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return {"near": xyz[near], "ground": rest[ground], "out": above[~inside], "kept": above[inside]}


def make_cloud(xyz, colors):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    colors = np.asarray(colors, dtype=np.float64)
    if colors.ndim == 1:
        pcd.paint_uniform_color(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_ground(plane, size=40.0, step=2.0):
    """地面平面畫成網格線（os1 座標，以 LiDAR 正下方為中心）。"""
    n, d = plane
    a = np.cross(n, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    o = -d * n  # 原點在平面上的投影
    ticks = np.arange(-size / 2, size / 2 + 1e-6, step)
    pts, lines = [], []
    for t in ticks:
        for u, v in ((a, b), (b, a)):
            pts += [o + t * u - size / 2 * v, o + t * u + size / 2 * v]
            lines.append((len(pts) - 2, len(pts) - 1))
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(pts)), o3d.utility.Vector2iVector(lines))
    ls.paint_uniform_color((0.35, 0.25, 0.12))
    return ls


def pixel_to_lidar(calib, u, v, z):
    """影像像素 (u, v) 在相機深度 z 的點 → os1 座標。解 P·[X,1] = w·[u,v,1] 與 depth(X) = z。"""
    P, T = calib["P_lidar_to_rect"], calib["T_lidar_to_cam"]
    A = np.zeros((4, 4))
    rhs = np.zeros(4)
    A[:3, :3] = P[:, :3]
    A[:3, 3] = -np.array([u, v, 1.0])
    rhs[:3] = -P[:, 3]
    A[3, :3] = T[2, :3]
    rhs[3] = z - T[2, 3]
    return np.linalg.solve(A, rhs)[:3]


def make_frustum(calib, depth):
    """相機視野：相機中心到影像四角在 depth 處的四條線 + 遠端矩形。"""
    T = calib["T_lidar_to_cam"]
    center = -T[:3, :3].T @ T[:3, 3]
    w, h = calib["width"], calib["height"]
    corners = [pixel_to_lidar(calib, u, v, depth) for u, v in ((0, 0), (w, 0), (w, h), (0, h))]
    pts = np.vstack([center, *corners])
    lines = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
    ls.paint_uniform_color((0.9, 0.9, 0.9))
    return ls


def make_boxes(boxes, classes):
    geoms = []
    for box in boxes:
        if classes and box["classId"] not in classes:
            continue
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(bbox_corners(box)),
                                  o3d.utility.Vector2iVector(BOX_EDGES))
        b, g, r = class_color(box["classId"])
        ls.paint_uniform_color((r / 255, g / 255, b / 255))
        geoms.append(ls)
    return geoms


class Viewer:
    def __init__(self, seq, calib, frames, start_idx, cfg, args):
        self.seq, self.calib, self.frames, self.idx = seq, calib, frames, start_idx
        self.cfg, self.args = cfg, args
        self.show = {"removed": not args.hide_removed, "ground": True, "frustum": True, "boxes": not args.no_bbox}
        self.color_mode = 0
        self.geoms = []

    def kept_colors(self, fp):
        if COLOR_MODES[self.color_mode] == "dist":
            return colormap(np.linalg.norm(fp.xyz_cam, axis=1), 0, self.args.max_range)
        return colormap(height_above(fp.xyz, fp.plane), 0, self.args.max_height)

    def load(self, vis, reset_view=False):
        frame = self.frames[self.idx]
        for g in self.geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        points = self.seq.read_lidar(frame)
        kw = self.cfg.point_kwargs()
        fp = prepare_points(points, self.calib, **kw)
        parts = split_points(points, self.calib, fp, **kw)
        if len(parts["kept"]) != len(fp.xyz):
            print(f"  警告：split_points 保留 {len(parts['kept'])} 點，prepare_points 保留 {len(fp.xyz)} 點，"
                  "前處理邏輯可能已改動，請同步更新本工具")

        self.geoms = [make_cloud(fp.xyz, self.kept_colors(fp))]
        if self.show["removed"]:
            self.geoms += [make_cloud(parts["ground"], GROUND_RGB), make_cloud(parts["out"], OUT_RGB)]
        if self.show["ground"]:
            self.geoms.append(make_ground(fp.plane))
        if self.show["frustum"]:
            self.geoms.append(make_frustum(self.calib, self.args.frustum_depth))
        boxes = gt_boxes(self.seq.read_bboxes(frame))
        if self.show["boxes"]:
            self.geoms += make_boxes(boxes, self.args.classes)
        for i, g in enumerate(self.geoms):
            vis.add_geometry(g, reset_bounding_box=reset_view and i == 0)

        n, d = fp.plane
        print(f"[{self.idx + 1}/{len(self.frames)}] seq {self.seq.seq} frame {frame}: 原始 {len(points)} 點 → "
              f"近點 -{len(parts['near'])}、地面 -{len(parts['ground'])}、影像外 -{len(parts['out'])} → "
              f"保留 {len(fp.xyz)} 點 | 地面 n=({n[0]:.3f},{n[1]:.3f},{n[2]:.3f}) d={d:.3f} | "
              f"{len(boxes)} 個 bbox | 上色：{COLOR_MODES[self.color_mode]}")

    def step(self, delta):
        def cb(vis):
            self.idx = (self.idx + delta) % len(self.frames)
            self.load(vis)
            return True
        return cb

    def toggle(self, name):
        def cb(vis):
            self.show[name] = not self.show[name]
            self.load(vis)
            return True
        return cb

    def cycle_color(self, vis):
        self.color_mode = (self.color_mode + 1) % len(COLOR_MODES)
        self.load(vis)
        return True


def main():
    paths = load_paths()
    default = DistanceConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"], help="CODa 根目錄")
    p.add_argument("--cam", default="cam0")
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--frame", type=int, help="從這個 frame 開始；預設為第一幀")
    p.add_argument("--annotated-only", action="store_true", help="只瀏覽有 3D bbox 標註的 frame")
    g = p.add_argument_group("前處理參數（見 q1/pipeline.py DistanceConfig）")
    for name in default.point_kwargs():
        g.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(default, name))
    p.add_argument("--hide-removed", action="store_true", help="一開始不畫被丟掉的點（可按 R 切換）")
    p.add_argument("--no-bbox", action="store_true", help="一開始不畫 3D bbox（可按 B 切換）")
    p.add_argument("--classes", nargs="*", default=TARGET_CLASSES, choices=TARGET_CLASSES, help="只畫這些類別的 bbox")
    p.add_argument("--max-range", type=float, default=50.0, help="距離上色的上限（m）")
    p.add_argument("--max-height", type=float, default=3.0, help="離地高度上色的上限（m）")
    p.add_argument("--frustum-depth", type=float, default=30.0, help="相機視野畫到多深（m）")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--zoom", type=float, default=0.03, help="初始視角縮放，越小越近")
    p.add_argument("--save", action="store_true", help="不互動，每幀截圖存到 --outdir")
    p.add_argument("--num", type=int, help="--save 時最多存幾幀")
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "vis_frame_points")
    args = p.parse_args()
    cfg = DistanceConfig(**{k: getattr(args, k) for k in default.point_kwargs()})

    seq = CodaSequence(args.root, args.seq, args.cam)
    calib = seq.read_calib()
    frames = seq.lidar_frames()
    if args.annotated_only:
        frames = [f for f in frames if seq.bbox_path(f).exists()]
    if not frames:
        sys.exit(f"seq {args.seq} 在 {args.root} 找不到點雲")
    if args.frame is not None and args.frame not in frames:
        sys.exit(f"frame {args.frame} 不在可瀏覽的 frame 中；前幾個：{frames[:10]}")
    start = frames.index(args.frame) if args.frame is not None else 0
    print(f"seq {args.seq}: {len(frames)} 幀，點雲來源 {seq.lidar_dir()[1]}/os1，相機 {args.cam}，"
          f"前處理參數 {cfg.point_kwargs()}")

    viewer = Viewer(seq, calib, frames, start, cfg, args)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"FramePoints seq {args.seq}", width=1280, height=800, visible=not args.save)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])
    viewer.load(vis, reset_view=True)
    ctr = vis.get_view_control()
    # 從車後上方往前看：os1 座標 +x 前、+y 左、+z 上
    ctr.set_front([-1, 0, 0.6])
    ctr.set_up([0, 0, 1])
    ctr.set_lookat([10, 0, 0])
    ctr.set_zoom(args.zoom)

    if args.save:
        outdir = args.outdir / str(args.seq)
        outdir.mkdir(parents=True, exist_ok=True)
        end = len(frames) if args.num is None else min(len(frames), start + args.num)
        for i in range(start, end):
            viewer.idx = i
            viewer.load(vis)
            vis.poll_events()
            vis.update_renderer()
            # Open3D 寫檔不支援中文路徑，先取影像再用 OpenCV 寫
            img = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
            cv2.imencode(".jpg", img[:, :, ::-1])[1].tofile(outdir / f"{frames[i]}.jpg")
        vis.destroy_window()
        print(f"截圖存到 {outdir}")
        return

    for key in (ord("N"), KEY_RIGHT):
        vis.register_key_callback(key, viewer.step(1))
    for key in (ord("P"), KEY_LEFT):
        vis.register_key_callback(key, viewer.step(-1))
    vis.register_key_callback(ord("R"), viewer.toggle("removed"))
    vis.register_key_callback(ord("G"), viewer.toggle("ground"))
    vis.register_key_callback(ord("F"), viewer.toggle("frustum"))
    vis.register_key_callback(ord("B"), viewer.toggle("boxes"))
    vis.register_key_callback(ord("C"), viewer.cycle_color)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
