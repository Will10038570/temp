"""用 Open3D 逐幀檢視 CODa 一個 sequence 的 os1 點雲，點依距離上色，可疊上 3D bbox 標註。

優先讀 3d_raw/os1，沒有時改讀 3d_comp/os1（tiny split 只有 3d_comp）。

範例（在 src/ 下執行）：
  python tools/vis_coda_pointcloud.py --seq 0
  python tools/vis_coda_pointcloud.py --seq 20 --frame 956 --classes person bicycle
  python tools/vis_coda_pointcloud.py --seq 0 --save      # 不開互動視窗，每幀存成截圖

互動按鍵：
  N / →  下一幀        P / ←  上一幀
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
from q1.data.projection import BOX_EDGES, bbox_corners  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from vis.draw import class_color, colormap  # noqa: E402

KEY_RIGHT, KEY_LEFT = 262, 263  # GLFW key codes


def make_cloud(points, max_range, min_range):
    xyz = points[:, :3]
    dist = np.linalg.norm(xyz, axis=1)
    keep = dist >= min_range  # 沒有回波的點落在原點附近
    xyz, dist = xyz[keep], dist[keep]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    pcd.colors = o3d.utility.Vector3dVector(colormap(dist, 0, max_range))
    return pcd


def make_boxes(boxes, classes):
    geoms = []
    for box in gt_boxes(boxes):  # CODa 類別歸類成 person / bicycle / car，其餘丟掉
        if classes and box["classId"] not in classes:
            continue
        ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(bbox_corners(box)),
                                  o3d.utility.Vector2iVector(BOX_EDGES))
        b, g, r = class_color(box["classId"])
        ls.paint_uniform_color((r / 255, g / 255, b / 255))
        geoms.append(ls)
    return geoms


class Viewer:
    def __init__(self, seq, frames, start_idx, args):
        self.seq, self.frames, self.idx, self.args = seq, frames, start_idx, args
        self.show_boxes = not args.no_bbox
        self.geoms = []

    def load(self, vis, reset_view=False):
        frame = self.frames[self.idx]
        for g in self.geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        boxes = gt_boxes(self.seq.read_bboxes(frame))
        self.geoms = [make_cloud(self.seq.read_lidar(frame), self.args.max_range, self.args.min_range)]
        if self.show_boxes:
            self.geoms += make_boxes(boxes, self.args.classes)
        for i, g in enumerate(self.geoms):
            vis.add_geometry(g, reset_bounding_box=reset_view and i == 0)
        n_pts = len(self.geoms[0].points)
        print(f"[{self.idx + 1}/{len(self.frames)}] seq {self.seq.seq} frame {frame}: "
              f"{n_pts} 點, {len(boxes)} 個 bbox" + ("" if boxes else "（此幀無標註）"))

    def step(self, delta):
        def cb(vis):
            self.idx = (self.idx + delta) % len(self.frames)
            self.load(vis)
            return True
        return cb

    def toggle_boxes(self, vis):
        self.show_boxes = not self.show_boxes
        self.load(vis)
        return True


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"], help="CODa 根目錄")
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--frame", type=int, help="從這個 frame 開始；預設為第一幀")
    p.add_argument("--annotated-only", action="store_true", help="只瀏覽有 3D bbox 標註的 frame")
    p.add_argument("--no-bbox", action="store_true", help="一開始不畫 3D bbox（可按 B 切換）")
    p.add_argument("--classes", nargs="*", default=TARGET_CLASSES, choices=TARGET_CLASSES, help="只畫這些類別的 bbox（歸類後名稱）")
    p.add_argument("--max-range", type=float, default=50.0, help="距離上色的上限（m），超過的點同為最遠色")
    p.add_argument("--min-range", type=float, default=0.5, help="小於此距離（m）的點不畫")
    p.add_argument("--point-size", type=float, default=1.5)
    p.add_argument("--zoom", type=float, default=0.03, help="初始視角縮放，越小越近")
    p.add_argument("--save", action="store_true", help="不互動，每幀截圖存到 --outdir")
    p.add_argument("--num", type=int, help="--save 時最多存幾幀")
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "vis_pointcloud")
    args = p.parse_args()

    seq = CodaSequence(args.root, args.seq)
    frames = seq.lidar_frames()
    if args.annotated_only:
        frames = [f for f in frames if seq.bbox_path(f).exists()]
    if not frames:
        sys.exit(f"seq {args.seq} 在 {args.root} 找不到點雲")
    if args.frame is not None and args.frame not in frames:
        sys.exit(f"frame {args.frame} 不在可瀏覽的 frame 中；前幾個：{frames[:10]}")
    start = frames.index(args.frame) if args.frame is not None else 0
    print(f"seq {args.seq}: {len(frames)} 幀，點雲來源 {seq.lidar_dir()[1]}/os1")

    viewer = Viewer(seq, frames, start, args)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"CODa seq {args.seq}", width=1280, height=800, visible=not args.save)
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
            out = outdir / f"{frames[i]}.jpg"
            cv2.imencode(".jpg", img[:, :, ::-1])[1].tofile(out)
        vis.destroy_window()
        print(f"截圖存到 {outdir}")
        return

    for key in (ord("N"), KEY_RIGHT):
        vis.register_key_callback(key, viewer.step(1))
    for key in (ord("P"), KEY_LEFT):
        vis.register_key_callback(key, viewer.step(-1))
    vis.register_key_callback(ord("B"), viewer.toggle_boxes)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
