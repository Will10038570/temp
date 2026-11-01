"""用 Open3D 逐幀 3D 檢視距離估計流程（q1/pipeline.py estimate_frame）選出的物件點，疊上 3D bbox GT，
檢查選群有沒有圈到 GT 物件上。

讀現有的 YOLO 結果（detections.csv），不重跑 YOLO。只瀏覽 detections.csv 裡有偵測框的 frame。
畫的點是前處理後的 FramePoints（去近點、去地面、只留影像內的點，見 q1/distance/points.py）。

範例（在 src/ 下執行）：
  python tools/vis_cluster_3d.py --seq 0
  python tools/vis_cluster_3d.py --seq 0 --frame 481 --annotated-only
  python tools/vis_cluster_3d.py --seq 0 --w-cov 0.3 --w-center 0.7      # 換選群參數
  python tools/vis_cluster_3d.py --seq 0 --save --num 20                  # 不開互動視窗，每幀存成截圖

畫面：
  選中的點（所有偵測框 res["final"] 的聯集）  紅色
  沒被選到的 FramePoints                      藍色
  3D bbox GT                                   綠色（只畫至少一個角點投影在影像內、且在相機前方的 box）
  YOLO 標籤「#編號 類別 conf=信心分數」                 白色，浮在每個偵測框選中點的上方
                                               （沒選到點的偵測框，標在框內點的中位數位置）

終端機會列出每個偵測框：類別、conf、status、選中點數、選中點落在任一 GT box 內的比例（in_gt）、距離。
標籤的 #編號 對應終端機的編號。

互動按鍵：
  N / →  下一幀        P / ←  上一幀
  B      顯示／隱藏 3D bbox
  L      顯示／隱藏 YOLO 標籤
  Q      離開（Open3D 內建）
  滑鼠左鍵旋轉、右鍵或 Ctrl+左鍵平移、滾輪縮放
"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.data.classes import TARGET_CLASSES, gt_boxes  # noqa: E402
from q1.data.coda import CodaSequence  # noqa: E402
from q1.data.projection import BOX_EDGES, bbox_corners, in_box3d, project_points  # noqa: E402
from q1.detection.yolo import load_detections  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from q1.pipeline import DistanceConfig, estimate_frame  # noqa: E402

KEY_RIGHT, KEY_LEFT = 262, 263  # GLFW key codes
SELECTED_RGB = (1.0, 0.15, 0.15)
UNSELECTED_RGB = (0.2, 0.45, 1.0)
BOX_RGB = (0.1, 0.9, 0.2)
LABEL_RGB = (1.0, 1.0, 1.0)  # 不用類別色：bicycle 偏紅、car 偏綠，會跟選中點、GT box 混淆
# 文字網格在 XY 平面、朝 +z；轉成：字往 -y 寫（從車後往前看是往右）、字的上方朝 +z、正面朝 -x（面向預設視角）
TEXT_ROT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def make_cloud(xyz, rgb):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64).reshape(-1, 3)))
    pcd.paint_uniform_color(rgb)
    return pcd


def make_box(corners):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(corners), o3d.utility.Vector2iVector(BOX_EDGES))
    ls.paint_uniform_color(BOX_RGB)
    return ls


def make_label(text, anchor, height, rgb):
    """3D 文字網格，底邊中心放在 anchor（os1 座標），字高約 height 公尺。"""
    mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0).to_legacy()
    v = np.asarray(mesh.vertices)
    v = (v - [(v[:, 0].min() + v[:, 0].max()) / 2, v[:, 1].min(), 0]) * (height / 14.0)  # 字高約 14 單位
    mesh.vertices = o3d.utility.Vector3dVector(v @ TEXT_ROT.T + anchor)
    mesh.paint_uniform_color(rgb)
    return mesh


def in_image(corners, calib, min_depth):
    """3D box 是否出現在影像內：至少一個角點在相機前方（深度 > min_depth）且投影落在影像範圍內。
    判斷方式與 prepare_points 對點的影像範圍裁切相同。"""
    uv, depth = project_points(corners, calib)
    w, h = calib["width"], calib["height"]
    ok = (depth > min_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return bool(ok.any())


class Viewer:
    def __init__(self, seq, calib, frames, dets, start_idx, cfg, args):
        self.seq, self.calib, self.frames, self.dets, self.idx = seq, calib, frames, dets, start_idx
        self.cfg, self.args = cfg, args
        self.show = {"boxes": not args.no_bbox, "labels": not args.no_label}
        self.geoms = []

    def labels(self, fp, results):
        geoms = []
        for j, (pred, res) in enumerate(results):
            pts = fp.xyz[res["final"]] if res["final"].any() else fp.xyz[res["in_box"]]
            if not len(pts):
                continue
            anchor = np.median(pts, axis=0)
            anchor[2] = pts[:, 2].max() + 0.2
            geoms.append(make_label(f"#{j} {pred['category']} conf={pred['score']:.2f}", anchor,
                                    self.args.label_size, LABEL_RGB))
        return geoms

    def load(self, vis, reset_view=False):
        frame = self.frames[self.idx]
        for g in self.geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        preds = self.dets.get((int(self.seq.seq), frame), [])
        fp, results = estimate_frame(self.seq.read_lidar(frame), self.calib, preds, self.cfg, set(self.args.classes))

        selected = np.zeros(len(fp.xyz), bool)
        for _, res in results:
            selected |= res["final"]

        boxes = [b for b in gt_boxes(self.seq.read_bboxes(frame)) if b["classId"] in self.args.classes]
        corners = [bbox_corners(b) for b in boxes]
        n_all = len(boxes)
        keep = [in_image(c, self.calib, self.cfg.min_depth) for c in corners]
        boxes = [b for b, k in zip(boxes, keep) if k]
        corners = [c for c, k in zip(corners, keep) if k]

        self.geoms = [make_cloud(fp.xyz[~selected], UNSELECTED_RGB), make_cloud(fp.xyz[selected], SELECTED_RGB)]
        if self.show["boxes"]:
            self.geoms += [make_box(c) for c in corners]
        if self.show["labels"]:
            self.geoms += self.labels(fp, results)
        for i, g in enumerate(self.geoms):
            vis.add_geometry(g, reset_bounding_box=reset_view and i == 0)

        print(f"[{self.idx + 1}/{len(self.frames)}] seq {self.seq.seq} frame {frame}: "
              f"FramePoints {len(fp.xyz)} 點、{len(results)} 個偵測框、選中 {selected.sum()} 點、"
              f"影像內 GT box {len(boxes)}/{n_all} 個" + ("" if n_all else "（此幀無標註）"))
        for j, (pred, res) in enumerate(results):
            pts = fp.xyz[res["final"]]
            if len(pts) and corners:
                hit = np.zeros(len(pts), bool)
                for c in corners:
                    hit |= in_box3d(pts, c)
                in_gt = f"{hit.mean() * 100:5.1f}%"
            else:
                in_gt = "  -  "
            print(f"   #{j:<2d} {pred['category']:<8s} conf={pred['score']:.2f}  {res['status']:<10s} "
                  f"n={int(res['final'].sum()):<4d} in_gt={in_gt}  dist={res['dist']:.2f}m")

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


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"], help="CODa 根目錄")
    p.add_argument("--dets", type=Path, default=paths["outputs"] / "vis_lidar2img_yolo" / "detections.csv")
    p.add_argument("--cam", default="cam0")
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--frame", type=int, help="從這個 frame 開始；預設為第一幀")
    p.add_argument("--annotated-only", action="store_true", help="只瀏覽有 3D bbox 標註的 frame")
    p.add_argument("--classes", nargs="+", default=TARGET_CLASSES, choices=TARGET_CLASSES,
                   help="只處理這些類別的偵測框與 GT box")
    p.add_argument("--conf", type=float, default=0.25)
    g = p.add_argument_group("距離估計參數（見 q1/pipeline.py DistanceConfig）")
    for f in fields(DistanceConfig):
        g.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    p.add_argument("--no-bbox", action="store_true", help="一開始不畫 3D bbox（可按 B 切換）")
    p.add_argument("--no-label", action="store_true", help="一開始不畫 YOLO 標籤（可按 L 切換）")
    p.add_argument("--label-size", type=float, default=0.35, help="標籤字高（m）")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--zoom", type=float, default=0.03, help="初始視角縮放，越小越近")
    p.add_argument("--save", action="store_true", help="不互動，每幀截圖存到 --outdir")
    p.add_argument("--num", type=int, help="--save 時最多存幾幀")
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "vis_cluster_3d")
    args = p.parse_args()
    cfg = DistanceConfig(**{f.name: getattr(args, f.name) for f in fields(DistanceConfig)})

    dets = load_detections(args.dets, set(args.classes), args.conf)
    seq = CodaSequence(args.root, args.seq, args.cam)
    frames = sorted(f for s, f in dets if s == args.seq)
    if args.annotated_only:
        frames = [f for f in frames if seq.bbox_path(f).exists()]
    if not frames:
        sys.exit(f"seq {args.seq} 在 {args.dets} 沒有偵測結果")
    if args.frame is not None and args.frame not in frames:
        sys.exit(f"frame {args.frame} 不在可瀏覽的 frame 中；前幾個：{frames[:10]}")
    start = frames.index(args.frame) if args.frame is not None else 0
    print(f"seq {args.seq}: {len(frames)} 幀有偵測框，相機 {args.cam}")

    viewer = Viewer(seq, seq.read_calib(), frames, dets, start, cfg, args)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"Cluster 3D seq {args.seq}", width=1280, height=800, visible=not args.save)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.light_on = False  # 文字網格不受光照影響，顏色才不會變暗
    opt.mesh_show_back_face = True  # 轉到背面時文字也看得到
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
    vis.register_key_callback(ord("B"), viewer.toggle("boxes"))
    vis.register_key_callback(ord("L"), viewer.toggle("labels"))
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
