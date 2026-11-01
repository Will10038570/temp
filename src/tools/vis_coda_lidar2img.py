"""把 CODa 的 os1 點雲投影到 RGB（2d_rect）影像上，點依相機深度上色，用來檢查 LiDAR→相機的標定。

投影用 calib_os1_to_{cam}.yaml 的 projection_matrix（與 vis_coda_bbox3d.py 相同）。
看標定準不準時建議：
  - 看電線桿、柱子、建築邊緣、門框等垂直邊，點雲邊界有沒有貼齊影像邊緣
  - 看前景與背景交界的深度跳變是否落在物體輪廓上
  - 挑機器人靜止或慢速的 frame，排除相機/LiDAR 時間不同步造成的偏移
  - 加 --edges 把影像換成 Canny 邊緣圖，比較容易看出偏移

範例（在 src/ 下執行）：
  python tools/vis_coda_lidar2img.py --seq 0                      # 前 10 個有影像的 frame
  python tools/vis_coda_lidar2img.py --seq 0 --frame 683 --show --edges
  python tools/vis_coda_lidar2img.py --seq 0 --stride 50 --num 20 --bbox
  python tools/vis_coda_lidar2img.py --list frames.csv --outdir ../outputs/vis_lidar2img_selected  # CSV 需有 seq, frame 欄位
  python tools/vis_coda_lidar2img.py --all-seqs --all --yolo     # 全部 frame 跑 YOLO，框疊在點雲投影圖上
輸出：outputs/vis_lidar2img/{cam}/{seq}/{frame}.jpg
      加 --yolo 時預設 outputs/vis_lidar2img_yolo/{cam}/{seq}/{frame}.jpg，另存 timing.json 與 detections.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.data.classes import gt_boxes  # noqa: E402
from q1.data.coda import CodaSequence  # noqa: E402
from q1.data.projection import BOX_EDGES, project_bbox, project_points  # noqa: E402
from q1.detection.yolo import DET_FIELDS, YoloDetector, detection_rows  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from vis.draw import class_color, colormap, draw_detections  # noqa: E402


def make_background(img, args):
    """--edges：深色底 + 白色 Canny 邊緣；否則把原圖調暗，讓點比較顯眼。"""
    if args.edges:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
        bg = (gray * 0.25).astype(np.uint8)
        bg[edges > 0] = 255
        return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    return (img * args.dim).astype(np.uint8)


def draw_points(img, points, calib, args):
    """把點投到影像上，回傳畫上去的點數。遠的先畫、近的後畫，讓前景蓋住背景。"""
    xyz = points[:, :3].astype(np.float64)
    xyz = xyz[np.linalg.norm(xyz, axis=1) >= args.min_range]  # 沒有回波的點落在原點附近
    uv, depth = project_points(xyz, calib)
    h, w = img.shape[:2]
    keep = ((depth > args.min_depth) & (depth < args.max_depth)
            & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
    uv, depth = uv[keep], depth[keep]
    order = np.argsort(-depth)
    uv, depth = uv[order], depth[order]
    colors = (colormap(depth, args.min_depth, args.color_max)[:, ::-1] * 255).astype(np.uint8)  # RGB → BGR
    pts = uv.round().astype(np.int32)
    for (x, y), c in zip(pts, colors):
        cv2.circle(img, (int(x), int(y)), args.radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return len(pts)


def draw_boxes(img, boxes, calib):
    """只畫 person / bicycle / car 的 GT（CODa 類別先歸類，見 q1/data/classes.py）。"""
    for box in gt_boxes(boxes):
        res = project_bbox(box, calib)
        if res is None:
            continue
        pts = res[0].round().astype(np.int32)
        color = class_color(box["classId"])
        for a, b in BOX_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 1, cv2.LINE_AA)


def draw_title(img, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(img, text, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_frame(seq, frame, calib, args, yolo=None):
    raw = seq.read_image(frame)
    preds = yolo(raw, seq.seq, frame) if yolo else None  # 用原圖推論，不是調暗後的圖
    img = make_background(raw, args)
    n = draw_points(img, seq.read_lidar(frame), calib, args)
    if args.bbox:
        draw_boxes(img, seq.read_bboxes(frame), calib)
    title = f"seq {seq.seq} {seq.cam} frame {frame}  pts {n}  depth {args.min_depth:g}-{args.color_max:g}m"
    if preds is not None:
        draw_detections(img, preds, args.conf, title=title)  # 會畫框並蓋掉標題列
    else:
        draw_title(img, title)
    return img, n, preds


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"], help="CODa 根目錄")
    p.add_argument("--seq", type=int, nargs="+", help="序列編號，可給多個（不用 --list / --all-seqs 時必填）")
    p.add_argument("--all-seqs", action="store_true", help="處理 root 下所有序列")
    p.add_argument("--list", type=Path, help="CSV 清單（需有 seq, frame 欄位），畫出清單中所有 frame")
    p.add_argument("--cam", default="cam0", choices=["cam0", "cam1"])
    p.add_argument("--frame", type=int, nargs="*", help="指定 frame；不指定時從第一幀開始取")
    p.add_argument("--num", type=int, default=10, help="不指定 --frame 時最多畫幾幀")
    p.add_argument("--stride", type=int, default=1, help="每隔幾幀取一幀")
    p.add_argument("--all", action="store_true", help="畫出所有有點雲與影像的 frame（仍套用 --stride）")
    p.add_argument("--min-depth", type=float, default=0.5, help="相機深度小於此值（m）的點不畫")
    p.add_argument("--max-depth", type=float, default=80.0, help="相機深度大於此值（m）的點不畫")
    p.add_argument("--color-max", type=float, default=30.0, help="深度上色的上限（m），近藍遠紅")
    p.add_argument("--min-range", type=float, default=0.5, help="離 LiDAR 小於此距離（m）的點不畫")
    p.add_argument("--radius", type=int, default=1, help="點的半徑（像素）")
    p.add_argument("--dim", type=float, default=0.6, help="背景影像亮度倍率")
    p.add_argument("--edges", action="store_true", help="背景改用 Canny 邊緣圖，方便看點雲邊界是否對齊")
    p.add_argument("--bbox", action="store_true", help="同時疊上 3D bbox 標註（只畫 person / bicycle / car）")
    p.add_argument("--show", action="store_true", help="用視窗顯示（任意鍵下一張，q 離開）")
    p.add_argument("--outdir", type=Path, help="預設 outputs/vis_lidar2img（加 --yolo 時 outputs/vis_lidar2img_yolo）")
    g = p.add_argument_group("YOLO 2D 偵測框")
    g.add_argument("--yolo", action="store_true", help="對 RGB 原圖跑 YOLO，把偵測框畫在點雲投影圖上，並統計推論時間")
    g.add_argument("--weights", type=Path, default=paths["outputs"].parent / "models" / "pretrained" / "yolo11s.pt")
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--conf", type=float, default=0.25, help="信心門檻（推論與畫圖都用這個值）")
    g.add_argument("--device", default="0")
    g.add_argument("--half", action="store_true", help="FP16 推論")
    args = p.parse_args()
    if args.outdir is None:
        args.outdir = paths["outputs"] / ("vis_lidar2img_yolo" if args.yolo else "vis_lidar2img")

    # 組出要畫的 (seq, frame) 清單
    if args.list:
        with open(args.list, encoding="utf-8") as f:
            items = [(int(r["seq"]), int(r["frame"])) for r in csv.DictReader(f)]
    else:
        if args.all_seqs:
            seq_ids = sorted(int(d.name) for d in (args.root / "2d_rect" / args.cam).iterdir() if d.is_dir())
        elif args.seq:
            seq_ids = args.seq
        else:
            p.error("需要 --seq、--all-seqs 或 --list")
        items = []
        for s in seq_ids:
            seq = CodaSequence(args.root, s, args.cam)
            frames = [f for f in seq.lidar_frames() if seq.image_path(f).exists()]
            if not frames:
                print(f"seq {s} 找不到同時有點雲與 {args.cam} 影像的 frame，略過")
                continue
            if args.frame:
                missing = sorted(set(args.frame) - set(frames))
                if missing:
                    sys.exit(f"seq {s} 這些 frame 沒有點雲或影像：{missing}；可用的前幾個：{frames[:10]}")
                frames = args.frame
            else:
                frames = frames[::args.stride]
                if not args.all:
                    frames = frames[:args.num]
            print(f"seq {s}: 點雲來源 {seq.lidar_dir()[1]}/os1，共 {len(frames)} 幀")
            items += [(s, f) for f in frames]

    yolo = YoloDetector(args.weights, args.imgsz, args.conf, args.device, args.half) if args.yolo else None
    det_rows = []
    calibs = {}
    t_start = time.perf_counter()
    for i, (s, frame) in enumerate(items):
        seq = CodaSequence(args.root, s, args.cam)
        if s not in calibs:
            calibs[s] = seq.read_calib()
        img, n, preds = draw_frame(seq, frame, calibs[s], args, yolo)
        if yolo:
            det_rows += detection_rows(s, frame, preds)
        outdir = args.outdir / args.cam / str(s)
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"{frame}.jpg"
        cv2.imencode(".jpg", img)[1].tofile(out)  # tofile 支援中文路徑
        extra = f"  yolo {yolo.records[-1][2]:.1f} ms, {yolo.records[-1][6]} det" if yolo else ""
        print(f"[{i + 1}/{len(items)}] seq {s} frame {frame}: {n} pts{extra} -> {out}")
        if args.show:
            cv2.imshow("CODa LiDAR -> image", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    total_s = time.perf_counter() - t_start
    cv2.destroyAllWindows()
    print(f"共畫出 {len(items)} 個 frame，總耗時 {total_s:.1f} s -> {args.outdir / args.cam}")

    if yolo and yolo.records:
        summary = yolo.summary(total_s)
        with open(args.outdir / "timing.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        with open(args.outdir / "detections.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(DET_FIELDS)
            w.writerows(det_rows)
        st = summary["stats_excluding_warmup"]
        print(f"\nYOLO {summary['num_frames']} 張，GPU {summary['gpu']}，imgsz {args.imgsz}，half {args.half}")
        print(f"  warmup 第一張 {summary['warmup_frame_ms']:.0f} ms；YOLO 累計 {summary['yolo_total_s']:.1f} s")
        for k in ["wall_ms", "preprocess_ms", "inference_ms", "postprocess_ms"]:
            print(f"  {k:15s} mean {st[k]['mean']:6.2f}  median {st[k]['median']:6.2f}  p95 {st[k]['p95']:6.2f}")
        print(f"  ≈ {summary['fps_wall']:.1f} FPS（僅 YOLO predict），峰值 GPU 記憶體 {summary['max_gpu_mem_MB']:.0f} MB")
        print(f"  -> {args.outdir / 'timing.json'}, {args.outdir / 'detections.csv'}")


if __name__ == "__main__":
    main()
