"""把 CODa 的 3D bbox 標註投影到 RGB 影像上，畫出 8 個角點與 12 條邊。

範例（在 src/ 下執行）：
  python tools/vis_coda_bbox3d.py --seq 0                  # seq 0 前 10 個有標註的 frame
  python tools/vis_coda_bbox3d.py --seq 0 --frame 683 --show
  python tools/vis_coda_bbox3d.py --seq 0 --all
輸出：outputs/vis_bbox3d/{cam}/{seq}/{frame}.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.data.classes import TARGET_CLASSES, gt_boxes  # noqa: E402
from q1.data.coda import CodaSequence  # noqa: E402
from q1.data.projection import BOX_EDGES, project_bbox  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from vis.draw import class_color  # noqa: E402


def draw_frame(seq, frame, calib, args):
    img = seq.read_image(frame)
    drawn = 0
    for box in gt_boxes(seq.read_bboxes(frame)):  # CODa 類別歸類成 person / bicycle / car，其餘丟掉
        cls = box["classId"]
        occl = box.get("labelAttributes", {}).get("isOccluded", "?")
        if args.classes and cls not in args.classes:
            continue
        if args.hide_full_occluded and occl == "Full":
            continue
        res = project_bbox(box, calib)
        if res is None:
            continue
        corners = res[0]
        color = class_color(cls)
        pts = corners.round().astype(np.int32)
        for a, b in BOX_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(img, tuple(pt), 4, color, -1, cv2.LINE_AA)
        # 標籤放在最上方的角點旁，並裁到影像內
        merged = f"/{box['codaClass']}" if cls == "car" and box["codaClass"] != "Car" else ""  # 被歸成 car 的車種
        label = f"{cls}{merged} [{occl}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx, ly = pts[pts[:, 1].argmin()]
        lx = int(np.clip(lx, 0, img.shape[1] - tw - 4))
        ly = int(np.clip(ly - 6, th + 4, img.shape[0] - 1))
        cv2.rectangle(img, (lx, ly - th - 4), (lx + tw + 4, ly), color, -1)
        cv2.putText(img, label, (lx + 2, ly - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        drawn += 1
    title = f"seq {seq.seq} frame {frame}  boxes {drawn}"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(img, title, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img, drawn


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"], help="CODa 根目錄")
    p.add_argument("--seq", type=int, required=True)
    p.add_argument("--cam", default="cam0", choices=["cam0", "cam1"])
    p.add_argument("--frame", type=int, nargs="*", help="指定 frame；不指定時取前 --num 個有標註的 frame")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--all", action="store_true", help="畫出這個 sequence 所有有標註的 frame")
    p.add_argument("--classes", nargs="*", default=TARGET_CLASSES, choices=TARGET_CLASSES, help="只畫這些類別（歸類後名稱）")
    p.add_argument("--hide-full-occluded", action="store_true", help="不畫 isOccluded=Full 的物件")
    p.add_argument("--show", action="store_true", help="用視窗顯示（任意鍵下一張，q 離開）")
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "vis_bbox3d")
    args = p.parse_args()

    seq = CodaSequence(args.root, args.seq, args.cam)
    calib = seq.read_calib()
    frames = seq.annotated_frames()
    if not frames:
        sys.exit(f"seq {args.seq} 在 {args.root} 找不到有標註且有影像的 frame")
    if args.frame:
        missing = sorted(set(args.frame) - set(frames))
        if missing:
            sys.exit(f"這些 frame 沒有標註或影像：{missing}；可用的前幾個：{frames[:10]}")
        frames = args.frame
    elif not args.all:
        frames = frames[:args.num]

    outdir = args.outdir / args.cam / str(args.seq)
    outdir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img, drawn = draw_frame(seq, frame, calib, args)
        out = outdir / f"{frame}.jpg"
        cv2.imencode(".jpg", img)[1].tofile(out)  # tofile 支援中文路徑
        print(f"frame {frame}: {drawn} boxes -> {out}")
        if args.show:
            cv2.imshow("CODa 3D bbox", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
