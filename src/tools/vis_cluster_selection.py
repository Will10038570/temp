"""視覺化距離估計流程（q1/pipeline.py）在每個 YOLO 框內選出的物件點群，用來檢查選群規則。

讀現有的 YOLO 結果（detections.csv），不重跑 YOLO。圖例見 vis/draw.py draw_cluster_selection。
另存每個偵測框的放大裁切圖（--crops），小物件比較容易看清楚。

範例（在 src/ 下執行）：
  python tools/vis_cluster_selection.py --all                         # 所有 seq 的所有 frame
  python tools/vis_cluster_selection.py --num 30 --crops               # 隨機抽 30 個 frame
  python tools/vis_cluster_selection.py --seq 0 --frame 481 --show
  python tools/vis_cluster_selection.py --num 50 --classes person car --w-cov 0.3 --w-center 0.7
輸出：outputs/vis_cluster/{seq}/{frame}.jpg、clusters.csv（每個偵測框一列）、crops/（加 --crops 時）
"""
import argparse
import csv
import sys
from collections import Counter
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.data.coda import CodaSequence  # noqa: E402
from q1.detection.yolo import load_detections  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from q1.pipeline import RESULT_FIELDS, TARGET_CLASSES, DistanceConfig, estimate_frame, result_row  # noqa: E402
from vis.draw import WHITE, crop_box, draw_cluster_selection, put_text  # noqa: E402


def box_area(pred):
    x1, y1, x2, y2 = pred["bbox"]
    return (x2 - x1) * (y2 - y1)


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"])
    p.add_argument("--dets", type=Path, default=paths["outputs"] / "vis_lidar2img_yolo" / "detections.csv")
    p.add_argument("--cam", default="cam0")
    p.add_argument("--seq", type=int, help="只看這個序列")
    p.add_argument("--frame", type=int, nargs="*", help="指定 frame（需搭配 --seq）")
    p.add_argument("--num", type=int, default=30, help="隨機抽幾個 frame（不指定 --frame / --all 時）")
    p.add_argument("--all", action="store_true", help="畫出所有有偵測結果的 frame（可配合 --seq 只跑一個序列）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classes", nargs="+", default=TARGET_CLASSES)
    p.add_argument("--conf", type=float, default=0.25)
    g = p.add_argument_group("距離估計參數（見 q1/pipeline.py DistanceConfig）")
    for f in fields(DistanceConfig):
        g.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--dim", type=float, default=0.5, help="背景影像亮度倍率")
    p.add_argument("--crops", action="store_true", help="另存每個偵測框的放大裁切圖")
    p.add_argument("--show", action="store_true")
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "vis_cluster")
    args = p.parse_args()
    cfg = DistanceConfig(**{f.name: getattr(args, f.name) for f in fields(DistanceConfig)})

    dets = load_detections(args.dets, set(args.classes), args.conf)
    keys = sorted(k for k in dets if args.seq is None or k[0] == args.seq)
    if args.frame:
        keys = [k for k in keys if k[1] in set(args.frame)]
    elif not args.all:
        pick = np.random.default_rng(args.seed).choice(len(keys), min(args.num, len(keys)), replace=False)
        keys = [keys[i] for i in sorted(pick)]
    if not keys:
        sys.exit("沒有符合條件的 frame")

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.crops:
        (args.outdir / "crops").mkdir(exist_ok=True)
    rows, calibs = [], {}
    for n_done, (s, frame) in enumerate(keys, 1):
        seq = CodaSequence(args.root, s, args.cam)
        if s not in calibs:
            calibs[s] = seq.read_calib()
        fp, results = estimate_frame(seq.read_lidar(frame), calibs[s], dets[(s, frame)], cfg, set(args.classes))

        img = (seq.read_image(frame) * args.dim).astype(np.uint8)
        for pred, res in sorted(results, key=lambda t: -box_area(t[0])):  # 大框先畫，小物件不會被蓋住
            draw_cluster_selection(img, pred, res, fp.uv, fp.xyz_cam, args.radius)
        put_text(img, f"seq {s} frame {frame}  dets {len(results)}  shrink {cfg.shrink} eps>={cfg.eps_min}",
                 8, 24, WHITE, 0.6)
        out = args.outdir / str(s) / f"{frame}.jpg"
        out.parent.mkdir(exist_ok=True)
        cv2.imencode(".jpg", img)[1].tofile(out)  # tofile 支援中文路徑

        for j, (pred, res) in enumerate(results):
            rows.append(result_row(s, frame, j, pred, res))
            if args.crops:
                cv2.imencode(".jpg", crop_box(img, pred["bbox"]))[1].tofile(
                    args.outdir / "crops" / f"{s}_{frame}_{j}_{pred['category']}.jpg")
        n_ok = sum(r["status"] == "valid" for _, r in results)
        print(f"[{n_done}/{len(keys)}] seq {s} frame {frame}: {len(results)} dets, {n_ok} valid -> {out}")
        if args.show:
            cv2.imshow("cluster selection", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

    with open(args.outdir / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(RESULT_FIELDS)
        w.writerows(rows)
    si, ii = RESULT_FIELDS.index("status"), RESULT_FIELDS.index("iou")
    st = Counter(r[si] for r in rows)
    ious = np.array([r[ii] for r in rows if r[si] == "ok"], dtype=float)
    print(f"\n{len(rows)} 個偵測框：" + "、".join(f"{k} {v}" for k, v in sorted(st.items())))
    if len(ious):
        print(f"ok 的 IoU：median {np.median(ious):.2f}，< 0.3 的有 {(ious < 0.3).sum()} 個")
    print(f"-> {args.outdir}（clusters.csv）")


if __name__ == "__main__":
    main()
