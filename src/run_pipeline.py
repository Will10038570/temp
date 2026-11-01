"""距離估計主流程：對每個 frame 跑 YOLO（或讀現有偵測結果），從框內點雲選出物件點並估計距離。

frame 來源（擇一）：
  --list CSV（需有 seq, frame 欄位，預設為 {coda_root}/kept_frames.csv）
  --seq S [S ...] / --all-seqs：該序列所有同時有點雲與影像的 frame
偵測框來源：
  預設跑 YOLO；給 --dets 時改讀現有的 detections.csv（不重跑 YOLO）

範例（在 src/ 下執行）：
  python run_pipeline.py                                   # kept_frames.csv 全部，跑 YOLO
  python run_pipeline.py --dets ../outputs/vis_lidar2img_yolo/detections.csv
  python run_pipeline.py --seq 0 --num 20
輸出（--outdir，預設 outputs/distance）：
  distances.csv   每個偵測框一列：狀態、各階段點數、距離（dist_m 為相機座標歐氏距離中位數）
  detections.csv  YOLO 原始結果（只在跑 YOLO 時）
  frames.csv      這次處理的所有 frame（含沒有偵測框的），給 run_eval.py 算 FN
  timing.json     YOLO 與距離估計的耗時、資料根目錄、相機、DistanceConfig
接著可以跑 run_eval.py 和 CODa 3D bbox GT 比較。
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q1.data.coda import CodaSequence  # noqa: E402
from q1.detection.yolo import DET_FIELDS, YoloDetector, detection_rows, load_detections  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from q1.pipeline import RESULT_FIELDS, TARGET_CLASSES, DistanceConfig, estimate_frame, result_row  # noqa: E402


def list_frames(args):
    if args.seq or args.all_seqs:
        seq_ids = args.seq or sorted(int(d.name) for d in (args.root / "2d_rect" / args.cam).iterdir() if d.is_dir())
        items = []
        for s in seq_ids:
            seq = CodaSequence(args.root, s, args.cam)
            items += [(s, f) for f in seq.lidar_frames() if seq.image_path(f).exists()]
    else:
        with open(args.list, encoding="utf-8") as f:
            items = [(int(r["seq"]), int(r["frame"])) for r in csv.DictReader(f)]
    return items[:args.num] if args.num else items


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"])
    p.add_argument("--cam", default="cam0", choices=["cam0", "cam1"])
    p.add_argument("--list", type=Path, default=paths["coda_root"] / "kept_frames.csv")
    p.add_argument("--seq", type=int, nargs="+")
    p.add_argument("--all-seqs", action="store_true")
    p.add_argument("--num", type=int, help="最多處理幾個 frame（除錯用）")
    p.add_argument("--classes", nargs="+", default=TARGET_CLASSES, help="要估距離的 COCO 類別")
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "distance")
    g = p.add_argument_group("2D 偵測")
    g.add_argument("--dets", type=Path, help="讀現有的 detections.csv，不跑 YOLO")
    g.add_argument("--weights", type=Path, default=paths["outputs"].parent / "models" / "pretrained" / "yolo11s.pt")
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--conf", type=float, default=0.25)
    g.add_argument("--device", default="0")
    g.add_argument("--half", action="store_true")
    g = p.add_argument_group("距離估計參數（見 q1/pipeline.py DistanceConfig）")
    for f in fields(DistanceConfig):
        g.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default),
                       default=f.default)
    args = p.parse_args()
    cfg = DistanceConfig(**{f.name: getattr(args, f.name) for f in fields(DistanceConfig)})

    items = list_frames(args)
    if not items:
        sys.exit("沒有要處理的 frame")
    if args.dets:
        dets, yolo = load_detections(args.dets, conf=args.conf), None
    else:
        dets, yolo = None, YoloDetector(args.weights, args.imgsz, args.conf, args.device, args.half)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows, det_rows, dist_ms, calibs = [], [], [], {}
    t_start = time.perf_counter()
    for i, (s, frame) in enumerate(items, 1):
        seq = CodaSequence(args.root, s, args.cam)
        if s not in calibs:
            calibs[s] = seq.read_calib()
        if yolo:
            preds = yolo(seq.read_image(frame), s, frame)
            det_rows += detection_rows(s, frame, preds)
        else:
            preds = dets.get((s, frame), [])
        points = seq.read_lidar(frame)
        t0 = time.perf_counter()
        _, results = estimate_frame(points, calibs[s], preds, cfg, set(args.classes))
        dist_ms.append((time.perf_counter() - t0) * 1000)
        rows += [result_row(s, frame, j, pred, res) for j, (pred, res) in enumerate(results)]
        n_ok = sum(r["status"] == "ok" for _, r in results)
        print(f"[{i}/{len(items)}] seq {s} frame {frame}: {len(results)} dets, {n_ok} ok, {dist_ms[-1]:.0f} ms")
    total_s = time.perf_counter() - t_start

    with open(args.outdir / "distances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(RESULT_FIELDS)
        w.writerows(rows)
    with open(args.outdir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frame"])
        w.writerows(items)
    timing = {"num_frames": len(items), "total_s": total_s, "root": str(args.root), "cam": args.cam,
              "config": asdict(cfg),
              "distance_ms": {"mean": float(np.mean(dist_ms)), "median": float(np.median(dist_ms)),
                              "p95": float(np.percentile(dist_ms, 95))}}
    if yolo:
        timing["yolo"] = yolo.summary(total_s)
        with open(args.outdir / "detections.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(DET_FIELDS)
            w.writerows(det_rows)
    with open(args.outdir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=1)

    status = [r[RESULT_FIELDS.index("status")] for r in rows]
    print(f"\n{len(items)} 個 frame、{len(rows)} 個偵測框：" +
          "、".join(f"{k} {status.count(k)}" for k in sorted(set(status))))
    print(f"距離估計每幀 mean {timing['distance_ms']['mean']:.1f} ms，總耗時 {total_s:.1f} s -> {args.outdir}")


if __name__ == "__main__":
    main()
