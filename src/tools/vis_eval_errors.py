"""把 run_eval.py 的配對結果畫在影像上，看 FP / FN 為什麼多。

每張圖畫：
  TP  綠色粗框 = YOLO 框，綠色細框 = 配對的 GT 投影框，標 IoU
  FP  紅框，標類別、score、和同類 GT 的最大 IoU；若和非目標類別的 CODa 標註重疊，標出那個類別
  FN  藍框（GT 投影框），標類別、最大 IoU、遮擋程度、距離；GT 的 8 個投影角點畫成小點
  best IoU 在 [0.3, 門檻) 的 FP/FN 加註 "near"：框到了但對不齊，門檻 0.3 時就會變 TP

挑圖：FP 最多、FN 最多、三種都有（隨機）各約 1/3，每個 sequence 限量以分散場景。

範例（在 src/ 下執行）：
  python tools/vis_eval_errors.py
  python tools/vis_eval_errors.py --n 100 --iou 0.5 --outdir ../outputs/vis_eval
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.data.classes import CODA_TO_TARGET, gt_boxes  # noqa: E402
from q1.data.coda import CodaSequence  # noqa: E402
from q1.data.projection import project_bbox  # noqa: E402
from q1.distance.cluster import box_iou  # noqa: E402
from q1.evaluation import greedy_match, project_gt  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from q1.pipeline import DistanceConfig  # noqa: E402

GREEN, RED, BLUE, GRAY = (0, 200, 0), (0, 0, 255), (255, 120, 0), (160, 160, 160)


def label(img, text, x, y, color, scale=0.45):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(min(max(x, 0), img.shape[1] - tw - 4))
    y = int(min(max(y, th + 4), img.shape[0] - 2))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def rect(img, b, color, t):
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, t)


def pick_frames(eval_dir, n, max_per_seq, seed):
    cnt = defaultdict(Counter)
    with open(eval_dir / "matches.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cnt[(int(r["seq"]), int(r["frame"]))]["TP"] += 1
    with open(eval_dir / "unmatched.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cnt[(int(r["seq"]), int(r["frame"]))][r["type"]] += 1
    keys = list(cnt)
    rng = random.Random(seed)
    rng.shuffle(keys)  # 同分時隨機
    chosen, per_seq = [], Counter()

    def take(order, k, tag):
        got = 0
        for key in order:
            if got >= k:
                break
            if key in {c[0] for c in chosen} or per_seq[(tag, key[0])] >= max_per_seq:
                continue
            chosen.append((key, tag))
            per_seq[(tag, key[0])] += 1
            got += 1

    k = n // 3
    take(sorted(keys, key=lambda x: -cnt[x]["FP"]), k, "fp_heavy")
    take(sorted(keys, key=lambda x: -cnt[x]["FN"]), k, "fn_heavy")
    take([x for x in keys if all(cnt[x][t] for t in ("TP", "FP", "FN"))], n - 2 * k, "mixed")
    return chosen, cnt


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dist-dir", type=Path, default=paths["outputs"] / "distance")
    p.add_argument("--eval-dir", type=Path, default=paths["outputs"] / "eval")
    p.add_argument("--root", type=Path, default=paths["coda_root"], help="CODa 根目錄")
    p.add_argument("--cam", default="cam0")
    p.add_argument("--min-depth", type=float, default=DistanceConfig().min_depth)
    p.add_argument("--iou", type=float, default=None, help="預設用 eval-dir/summary.json 的門檻")
    p.add_argument("--near", type=float, default=0.3, help="best IoU >= 此值的 FP/FN 標 near")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-per-seq", type=int, default=4, help="每類挑圖時每個 sequence 最多幾張")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "vis_eval")
    args = p.parse_args()

    with open(args.eval_dir / "summary.json", encoding="utf-8") as f:
        iou_thr = args.iou if args.iou is not None else json.load(f)["iou_thr"]
    root, cam, min_depth = args.root, args.cam, args.min_depth

    preds_by = defaultdict(list)
    with open(args.dist_dir / "distances.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            preds_by[(int(r["seq"]), int(r["frame"]))].append(r)

    chosen, cnt = pick_frames(args.eval_dir, args.n, args.max_per_seq, args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    seqs, calibs = {}, {}
    index = []
    for i, ((s, frame), tag) in enumerate(chosen, 1):
        if s not in seqs:
            seqs[s] = CodaSequence(root, s, cam)
            calibs[s] = seqs[s].read_calib()
        seq, calib = seqs[s], calibs[s]
        img = seq.read_image(frame)
        raw = seq.read_bboxes(frame)
        gts = project_gt(gt_boxes(raw), calib, min_depth)
        others = []  # 非目標類別的 CODa 標註（用來解釋 FP 框到了什麼）
        for b in raw:
            if b["classId"] not in CODA_TO_TARGET:
                pr = project_bbox(b, calib, min_depth)
                if pr is not None:
                    others.append((b["classId"], pr[1]))
        preds = sorted(preds_by.get((s, frame), []), key=lambda r: int(r["det"]))
        pb = [[float(r[k]) for k in ("x1", "y1", "x2", "y2")] for r in preds]
        matches, fp_idx, fn_idx, iou = greedy_match(pb, [float(r["score"]) for r in preds],
                                                    [g["bbox2d"] for g in gts], iou_thr,
                                                    pred_classes=[r["class"] for r in preds],
                                                    gt_classes=[g["box"]["classId"] for g in gts])
        same = np.array([[1.0 if r["class"] == g["box"]["classId"] else 0.0 for g in gts] for r in preds]
                        ).reshape(len(preds), len(gts))
        iou_same = np.where(same > 0, iou, 0.0)

        for gi in fn_idx:  # FN 先畫，放最底層
            g = gts[gi]
            b, box = g["bbox2d"], g["box"]
            uv, _ = project_bbox(box, calib, min_depth)
            for q in uv:
                cv2.circle(img, (int(q[0]), int(q[1])), 2, BLUE, -1)
            rect(img, b, BLUE, 2)
            best = float(iou_same[:, gi].max()) if len(preds) else 0.0
            occ = box.get("labelAttributes", {}).get("isOccluded", "?")
            d = float(np.hypot(box["cX"], box["cY"]))
            near = " near" if best >= args.near else ""
            label(img, f"FN {box['classId']} iou{best:.2f} occ:{occ} {d:.0f}m{near}", b[0], b[3] + 14, BLUE)
        for pi, gi, v in matches:
            rect(img, gts[gi]["bbox2d"], GREEN, 1)
            rect(img, pb[pi], GREEN, 2)
            label(img, f"TP {preds[pi]['class']} iou{v:.2f}", pb[pi][0], pb[pi][1] - 2, GREEN)
        for pi in fp_idx:
            r, b = preds[pi], pb[pi]
            rect(img, b, RED, 2)
            best = float(iou_same[pi].max()) if len(gts) else 0.0
            txt = f"FP {r['class']} {float(r['score']):.2f} iou{best:.2f}"
            if best >= args.near:
                txt += " near"
            o = max(((box_iou(b, ob), c) for c, ob in others), default=(0.0, ""))
            if o[0] >= 0.3:
                txt += f" ~{o[1]}"
            label(img, txt, b[0], b[1] - 2, RED)

        c = cnt[(s, frame)]
        head = f"[{tag}] seq {s} frame {frame}  IoU>={iou_thr}  TP {c['TP']}  FP {c['FP']}  FN {c['FN']}"
        (tw, th), _ = cv2.getTextSize(head, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
        cv2.putText(img, head, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        name = f"{i:03d}_{tag}_s{s}_f{frame}.jpg"
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(args.outdir / name))
        index.append([name, tag, s, frame, c["TP"], c["FP"], c["FN"]])
        print(f"[{i}/{len(chosen)}] {name}")

    with open(args.outdir / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "tag", "seq", "frame", "TP", "FP", "FN"])
        w.writerows(index)
    print(f"-> {args.outdir}")


if __name__ == "__main__":
    main()
