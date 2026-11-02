"""距離估計的量化驗證，接在 run_pipeline.py 之後執行：讀它的輸出，和 CODa 3D bbox GT 比較。

  1. 配對：GT 3D bbox 8 個角點投影到影像取外接矩形，和 YOLO 框做依類別的 greedy 配對（同類別 IoU >= --iou）。
     類別不符或 IoU 不足的 YOLO 框記為 FP、GT 記為 FN，只記錄編號與數量，不算距離。
  2. 距離：配對成功的，GT 距離 = FramePoints 中落在 GT 3D box 內的點的相機座標歐氏距離中位數，
     和 pipeline 的 dist_m（選中 cluster 點的中位數）比較。err_m = dist_m − gt_dist_m（有正負號）。
     FramePoints 用 timing.json 記錄的 DistanceConfig 重建，和 pipeline 用的是同一份點（去近點、去地面、只留影像內）。

範例（在 src/ 下執行）：
  python run_pipeline.py
  python run_eval.py
  python run_eval.py --dist-dir ../outputs/distance_trt_fp16 --outdir ../outputs/eval_trt_fp16 --iou 0.3
輸出（--outdir，預設 outputs/eval）：
  matches.csv    每個配對一列；eval 欄位為 ok 才有算距離誤差
  unmatched.csv  FP（YOLO 框）與 FN（GT）各一列，含它和對方最大的 IoU（方便判斷是不是門檻太嚴）
  summary.json   TP / FP / FN 數量、類別正確率、距離誤差統計（全部與各類別）
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from q1.data.classes import gt_boxes  # noqa: E402
from q1.data.coda import CodaSequence  # noqa: E402
from q1.distance.points import prepare_points  # noqa: E402
from q1.evaluation import error_stats, greedy_match, gt_distance, project_gt  # noqa: E402
from q1.paths import load_paths  # noqa: E402
from q1.pipeline import DistanceConfig  # noqa: E402

MATCH_FIELDS = ["seq", "frame", "det", "gt_id", "iou", "pred_class", "gt_class", "class_match", "score",
                "status", "n_final", "gt_n_points", "dist_m", "gt_dist_m", "err_m", "abs_err_m", "eval"]
UNMATCHED_FIELDS = ["seq", "frame", "type", "id", "class", "score", "best_iou", "x1", "y1", "x2", "y2"]


def load_distances(path):
    """distances.csv → {(seq, frame): [row, ...]}，依 det 編號排序。"""
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(int(r["seq"]), int(r["frame"]))].append(r)
    for rows in out.values():
        rows.sort(key=lambda r: int(r["det"]))
    return out


def load_frames(dist_dir, fallback):
    path = dist_dir / "frames.csv"
    if not path.exists():
        print(f"注意：{path} 不存在（舊版 run_pipeline 的輸出），改用 {fallback} 當作 frame 清單；"
              "若 pipeline 當時只跑了部分 frame，FN 會多算。重跑 run_pipeline.py 即可解決。")
        path = fallback
    with open(path, encoding="utf-8") as f:
        return [(int(r["seq"]), int(r["frame"])) for r in csv.DictReader(f)]


def r3(v):
    return round(v, 3) if np.isfinite(v) else ""


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dist-dir", type=Path, default=paths["outputs"] / "distance",
                   help="run_pipeline.py 的輸出資料夾（distances.csv、timing.json、frames.csv）")
    p.add_argument("--root", type=Path, help="CODa 根目錄；預設用 timing.json 記錄的，沒有時用 paths.yaml")
    p.add_argument("--cam", help="預設用 timing.json 記錄的，沒有時用 cam0")
    p.add_argument("--list", type=Path, default=paths["coda_root"] / "kept_frames.csv",
                   help="--dist-dir 沒有 frames.csv 時改用這份 frame 清單")
    p.add_argument("--iou", type=float, default=0.5, help="配對的 IoU 門檻")
    p.add_argument("--min-gt-points", type=int, default=5, help="GT box 內的點少於此值時不算距離誤差")
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "eval")
    args = p.parse_args()

    with open(args.dist_dir / "timing.json", encoding="utf-8") as f:
        timing = json.load(f)
    cfg = DistanceConfig(**timing["config"])
    root = args.root or Path(timing.get("root", paths["coda_root"]))
    cam = args.cam or timing.get("cam", "cam0")
    dists = load_distances(args.dist_dir / "distances.csv")
    frames = load_frames(args.dist_dir, args.list)

    match_rows, unmatched_rows = [], []
    n_pred = n_gt = 0
    no_anno = []
    seqs, calibs = {}, {}
    for i, (s, frame) in enumerate(frames, 1):
        if s not in seqs:
            seqs[s] = CodaSequence(root, s, cam)
            calibs[s] = seqs[s].read_calib()
        seq, calib = seqs[s], calibs[s]
        if not seq.bbox_path(frame).exists():
            no_anno.append((s, frame))  # 沒有標註的 frame 無法判斷對錯，整幀跳過
            continue
        preds = dists.get((s, frame), [])
        pred_boxes = [[float(r[k]) for k in ("x1", "y1", "x2", "y2")] for r in preds]
        gts = project_gt(gt_boxes(seq.read_bboxes(frame)), calib, cfg.min_depth)
        n_pred += len(preds)
        n_gt += len(gts)
        matches, fp_idx, fn_idx, iou = greedy_match(pred_boxes, [float(r["score"]) for r in preds],
                                                    [g["bbox2d"] for g in gts], args.iou,
                                                    pred_classes=[r["class"] for r in preds],
                                                    gt_classes=[g["box"]["classId"] for g in gts])

        fp = prepare_points(seq.read_lidar(frame), calib, **cfg.point_kwargs()) if matches else None
        for pi, gi, v in matches:
            r, g = preds[pi], gts[gi]["box"]
            gt_n, gt_d = gt_distance(fp, gts[gi]["corners"])
            d = float(r["dist_m"]) if r["dist_m"] not in ("", "nan") else float("nan")
            if r["status"] != "valid":
                ev = f"pred_{r['status']}"
            elif gt_n < args.min_gt_points:
                ev = "gt_few_points"
            else:
                ev = "ok"
            err = d - gt_d if ev == "ok" else float("nan")
            match_rows.append([s, frame, int(r["det"]), g["instanceId"], round(v, 3), r["class"], g["classId"],
                               int(r["class"] == g["classId"]), r["score"], r["status"], r["n_final"], gt_n,
                               r3(d), r3(gt_d), r3(err), r3(abs(err)), ev])
        for pi in fp_idx:
            r = preds[pi]
            best = float(iou[pi].max()) if iou.shape[1] else 0.0
            unmatched_rows.append([s, frame, "FP", int(r["det"]), r["class"], r["score"], round(best, 3),
                                   *pred_boxes[pi]])
        for gi in fn_idx:
            g = gts[gi]
            best = float(iou[:, gi].max()) if iou.shape[0] else 0.0
            unmatched_rows.append([s, frame, "FN", g["box"]["instanceId"], g["box"]["classId"], "", round(best, 3),
                                   *(round(float(v), 1) for v in g["bbox2d"])])
        if i % 100 == 0 or i == len(frames):
            print(f"[{i}/{len(frames)}] TP {len(match_rows)}  FP/FN {len(unmatched_rows)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, fields_, rows in (("matches.csv", MATCH_FIELDS, match_rows),
                                ("unmatched.csv", UNMATCHED_FIELDS, unmatched_rows)):
        with open(args.outdir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(fields_)
            w.writerows(rows)

    col = {k: i for i, k in enumerate(MATCH_FIELDS)}
    tp = len(match_rows)
    n_fp = sum(r[2] == "FP" for r in unmatched_rows)
    n_fn = sum(r[2] == "FN" for r in unmatched_rows)
    ok = [r for r in match_rows if r[col["eval"]] == "ok"]
    summary = {
        "dist_dir": str(args.dist_dir), "iou_thr": args.iou, "min_gt_points": args.min_gt_points,
        "num_frames": len(frames), "frames_without_annotation": len(no_anno),
        "num_pred": n_pred, "num_gt_in_image": n_gt, "TP": tp, "FP": n_fp, "FN": n_fn,
        "precision": tp / n_pred if n_pred else None, "recall": tp / n_gt if n_gt else None,
        "class_accuracy_on_TP": sum(r[col["class_match"]] for r in match_rows) / tp if tp else None,
        "FP_by_class": dict(Counter(r[4] for r in unmatched_rows if r[2] == "FP")),
        "FN_by_class": dict(Counter(r[4] for r in unmatched_rows if r[2] == "FN")),
        "eval_status_on_TP": dict(Counter(r[col["eval"]] for r in match_rows)),
        "distance_error": error_stats([r[col["err_m"]] for r in ok]),
        "distance_error_by_gt_class": {c: error_stats([r[col["err_m"]] for r in ok if r[col["gt_class"]] == c])
                                       for c in sorted({r[col["gt_class"]] for r in ok})},
    }
    with open(args.outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(f"\n{len(frames)} 個 frame（{len(no_anno)} 個沒有標註，已跳過）；IoU 門檻 {args.iou}")
    print(f"YOLO 框 {n_pred}、影像內 GT {n_gt} → TP {tp}、FP {n_fp}、FN {n_fn}")
    if tp:
        print(f"TP 類別正確率 {summary['class_accuracy_on_TP']:.3f}；TP 距離狀態 {summary['eval_status_on_TP']}")
    e = summary["distance_error"]
    if e["n"]:
        print(f"距離誤差（{e['n']} 個，估計 - GT）：平均 {e['mean_signed_m']:+.3f} m、中位數 {e['median_signed_m']:+.3f} m、"
              f"MAE {e['mae_m']:.3f} m、|誤差| 中位數 {e['median_abs_m']:.3f} m、P90 {e['p90_abs_m']:.3f} m")
        for c, st in summary["distance_error_by_gt_class"].items():
            print(f"  {c:<8s} n={st['n']:<5d} 平均 {st['mean_signed_m']:+.3f} m  MAE {st['mae_m']:.3f} m")
    print(f"-> {args.outdir}")


if __name__ == "__main__":
    main()
