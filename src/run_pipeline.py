"""距離估計主流程：對每個 frame 跑 YOLO（或讀現有偵測結果），從框內點雲選出物件點並估計距離。

frame 來源（擇一）：
  --list CSV（需有 seq, frame 欄位，預設為 paths.yaml 的 kept_frames）
  --seq S [S ...] / --all-seqs：該序列所有同時有點雲與影像的 frame

範例（在 src/ 下執行）：
  python run_pipeline.py                                   # kept_frames.csv 全部，跑 YOLO
  python run_pipeline.py --seq 0 --num 20
  python run_pipeline.py --warmup 5                        # profiling 統計排除前 5 個 frame
輸出（--outdir，預設 outputs/distance）：
  distances.csv   每個偵測框一列：狀態、各階段點數、距離（dist_m 為相機座標歐氏距離中位數）
  detections.csv  YOLO 原始結果（只在跑 YOLO 時）
  frames.csv      這次處理的所有 frame（含沒有偵測框的），給 run_eval.py 算 FN
  profiling.json  各 stage（io / yolo / prepare / cluster）的 wall time 總和、CPU time 總和、
                  VRAM peak（各 frame 峰值的最大值），排除前 --warmup 個 frame（預設 3）；
                  meta 欄位記錄實際排除幾個
  profiling.md    同 profiling.json 的統計，轉成 Markdown 表格（另附 weights、總耗時等執行資訊）
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
from q1.detection.yolo import DET_FIELDS, YoloDetector, detection_rows  # noqa: E402
from q1.profiler import StageTimer, compute_summary, write_summary_md  # noqa: E402
from q1.paths import load_config, load_paths  # noqa: E402
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


def _run(args, dist_cfg: DistanceConfig) -> None:
    """Pipeline 核心：list_frames → YOLO → 距離估計 → 寫輸出檔。"""
    items = list_frames(args)
    if not items:
        print("沒有要處理的 frame")
        return
    yolo = YoloDetector(args.weights, args.imgsz, args.conf, args.device, args.half,
                        iou=getattr(args, "iou_nms", 0.7))
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows, det_rows, prof_records, calibs = [], [], [], {}
    t_start = time.perf_counter()
    for i, (s, frame) in enumerate(items, 1):
        seq = CodaSequence(args.root, s, args.cam)
        if s not in calibs:
            calibs[s] = seq.read_calib()
        with StageTimer() as t_io:
            img    = seq.read_image(frame)
            points = seq.read_lidar(frame)
        with StageTimer() as t_yolo:
            preds = yolo(img, s, frame)
        det_rows += detection_rows(s, frame, preds)
        _, results, stage_times = estimate_frame(points, calibs[s], preds, dist_cfg, set(args.classes))
        prof_records.append({"seq": s, "frame": frame,
                             "io":      t_io.result,
                             "yolo":    t_yolo.result,
                             "prepare": stage_times["prepare"],
                             "cluster": stage_times["cluster"]})
        rows += [result_row(s, frame, j, pred, res) for j, (pred, res) in enumerate(results)]
        n_ok = sum(r["status"] == "valid" for _, r in results)
        print(f"[{i}/{len(items)}] seq {s} frame {frame}: {len(results)} dets, {n_ok} valid, "
              f"yolo {t_yolo.result['wall_ms']:.0f} ms")
    total_s = time.perf_counter() - t_start

    with open(args.outdir / "distances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(RESULT_FIELDS)
        w.writerows(rows)
    with open(args.outdir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frame"])
        w.writerows(items)
    profiling = compute_summary(prof_records, args.warmup)
    if profiling["meta"]["warmup_skipped"] < args.warmup:
        print(f"注意：frame 數（{len(prof_records)}）不超過 --warmup {args.warmup}，"
              "profiling 統計改用全部 frame（含 warmup）")
    with open(args.outdir / "profiling.json", "w", encoding="utf-8") as f:
        json.dump(profiling, f, ensure_ascii=False, indent=1)
    write_summary_md(profiling, args.outdir / "profiling.md",
                     {"weights": args.weights, "outdir": args.outdir,
                      "imgsz": args.imgsz, "conf": args.conf,
                      "total_time_s": f"{total_s:.1f}"})
    with open(args.outdir / "detections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(DET_FIELDS)
        w.writerows(det_rows)

    status = [r[RESULT_FIELDS.index("status")] for r in rows]
    print(f"\n{len(items)} 個 frame、{len(rows)} 個偵測框：" +
          "、".join(f"{k} {status.count(k)}" for k in sorted(set(status))))
    smry = profiling
    print("wall_s total — " + "  ".join(f"{k}:{smry[k]['wall_s']:.1f}"
                                        for k in ["io", "yolo", "prepare", "cluster"]) +
          f"  loop:{total_s:.1f}s -> {args.outdir}")
    print(f"（profiling 統計排除前 {smry['meta']['warmup_skipped']} 個 warmup frame，"
          f"使用 {smry['meta']['frames_used']} 個）")

def main():
    # 若帶 --precision，從 paths.yaml + pipeline.yaml（+ quantization.yaml 的 export）讀取所有參數
    if "--precision" in sys.argv:
        import argparse as _ap
        _p = _ap.ArgumentParser(add_help=False)
        _p.add_argument("--precision", choices=["fp32", "fp16", "int8", "fp8"])
        _ns, _ = _p.parse_known_args()
        if _ns.precision:
            run_from_config(_ns.precision)
            return

    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"])
    p.add_argument("--cam", default="cam0", choices=["cam0", "cam1"])
    p.add_argument("--list", type=Path, default=paths["kept_frames"])
    p.add_argument("--seq", type=int, nargs="+")
    p.add_argument("--all-seqs", action="store_true")
    p.add_argument("--num", type=int, help="最多處理幾個 frame（除錯用）")
    p.add_argument("--classes", nargs="+", default=TARGET_CLASSES, help="要估距離的 COCO 類別")
    p.add_argument("--outdir", type=Path, default=paths["outputs"] / "distance")
    p.add_argument("--warmup", type=int, default=3,
                   help="profiling.json 統計時排除前幾個 frame（仍會正常輸出偵測與距離結果）")
    g = p.add_argument_group("2D 偵測")
    g.add_argument("--weights", type=Path, default=paths["weights"])
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--conf", type=float, default=0.25)
    g.add_argument("--device", default="0")
    g.add_argument("--half", action="store_true")
    g = p.add_argument_group("距離估計參數（見 q1/pipeline.py DistanceConfig）")
    for f in fields(DistanceConfig):
        g.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default),
                       default=f.default)
    args = p.parse_args()
    dist_cfg = DistanceConfig(**{f.name: getattr(args, f.name) for f in fields(DistanceConfig)})
    _run(args, dist_cfg)


def run_from_config(precision: str) -> None:
    """執行指定精度 TRT engine 的 pipeline。

    paths.yaml：資料與輸出路徑；pipeline.yaml：cam、conf、iou_nms、num_frames、warmup、distance；
    quantization.yaml 的 export：engine 的 imgsz 與 device（導出時就固定了，所以以量化設定為準）。
    """
    from dataclasses import fields as dc_fields

    paths   = load_paths()
    p_cfg   = load_config("pipeline")
    exp_cfg = load_config("quantization")["export"]
    d_cfg   = p_cfg.get("distance") or {}

    args = argparse.Namespace(
        root      = paths["coda_root"],
        cam       = p_cfg["cam"],
        list      = paths["kept_frames"],
        seq       = None,
        all_seqs  = False,
        num       = p_cfg.get("num_frames"),
        classes   = TARGET_CLASSES,
        outdir    = paths["outputs"] / f"distance_{precision}",
        weights   = paths["outputs"] / "engines" / f"{precision}.engine",
        imgsz     = exp_cfg["imgsz"],
        conf      = p_cfg["conf"],
        iou_nms   = p_cfg["iou_nms"],
        device    = exp_cfg["device"],
        half      = False,
        warmup    = p_cfg.get("warmup", 3),
    )

    dc_defaults = {f.name: f.default for f in dc_fields(DistanceConfig)}
    dist_cfg = DistanceConfig(**{k: d_cfg.get(k, v) for k, v in dc_defaults.items()})
    _run(args, dist_cfg)


if __name__ == "__main__":
    main()
