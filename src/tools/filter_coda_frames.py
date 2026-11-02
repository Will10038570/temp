"""只保留含有 COCO 對應類別的 CODa frame，其餘 frame 的檔案全部刪除。

保留條件（全部符合才保留）：
  1. RGB 影像（2d_rect/cam0）、LiDAR 點雲（3d_comp/os1）、3D bbox 標註（3d_bbox/os1）三者都存在
  2. bbox 標註中至少有一個物件屬於 --classes（預設 KEEP_CLASSES，即 Pedestrian / Bike / Car），
     且指定 --occlusion 時，該物件的 isOccluded 也要符合
  3. 指定 --max-yaw-rate 時，機器人的 |yaw rate|（°/s，由 poses/dense 計算）不超過此值。
     轉彎時 LiDAR 與相機的取樣時間差會讓點雲投影到影像上產生水平偏移（7°/s 約偏 15 px）
刪除範圍是所有逐 frame 的資料夾（2d_*、3d_*），calibrations / poses / timestamps / metadata 不動。

範例（在 src/ 下執行）：
  python tools/filter_coda_frames.py            # 只統計，不刪除
  python tools/filter_coda_frames.py --apply    # 實際刪除
  python tools/filter_coda_frames.py --classes Pedestrian Bike Car --occlusion None
  python tools/filter_coda_frames.py --max-yaw-rate 3 --apply   # 另外刪除轉彎中的 frame
輸出：{root}/kept_frames.csv（seq, frame, 該 frame 含有的保留類別）
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.data.classes import KEEP_CLASSES  # noqa: E402
from q1.data.coda import yaw_rates  # noqa: E402
from q1.paths import load_paths  # noqa: E402

# 檔名結尾的 _{seq}_{frame}.{ext}
FRAME_RE = re.compile(r"_(\d+)_(\d+)\.\w+$")


def frames_to_keep(root, classes, occlusion=None, max_yaw_rate=None):
    """含有保留類別（且遮擋等級符合 occlusion），且 RGB（cam0）、LiDAR、bbox 三者都齊全的 frame。
    指定 max_yaw_rate 時，|yaw rate| 超過的 frame 也不保留；pose 無效、算不出 yaw rate 的 frame 保留。
    回傳 (keep, 因轉彎排除的 frame 數, 算不出 yaw rate 而保留的 frame 數)。
    """
    keep, yaw_cache, turning, unknown = {}, {}, 0, 0
    for p in (root / "3d_bbox" / "os1").glob("*/*.json"):
        seq, frame = FRAME_RE.search(p.name).groups()
        image = root / "2d_rect" / "cam0" / seq / f"2d_rect_cam0_{seq}_{frame}.png"
        lidar = root / "3d_comp" / "os1" / seq / f"3d_comp_os1_{seq}_{frame}.bin"
        if not (image.is_file() and lidar.is_file()):
            continue
        with open(p, encoding="utf-8") as f:
            found = {b["classId"] for b in json.load(f)["3dbbox"]
                     if b["classId"] in classes
                     and (occlusion is None or b.get("labelAttributes", {}).get("isOccluded") == occlusion)}
        if not found:
            continue
        if max_yaw_rate is not None:
            if seq not in yaw_cache:
                yaw_cache[seq] = yaw_rates(root, seq)
            rate = yaw_cache[seq][int(frame)]
            if np.isnan(rate):
                unknown += 1
            elif abs(rate) > max_yaw_rate:
                turning += 1
                continue
        keep[(seq, frame)] = sorted(found)
    return keep, turning, unknown


def main():
    paths = load_paths()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=paths["coda_root"], help="CODa 根目錄")
    p.add_argument("--apply", action="store_true", help="實際刪除；不加只統計")
    p.add_argument("--classes", nargs="+", default=list(KEEP_CLASSES), help="保留含有這些 CODa 類別的 frame")
    p.add_argument("--occlusion", help="只算 isOccluded 等於此值的物件，例如 None")
    p.add_argument("--max-yaw-rate", type=float, help="|yaw rate| 超過此值（°/s）的 frame 也刪除，例如 3")
    args = p.parse_args()
    unknown = set(args.classes) - KEEP_CLASSES.keys()
    if unknown:
        sys.exit(f"未知類別 {sorted(unknown)}；可用：{list(KEEP_CLASSES)}")

    keep, turning, unknown = frames_to_keep(args.root, set(args.classes), args.occlusion, args.max_yaw_rate)
    if args.max_yaw_rate is not None:
        print(f"|yaw rate| > {args.max_yaw_rate:g}°/s 而排除的 frame：{turning}；pose 無效、未判斷而保留：{unknown}")
    if not keep:
        sys.exit(f"{args.root} 找不到含有保留類別的標註，停止（避免刪光所有資料）")

    removed, kept, freed = Counter(), Counter(), 0
    for d in sorted(args.root.glob("[23]d_*/*")):
        for f in d.rglob("*"):
            m = FRAME_RE.search(f.name)
            if not f.is_file() or not m:
                continue
            if m.groups() in keep:
                kept[d.relative_to(args.root).as_posix()] += 1
                continue
            removed[d.relative_to(args.root).as_posix()] += 1
            freed += f.stat().st_size
            if args.apply:
                f.unlink()

    for d in sorted(kept.keys() | removed.keys()):
        print(f"{d:24s} 保留 {kept[d]:6d}  刪除 {removed[d]:6d}")
    frame_count = Counter(c for classes in keep.values() for c in classes)
    print(f"\n保留 {len(keep)} 個 frame，來自 {len({s for s, _ in keep})} 個序列；各類別出現的 frame 數：")
    for c, n in frame_count.most_common():
        print(f"  {c:16s} {n:6d}")
    print(f"\n{'已刪除' if args.apply else '將刪除'} {sum(removed.values())} 個檔案，{freed / 1e9:.1f} GB")

    if args.apply:
        out = args.root / "kept_frames.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seq", "frame", "classes"])
            for (seq, frame), classes in sorted(keep.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
                w.writerow([seq, frame, ";".join(classes)])
        print(f"保留清單 -> {out}")


if __name__ == "__main__":
    main()
