"""校正幀抽樣：從 kept_frames.csv 跨 seq 分層抽取 PTQ 校正集。

讀取 trt_quant.yaml 的 calibration 區段：
  n     - 目標幀數
  seed  - 隨機種子
  cam   - 供 print 統計用（不影響抽樣邏輯）

輸出：
  {outdir}/calib_frames.csv  (seq, frame)
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._config import load_config  # noqa: E402


def sample_frames(list_csv: Path, n: int, seed: int) -> list[tuple[int, int]]:
    """Stratified sample by seq. Returns [(seq, frame), ...]，total <= n。"""
    by_seq: dict[int, list[int]] = defaultdict(list)
    with open(list_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_seq[int(row["seq"])].append(int(row["frame"]))

    rng = random.Random(seed)
    seqs = sorted(by_seq)
    per_seq = n // len(seqs)
    result: list[tuple[int, int]] = []

    for seq in seqs:
        frames = by_seq[seq]
        k = min(per_seq, len(frames))
        result.extend((seq, f) for f in rng.sample(frames, k))

    # 若因某些 seq 幀數不足而還差幾幀，從剩餘幀補足
    used = {(s, f) for s, f in result}
    remaining = [(s, f) for s in seqs for f in by_seq[s] if (s, f) not in used]
    rng.shuffle(remaining)
    result.extend(remaining[: n - len(result)])
    result.sort()
    return result


def print_stats(samples: list[tuple[int, int]], all_csv: Path) -> None:
    total_by_seq: dict[int, int] = defaultdict(int)
    with open(all_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total_by_seq[int(row["seq"])] += 1

    sampled_by_seq: dict[int, int] = defaultdict(int)
    for seq, _ in samples:
        sampled_by_seq[seq] += 1

    print(f"校正集：{len(samples)} 幀，涵蓋 {len(sampled_by_seq)} 個 seq")
    for seq in sorted(sampled_by_seq):
        print(f"  seq {seq:3d}: {sampled_by_seq[seq]:4d} / {total_by_seq[seq]:4d} 幀")


def main() -> None:
    cfg = load_config()
    cal = cfg["calibration"]
    paths = cfg["paths"]

    kept = paths["kept_frames"]
    outdir = paths["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "calib_frames.csv"

    samples = sample_frames(kept, cal["n"], cal["seed"])

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "frame"])
        w.writerows(samples)

    print_stats(samples, kept)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
