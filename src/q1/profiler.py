import time

import psutil

try:
    import torch as _torch
    _CUDA = _torch.cuda.is_available()
except ImportError:
    _CUDA = False


class StageTimer:
    def __enter__(self):
        proc = psutil.Process()
        if _CUDA:
            _torch.cuda.reset_peak_memory_stats()
        ct = proc.cpu_times()
        self._t0   = time.perf_counter()
        self._cpu0 = ct.user + ct.system
        self._ram0 = proc.memory_info().rss
        return self

    def __exit__(self, *_):
        self._t1   = time.perf_counter()
        proc = psutil.Process()
        ct = proc.cpu_times()
        self._cpu1     = ct.user + ct.system
        self._ram1     = proc.memory_info().rss
        self._gpu_peak = _torch.cuda.max_memory_allocated() / 1e6 if _CUDA else 0.0

    @property
    def result(self) -> dict:
        return {
            "wall_ms":      (self._t1  - self._t0)  * 1000,
            "cpu_ms":       (self._cpu1 - self._cpu0) * 1000,
            "gpu_peak_mb":  self._gpu_peak,
            "ram_delta_mb": (self._ram1 - self._ram0) / 1e6,
        }


STAGES = ["io", "yolo", "prepare", "cluster"]


def compute_summary(per_frame: list, warmup: int = 0) -> dict:
    """把逐 frame 的 StageTimer 結果彙整成各 stage 的總計（不輸出逐 frame 資料）。

    每個 stage 只輸出三項：
      wall_s        納入統計的 frame 的 wall time 總和（秒）
      cpu_s         納入統計的 frame 的 CPU time 總和（秒）
      vram_peak_mb  納入統計的 frame 中 GPU 記憶體峰值的最大值（MB）

    warmup：前幾個 frame 不計入統計（第一次推論含 CUDA / TRT kernel 選擇，速度偏慢）。
    frame 數不超過 warmup 時無法排除，改用全部 frame 並在 meta.warmup_skipped 記 0。
    """
    skipped = warmup if 0 < warmup < len(per_frame) else 0
    used    = per_frame[skipped:]
    summary = {"meta": {"num_frames": len(per_frame),
                        "warmup_requested": warmup,
                        "warmup_skipped": skipped,
                        "frames_used": len(used)}}
    for stage in STAGES:
        recs = [f[stage] for f in used]
        summary[stage] = {
            "wall_s":       round(sum(r["wall_ms"] for r in recs) / 1000, 3),
            "cpu_s":        round(sum(r["cpu_ms"] for r in recs) / 1000, 3),
            "vram_peak_mb": round(max((r["gpu_peak_mb"] for r in recs), default=0.0), 1),
        }
    summary["total"] = {
        "wall_s":       round(sum(summary[s]["wall_s"] for s in STAGES), 3),
        "cpu_s":        round(sum(summary[s]["cpu_s"] for s in STAGES), 3),
        "vram_peak_mb": max(summary[s]["vram_peak_mb"] for s in STAGES),
    }
    return summary


def write_summary_md(summary: dict, path, run_info: dict | None = None) -> None:
    """把 compute_summary 的結果寫成人看的 Markdown 表格（內容與 profiling.json 相同）。

    run_info：額外要列在檔頭的執行資訊（例如 weights、總耗時），key/value 原樣輸出。
    """
    from datetime import datetime
    meta = summary["meta"]
    lines = ["# Profiling Summary", "",
             f"產生時間：{datetime.now():%Y-%m-%d %H:%M:%S}", "",
             "## 執行資訊", "", "| 項目 | 值 |", "|---|---|"]
    for k, v in (run_info or {}).items():
        lines.append(f"| {k} | `{v}` |")
    for k in ["num_frames", "warmup_requested", "warmup_skipped", "frames_used"]:
        lines.append(f"| {k} | {meta[k]} |")
    lines += ["", "## 各階段總計", "",
              "| Stage | Wall time (s) | CPU time (s) | VRAM peak (MB) |",
              "|---|---:|---:|---:|"]
    for stage in STAGES + ["total"]:
        m = summary[stage]
        name = f"**{stage}**" if stage == "total" else stage
        lines.append(f"| {name} | {m['wall_s']:.3f} | {m['cpu_s']:.3f} | {m['vram_peak_mb']:.1f} |")
    lines += ["", "> Wall / CPU time 為納入統計的 frame 加總；VRAM peak 為各 frame 峰值中的最大值。"
              "統計已排除前 `warmup_skipped` 個 frame；欄位定義見專案根目錄 `profiling.md`。", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
