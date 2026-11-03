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


def compute_summary(per_frame: list) -> dict:
    import numpy as np
    stages  = ["io", "yolo", "prepare", "cluster"]
    metrics = ["wall_ms", "cpu_ms", "gpu_peak_mb", "ram_delta_mb"]
    summary = {}
    for stage in stages:
        summary[stage] = {}
        for metric in metrics:
            vals = np.array([f[stage][metric] for f in per_frame])
            summary[stage][metric] = {
                "mean": round(float(vals.mean()), 3),
                "p95":  round(float(np.percentile(vals, 95)), 3),
            }
    return summary
