"""ultralytics YOLO 2D 偵測，逐張推論並記錄耗時。第一張當 warmup，不計入統計。

只輸出目標類別（q1/data/classes.py）：推論時用 classes= 只保留 COCO_TO_TARGET 內的類別
（person、bicycle、car，其他 COCO 類別一律不輸出），再依 class_map 換成目標類別名稱。
若 class_map 把多個 COCO 類別對到同一目標類別，歸類後的 NMS 會把重複框合併；目前是一對一，這一步不會刪框。
"""
import csv
import time

from q1.data.classes import COCO_TO_TARGET


class YoloDetector:
    def __init__(self, weights, imgsz=1280, conf=0.25, device="0", half=False, iou=0.7, class_map=COCO_TO_TARGET):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.model = YOLO(str(weights))
        self.weights, self.imgsz, self.conf, self.device = weights, imgsz, conf, device
        # TRT engine 的精度已 bake-in，half 參數在此無意義
        self.half = None if str(weights).endswith(".engine") else half
        self.iou, self.class_map = iou, class_map
        self.class_ids = [i for i, n in self.model.names.items() if n in class_map]
        self.targets = sorted(set(class_map.values()))
        self.records = []  # (seq, frame, wall_ms, pre_ms, inf_ms, post_ms, n_det)

    def _sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()

    def __call__(self, img, seq_id=None, frame=None):
        """img：BGR 影像。回傳 list of {"bbox": [x1, y1, x2, y2], "category", "raw_class", "score"}。
        category 為歸類後的目標類別，raw_class 為 YOLO 原始的 COCO 類別。"""
        from torchvision.ops import batched_nms
        self._sync()
        t0 = time.perf_counter()
        extra = {"half": True} if self.half else {}  # TRT engine 時 self.half is None，自動跳過
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device,
                               classes=self.class_ids, verbose=False, **extra)[0]
        boxes, scores, cls = r.boxes.xyxy, r.boxes.conf, r.boxes.cls.long()
        raw = [self.model.names[int(c)] for c in cls.tolist()]
        tgt = self.torch.tensor([self.targets.index(self.class_map[n]) for n in raw], dtype=self.torch.long,
                                device=boxes.device)
        keep = batched_nms(boxes.float(), scores.float(), tgt, self.iou).tolist() if len(raw) else []
        self._sync()
        wall = (time.perf_counter() - t0) * 1000
        boxes, scores = boxes.cpu(), scores.cpu()
        preds = [{"bbox": boxes[i].tolist(), "category": self.targets[int(tgt[i])], "raw_class": raw[i],
                  "score": float(scores[i])} for i in keep]
        sp = r.speed
        self.records.append((seq_id, frame, wall, sp["preprocess"], sp["inference"], sp["postprocess"], len(preds)))
        return preds

    @property
    def device_name(self):
        return str(self.model.predictor.device) if self.model.predictor else self.device

    def summary(self, total_s=None):
        import numpy as np
        rec = self.records[1:] if len(self.records) > 1 else self.records
        cols = np.array([r[2:6] for r in rec], dtype=np.float64)
        stats = {}
        for i, name in enumerate(["wall_ms", "preprocess_ms", "inference_ms", "postprocess_ms"]):
            stats[name] = {"mean": float(cols[:, i].mean()), "median": float(np.median(cols[:, i])),
                           "p95": float(np.percentile(cols[:, i], 95)), "sum_s": float(cols[:, i].sum() / 1000)}
        cuda = self.torch.cuda.is_available()
        return {
            "weights": str(self.weights), "imgsz": self.imgsz, "conf": self.conf, "iou": self.iou,
            "class_map": self.class_map,
            "half": self.half, "device": self.device_name,
            "engine_mode": "trt" if self.half is None else "pytorch",
            "gpu": self.torch.cuda.get_device_name(0) if cuda else None,
            "num_frames": len(self.records), "warmup_frame_ms": self.records[0][2],
            "yolo_total_s": float(sum(r[2] for r in self.records) / 1000),
            "script_total_s": total_s,
            "fps_wall": 1000 / stats["wall_ms"]["mean"],
            "stats_excluding_warmup": stats,
            "max_gpu_mem_MB": self.torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        }


DET_FIELDS = ["seq", "frame", "class", "raw_class", "score", "x1", "y1", "x2", "y2"]


def detection_rows(seq_id, frame, preds):
    return [(seq_id, frame, p["category"], p.get("raw_class", p["category"]), round(p["score"], 4),
             *(round(v, 1) for v in p["bbox"])) for p in preds]


def load_detections(path, classes=None, conf=0.0):
    """讀 detections.csv → {(seq, frame): [pred, ...]}。classes 為 None 時不篩類別。"""
    from collections import defaultdict
    dets = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (classes is not None and r["class"] not in classes) or float(r["score"]) < conf:
                continue
            dets[(int(r["seq"]), int(r["frame"]))].append({
                "category": r["class"], "score": float(r["score"]),
                "bbox": [float(r[k]) for k in ("x1", "y1", "x2", "y2")]})
    return dict(dets)
