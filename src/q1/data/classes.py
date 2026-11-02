"""目標類別定義：只處理 person、bicycle、car 三類，其餘全部忽略。

YOLO 預測（COCO 類別）與 CODa 3D bbox 標註（CODa 類別）都先對應到這三類；
不在對應表內的類別一律丟棄，不推論、不畫圖、不估距離。
"""

# 目標類別（沿用 COCO 名稱）
TARGET_CLASSES = ["person", "bicycle", "car"]

# YOLO（COCO）類別 → 目標類別。只保留三類，bus、truck、motorcycle 等不併入 car，直接忽略
COCO_TO_TARGET = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
}

# CODa 標註類別 → 目標類別。只保留三類；Bus、各種 Truck／Vehicle 不併入 car
# （YOLO 分不出這麼多種車輛），和 Motorcycle、Scooter 等一樣忽略
CODA_TO_TARGET = {
    "Pedestrian": "person",
    "Bike": "bicycle",
    "Car": "car",
}

# 相容舊名稱（filter_coda_frames.py 用 key 篩 CODa 類別）
KEEP_CLASSES = CODA_TO_TARGET


def gt_boxes(boxes):
    """CODa 3D bbox 標註 list → 只留目標類別，classId 換成目標類別名稱（原類別存在 codaClass）。"""
    return [{**b, "classId": CODA_TO_TARGET[b["classId"]], "codaClass": b["classId"]}
            for b in boxes if b["classId"] in CODA_TO_TARGET]
