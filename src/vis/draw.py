import zlib

import cv2
import numpy as np


def class_color(name):
    """同一類別固定同一種顏色（BGR，0-255）。"""
    h = zlib.crc32(name.encode())
    hsv = np.uint8([[[h % 180, 200, 255]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def colormap(values, vmin, vmax):
    """數值 → Nx3 RGB（0-1），用 OpenCV 的 TURBO colormap，近藍遠紅。"""
    t = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    idx = (t * 255).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(idx, cv2.COLORMAP_TURBO).reshape(-1, 3)
    return bgr[:, ::-1] / 255.0


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_detections(img, preds, conf_thr, title=""):
    """畫 YOLO 預測框：依類別上色，框上方標類別與 conf。只畫 score >= conf_thr 的框。"""
    shown = [p for p in preds if p["score"] >= conf_thr]
    for p in shown:
        x1, y1, x2, y2 = (int(round(v)) for v in p["bbox"])
        color = class_color(p["category"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _label(img, f"{p['category']} {p['score']:.2f}", x1, y1, color)
    head = f"{title}  pred {len(shown)} (conf>={conf_thr})"
    (tw, th), _ = cv2.getTextSize(head, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(img, head, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# ---------- 物件點群選擇（q1/distance/cluster.py 的結果） ----------

# 候選群的顏色（BGR），選中的群另外用白框標示
PALETTE = [(0, 255, 255), (255, 0, 255), (0, 165, 255), (255, 255, 0), (0, 255, 0),
           (255, 128, 0), (128, 0, 255), (0, 128, 255)]
GRAY = (130, 130, 130)
WHITE = (255, 255, 255)


def put_text(img, text, x, y, color, scale=0.45):
    """黑底文字，自動裁到影像內。"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(min(max(x, 0), img.shape[1] - tw - 4))
    y = int(max(y, th + 4))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _ipt(p):
    return int(round(p[0])), int(round(p[1]))


def draw_cluster_selection(img, pred, res, uv, xyz_cam, r=2):
    """畫一個偵測框的選群結果：
      YOLO 框（類別色）、core 框（細灰線）
      core 內的候選群：不同顏色的點，標 #id、點數、距離、分數（選中的前面加 *）
      非候選的小群：灰點；完整框內、core 外、沒被長回來的點：暗灰小點
      選中的群（含長回來的點）：白邊大點 + 白色 2D 外框
    uv / xyz_cam 為 FramePoints 的欄位，res 為 select_object_points 的結果。
    """
    box = pred["bbox"]
    color = class_color(pred["category"])
    cv2.rectangle(img, _ipt(box[:2]), _ipt(box[2:]), color, 2)
    cb = res["core_box"]
    cv2.rectangle(img, _ipt(cb[:2]), _ipt(cb[2:]), GRAY, 1)

    core_idx = np.flatnonzero(res["core"])
    labels = res["labels"]
    cands = [c for c in res["clusters"] if c["candidate"]]
    cand_color = {c["id"]: PALETTE[k % len(PALETTE)] for k, c in enumerate(cands)}
    for p in uv[res["in_box"] & ~res["core"] & ~res["final"]]:
        cv2.circle(img, _ipt(p), max(r - 1, 1), (70, 70, 70), -1)
    for i, p in zip(labels, uv[core_idx]):
        cv2.circle(img, _ipt(p), r, cand_color.get(i, GRAY), -1)
    if res["chosen"] is not None:
        c = cand_color[res["chosen"]]
        for p in uv[res["final"]]:
            cv2.circle(img, _ipt(p), r + 1, WHITE, -1)
            cv2.circle(img, _ipt(p), r, c, -1)
        e = res["extent"]
        cv2.rectangle(img, _ipt(e[:2]), _ipt(e[2:]), WHITE, 1)

    for c in cands:
        d = np.median(np.linalg.norm(xyz_cam[core_idx[labels == c["id"]]], axis=1))
        mark = "*" if c["id"] == res["chosen"] else ""
        put_text(img, f"{mark}#{c['id']} n{c['n']} {d:.1f}m s{c['score']:.2f}",
                 c["med_uv"][0] + 4, c["med_uv"][1], cand_color[c["id"]], 0.4)

    if res["chosen"] is not None:
        head = (f"{pred['category']} {res['dist']:.1f}m n{int(res['final'].sum())} "
                f"IoU{res['iou']:.2f} cov{res['cov_w']:.2f}x{res['cov_h']:.2f}")
        if res["status"] != "ok":
            head += f" [{res['status']}]"
    else:
        head = f"{pred['category']} [{res['status']}]"
    put_text(img, head, box[0], box[1] - 2, color, 0.5)


def crop_box(img, box, pad=0.3, min_h=320):
    """裁出 box 周圍區域並放大到至少 min_h 高，方便看小物件。"""
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * pad + 20, (y2 - y1) * pad + 20
    h, w = img.shape[:2]
    a, b = int(max(x1 - pw, 0)), int(max(y1 - ph - 20, 0))
    c, d = int(min(x2 + pw, w)), int(min(y2 + ph, h))
    out = img[b:d, a:c]
    s = max(1.0, min_h / max(out.shape[0], 1))
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
