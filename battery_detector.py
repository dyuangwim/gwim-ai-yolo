# battery_detector.py
import os
import cv2
import numpy as np
from ultralytics import YOLO

def nms_numpy(boxes, scores, iou_thr=0.55, topk=80):
    """Simple NMS for full-image battery detection."""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0 and len(keep) < topk:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]

    return np.asarray(keep, dtype=int)

class BatteryDetector:
    """
    Battery detector using a normal .pt model (recommended).
    如果你传入的是 NCNN 路径（*.ncnn.param），Ultralytics 也会跑，
    但目前这份 NCNN 模型有问题，所以建议只用 .pt。
    """

    def __init__(self, weights: str, imgsz: int = 416, conf: float = 0.50, threads: int = 4):
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ.setdefault("NCNN_VERBOSE", "0")

        self.model = YOLO(weights)      # 这里既可以是 .pt，也可以是 ncnn，但推荐 pt
        self.imgsz = int(imgsz)
        self.conf = float(conf)

        # 预热
        _ = self.model.predict(
            source=np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
            imgsz=self.imgsz, conf=self.conf, verbose=False
        )

    def detect_full(self, bgr):
        """
        在整张图上跑一次电池 YOLO，返回：
        [ { "xyxy": (x1,y1,x2,y2), "conf": float }, ... ]
        坐标是全图坐标。
        """
        r = self.model.predict(
            source=bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=0.55,
            max_det=80,
            verbose=False
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = (r.boxes.conf if r.boxes.conf is not None
                 else np.zeros((len(boxes), 1))).cpu().numpy().reshape(-1)

        # 再做一次 NMS，保证干净
        keep = nms_numpy(boxes, confs, iou_thr=0.55, topk=80)

        out = []
        for i in keep:
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            c = float(confs[i])
            out.append({"xyxy": (x1, y1, x2, y2), "conf": c})
        return out
