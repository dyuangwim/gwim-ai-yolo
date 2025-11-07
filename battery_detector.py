# battery_detector.py
import os
import cv2
import numpy as np
from ultralytics import YOLO


class BatteryDetector:
    """
    Battery detector using the SAME call style as your Colab test:

        model('/path/to/img.jpg')   # 在 Colab
        model(bgr_ndarray)         # 在 Raspberry Pi

    - 不再强行指定 imgsz / iou / max_det
    - 完全使用 Ultralytics 默认参数 (imgsz=640, conf=0.25, iou=0.7)
    - 也不再做额外 NMS，直接使用模型内置 NMS 的结果
    """

    def __init__(self, weights: str, imgsz: int = 640, conf: float = None, threads: int = 4):
        # 线程数还是可以设，避免 CPU 线程乱飙
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ.setdefault("NCNN_VERBOSE", "0")

        # 这里直接加载 .pt（你现在传的是 /home/pi/models/battery.pt）
        self.model = YOLO(weights)

        # imgsz 目前只做记录，不强行传给 model()
        self.imgsz = int(imgsz)
        # 如果 conf=None，就用 Ultralytics 默认 0.25
        self.conf = conf

    def detect_full(self, bgr):
        """
        在整张 BGR 图像上跑一次电池检测。
        返回:
            [ { "xyxy": (x1,y1,x2,y2), "conf": float }, ... ]
        坐标是全图坐标。
        """

        # —— 核心：和你 Colab 一样的调用方式 ——
        if self.conf is None:
            # 完全使用默认参数 (imgsz=640, conf=0.25, iou=0.7)
            r = self.model(bgr, verbose=False)[0]
        else:
            # 只覆盖置信度，其他参数仍用默认
            r = self.model(bgr, conf=self.conf, verbose=False)[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = (r.boxes.conf if r.boxes.conf is not None
                 else np.zeros((len(boxes), 1))).cpu().numpy().reshape(-1)

        out = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i].tolist())
            c = float(confs[i])
            out.append({"xyxy": (x1, y1, x2, y2), "conf": c})
        return out
