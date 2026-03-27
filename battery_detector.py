# battery_detector.py
import os
import cv2
import numpy as np
from ultralytics import YOLO


class BatteryDetector:
    """
    Battery detector using the SAME call style as your Colab test:

        model('/path/to/img.jpg')  
        model(bgr_ndarray)        

    - No longer forcibly specifying imgsz / iou / max_det
    - Use Ultralytics default parameters completely (imgsz=640, conf=0.25, iou=0.7)
    - No longer perform additional NMS, directly use the model's built-in NMS results
    """

    def __init__(self, weights: str, imgsz: int = 640, conf: float = None, threads: int = 4):
        # The number of threads can still be set to prevent CPU thread spikes.
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ.setdefault("NCNN_VERBOSE", "0")

        # Here, we'll directly load the .pt file (you're currently uploading /home/pi/models/battery.pt).
        self.model = YOLO(weights)

        # imgsz currently only records data and does not forcibly pass it to model().
        self.imgsz = int(imgsz)
        # If conf=None, Ultralytics default version 0.25 will be used.
        self.conf = conf

    def detect_full(self, bgr):
        """
        Run a battery detection test once across the entire BGR image.
        
        Returns:
        [ { "xyxy": (x1,y1,x2,y2), "conf": float }, ... ]
        The coordinates are full-image coordinates.
        """

        # —— Core: Same calling method as your Colab. ——
        if self.conf is None:
            # Use all default parameters (imgsz=640, conf=0.25, iou=0.7)
            r = self.model(bgr, verbose=False)[0]
        else:
            # Only override the confidence threshold, keep other parameters as default
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
