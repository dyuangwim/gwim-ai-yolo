# battery_detector.py
import os, cv2, numpy as np
from ultralytics import YOLO

class BatteryDetector:
    def __init__(self, weights:str, imgsz:int=416, conf:float=0.35, threads:int=4):
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("NCNN_THREADS", str(threads))
        os.environ.setdefault("NCNN_VERBOSE", "0")
        self.model = YOLO(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)

        _ = self.model.predict(
            source=np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
            imgsz=self.imgsz, conf=self.conf, verbose=False
        )

    def detect(self, bgr_roi):
        """对卡纸 ROI 检测电池，返回电池框列表（局部坐标系）。"""
        r = self.model.predict(
            source=bgr_roi, imgsz=self.imgsz, conf=self.conf,
            verbose=False
        )[0]
        out=[]
        if r.boxes is not None and len(r.boxes)>0:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                c = float(b.conf[0]) if b.conf is not None else 0.0
                out.append({"xyxy": (x1,y1,x2,y2), "conf": c})
        return out
