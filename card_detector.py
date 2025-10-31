import os, cv2, numpy as np
from ultralytics import YOLO

def _resolve_weights(p:str)->str:
    if os.path.isdir(p):
        for name in ("model.ncnn.param","best.ncnn.param"):
            cand = os.path.join(p, name)
            if os.path.exists(cand):
                return cand
    return p  # 已是文件路径

class CardDetector:
    def __init__(self, weights:str, imgsz:int=640, conf:float=0.35, threads:int=4):
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("NCNN_THREADS", str(threads))
        os.environ.setdefault("NCNN_VERBOSE", "0")

        weights = _resolve_weights(weights)
        self.model = YOLO(weights, task="detect")   # 显式指定 task
        self.imgsz = int(imgsz)
        self.conf = float(conf)

        _ = self.model.predict(
            source=np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
            imgsz=self.imgsz, conf=self.conf, verbose=False
        )

    def detect(self, bgr):
        r = self.model.predict(source=bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out=[]
        if r.boxes is not None and len(r.boxes)>0:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                c = float(b.conf[0]) if b.conf is not None else 0.0
                out.append({"xyxy": (x1,y1,x2,y2), "conf": c})
        return out
