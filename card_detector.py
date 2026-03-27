# card_detector.py
import os, cv2, numpy as np
from ultralytics import YOLO

def nms_numpy(boxes, scores, iou_thr=0.55, topk=50):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < topk:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return np.asarray(keep, dtype=int)

def _load_ncnn(weights_path:str):
    last_err = None
    if os.path.isdir(weights_path):
        # Try loading the directory directly first (some versions support this).
        try:
            return YOLO(weights_path, task="detect")
        except Exception as e:
            last_err = e
            # Try common filenames again
            for name in ("model.ncnn.param", "best.ncnn.param"):
                p = os.path.join(weights_path, name)
                if os.path.exists(p):
                    try:
                        return YOLO(p, task="detect")
                    except Exception as e2:
                        last_err = e2
    else:
        try:
            return YOLO(weights_path, task="detect")
        except Exception as e:
            last_err = e
            d = os.path.dirname(weights_path)
            if os.path.isdir(d):
                try:
                    return YOLO(d, task="detect")
                except Exception as e2:
                    last_err = e2
    raise last_err or RuntimeError("Unable to load NCNN model")

class CardDetector:
    def __init__(self, weights:str, imgsz:int=640, conf:float=0.50, threads:int=4):
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("NCNN_THREADS", str(threads))
        os.environ.setdefault("NCNN_VERBOSE", "0")
        self.model = _load_ncnn(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        # preheating
        _ = self.model.predict(
            source=np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
            imgsz=self.imgsz, conf=self.conf, verbose=False
        )

    def detect(self, bgr):
        r = self.model.predict(
            source=bgr, imgsz=self.imgsz, conf=self.conf,
            iou=0.55, max_det=80, verbose=False
        )[0]
        out=[]
        if r.boxes is not None and len(r.boxes)>0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = (r.boxes.conf if r.boxes.conf is not None else np.zeros((len(boxes),1))).cpu().numpy().reshape(-1)
            # Perform local NMS only once, and never use the condition len(boxes)>100 again.
            keep = nms_numpy(boxes, confs, iou_thr=0.55, topk=20)
            for i in keep:
                x1,y1,x2,y2 = map(int, boxes[i].tolist())
                c = float(confs[i])
                out.append({"xyxy": (x1,y1,x2,y2), "conf": c})
        return out
