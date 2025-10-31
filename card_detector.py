import os, cv2, numpy as np
from ultralytics import YOLO

def _load_ncnn(weights_path:str):
    """
    兼容两种写法：
    1) 传目录（推荐，老版本 Ultralytics 只认目录）
    2) 传文件：.../model.ncnn.param
    我们先尝试“目录”，失败再尝试“文件”，最后再回退父目录。
    """
    last_err = None
    # A. 目录优先（你的环境喜欢这种）
    if os.path.isdir(weights_path):
        try:
            return YOLO(weights_path, task="detect")
        except Exception as e:
            last_err = e
            # 再试目录里的 .param
            for name in ("model.ncnn.param", "best.ncnn.param"):
                p = os.path.join(weights_path, name)
                if os.path.exists(p):
                    try:
                        return YOLO(p, task="detect")
                    except Exception as e2:
                        last_err = e2
    else:
        # B. 传的是文件（.param）
        try:
            return YOLO(weights_path, task="detect")
        except Exception as e:
            last_err = e
            # 回退用父目录
            d = os.path.dirname(weights_path)
            if os.path.isdir(d):
                try:
                    return YOLO(d, task="detect")
                except Exception as e2:
                    last_err = e2
    raise last_err

class CardDetector:
    def __init__(self, weights:str, imgsz:int=640, conf:float=0.35, threads:int=4):
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("NCNN_THREADS", str(threads))
        os.environ.setdefault("NCNN_VERBOSE", "0")

        self.model = _load_ncnn(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)

        # 预热
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
