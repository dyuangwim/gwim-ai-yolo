# yolo_detector.py
import os, time
import cv2, numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

class YoloDetector:
    def __init__(self, weights, imgsz=320, conf=0.35, preview=False, threads=4):
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["NCNN_THREADS"] = str(threads)
        os.environ.setdefault("NCNN_VERBOSE", "0")
        self.imgsz = int(imgsz)
        self.conf = float(conf)

        # Camera
        self.picam2 = Picamera2()
        self.main_w, self.main_h = 1280, 960
        cfg = (self.picam2.create_preview_configuration if preview else self.picam2.create_video_configuration)(
            main={"size": (self.main_w, self.main_h), "format": "YUV420"},
            controls={"FrameRate": 30}
        )
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(1.0)

        # YOLO
        self.model = YOLO(weights)
        # warmup
        _ = self.model.predict(source=np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
                               imgsz=self.imgsz, verbose=False)

        # scale factors (inference->camera)
        self.sx, self.sy = self.main_w/float(self.imgsz), self.main_h/float(self.imgsz)

    def capture_bgr(self):
        yuv = self.picam2.capture_array("main")
        try:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        except Exception:
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    def track_once(self, frame_bgr):
        infer = cv2.resize(frame_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        r = self.model.track(source=infer, imgsz=self.imgsz, conf=self.conf,
                             verbose=False, persist=True)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
            order = np.argsort((-r.boxes.conf.cpu().numpy()).flatten())
            for i in order:
                b = r.boxes[int(i)]
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                # scale back to camera coords
                X1, Y1, X2, Y2 = int(x1*self.sx), int(y1*self.sy), int(x2*self.sx), int(y2*self.sy)
                dets.append({
                    "xyxy": (X1, Y1, X2, Y2),
                    "conf": float(b.conf[0]) if b.conf is not None else 0.0,
                    "track_id": int(b.id[0])
                })
        return dets

    def close(self):
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass
