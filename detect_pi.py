#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser(description="Raspberry Pi + Picamera2 + YOLO (NCNN) Realtime")
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model",
                    help="Folder containing model.ncnn.param/bin")
    ap.add_argument("--imgsz", type=int, default=320, help="inference size (must be multiple of 32)")
    ap.add_argument("--conf", type=float, default=0.60, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.60, help="NMS IOU threshold")
    ap.add_argument("--max_det", type=int, default=40, help="max detections per image")
    ap.add_argument("--min_area", type=int, default=8000, help="min bbox area to keep")
    ap.add_argument("--topk_draw", type=int, default=12, help="max number of boxes to draw")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true", help="no imshow (for SSH)", default=True)
    ap.add_argument("--no_labels", action="store_true", help="no text labels, only boxes", default=False)

    # 相机设置
    ap.add_argument("--shutter", type=int, default=3500, help="ExposureTime(us)")
    ap.add_argument("--gain", type=float, default=8.0, help="AnalogueGain")
    ap.add_argument("--warmup", type=float, default=1.0, help="AWB/AE warmup")
    ap.add_argument("--lock_awb", action="store_true", help="lock AWB after warmup")

    # 可选 ROI (x, y, w, h)
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="crop ROI to reduce background interference")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label, conf, no_labels=False):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    if no_labels:
        return
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def mask_roi(img_bgr, roi):
    if not roi: return img_bgr, (0,0)
    x,y,w,h = roi
    x2, y2 = x + w, y + h
    x = max(0,x); y = max(0,y)
    x2 = min(img_bgr.shape[1], x2)
    y2 = min(img_bgr.shape[0], y2)
    return img_bgr[y:y2, x:x2], (x, y)

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    # ==== 初始化相机 ====
    picam2 = Picamera2()
    lo_side = args.imgsz
    cfg = picam2.create_video_configuration(
        main = {"size": (1280, 960), "format": "RGB888"},
        lores= {"size": (lo_side, lo_side), "format": "YUV420"},
        controls={
            "FrameDurationLimits": (33333, 33333),
            "AeEnable": True,
            "AwbEnable": True,
            "NoiseReductionMode": 0,
            "Sharpness": 1.3
        }
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(max(0.5, args.warmup))
    controls = {"AeEnable": False, "ExposureTime": args.shutter, "AnalogueGain": args.gain}
    if args.lock_awb:
        md = picam2.capture_metadata()
        controls.update({"AwbEnable": False, "ColourGains": md.get("ColourGains",(1.0,1.0))})
    picam2.set_controls(controls)

    # ==== 加载 NCNN 模型 ====
    wdir = args.weights
    assert os.path.isdir(wdir), f"weights folder not found: {wdir}"
    assert os.path.exists(os.path.join(wdir,"model.ncnn.param")), "missing model.ncnn.param"
    assert os.path.exists(os.path.join(wdir,"model.ncnn.bin")), "missing model.ncnn.bin"
    model = YOLO(wdir)

    backend_name = getattr(getattr(model, "predictor", None), "backend", None)
    print(f"[INFO] Backend:", backend_name)
    if backend_name is None:
        print("❌ WARNING: NCNN backend not loaded, fallback may be slow!")

    fps_hist, last_save = [], 0.0
    try:
        while True:
            t0 = time.time()
            req = picam2.capture_request()
            lo = req.make_array('lores')
            main_rgb = req.make_array('main')
            req.release()

            w = lo.shape[1]
            h = lo.shape[0] * 2 // 3
            lo = lo.reshape((h*3//2, w))
            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)

            # ROI
            roi_img, (ox, oy) = mask_roi(lo_bgr, args.roi)

            # YOLO 推理
            r0 = model.predict(
                source=roi_img,
                imgsz=roi_img.shape[1],
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                verbose=False
            )[0]

            main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
            mh, mw = main_bgr.shape[:2]
            sx, sy = mw/float(w), mh/float(h)

            dets = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                clses = r0.boxes.cls.cpu().numpy() if r0.boxes.cls is not None else None
                for i,(x1,y1,x2,y2) in enumerate(xyxy):
                    # --- NaN 检查 ---
                    if any(map(lambda v: v != v or v is None, [x1,y1,x2,y2])):
                        continue
                    x1 += ox; x2 += ox; y1 += oy; y2 += oy
                    area = (x2 - x1) * (y2 - y1)
                    if area < args.min_area:
                        continue
                    dets.append((float(confs[i]), int(clses[i]) if clses is not None else 0, x1,y1,x2,y2))

            dets.sort(key=lambda z: z[0], reverse=True)
            dets = dets[:max(1,args.topk_draw)]

            for conf, cls_id, x1,y1,x2,y2 in dets:
                # 再次防护
                if any(map(lambda v: v != v or v is None, [x1,y1,x2,y2])):
                    continue
                X1,Y1,X2,Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                label = (r0.names.get(cls_id,"battery") if hasattr(r0,"names") else "battery")
                draw_box(main_bgr, X1,Y1,X2,Y2, label, conf, no_labels=args.no_labels)

            dt_ms = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(1.0, dt_ms))
            if len(fps_hist) > 30: fps_hist.pop(0)
            fps_avg = sum(fps_hist)/len(fps_hist)
            cv2.putText(main_bgr, f"Detections:{len(dets)} | {dt_ms:.1f} ms ({fps_avg:.1f} FPS avg)",
                        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            now = time.time()
            if (len(dets) == 0) and (now - last_save > 1.0):
                ensure_dir(args.save_dir)
                cv2.imwrite(os.path.join(args.save_dir, f"hard_0_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"), main_bgr)
                last_save = now

            if not args.headless:
                cv2.imshow("Battery Detection - Pi", main_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
