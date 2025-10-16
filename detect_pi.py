# ~/detect_pi.py
import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best.pt")
    ap.add_argument("--imgsz", type=int, default=320)             # 小一档，提 FPS
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=2000)         # lores 上面积阈值稍小
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")

    # 相机相关：先让 AE/AWB 收敛后锁定（避免偏色/抖动）
    ap.add_argument("--shutter", type=int, default=8000,          # us，1/125s；光线不足就加到 12000
                    help="ExposureTime in microseconds when locking AE")
    ap.add_argument("--gain", type=float, default=2.0,            # 2~8 合理，越大越亮但噪点增
                    help="AnalogueGain when locking AE")
    ap.add_argument("--warmup", type=float, default=1.0,          # 让 AE/AWB 先自动 1 秒
                    help="seconds to let AE/AWB settle before locking")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    txt=f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), (0,255,0), -1)
    cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    picam2 = Picamera2()
    # main 用于显示，lores 用于推理
    config = picam2.create_video_configuration(
        main  = {"size": (1280, 960), "format": "RGB888"},
        lores = {"size": (640, 480),  "format": "RGB888"},
        controls={
            "FrameDurationLimits": (33333, 33333),  # 30fps 预算
            "AeEnable": True,
            "AwbEnable": True
        }
    )
    picam2.configure(config)
    picam2.start()

    # 让 AE/AWB 先收敛再锁定（避免偏色/变亮变暗）
    time.sleep(max(0.2, args.warmup))
    md = picam2.capture_metadata()
    # 锁定当前曝光和白平衡
    picam2.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": args.shutter,          # 手动设定更稳定
        "AnalogueGain": args.gain,
        # 也可用 md 的 ColourGains 锁定白平衡（可选）：
        # "ColourGains": md.get("ColourGains", (1.0, 1.0)),
    })

    model = YOLO(args.weights)
    fps_hist, last_save = [], 0

    try:
        while True:
            t0 = time.time()

            # 取 lores 做推理（小分辨率更快）
            lo = picam2.capture_array("lores")           # RGB
            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_RGB2BGR)

            results = model.predict(
                source=lo_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False
            )
            dets = []
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    area = max(0,(x2-x1))*max(0,(y2-y1))
                    if area < args.min_area: continue
                    dets.append((x1,y1,x2,y2,float(b.conf[0]), int(b.cls[0]) if b.cls is not None else 0))

            # 拿 main 画面用于显示（清晰）
            main_img = picam2.capture_array()            # main 默认流
            # 把 lores 的框按比例映射到 main 上
            lh, lw = lo.shape[:2]
            mh, mw = main_img.shape[:2]
            sx, sy = mw/float(lw), mh/float(lh)

            det_count, low_conf = 0, False
            for (x1,y1,x2,y2,conf,cls_id) in dets:
                X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                label = model.names.get(cls_id, "battery")
                draw_box(main_img, X1,Y1,X2,Y2, label, conf)
                det_count += 1
                if conf < (args.conf + 0.10): low_conf = True

            infer_ms = (time.time() - t0) * 1000.0
            fps_hist.append(1000.0 / max(1.0, infer_ms))
            if len(fps_hist) > 30: fps_hist.pop(0)
            fps_avg = sum(fps_hist)/len(fps_hist)
            hud = f"Detections: {det_count} | {infer_ms:.1f} ms ({fps_avg:.1f} FPS avg)"
            cv2.putText(main_img, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 回捞 hard cases：无检出或低置信
            now = time.time()
            if (det_count == 0 or low_conf) and (now - last_save > 1.0):
                p = os.path.join(args.save_dir, f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
                cv2.imwrite(p, cv2.cvtColor(main_img, cv2.COLOR_RGB2BGR))
                last_save = now

            if not args.headless:
                cv2.imshow("Battery Detection - Pi4 (lores infer + main display)", cv2.cvtColor(main_img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
