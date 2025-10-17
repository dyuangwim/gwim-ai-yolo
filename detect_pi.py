# detect_pi.py — LOCKED BGR PIPELINE (confirmed-good on your Pi)
# - 强制相机主流 BGR888（三通道）
# - 推理/显示/保存全程 BGR，不做任何 RGB<->BGR 转换
# - 保险裁掉 4 通道（若驱动意外给 BGRA）
import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-debug", action="store_true",
                    help="启动后保存一帧 /home/pi/view_debug.jpg 以核对颜色")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img_bgr, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img_bgr, (x1, y1-th-6), (x1+tw+4, y1), (0,255,0), -1)
    cv2.putText(img_bgr, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    # ✅ 强制 3 通道 BGR888（与你机器颜色一致）
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (1280, 960), "format": "BGR888"},
        controls={"AeEnable": True, "AwbEnable": True}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1.0)  # 让 AE/AWB 收敛

    model = YOLO(args.weights)
    last_save = 0
    fps_hist = []

    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", 960, 720)

    if args.save_debug:
        f0 = picam2.capture_array("main")
        if f0.ndim == 3 and f0.shape[2] == 4:  # 保险：裁掉 alpha
            f0 = f0[:, :, :3]
        cv2.imwrite("/home/pi/view_debug.jpg", f0)
        print("Saved /home/pi/view_debug.jpg", f0.shape)

    try:
        while True:
            t0 = time.time()

            # 采集（BGR888）
            frame = picam2.capture_array("main")
            if frame.ndim == 3 and frame.shape[2] == 4:  # 极少数情况下驱动给 BGRA
                frame = frame[:, :, :3]

            # 推理：Ultralytics 支持 numpy BGR（与 cv2.imread 一致）
            infer_in = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            r = model.predict(source=infer_in, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # 画框/显示都在 BGR 上完成
            h, w = frame.shape[:2]
            sx = w / float(args.imgsz); sy = h / float(args.imgsz)

            det_count, low_conf = 0, False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    if (x2-x1)*(y2-y1) < args.min_area: continue
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    label = r.names.get(cls_id, "battery")
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    draw_box(frame, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

            dt = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0)); fps_hist = fps_hist[-30:]
            hud = f"Detections:{det_count} | {dt:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS)"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 保存 hard cases
            now = time.time()
            if (det_count==0 or low_conf) and (now-last_save>1.0):
                fn = f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame)
                last_save = now

            if not args.headless:
                cv2.imshow("Battery Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
