# detect_pi2.py — RPi4 + NCNN 稳定版（lores=YUV420 → BGR 转换；main=BGR；可选 headless）
import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img_bgr, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    txt=f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img_bgr, (x1, y0), (x1+tw+4, y0+th+6), (0,255,0), -1)
    cv2.putText(img_bgr, txt, (x1+2, y0+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    # 可选：限制线程，减少争用
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("NCNN_THREADS", "4")

    args = parse_args()
    ensure_dir(args.save_dir)

    # ---- Camera ----
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    cfg = picam2.create_video_configuration(
        main = {"size": (main_w, main_h), "format": "BGR888"},   # 显示用，BGR 直接画
        lores= {"size": (args.imgsz, args.imgsz), "format": "YUV420"},  # 推理用，必须 YUV
        controls={"AeEnable": True, "AwbEnable": True}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.8)

    if args.save_debug:
        f0 = picam2.capture_array("main")
        if f0.ndim == 3 and f0.shape[2] == 4: f0 = f0[:, :, :3]
        cv2.imwrite("/home/pi/view_debug.jpg", f0)
        print("Saved /home/pi/view_debug.jpg", f0.shape)

    # ---- Model ----
    model = YOLO(args.weights)

    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", main_w, main_h)

    last_save = 0
    fps_hist = []

    try:
        while True:
            t0 = time.time()

            # lores：YUV420 -> BGR (I420)
            yuv = picam2.capture_array("lores")   # shape: (H*3/2, W), planar
            # OpenCV 期望的 I420 格式高度为 H*3/2，宽为 W
            infer_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            # 防止尺寸跑偏（少数驱动会给到非 imgsz×imgsz）
            if infer_bgr.shape[:2] != (args.imgsz, args.imgsz):
                infer_bgr = cv2.resize(infer_bgr, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)

            # main：BGR888，直接显示
            frame_bgr = picam2.capture_array("main")
            if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4: frame_bgr = frame_bgr[:, :, :3]

            # 推理（直接喂 BGR，Ultralytics 内部会处理颜色）
            r = model.predict(source=infer_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # 将 lores 坐标映射到 main 尺寸
            lh, lw = args.imgsz, args.imgsz
            mh, mw = frame_bgr.shape[:2]
            sx, sy = mw/float(lw), mh/float(lh)

            det_count, low_conf = 0, False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    if (X2-X1)*(Y2-Y1) < args.min_area:
                        continue
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    label = r.names.get(cls_id, "battery")
                    draw_box(frame_bgr, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

            dt = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0));  fps_hist = fps_hist[-30:]
            hud = f"Det:{det_count} | {dt:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS)"
            cv2.putText(frame_bgr, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # hard cases
            now = time.time()
            if (det_count==0 or low_conf) and (now-last_save>1.0):
                fn=f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame_bgr)
                last_save = now

            if not args.headless:
                cv2.imshow("Battery Detection", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
