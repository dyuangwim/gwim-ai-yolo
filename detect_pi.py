import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best.pt")
    ap.add_argument("--imgsz", type=int, default=320)       # 320 或 416，和 lores 对齐
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)   # lores 上阈值稍小
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--shutter", type=int, default=6000)    # 1/166s；再不清就 4000
    ap.add_argument("--gain", type=float, default=4.0)      # 4~8 视光线而定，配合加灯
    ap.add_argument("--warmup", type=float, default=1.5)    # 让 AWB/AE 先收敛
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    txt=f"{label} {conf:.2f}"; (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), (0,255,0), -1)
    cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    picam2 = Picamera2()
    # Pi 4：lores 必须 YUV420；尺寸与 imgsz 对齐，减少额外 resize
    config = picam2.create_video_configuration(
        main  = {"size": (1280, 960), "format": "RGB888"},
        lores = {"size": (args.imgsz, args.imgsz), "format": "YUV420"},
        controls={ "FrameDurationLimits": (33333, 33333), "AeEnable": True, "AwbEnable": True }
    )
    picam2.configure(config)
    picam2.start()

    # 先让 AE/AWB 自稳，再锁参数（避免偏紫/忽明忽暗）
    time.sleep(max(0.5, args.warmup))
    md = picam2.capture_metadata()
    # 锁定当前白平衡增益（若可取到）
    colour_gains = md.get("ColourGains", (1.0, 1.0))
    picam2.set_controls({
        "AeEnable": False, "AwbEnable": False,
        "ExposureTime": args.shutter, "AnalogueGain": args.gain,
        "ColourGains": colour_gains
    })

    model = YOLO(args.weights)
    fps_hist, last_save = [], 0

    try:
        while True:
            t0 = time.time()
            # 一次采集两路，减少内存搬运
            req = picam2.capture_request()
            lo = req.make_array('lores')   # YUV420: (h*3/2, w)
            main_rgb = req.make_array('main')
            req.release()

            # YUV420 -> BGR（I420）
            lo_w = lo.shape[1]; lo_h = lo.shape[0] * 2 // 3
            lo = lo.reshape((lo_h * 3 // 2, lo_w))
            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)

            # YOLO 推理（在 lores 上）
            r = model.predict(source=lo_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # 准备 main 显示
            main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
            mh, mw = main_bgr.shape[:2]
            sx = mw / float(args.imgsz); sy = mh / float(args.imgsz)

            det_count, low_conf = 0, False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    if (x2-x1)*(y2-y1) < args.min_area: continue
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    label = model.names.get(cls_id, "battery")
                    # lores → main 映射（两路同为正方形 lores 与 4:3 main）
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    draw_box(main_bgr, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

            infer_ms = (time.time() - t0) * 1000.0
            fps_hist.append(1000.0 / max(1.0, infer_ms))
            if len(fps_hist) > 30: fps_hist.pop(0)
            hud = f"Detections: {det_count} | {infer_ms:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS avg)"
            cv2.putText(main_bgr, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 回捞 hard cases
            now = time.time()
            if (det_count == 0 or low_conf) and (now - last_save > 1.0):
                cv2.imwrite(os.path.join(args.save_dir, f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"), main_bgr)
                last_save = now

            if not args.headless:
                cv2.imshow("Battery Detection - Pi4 (dual-stream optimized)", main_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
