# detect_pi.py  — Pi Camera v3 / Picamera2 + Ultralytics YOLO
# 目标：稳定帧率、不偏色、在 lores 跑推理、在 main 高分辨显示框
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

    # 曝光相关（单位：微秒 / 倍数），建议配合补光
    ap.add_argument("--shutter", type=int, default=6000)   # 1/166s
    ap.add_argument("--gain", type=float, default=4.0)

    # 画面稳定：预热时长；是否锁AWB
    ap.add_argument("--warmup", type=float, default=1.2)
    ap.add_argument("--lock-awb", action="store_true",
                    help="仅当元数据存在有效 ColourGains 时才锁定白平衡；默认不锁以避免偏色")
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
    # main 用 BGR888（OpenCV 直用），lores 用 YUV420 做推理
    config = picam2.create_video_configuration(
        main  = {"size": (1280, 960), "format": "BGR888"},
        lores = {"size": (args.imgsz, args.imgsz), "format": "YUV420"},
        controls={
            # 锁帧间隔 ~30fps；AE先开着预热
            "FrameDurationLimits": (33333, 33333),
            "AeEnable": True,
            "AwbEnable": True
        }
    )
    picam2.configure(config)
    picam2.start()

    # 让 AE/AWB 收敛
    time.sleep(max(0.5, args.warmup))

    # 只锁 AE（避免亮度忽明忽暗），AWB 默认不锁以避免偏色
    md = picam2.capture_metadata()
    controls = {
        "AeEnable": False,
        "ExposureTime": args.shutter,
        "AnalogueGain": args.gain,
        # AwbEnable 继续 True：不锁白平衡最安全
        "AwbEnable": True
    }

    # 若用户强制要求锁 AWB，且元数据里拿到了有效的 ColourGains，才上锁
    if args.lock_awb:
        cg = md.get("ColourGains", None)
        if isinstance(cg, tuple) and len(cg) == 2 and cg[0] > 0.1 and cg[1] > 0.1:
            controls["AwbEnable"] = False
            controls["ColourGains"] = cg
            print(f"[i] Lock AWB with ColourGains={cg}")
        else:
            print("[i] Skip AWB lock: valid ColourGains not found in metadata")

    picam2.set_controls(controls)

    model = YOLO(args.weights)
    fps_hist, last_save = [], 0

    try:
        while True:
            t0 = time.time()
            req = picam2.capture_request()
            lo = req.make_array('lores')      # YUV420 (I420)
            main_bgr = req.make_array('main') # 已是 BGR888
            req.release()

            # lores: YUV420 → BGR（I420）
            lo_w = lo.shape[1]
            lo_h = lo.shape[0] * 2 // 3
            lo = lo.reshape((lo_h * 3 // 2, lo_w))
            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)

            # YOLO 在 lores 上推理
            r = model.predict(source=lo_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            mh, mw = main_bgr.shape[:2]
            sx = mw / float(args.imgsz)
            sy = mh / float(args.imgsz)

            det_count, low_conf = 0, False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    if (x2-x1)*(y2-y1) < args.min_area:
                        continue
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    label = model.names.get(cls_id, "battery")

                    # lores → main 映射
                    X1, Y1 = int(x1*sx), int(y1*sy)
                    X2, Y2 = int(x2*sx), int(y2*sy)
                    draw_box(main_bgr, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10):
                        low_conf = True

            infer_ms = (time.time() - t0) * 1000.0
            fps_hist.append(1000.0 / max(1.0, infer_ms))
            if len(fps_hist) > 30: fps_hist.pop(0)

            hud = f"Detections: {det_count} | {inferencia:=infer_ms:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS avg)"
            cv2.putText(main_bgr, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 回捞 hard cases（低置信或未检出）
            now = time.time()
            if (det_count == 0 or low_conf) and (now - last_save > 1.0):
                fn = f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), main_bgr)
                last_save = now

            if not args.headless:
                cv2.imshow("Battery Detection (Pi dual-stream)", main_bgr)
                # 让窗口自适应，减少拉伸造成的观感偏差
                cv2.waitKey(1)  # Esc 退出在终端处理 Ctrl+C 即可

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
