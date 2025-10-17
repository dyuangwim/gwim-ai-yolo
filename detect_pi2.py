# detect_pi2.py — RPi4 + NCNN 稳定版（main=RGB888 显示；lores=YUV420→RGB 推理）
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
    # YUV 子格式（默认 i420，颜色不对可改 yv12/nv12/nv21）
    ap.add_argument("--yuv", default="i420", choices=["i420","yv12","nv12","nv21"])
    # 极端兜底：如果仍有偏紫，可再交换 R/B
    ap.add_argument("--swap-rb", action="store_true", help="显示与推理都交换 R/B 通道")
    # 预热后锁白平衡，减少色偏漂移
    ap.add_argument("--lock-awb", action="store_true")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def yuv420_to_rgb(yuv_img, mode):
    if mode == "i420":
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)
    if mode == "yv12":
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_YV12)
    if mode == "nv12":
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_NV12)
    if mode == "nv21":
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_NV21)
    return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)

def draw_box(img_bgr, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    txt=f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img_bgr, (x1, y0), (x1+tw+4, y0+th+6), (0,255,0), -1)
    cv2.putText(img_bgr, txt, (x1+2, y0+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    # 线程上限，避免争用
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("NCNN_THREADS", "4")

    args = parse_args()
    ensure_dir(args.save_dir)

    # ---- Camera：main=RGB888（用于显示）；lores=YUV420（用于推理） ----
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    cfg = picam2.create_video_configuration(
        main = {"size": (main_w, main_h), "format": "RGB888"},
        lores= {"size": (args.imgsz, args.imgsz), "format": "YUV420"},
        controls={"AeEnable": True, "AwbEnable": True}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.8)

    # 可选：锁 AWB，避免色温漂移
    if args.lock_awb:
        try:
            md = picam2.capture_metadata()
            gains = md.get("ColourGains", None)
            if gains:
                picam2.set_controls({"AwbEnable": False, "ColourGains": gains})
                print(f"[AWB] Locked gains={gains}")
        except Exception as e:
            print(f"[AWB] lock failed: {e}")

    if args.save_debug:
        f0 = picam2.capture_array("main")   # RGB
        if f0.ndim == 3 and f0.shape[2] == 4: f0 = f0[:, :, :3]
        # 保存为 BGR 以便系统看图工具正确显示
        view0 = cv2.cvtColor(f0, cv2.COLOR_RGB2BGR)
        cv2.imwrite("/home/pi/view_debug.jpg", view0)
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

            # lores：YUV420 平面 → RGB（给模型）
            yuv = picam2.capture_array("lores")   # shape: (H*3/2, W)
            infer_rgb = yuv420_to_rgb(yuv, args.yuv)
            if infer_rgb.shape[:2] != (args.imgsz, args.imgsz):
                infer_rgb = cv2.resize(infer_rgb, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)

            # 若需要兜底交换 R/B（极个别相机栈）
            if args.swap-rb:
                infer_rgb = infer_rgb[..., ::-1]  # RGB<->BGR

            # main：RGB → 显示前转 BGR
            main_rgb = picam2.capture_array("main")    # (H, W, 3) RGB
            if main_rgb.ndim == 3 and main_rgb.shape[2] == 4: main_rgb = main_rgb[:, :, :3]
            frame_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)

            # 推理（模型内部按 RGB 处理最稳）
            r = model.predict(source=infer_rgb, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # lores → main 尺度映射
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
            hud = f"Det:{det_count} | {dt:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS) | yuv={args.yuv}"
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
