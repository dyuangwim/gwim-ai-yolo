# detect_pi2.py — RPi4 + NCNN 稳定版（lores=YUV420 → BGR；可选 I420/YV12/NV12/NV21；支持锁AWB）
import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    # 新增：YUV 子格式选择
    ap.add_argument("--yuv", default="i420", choices=["i420","yv12","nv12","nv21"],
                    help="lores=YUV420 转 BGR 的方式，默认 i420；颜色不对就换 yv12/nv12/nv21 试")
    # 新增：锁白平衡
    ap.add_argument("--lock-awb", action="store_true", help="预热后锁定 AWB，避免颜色漂移")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img_bgr, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    txt=f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img_bgr, (x1, y0), (x1+tw+4, y0+th+6), (0,255,0), -1)
    cv2.putText(img_bgr, txt, (x1+2, y0+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def yuv420_to_bgr(yuv_img, code):
    if code == "i420":
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_I420)
    if code == "yv12":
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_YV12)
    if code == "nv12":
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_NV12)
    if code == "nv21":
        return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_NV21)
    # fallback
    return cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR_I420)

def main():
    # 强制设置环境变量，确保 NCNN/OpenCV 使用多线程
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("NCNN_THREADS", "4")

    args = parse_args()
    ensure_dir(args.save_dir)

    # ---- Camera ----
    picam2 = Picamera2()
    # 保持主流高分辨率用于显示和保存（BGR888）
    main_w, main_h = 1280, 960 
    # 低分辨率 lores 流用于推理（YUV420 格式，效率最高） [3]
    cfg = picam2.create_video_configuration(
        main = {"size": (main_w, main_h), "format": "BGR888"},
        lores= {"size": (args.imgsz, args.imgsz), "format": "YUV420"},
        controls={"AeEnable": True, "AwbEnable": True}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.8)  # 预热

    # 可选：锁 AWB
    if args.lock_awb:
        try:
            md = picam2.capture_metadata()
            # ColourGains 是 (R,G)；没有就不锁
            gains = md.get("ColourGains", None)
            if gains:
                picam2.set_controls({"AwbEnable": False, "ColourGains": gains})
                print(f" Locked with gains={gains}")
        except Exception as e:
            print(f" lock failed: {e}")

    # 调试保存
    if args.save_debug:
        f0 = picam2.capture_array("main")
        if f0.ndim == 3 and f0.shape[1] == 4: f0 = f0[:, :, :3]
        cv2.imwrite("/home/pi/view_debug.jpg", f0)
        print("Saved /home/pi/view_debug.jpg", f0.shape)

    # ---- Model ----
    model = YOLO(args.weights)

    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", main_w, main_h)

    last_save = 0
    fps_hist =

    try:
        while True:
            t0 = time.time()

            # 1. 帧捕获：lores YUV420 平面
            yuv = picam2.capture_array("lores")    # shape: (H*3/2, W)
            
            # 2. YUV -> BGR 转换（OpenCV格式）
            infer_bgr = yuv420_to_bgr(yuv, args.yuv)

            # --- 关键修复：颜色通道对齐 ---
            # NCNN/YOLO模型通常期望 RGB 输入。
            # 我们将 BGR 图像显式转换为 RGB 格式，用于推理。
            infer_rgb = cv2.cvtColor(infer_bgr, cv2.COLOR_BGR2RGB) # <--- 修复紫色画面的关键步骤

            # 3. 保底 resize (通常 lores 流已是正确尺寸)
            if infer_rgb.shape[:2]!= (args.imgsz, args.imgsz):
                infer_rgb = cv2.resize(infer_rgb, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            
            # 4. 捕获 main 流（用于显示/保存，是 BGR888 格式）
            frame_bgr = picam2.capture_array("main")
            if frame_bgr.ndim == 3 and frame_bgr.shape[1] == 4: frame_bgr = frame_bgr[:, :, :3]

            # 5. 推理 (使用 infer_rgb)
            # 使用 numpy 数组作为 source，而不是 BGR 图像
            r = model.predict(source=infer_rgb, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # 6. 后处理与绘制
            # lores → main 尺度映射
            lh, lw = args.imgsz, args.imgsz
            mh, mw = frame_bgr.shape[:2]
            sx, sy = mw/float(lw), mh/float(lh)

            det_count, low_conf = 0, False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    # 获取推理结果（lores 坐标）
                    x1,y1,x2,y2 = map(int, b.xyxy.tolist())
                    # 映射到主画面 (main) 坐标
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    
                    if (X2-X1)*(Y2-Y1) < args.min_area: continue
                    
                    conf = float(b.conf)
                    cls_id = int(b.cls) if b.cls is not None else 0
                    label = r.names.get(cls_id, "battery")
                    
                    # 在 main BGR 帧上绘制
                    draw_box(frame_bgr, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

            dt = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0)); fps_hist = fps_hist[-30:]
            hud = f"Det:{det_count} | {dt:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS) | yuv={args.yuv}"
            cv2.putText(frame_bgr, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # hard cases 保存
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
