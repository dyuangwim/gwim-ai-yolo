import os, time, argparse
from datetime import datetime
import cv2
import numpy as np
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
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    txt=f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y0), (x1+tw+4, y0+th+6), (0,255,0), -1)
    cv2.putText(img, txt, (x1+2, y0+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    # 线程设置
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("NCNN_THREADS", "4")

    args = parse_args()
    ensure_dir(args.save_dir)

    # ---- Camera 设置 ----
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    
    # 重要：尝试不同的格式
    print("Testing camera formats...")
    
    # 方案1：先尝试 BGR888
    try:
        cfg = picam2.create_video_configuration(
            main={"size": (main_w, main_h), "format": "BGR888"},
            controls={"AeEnable": True, "AwbEnable": True}
        )
        picam2.configure(cfg)
        picam2.start()
        time.sleep(2.0)
        
        # 测试捕获
        test_frame = picam2.capture_array("main")
        print(f"[TEST] BGR888 format - frame shape: {test_frame.shape}")
        cv2.imwrite("/home/pi/test_bgr.jpg", test_frame)
        
    except Exception as e:
        print(f"BGR888 failed: {e}")
        picam2.stop()
        # 方案2：尝试 RGB888
        try:
            cfg = picam2.create_video_configuration(
                main={"size": (main_w, main_h), "format": "RGB888"},
                controls={"AeEnable": True, "AwbEnable": True}
            )
            picam2.configure(cfg)
            picam2.start()
            time.sleep(2.0)
            
            test_frame = picam2.capture_array("main")
            print(f"[TEST] RGB888 format - frame shape: {test_frame.shape}")
            cv2.imwrite("/home/pi/test_rgb.jpg", test_frame)
            
        except Exception as e2:
            print(f"RGB888 also failed: {e2}")
            return

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

            # 直接捕获主图像流
            frame = picam2.capture_array("main")
            
            # 确保是3通道
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            # 保存原始图像用于调试
            cv2.imwrite("/home/pi/debug_frame_raw.jpg", frame)

            # 准备推理图像：调整大小
            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite("/home/pi/debug_infer_input.jpg", infer_img)

            # 推理
            r = model.predict(source=infer_img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # 尺度映射
            lh, lw = args.imgsz, args.imgsz
            mh, mw = frame.shape[:2]
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
                    draw_box(frame, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

            dt = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0));  fps_hist = fps_hist[-30:]
            hud = f"Det:{det_count} | {dt:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS)"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 直接显示，不进行颜色转换
            if not args.headless:
                cv2.imshow("Battery Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # hard cases
            now = time.time()
            if (det_count==0 or low_conf) and (now-last_save>1.0):
                fn=f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame)
                last_save = now

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
