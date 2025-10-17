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

def fix_color_channels(img):
    """
    修复颜色通道问题
    紫色通常意味着 B 和 R 通道交换了
    """
    # 方案1: 直接交换 R 和 B 通道
    img_fixed = img.copy()
    img_fixed[:, :, 0] = img[:, :, 2]  # B <- R
    img_fixed[:, :, 2] = img[:, :, 0]  # R <- B
    return img_fixed

def main():
    # 线程设置
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("NCNN_THREADS", "4")

    args = parse_args()
    ensure_dir(args.save_dir)

    # ---- Camera 设置 ----
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    
    print("Testing different camera configurations...")
    
    # 方案1: 使用 YUV420 然后转换
    try:
        print("Trying YUV420 + manual conversion...")
        cfg = picam2.create_video_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"AeEnable": True, "AwbEnable": True}
        )
        picam2.configure(cfg)
        picam2.start()
        time.sleep(2.0)
        
        # 捕获 YUV 并手动转换
        yuv_frame = picam2.capture_array("main")
        print(f"YUV frame shape: {yuv_frame.shape}")
        
        # 尝试不同的 YUV 转换
        try:
            # 方法1: NV12 转换
            if len(yuv_frame.shape) == 2:  # 单通道 YUV
                h, w = yuv_frame.shape
                if h == main_h * 3 // 2:  # YUV420 格式
                    bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                    print("Used YUV2BGR_I420 conversion")
                else:
                    bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)
                    print("Used YUV2BGR_NV12 conversion")
            else:
                bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                print("Used YUV2BGR_I420 conversion (multi-channel)")
                
        except Exception as e:
            print(f"YUV conversion failed: {e}")
            # 尝试其他转换方法
            try:
                bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV21)
                print("Used YUV2BGR_NV21 conversion")
            except:
                try:
                    bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_YV12)
                    print("Used YUV2BGR_YV12 conversion")
                except:
                    print("All YUV conversions failed")
                    bgr_frame = np.zeros((main_h, main_w, 3), dtype=np.uint8)
        
        cv2.imwrite("/home/pi/test_yuv_converted.jpg", bgr_frame)
        print("Saved YUV converted image")
        
    except Exception as e:
        print(f"YUV420 failed: {e}")
        picam2.stop()
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

            # 捕获 YUV 帧并转换
            yuv_frame = picam2.capture_array("main")
            
            # YUV 转 BGR
            try:
                if len(yuv_frame.shape) == 2:
                    h, w = yuv_frame.shape
                    if h == main_h * 3 // 2:
                        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                    else:
                        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)
                else:
                    frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            except:
                try:
                    frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV21)
                except:
                    frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_YV12)
            
            # 如果颜色还是紫色，强制修复
            if frame.mean(axis=2).mean() > 0:  # 确保不是全黑图像
                # 检查是否是紫色（B和R通道交换）
                b_mean = frame[:,:,0].mean()
                r_mean = frame[:,:,2].mean()
                if r_mean > b_mean * 1.5:  # 如果红色平均值远大于蓝色，可能是通道交换
                    print("Detected possible B/R channel swap, fixing...")
                    frame = fix_color_channels(frame)
            
            # 保存调试图像
            cv2.imwrite("/home/pi/debug_current_frame.jpg", frame)

            # 准备推理图像
            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            
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
            fps_hist.append(1000.0/max(dt,1.0))
            fps_hist = fps_hist[-30:]
            hud = f"Det:{det_count} | {dt:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS)"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 显示
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
