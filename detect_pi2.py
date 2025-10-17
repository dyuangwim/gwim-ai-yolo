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
    ap.add_argument("--preview", action="store_true", help="Use preview configuration for better performance")
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
    # 性能优化：设置线程和环境变量
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["NCNN_THREADS"] = "4"
    # 禁用调试输出以提高性能
    os.environ["NCNN_VERBOSE"] = "0"
    
    args = parse_args()
    ensure_dir(args.save_dir)

    # ---- Camera 设置 - 性能优化 ----
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    
    # 使用预览配置提高性能
    if args.preview:
        print("Using preview configuration for better performance...")
        config = picam2.create_preview_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"AeEnable": True, "AwbEnable": True, "FrameRate": 30}
        )
    else:
        config = picam2.create_video_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"AeEnable": True, "AwbEnable": True, "FrameRate": 30}
        )
    
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)  # 预热

    # ---- Model 预热 ----
    print("Warming up model...")
    model = YOLO(args.weights)
    # 预热推理
    warmup_frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        _ = model.predict(source=warmup_frame, imgsz=args.imgsz, verbose=False)
    print("Model warmup completed")

    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", main_w, main_h)

    last_save = 0
    fps_hist = []
    frame_count = 0
    start_time = time.time()

    # 预计算缩放因子
    lh, lw = args.imgsz, args.imgsz
    mh, mw = main_h, main_w
    sx, sy = mw/float(lw), mh/float(lh)

    try:
        while True:
            t0 = time.time()

            # 性能优化：减少不必要的操作
            yuv_frame = picam2.capture_array("main")
            
            # YUV 转 BGR - 使用最快的转换方法
            try:
                if len(yuv_frame.shape) == 2 and yuv_frame.shape[0] == main_h * 3 // 2:
                    frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                else:
                    frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)
            except:
                frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

            # 准备推理图像 - 使用较快的插值方法
            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            
            # 推理
            r = model.predict(source=infer_img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # 检测处理
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

            # FPS 计算优化
            frame_count += 1
            dt = (time.time()-t0)*1000.0
            current_fps = 1000.0 / max(dt, 1.0)
            fps_hist.append(current_fps)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            
            avg_fps = sum(fps_hist) / len(fps_hist)
            hud = f"Det:{det_count} | {dt:.1f}ms | {avg_fps:.1f}FPS"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 显示
            if not args.headless:
                cv2.imshow("Battery Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # 每100帧打印一次性能统计
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames in {elapsed:.1f}s, Average FPS: {frame_count/elapsed:.1f}")

            # hard cases - 优化保存逻辑
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
        
        # 最终性能报告
        total_time = time.time() - start_time
        print(f"\n=== Performance Summary ===")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {frame_count/total_time:.1f}")

if __name__ == "__main__":
    main()
