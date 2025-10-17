import os, time, argparse
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model")  # 默认用原模型
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1000)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--use-optimized", action="store_true", help="Use optimized model if available")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box_fast(img, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    txt = f"{label} {conf:.1f}"
    cv2.putText(img, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["NCNN_THREADS"] = "4"
    os.environ["NCNN_VERBOSE"] = "0"
    
    args = parse_args()
    ensure_dir(args.save_dir)

    print("Initializing camera...")
    
    # 相机配置
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    
    config = picam2.create_video_configuration(
        main={"size": (main_w, main_h), "format": "YUV420"},
        lores={"size": (args.imgsz, args.imgsz), "format": "YUV420"},
        controls={"AeEnable": True, "AwbEnable": True, "FrameRate": 30}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)

    print("Loading model...")
    
    # 智能选择模型
    optimized_path = "/home/pi/models/optimized_model"
    original_path = "/home/pi/models/best_ncnn_model"
    
    if args.use_optimized and os.path.exists(optimized_path):
        print("Using optimized model...")
        model = YOLO(optimized_path)
    else:
        print("Using original model...")
        model = YOLO(original_path)
    
    print("Model loaded successfully!")

    # 预热
    print("Warming up model...")
    warmup_frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    _ = model.predict(source=warmup_frame, imgsz=args.imgsz, verbose=False)
    print("Warmup completed")

    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", 800, 600)

    fps_hist = []
    frame_count = 0
    start_time = time.time()
    
    # 缩放因子
    lh, lw = args.imgsz, args.imgsz
    mh, mw = main_h, main_w
    sx, sy = mw / lw, mh / lh

    print("Starting detection loop...")
    
    try:
        while True:
            frame_start = time.time()

            # 使用 lores 流进行推理
            yuv_lores = picam2.capture_array("lores")
            infer_bgr = cv2.cvtColor(yuv_lores, cv2.COLOR_YUV2BGR_I420)
            
            # 捕获主流用于显示
            main_frame = picam2.capture_array("main")
            if main_frame.ndim == 3 and main_frame.shape[2] == 4:
                main_frame = main_frame[:, :, :3]
            elif len(main_frame.shape) == 2:
                main_frame = cv2.cvtColor(main_frame, cv2.COLOR_YUV2BGR_I420)

            # 推理
            predict_start = time.time()
            results = model.predict(source=infer_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)
            predict_time = (time.time() - predict_start) * 1000
            
            r = results[0]

            # 处理检测结果
            det_count = 0
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                    
                    area = (X2 - X1) * (Y2 - Y1)
                    if area >= args.min_area:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = r.names[cls_id]
                        draw_box_fast(main_frame, X1, Y1, X2, Y2, label, conf)
                        det_count += 1

            # 计算FPS
            frame_time = (time.time() - frame_start) * 1000
            current_fps = 1000.0 / max(frame_time, 1.0)
            fps_hist.append(current_fps)
            if len(fps_hist) > 20:
                fps_hist.pop(0)
            
            avg_fps = sum(fps_hist) / len(fps_hist)
            
            # 显示信息
            info_text = f"FPS: {avg_fps:.1f} | Det: {det_count} | Infer: {predict_time:.1f}ms"
            cv2.putText(main_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示
            if not args.headless:
                cv2.imshow("Battery Detection", main_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            frame_count += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                overall_fps = frame_count / elapsed
                print(f"Frame {frame_count}: Current FPS: {avg_fps:.1f}, Overall FPS: {overall_fps:.1f}")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"\n=== Performance Summary ===")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {frame_count/total_time:.1f}")

if __name__ == "__main__":
    main()
