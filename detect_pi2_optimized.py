import os, time, argparse
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/optimized_model")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1000)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box_fast(img, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
    txt = f"{label} {conf:.1f}"
    cv2.putText(img, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

def main():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["NCNN_THREADS"] = "4"
    os.environ["NCNN_VERBOSE"] = "0"
    
    args = parse_args()
    ensure_dir(args.save_dir)

    # 相机配置
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1024, 768), "format": "YUV420"},
        controls={"AeEnable": True, "AwbEnable": True, "FrameRate": 30}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0)

    # 尝试加载优化模型，如果失败则回退到原模型
    try:
        print("Loading optimized model...")
        model = YOLO(args.weights)
        print("Optimized model loaded successfully!")
    except Exception as e:
        print(f"Optimized model failed: {e}")
        print("Falling back to original model...")
        model = YOLO("/home/pi/models/best_ncnn_model")
    
    # 预热
    warmup_img = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    _ = model.predict(source=warmup_img, imgsz=args.imgsz, verbose=False)

    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", 640, 480)

    fps_hist = []
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            t0 = time.time()

            # 捕获和处理帧
            yuv_frame = picam2.capture_array("main")
            if len(yuv_frame.shape) == 2:
                frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            else:
                frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)

            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_NEAREST)
            
            # 推理
            r = model.predict(source=infer_img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # 处理检测结果
            det_count = 0
            if r.boxes is not None:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = int(x1*4), int(y1*4), int(x2*4), int(y2*4)  # 简化缩放
                    if (X2-X1)*(Y2-Y1) >= args.min_area:
                        conf = float(b.conf[0])
                        label = r.names.get(int(b.cls[0]), "battery")
                        draw_box_fast(frame, X1, Y1, X2, Y2, label, conf)
                        det_count += 1

            # FPS 显示
            fps = 1000.0 / max((time.time()-t0)*1000.0, 1.0)
            fps_hist.append(fps)
            if len(fps_hist) > 20:
                fps_hist.pop(0)
            
            avg_fps = sum(fps_hist) / len(fps_hist)
            cv2.putText(frame, f"FPS:{avg_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            if not args.headless:
                cv2.imshow("Battery Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames, Current FPS: {avg_fps:.1f}")

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"\n最终性能: {frame_count} frames, {total_time:.1f}s, 平均FPS: {frame_count/total_time:.1f}")

if __name__ == "__main__":
    main()
