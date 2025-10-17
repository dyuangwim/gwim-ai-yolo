import os, time, argparse
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model")
    ap.add_argument("--imgsz", type=int, default=256)  # 降低到256
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=800)  # 降低面积要求
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no_display", action="store_true", help="完全禁用显示相关操作")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box_fast(img, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)  # 更细的线
    if not args.no_display:
        txt = f"{label} {conf:.1f}"
        cv2.putText(img, txt, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

def main():
    # 更激进的性能设置
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["NCNN_THREADS"] = "4"
    os.environ["NCNN_VERBOSE"] = "0"
    os.environ["OPENCV_OPENCL_RUNTIME"] = ""  # 禁用OpenCL
    
    args = parse_args()
    ensure_dir(args.save_dir)

    print("Initializing camera...")
    
    # 更低的相机分辨率
    picam2 = Picamera2()
    main_w, main_h = 1024, 768  # 降低显示分辨率
    
    config = picam2.create_preview_configuration(  # 使用preview配置，更快
        main={"size": (main_w, main_h), "format": "YUV420"},
        controls={
            "AeEnable": True, 
            "AwbEnable": True, 
            "FrameRate": 30,
            "ExposureTime": 15000,  # 固定曝光减少计算
        }
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.5)  # 减少预热时间

    print("Loading model...")
    
    # 直接使用优化模型
    model_path = "/home/pi/models/optimized_model" if os.path.exists("/home/pi/models/optimized_model") else args.weights
    print(f"Using model: {model_path}")
    
    model = YOLO(model_path)
    print("Model loaded successfully!")

    # 快速预热
    print("Quick warmup...")
    warmup_frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    for _ in range(2):
        _ = model.predict(source=warmup_frame, imgsz=args.imgsz, verbose=False)
    print("Warmup completed")

    if not args.headless and not args.no_display:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", 640, 480)  # 更小的显示窗口

    fps_hist = []
    frame_count = 0
    start_time = time.time()
    
    # 缩放因子
    lh, lw = args.imgsz, args.imgsz
    mh, mw = main_h, main_w
    sx, sy = mw / lw, mh / lh

    print("Starting ultra-optimized detection loop...")
    
    try:
        while True:
            frame_start = time.time()

            # 单次捕获，用于推理和显示
            yuv_frame = picam2.capture_array("main")
            
            # 快速YUV转换
            if len(yuv_frame.shape) == 2:
                frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            else:
                frame = yuv_frame  # 如果已经是BGR
            
            # 准备推理图像 - 使用最快的方法
            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_NEAREST)

            # 推理
            predict_start = time.time()
            results = model.predict(
                source=infer_img, 
                imgsz=args.imgsz, 
                conf=args.conf, 
                verbose=False,
                max_det=3  # 限制最大检测数量
            )
            predict_time = (time.time() - predict_start) * 1000
            
            r = results[0]

            # 快速处理检测结果
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
                        draw_box_fast(frame, X1, Y1, X2, Y2, label, conf)
                        det_count += 1
                        if det_count >= 3:  # 最多处理3个检测
                            break

            # 优化的FPS计算
            frame_time = (time.time() - frame_start) * 1000
            current_fps = 1000.0 / max(frame_time, 1.0)
            fps_hist.append(current_fps)
            if len(fps_hist) > 10:  # 更短的平滑窗口
                fps_hist.pop(0)
            
            avg_fps = sum(fps_hist) / len(fps_hist)
            
            # 只在需要时添加文本
            if not args.no_display:
                info_text = f"FPS:{avg_fps:.1f} D:{det_count} I:{predict_time:.0f}ms"
                cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # 显示
            if not args.headless and not args.no_display:
                cv2.imshow("Battery Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break

            frame_count += 1
            
            # 减少输出频率
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                overall_fps = frame_count / elapsed
                print(f"Frames: {frame_count}, FPS: {overall_fps:.1f}")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        picam2.stop()
        if not args.headless and not args.no_display:
            cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        print(f"\n=== 最终性能 ===")
        print(f"总帧数: {frame_count}")
        print(f"总时间: {total_time:.1f}s")
        print(f"平均FPS: {frame_count/total_time:.1f}")

if __name__ == "__main__":
    main()
