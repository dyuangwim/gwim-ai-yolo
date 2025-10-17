import os, time, argparse
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import threading

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/optimized_model")  # 使用优化后的模型
    ap.add_argument("--imgsz", type=int, default=256)  # 降低推理尺寸
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1000)  # 调整面积阈值
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true", default=True)
    ap.add_argument("--display_size", type=int, default=640)  # 降低显示分辨率
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box_fast(img, x1, y1, x2, y2, label, conf):
    """快速绘制边界框"""
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)  #  thinner border
    txt = f"{label} {conf:.1f}"
    cv2.putText(img, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

class CameraBuffer:
    """相机缓冲区管理，减少等待时间"""
    def __init__(self, picam2):
        self.picam2 = picam2
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        
    def start_capture(self):
        def capture_loop():
            while self.running:
                try:
                    yuv_frame = self.picam2.capture_array("main")
                    # 快速 YUV 转 BGR
                    if len(yuv_frame.shape) == 2 and yuv_frame.shape[0] == self.picam2.camera_config['main']['size'][1] * 3 // 2:
                        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
                    else:
                        bgr_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_NV12)
                    
                    with self.lock:
                        self.frame = bgr_frame
                except Exception as e:
                    print(f"Capture error: {e}")
                    time.sleep(0.01)
        
        self.thread = threading.Thread(target=capture_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

def main():
    # === 性能优化设置 ===
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["NCNN_THREADS"] = "4" 
    os.environ["NCNN_VERBOSE"] = "0"
    
    # 设置进程优先级
    os.nice(-10)  # 提高进程优先级
    
    args = parse_args()
    ensure_dir(args.save_dir)

    # === 相机配置优化 ===
    picam2 = Picamera2()
    
    # 根据显示参数调整分辨率
    display_size = (args.display_size, int(args.display_size * 0.75)) if args.display_size < 800 else (640, 480)
    main_size = (1024, 768)  # 适中的主分辨率
    
    # 使用预览配置 + 性能优化参数
    config = picam2.create_preview_configuration(
        main={"size": main_size, "format": "YUV420"},
        controls={
            "AeEnable": True, 
            "AwbEnable": True, 
            "FrameRate": 30,
            "ExposureTime": 10000,  # 固定曝光时间
            "AnalogueGain": 1.0,    # 固定模拟增益
        }
    )
    
    picam2.configure(config)
    picam2.set_controls({"ExposureTime": 10000, "AnalogueGain": 1.0})
    picam2.start()
    
    # 使用相机缓冲区
    cam_buffer = CameraBuffer(picam2)
    cam_buffer.start_capture()
    time.sleep(2.0)  # 预热

    # === 模型预热 ===
    print("Loading and warming up model...")
    model = YOLO(args.weights)
    
    # 更彻底的预热
    warmup_data = [np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8) for _ in range(5)]
    for warmup_img in warmup_data:
        _ = model.predict(source=warmup_img, imgsz=args.imgsz, verbose=False)
    print("Model warmup completed")

    # === 显示设置 ===
    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", display_size[0], display_size[1])

    # === 性能监控 ===
    last_save = 0
    fps_hist = []
    frame_count = 0
    start_time = time.time()
    
    # 预计算缩放因子
    lh, lw = args.imgsz, args.imgsz
    mh, mw = main_size[1], main_size[0]
    sx, sy = mw/float(lw), mh/float(lh)

    print("Starting main loop...")
    
    try:
        while True:
            frame_start = time.time()

            # 从缓冲区获取帧（非阻塞）
            frame = cam_buffer.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # 准备推理图像 - 使用最快速的调整大小方法
            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_NEAREST)
            
            # 推理
            predict_start = time.time()
            r = model.predict(source=infer_img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            predict_time = (time.time() - predict_start) * 1000

            # 快速检测处理
            det_count = 0
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    area = (X2-X1)*(Y2-Y1)
                    if area < args.min_area:
                        continue
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    label = r.names.get(cls_id, "battery")
                    draw_box_fast(frame, X1, Y1, X2, Y2, label, conf)
                    det_count += 1

            # FPS 计算和显示
            frame_time = (time.time() - frame_start) * 1000
            current_fps = 1000.0 / max(frame_time, 1.0)
            fps_hist.append(current_fps)
            if len(fps_hist) > 20:  # 更短的平滑窗口
                fps_hist.pop(0)
            
            avg_fps = sum(fps_hist) / len(fps_hist)
            
            # 简化的 HUD
            hud_text = f"FPS:{avg_fps:.1f} | Det:{det_count} | Infer:{predict_time:.1f}ms"
            cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 显示
            if not args.headless:
                display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Battery Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_count += 1
            
            # 性能报告
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                print(f"Frames: {frame_count}, Avg FPS: {current_fps:.1f}, Latest: {avg_fps:.1f} FPS")

            # 优化的保存逻辑
            now = time.time()
            if det_count == 0 and (now - last_save) > 2.0:  # 减少保存频率
                fn = f"hard_{datetime.now().strftime('%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame)
                last_save = now

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cam_buffer.stop()
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        
        # 最终报告
        total_time = time.time() - start_time
        print(f"\n=== 最终性能报告 ===")
        print(f"总帧数: {frame_count}")
        print(f"总时间: {total_time:.1f}s") 
        print(f"平均 FPS: {frame_count/total_time:.1f}")

if __name__ == "__main__":
    main()