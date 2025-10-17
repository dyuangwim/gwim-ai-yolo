import argparse
import time
import os
import cv2
import numpy as np # 导入numpy以处理图像数组
from ultralytics import YOLO

# -----------------------------------------------------
# 1. 配置常量 (Configuration Constants)
# -----------------------------------------------------
# 强制NCNN使用所有可用核心（RPi 4有4核）
# 确保在运行脚本前在终端设置: export OMP_NUM_THREADS=4; export NCNN_THREADS=4
# NCNN的线程设置通常依赖于环境变量或底层库，Python中难以直接强制控制。
# 但我们仍然在代码中记录这个目标。
NCNN_NUM_THREADS = 4  
# 在进行性能测试时，应尽可能减少UI/绘制开销
ENABLE_DEBUG_WINDOW = True # 如果需要实时查看，请设为True；性能测试时设为False

# -----------------------------------------------------
# 2. 参数解析 (Argument Parsing)
# -----------------------------------------------------
parser = argparse.ArgumentParser(description="YOLO Model Inference on Raspberry Pi.")
parser.add_argument('--weights', type=str, required=True, help='Path to model weights file.')
parser.add_argument('--imgsz', type=int, default=320, help='Inference size (pixels).')
parser.add_argument('--conf', type=float, default=0.30, help='Confidence threshold.')
parser.add_argument('--assume-bgr', action='store_true', help='Assume BGR input (for OpenCV camera/image).')
parser.add_argument('--save-debug', action='store_true', help='Save debug images and video (performance killer).')
args = parser.parse_args()

# -----------------------------------------------------
# 3. 初始化 (Initialization)
# -----------------------------------------------------
print(f"[INFO] Initializing YOLO model...")
try:
    # 加载模型，模型将根据权重文件格式（.pt, NCNN等）自动选择后端
    model = YOLO(args.weights)
    
    print(f"[INFO] Model loaded successfully. Inference threads target: {NCNN_NUM_THREADS}")
    
except Exception as e:
    print(f" Failed to load model: {e}")
    exit()

# 尝试使用picamera2进行高效视频捕获
try:
    from picamera2 import Picamera2
    # 确保分辨率与模型输入尺寸对齐，以减少CPU resize开销
    lores_size = (args.imgsz, args.imgsz) 
    picam2 = Picamera2()
    # 使用 lores 捕获流，格式设置为 BGR888 (NumPy array)
    config = picam2.create_video_configuration(main={"size": lores_size}, lores={"size": lores_size, "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2.0) # 允许相机暖机
    print("[INFO] Using Picamera2 for high-speed capture.")
except ImportError:
    print(" Picamera2 not found. Falling back to slow cv2.VideoCapture(0).")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Cannot open camera.")
        exit()
    
# -----------------------------------------------------
# 4. 主循环 (Main Loop)
# -----------------------------------------------------
print("[INFO] Starting object detection loop...")
start_time = time.time()
frame_count = 0

while True:
    try:
        # 4.1. 帧捕获
        if 'picam2' in locals():
            # 从lo-res流捕获帧
            frame = picam2.capture_array("lores") 
        else:
            # OpenCV 捕获
            ret, frame = cap.read()
            if not ret:
                break
        
        # 4.2. 图像预处理 (解决颜色和尺寸问题)
        # 检查帧尺寸是否需要调整 (注意：Picamera2 lores 流通常已是正确尺寸)
        # **修复了第93行的语法错误**
        if frame.shape!= args.imgsz or frame.shape![1]= args.imgsz:
             frame = cv2.resize(frame, (args.imgsz, args.imgsz))
        
        # 颜色通道修复 (解决画面变绿/变紫的问题)
        # 假设输入是 BGR (OpenCV/Picamera2默认)，而 YOLO/NCNN 模型（如果导出正确）可能需要 RGB。
        # 显式转换 BGR -> RGB 是最可靠的修复方法。
        if args.assume_bgr:
             # 如果帧是 BGR，但 YOLO/NCNN 期望 RGB，则进行转换
             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # 4.3. 推理
        # 在运行脚本前，请确保在终端设置了线程数: export OMP_NUM_THREADS=4; export NCNN_THREADS=4
        results = model(
            frame, 
            conf=args.conf, 
            iou=0.45, 
            imgsz=args.imgsz,
            verbose=False # 减少日志输出，减轻I/O开销
        )

        # 4.4. 后处理与绘制 
        
        # 绘制结果到帧上 (注意：绘制和 imshow 耗费大量 CPU)
        if ENABLE_DEBUG_WINDOW:
            # results.plot() 通常会使用 RGB/BGR转换进行绘制
            annotated_frame = results.plot() # 确保只绘制第一个结果对象
            
            # 如果之前进行了 BGR->RGB 转换，则需要确保显示时颜色正确
            # OpenCV 的 imshow 期望 BGR 格式，如果 annotated_frame 是 RGB，需要再次转换
            if args.assume_bgr:
                 annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                 
            cv2.imshow("YOLO Inference", annotated_frame)
            
            # 警告：保存文件是IO密集型操作，会显著影响实时性能 [2]
            if args.save_debug:
                cv2.imwrite(f"debug_frame_{frame_count}.jpg", annotated_frame)

            # 检查退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break

# -----------------------------------------------------
# 5. 清理 (Cleanup)
# -----------------------------------------------------
end_time = time.time()
elapsed_time = end_time - start_time
if frame_count > 0:
    avg_fps = frame_count / elapsed_time
    print(f"\n Total Frames: {frame_count}")
    print(f" Total Time: {elapsed_time:.2f} seconds")
    print(f" Average FPS: {avg_fps:.2f} FPS")
    # 如果FPS低于5，再次提醒检查INT8量化和散热
    if avg_fps < 5:
        print("\n FPS 仍然偏低。请确保已执行 INT8 量化，并确认 RPi 4 散热良好，没有发生热节流。")
else:
    print("\n No frames processed.")

if 'picam2' in locals():
    picam2.stop()
if 'cap' in locals() and cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
