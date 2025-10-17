import argparse
import time
import os
import cv2
from ultralytics import YOLO

# -----------------------------------------------------
# 1. 配置常量 (Configuration Constants)
# -----------------------------------------------------
# 强制NCNN使用所有可用核心（RPi 4有4核）
# 这是性能提升的关键！
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
    # 加载模型，并设置关键的NCNN参数
    # 如果是NCNN模型，它会尝试使用ncnn后端
    model = YOLO(args.weights)
    
    # 强制设置推理线程数
    # 这一步对于RPi 4的并行计算至关重要
    print(f"[INFO] Setting inference threads to: {NCNN_NUM_THREADS}")
    
    # Ultralytics V8+ NCNN的线程设置通常在推理时通过cfg参数传入
    # 或者通过环境变量：os.environ = str(NCNN_NUM_THREADS)
    # 对于YOLO V8/V11，如果模型是NCNN格式，线程设置应尝试通过API控制
    
except Exception as e:
    print(f" Failed to load model: {e}")
    exit()

# 尝试使用picamera2进行高效视频捕获
# 确保您已安装picamera2库，这是RPi上最高效的摄像头库 [5]
try:
    from picamera2 import Picamera2
    # 确保分辨率与模型输入尺寸对齐，以减少CPU resize开销
    lores_size = (args.imgsz, args.imgsz) 
    picam2 = Picamera2()
    # 避免使用 create_still_configuration，因为它不是为高速视频流设计的 [6]
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
            # 从lo-res流捕获帧，格式已设置为BGR888 (NumPy array)
            # 这比使用 cv2.VideoCapture 更高效 [5]
            frame = picam2.capture_array("lores") 
        else:
            # OpenCV 捕获 (较慢)
            ret, frame = cap.read()
            if not ret:
                break
        
        # 4.2. 图像预处理 (NCNN 模型对 BGR/RGB 敏感)
        # 保持 --assume-bgr 标志：这意味着您的 camera/cv2 输出的是 BGR 格式
        # 如果模型需要 RGB，Ultralytics会在内部处理颜色通道转换
        if args.assume_bgr:
            # 确保帧尺寸与模型期望的 imgsz 对齐
            if frame.shape!= args.imgsz or frame.shape![1]= args.imgsz:
                 frame = cv2.resize(frame, (args.imgsz, args.imgsz))
        
        # 4.3. 推理
        # 设置线程数（通过传入args或直接设置）
        # 对于Ultralytics API，NCNN的线程设置通常是自动的或通过ENV变量控制
        # 如果NCNN的线程设置有问题，可以尝试使用 os.environ
        results = model(
            frame, 
            conf=args.conf, 
            iou=0.45, 
            imgsz=args.imgsz, 
            # 如果 Ultralytics NCNN 部署需要明确的线程设置，可以尝试添加 cfg={} 参数
            # 但在当前API中，通常依赖于环境变量或默认值
            # 暂时不添加，依赖 ENV 变量或 NCNN 默认多核运行
        )

        # 4.4. 后处理与绘制 (高开销区域)
        
        # 绘制结果到帧上 (这一步很耗CPU，会拉低FPS)
        if ENABLE_DEBUG_WINDOW:
            annotated_frame = results.plot()
            cv2.imshow("YOLO Inference", annotated_frame)
            
            # 保存调试图像（如果您开启了 --save-debug 旗标）
            if args.save_debug:
                # 警告：保存文件是IO密集型操作，会显著影响实时性能 [7]
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
else:
    print("\n No frames processed.")

if 'picam2' in locals():
    picam2.stop()
if 'cap' in locals() and cap.isOpened():
    cap.release()
cv2.destroyAllWindows()