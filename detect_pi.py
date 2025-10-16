import os
import time
import argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2
# 注意：确保您的系统已安装 picamera2, ultralytics, opencv-python, 以及 ncnn 相关的依赖

def parse_args():
    """解析命令行参数"""
    ap = argparse.ArgumentParser()
    # 确保此处指向您 NCNN 转换后的模型路径 (文件夹路径)
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model",
                    help="Path to the directory containing model.ncnn.param and model.ncnn.bin")
    ap.add_argument("--imgsz", type=int, default=416, 
                    help="Inference size (lores stream size), recommended 416x416 for speed")
    ap.add_argument("--conf", type=float, default=0.30,
                    help="Object confidence threshold")
    ap.add_argument("--min_area", type=int, default=1800,
                    help="Minimum detection area (in lores size) to be considered valid")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases",
                    help="Directory to save hard case images")
    ap.add_argument("--headless", action="store_true",
                    help="Run without displaying the video window (for remote/server use)")

    # 曝光相关：锁 AE（防止 FPS 波动），AWB 默认不锁（防止偏色）
    ap.add_argument("--shutter", type=int, default=4000,
                    help="ExposureTime (us) to lock AE (short shutter to prevent motion blur)")
    ap.add_argument("--gain", type=float, default=6.0,
                    help="AnalogueGain to lock AE (compensate for short shutter)")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds to let AE/AWB settle before locking")

    # 如果你想 *明确* 锁 AWB，再加 --lock_awb 才会锁
    ap.add_argument("--lock_awb", action="store_true", help="lock AWB using current ColourGains")
    return ap.parse_args()

def ensure_dir(p):
    """确保目录存在"""
    os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label, conf):
    """在图像上绘制检测框和标签"""
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    # 1. 初始化 Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        # main 流用于显示 (1280x960 兼顾性能和清晰度)
        main={"size": (1280, 960), "format": "RGB888"},
        # lores 流用于推理 (大小由 args.imgsz 决定)
        lores={"size": (args.imgsz, args.imgsz), "format": "YUV420"},
        controls={
            "FrameDurationLimits": (33333, 33333),  # 锁定 30fps 预算 (1/30s = 33333 us)
            "AeEnable": True,                       # 初始启用 AE/AWB
            "AwbEnable": True,
            "NoiseReductionMode": 0,                # 0=Off，提高细节清晰度
            "Sharpness": 1.3                        # 略提锐度
        }
    )
    picam2.configure(config)
    picam2.start()

    # 2. 曝光/白平衡预热与锁定
    # 等待 AE/AWB 稳定
    time.sleep(max(0.5, args.warmup))

    # 锁定 AE (关闭自动曝光，设置快门和增益)
    controls = {"AeEnable": False, "ExposureTime": args.shutter, "AnalogueGain": args.gain}

    # 如果要求，锁定 AWB
    if args.lock_awb:
        md = picam2.capture_metadata()
        cg = md.get("ColourGains", (1.0, 1.0))
        # 如果获取 ColourGains 失败，cg 可能是 None，这里用默认值 (1.0, 1.0)
        if cg:
             controls.update({"AwbEnable": False, "ColourGains": cg})
        else:
             print("Warning: Failed to retrieve ColourGains. AWB lock skipped.")

    # 应用锁定控制
    picam2.set_controls(controls)

    # 3. 加载 YOLO 模型 (将自动加载 NCNN 转换的模型)
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"Error loading model from {args.weights}. Check if the folder contains 'model.ncnn.param' and 'model.ncnn.bin'.")
        print(f"Details: {e}")
        picam2.stop()
        return

    # 4. 实时推理循环
    fps_hist = []
    last_save = 0

    try:
        while True:
            t0 = time.time()

            # 捕获请求
            req = picam2.capture_request()
            # 获取 lores 流 (用于推理) - 格式: YUV420, 维度: (H*1.5, W)
            lo = req.make_array('lores')
            # 获取 main 流 (用于显示和绘制) - 格式: RGB888, 维度: (H, W, 3)
            main_rgb = req.make_array('main')
            req.release()

            # **关键修正：YUV420 转换**
            # lo 数组已经是正确的 YUV420 格式，直接转换即可，无需手动 reshape/计算高度。
            # 原代码中第 88-90 行的错误手动计算已被移除。
            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)

            # YOLO 推理（在 lores 上）
            results = model.predict(source=lo_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)

            # 准备 main 显示（RGB→BGR 便于绘制）
            main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
            mh, mw = main_bgr.shape[:2]

            # 计算 lores 到 main 的缩放比例
            sx, sy = mw / float(args.imgsz), mh / float(args.imgsz)

            det_count, low_conf = 0, False

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                r_boxes = results[0].boxes
                for b in r_boxes:
                    # 获取 lores 上的坐标
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist()) 

                    # 面积筛选
                    if (x2 - x1) * (y2 - y1) < args.min_area:
                        continue

                    conf = float(b.conf)
                    cls = int(b.cls) if b.cls is not None else 0

                    label = model.names.get(cls, f"Object {cls}")

                    # 缩放到 main 尺寸的坐标
                    X1, Y1 = int(x1 * sx), int(y1 * sy)
                    X2, Y2 = int(x2 * sx), int(y2 * sy)
                    
                    # 绘制
                    draw_box(main_bgr, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    
                    # 记录低置信度情况 (硬核案例)
                    if conf < (args.conf + 0.10):
                        low_conf = True

            # 性能计算和显示
            infer_ms = (time.time() - t0) * 1000.0
            fps = 1000.0 / max(1.0, infer_ms)
            fps_hist.append(fps)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            fps_avg = sum(fps_hist) / len(fps_hist)
            
            # 绘制 FPS 和检测数
            cv2.putText(main_bgr, 
                        f"Detections: {det_count} | {infer_ms:.1f} ms ({fps_avg:.1f} FPS avg)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 回捞 hard cases (无检测结果 或 有低置信度结果，且距离上次保存超过 1.0 秒)
            now = time.time()
            if (det_count == 0 or low_conf) and (now - last_save > 1.0):
                # 文件名格式: hard_计数_时间戳.jpg
                filename = f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, filename), main_bgr)
                last_save = now

            # 显示窗口
            if not args.headless:
                cv2.imshow("Battery Detection - Pi4 (AE/AWB Locked)", main_bgr)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 键退出
                    break

    except KeyboardInterrupt:
        print("\nExiting program...")
        pass
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # 清理资源
        picam2.stop()
        cv2.destroyAllWindows()
        print("Picamera2 stopped and windows closed.")


if __name__ == "__main__":
    main()
