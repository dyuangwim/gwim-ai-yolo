import os, time, argparse
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model")
    ap.add_argument("--imgsz", type=int, default=192)  # 大幅降低推理尺寸
    ap.add_argument("--conf", type=float, default=0.25)  # 稍低置信度
    ap.add_argument("--min_area", type=int, default=500)  # 更低的面积要求
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no_save", action="store_true", help="不保存图片提升性能")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box_minimal(img, x1, y1, x2, y2):
    """最简绘制，只画框不写文字"""
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)

def main():
    # 性能优化设置
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["NCNN_THREADS"] = "4" 
    os.environ["NCNN_VERBOSE"] = "0"
    
    args = parse_args()
    if not args.no_save:
        ensure_dir(args.save_dir)

    print("🚀 启动超优化版本...")
    
    # 相机配置 - 最小化分辨率
    picam2 = Picamera2()
    display_w, display_h = 640, 480  # 最小显示分辨率
    
    config = picam2.create_preview_configuration(
        main={"size": (display_w, display_h), "format": "YUV420"},  # 显示和推理用同一分辨率
        controls={
            "AeEnable": True, 
            "AwbEnable": False,  # 禁用自动白平衡
            "FrameRate": 30,
        }
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # 最小预热

    print("📦 加载模型...")
    # 使用原模型，确保稳定
    model = YOLO(args.weights)
    print("✅ 模型加载成功!")

    # 极速预热
    print("🔥 快速预热...")
    warmup_img = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    _ = model.predict(source=warmup_img, imgsz=args.imgsz, verbose=False)
    print("✅ 预热完成")

    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", display_w, display_h)

    # 性能跟踪
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps_history = []

    print("🎯 开始检测循环...")
    
    try:
        while True:
            frame_start = time.perf_counter()

            # 🚀 单次捕获 + 快速处理
            yuv_frame = picam2.capture_array("main")
            
            # 快速YUV转BGR
            if len(yuv_frame.shape) == 2:
                display_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            else:
                display_frame = yuv_frame
            
            # 🚀 推理图像准备 - 使用同一帧，避免重复捕获
            infer_img = cv2.resize(display_frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_NEAREST)

            # 🚀 推理
            predict_start = time.perf_counter()
            results = model.predict(
                source=infer_img, 
                imgsz=args.imgsz, 
                conf=args.conf, 
                verbose=False,
                max_det=2,  # 限制检测数量
                half=False   # 禁用半精度
            )
            predict_time = (time.perf_counter() - predict_start) * 1000
            
            r = results[0]

            # 🚀 快速检测处理
            det_count = 0
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # 直接缩放坐标 (192->640)
                    scale = display_w / args.imgsz
                    X1, Y1, X2, Y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                    
                    area = (X2 - X1) * (Y2 - Y1)
                    if area >= args.min_area:
                        draw_box_minimal(display_frame, X1, Y1, X2, Y2)
                        det_count += 1
                        if det_count >= 2:  # 最多2个检测
                            break

            # 🚀 极简FPS计算
            frame_time = (time.perf_counter() - frame_start) * 1000
            current_fps = 1000.0 / max(frame_time, 1.0)
            fps_history.append(current_fps)
            if len(fps_history) > 5:  # 极短窗口
                fps_history.pop(0)
            
            avg_fps = sum(fps_history) / len(fps_history)

            # 🚀 最小化显示开销
            if not args.headless:
                cv2.imshow("Battery Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_count += 1
            
            # 🚀 减少性能输出
            current_time = time.time()
            if current_time - last_fps_time >= 5.0:  # 每5秒输出一次
                elapsed = current_time - start_time
                overall_fps = frame_count / elapsed
                print(f"📊 帧数: {frame_count}, 实时FPS: {avg_fps:.1f}, 平均FPS: {overall_fps:.1f}, 推理: {predict_time:.1f}ms")
                last_fps_time = current_time

            # 🚀 可选保存
            if not args.no_save and det_count == 0 and (time.time() - start_time) % 10 < 0.1:
                fn = f"hard_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), display_frame)

    except KeyboardInterrupt:
        print("\n🛑 用户停止")
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        
        # 最终报告
        total_time = time.time() - start_time
        final_fps = frame_count / total_time
        print(f"\n🎉 最终性能报告:")
        print(f"   总帧数: {frame_count}")
        print(f"   总时间: {total_time:.1f}秒") 
        print(f"   平均FPS: {final_fps:.1f}")
        
        if final_fps >= 10:
            print("   🚀 优秀性能!")
        elif final_fps >= 7:
            print("   ✅ 良好性能!")
        else:
            print("   ⚠️  需要进一步优化")

if __name__ == "__main__":
    main()
