import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    # 确保此处指向您 NCNN 转换后的模型路径
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model") 
    ap.add_argument("--imgsz", type=int, default=416)       # 建议使用 416x416 加速推理 
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")

    # 曝光相关：我们锁 AE（防止 FPS 波动），AWB 默认不锁（防止偏色）
    # 短快门（4000 us 或更短）以防止拖影，提高清晰度
    ap.add_argument("--shutter", type=int, default=4000,    
                    help="ExposureTime (us) to lock AE")
    # 增加增益来弥补短快门损失的亮度
    ap.add_argument("--gain", type=float, default=6.0,      
                    help="AnalogueGain to lock AE")
    ap.add_argument("--warmup", type=float, default=1.0,    # 让 AE/AWB 先稳定 
                    help="seconds to let AE/AWB settle")

    # 如果你想 *明确* 锁 AWB，再加 --lock_awb 才会锁
    ap.add_argument("--lock_awb", action="store_true", help="lock AWB using current ColourGains")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    txt=f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), (0,255,0), -1)
    cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main  = {"size": (1280, 960), "format": "RGB888"},   # 显示用
        lores = {"size": (args.imgsz, args.imgsz), "format": "YUV420"},  # 推理用 
        controls={
            "FrameDurationLimits": (33333, 33333),  # 锁定 30fps 预算 
            "AeEnable": True,                       # 先开着 AWB/AE 
            "AwbEnable": True,                      
            "NoiseReductionMode": 0,                # 0=Off，提高细节清晰度
            "Sharpness": 1.3                        # 略提锐度
        }
    )
    picam2.configure(config)
    picam2.start()

    # 让 AE/AWB 先收敛
    time.sleep(max(0.5, args.warmup)) 

    # 锁 AE（曝光），AWB 默认继续开着；若传了 --lock_awb 才锁
    controls = {"AeEnable": False, "ExposureTime": args.shutter, "AnalogueGain": args.gain}
    
    if args.lock_awb:
        md = picam2.capture_metadata()
        cg = md.get("ColourGains", (1.0, 1.0)) # 读取收敛后的颜色增益 
        controls.update({"AwbEnable": False, "ColourGains": cg})
        
    picam2.set_controls(controls) # 设定锁定的控制参数 

    # 加载模型
    model = YOLO(args.weights)
    fps_hist, last_save =, 0 # <<< 修复：删除多余的逗号

    try:
        while True:
            t0 = time.time()
            
            # 一次取两路
            req = picam2.capture_request()
            lo = req.make_array('lores')    # YUV420: (h*3/2, w)
            main_rgb = req.make_array('main')
            req.release()

            # CPU 密集型：YUV420 -> BGR（I420） - 仍是 Pi 4 的性能瓶颈 
            w = lo.shape[1]; h = lo.shape * 2 // 3 # 修复：确保 lo.shape 是正确的维度
            lo = lo.reshape((h * 3 // 2, w))
            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)

            # YOLO 推理（在 lores 上）
            # 注意: 如果使用的是 NCNN 模型，model.predict 方法可能会要求输入尺寸匹配 
            # 这里的 imgsz=args.imgsz (416) 应该与 lores 流的尺寸匹配。
            r = model.predict(source=lo_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False) 
            
            # 准备 main 显示（RGB→BGR 便于绘制）
            main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
            mh, mw = main_bgr.shape[:2]
            sx, sy = mw/float(args.imgsz), mh/float(args.imgsz)

            det_count, low_conf = 0, False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    # 确保 xyxy 是列表或数组，并且取第一个元素
                    x1,y1,x2,y2 = map(int, b.xyxy.tolist()) 
                    if (x2-x1)*(y2-y1) < args.min_area: continue
                    conf = float(b.conf) # 确保 conf 取第一个元素
                    cls = int(b.cls) if b.cls is not None else 0 # 确保 cls 取第一个元素
                    label = model.names.get(cls, "battery")
                    X1,Y1,X2,Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    draw_box(main_bgr, X1,Y1,X2,Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

            infer_ms = (time.time() - t0) * 1000.0
            fps_hist.append(1000.0 / max(1.0, infer_ms))
            if len(fps_hist) > 30: fps_hist.pop(0)
            fps_avg = sum(fps_hist)/len(fps_hist)
            cv2.putText(main_bgr, f"Detections: {det_count} | {infer_ms:.1f} ms ({fps_avg:.1f} FPS avg)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 回捞 hard cases
            now = time.time()
            if (det_count == 0 or low_conf) and (now - last_save > 1.0):
                cv2.imwrite(os.path.join(args.save_dir, f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"), main_bgr)
                last_save = now

            if not args.headless:
                cv2.imshow("Battery Detection - Pi4 (AWB on, AE locked)", main_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
