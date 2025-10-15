import os, time, argparse, math
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/battery.pt", help="YOLO .pt path")
    ap.add_argument("--imgsz", type=int, default=416, help="inference size (416/512/640)")
    ap.add_argument("--conf", type=float, default=0.30, help="confidence threshold")
    ap.add_argument("--min_area", type=int, default=2500, help="min bbox area in px (过滤小假框)")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases", help="回捞误例/低置信结果")
    ap.add_argument("--headless", action="store_true", help="无显示器运行，不弹窗")
    return ap.parse_args()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label, conf):
    color = (0, 255, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    # 1) Camera 配置（RPI-CAM-GS 推荐 1280x960，1:1.33，留足分辨率计数/后续OCR）
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 960), "format": "RGB888"},  # RGB888 -> OpenCV BGR 转换更快
        controls={"FrameDurationLimits": (33333, 33333)}  # ~30fps; 可按光线降低为 50-60fps上限
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.3)  # 预热

    # 2) 加载 YOLO
    model = YOLO(args.weights)

    fps_hist = []
    last_save_ts = 0

    try:
        while True:
            frame = picam2.capture_array()  # RGB
            # OpenCV 用 BGR，转换一下
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame.shape[:2]

            t0 = time.time()
            # 3) 推理（Ultralytics 对 numpy 支持很好）
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False
            )
            t1 = time.time()

            det_count = 0
            low_conf_flag = False

            # 4) 解析结果 + 面积过滤 + 绘制
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
                        conf = float(b.conf[0])
                        cls_id = int(b.cls[0]) if b.cls is not None else 0
                        label = model.names.get(cls_id, "battery")

                        x1, y1, x2, y2 = map(lambda x: int(max(0, min(x, 10**6))), xyxy)
                        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                        if area < args.min_area:
                            continue  # 面积太小，视为假框

                        det_count += 1
                        draw_box(frame, x1, y1, x2, y2, label, conf)
                        if conf < (args.conf + 0.10):  # 比门槛高一点仍然认为偏低 -> 回捞
                            low_conf_flag = True

            # 5) HUD / 计数显示
            infer_ms = (t1 - t0) * 1000
            fps = 1000.0 / max(1.0, infer_ms)
            fps_hist.append(fps)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            fps_avg = sum(fps_hist) / max(1, len(fps_hist))

            hud = f"Detections: {det_count} | {infer_ms:.1f} ms ({fps_avg:.1f} FPS avg)"
            cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 6) 回捞：无检出或低置信，且与上次保存间隔>=1s，避免刷屏
            now_ts = time.time()
            if (det_count == 0 or low_conf_flag) and (now_ts - last_save_ts > 1.0):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                p = os.path.join(args.save_dir, f"hard_{det_count}_{ts}.jpg")
                cv2.imwrite(p, frame)
                last_save_ts = now_ts

            # 7) 显示/输出
            if not args.headless:
                cv2.imshow("Battery Detection - Pi4", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                # 纯控制台模式
                pass

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
