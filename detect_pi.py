# detect_pi.py — COLOR-PURE & NO-BLACK-SCREEN BASELINE
# 只开默认 AE/AWB，不写任何曝光/白平衡/帧时长；单流 BGR888，绝不改色链路。
import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
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

    # 1) 只开主流 BGR888（OpenCV 原生），让 ISP + AE/AWB 全自动
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 960), "format": "BGR888"},   # 4:3；你也可改成 (1280, 720)
        controls={"AeEnable": True, "AwbEnable": True}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # 给 AE/AWB 一点时间收敛

    # 2) 模型
    model = YOLO(args.weights)

    last_save = 0
    fps_hist = []

    try:
        while True:
            t0 = time.time()

            # 3) 直接拿 BGR888 给 OpenCV 显示
            req = picam2.capture_request()
            frame_bgr = req.make_array("main")   # 已经是 BGR888，不需要再转换
            req.release()

            # 4) 推理只做一次“BGR->RGB + resize”
            infer_in = cv2.resize(frame_bgr, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            infer_in = cv2.cvtColor(infer_in, cv2.COLOR_BGR2RGB)  # YOLO 习惯 RGB
            r = model.predict(source=infer_in, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            h, w = frame_bgr.shape[:2]
            sx = w / float(args.imgsz); sy = h / float(args.imgsz)

            det_count, low_conf = 0, False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    if (x2-x1)*(y2-y1) < args.min_area: continue
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    label = r.names.get(cls_id, "battery")
                    X1,Y1,X2,Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    draw_box(frame_bgr, X1,Y1,X2,Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

            dt = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0));  fps_hist = fps_hist[-30:]
            hud=f"Detections:{det_count} | {dt:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS)"
            cv2.putText(frame_bgr, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # 低置信或无检出，保存硬样本
            now=time.time()
            if (det_count==0 or low_conf) and (now-last_save>1.0):
                cv2.imwrite(os.path.join(args.save_dir, f"hard_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"), frame_bgr)
                last_save=now

            if not args.headless:
                cv2.imshow("Battery Detection — COLOR PURE", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
