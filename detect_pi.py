import os, time, argparse

from datetime import datetime

import cv2

from ultralytics import YOLO

from picamera2 import Picamera2



def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--weights", default="/home/pi/models/battery.pt")

    ap.add_argument("--imgsz", type=int, default=320)       # 小一档提FPS

    ap.add_argument("--conf", type=float, default=0.30)

    ap.add_argument("--min_area", type=int, default=2000)   # lores 上阈值小一些

    ap.add_argument("--save_dir", default="/home/pi/hard_cases")

    ap.add_argument("--headless", action="store_true")



    # 相机/曝光

    ap.add_argument("--shutter", type=int, default=8000,    # us，1/125s；看现场可改 6000/4000

                    help="ExposureTime in microseconds when locking AE")

    ap.add_argument("--gain", type=float, default=2.0,      # 2~8 合理，越大越亮但噪

                    help="AnalogueGain when locking AE")

    ap.add_argument("--warmup", type=float, default=1.0,    # 先让 AE/AWB 自稳

                    help="seconds to let AE/AWB settle before locking")

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

    # main 用于显示（RGB），lores 用于推理（YUV420）

    config = picam2.create_video_configuration(

        main  = {"size": (1280, 960), "format": "RGB888"},

        lores = {"size": (640, 480),  "format": "YUV420"},

        controls={

            "FrameDurationLimits": (33333, 33333),  # 30 fps 预算

            "AeEnable": True,

            "AwbEnable": True

        }

    )

    picam2.configure(config)

    picam2.start()



    # 让 AE/AWB 先收敛再锁定（避免偏色/忽明忽暗）

    time.sleep(max(0.2, args.warmup))

    picam2.set_controls({

        "AeEnable": False,

        "AwbEnable": False,

        "ExposureTime": args.shutter,

        "AnalogueGain": args.gain,

    })



    model = YOLO(args.weights)

    fps_hist, last_save = [], 0



    try:

        while True:

            t0 = time.time()



            # 1) lores 推理：YUV420 -> BGR

            lo = picam2.capture_array("lores")  # shape = (h*3/2, w), YUV420P

            # 从 (h*3/2, w) 反推 h,w

            lo_w = lo.shape[1]

            lo_h = lo.shape[0] * 2 // 3

            lo = lo.reshape((lo_h * 3 // 2, lo_w))

            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)



            results = model.predict(

                source=lo_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False

            )

            dets = []

            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:

                for b in r.boxes:

                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())

                    area = max(0,(x2-x1))*max(0,(y2-y1))

                    if area < args.min_area: continue

                    dets.append((x1,y1,x2,y2,float(b.conf[0]), int(b.cls[0]) if b.cls is not None else 0))



            # 2) main 显示：RGB -> BGR（便于绘制/显示）

            main_rgb = picam2.capture_array()   # main

            main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)



            # lores -> main 的坐标映射

            mh, mw = main_bgr.shape[:2]

            sx, sy = mw / float(lo_w), mh / float(lo_h)



            det_count, low_conf = 0, False

            for (x1,y1,x2,y2,conf,cls_id) in dets:

                X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)

                label = model.names.get(cls_id, "battery")

                draw_box(main_bgr, X1,Y1,X2,Y2, label, conf)

                det_count += 1

                if conf < (args.conf + 0.10): low_conf = True



            infer_ms = (time.time() - t0) * 1000.0

            fps_hist.append(1000.0 / max(1.0, infer_ms))

            if len(fps_hist) > 30: fps_hist.pop(0)

            fps_avg = sum(fps_hist)/len(fps_hist)

            hud = f"Detections: {det_count} | {infer_ms:.1f} ms ({fps_avg:.1f} FPS avg)"

            cv2.putText(main_bgr, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)



            # 回捞 hard cases

            now = time.time()

            if (det_count == 0 or low_conf) and (now - last_save > 1.0):

                p = os.path.join(args.save_dir, f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")

                cv2.imwrite(p, main_bgr)

                last_save = now



            if not args.headless:

                cv2.imshow("Battery Detection - Pi4 (lores infer + main display)", main_bgr)

                if cv2.waitKey(1) & 0xFF == 27: break



    except KeyboardInterrupt:

        pass

    finally:

        picam2.stop()

        cv2.destroyAllWindows()



if __name__ == "__main__":

    main()
