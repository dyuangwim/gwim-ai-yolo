#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2


def parse_args():
    ap = argparse.ArgumentParser(description="Raspberry Pi + Picamera2 + YOLO (NCNN) realtime detector")
    # 建议：指向含有 model.ncnn.param / model.ncnn.bin 的“目录”
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model",
                    help="Path to NCNN model folder (contains model.ncnn.param/bin)")
    ap.add_argument("--imgsz", type=int, default=416, help="inference size for lores stream (square)")
    ap.add_argument("--conf", type=float, default=0.30, help="confidence threshold")
    ap.add_argument("--min_area", type=int, default=1800, help="min bbox area (filter tiny)")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases", help="folder to save hard samples")
    ap.add_argument("--headless", action="store_true", help="no cv2.imshow (for SSH / no GUI)")

    # 相机控制：先让 AE/AWB 收敛，再锁 AE（快门/增益）。AWB 默认不锁，只有传 --lock_awb 才锁。
    ap.add_argument("--shutter", type=int, default=4000, help="ExposureTime(us) to lock AE")
    ap.add_argument("--gain", type=float, default=6.0, help="AnalogueGain to lock AE")
    ap.add_argument("--warmup", type=float, default=1.0, help="seconds to let AE/AWB settle")
    ap.add_argument("--lock_awb", action="store_true", help="lock AWB using settled ColourGains")

    return ap.parse_args()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def draw_box(img, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(img, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    # ==== 1) 初始化相机 ====
    picam2 = Picamera2()
    # lores 用于推理（YUV420），main 用于显示（RGB888）
    # imgsz 尽量用偶数（YUV420 对齐），Picamera2 会处理；若担心，取偶：args.imgsz - args.imgsz % 2
    lo_side = args.imgsz if args.imgsz % 2 == 0 else (args.imgsz - 1)

    video_config = picam2.create_video_configuration(
        main={
            "size": (1280, 960),
            "format": "RGB888"
        },
        lores={
            "size": (lo_side, lo_side),
            "format": "YUV420"  # I420: (h*3/2, w)
        },
        controls={
            # 30 fps 预算；如需更高帧率可压尺寸/关显示
            "FrameDurationLimits": (33333, 33333),
            # 先让 AE/AWB 开启并收敛
            "AeEnable": True,
            "AwbEnable": True,
            # 细节更清晰（利于读字）
            "NoiseReductionMode": 0,  # Off
            "Sharpness": 1.3
        }
    )
    picam2.configure(video_config)
    picam2.start()

    # 让 AE / AWB 收敛
    time.sleep(max(0.5, args.warmup))

    # 锁 AE（快门 / 增益）；AWB 默认不锁（颜色更稳）
    cam_controls = {
        "AeEnable": False,
        "ExposureTime": args.shutter,
        "AnalogueGain": args.gain,
    }
    if args.lock_awb:
        md = picam2.capture_metadata()
        cg = md.get("ColourGains", (1.0, 1.0))
        cam_controls.update({"AwbEnable": False, "ColourGains": cg})
    picam2.set_controls(cam_controls)

    # ==== 2) 加载 YOLO（NCNN） ====
    # 注意：Ultralytics 会尝试安装 ncnn 依赖，第一次加载稍慢；确保 weights 指向「目录」。
    model_path = args.weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"weights path not found: {model_path}")
    # 目录内需含有 model.ncnn.param / model.ncnn.bin
    if os.path.isdir(model_path):
        if not (os.path.exists(os.path.join(model_path, "model.ncnn.param"))
                and os.path.exists(os.path.join(model_path, "model.ncnn.bin"))):
            raise FileNotFoundError(
                f"NCNN files not found in folder: {model_path}\n"
                f"Expect: model.ncnn.param / model.ncnn.bin"
            )
    model = YOLO(model_path)

    fps_hist = []
    last_save = 0.0

    try:
        while True:
            t0 = time.time()

            # ==== 3) 取一帧（两路）====
            req = picam2.capture_request()
            lo = req.make_array('lores')     # YUV420: shape == (h*3/2, w)
            main_rgb = req.make_array('main')  # RGB888
            req.release()

            # ==== 4) YUV420 -> BGR ====
            # lo.shape = (h*3/2, w)  => 取 w、h
            w = lo.shape[1]
            h = lo.shape[0] * 2 // 3  # ← 修正：用第0维（不是整个 tuple）
            # 这里 lo 已经是 (h*3/2, w)，reshape 一次以确保连续内存形状正确
            lo = lo.reshape((h * 3 // 2, w))
            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)  # 得到 h x w x 3

            # ==== 5) YOLO 推理（在 lores 上）====
            results = model.predict(source=lo_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)
            r = results[0]

            # ==== 6) 把框映射回 main 并绘制 ====
            main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
            mh, mw = main_bgr.shape[:2]
            sx, sy = mw / float(w), mh / float(h)

            det_count = 0
            low_conf = False

            if r.boxes is not None and len(r.boxes) > 0:
                # Ultralytics Boxes API
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None

                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    area = (x2 - x1) * (y2 - y1)
                    if area < args.min_area:
                        continue
                    conf = float(confs[i])
                    cls_id = int(clses[i]) if clses is not None else 0
                    label = r.names.get(cls_id, "object") if hasattr(r, "names") else "object"

                    X1, Y1, X2, Y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                    draw_box(main_bgr, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10):
                        low_conf = True

            # ==== 7) FPS 统计 & Hard cases 回捞 ====
            dt_ms = (time.time() - t0) * 1000.0
            fps = 1000.0 / max(1.0, dt_ms)
            fps_hist.append(fps)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            fps_avg = sum(fps_hist) / len(fps_hist)

            cv2.putText(main_bgr,
                        f"Detections:{det_count} | {dt_ms:.1f} ms ({fps_avg:.1f} FPS avg)",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            now = time.time()
            if (det_count == 0 or low_conf) and (now - last_save > 1.0):
                ensure_dir(args.save_dir)
                out_path = os.path.join(
                    args.save_dir,
                    f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )
                cv2.imwrite(out_path, main_bgr)
                last_save = now

            # ==== 8) 显示 ====
            if not args.headless:
                cv2.imshow("Battery Detection - Pi (AWB on, AE locked)", main_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
