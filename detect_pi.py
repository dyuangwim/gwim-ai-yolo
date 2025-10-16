#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, argparse
from datetime import datetime
import cv2
from ultralytics import YOLO
from picamera2 import Picamera2

def parse_args():
    ap = argparse.ArgumentParser(description="Raspberry Pi + Picamera2 + YOLO(NCNN) realtime")
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model",
                    help="Folder containing model.ncnn.param/bin")
    # —— 推理/后处理更“保守”，减少假框、提升稳定性 ——
    ap.add_argument("--imgsz", type=int, default=352, help="lores inference size (square)")
    ap.add_argument("--conf", type=float, default=0.60, help="confidence threshold (↑抬高)")
    ap.add_argument("--iou", type=float, default=0.60, help="NMS IOU threshold (↑更严格)")
    ap.add_argument("--max_det", type=int, default=40, help="最多保留多少框（限流）")
    ap.add_argument("--min_area", type=int, default=8000, help="过滤很小的框（像素面积）")
    ap.add_argument("--topk_draw", type=int, default=12, help="最多绘制多少框（再限流）")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases", help="hard cases 目录")
    ap.add_argument("--headless", action="store_true", help="无窗口（建议SSH）", default=True)
    ap.add_argument("--no_labels", action="store_true", help="不画文字，只画框（更快）")

    # 相机：默认 AE 锁、AWB 不锁（不再发紫）
    ap.add_argument("--shutter", type=int, default=3500, help="曝光(us)，更短更清晰")
    ap.add_argument("--gain", type=float, default=8.0, help="增益，配合短快门提亮")
    ap.add_argument("--warmup", type=float, default=1.0, help="预热收敛秒数")
    ap.add_argument("--lock_awb", action="store_true", help="若指定则锁AWB")
    # 可选 ROI（减少背景干扰 + 加速）：x,y,w,h 按 lores 分辨率的绝对像素
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X","Y","W","H"),
                    help="只在 lores 的该矩形里推理。例如: --roi 40 40 272 272")

    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label, conf, no_labels=False):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    if no_labels: return
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1-th-6), (x1+tw+4, y1), (0,255,0), -1)
    cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def mask_roi(img_bgr, roi):
    if not roi: return img_bgr, (0,0)  # 无ROI
    x,y,w,h = roi
    x = max(0, x); y = max(0, y); w = max(1, w); h = max(1, h)
    x2, y2 = min(img_bgr.shape[1], x+w), min(img_bgr.shape[0], y+h)
    # 只对 ROI 区域送检；回绘时需要坐标偏移 (ox, oy)
    return img_bgr[y:y2, x:x2], (x, y)

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    # ==== 相机 ====
    picam2 = Picamera2()
    lo_side = args.imgsz if args.imgsz % 2 == 0 else (args.imgsz - 1)
    cfg = picam2.create_video_configuration(
        main = {"size": (1280, 960), "format": "RGB888"},
        lores= {"size": (lo_side, lo_side), "format": "YUV420"},
        controls={
            "FrameDurationLimits": (33333, 33333), # 30fps预算
            "AeEnable": True, "AwbEnable": True,
            "NoiseReductionMode": 0, "Sharpness": 1.3
        }
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(max(0.5, args.warmup))
    controls = {"AeEnable": False, "ExposureTime": args.shutter, "AnalogueGain": args.gain}
    if args.lock_awb:
        md = picam2.capture_metadata()
        controls.update({"AwbEnable": False, "ColourGains": md.get("ColourGains",(1.0,1.0))})
    picam2.set_controls(controls)

    # ==== 模型（NCNN） ====
    wdir = args.weights
    assert os.path.isdir(wdir), f"weights 必须指向目录: {wdir}"
    assert os.path.exists(os.path.join(wdir,"model.ncnn.param")), "缺少 model.ncnn.param"
    assert os.path.exists(os.path.join(wdir,"model.ncnn.bin")),   "缺少 model.ncnn.bin"
    model = YOLO(wdir)

    # 尝试读取 backend 名称（不同版本字段不同，尽量不崩）
    backend_name = getattr(getattr(model, "predictor", None), "backend", None)
    print(f"[INFO] Backend: {backend_name}")  # 如果是 NCNN 会体现；若为空，通常是没走到 NCNN

    fps_hist, last_save = [], 0.0
    try:
        while True:
            t0 = time.time()
            req = picam2.capture_request()
            lo = req.make_array('lores')     # (h*3/2, w)
            main_rgb = req.make_array('main')
            req.release()

            w = lo.shape[1]
            h = lo.shape[0] * 2 // 3   # ← 正确取高（修正点）  # 参见：:contentReference[oaicite:2]{index=2}
            lo = lo.reshape((h*3//2, w))
            lo_bgr = cv2.cvtColor(lo, cv2.COLOR_YUV2BGR_I420)

            # 可选 ROI：缩小推理范围 + 降低误检
            roi_img, (ox, oy) = mask_roi(lo_bgr, args.roi)

            # —— 只在 ROI 上推理；强约束 conf/iou/max_det（限流）——
            r0 = model.predict(
                source=roi_img,
                imgsz=roi_img.shape[1],   # 与 ROI 宽对齐即可（方形 ROI 建议）
                conf=args.conf, iou=args.iou, max_det=args.max_det,
                agnostic_nms=False, verbose=False
            )[0]

            main_bgr = cv2.cvtColor(main_rgb, cv2.COLOR_RGB2BGR)
            mh, mw = main_bgr.shape[:2]
            sx, sy = mw/float(w), mh/float(h)

            dets = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                clses = r0.boxes.cls.cpu().numpy() if r0.boxes.cls is not None else None
                for i,(x1,y1,x2,y2) in enumerate(xyxy):
                    # ROI 坐标 → lores 坐标（加回偏移），再映射到 main
                    x1 += ox; x2 += ox; y1 += oy; y2 += oy
                    area = (x2-x1)*(y2-y1)
                    if area < args.min_area:  # 二次过滤
                        continue
                    dets.append((float(confs[i]), int(clses[i]) if clses is not None else 0, x1,y1,x2,y2))

            # 只画 Top-K，避免绘制耗时 & 屏幕被刷爆
            dets.sort(key=lambda z: z[0], reverse=True)
            dets = dets[:max(1, args.topk_draw)]

            det_count = len(dets)
            for conf, cls_id, x1,y1,x2,y2 in dets:
                X1,Y1,X2,Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                label = (r0.names.get(cls_id,"battery") if hasattr(r0,"names") else "battery")
                draw_box(main_bgr, X1,Y1,X2,Y2, label, conf, no_labels=args.no_labels)

            # 统计 + 回捞
            dt_ms = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(1.0, dt_ms))
            if len(fps_hist) > 30: fps_hist.pop(0)
            fps_avg = sum(fps_hist)/len(fps_hist)
            cv2.putText(main_bgr, f"Detections:{det_count} | {dt_ms:.1f} ms ({fps_avg:.1f} FPS avg)",
                        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            now = time.time()
            if (det_count == 0) and (now-last_save>1.0):
                ensure_dir(args.save_dir)
                cv2.imwrite(os.path.join(args.save_dir, f"hard_0_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"), main_bgr)
                last_save = now

            if not args.headless:
                cv2.imshow("Battery Detection - Pi (AWB on, AE locked)", main_bgr)
                if cv2.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
