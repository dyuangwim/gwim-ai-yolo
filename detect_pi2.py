import os, time, argparse
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# ---- YUV420 -> BGR 转换函数表 ----
YUV2BGR_CODE = {
    "i420": cv2.COLOR_YUV2BGR_I420,
    "yv12": cv2.COLOR_YUV2BGR_YV12,
    "nv12": cv2.COLOR_YUV2BGR_NV12,
    "nv21": cv2.COLOR_YUV2BGR_NV21,
}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-debug", action="store_true")
    # yuv=auto 时自动探测；否则用指定的 i420/yv12/nv12/nv21
    ap.add_argument("--yuv", default="auto", choices=["auto","i420","yv12","nv12","nv21"])
    # 允许用户强制 swap_rb；默认 auto 时会自动评估是否需要
    ap.add_argument("--swap-rb", dest="swap_rb", action="store_true")
    ap.add_argument("--no-swap-rb", dest="swap_rb", action="store_false")
    ap.set_defaults(swap_rb=None)  # None=自动决策；True/False=用户强制
    # 可选：预热后锁 AWB
    ap.add_argument("--lock-awb", action="store_true")
    return ap.parse_args()

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def mse(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def choose_best_yuv_and_swap(main_bgr, yuv_lores, imgsz, user_yuv, user_swap):
    """
    在启动时自动选择 YUV 子格式与是否需要 R/B 交换。
    返回 (chosen_yuv, do_swap_rb)
    """
    # 将 main_bgr 降采样到 imgsz×imgsz 作为比对基准
    ref = cv2.resize(main_bgr, (imgsz, imgsz), interpolation=cv2.INTER_AREA)

    # 若用户指定 yuv 格式，就只测这一种；否则四种都测
    yuv_modes = [user_yuv] if user_yuv in YUV2BGR_CODE else list(YUV2BGR_CODE.keys())

    best = (1e12, "i420", False)  # (mse, mode, swap)
    for mode in yuv_modes:
        bgr = cv2.cvtColor(yuv_lores, YUV2BGR_CODE[mode])
        if bgr.shape[:2] != (imgsz, imgsz):
            bgr = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

        # 两种 swap 方案：不交换 / 交换（BGR<->RGB）
        candidates = [(bgr, False)]
        bgr_swapped = bgr[..., ::-1]
        candidates.append((bgr_swapped, True))

        for cand, is_swapped in candidates:
            if user_swap is not None and is_swapped != user_swap:
                continue  # 用户强制 swap 选择时，跳过不符合的
            score = mse(ref, cand)
            if score < best[0]:
                best = (score, mode, is_swapped)

    return best[1], best[2]

def draw_box(img_bgr, x1, y1, x2, y2, label, conf):
    cv2.rectangle(img_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    txt=f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img_bgr, (x1, y0), (x1+tw+4, y0+th+6), (0,255,0), -1)
    cv2.putText(img_bgr, txt, (x1+2, y0+th+2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def main():
    # 线程上限，避免争用
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("NCNN_THREADS", "4")

    args = parse_args()
    ensure_dir(args.save_dir)

    # ---- Camera：main=BGR888（用于显示）；lores=YUV420（用于推理） ----
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    cfg = picam2.create_video_configuration(
        main = {"size": (main_w, main_h), "format": "BGR888"},
        lores= {"size": (args.imgsz, args.imgsz), "format": "YUV420"},
        controls={"AeEnable": True, "AwbEnable": True}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.8)

    # 可选：锁 AWB
    if args.lock_awb:
        try:
            md = picam2.capture_metadata()
            gains = md.get("ColourGains", None)
            if gains:
                picam2.set_controls({"AwbEnable": False, "ColourGains": gains})
                print(f"[AWB] Locked gains={gains}")
        except Exception as e:
            print(f"[AWB] lock failed: {e}")

    # 启动时抓一帧 main（BGR）和一帧 lores（YUV），自动确定 yuv 模式与是否 swap
    main_probe = picam2.capture_array("main")
    yuv_probe  = picam2.capture_array("lores")
    if main_probe.ndim == 3 and main_probe.shape[2] == 4: main_probe = main_probe[:, :, :3]

    chosen_mode, need_swap = choose_best_yuv_and_swap(main_probe, yuv_probe, args.imgsz, args.yuv, args.swap_rb)
    print(f"[AUTO] YUV mode -> {chosen_mode} ; swap_rb -> {need_swap}")

    if args.save_debug:
        cv2.imwrite("/home/pi/view_debug.jpg", main_probe)
        # 保存自动选择的转换结果，便于核对
        cand = cv2.cvtColor(yuv_probe, YUV2BGR_CODE[chosen_mode])
        if cand.shape[:2] != (args.imgsz, args.imgsz):
            cand = cv2.resize(cand, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        if need_swap:
            cand = cand[..., ::-1]
        cv2.imwrite("/home/pi/lores_debug.jpg", cand)
        print("Saved /home/pi/view_debug.jpg and /home/pi/lores_debug.jpg")

    # ---- Model ----
    model = YOLO(args.weights)

    if not args.headless:
        cv2.namedWindow("Battery Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery Detection", main_w, main_h)

    last_save = 0
    fps_hist = []

    try:
        while True:
            t0 = time.time()

            # lores：YUV420 -> BGR（使用自动选定的模式）
            yuv = picam2.capture_array("lores")
            infer_bgr = cv2.cvtColor(yuv, YUV2BGR_CODE[chosen_mode])
            if infer_bgr.shape[:2] != (args.imgsz, args.imgsz):
                infer_bgr = cv2.resize(infer_bgr, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            if need_swap:
                infer_bgr = infer_bgr[..., ::-1]  # BGR<->RGB 交换

            # main：BGR888（直接用于显示与画框）
            frame_bgr = picam2.capture_array("main")
            if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4: frame_bgr = frame_bgr[:, :, :3]

            # 推理（Ultralytics 能接收 BGR/RGB ndarray；这里直接给 BGR）
            r = model.predict(source=infer_bgr, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            # lores → main 尺度映射
            lh, lw = args.imgsz, args.imgsz
            mh, mw = frame_bgr.shape[:2]
            sx, sy = mw/float(lw), mh/float(lh)

            det_count, low_conf = 0, False
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    if (X2-X1)*(Y2-Y1) < args.min_area:
                        continue
                    conf = float(b.conf[0])
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    label = r.names.get(cls_id, "battery")
                    draw_box(frame_bgr, X1, Y1, X2, Y2, label, conf)
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

            dt = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0));  fps_hist = fps_hist[-30:]
            hud = f"Det:{det_count} | {dt:.1f} ms ({sum(fps_hist)/len(fps_hist):.1f} FPS) | yuv={chosen_mode} | swap={int(need_swap)}"
            cv2.putText(frame_bgr, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if not args.headless:
                cv2.imshow("Battery Detection", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # hard cases
            now = time.time()
            if (det_count==0 or low_conf) and (now-last_save>1.0):
                fn=f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame_bgr)
                last_save = now

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
