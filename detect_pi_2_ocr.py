import os, time, argparse
from datetime import datetime
import re
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# =============== Optional OCR =================
try:
    import pytesseract
    TESS_OK = True
except Exception:
    TESS_OK = False

# =============== Battery patterns =============
BATTERY_MODELS = ["1616", "1620", "2016", "2025", "2032"]
RE_CR   = re.compile(r"CR\s*(1616|1620|2016|2025|2032)")
RE_DIG4 = re.compile(r"(1616|1620|2016|2025|2032)")

# =============== Args =========================

def parse_args():
    ap = argparse.ArgumentParser(description="RPi5: Battery detect + OCR (label shows OCR text)")
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true")

    # Camera/colors
    ap.add_argument("--assume_bgr", action="store_true", help="Skip YUV->BGR if your feed is already BGR")

    # OCR controls (IMPORTANT: 'every' means frames, not seconds!)
    ap.add_argument("--ocr", action="store_true", help="Enable OCR")
    ap.add_argument("--ocr_every", type=int, default=3, help="Run OCR every N FRAMES (not seconds). Lower = more OCR, slower.")
    ap.add_argument("--ocr_pad", type=float, default=0.12, help="Padding ratio around bbox for OCR crop")
    ap.add_argument("--ocr_psm", type=int, default=7, help="Tesseract PSM: 7=single line, 8=single word, 6=block")
    ap.add_argument("--tess_cmd", default="", help="Path to tesseract binary if not in PATH")

    # Label mode
    ap.add_argument("--label_mode", choices=["ocr","det","both"], default="ocr",
                    help="What to show on the box label: OCR text, detector cls/conf, or both")

    # Threads
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()

# =============== Utils ========================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def draw_label_box(img, x1, y1, x2, y2, label_text, sub_text=None, color=(0,255,0)):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    if label_text:
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y0 = max(0, y1 - th - 8)
        cv2.rectangle(img, (x1, y0), (x1+tw+10, y0+th+10), color, -1)
        cv2.putText(img, label_text, (x1+5, y0+th+3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    if sub_text:
        (tw, th), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y2b = min(img.shape[0]-1, y2 + th + 12)
        cv2.rectangle(img, (x1, y2b-th-10), (x1+tw+10, y2b), (0,215,255), -1)
        cv2.putText(img, sub_text, (x1+5, y2b-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)


# =============== OCR ==========================

def _prep_variants(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.bilateralFilter(g, 5, 50, 50)
    g2 = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ada = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    return [g2, otsu, cv2.bitwise_not(otsu), ada]


def _tess(img, psm=7):
    if not TESS_OK:
        return "", 0.0
    cfg = f"--psm {psm} -c tessedit_char_whitelist=CR0123456789."
    try:
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=cfg)
        txt = " ".join([w for w in d.get("text", []) if w])
        confs = [float(c) for c in d.get("conf", []) if str(c).isdigit() or (isinstance(c, (int,float)) and c >= 0)]
        conf = sum(confs)/len(confs) if confs else 0.0
        return txt, conf
    except Exception:
        try:
            return pytesseract.image_to_string(img, config=cfg), 0.0
        except Exception:
            return "", 0.0


def ocr_battery_crop(bgr_roi, psm=7):
    if bgr_roi is None or bgr_roi.size == 0:
        return None, "", 0.0
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    variants = _prep_variants(gray)

    best_txt, best_conf = "", -1
    for v in variants:
        t, c = _tess(v, psm=psm)
        clean = ''.join(ch for ch in t.upper() if ch.isalnum() or ch in 'CR.')
        if c > best_conf and clean.strip():
            best_txt, best_conf = clean, c

    # Normalize to CRxxxx if possible
    model = None
    m = RE_CR.search(best_txt)
    if m:
        model = f"CR{m.group(1)}"
    else:
        m2 = RE_DIG4.search(best_txt)
        if m2 and m2.group(1) in BATTERY_MODELS:
            model = f"CR{m2.group(1)}"

    return model, best_txt.strip(), float(best_conf)


# =============== Main =========================

def main():
    args = parse_args()

    # Threads/env
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["NCNN_THREADS"] = str(args.threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NCNN_VERBOSE", "0")

    if args.ocr and not TESS_OK:
        print("WARNING: Tesseract not found; OCR disabled.")
    if args.tess_cmd and TESS_OK:
        pytesseract.pytesseract.tesseract_cmd = args.tess_cmd

    ensure_dir(args.save_dir)

    # Camera
    picam2 = Picamera2()
    main_w, main_h = 1280, 720  # slightly lower than 960 for speed
    if args.preview:
        config = picam2.create_preview_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"AeEnable": True, "AwbEnable": True, "FrameRate": 30}
        )
    else:
        config = picam2.create_video_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"AeEnable": True, "AwbEnable": True, "FrameRate": 30}
        )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.5)

    # Model
    model = YOLO(args.weights)
    dummy = np.random.randint(0,255,(args.imgsz, args.imgsz, 3), np.uint8)
    _ = model.predict(source=dummy, imgsz=args.imgsz, verbose=False)

    if not args.headless:
        cv2.namedWindow("Battery+OCR", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery+OCR", main_w, main_h)

    # Scaling small->full
    sx, sy = main_w/float(args.imgsz), main_h/float(args.imgsz)

    # Stats
    frame_count = 0
    fps_hist = []
    total_ocr_ms, ocr_runs = 0.0, 0
    last_save = 0.0
    start = time.time()

    try:
        while True:
            t0 = time.time()
            yuv = picam2.capture_array("main")
            if args.assume_bgr:
                frame = yuv
            else:
                try:
                    if len(yuv.shape)==2 and yuv.shape[0]==main_h*3//2:
                        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    else:
                        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                except Exception:
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            small = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            r = model.predict(source=small, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            det_count, low_conf = 0, False
            do_ocr_this_frame = args.ocr and TESS_OK and (frame_count % args.ocr_every == 0)

            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    if (X2-X1)*(Y2-Y1) < args.min_area:
                        continue
                    det_count += 1

                    det_conf = float(box.conf[0]) if box.conf is not None else 0.0
                    if det_conf < (args.conf + 0.10):
                        low_conf = True
                    cls_id = int(box.cls[0]) if box.cls is not None else 0
                    det_label = r.names.get(cls_id, "battery")

                    # ----- OCR (per N frames) -----
                    ocr_text_for_label = None
                    sub_text = None
                    if do_ocr_this_frame:
                        o_t0 = time.time()
                        pad = args.ocr_pad
                        w, h = X2 - X1, Y2 - Y1
                        px, py = int(w*pad), int(h*pad)
                        cx1 = max(0, X1 - px); cy1 = max(0, Y1 - py)
                        cx2 = min(main_w, X2 + px); cy2 = min(main_h, Y2 + py)
                        crop = frame[cy1:cy2, cx1:cx2].copy()
                        model_code, raw_txt, conf_est = ocr_battery_crop(crop, psm=args.ocr_psm)
                        ocr_ms = (time.time()-o_t0)*1000.0
                        total_ocr_ms += ocr_ms; ocr_runs += 1

                        # 作为主标签显示：优先标准化型号，其次显示 raw 文本
                        if model_code:
                            ocr_text_for_label = model_code
                            sub_text = f"OCR {conf_est:.0f}"
                        elif raw_txt:
                            ocr_text_for_label = raw_txt[:18]
                            sub_text = f"OCR {conf_est:.0f}"
                        else:
                            ocr_text_for_label = "OCR?"
                            sub_text = None

                    # Decide label text based on mode
                    if args.label_mode == "ocr":
                        label_text = ocr_text_for_label or det_label
                    elif args.label_mode == "both":
                        base = ocr_text_for_label or det_label
                        label_text = f"{base} | {det_label} {det_conf:.2f}"
                    else:  # det
                        label_text = f"{det_label} {det_conf:.2f}"

                    draw_label_box(frame, X1, Y1, X2, Y2, label_text, sub_text)

            # HUD / FPS
            frame_count += 1
            dt = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0))
            if len(fps_hist) > 30: fps_hist.pop(0)
            avg_fps = sum(fps_hist)/len(fps_hist)

            hud = f"Det:{det_count} | {dt:.1f}ms | {avg_fps:.1f}FPS"
            if args.ocr:
                if TESS_OK:
                    hud += f" | OCR every {args.ocr_every}f"
                    if ocr_runs:
                        hud += f" (~{total_ocr_ms/ocr_runs:.0f}ms)"
                else:
                    hud += " | OCR missing"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # Show
            if not args.headless:
                cv2.imshow("Battery+OCR", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Save hard cases
            now = time.time()
            if (det_count==0 or low_conf) and (now-last_save>1.0):
                fn = f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame)
                last_save = now

            if frame_count % 120 == 0:
                elapsed = time.time()-start
                print(f"Processed {frame_count} frames in {elapsed:.1f}s, FPS={frame_count/elapsed:.1f}")

    except KeyboardInterrupt:
        pass
    finally:
        try: picam2.stop()
        except Exception: pass
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
