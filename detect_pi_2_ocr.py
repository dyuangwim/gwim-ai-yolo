import os, time, argparse
from datetime import datetime
import re
import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# -----------------------------
# Optional OCR (pytesseract)
# -----------------------------
try:
    import pytesseract
    TESS_OK = True
except Exception:
    TESS_OK = False

# -----------------------------
# Battery model patterns
# -----------------------------
BATTERY_MODELS = ["1616", "1620", "2016", "2025", "2032"]
BATTERY_REGEX_CR = re.compile(r"CR\s*(1616|1620|2016|2025|2032)")
BATTERY_REGEX_DIGIT = re.compile(r"(1616|1620|2016|2025|2032)")


# =============================
# Args
# =============================

def parse_args():
    ap = argparse.ArgumentParser(description="Raspberry Pi 5 — Battery detect + OCR (optimized)")
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model", help="YOLO weights folder/file (NCNN dir or .pt/.onnx)")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true", help="Use preview configuration for better performance")

    # Color / capture controls
    ap.add_argument("--assume_bgr", action="store_true", help="Treat capture as BGR (skip YUV->BGR). Use this if your colors are correct only with --assume-bgr.")

    # OCR controls
    ap.add_argument("--ocr", action="store_true", help="Enable OCR per detected battery")
    ap.add_argument("--ocr_every", type=int, default=5, help="Run OCR every N frames to save CPU")
    ap.add_argument("--ocr_pad", type=float, default=0.12, help="Crop padding ratio around battery for OCR [0-0.4]")
    ap.add_argument("--ocr_psm", type=int, default=7, help="Tesseract PSM (7=single line, 8=single word, 6=block)")
    ap.add_argument("--tess_cmd", default="", help="Path to tesseract binary if not in PATH")

    # Performance/env
    ap.add_argument("--threads", type=int, default=4, help="NCNN/OMP threads")
    return ap.parse_args()


# =============================
# Utils
# =============================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def draw_box(img, x1, y1, x2, y2, label, conf=None, info=None, color=(0,255,0)):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    txt = f"{label}"
    if conf is not None:
        txt += f" {conf:.2f}"
    if info:
        txt += f" | {info}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y0), (x1+tw+8), (y0+th+8), color, -1)
    cv2.putText(img, txt, (x1+4, y0+th+3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


# =============================
# OCR Preprocess & Run
# =============================

def _prep_roi(gray):
    # CLAHE reduces local lighting issues / glare; bilateral keeps edges; upscale helps OCR
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.bilateralFilter(g, 5, 50, 50)
    g2x = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Otsu + adaptive variants
    _, otsu = cv2.threshold(g2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ada = cv2.adaptiveThreshold(g2x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    return [g2x, otsu, cv2.bitwise_not(otsu), ada]


def _tess(img, psm=7):
    if not TESS_OK:
        return "", 0.0
    config = f"--psm {psm} -c tessedit_char_whitelist=CR0123456789."
    try:
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
        txt = " ".join([w for w in d.get("text", []) if w])
        confs = [float(c) for c in d.get("conf", []) if str(c).isdigit() or (isinstance(c, (int,float)) and c >= 0)]
        conf = sum(confs)/len(confs) if confs else 0.0
        return txt, conf
    except Exception:
        try:
            txt = pytesseract.image_to_string(img, config=config)
            return txt, 0.0
        except Exception:
            return "", 0.0


def ocr_battery_crop(bgr_roi, psm=7):
    if bgr_roi is None or bgr_roi.size == 0:
        return None, "CropErr", 0.0
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    variants = _prep_roi(gray)

    best_txt, best_conf = "", -1
    for v in variants:
        txt, conf = _tess(v, psm=psm)
        clean = ''.join(ch for ch in txt.upper() if ch.isalnum() or ch in 'CR.')
        if conf > best_conf and clean.strip():
            best_txt, best_conf = clean, conf

    # Normalize with regex (accept CRxxxx or bare 4 digits)
    model = None
    m = BATTERY_REGEX_CR.search(best_txt)
    if m:
        model = f"CR{m.group(1)}"
    else:
        m2 = BATTERY_REGEX_DIGIT.search(best_txt)
        if m2 and m2.group(1) in BATTERY_MODELS:
            model = f"CR{m2.group(1)}"

    return model, best_txt.strip(), float(best_conf)


# =============================
# Main
# =============================

def main():
    args = parse_args()

    # Threads & env (helps NCNN / OpenBLAS on Pi 5)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["NCNN_THREADS"] = str(args.threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NCNN_VERBOSE", "0")

    if args.ocr and not TESS_OK:
        print("WARNING: Tesseract import failed; OCR disabled.")
    if args.tess_cmd and TESS_OK:
        pytesseract.pytesseract.tesseract_cmd = args.tess_cmd

    ensure_dir(args.save_dir)

    # --- Camera ---
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
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
    time.sleep(2.0)

    # --- Model ---
    print("Loading model…")
    model = YOLO(args.weights)
    # Warmup on random noise to initialize backends
    warm = np.random.randint(0,255,(args.imgsz, args.imgsz, 3), np.uint8)
    for _ in range(2):
        _ = model.predict(source=warm, imgsz=args.imgsz, verbose=False)
    print("Model ready.")

    # Optional window
    if not args.headless:
        cv2.namedWindow("Battery+OCR", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Battery+OCR", main_w, main_h)

    last_save = 0.0
    frame_count = 0
    start_time = time.time()
    fps_hist = []

    # OCR profiling
    total_ocr_ms = 0.0
    ocr_runs = 0

    # For mapping resized inference back to full-res frame
    sx, sy = main_w/float(args.imgsz), main_h/float(args.imgsz)

    try:
        while True:
            t0 = time.time()
            yuv = picam2.capture_array("main")

            if args.assume_bgr:
                frame = yuv
            else:
                # Robust YUV420/NV12 fallback conversion
                try:
                    if len(yuv.shape) == 2 and yuv.shape[0] == main_h * 3 // 2:
                        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    else:
                        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
                except Exception:
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            r = model.predict(source=infer_img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

            det_count, low_conf = 0, False
            ocr_sample_frame = args.ocr and TESS_OK and (frame_count % args.ocr_every == 0)

            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    if (X2-X1)*(Y2-Y1) < args.min_area:
                        continue

                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    label = r.names.get(cls_id, "battery")
                    det_count += 1
                    if conf < (args.conf + 0.10):
                        low_conf = True

                    info = None
                    if ocr_sample_frame:
                        ocr_t0 = time.time()
                        pad = args.ocr_pad
                        w, h = X2 - X1, Y2 - Y1
                        px, py = int(w*pad), int(h*pad)
                        cx1 = max(0, X1 - px); cy1 = max(0, Y1 - py)
                        cx2 = min(main_w, X2 + px); cy2 = min(main_h, Y2 + py)
                        crop = frame[cy1:cy2, cx1:cx2].copy()
                        model_code, raw_txt, conf_est = ocr_battery_crop(crop, psm=args.ocr_psm)
                        ocr_ms = (time.time() - ocr_t0) * 1000.0
                        total_ocr_ms += ocr_ms
                        ocr_runs += 1

                        if model_code:
                            info = f"{model_code} (OCR {conf_est:.0f})"
                        else:
                            info = f"OCR? ({conf_est:.0f})"

                    draw_box(frame, X1, Y1, X2, Y2, label, conf, info)

            # HUD / FPS
            frame_count += 1
            dt_ms = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt_ms,1.0))
            if len(fps_hist) > 30:
                fps_hist.pop(0)
            avg_fps = sum(fps_hist)/len(fps_hist)

            hud = f"Det:{det_count} | {dt_ms:.1f}ms | {avg_fps:.1f}FPS"
            if args.ocr:
                if TESS_OK:
                    avg_ocr = (total_ocr_ms/ocr_runs) if ocr_runs else 0.0
                    hud += f" | OCR:on {avg_ocr:.1f}ms/every{args.ocr_every}"
                else:
                    hud += " | OCR:missing"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if not args.headless:
                cv2.imshow("Battery+OCR", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Save hard cases (no detect / low conf)
            now = time.time()
            if (det_count == 0 or low_conf) and (now - last_save > 1.0):
                fn = f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame)
                last_save = now

            if frame_count % 120 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames in {elapsed:.1f}s, Avg FPS: {frame_count/elapsed:.1f}")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        if not args.headless:
            cv2.destroyAllWindows()
        total = time.time() - start_time
        if total > 0:
            print("\n=== Performance Summary ===")
            print(f"Frames: {frame_count}")
            print(f"Time: {total:.1f}s")
            print(f"Overall FPS: {frame_count/total:.1f}")
            if args.ocr and TESS_OK and ocr_runs:
                print(f"OCR runs: {ocr_runs} | Avg OCR: {total_ocr_ms/ocr_runs:.1f} ms")


if __name__ == "__main__":
    main()
