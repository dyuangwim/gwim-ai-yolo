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
# Battery model patterns & Config
# -----------------------------
BATTERY_MODELS = ["1616", "1620", "2016", "2025", "2032"]
BATTERY_REGEX_CR = re.compile(r"CR\s*(1616|1620|2016|2025|2032)", re.IGNORECASE)
BATTERY_REGEX_DIGIT = re.compile(r"(1616|1620|2016|2025|2032)")

# Tesseract配置
TESSERACT_CONFIG = '--psm 6 -c tessedit_char_whitelist=C0123456789R.'


# =============================
# 1. 参数设置
# =============================

def parse_args():
    ap = argparse.ArgumentParser(description="Raspberry Pi 5 — Battery detect + OCR with Tracking")
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model", help="YOLO weights folder/file")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true", help="Use preview configuration")
    
    ap.add_argument("--assume_bgr", action="store_true", help="Treat capture as BGR")

    # OCR控制：推荐 ocr_every=1 或 2，用于传送带快速移动场景
    ap.add_argument("--ocr", action="store_true", help="Enable OCR per detected battery")
    ap.add_argument("--ocr_every", type=int, default=2, help="Run OCR every N frames (1=每帧, 2=每2帧)")
    ap.add_argument("--ocr_pad", type=float, default=0.15, help="Crop padding ratio [0-0.4]")
    
    ap.add_argument("--threads", type=int, default=4, help="NCNN/OMP threads")
    return ap.parse_args()


# =============================
# 2. 辅助函数
# =============================

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label_text, color=(0,255,0)):
    """绘制边界框和标签"""
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    
    cv2.rectangle(img, (x1, y0), (x1+tw+8, y0+th+8), color, -1)
    cv2.putText(img, label_text, (x1+4, y0+th+3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


# =============================
# 3. OCR核心逻辑
# =============================

def _prep_roi(bgr_roi):
    """鲁棒的OCR预处理"""
    if bgr_roi is None or bgr_roi.size == 0:
        return []
    
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    
    # CLAHE处理局部光照
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    
    # 双边滤波保留边缘
    g = cv2.bilateralFilter(g, 5, 50, 50)
    
    # 放大提高识别率
    g2x = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 多重二值化变体
    _, otsu = cv2.threshold(g2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rev_otsu = cv2.bitwise_not(otsu)
    ada = cv2.adaptiveThreshold(g2x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    
    return [g2x, otsu, rev_otsu, ada]


def ocr_battery_crop(bgr_roi, psm=6):
    """
    对裁剪区域运行OCR
    Returns: model_code (str: 如 'CR2032' 或 None), raw_txt (str), conf_est (float)
    """
    if not TESS_OK: 
        return None, "TessMissing", 0.0

    variants = _prep_roi(bgr_roi)
    if not variants: 
        return None, "CropErr", 0.0
    
    best_txt, best_conf = "", -1
    
    for v in variants:
        try:
            d = pytesseract.image_to_data(v, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)
            
            txt = " ".join([w for w in d.get("text", []) if w])
            confs = [float(c) for c in d.get("conf", []) if str(c).isdigit() or (isinstance(c, (int,float)) and c >= 0)]
            conf = sum(confs)/len(confs) if confs else 0.0
            
            clean = ''.join(ch for ch in txt.upper() if ch.isalnum() or ch in 'CR.')

            if conf > best_conf and clean.strip():
                best_txt, best_conf = clean, conf
        except Exception:
            pass 

    # 型号标准化
    model = None
    
    # 优先匹配 CRxxxx
    m = BATTERY_REGEX_CR.search(best_txt)
    if m:
        model = f"CR{m.group(1)}"
    else:
        # 其次匹配裸露的4位数字
        m2 = BATTERY_REGEX_DIGIT.search(best_txt)
        if m2:
            model = f"CR{m2.group(1)}"

    return model, best_txt.strip(), float(best_conf)


# =============================
# 4. 主循环
# =============================

def main():
    args = parse_args()

    # 性能环境变量
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["NCNN_THREADS"] = str(args.threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NCNN_VERBOSE", "0")

    if args.ocr and not TESS_OK:
        print("WARNING: Tesseract import failed; OCR disabled.")

    ensure_dir(args.save_dir)

    # --- 摄像头初始化 ---
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    
    if args.preview:
        config = picam2.create_preview_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"FrameRate": 30}
        )
    else:
        config = picam2.create_video_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"FrameRate": 30}
        )
        
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    # --- 模型加载 ---
    print("Loading model…")
    try:
        model = YOLO(args.weights)
        warm = np.random.randint(0,255,(args.imgsz, args.imgsz, 3), np.uint8)
        _ = model.predict(source=warm, imgsz=args.imgsz, verbose=False)
        print("Model ready.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 循环变量 ---
    last_save = 0.0
    frame_count = 0
    start_time = time.time()
    fps_hist = []

    # OCR性能追踪
    total_ocr_ms = 0.0
    ocr_runs = 0
    
    # 跟踪数据：存储每个track_id的OCR结果
    tracking_data = {} 
    
    # 坐标缩放比例
    sx, sy = main_w/float(args.imgsz), main_h/float(args.imgsz)

    try:
        while True:
            t0 = time.time()
            yuv = picam2.capture_array("main")

            # YUV -> BGR
            if args.assume_bgr:
                frame = yuv
            else:
                try:
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                except Exception:
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

            # 图像推理
            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            r = model.track(source=infer_img, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det_count, low_conf = 0, False
            
            current_track_ids = set()

            if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
                
                # 判断是否在本帧运行OCR
                ocr_sample_frame = args.ocr and TESS_OK and (frame_count % args.ocr_every == 0)
                
                for i in range(len(r.boxes)):
                    b = r.boxes[i]
                    track_id = int(b.id[0])
                    current_track_ids.add(track_id)
                    
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    
                    if (X2-X1)*(Y2-Y1) < args.min_area: 
                        continue

                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    det_count += 1
                    if conf < (args.conf + 0.10): 
                        low_conf = True

                    # 获取历史OCR结果
                    ocr_info = tracking_data.get(track_id, {'model': None, 'conf': 0.0, 'raw': ''})
                    model_code = ocr_info['model']
                    ocr_conf = ocr_info['conf']
                    
                    # === 关键改进：持续更新OCR，不只是第一次 ===
                    if ocr_sample_frame:
                        ocr_t0 = time.time()
                        
                        # 裁剪区域（带Padding）
                        pad = args.ocr_pad
                        w, h = X2 - X1, Y2 - Y1
                        px, py = int(w*pad), int(h*pad)
                        cx1 = max(0, X1 - px)
                        cy1 = max(0, Y1 - py)
                        cx2 = min(main_w, X2 + px)
                        cy2 = min(main_h, Y2 + py)
                        crop = frame[cy1:cy2, cx1:cx2].copy()
                        
                        new_model, raw_txt, conf_est = ocr_battery_crop(crop, psm=6)
                        ocr_ms = (time.time() - ocr_t0) * 1000.0
                        total_ocr_ms += ocr_ms
                        ocr_runs += 1

                        # 更新策略：如果新识别结果更好，或者还没有结果，就更新
                        if new_model and (not model_code or conf_est > ocr_conf):
                            tracking_data[track_id] = {
                                'model': new_model, 
                                'conf': conf_est,
                                'raw': raw_txt
                            }
                            model_code = new_model
                            ocr_conf = conf_est
                            
                    # === 绘制标签：优先显示OCR结果 ===
                    if model_code:
                        # 成功识别：显示型号 + YOLO置信度
                        label_text = f"{model_code} | Conf:{conf:.2f}"
                        color = (0, 255, 0)  # 绿色
                    else:
                        # 未识别：显示Battery + YOLO置信度
                        label_text = f"Battery | Conf:{conf:.2f}"
                        color = (0, 255, 255)  # 黄色
                        
                    draw_box(frame, X1, Y1, X2, Y2, label_text, color)
            
            # 清理不再存在的track_id
            keys_to_delete = [tid for tid in tracking_data if tid not in current_track_ids]
            for tid in keys_to_delete:
                del tracking_data[tid]
            
            # --- HUD / FPS ---
            frame_count += 1
            dt_ms = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt_ms, 1.0))
            if len(fps_hist) > 30: 
                fps_hist.pop(0)
            avg_fps = sum(fps_hist)/len(fps_hist)

            hud = f"Det:{det_count} | Frame:{dt_ms:.1f}ms | {avg_fps:.1f}FPS"
            if args.ocr:
                if TESS_OK:
                    avg_ocr = (total_ocr_ms/ocr_runs) if ocr_runs else 0.0
                    hud += f" | OCR:{avg_ocr:.1f}ms/每{args.ocr_every}帧"
                else:
                    hud += " | OCR:未安装"
            
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # --- 显示 ---
            if not args.headless:
                cv2.imshow("Battery+OCR", frame)
                if cv2.waitKey(1) & 0xFF == 27: 
                    break

            # --- 保存困难案例 ---
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
        import traceback
        traceback.print_exc()
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
        print("Program stopped.")

if __name__ == "__main__":
    main()
