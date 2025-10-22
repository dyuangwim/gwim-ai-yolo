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
    # Set tesseract command path if provided via argument (only needed if not in system PATH)
    import pytesseract
    TESS_OK = True
except Exception:
    TESS_OK = False

# -----------------------------
# Battery model patterns & Config
# -----------------------------
BATTERY_MODELS = ["1616", "1620", "2016", "2025", "2032"]
# 匹配 CRxxxx 或裸露的 xxxx
BATTERY_REGEX_CR = re.compile(r"CR\s*(1616|1620|2016|2025|2032)", re.IGNORECASE)
BATTERY_REGEX_DIGIT = re.compile(r"(1616|1620|2016|2025|2032)")

# Tesseract 配置: 仅允许识别数字、CR 和点。
TESSERACT_CONFIG = '--psm 6 -c tessedit_char_whitelist=C0123456789R.'


# =============================
# 1. 参数设置
# =============================

def parse_args():
    ap = argparse.ArgumentParser(description="Raspberry Pi 5 — Battery detect + OCR with Tracking")
    ap.add_argument("--weights", default="/home/pi/models/best_ncnn_model", help="YOLO weights folder/file (NCNN dir or .pt/.onnx)")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true", help="Use preview configuration for better performance")
    
    # 颜色 / 捕获控制
    ap.add_argument("--assume_bgr", action="store_true", help="Treat capture as BGR (skip YUV->BGR). Use this if your colors are correct only with --assume-bgr.")

    # OCR 控制：OCR_EVERY=1 是默认且推荐的，用于高精度场景。
    ap.add_argument("--ocr", action="store_true", help="Enable OCR per detected battery")
    ap.add_argument("--ocr_every", type=int, default=1, help="Run OCR every N frames. Lower is more accurate, higher is faster.")
    ap.add_argument("--ocr_pad", type=float, default=0.15, help="Crop padding ratio around battery for OCR [0-0.4]")
    
    # 性能/环境
    ap.add_argument("--threads", type=int, default=4, help="NCNN/OMP threads for YOLO")
    return ap.parse_args()


# =============================
# 2. 辅助函数
# =============================

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1, y1, x2, y2, label_text, color=(0,255,0)):
    """Draws bounding box with only the final label text (OCR result + Conf)."""
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    
    txt = label_text
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y0 = max(0, y1 - th - 6)
    
    # 绘制背景和文本
    cv2.rectangle(img, (x1, y0), (x1+tw+8, y0+th+8), color, -1)
    cv2.putText(img, txt, (x1+4, y0+th+3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


# =============================
# 3. OCR 核心逻辑
# =============================

def _prep_roi(bgr_roi):
    """鲁棒的 OCR 预处理: CLAHE, 双边滤波, 放大, 多重二值化变体."""
    if bgr_roi is None or bgr_roi.size == 0:
        return []
    
    # 1. 颜色空间转换与增强
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (限制对比度自适应直方图均衡化) 处理局部光照不均和反光
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    
    # 双边滤波 (Bilateral Filter) 保留边缘同时平滑图像，减少噪点
    g = cv2.bilateralFilter(g, 5, 50, 50)
    
    # 2. 放大 (提高识别率)
    g2x = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 3. 多重二值化变体 (增加鲁棒性)
    _, otsu = cv2.threshold(g2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 反转 Otsu (黑底白字)
    rev_otsu = cv2.bitwise_not(otsu) 
    # 高斯自适应阈值
    ada = cv2.adaptiveThreshold(g2x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    
    return [g2x, otsu, rev_otsu, ada]


def ocr_battery_crop(bgr_roi, psm=6):
    """
    对裁剪区域运行 OCR，并进行后处理。
    Returns: model_code (str: 如 'CR2032' 或 None), raw_txt (str), conf_est (float)
    """
    if not TESS_OK: return None, "TessMissing", 0.0

    variants = _prep_roi(bgr_roi)
    if not variants: return None, "CropErr", 0.0
    
    best_txt, best_conf = "", -1
    
    # 遍历所有预处理变体，找到置信度最高的文本
    for v in variants:
        try:
            # 使用 image_to_data 获取更精细的置信度和文本
            d = pytesseract.image_to_data(v, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)
            
            # 过滤空文本并拼接
            txt = " ".join([w for w in d.get("text", []) if w])
            # 计算非负且是数字的置信度平均值
            confs = [float(c) for c in d.get("conf", []) if str(c).isdigit() or (isinstance(c, (int,float)) and c >= 0)]
            conf = sum(confs)/len(confs) if confs else 0.0
            
            clean = ''.join(ch for ch in txt.upper() if ch.isalnum() or ch in 'CR.')

            if conf > best_conf and clean.strip():
                best_txt, best_conf = clean, conf
        except Exception:
            # Fallback to image_to_string if image_to_data fails (rare)
            pass 

    # --- 后处理和型号标准化 ---
    model = None
    
    # 1. 优先匹配 CRxxxx 格式
    m = BATTERY_REGEX_CR.search(best_txt)
    if m:
        model = f"CR{m.group(1)}"
    else:
        # 2. 其次匹配裸露的 4 位数字
        m2 = BATTERY_REGEX_DIGIT.search(best_txt)
        if m2:
            model = f"CR{m2.group(1)}" # 补上 CR

    return model, best_txt.strip(), float(best_conf)


# =============================
# 4. 主循环
# =============================

def main():
    args = parse_args()

    # 性能环境变量设置
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["NCNN_THREADS"] = str(args.threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NCNN_VERBOSE", "0")

    if args.ocr and not TESS_OK:
        print("WARNING: Tesseract import failed; OCR disabled.")

    ensure_dir(args.save_dir)

    # --- 摄像头初始化 ---
    picam2 = Picamera2()
    main_w, main_h = 1280, 960 # 推荐分辨率
    
    if args.preview:
        config = picam2.create_preview_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"FrameRate": 30} # 锁定帧率
        )
    else:
        config = picam2.create_video_configuration(
            main={"size": (main_w, main_h), "format": "YUV420"},
            controls={"FrameRate": 30} # 锁定帧率
        )
        
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0) # 等待摄像头稳定

    # --- 模型加载 ---
    print("Loading model…")
    try:
        model = YOLO(args.weights)
        # Warmup
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

    # OCR 性能和跟踪
    total_ocr_ms = 0.0
    ocr_runs = 0
    # 核心：用于存储和继承每一帧的 OCR 结果 {track_id: {'model': 'CR2032', 'conf': 95}}
    tracking_data = {} 
    
    # For mapping resized inference back to full-res frame
    sx, sy = main_w/float(args.imgsz), main_h/float(args.imgsz)

    try:
        while True:
            t0 = time.time()
            yuv = picam2.capture_array("main")

            # YUV -> BGR 转换
            if args.assume_bgr:
                 frame = yuv
            else:
                 try:
                     frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                 except Exception:
                     frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12) # Fallback

            # 图像推理
            infer_img = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            # 使用 track 而不是 predict 来启用跟踪功能 (更稳定)
            r = model.track(source=infer_img, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det_count, low_conf = 0, False
            
            # --- 处理新的 tracking_data ---
            current_track_ids = set()

            if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
                
                # 检查是否需要运行 OCR
                ocr_sample_frame = args.ocr and TESS_OK and (frame_count % args.ocr_every == 0)
                
                for i in range(len(r.boxes)):
                    b = r.boxes[i]
                    # 确保是跟踪框 (YOLOv8的track模式下会有 id)
                    track_id = int(b.id[0]) 
                    current_track_ids.add(track_id)
                    
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    
                    # 缩放坐标到原始图像
                    X1, Y1, X2, Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    
                    if (X2-X1)*(Y2-Y1) < args.min_area: continue

                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    cls_id = int(b.cls[0]) if b.cls is not None else 0
                    
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

                    # 默认使用跟踪数据中的信息
                    ocr_info = tracking_data.get(track_id, {'model': None, 'conf': 0.0})
                    model_code = ocr_info['model']
                    ocr_conf = ocr_info['conf']
                    
                    # --- 运行 OCR 并更新跟踪数据 ---
                    if ocr_sample_frame and not model_code:
                        ocr_t0 = time.time()
                        
                        # 裁剪区域 (带 Padding)
                        pad = args.ocr_pad
                        w, h = X2 - X1, Y2 - Y1
                        px, py = int(w*pad), int(h*pad)
                        cx1 = max(0, X1 - px); cy1 = max(0, Y1 - py)
                        cx2 = min(main_w, X2 + px); cy2 = min(main_h, Y2 + py)
                        crop = frame[cy1:cy2, cx1:cx2].copy()
                        
                        model_code, _, conf_est = ocr_battery_crop(crop, psm=6) # 强制 PSM 6/单块
                        ocr_ms = (time.time() - ocr_t0) * 1000.0
                        total_ocr_ms += ocr_ms
                        ocr_runs += 1

                        if model_code:
                            # 只有成功识别出型号时才更新 tracking_data
                            tracking_data[track_id] = {'model': model_code, 'conf': conf_est}
                            ocr_conf = conf_est
                            
                    # --- 绘制标签 ---
                    
                    # 优先显示 OCR 结果，其次显示 YOLO 置信度
                    if model_code:
                        label_text = f"{model_code} | Conf:{conf:.2f}"
                        color = (0, 255, 0) # 绿色: 识别成功
                    else:
                        label_text = f"Battery | Conf:{conf:.2f}"
                        color = (0, 255, 255) # 黄色: 仅检测成功，等待 OCR 结果
                        
                    draw_box(frame, X1, Y1, X2, Y2, label_text, color)
            
            # --- 清理跟踪数据 ---
            # 移除不再在画面中的 track_id
            keys_to_delete = [tid for tid in tracking_data if tid not in current_track_ids]
            for tid in keys_to_delete:
                del tracking_data[tid]
            
            # --- HUD / FPS ---
            frame_count += 1
            dt_ms = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt_ms, 1.0))
            if len(fps_hist) > 30: fps_hist.pop(0)
            avg_fps = sum(fps_hist)/len(fps_hist)

            hud = f"Det:{det_count} | F-Time:{dt_ms:.1f}ms | {avg_fps:.1f}FPS"
            if args.ocr:
                if TESS_OK:
                    avg_ocr = (total_ocr_ms/ocr_runs) if ocr_runs else 0.0
                    hud += f" | OCR Avg:{avg_ocr:.1f}ms (Every {args.ocr_every})"
                else:
                    hud += " | OCR:missing"
            
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # --- 显示 ---
            if not args.headless:
                cv2.imshow("Battery+OCR", frame)
                if cv2.waitKey(1) & 0xFF == 27: break

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
        import traceback; traceback.print_exc()
    finally:
        try: picam2.stop()
        except Exception: pass
        if not args.headless: cv2.destroyAllWindows()
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
