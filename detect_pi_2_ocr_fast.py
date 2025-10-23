# detect_pi_2_ocr_fast.py
# Raspberry Pi 5 | YOLO + Budgeted OCR + Lock + Blue-Ring Version Check
# - 单进程“限额 + 锁定 + 蓝环”优化版
# - 需要: ultralytics, opencv-python, picamera2, (可选) pytesseract
# - NCNN or onnx/pt 权重皆可；传入 --weights 即可

import os, time, argparse, re
from datetime import datetime
import cv2, numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# ---------------- OCR 正则 & Tesseract 配置 ----------------
BATTERY_REGEX_CR     = re.compile(r"CR\s*(1616|1620|2016|2025|2032)", re.I)
BATTERY_REGEX_DIGIT  = re.compile(r"(1616|1620|2016|2025|2032)")
TESSERACT_CONFIG     = "--psm 7 -c tessedit_char_whitelist=C0123456789R"

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("RPi5 YOLO + budgeted OCR + blue-ring ver")
    ap.add_argument("--weights", required=True, help="模型路径（NCNN 目录/pt/onnx 均可）")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=1800, help="最小框面积过滤")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true", help="使用 create_preview_configuration")
    ap.add_argument("--assume_bgr", action="store_true", help="若相机已返回BGR则启用")

    # OCR 控制
    ap.add_argument("--ocr", action="store_true", help="启用 OCR")
    ap.add_argument("--ocr_every", type=int, default=6, help="每 N 帧尝试 OCR 一次")
    ap.add_argument("--ocr_budget", type=int, default=3, help="每帧最多 OCR 的 ROI 个数")
    ap.add_argument("--ocr_lock_conf", type=float, default=55.0, help="达到该均值置信即锁定")
    ap.add_argument("--ocr_pad", type=float, default=0.12, help="ROI 外扩比例（四周）")

    # 预设批次信息（用于上色判断）
    ap.add_argument("--expected_model", default="", help="例如: CR2025（忽略大小写/空格）")
    ap.add_argument("--expected_version", default="", help="v1 / v2 / v3（目前脚本判断 v1/v2）")

    return ap.parse_args()

# ---------------- 工具 ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1,y1,x2,y2, label, color=(0,255,0)):
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img,(x1,y0),(x1+tw+8,y0+th+8),color,-1)
    cv2.putText(img,label,(x1+4,y0+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def norm_model_str(s: str) -> str:
    if not s: return ""
    s = s.upper().replace(" ", "")
    if s.startswith("CR") and len(s) >= 4:
        return "CR" + s[2:]
    m = BATTERY_REGEX_CR.search(s)
    if m: return "CR" + m.group(1)
    m2 = BATTERY_REGEX_DIGIT.search(s)
    if m2: return "CR" + m2.group(1)
    return s

# ---------------- 蓝环版本识别（启发式） ----------------
def classify_version_blue_ring(bgr_roi, inner=0.45, outer=0.85,
                               blue_h=(90, 135), sat_min=60, val_min=60,
                               ratio_thr=0.06):
    """
    简易蓝环检测：统计环形区域里“蓝色像素”占比。
    inner/outer: 内外环半径相对短边的一半
    blue_h: HSV H 通道蓝色范围（OpenCV 标度 0-180）
    sat_min, val_min: S/V 最小值
    ratio_thr: 超过则判断 v2
    """
    if bgr_roi is None or bgr_roi.size == 0: return "v?"
    h, w = bgr_roi.shape[:2]
    s = min(h, w)
    cx, cy = w // 2, h // 2
    r_in, r_out = int(s * 0.5 * inner), int(s * 0.5 * outer)

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    lower = np.array([blue_h[0], sat_min, val_min], dtype=np.uint8)
    upper = np.array([blue_h[1], 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lower, upper)

    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    ring_mask = ((dist2 >= r_in * r_in) & (dist2 <= r_out * r_out)).astype(np.uint8) * 255

    inter = cv2.bitwise_and(mask_blue, ring_mask)
    ring_count = int(np.count_nonzero(ring_mask))
    if ring_count <= 0: return "v?"

    blue_ratio = float(np.count_nonzero(inter)) / ring_count
    return "v2" if blue_ratio >= ratio_thr else "v1"

# ---------------- OCR ----------------
def try_import_tess():
    try:
        import pytesseract
        return pytesseract
    except Exception:
        return None

def fast_ocr_preproc(bgr_roi, max_w=160):
    if bgr_roi is None or bgr_roi.size == 0:
        return None
    h, w = bgr_roi.shape[:2]
    if w > max_w:
        s = max_w / float(w)
        bgr_roi = cv2.resize(bgr_roi, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    # 单一二值化（Otsu）
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu

def run_ocr_once(pytesseract, roi):
    # 返回 (model_code, raw_text, mean_conf)
    d = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)
    txt = " ".join([w for w in d.get("text", []) if w])
    confs = []
    for c in d.get("conf", []):
        try:
            f = float(c)
            if f >= 0: confs.append(f)
        except: pass
    mean_conf = sum(confs)/len(confs) if confs else 0.0
    clean = ''.join(ch for ch in txt.upper() if ch.isalnum() or ch in 'CR.')
    model = None
    m = BATTERY_REGEX_CR.search(clean)
    if m: model = f"CR{m.group(1)}"
    else:
        m2 = BATTERY_REGEX_DIGIT.search(clean)
        if m2: model = f"CR{m2.group(1)}"
    return model, clean.strip(), float(mean_conf)

# ---------------- 主程序 ----------------
def main():
    args = parse_args()
    os.environ["OMP_NUM_THREADS"] = "4"; os.environ["NCNN_THREADS"] = "4"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("NCNN_VERBOSE", "0")
    ensure_dir(args.save_dir)

    # 预设信息
    expected_model = norm_model_str(args.expected_model)
    expected_ver   = args.expected_version.lower().strip() if args.expected_version else ""

    # OCR 可选
    pytesseract = try_import_tess()
    if args.ocr and pytesseract is None:
        print("WARNING: pytesseract 不可用，OCR 已禁用。")
        args.ocr = False

    # 相机
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    cfg = (picam2.create_preview_configuration if args.preview else picam2.create_video_configuration)(
        main={"size": (main_w, main_h), "format": "YUV420"}, controls={"FrameRate": 30}
    )
    picam2.configure(cfg); picam2.start(); time.sleep(1.0)

    # YOLO
    print("Loading model…")
    model = YOLO(args.weights)
    warm = np.random.randint(0,255,(args.imgsz,args.imgsz,3),np.uint8)
    _ = model.predict(source=warm, imgsz=args.imgsz, verbose=False)
    print("Model ready.")

    # 状态缓存
    # tid -> {model:str|None, mconf:float, raw:str, locked:bool, ver:str|None}
    tracks = {}

    sx, sy = main_w / float(args.imgsz), main_h / float(args.imgsz)
    last_save = 0.0; fps_hist=[]; frame_idx=0; start=time.time()
    total_ocr_ms=0.0; ocr_runs=0

    try:
        while True:
            t0 = time.time()
            yuv = picam2.capture_array("main")
            if args.assume_bgr:
                frame = yuv
            else:
                # 优先 I420，不行再 NV12
                try:
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                except:
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

            infer = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            r = model.track(source=infer, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det_count = 0; low_conf = False
            current_ids = set()
            budget = args.ocr_budget if args.ocr else 0
            do_ocr_this_frame = args.ocr and (frame_idx % args.ocr_every == 0)

            if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
                # 先按置信度排序，优先对高置信度做 OCR（更稳）
                order = np.argsort((-r.boxes.conf.cpu().numpy()).flatten())
                for i in order:
                    b = r.boxes[int(i)]
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1,Y1,X2,Y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                    if (X2-X1)*(Y2-Y1) < args.min_area: continue

                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    det_count += 1
                    if conf < (args.conf + 0.10): low_conf = True

                    tid = int(b.id[0])
                    current_ids.add(tid)
                    info = tracks.get(tid, {"model":None,"mconf":0.0,"raw":"","locked":False,"ver":None})

                    # 蓝环版本只判一次（或未判定时）
                    if info["ver"] is None:
                        # 用略小的中央 ROI，避免背景干扰
                        w, h = X2-X1, Y2-Y1
                        shrink = 0.15
                        cx1 = X1 + int(w*shrink); cy1 = Y1 + int(h*shrink)
                        cx2 = X2 - int(w*shrink); cy2 = Y2 - int(h*shrink)
                        if cx2>cx1 and cy2>cy1:
                            center_roi = frame[cy1:cy2, cx1:cx2]
                            info["ver"] = classify_version_blue_ring(center_roi)

                    # 仅当：到采样帧 + 未锁定 + 预算>0 才跑 OCR
                    if do_ocr_this_frame and (not info["locked"]) and budget>0:
                        pad = args.ocr_pad
                        w, h = X2-X1, Y2-Y1
                        px, py = int(w*pad), int(h*pad)
                        ox1, oy1 = max(0, X1-px), max(0, Y1-py)
                        ox2, oy2 = min(main_w, X2+px), min(main_h, Y2+py)
                        crop = frame[oy1:oy2, ox1:ox2]
                        roi = fast_ocr_preproc(crop, max_w=160)

                        if roi is not None:
                            t_ocr0 = time.time()
                            model_code, raw_txt, mean_conf = run_ocr_once(pytesseract, roi)
                            ocr_ms = (time.time() - t_ocr0) * 1000.0
                            total_ocr_ms += ocr_ms; ocr_runs += 1
                            budget -= 1

                            # 更新更优结果或首次结果
                            if model_code and (mean_conf > info["mconf"] or not info["model"]):
                                info["model"] = norm_model_str(model_code)
                                info["mconf"] = mean_conf
                                info["raw"]   = raw_txt
                                if mean_conf >= args.ocr_lock_conf:
                                    info["locked"] = True

                    # 颜色规则
                    label_model = info["model"] if info["model"] else "?"
                    label_ver   = info["ver"] if info["ver"] else "v?"
                    label_conf  = f"{conf:.2f}"
                    # 匹配判断
                    normalized_expected_model = norm_model_str(expected_model)
                    model_ok = (label_model and normalized_expected_model and (label_model == normalized_expected_model))
                    ver_ok   = (label_ver and expected_ver and (label_ver.lower()==expected_ver))
                    color = (0,255,255)  # 未知→青色
                    if label_model and normalized_expected_model:
                        if model_ok and (expected_ver=="" or ver_ok):
                            color = (0,255,0)     # 全部匹配 → 绿
                        elif model_ok and expected_ver!="" and (not ver_ok):
                            color = (0,255,255)   # 型号对、版本不符 → 黄
                        else:
                            color = (0,0,255)     # 型号不符 → 红

                    # 绘制
                    label = f"{label_model} {label_ver} | {label_conf}"
                    draw_box(frame, X1,Y1,X2,Y2, label, color)
                    tracks[tid] = info

            # 清理离场 track
            dead = [tid for tid in tracks.keys() if tid not in current_ids]
            for tid in dead: tracks.pop(tid, None)

            # HUD & FPS
            frame_idx += 1
            dt_ms = (time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt_ms,1.0))
            if len(fps_hist)>30: fps_hist.pop(0)
            avg_fps = sum(fps_hist)/len(fps_hist)

            hud = f"Det:{det_count} | {dt_ms:.1f}ms | {avg_fps:.1f}FPS"
            if args.ocr:
                hud += f" | OCR_budget:{args.ocr_budget}/F"
                if ocr_runs: hud += f" | OCR_avg:{(total_ocr_ms/ocr_runs):.1f}ms"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            if not args.headless:
                cv2.imshow("Battery + OCR (FAST)", frame)
                if cv2.waitKey(1) & 0xFF == 27: break

            # 保存困难样本
            now = time.time()
            if (det_count==0 or low_conf) and (now-last_save>1.0):
                fn=f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame); last_save=now

            if frame_idx%120==0:
                el = time.time()-start
                print(f"Processed {frame_idx} in {el:.1f}s, Avg FPS: {frame_idx/el:.1f}")
    finally:
        try: picam2.stop()
        except: pass
        if not args.headless: cv2.destroyAllWindows()
        el=time.time()-start
        print("\n=== Summary ===")
        print(f"Frames: {frame_idx} | Time: {el:.1f}s | FPS: {frame_idx/max(el,1):.1f}")
        if args.ocr and ocr_runs:
            print(f"OCR runs: {ocr_runs} | Avg OCR: {total_ocr_ms/ocr_runs:.1f} ms")
        print("Bye.")

if __name__ == "__main__":
    main()
