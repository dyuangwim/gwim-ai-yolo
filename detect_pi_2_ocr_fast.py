# detect_pi_2_ocr_fast_v3.py
# Raspberry Pi 5 | YOLO + Battery OCR + Lock + BlueRing | NO Card OCR

import os, time, argparse, re
from datetime import datetime
import cv2, numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# -------------------- OCR 正则与配置 --------------------
BATTERY_REGEX_CR = re.compile(r"CR\s*(1616|1620|2016|2025|2032)", re.I)
BATTERY_REGEX_DIGIT = re.compile(r"(1616|1620|2016|2025|2032)")
TESSERACT_CONFIG = "--oem 1 --psm 11 -c tessedit_char_whitelist=C0123456789R"

# -------------------- CLI 参数 --------------------
def parse_args():
    ap = argparse.ArgumentParser("RPi5 Battery OCR + Version Detect (NO Card OCR)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--assume_bgr", action="store_true")

    # OCR 控制
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ocr_every", type=int, default=6)
    ap.add_argument("--ocr_budget", type=int, default=3)
    ap.add_argument("--ocr_lock_conf", type=float, default=55.0)
    ap.add_argument("--ocr_pad", type=float, default=0.12)

    # 批次预设信息
    ap.add_argument("--expected_model", default="", help="例如 CR2025")
    ap.add_argument("--expected_version", default="", help="v1 / v2 / v3")
    return ap.parse_args()

# -------------------- 工具函数 --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def norm_model_str(s):
    if not s: return ""
    s = s.upper().replace(" ", "")
    m = BATTERY_REGEX_CR.search(s) or BATTERY_REGEX_DIGIT.search(s)
    return "CR" + m.group(1) if m else s

def draw_box(img, x1,y1,x2,y2, label, color=(0,255,0)):
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img,(x1,y0),(x1+tw+8,y0+th+8),color,-1)
    cv2.putText(img,label,(x1+4,y0+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

# -------------------- 蓝环版本检测 --------------------
def classify_version_blue_ring(bgr_roi, inner=0.45, outer=0.85,
                               blue_h=(90,135), sat_min=60, val_min=60,
                               ratio_thr=0.06):
    """通过HSV统计蓝色环形区域比例区分v1/v2"""
    if bgr_roi is None or bgr_roi.size == 0: return "v?"
    h,w = bgr_roi.shape[:2]; s=min(h,w); cx,cy=w//2,h//2
    r_in, r_out = int(s*0.5*inner), int(s*0.5*outer)
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    lower = np.array([blue_h[0], sat_min, val_min], np.uint8)
    upper = np.array([blue_h[1], 255, 255], np.uint8)
    mask_blue = cv2.inRange(hsv, lower, upper)
    yy,xx = np.ogrid[:h,:w]; dist2 = (xx-cx)**2 + (yy-cy)**2
    ring = ((dist2>=r_in*r_in)&(dist2<=r_out*r_out)).astype(np.uint8)*255
    inter = cv2.bitwise_and(mask_blue, ring)
    denom = np.count_nonzero(ring)
    if denom<=0: return "v?"
    ratio = float(np.count_nonzero(inter)) / denom
    return "v2" if ratio >= ratio_thr else "v1"

# -------------------- OCR 函数 --------------------
def try_import_tess():
    try:
        import pytesseract
        return pytesseract
    except Exception:
        return None

def prep_gray(bgr, max_w=220):
    """灰度化+放大+轻度高斯模糊"""
    if bgr is None or bgr.size==0: return None
    h,w = bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),0)
    return gray

def bin_variants(gray):
    """生成多种二值化图像供OCR"""
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ada = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,7)
    return [otsu, cv2.bitwise_not(otsu), ada, cv2.bitwise_not(ada)]

def run_ocr_once(pytesseract, img, timeout_s=1.2):
    try:
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                      config=TESSERACT_CONFIG, timeout=timeout_s)
    except TypeError:
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                      config=TESSERACT_CONFIG)
    txt = " ".join([w for w in d.get("text",[]) if w])
    confs = []
    for c in d.get("conf",[]):
        try:
            f = float(c)
            if f>=0: confs.append(f)
        except: pass
    mean_conf = sum(confs)/len(confs) if confs else 0.0
    clean = ''.join(ch for ch in txt.upper() if ch.isalnum() or ch in 'CR.')
    model=None
    m = BATTERY_REGEX_CR.search(clean)
    if m: model=f"CR{m.group(1)}"
    else:
        m2 = BATTERY_REGEX_DIGIT.search(clean)
        if m2: model=f"CR{m2.group(1)}"
    return model, clean.strip(), float(mean_conf)

def ocr_roi(pytesseract, bgr, lock_thr=55.0):
    """电池ROI OCR，多种阈值 + 180°兜底"""
    gray = prep_gray(bgr, max_w=220)
    if gray is None: return None, "", 0.0
    best = ("", "", 0.0)
    for variant in bin_variants(gray):
        m, raw, mc = run_ocr_once(pytesseract, variant)
        if mc > best[2]:
            best = (m, raw, mc)
        if m and mc >= lock_thr:
            return m, raw, mc
    # 再试180度
    rot = cv2.rotate(gray, cv2.ROTATE_180)
    for variant in bin_variants(rot):
        m, raw, mc = run_ocr_once(pytesseract, variant)
        if mc > best[2]:
            best = (m, raw, mc)
        if m and mc >= lock_thr:
            return m, raw, mc
    return best

# -------------------- 主程序 --------------------
def main():
    args = parse_args()
    os.environ["OMP_NUM_THREADS"]="4"
    os.environ["NCNN_THREADS"]="4"
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("NCNN_VERBOSE","0")
    ensure_dir(args.save_dir)

    pytesseract = try_import_tess()
    if args.ocr and pytesseract is None:
        print("❌ pytesseract 未安装，OCR将被禁用。")
        args.ocr = False

    expected_model = norm_model_str(args.expected_model)
    expected_ver   = args.expected_version.lower().strip() if args.expected_version else ""

    # 初始化相机
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    cfg = (picam2.create_preview_configuration if args.preview else picam2.create_video_configuration)(
        main={"size":(main_w,main_h),"format":"YUV420"}, controls={"FrameRate":30})
    picam2.configure(cfg); picam2.start(); time.sleep(1.0)

    # YOLO
    print("加载模型中…")
    model = YOLO(args.weights)
    _ = model.predict(source=np.zeros((args.imgsz,args.imgsz,3),np.uint8), imgsz=args.imgsz, verbose=False)
    print("✅ 模型加载完成。")

    tracks = {}
    sx, sy = main_w/float(args.imgsz), main_h/float(args.imgsz)
    fps_hist=[]; frame_idx=0; start=time.time()
    total_ocr_ms=0.0; ocr_runs=0; last_save=0.0

    try:
        while True:
            t0 = time.time()
            yuv = picam2.capture_array("main")
            frame = yuv if args.assume_bgr else cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            infer = cv2.resize(frame,(args.imgsz,args.imgsz), interpolation=cv2.INTER_LINEAR)
            r = model.track(source=infer, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det_count, low_conf = 0, False
            ids=set(); budget=args.ocr_budget if args.ocr else 0
            do_ocr = args.ocr and (frame_idx % args.ocr_every == 0)

            if r.boxes is not None and len(r.boxes)>0 and r.boxes.id is not None:
                order = np.argsort((-r.boxes.conf.cpu().numpy()).flatten())
                for i in order:
                    b = r.boxes[int(i)]
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1,Y1,X2,Y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                    if (X2-X1)*(Y2-Y1)<args.min_area: continue
                    conf=float(b.conf[0]) if b.conf is not None else 0.0
                    det_count+=1
                    if conf<(args.conf+0.10): low_conf=True

                    tid=int(b.id[0]); ids.add(tid)
                    info=tracks.get(tid, {"model":None,"mconf":0.0,"locked":False,"ver":None})

                    # 蓝环版本检测（仅一次）
                    if info["ver"] is None:
                        w,h=X2-X1,Y2-Y1
                        shrink=0.15
                        cx1,cy1 = X1+int(w*shrink), Y1+int(h*shrink)
                        cx2,cy2 = X2-int(w*shrink), Y2-int(h*shrink)
                        if cx2>cx1 and cy2>cy1:
                            c_roi = frame[cy1:cy2, cx1:cx2]
                            info["ver"] = classify_version_blue_ring(c_roi)

                    # OCR逻辑
                    if do_ocr and (not info["locked"]) and budget>0:
                        pad=args.ocr_pad
                        w,h=X2-X1,Y2-Y1
                        px,py=int(w*pad),int(h*pad)
                        ox1,oy1=max(0,X1-px),max(0,Y1-py)
                        ox2,oy2=min(main_w,X2+px),min(main_h,Y2+py)
                        crop=frame[oy1:oy2, ox1:ox2]

                        t_ocr0=time.time()
                        m,raw,mc=ocr_roi(pytesseract,crop,lock_thr=args.ocr_lock_conf)
                        ocr_ms=(time.time()-t_ocr0)*1000.0
                        total_ocr_ms+=ocr_ms; ocr_runs+=1; budget-=1

                        if m and (mc>info["mconf"] or not info["model"]):
                            info["model"]=norm_model_str(m)
                            info["mconf"]=mc
                            if mc>=args.ocr_lock_conf: info["locked"]=True

                    # 匹配与颜色
                    label_model=info["model"] if info["model"] else "?"
                    label_ver=info["ver"] if info["ver"] else "v?"
                    color=(0,255,255)
                    if label_model and expected_model:
                        if label_model==expected_model and (not expected_ver or label_ver.lower()==expected_ver):
                            color=(0,255,0)
                        elif label_model==expected_model and expected_ver and label_ver.lower()!=expected_ver:
                            color=(0,255,255)
                        else:
                            color=(0,0,255)
                    label=f"{label_model} {label_ver} | {conf:.2f}"
                    draw_box(frame,X1,Y1,X2,Y2,label,color)
                    tracks[tid]=info

            # 清理离场目标
            for tid in [t for t in list(tracks.keys()) if t not in ids]:
                tracks.pop(tid,None)

            # FPS与HUD
            frame_idx+=1
            dt=(time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0))
            if len(fps_hist)>30: fps_hist.pop(0)
            fps=sum(fps_hist)/len(fps_hist)
            hud=f"Det:{det_count} | {dt:.1f}ms | {fps:.1f}FPS"
            if args.ocr and ocr_runs:
                hud+=f" | OCRavg:{(total_ocr_ms/ocr_runs):.1f}ms"
            cv2.putText(frame,hud,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            if not args.headless:
                cv2.imshow("Battery OCR (No Card)",frame)
                if cv2.waitKey(1)&0xFF==27: break

            # 保存困难样本
            now=time.time()
            if (det_count==0 or low_conf) and (now-last_save>1.0):
                fn=f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir,fn),frame); last_save=now

    finally:
        try: picam2.stop()
        except: pass
        if not args.headless: cv2.destroyAllWindows()
        el=time.time()-start
        print(f"\nFrames:{frame_idx} | Time:{el:.1f}s | FPS:{frame_idx/max(el,1):.1f}")
        if args.ocr and ocr_runs:
            print(f"OCR runs:{ocr_runs} | Avg OCR:{total_ocr_ms/ocr_runs:.1f}ms")
        print("Bye.")

if __name__=="__main__":
    main()
