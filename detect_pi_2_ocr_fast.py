# detect_pi_2_ocr_paddle_v2.py
# RPi5 | YOLO + PaddleOCR(英文) | 只读电池表面 | 限额+锁定 | 颜色规则(绿/红/白)
# - 兼容新版 PaddleOCR：不再传 cls；优先使用 predict()，否则退回 ocr()
# - 兼容两种返回结构（老版list嵌套，或新版dict/tuple结构）

import os, time, argparse, re
from datetime import datetime
import cv2, numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# ---------------- 型号抽取规则 ----------------
RE_CR     = re.compile(r"CR\s*(1616|1620|2016|2025|2032)", re.I)
RE_DIGITS = re.compile(r"(1616|1620|2016|2025|2032)")

def norm_model(s: str) -> str:
    if not s: return ""
    s = s.upper().replace(" ", "")
    m = RE_CR.search(s) or RE_DIGITS.search(s)
    return "CR" + m.group(1) if m else ""

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("RPi5 YOLO + PaddleOCR (battery surface only)")
    ap.add_argument("--weights", required=True, help="YOLO 权重（NCNN目录/pt/onnx）")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--assume_bgr", action="store_true")

    # OCR 调度
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ocr_every", type=int, default=6, help="每 N 帧尝试 OCR 一次")
    ap.add_argument("--ocr_budget", type=int, default=3, help="每帧最多 OCR ROI 数")
    ap.add_argument("--ocr_lock_conf", type=float, default=0.55, help="达到该置信度即锁定(0~1)")
    ap.add_argument("--ocr_pad", type=float, default=0.12, help="ROI 外扩比例")

    # 目标批次（用于上色）
    ap.add_argument("--expected_model", default="", help="例如 CR2025；留空则全部白色为未知")
    return ap.parse_args()

# ---------------- 绘制 ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1,y1,x2,y2, label, color=(255,255,255)):
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(img,(x1,y0),(x1+tw+8,y0+th+8),color,-1)
    cv2.putText(img,label,(x1+4,y0+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

# ---------------- PaddleOCR ----------------
def init_paddle_ocr():
    # 新版弃用 use_angle_cls；改为 use_textline_orientation
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_textline_orientation=False, lang='en')  # 不传 show_log / cls
    return ocr

def prep_roi_for_ocr(bgr, max_w=240):
    """缩放到适中宽度+轻度去噪，返回 RGB（PaddleOCR 接受 RGB ndarray）"""
    if bgr is None or bgr.size==0: return None
    h,w = bgr.shape[:2]
    if w > max_w:
        s = max_w/float(w)
        bgr = cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
    bgr = cv2.GaussianBlur(bgr,(3,3),0)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _call_paddle_ocr(ocr, rgb_img):
    """
    适配不同 PaddleOCR 版本：
    - 新版推荐 ocr.predict(img)
    - 旧版是 ocr.ocr(img)
    都返回一个 Python 对象，我们后面用 _parse_ocr_result 统一解析
    """
    if hasattr(ocr, "predict"):
        return ocr.predict(rgb_img)   # 不传 cls
    else:
        return ocr.ocr(rgb_img)       # 旧接口

def _parse_ocr_result(res):
    """
    解析 PaddleOCR 的返回，兼容多版本：
    目标：拿到 texts(list[str]) 和每段文字对应的概率 probs(list[float])
    可能的形态：
      A) 旧：res -> [ [ [box, (text, prob)], ... ] ]
      B) 新：res -> [ { 'text':..., 'score':..., 'points':... }, ... ] 或类似
      C) 新：res -> [ [ (points), (text, prob) ], ... ]（某些组合包）
    """
    texts = []
    probs = []

    if res is None:
        return texts, probs

    # 如果是 [[...]] 的嵌套，先把第一层取出来
    if isinstance(res, list) and len(res) == 1 and isinstance(res[0], list):
        res = res[0]

    # 逐条解析
    if isinstance(res, list):
        for item in res:
            # 旧版：item = [box, (text, prob)]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cand = item[1]
                if isinstance(cand, (list, tuple)) and len(cand) >= 2 and isinstance(cand[0], str):
                    texts.append(cand[0].strip())
                    try: probs.append(float(cand[1]))
                    except: probs.append(0.0)
                    continue
            # 新版：item 是 dict
            if isinstance(item, dict):
                txt = item.get('text') or item.get('transcription') or item.get('label') or ""
                sc  = item.get('score') or item.get('confidence') or item.get('prob') or 0.0
                texts.append(str(txt).strip())
                try: probs.append(float(sc))
                except: probs.append(0.0)
                continue
    return texts, probs

def paddle_ocr_once(ocr, rgb_img):
    """
    返回 (model_code, raw_text, conf)
    conf：命中型号文本的最大概率；若未命中，返回文本中最高概率（仅参考）
    """
    res = _call_paddle_ocr(ocr, rgb_img)
    texts, probs = _parse_ocr_result(res)
    if not texts:
        return "", "", 0.0

    raw = " ".join(texts).upper()
    best_model = ""
    best_prob  = 0.0
    for t, p in zip(texts, probs if probs else [0.0]*len(texts)):
        m = RE_CR.search(t) or RE_DIGITS.search(t)
        if m and float(p) > best_prob:
            best_prob  = float(p)
            best_model = t

    model = norm_model(best_model if best_model else raw)
    conf  = float(best_prob if model else (max(probs) if probs else 0.0))
    return model, raw, conf

# ---------------- 主程序 ----------------
def main():
    args = parse_args()
    os.environ["OMP_NUM_THREADS"]="4"; os.environ["NCNN_THREADS"]="4"
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("NCNN_VERBOSE","0")
    ensure_dir(args.save_dir)

    expected_model = norm_model(args.expected_model)

    # OCR
    if args.ocr:
        try:
            ocr = init_paddle_ocr()
        except Exception as e:
            print("❌ PaddleOCR 初始化失败：", e)
            args.ocr = False
            ocr = None
    else:
        ocr = None

    # Camera
    picam2 = Picamera2()
    main_w, main_h = 1280, 960
    cfg = (picam2.create_preview_configuration if args.preview else picam2.create_video_configuration)(
        main={"size":(main_w,main_h),"format":"YUV420"}, controls={"FrameRate":30}
    )
    picam2.configure(cfg); picam2.start(); time.sleep(1.0)

    # YOLO
    print("Loading YOLO…")
    model = YOLO(args.weights)
    warm = np.zeros((args.imgsz,args.imgsz,3), np.uint8)
    _ = model.predict(source=warm, imgsz=args.imgsz, verbose=False)
    print("✅ YOLO ready.")

    # 状态缓存：tid -> {"model":str , "conf":float, "locked":bool}
    tracks = {}
    sx, sy = main_w/float(args.imgsz), main_h/float(args.imgsz)
    last_save=0.0; fps_hist=[]; frame_idx=0; start=time.time()
    total_ocr_ms=0.0; ocr_runs=0

    try:
        while True:
            t0=time.time()
            yuv = picam2.capture_array("main")
            frame = yuv if args.assume_bgr else cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            infer = cv2.resize(frame,(args.imgsz,args.imgsz), interpolation=cv2.INTER_LINEAR)
            r = model.track(source=infer, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det=0; low=False; ids=set()
            budget = args.ocr_budget if args.ocr else 0
            do_ocr = args.ocr and (frame_idx % args.ocr_every == 0)

            if r.boxes is not None and len(r.boxes)>0 and r.boxes.id is not None:
                order = np.argsort((-r.boxes.conf.cpu().numpy()).flatten())
                for i in order:
                    b = r.boxes[int(i)]
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    X1,Y1,X2,Y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                    if (X2-X1)*(Y2-Y1) < args.min_area: continue
                    conf = float(b.conf[0]) if b.conf is not None else 0.0
                    det += 1
                    if conf < (args.conf+0.10): low=True

                    tid = int(b.id[0]); ids.add(tid)
                    info = tracks.get(tid, {"model":"", "conf":0.0, "locked":False})

                    # OCR（只读电池ROI；达到阈值锁定，不再重复）
                    if do_ocr and (not info["locked"]) and budget>0 and ocr is not None:
                        pad=args.ocr_pad; w,h = X2-X1, Y2-Y1
                        px,py = int(w*pad), int(h*pad)
                        ox1,oy1 = max(0, X1-px), max(0, Y1-py)
                        ox2,oy2 = min(main_w, X2+px), min(main_h, Y2+py)
                        crop = frame[oy1:oy2, ox1:ox2]

                        rgb = prep_roi_for_ocr(crop, max_w=240)
                        if rgb is not None:
                            t_ocr0=time.time()
                            m, raw, prob = paddle_ocr_once(ocr, rgb)
                            ocr_ms=(time.time()-t_ocr0)*1000.0
                            total_ocr_ms+=ocr_ms; ocr_runs+=1; budget-=1

                            if m and (prob>info["conf"] or not info["model"]):
                                info["model"]=m
                                info["conf"]=prob
                                if prob >= args.ocr_lock_conf:
                                    info["locked"]=True

                    # 上色与标签（绿=匹配；红=读到但不匹配；白=未知或未设 expected）
                    label_model = info["model"] if info["model"] else "?"
                    color = (255,255,255)
                    if expected_model:
                        if info["model"] == expected_model:
                            color = (0,255,0)
                        elif info["model"]:
                            color = (0,0,255)
                        else:
                            color = (255,255,255)

                    label = f"{label_model} | {conf:.2f}"
                    draw_box(frame, X1,Y1,X2,Y2, label, color)
                    tracks[tid]=info

            # 清理离场
            for tid in [t for t in list(tracks.keys()) if t not in ids]:
                tracks.pop(tid, None)

            # HUD
            frame_idx+=1
            dt=(time.time()-t0)*1000.0
            fps_hist.append(1000.0/max(dt,1.0))
            if len(fps_hist)>30: fps_hist.pop(0)
            fps=sum(fps_hist)/len(fps_hist)
            hud=f"Det:{det} | {dt:.1f}ms | {fps:.1f}FPS"
            if args.ocr and ocr_runs:
                hud+=f" | OCRavg:{(total_ocr_ms/ocr_runs):.1f}ms"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            if not args.headless:
                cv2.imshow("Battery+PaddleOCR", frame)
                if cv2.waitKey(1)&0xFF==27: break

            # hard cases
            now=time.time()
            if (det==0 or low) and (now-last_save>1.0):
                fn=f"hard_{det}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame); last_save=now

    finally:
        try: picam2.stop()
        except: pass
        if not args.headless: cv2.destroyAllWindows()
        el=time.time()-start
        print(f"\nFrames:{frame_idx} | Time:{el:.1f}s | FPS:{frame_idx/max(el,1):.1f}")
        if args.ocr and ocr_runs:
            print(f"OCR runs:{ocr_runs} | Avg OCR:{total_ocr_ms/ocr_runs:.1f} ms")
        print("Bye.")

if __name__=="__main__":
    main()
