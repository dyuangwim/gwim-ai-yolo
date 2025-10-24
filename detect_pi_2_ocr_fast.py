# detect_pi_2_ocr_paddle_v5_probe.py
# YOLO + PaddleOCR(ONNX) | 只读电池表面 | 限额+锁定 | 颜色规则 | 全面探针：HUD+日志+保存裁图

import os, time, argparse, re
from datetime import datetime
import cv2, numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2

# ---------- 型号抽取 ----------
RE_CR     = re.compile(r"CR\s*(1616|1620|2016|2025|2032)", re.I)
RE_DIGITS = re.compile(r"(1616|1620|2016|2025|2032)")
def norm_model(s: str) -> str:
    if not s: return ""
    s = s.upper().replace(" ", "")
    m = RE_CR.search(s) or RE_DIGITS.search(s)
    return "CR" + m.group(1) if m else ""

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser("RPi5 YOLO + PaddleOCR(ONNX) battery-surface-only (probe)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--assume_bgr", action="store_true")

    # OCR 调度
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ocr_every", type=int, default=8)
    ap.add_argument("--ocr_budget", type=int, default=2)
    ap.add_argument("--ocr_lock_conf", type=float, default=0.55)  # 0~1
    ap.add_argument("--ocr_pad", type=float, default=0.10)
    ap.add_argument("--center_shrink", type=float, default=0.18, help="中心收缩比例(0~0.4)")

    # 目标
    ap.add_argument("--expected_model", default="")

    # 调试/可视化/保存
    ap.add_argument("--ocr_show_raw", action="store_true")
    ap.add_argument("--log_ocr", action="store_true")
    ap.add_argument("--ocr_dump", action="store_true")
    ap.add_argument("--ocr_dump_dir", default="/home/pi/ocr_dumps")
    ap.add_argument("--debug_ocr_once", action="store_true")
    return ap.parse_args()

# ---------- 绘制 ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1,y1,x2,y2, label, color=(255,255,255)):
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    y0=max(0,y1-th-6)
    cv2.rectangle(img,(x1,y0),(x1+tw+8,y0+th+8),color,-1)
    cv2.putText(img,label,(x1+4,y0+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def draw_small_text(img, x, y, text, color=(255,255,255)):
    maxw = 36
    lines=[]; t=text
    while len(t)>maxw:
        lines.append(t[:maxw]); t=t[maxw:]
    if t: lines.append(t)
    y0=y
    for ln in lines:
        cv2.putText(img, ln, (x,y0), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)
        y0+=12

# ---------- PaddleOCR (ONNX backend) ----------
def init_paddle_ocr_onnx():
    from paddleocr import PaddleOCR
    return PaddleOCR(use_textline_orientation=False, lang='en', use_onnx=True)

def _unsharp_mask(gray, amount=1.0, radius=3):
    blur = cv2.GaussianBlur(gray, (radius|1, radius|1), 0)
    return cv2.addWeighted(gray, 1+amount, blur, -amount, 0)

def _clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def prep_roi_for_ocr(bgr, max_w=320, gamma=1.1):
    """放大→灰度→CLAHE→unsharp→轻去噪→RGB；返回两角度(0°,180°) + 中间过程图"""
    if bgr is None or bgr.size==0: return None, None, None, None
    h,w=bgr.shape[:2]
    if w>max_w:
        s=max_w/float(w)
        bgr=cv2.resize(bgr,(int(w*s),int(h*s)),interpolation=cv2.INTER_LINEAR)

    if gamma!=1.0:
        table=np.array([((i/255.0)**(1.0/gamma))*255 for i in np.arange(256)]).astype("uint8")
        bgr=cv2.LUT(bgr, table)

    gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray=_clahe(gray)
    gray=_unsharp_mask(gray, amount=1.0, radius=3)
    gray=cv2.GaussianBlur(gray,(3,3),0)

    rgb0=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb180=cv2.rotate(rgb0, cv2.ROTATE_180)
    return rgb0, rgb180, bgr, gray

def _call_paddle_ocr(ocr, rgb_img):
    if hasattr(ocr,"predict"):
        return ocr.predict(rgb_img)
    return ocr.ocr(rgb_img)

def _parse_ocr_result(res):
    texts, probs=[], []
    if res is None: return texts, probs
    if isinstance(res, list) and len(res)==1 and isinstance(res[0], list):
        res=res[0]
    if isinstance(res, list) and len(res)>0 and isinstance(res[0], dict):
        for item in res:
            if "rec" in item and isinstance(item["rec"], (list,tuple)):
                for r in item["rec"]:
                    if isinstance(r,(list,tuple)) and len(r)>=2:
                        texts.append(str(r[0]).strip())
                        try: probs.append(float(r[1]))
                        except: probs.append(0.0)
            elif "rec_res" in item and isinstance(item["rec_res"], (list,tuple)):
                for r in item["rec_res"]:
                    if isinstance(r,(list,tuple)) and len(r)>=2:
                        texts.append(str(r[0]).strip())
                        try: probs.append(float(r[1]))
                        except: probs.append(0.0)
            else:
                txt=item.get('text') or item.get('transcription') or item.get('label') or ""
                sc =item.get('score') or item.get('confidence') or item.get('prob') or 0.0
                if txt!="":
                    texts.append(str(txt).strip())
                    try: probs.append(float(sc))
                    except: probs.append(0.0)
        return texts, probs
    if isinstance(res, list):
        for it in res:
            if isinstance(it,(list,tuple)) and len(it)>=2:
                cand=it[1]
                if isinstance(cand,(list,tuple)) and len(cand)>=2 and isinstance(cand[0],str):
                    texts.append(cand[0].strip())
                    try: probs.append(float(cand[1]))
                    except: probs.append(0.0)
    return texts, probs

def paddle_ocr_once(ocr, rgb_img, debug_once=False, printed=[False]):
    """一定返回 ran=True，只要函数被调用；即使 texts 为空。"""
    res=_call_paddle_ocr(ocr, rgb_img)
    texts, probs=_parse_ocr_result(res)
    if (not texts) and debug_once and (not printed[0]):
        printed[0]=True
        try: print("DEBUG OCR raw result (truncated):", str(res)[:400])
        except Exception: print("DEBUG OCR raw result: <unprintable>")
    if not texts:
        return True, "", "", 0.0  # ran=True, raw为空
    raw=" ".join(texts).upper()
    best_model=""; best_prob=0.0
    for t,p in zip(texts, probs if probs else [0.0]*len(texts)):
        m=RE_CR.search(t) or RE_DIGITS.search(t)
        if m and float(p)>best_prob:
            best_model=t; best_prob=float(p)
    model=norm_model(best_model if best_model else raw)
    conf=float(best_prob if model else (max(probs) if probs else 0.0))
    return True, model, raw, conf

# ---------- 主程序 ----------
def main():
    args=parse_args()
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"
    os.environ["NCNN_THREADS"]="4"; os.environ.setdefault("NCNN_VERBOSE","0")
    ensure_dir(args.save_dir)
    if args.ocr_dump: ensure_dir(args.ocr_dump_dir)

    expected_model=norm_model(args.expected_model)

    # OCR init
    ocr=None; ocr_backend="OFF"
    if args.ocr:
        try:
            ocr=init_paddle_ocr_onnx()
            ocr_backend="ONNX"
            print(f"✅ PaddleOCR init OK. backend={ocr_backend}")
        except Exception as e:
            print("❌ PaddleOCR init failed:", e)
            args.ocr=False
    else:
        print("ℹ️ OCR disabled (no --ocr)")

    # Camera
    picam2=Picamera2()
    main_w, main_h=1280, 960
    cfg=(picam2.create_preview_configuration if args.preview else picam2.create_video_configuration)(
        main={"size":(main_w,main_h),"format":"YUV420"}, controls={"FrameRate":30})
    picam2.configure(cfg); picam2.start(); time.sleep(1.0)

    # YOLO
    print("Loading YOLO…")
    model=YOLO(args.weights)
    _=model.predict(source=np.zeros((args.imgsz,args.imgsz,3),np.uint8), imgsz=args.imgsz, verbose=False)
    print("✅ YOLO ready.")

    tracks={}
    sx, sy=main_w/float(args.imgsz), main_h/float(args.imgsz)
    last_save=0.0; fps_hist=[]; frame_idx=0; start=time.time()
    total_ocr_ms=0.0; ocr_runs=0; ocr_calls_total=0

    try:
        while True:
            t0=time.time()
            yuv=picam2.capture_array("main")
            frame=yuv if args.assume_bgr else cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            infer=cv2.resize(frame,(args.imgsz,args.imgsz), interpolation=cv2.INTER_LINEAR)
            r=model.track(source=infer, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det=0; low=False; ids=set()
            budget=args.ocr_budget if args.ocr else 0
            do_ocr = args.ocr and (frame_idx % args.ocr_every == 0)

            if r.boxes is not None and len(r.boxes)>0 and r.boxes.id is not None:
                order=np.argsort((-r.boxes.conf.cpu().numpy()).flatten())
                for i in order:
                    b=r.boxes[int(i)]
                    x1,y1,x2,y2=map(int, b.xyxy[0].tolist())
                    X1,Y1,X2,Y2=int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                    if (X2-X1)*(Y2-Y1)<args.min_area: continue
                    conf=float(b.conf[0]) if b.conf is not None else 0.0
                    det+=1
                    if conf<(args.conf+0.10): low=True

                    tid=int(b.id[0]); ids.add(tid)
                    info=tracks.get(tid, {"model":"", "conf":0.0, "locked":False, "raw":"", "tried":False})

                    # OCR：只读中心ROI
                    if do_ocr and (not info["locked"]) and budget>0 and ocr is not None:
                        pad=args.ocr_pad; w,h=X2-X1, Y2-Y1
                        px,py=int(w*pad),int(h*pad)
                        ox1,oy1=max(0,X1-px), max(0,Y1-py)
                        ox2,oy2=min(main_w,X2+px), min(main_h,Y2+py)
                        crop=frame[oy1:oy2, ox1:ox2]

                        shrink=max(0.0, min(0.4, args.center_shrink))
                        cw,ch=ox2-ox1, oy2-oy1
                        cx1=ox1+int(cw*shrink); cy1=oy1+int(ch*shrink)
                        cx2=ox2-int(cw*shrink); cy2=oy2-int(ch*shrink)
                        if cx2>cx1 and cy2>cy1:
                            crop=frame[cy1:cy2, cx1:cx2]

                        rgb0, rgb180, bgr_scaled, gray_scaled = prep_roi_for_ocr(crop, max_w=320, gamma=1.1)
                        if rgb0 is not None:
                            ocr_calls_total += 1
                            info["tried"]=True
                            t_ocr0=time.time()
                            ran0, m0, raw0, p0 = paddle_ocr_once(ocr, rgb0, debug_once=args.debug_ocr_once)
                            ran1, m1, raw1, p1 = paddle_ocr_once(ocr, rgb180, debug_once=False)
                            if p1>p0: m, raw, prob = m1, raw1, p1
                            else:     m, raw, prob = m0, raw0, p0
                            ocr_ms=(time.time()-t_ocr0)*1000.0
                            total_ocr_ms+=ocr_ms; ocr_runs+=1; budget-=1

                            # 没文本也要有占位，方便显示
                            if raw=="":
                                raw="[EMPTY]"
                            info["raw"]=raw

                            if args.log_ocr:
                                print(f"[OCR] tid={tid} tried=YES ms={ocr_ms:.1f} prob={prob:.3f} model={m or '?'} raw='{raw}'")

                            if args.ocr_dump:
                                ts=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                base=f"tid{tid}_{prob:.2f}_{(m or 'UNK')}_{ts}"
                                cv2.imwrite(os.path.join(args.ocr_dump_dir, base+"_crop.jpg"), crop)
                                if bgr_scaled is not None:
                                    cv2.imwrite(os.path.join(args.ocr_dump_dir, base+"_scaled.jpg"), bgr_scaled)
                                if gray_scaled is not None:
                                    cv2.imwrite(os.path.join(args.ocr_dump_dir, base+"_gray.jpg"), gray_scaled)

                            if m and (prob>info["conf"] or not info["model"]):
                                info["model"]=m; info["conf"]=prob
                                if prob>=args.ocr_lock_conf: info["locked"]=True

                    # 颜色规则
                    label_model = info["model"] if info["model"] else "?"
                    color=(255,255,255)
                    if expected_model:
                        if info["model"] == expected_model: color=(0,255,0)
                        elif info["model"]: color=(0,0,255)

                    label=f"{label_model} | {conf:.2f}"
                    draw_box(frame, X1,Y1,X2,Y2, label, color)

                    # 画 RAW 或占位
                    if args.ocr_show_raw:
                        raw_short = info.get("raw","")
                        if len(raw_short)>42: raw_short=raw_short[:42]+"…"
                        draw_small_text(frame, X1+4, Y2+14, "RAW: "+(raw_short if raw_short else "[NO-OCR]"), (200,200,200))

                    tracks[tid]=info

            # 清理离场
            for tid in [t for t in list(tracks.keys()) if t not in ids]:
                tracks.pop(tid, None)

            # HUD/FPS + OCR 状态
            frame_idx+=1
            dt=(time.time()-t0)*1000.0
            fps=(1000.0/max(dt,1.0))
            fps_hist.append(fps)
            if len(fps_hist)>30: fps_hist.pop(0)
            fps_avg=sum(fps_hist)/len(fps_hist)

            hud=f"Det:{det} | {dt:.1f}ms | {fps_avg:.1f}FPS"
            hud+=f" | OCR:{'ON' if args.ocr else 'OFF'}/{ocr_backend}"
            hud+=f" every:{args.ocr_every} budget:{args.ocr_budget} calls:{ocr_calls_total}"
            if args.ocr and ocr_runs: hud+=f" | OCRavg:{(total_ocr_ms/ocr_runs):.1f}ms"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            if not args.headless:
                cv2.imshow("Battery+PaddleOCR(ONNX)-v5probe", frame)
                if cv2.waitKey(1)&0xFF==27: break

            # 保存困难样本
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
