# detect_pi_2_ocr_async.py — RPi5 YOLO (NCNN) + Async OCR (PaddleOCR ONNX→fallback Tesseract)
# 目标：把OCR放到独立进程，不阻塞主检测。支持“限额+锁定”、批量16颗、颜色规则（预设型号=青色、错误=红色）。
# 参考你现有脚本：detect_pi2.py（仅YOLO快）与 detect_pi_2_ocr.py（同步OCR较慢）和 detect_pi_2_ocr_fast 的限额/锁定思路。
#
# 快速使用：
#   python3 detect_pi_2_ocr_async.py \
#       --weights /home/pi/models/best_ncnn_model \
#       --imgsz 320 --conf 0.30 --expected_model CR2025 --ocr --ocr_every 8 \
#       --ocr_budget 2 --ocr_lock_conf 0.55 --center_shrink 0.18 --preview
#
# 关键参数：
#   --ocr_every N      每 N 帧触发一次 OCR 调度（默认8）
#   --ocr_budget K     每次调度最多K个ROI进入OCR队列（默认2）
#   --ocr_lock_conf P  概率>=P即“锁定”，不再重复OCR（默认0.55）
#   --center_shrink R  对ROI做中心收缩 R（0~0.4，默认0.18），只读表面文字区
#   --expected_model   预设型号（如 CR2025）。匹配=青色，不匹配=红色；未知=白/黄
#
# 依赖：
#   pip install ultralytics paddleocr opencv-python-headless picamera2 numpy
#   （PaddleOCR 在RPi上用CPU+ONNX后端；若PaddleOCR不可用，自动降级用pytesseract）

import os, time, argparse, re, sys
from datetime import datetime
import cv2, numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import multiprocessing as mp
import queue

# ---------- 型号解析 ----------
RE_CR     = re.compile(r"CR\s*(1616|1620|2016|2025|2032|1650)", re.I)
RE_DIGITS = re.compile(r"(1616|1620|2016|2025|2032|1650)")

def norm_model(s: str) -> str:
    if not s: return ""
    s = s.upper().replace(" ", "")
    m = RE_CR.search(s) or RE_DIGITS.search(s)
    return "CR" + m.group(1) if m else ""

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser("RPi5 YOLO + Async OCR (Paddle→Tesseract) with budget+lock")
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
    ap.add_argument("--ocr_lock_conf", type=float, default=0.55)
    ap.add_argument("--ocr_pad", type=float, default=0.10)
    ap.add_argument("--center_shrink", type=float, default=0.18, help="中心收缩比例(0~0.4)")
    ap.add_argument("--ocr_dump", action="store_true")
    ap.add_argument("--ocr_dump_dir", default="/home/pi/ocr_dumps")

    # 目标
    ap.add_argument("--expected_model", default="")

    # 调试/显示
    ap.add_argument("--ocr_show_raw", action="store_true")
    ap.add_argument("--debug_ocr_once", action="store_true")

    # 线程/性能
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()

# ---------- 画图 ----------

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1,y1,x2,y2, label, color=(255,255,255)):
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    y0=max(0,y1-th-6)
    cv2.rectangle(img,(x1,y0),(x1+tw+8,y0+th+8),color,-1)
    cv2.putText(img,label,(x1+4,y0+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def draw_small_text(img, x, y, text, color=(200,200,200)):
    maxw=40
    t=text
    while t:
        ln=t[:maxw]
        cv2.putText(img, ln, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)
        y+=12
        t=t[maxw:]

# ---------- OCR Worker 实现 ----------

# 轻量预处理：放大→灰度→CLAHE→锐化→轻去噪→两角度

def _unsharp_mask(gray, amount=1.0, radius=3):
    blur = cv2.GaussianBlur(gray, (radius|1, radius|1), 0)
    return cv2.addWeighted(gray, 1+amount, blur, -amount, 0)


def prep_roi_for_ocr(bgr, max_w=320, gamma=1.1):
    if bgr is None or bgr.size==0: return None, None, None, None
    h,w=bgr.shape[:2]
    if w>max_w:
        s=max_w/float(w)
        bgr=cv2.resize(bgr,(int(w*s),int(h*s)),interpolation=cv2.INTER_LINEAR)
    if gamma!=1.0:
        table=np.array([((i/255.0)**(1.0/gamma))*255 for i in np.arange(256)]).astype("uint8")
        bgr=cv2.LUT(bgr, table)
    gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray=clahe.apply(gray)
    gray=_unsharp_mask(gray, amount=1.0, radius=3)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    rgb0=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb180=cv2.rotate(rgb0, cv2.ROTATE_180)
    return rgb0, rgb180, bgr, gray


def _call_paddle_ocr(ocr, rgb_img):
    if hasattr(ocr,"predict"): return ocr.predict(rgb_img)
    return ocr.ocr(rgb_img)


def _parse_ocr_result(res):
    texts, probs=[], []
    if res is None: return texts, probs
    # 兼容不同返回结构
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
                if txt:
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
    res=_call_paddle_ocr(ocr, rgb_img)
    texts, probs=_parse_ocr_result(res)
    if (not texts) and debug_once and (not printed[0]):
        printed[0]=True
        try: print("DEBUG OCR raw result (truncated):", str(res)[:400])
        except Exception: print("DEBUG OCR raw result: <unprintable>")
    if not texts:
        return True, "", "", 0.0
    raw=" ".join(texts).upper()
    best_model=""; best_prob=0.0
    for t,p in zip(texts, probs if probs else [0.0]*len(texts)):
        m=RE_CR.search(t) or RE_DIGITS.search(t)
        if m and float(p)>best_prob:
            best_model=t; best_prob=float(p)
    model=norm_model(best_model if best_model else raw)
    conf=float(best_prob if model else (max(probs) if probs else 0.0))
    return True, model, raw, conf


def tesseract_ocr_once(bgr_img):
    try:
        import pytesseract
    except Exception:
        return True, "", "", 0.0
    gray=cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    _,bin_img=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    d=pytesseract.image_to_data(bin_img, output_type=pytesseract.Output.DICT,
                                config='--psm 6 -c tessedit_char_whitelist=C0123456789R.')
    texts=[w for w in d.get("text",[]) if w.strip()]
    confs=[float(c) for c in d.get("conf",[]) if str(c).isdigit() or (isinstance(c,(int,float)) and float(c)>=0)]
    if not texts:
        return True, "", "", 0.0
    raw=" ".join(texts).upper()
    m=RE_CR.search(raw) or RE_DIGITS.search(raw)
    model=norm_model(m.group(0) if m else raw)
    conf=float(sum(confs)/len(confs)) if confs else 0.0
    return True, model, raw, conf


def ocr_worker(input_q: mp.Queue, output_q: mp.Queue, backend: str, debug_once: bool=False):
    """独立进程：初始化OCR后循环消费任务。任务数据结构：
    {
      'tid': int, 'crop': np.ndarray(BGR), 'ts': float
    }
    输出：{'tid':int, 'ran':bool, 'model':str, 'raw':str, 'prob':float, 'ms':float}
    """
    ocr=None; backend_used="OFF"
    if backend.upper()=="PADDLE":
        try:
            from paddleocr import PaddleOCR
            ocr=PaddleOCR(use_textline_orientation=False, lang='en', use_onnx=True)
            backend_used="PADDLE"
            print("[OCR-Worker] PaddleOCR init OK (ONNX)")
        except Exception as e:
            print("[OCR-Worker] Paddle init failed, fallback to Tesseract:", e)
            backend_used="TESSERACT"
    else:
        backend_used="TESSERACT"

    while True:
        try:
            task=input_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if task is None:
            break  # poison pill
        crop=task['crop']
        t0=time.time()
        try:
            if backend_used=="PADDLE" and ocr is not None:
                rgb0, rgb180, bgr_scaled, gray_scaled = prep_roi_for_ocr(crop, max_w=320, gamma=1.1)
                if rgb0 is None:
                    ran, model, raw, prob = True, "", "", 0.0
                else:
                    ran0, m0, raw0, p0 = paddle_ocr_once(ocr, rgb0, debug_once=debug_once)
                    ran1, m1, raw1, p1 = paddle_ocr_once(ocr, rgb180, debug_once=False)
                    if p1>p0: model, raw, prob = m1, raw1, p1
                    else:     model, raw, prob = m0, raw0, p0
            else:
                ran, model, raw, prob = tesseract_ocr_once(crop)
        except Exception as e:
            model, raw, prob = "", f"ERR:{e}", 0.0
        ms=(time.time()-t0)*1000.0
        output_q.put({'tid': task['tid'], 'ran': True, 'model': model, 'raw': raw, 'prob': float(prob), 'ms': ms})

# ---------- 主程序 ----------

def main():
    args=parse_args()
    # 性能环境
    os.environ["OMP_NUM_THREADS"]=str(args.threads)
    os.environ["NCNN_THREADS"]=str(args.threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("NCNN_VERBOSE","0")

    ensure_dir(args.save_dir)
    if args.ocr_dump: ensure_dir(args.ocr_dump_dir)

    expected_model=norm_model(args.expected_model)

    # Camera
    picam2=Picamera2()
    main_w, main_h=1280, 960
    cfg=(picam2.create_preview_configuration if args.preview else picam2.create_video_configuration)(
        main={"size":(main_w,main_h),"format":"YUV420"}, controls={"FrameRate":30})
    picam2.configure(cfg); picam2.start(); time.sleep(1.0)

    # YOLO 模型
    print("Loading YOLO…")
    model=YOLO(args.weights)
    _=model.predict(source=np.zeros((args.imgsz,args.imgsz,3),np.uint8), imgsz=args.imgsz, verbose=False)
    print("✅ YOLO ready.")

    # 多进程 OCR
    input_q=mp.Queue(maxsize=32)
    output_q=mp.Queue(maxsize=64)
    ocr_backend = "PADDLE" if args.ocr else "OFF"
    worker=None
    if args.ocr:
        worker=mp.Process(target=ocr_worker, args=(input_q,output_q,ocr_backend,args.debug_ocr_once), daemon=True)
        worker.start()
        print(f"✅ OCR Worker started. backend={ocr_backend}")
    else:
        print("ℹ️ OCR disabled (no --ocr)")

    # 运行时状态
    tracks={}  # tid -> {model, conf, locked, raw, tried}
    sx, sy=main_w/float(args.imgsz), main_h/float(args.imgsz)
    fps_hist=[]; frame_idx=0; start=time.time()
    last_save=0.0
    total_ocr_ms=0.0; ocr_runs=0; ocr_calls_total=0

    try:
        while True:
            t0=time.time()
            yuv=picam2.capture_array("main")
            frame=yuv if args.assume_bgr else cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            infer=cv2.resize(frame,(args.imgsz,args.imgsz), interpolation=cv2.INTER_LINEAR)
            r=model.track(source=infer, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det=0; low=False; ids=set()
            do_ocr = args.ocr and (frame_idx % args.ocr_every == 0)
            budget = args.ocr_budget if args.ocr else 0

            if r.boxes is not None and len(r.boxes)>0 and r.boxes.id is not None:
                # 高置信度优先
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

                    # 调度OCR任务（在Worker处理，不阻塞）
                    if do_ocr and (not info["locked"]) and budget>0 and args.ocr:
                        pad=args.ocr_pad; w,h=X2-X1, Y2-Y1
                        px,py=int(w*pad),int(h*pad)
                        ox1,oy1=max(0,X1-px), max(0,Y1-py)
                        ox2,oy2=min(main_w,X2+px), min(main_h,Y2+py)
                        crop=frame[oy1:oy2, ox1:ox2]
                        # 中心收缩，仅读表面文字
                        shrink=max(0.0, min(0.4, args.center_shrink))
                        cw,ch=ox2-ox1, oy2-oy1
                        cx1=ox1+int(cw*shrink); cy1=oy1+int(ch*shrink)
                        cx2=ox2-int(cw*shrink); cy2=oy2-int(ch*shrink)
                        if cx2>cx1 and cy2>cy1:
                            crop=frame[cy1:cy2, cx1:cx2]
                        if crop is not None and crop.size>0:
                            try:
                                input_q.put_nowait({'tid': tid, 'crop': crop, 'ts': time.time()})
                                ocr_calls_total += 1
                                info["tried"]=True
                                budget-=1
                            except queue.Full:
                                pass

                    # 显示颜色规则
                    label_model = info["model"] if info["model"] else "?"
                    # 颜色：匹配=青色，不匹配=红色，未知=白
                    color=(255,255,255)
                    if expected_model:
                        if info["model"] == expected_model: color=(255,255,0)  # Cyan (B,G,R)=(255,255,0)
                        elif info["model"]: color=(0,0,255)

                    label=f"{label_model} | {conf:.2f}"
                    draw_box(frame, X1,Y1,X2,Y2, label, color)
                    if args.ocr_show_raw:
                        raw_short = info.get("raw","")
                        if len(raw_short)>42: raw_short=raw_short[:42]+"…"
                        draw_small_text(frame, X1+4, Y2+14, "RAW: "+(raw_short if raw_short else "[NO-OCR]"))

                    tracks[tid]=info

            # 读取OCR结果（非阻塞）
            if args.ocr:
                while True:
                    try:
                        res=output_q.get_nowait()
                    except queue.Empty:
                        break
                    tid=res['tid']
                    info=tracks.get(tid)
                    if info is None: continue
                    model=res.get('model') or ""
                    raw=res.get('raw') or ""
                    prob=float(res.get('prob') or 0.0)
                    info['raw']=raw if raw else info.get('raw','')
                    if model and (prob>info['conf'] or not info['model']):
                        info['model']=norm_model(model)
                        info['conf']=prob
                        if prob>=args.ocr_lock_conf:
                            info['locked']=True
                    tracks[tid]=info
                    total_ocr_ms+=float(res.get('ms',0.0)); ocr_runs+=1

            # 清理离场的track
            for tid in [t for t in list(tracks.keys()) if t not in ids]:
                tracks.pop(tid, None)

            # HUD
            frame_idx+=1
            dt=(time.time()-t0)*1000.0
            fps=(1000.0/max(dt,1.0))
            fps_hist.append(fps)
            if len(fps_hist)>30: fps_hist.pop(0)
            fps_avg=sum(fps_hist)/len(fps_hist)

            hud=f"Det:{det} | {dt:.1f}ms | {fps_avg:.1f}FPS"
            hud+=f" | OCR:{'ON' if args.ocr else 'OFF'}"
            if args.ocr:
                hud+=f" q:{output_q.qsize()}/{input_q.qsize()} calls:{ocr_calls_total}"
                if ocr_runs: hud+=f" avg:{(total_ocr_ms/max(ocr_runs,1)):.1f}ms"
            if expected_model:
                hud+=f" | EXPECT:{expected_model}"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            if not args.headless:
                cv2.imshow("Battery+AsyncOCR", frame)
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
        if args.ocr and worker is not None and worker.is_alive():
            try:
                input_q.put(None)
                worker.terminate()  # 保守起见强停
                worker.join(timeout=0.5)
            except Exception: pass
        el=time.time()-start
        print(f"\nFrames:{frame_idx} | Time:{el:.1f}s | FPS:{frame_idx/max(el,1):.1f}")
        if args.ocr and ocr_runs:
            print(f"OCR runs:{ocr_runs} | Avg OCR:{total_ocr_ms/max(ocr_runs,1):.1f} ms")
        print("Bye.")

if __name__=="__main__":
    # Linux/RPi 默认是 fork，足够稳定；保持默认即可。
    # 如需显式： mp.set_start_method('fork', force=True)
    main()
