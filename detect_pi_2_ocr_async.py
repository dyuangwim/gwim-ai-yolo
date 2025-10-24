# detect_pi_2_ocr_async.py — RPi5 YOLO (NCNN) + Async OCR
# 稳态 + 强化版（针对你现场“长时间不出文本”的问题做了硬改）：
# 1) 强化 Tesseract 预处理：放大→CLAHE→锐化→多种二值化（OTSU/自适应/反相）→0°+180° 双角度，
#    逐个跑并打分（命中 CRxxxx > 纯数字 > 文本长度），选得分最高的方案。
# 2) 可选保存每个送 OCR 的 crop（--ocr_dump）以便排错；可在 HUD 下方显示 RAW 文本（--ocr_show_raw）。
# 3) 调整默认调度：ocr_every=4、ocr_budget=4（方便快速锁定；上线可改回 8/2）。
# 4) 彻底修复关停顺序（避免 allocator 异常），保留限额+锁定+中心收缩+颜色规则。
#
# 运行示例（先用 Tesseract 验证流程）：
#   python3 detect_pi_2_ocr_async_strong.py \
#     --weights /home/pi/models/best_ncnn_model --imgsz 320 --conf 0.30 \
#     --expected_model CR2025 --ocr --ocr_backend TESSERACT \
#     --ocr_every 4 --ocr_budget 4 --ocr_lock_conf 0.55 \
#     --center_shrink 0.12 --ocr_pad 0.18 --ocr_show_raw --ocr_dump --preview

import os, time, argparse, re, signal
from datetime import datetime
import cv2, numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import multiprocessing as mp
import queue

# ---------- 型号解析 ----------
RE_CR     = re.compile(r"CR\s*(1616|1620|2016|2025|2032|1650|1632)", re.I)
RE_DIGITS = re.compile(r"(1616|1620|2016|2025|2032|1650|1632)")

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
    ap.add_argument("--ocr_backend", choices=["AUTO","PADDLE","TESSERACT"], default="AUTO")
    ap.add_argument("--ocr_every", type=int, default=4)      # 强化：默认更频繁
    ap.add_argument("--ocr_budget", type=int, default=4)     # 强化：默认更大预算
    ap.add_argument("--ocr_lock_conf", type=float, default=0.55)
    ap.add_argument("--ocr_pad", type=float, default=0.18)   # 强化：更大 padding
    ap.add_argument("--center_shrink", type=float, default=0.12, help="中心收缩比例(0~0.4)")
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
    maxw=52
    t=text
    while t:
        ln=t[:maxw]
        cv2.putText(img, ln, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1)
        y+=14
        t=t[maxw:]

# ---------- OCR Worker 实现 ----------

def _unsharp_mask(gray, amount=1.0, radius=3):
    blur = cv2.GaussianBlur(gray, (radius|1, radius|1), 0)
    return cv2.addWeighted(gray, 1+amount, blur, -amount, 0)


def _prep_basic(bgr, target_w=480, gamma=1.10):
    if bgr is None or bgr.size==0: return None
    h,w=bgr.shape[:2]
    if w<target_w:
        s=target_w/float(w)
        bgr=cv2.resize(bgr,(int(w*s),int(h*s)),interpolation=cv2.INTER_CUBIC)
    if gamma!=1.0:
        table=np.array([((i/255.0)**(1.0/gamma))*255 for i in np.arange(256)]).astype("uint8")
        bgr=cv2.LUT(bgr, table)
    gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray=clahe.apply(gray)
    gray=_unsharp_mask(gray, amount=1.0, radius=3)
    return gray


def _binarize_family(gray):
    outs=[]
    _,t1=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);   outs.append(t1)
    _,t2=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU); outs.append(t2)
    t3=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,8); outs.append(t3)
    t4=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,8); outs.append(t4)
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    outs+=[cv2.morphologyEx(x, cv2.MORPH_OPEN, k) for x in outs]
    return outs


def _run_tess(img_bin):
    import pytesseract
    cfg='--psm 6 -c tessedit_char_whitelist=C0123456789R.'
    d=pytesseract.image_to_data(img_bin, output_type=pytesseract.Output.DICT, config=cfg)
    texts=[w for w in d.get('text',[]) if str(w).strip()]
    confs=[float(c) for c in d.get('conf',[]) if str(c).lstrip('-').isdigit()]
    raw=' '.join(texts).upper() if texts else ''
    avg=float(sum(confs)/len(confs)) if confs else 0.0
    return raw, avg


def tesseract_ocr_once_strong(bgr_img):
    gray=_prep_basic(bgr_img)
    if gray is None: return True, '', '', 0.0
    bins=_binarize_family(gray)
    bins+= [cv2.rotate(x, cv2.ROTATE_180) for x in list(bins)]
    best=(None, -1e9, 0.0)
    for b in bins:
        raw, avg=_run_tess(b)
        score=0.0
        if RE_CR.search(raw): score+=100.0
        elif RE_DIGITS.search(raw): score+=60.0
        score+=min(len(raw),40)*0.5
        score+=avg*0.1
        if score>best[1]: best=(raw, score, avg)
    raw=best[0] or ''
    m=RE_CR.search(raw) or RE_DIGITS.search(raw)
    model_out=norm_model(m.group(0) if m else raw)
    return True, model_out, raw, best[2]


def _call_paddle_ocr(ocr, rgb_img):
    return ocr.ocr(rgb_img, det=False, rec=True, cls=False)


def _parse_ocr_result(res):
    texts, probs=[], []
    if res is None: return texts, probs
    if isinstance(res, list) and len(res)==1 and isinstance(res[0], list): res=res[0]
    if isinstance(res, list) and len(res)>0 and isinstance(res[0], dict):
        for item in res:
            txt=item.get('text') or item.get('transcription') or item.get('label') or ''
            sc =item.get('score') or item.get('confidence') or item.get('prob') or 0.0
            if txt: texts.append(str(txt).strip()); probs.append(float(sc) if sc else 0.0)
        return texts, probs
    if isinstance(res, list):
        for it in res:
            if isinstance(it,(list,tuple)) and len(it)>=2:
                cand=it[1]
                if isinstance(cand,(list,tuple)) and len(cand)>=2 and isinstance(cand[0],str):
                    texts.append(cand[0].strip()); probs.append(float(cand[1]) if len(cand)>1 else 0.0)
    return texts, probs


def paddle_ocr_once(ocr, bgr_img):
    gray=_prep_basic(bgr_img, target_w=480)
    rgb0=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb180=cv2.rotate(rgb0, cv2.ROTATE_180)
    res0=_call_paddle_ocr(ocr, rgb0);  t0,p0=_parse_ocr_result(res0); raw0=' '.join(t0).upper(); sc0=max(p0) if p0 else 0.0
    res1=_call_paddle_ocr(ocr, rgb180);t1,p1=_parse_ocr_result(res1);raw1=' '.join(t1).upper(); sc1=max(p1) if p1 else 0.0
    if sc1>sc0: raw=raw1; conf=sc1
    else:       raw=raw0; conf=sc0
    m=RE_CR.search(raw) or RE_DIGITS.search(raw)
    return True, norm_model(m.group(0) if m else raw), raw, conf


def ocr_worker(input_q: mp.Queue, output_q: mp.Queue, backend: str):
    ocr=None; backend_used='TESSERACT'
    if backend.upper() in ('AUTO','PADDLE'):
        try:
            from paddleocr import PaddleOCR
            ocr=PaddleOCR(lang='en', use_angle_cls=False)
            backend_used='PADDLE'; print('[OCR-Worker] PaddleOCR init OK')
        except Exception as e:
            print('[OCR-Worker] Paddle init failed, fallback to Tesseract:', e)
            backend_used='TESSERACT'

    while True:
        try:
            task=input_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if task is None: break
        crop=task['crop']
        t0=time.time()
        try:
            if backend_used=='PADDLE' and ocr is not None:
                _, model_out, raw, prob = paddle_ocr_once(ocr, crop)
            else:
                _, model_out, raw, prob = tesseract_ocr_once_strong(crop)
        except Exception as e:
            model_out, raw, prob = '', f'ERR:{e}', 0.0
        ms=(time.time()-t0)*1000.0
        output_q.put({'tid': task['tid'], 'model': model_out, 'raw': raw, 'prob': float(prob), 'ms': ms})

# ---------- 主程序 ----------

def main():
    args=parse_args()

    os.environ['OMP_NUM_THREADS']=str(args.threads)
    os.environ['NCNN_THREADS']=str(args.threads)
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('NCNN_VERBOSE','0')

    ensure_dir(args.save_dir)
    if args.ocr_dump: ensure_dir(args.ocr_dump_dir)

    expected_model=norm_model(args.expected_model)

    picam2=Picamera2()
    main_w, main_h=1280, 960
    cfg=(picam2.create_preview_configuration if args.preview else picam2.create_video_configuration)(
        main={'size':(main_w,main_h),'format':'YUV420'}, controls={'FrameRate':30})
    picam2.configure(cfg); picam2.start(); time.sleep(1.0)

    print('Loading YOLO…')
    yolo=YOLO(args.weights)
    _=yolo.predict(source=np.zeros((args.imgsz,args.imgsz,3),np.uint8), imgsz=args.imgsz, verbose=False)
    print('✅ YOLO ready.')

    input_q=mp.Queue(maxsize=64)
    output_q=mp.Queue(maxsize=64)
    backend = ('PADDLE' if args.ocr_backend=='PADDLE' else ('TESSERACT' if args.ocr_backend=='TESSERACT' else 'AUTO'))
    worker=None
    if args.ocr:
        worker=mp.Process(target=ocr_worker, args=(input_q,output_q,backend), daemon=True)
        worker.start(); print(f'✅ OCR Worker started. backend={backend}')
    else:
        print('ℹ️ OCR disabled (no --ocr)')

    tracks={}
    sx, sy=main_w/float(args.imgsz), main_h/float(args.imgsz)
    fps_hist=[]; frame_idx=0; start=time.time()
    last_save=0.0
    total_ocr_ms=0.0; ocr_runs=0; ocr_calls_total=0

    stop_flag=[False]
    def handle_sigint(sig,frame): stop_flag[0]=True
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not stop_flag[0]:
            t0=time.time()
            try:
                yuv=picam2.capture_array('main')
            except Exception:
                continue
            frame=yuv if args.assume_bgr else cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

            infer=cv2.resize(frame,(args.imgsz,args.imgsz), interpolation=cv2.INTER_LINEAR)
            r=yolo.track(source=infer, imgsz=args.imgsz, conf=args.conf, verbose=False, persist=True)[0]

            det=0; low=False; ids=set()
            do_ocr = args.ocr and (frame_idx % args.ocr_every == 0)
            budget = args.ocr_budget if args.ocr else 0

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
                    info=tracks.get(tid, {'model':'', 'conf':0.0, 'locked':False, 'raw':'', 'tried':False})

                    if do_ocr and (not info['locked']) and budget>0 and args.ocr:
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
                        if crop is not None and crop.size>0:
                            if args.ocr_dump:
                                fn=f"{tid}_{int(time.time()*1000)}.jpg"; cv2.imwrite(os.path.join(args.ocr_dump_dir, fn), crop)
                            try:
                                input_q.put_nowait({'tid': tid, 'crop': crop, 'ts': time.time()})
                                ocr_calls_total += 1
                                info['tried']=True
                                budget-=1
                            except queue.Full:
                                pass

                    label_model = info['model'] if info['model'] else '?'
                    color=(255,255,255)
                    if expected_model:
                        if info['model'] == expected_model: color=(255,255,0)
                        elif info['model']: color=(0,0,255)
                    label=f"{label_model} | {conf:.2f}"
                    draw_box(frame, X1,Y1,X2,Y2, label, color)
                    if args.ocr_show_raw:
                        raw_short = info.get('raw','')
                        if len(raw_short)>80: raw_short=raw_short[:80]+'…'
                        draw_small_text(frame, X1+4, Y2+16, 'RAW: '+(raw_short if raw_short else '[NO-OCR]'))

                    tracks[tid]=info

            if args.ocr:
                while True:
                    try:
                        res=output_q.get_nowait()
                    except queue.Empty:
                        break
                    tid=res['tid']
                    info=tracks.get(tid)
                    if info is None: continue
                    ocr_text=res.get('model') or ''
                    raw=res.get('raw') or ''
                    prob=float(res.get('prob') or 0.0)
                    info['raw']=raw if raw else info.get('raw','')
                    if ocr_text and (prob>info['conf'] or not info['model']):
                        info['model']=norm_model(ocr_text)
                        info['conf']=prob
                        if prob>=args.ocr_lock_conf:
                            info['locked']=True
                    tracks[tid]=info
                    total_ocr_ms+=float(res.get('ms',0.0)); ocr_runs+=1

            for tid in [t for t in list(tracks.keys()) if t not in ids]:
                tracks.pop(tid, None)

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
                cv2.imshow('Battery+AsyncOCR', frame)
                if cv2.waitKey(1)&0xFF==27: break

            now=time.time()
            if (det==0 or low) and (now-last_save>1.0):
                fn=f"hard_{det}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame); last_save=now

    finally:
        if args.ocr and worker is not None and worker.is_alive():
            try:
                input_q.put(None)
                worker.join(timeout=1.0)
            except Exception:
                try: worker.terminate()
                except Exception: pass
        try:
            picam2.stop(); picam2.close()
        except Exception:
            pass
        if not args.headless:
            try: cv2.destroyAllWindows()
            except Exception: pass
        el=time.time()-start
        print(f"\nFrames:{frame_idx} | Time:{el:.1f}s | FPS:{frame_idx/max(el,1):.1f}")
        if args.ocr and ocr_runs:
            print(f"OCR runs:{ocr_runs} | Avg OCR:{total_ocr_ms/max(ocr_runs,1):.1f} ms")
        print('Bye.')

if __name__=='__main__':
    main()
