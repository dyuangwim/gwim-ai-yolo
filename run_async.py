# run_async.py
import os, time, argparse
import cv2, numpy as np
import multiprocessing as mp
from datetime import datetime

from yolo_detector import YoloDetector
from ocr_module import ocr_worker

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def draw_box(img, x1,y1,x2,y2, label, color=(255,255,255)):
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    y0=max(0,y1-th-6)
    cv2.rectangle(img,(x1,y0),(x1+tw+8,y0+th+8),color,-1)
    cv2.putText(img,label,(x1+4,y0+th+3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def parse_args():
    ap = argparse.ArgumentParser("YOLO main + Async OCR (split modules)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--headless", action="store_true")

    # OCR 调度参数
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ocr_every", type=int, default=6)       # 稍微勤一点
    ap.add_argument("--ocr_budget", type=int, default=4)      # 多给 1 个预算
    ap.add_argument("--ocr_lock_conf", type=float, default=0.60)
    ap.add_argument("--ocr_pad", type=float, default=0.22)
    ap.add_argument("--center_shrink", type=float, default=0.0)

    ap.add_argument("--expected_model", default="", help="如 CR2025，用于着色")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    det = YoloDetector(weights=args.weights, imgsz=args.imgsz, conf=args.conf,
                       preview=args.preview, threads=args.threads)

    input_q = mp.Queue(maxsize=64)
    output_q = mp.Queue(maxsize=64)
    worker = None
    if args.ocr:
        worker = mp.Process(target=ocr_worker, args=(input_q, output_q), daemon=True)
        worker.start(); print("✅ OCR worker started")

    tracks = {}  # tid -> {model, conf, locked, raw}
    fps_hist=[]; frame_idx=0; start=time.time()
    last_save=0.0; total_ocr_ms=0.0; ocr_runs=0; calls=0
    expected = args.expected_model.upper().replace(" ","")

    try:
        while True:
            t0 = time.time()
            frame = det.capture_bgr()
            dets = det.track_once(frame)

            det_count = 0; low=False; ids=set()
            do_ocr = args.ocr and (frame_idx % args.ocr_every == 0)
            budget = args.ocr_budget if args.ocr else 0

            for d in dets:
                (X1,Y1,X2,Y2) = d["xyxy"]
                if (X2-X1)*(Y2-Y1) < args.min_area: continue
                det_count += 1
                conf = d["conf"]; tid = d["track_id"]
                if conf < (args.conf + 0.10): low=True
                ids.add(tid)

                info = tracks.get(tid, {"model":"", "conf":0.0, "locked":False, "raw":""})

                # 调度 OCR：padding + （可选）中心收缩
                if do_ocr and (not info["locked"]) and budget>0 and args.ocr:
                    pad=args.ocr_pad; w,h=X2-X1, Y2-Y1
                    px,py=int(w*pad),int(h*pad)
                    ox1,oy1=max(0,X1-px), max(0,Y1-py)
                    ox2,oy2=min(det.main_w,X2+px), min(det.main_h,Y2+py)
                    crop=frame[oy1:oy2, ox1:ox2]

                    shrink=max(0.0, min(0.4, args.center_shrink))
                    if shrink>0.0:
                        cw,ch=ox2-ox1, oy2-oy1
                        cx1=ox1+int(cw*shrink); cy1=oy1+int(ch*shrink)
                        cx2=ox2-int(cw*shrink); cy2=oy2-int(ch*shrink)
                        if cx2>cx1 and cy2>cy1:
                            crop=frame[cy1:cy2, cx1:cx2]

                    if crop is not None and crop.size>0:
                        try:
                            input_q.put_nowait({"tid":tid, "crop":crop, "expected":expected})
                            calls+=1; budget-=1
                        except Exception:
                            pass

                # 颜色 & 文本（只有“已锁定且与 expected 不符”才红色）
                model = info["model"] if info["model"] else "?"
                if expected and info["locked"] and info["model"]:
                    if info["model"] == expected: color=(255,255,0)   # 青/黄：匹配
                    else:                          color=(0,0,255)     # 红：锁定但不匹配
                else:
                    color=(255,255,255)  # 未锁定或无 expected：白
                label=f"{model} | {conf:.2f}"
                draw_box(frame, X1,Y1,X2,Y2, label, color)
                tracks[tid]=info

            # 取回 OCR 结果
            if args.ocr:
                import queue
                while True:
                    try:
                        res = output_q.get_nowait()
                    except queue.Empty:
                        break
                    tid=res["tid"]; info=tracks.get(tid)
                    if info is None: continue
                    model=res.get("model","")
                    prob=float(res.get("prob",0.0))   # 现在是“纯 OCR 置信度”
                    raw=res.get("raw","")
                    info["raw"]=raw if raw else info.get("raw","")
                    if model and (prob>info["conf"] or not info["model"]):
                        info["model"]=model; info["conf"]=prob
                        # 只有命中 OCR 且置信度过阈值才锁定
                        if prob>=args.ocr_lock_conf:
                            info["locked"]=True
                    tracks[tid]=info
                    total_ocr_ms+=float(res.get("ms",0.0)); ocr_runs+=1

            # 清理离场
            for tid in [t for t in list(tracks.keys()) if t not in ids]:
                tracks.pop(tid, None)

            # HUD/FPS
            frame_idx+=1
            dt=(time.time()-t0)*1000.0
            fps=(1000.0/max(dt,1.0)); fps_hist.append(fps)
            if len(fps_hist)>30: fps_hist.pop(0)
            fps_avg=sum(fps_hist)/len(fps_hist)
            hud=f"Det:{det_count} | {dt:.1f}ms | {fps_avg:.1f}FPS | OCR:{'ON' if args.ocr else 'OFF'}"
            if args.ocr:
                hud+=f" calls:{calls}"
                if ocr_runs: hud+=f" avg:{(total_ocr_ms/max(ocr_runs,1)):.1f}ms"
            if expected:
                hud+=f" | EXPECT:{expected}"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            if not args.headless:
                cv2.imshow("Async YOLO + OCR", frame)
                if cv2.waitKey(1)&0xFF==27: break

            now=time.time()
            if (det_count==0 or low) and (now-last_save>1.0):
                fn=f"hard_{det_count}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame); last_save=now

    finally:
        det.close()
        if args.ocr and worker is not None and worker.is_alive():
            try:
                input_q.put(None); worker.join(timeout=1.0)
            except Exception:
                try: worker.terminate()
                except Exception: pass
        if not args.headless:
            try: cv2.destroyAllWindows()
            except Exception: pass
        el=time.time()-start
        print(f"\nFrames:{frame_idx} | Time:{el:.1f}s | FPS:{frame_idx/max(el,1):.1f}")
        if args.ocr and ocr_runs:
            print(f"OCR runs:{ocr_runs} | Avg OCR:{total_ocr_ms/max(ocr_runs,1):.1f} ms")
        print("Bye.")

if __name__ == "__main__":
    main()
