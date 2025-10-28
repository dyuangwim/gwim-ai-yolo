# run_async.py
import os, time, argparse, uuid, signal, sys
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
    ap.add_argument("--min_area", type=int, default=1600)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--headless", action="store_true")

    # OCR 调度参数
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ocr_every", type=int, default=2)
    ap.add_argument("--ocr_budget", type=int, default=-1)      # -1 自适应
    ap.add_argument("--ocr_lock_conf", type=float, default=0.50)
    ap.add_argument("--ocr_pad", type=float, default=0.10)
    ap.add_argument("--center_shrink", type=float, default=0.20)

    # 诊断与存盘
    ap.add_argument("--roi_dir", default="")
    ap.add_argument("--pre_dir", default="")
    ap.add_argument("--dump_rate", type=int, default=0)
    ap.add_argument("--max_dump_per_frame", type=int, default=8)

    # 速度/稳定
    ap.add_argument("--no_tess", action="store_true")
    ap.add_argument("--max_vars", type=int, default=4)

    # 质量门阈值
    ap.add_argument("--blur_thr", type=float, default=45.0, help="拉普拉斯方差阈值，低于此视为模糊")
    ap.add_argument("--glare_thr", type=float, default=0.24, help="高光像素比例阈值")

    ap.add_argument("--expected_model", default="")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()

def roi_quality_ok(bgr, blur_thr=45.0, glare_thr=0.24):
    """快速判断 ROI 是否值得送 OCR：足够清晰、反光不过多。"""
    if bgr is None or bgr.size==0: return False
    gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 模糊度：拉普拉斯方差
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < blur_thr: 
        return False
    # 高光占比：亮度>230 的比例
    glare_ratio = float((gray>230).sum()) / float(gray.size)
    if glare_ratio > glare_thr:
        return False
    return True

def tiny_fingerprint(bgr):
    """16x16 灰度均值二值化指纹，返回 bytes；用于判断 ROI 是否变化明显。"""
    if bgr is None or bgr.size==0: return b""
    g=cv2.cvtColor(cv2.resize(bgr,(16,16),interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    mean=int(g.mean()); bits=(g>mean).astype(np.uint8)
    # pack 256 bits -> 32 bytes
    return np.packbits(bits.reshape(-1)).tobytes()

def main():
    args = parse_args()
    ensure_dir(args.save_dir)
    if args.roi_dir: ensure_dir(args.roi_dir)
    if args.pre_dir: ensure_dir(args.pre_dir)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    stop_flag={"v":False}
    def _sigint(_a,_b):
        stop_flag["v"]=True
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    det = YoloDetector(weights=args.weights, imgsz=args.imgsz, conf=args.conf,
                       preview=args.preview, threads=args.threads)

    input_q = mp.Queue(maxsize=64)
    output_q = mp.Queue(maxsize=64)
    worker = None
    if args.ocr:
        worker = mp.Process(target=ocr_worker, args=(input_q, output_q), daemon=True)
        worker.start(); print("✅ OCR worker started")

    # tid -> info
    # info: model, conf, locked, raw, silent, seen, last_dump, fp (fingerprint), fresh_frames
    tracks = {}
    fps_hist=[]; frame_idx=0; start=time.time()
    last_save=0.0; total_ocr_ms=0.0; ocr_runs=0; calls=0
    expected = args.expected_model.upper().replace(" ","")

    try:
        while not stop_flag["v"]:
            t0 = time.time()
            frame = det.capture_bgr()
            dets = det.track_once(frame)

            valid=[]
            for d in dets:
                (X1,Y1,X2,Y2) = d["xyxy"]
                if (X2-X1)*(Y2-Y1) >= args.min_area:
                    valid.append(d)

            # 自适应预算 + 背压
            budget = 0
            if args.ocr:
                det_count=len(valid)
                budget = min(det_count, 8) if args.ocr_budget < 0 else min(det_count, args.ocr_budget)
                if input_q.qsize() > 48: budget = 0

            # 新目标优先：未锁定 + 没模型的排前
            valid.sort(key=lambda d: (
                tracks.get(d["track_id"], {}).get("locked", False),
                tracks.get(d["track_id"], {}).get("model","")!=""
            ))

            ids=set(); low=False; dumps_this_frame=0
            for d in valid:
                (X1,Y1,X2,Y2) = d["xyxy"]
                conf = d["conf"]; tid = d["track_id"]
                if conf < (args.conf + 0.10): low=True
                ids.add(tid)

                info = tracks.get(tid, {"model":"", "conf":0.0, "locked":False, "raw":"",
                                        "silent":0, "seen":0, "last_dump":-9999,
                                        "fp":b"", "fresh_frames":0})
                info["seen"] += 1
                if info["fresh_frames"] < 999999:  # 防止溢出
                    info["fresh_frames"] += 1

                do_ocr = args.ocr and (frame_idx % args.ocr_every == 0) and (not info["locked"]) and (budget>0)

                if do_ocr:
                    if info["silent"]>0:
                        info["silent"]-=1
                    else:
                        pad=args.ocr_pad; w,h=X2-X1, Y2-Y1
                        px,py=int(w*pad),int(h*pad)
                        ox1,oy1=max(0,X1-px), max(0,Y1-py)
                        ox2,oy2=min(det.main_w,X2+px), min(det.main_h,Y2+py)
                        crop=frame[oy1:oy2, ox1:ox2]

                        # 动态中心收缩：若高光过量则增强收缩
                        shrink=max(0.0, min(0.4, args.center_shrink))
                        if crop is not None and crop.size>0:
                            gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            glare_ratio=float((gray>230).sum())/float(gray.size)
                            if glare_ratio > args.glare_thr*1.2:
                                shrink=min(0.35, shrink+0.08)
                        if shrink>0.0 and crop is not None and crop.size>0:
                            cw,ch=ox2-ox1, oy2-oy1
                            cx1=ox1+int(cw*shrink); cy1=oy1+int(ch*shrink)
                            cx2=ox2-int(cw*shrink); cy2=oy2-int(ch*shrink)
                            if cx2>cx1 and cy2>cy1:
                                crop=frame[cy1:cy2, cx1:cx2]

                        # 质量门：模糊/高光过多不送 OCR
                        if crop is not None and crop.size>0 and roi_quality_ok(crop, args.blur_thr, args.glare_thr):

                            # 指纹去重：画面几乎没变就不送
                            fp=tiny_fingerprint(crop)
                            if fp == info.get("fp", b""):
                                info["silent"]=6  # 稍后再说
                            else:
                                info["fp"]=fp
                                # ROI 保存（限速）
                                if args.roi_dir and args.dump_rate>0 and dumps_this_frame<args.max_dump_per_frame:
                                    if info["seen"] - info["last_dump"] >= args.dump_rate:
                                        dump_id = f"tid{tid}_f{frame_idx}_{int(time.time()*1000)}"
                                        cv2.imwrite(os.path.join(args.roi_dir, f"roi_{dump_id}.jpg"), crop)
                                        info["last_dump"] = info["seen"]; dumps_this_frame += 1
                                    else:
                                        dump_id = f"tid{tid}_f{frame_idx}_{uuid.uuid4().hex[:6]}"
                                else:
                                    dump_id = f"tid{tid}_f{frame_idx}_{uuid.uuid4().hex[:6]}"

                                try:
                                    input_q.put_nowait({
                                        "tid":tid, "crop":crop, "expected":expected,
                                        "pre_dir": args.pre_dir,
                                        "max_vars": args.max_vars,
                                        "use_tess": (not args.no_tess),
                                        "dump_token": dump_id
                                    })
                                    calls+=1; budget-=1
                                    info["silent"]=6
                                except Exception:
                                    pass
                        else:
                            # 太模糊/太亮 → 再等几帧
                            info["silent"]=3

                # 上色与显示
                model = info["model"] if info["model"] else "?"
                color=(255,255,255)
                if expected and info["locked"] and info["model"]:
                    color=(255,255,0) if info["model"]==expected else (0,0,255)
                label=f"{model} | {info.get('conf',0.0):.2f}"
                draw_box(frame, X1,Y1,X2,Y2, label, color)
                tracks[tid]=info

            # 收取 OCR 结果
            if args.ocr:
                import queue
                while True:
                    try:
                        res=output_q.get_nowait()
                    except queue.Empty:
                        break
                    tid=res["tid"]; info=tracks.get(tid)
                    if info is None: continue
                    model=res.get("model",""); prob=float(res.get("prob",0.0))
                    raw=res.get("raw","")
                    if raw: info["raw"]=raw
                    changed=False
                    if model and (prob>info["conf"] or not info["model"]):
                        info["model"]=model; info["conf"]=prob; changed=True
                    if changed and prob>=args.ocr_lock_conf:
                        info["locked"]=True; info["silent"]=12
                    tracks[tid]=info
                    total_ocr_ms+=float(res.get("ms",0.0)); ocr_runs+=1

            # 清理离场
            for tid in [t for t in list(tracks.keys()) if t not in ids]:
                tracks.pop(tid, None)

            # HUD
            frame_idx+=1
            dt=(time.time()-t0)*1000.0
            fps=(1000.0/max(dt,1.0)); fps_hist.append(fps)
            if len(fps_hist)>30: fps_hist.pop(0)
            fps_avg=sum(fps_hist)/len(fps_hist)
            hud=f"Det:{len(valid)} | {dt:.1f}ms | {fps_avg:.1f}FPS | OCR:{'ON' if args.ocr else 'OFF'}"
            if args.ocr:
                hud+=f" calls:{calls}"
                if ocr_runs: hud+=f" avg:{(total_ocr_ms/max(ocr_runs,1)):.1f}ms"
            if expected: hud+=f" | EXPECT:{expected}"
            cv2.putText(frame, hud, (10,28), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

            if not args.headless:
                cv2.imshow("Async YOLO + OCR", frame)
                if cv2.waitKey(1)&0xFF==27: 
                    stop_flag["v"]=True

            now=time.time()
            if (len(valid)==0 or low) and (now-last_save>1.0):
                fn=f"hard_{len(valid)}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(os.path.join(args.save_dir, fn), frame); last_save=now

    finally:
        # 优雅收尾
        try:
            if args.ocr and worker is not None and worker.is_alive():
                input_q.put(None); worker.join(timeout=1.0)
        except Exception:
            try: worker.terminate()
            except Exception: pass
        try:
            if not args.headless: cv2.destroyAllWindows()
        except Exception: pass
        try:
            det.close()
        except Exception: pass

        el=time.time()-start
        print(f"\nFrames:{frame_idx} | Time:{el:.1f}s | FPS:{frame_idx/max(el,1):.1f}")
        if args.ocr and ocr_runs:
            print(f"OCR runs:{ocr_runs} | Avg OCR:{total_ocr_ms/max(ocr_runs,1):.1f} ms")
        print("Bye.")

if __name__ == "__main__":
    main()
