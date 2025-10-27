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
    ap.add_argument("--min_area", type=int, default=1600)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--headless", action="store_true")

    # OCR 调度参数
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--ocr_every", type=int, default=2)         # 更频繁：降低堆积
    ap.add_argument("--ocr_budget", type=int, default=-1,       # -1 = 自适应：min(det_count, 8)
                    help="-1 for auto; otherwise max OCR ROIs per frame")
    ap.add_argument("--ocr_lock_conf", type=float, default=0.50)
    ap.add_argument("--ocr_pad", type=float, default=0.12)
    ap.add_argument("--center_shrink", type=float, default=0.18)

    ap.add_argument("--expected_model", default="", help="如 CR2025，用于着色")
    ap.add_argument("--save_dir", default="/home/pi/hard_cases")
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    # 限制并行库线程，避免互拖
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    det = YoloDetector(weights=args.weights, imgsz=args.imgsz, conf=args.conf,
                       preview=args.preview, threads=args.threads)

    input_q = mp.Queue(maxsize=64)
    output_q = mp.Queue(maxsize=64)
    worker = None
    if args.ocr:
        worker = mp.Process(target=ocr_worker, args=(input_q, output_q), daemon=True)
        worker.start(); print("✅ OCR worker started")

    tracks = {}  # tid -> {model, conf, locked, raw, silent}
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

            # 先统计本帧有效对象数
            valid_dets=[]
            for d in dets:
                (X1,Y1,X2,Y2) = d["xyxy"]
                if (X2-X1)*(Y2-Y1) < args.min_area: 
                    continue
                valid_dets.append(d)
            det_count = len(valid_dets)

            # 自适应预算 + 背压
            budget = 0
            if args.ocr:
                budget = min(det_count, 8) if args.ocr_budget < 0 else min(det_count, args.ocr_budget)
                if input_q.qsize() > 48:    # 背压：队列大就暂时不送，防堆积
                    budget = 0

            for d in valid_dets:
                (X1,Y1,X2,Y2) = d["xyxy"]
                conf = d["conf"]; tid = d["track_id"]
                if conf < (args.conf + 0.10): low=True
                ids.add(tid)

                info = tracks.get(tid, {"model":"", "conf":0.0, "locked":False, "raw":"", "silent":0})

                # 调度 OCR：padding + （可选）中心收缩 + 静默节流
                if do_ocr and (not info["locked"]) and budget>0 and args.ocr:
                    if info["silent"] > 0:
                        info["silent"] -= 1
                    else:
                        pad=args.ocr_pad; w,h=X2-X1, Y2-Y1
                        px,py=int(w*pad),int(h*pad)
                        ox1,oy1=max(0,X1-px), max(0,Y1-py)
                        ox2,oy2=min(det.main_w,X2+px), min(det.main_h,Y2+py)
                        crop=frame[oy1:oy2, ox1:ox2]

                        shrink=max(0.0, min(0.4, args.center_shrink))
                        if shrink>0.0:
                            cw,ch=
