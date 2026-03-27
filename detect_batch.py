# detect_batch.py
import os, cv2, json, time, csv, argparse
from datetime import datetime
import numpy as np

from card_detector import CardDetector
from battery_detector import BatteryDetector
from utils_hw import Trigger, Buzzer


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def draw_box(img, box, label=None, color=(0,255,255), thick=2):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)
    if label:
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1,y0), (x1+tw+8, y0+th+8), color, -1)
        cv2.putText(img, label, (x1+4, y0+th+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)


def auto_rotate_if_needed(bgr, rotate:int):
    if rotate==0:   return bgr
    if rotate==90:  return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if rotate==180: return cv2.rotate(bgr, cv2.ROTATE_180)
    if rotate==270: return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr


def capture_from_camera(width=1920, height=1080):
    from picamera2 import Picamera2
    picam2 = Picamera2()
    cfg = picam2.create_still_configuration(
        main={"size": (width, height), "format":"RGB888"}
    )
    picam2.configure(cfg)
    picam2.start(); time.sleep(0.6)
    arr = picam2.capture_array("main")
    picam2.stop(); picam2.close()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------- NMS + Card Filtering ----------

def nms_numpy_local(boxes, scores, iou_thr=0.5, topk=32):
    if len(boxes) == 0:
        return []
    boxes  = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)

    x1,y1,x2,y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < topk:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def filter_cards(cards, img_w, img_h, min_w=100, min_h=80):
    boxes=[]; scores=[]
    for c in cards:
        x1,y1,x2,y2 = c["xyxy"]
        w,h = x2-x1, y2-y1
        if w < min_w or h < min_h:
            continue
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            continue
        boxes.append([x1,y1,x2,y2])
        scores.append(c["conf"])
    if not boxes:
        return []
    keep_idx = nms_numpy_local(boxes, scores, iou_thr=0.5, topk=32)
    return [{"xyxy": boxes[i], "conf": scores[i]} for i in keep_idx]


# ---------- Only merge duplicate boxes; the "maximum expected number" limit is no longer applied. ----------

def dedup_batteries(bats, max_keep=None):
    """
    Merge duplicate battery boxes (boxes with very close center points).
    - No longer use expected, treat it as a "quantity limit"
    - max_keep is just a safety limit (e.g., 8 / 16) to prevent too many boxes in extreme cases
    """
    if not bats:
        return []

    # Arranged by confidence level from highest to lowest
    cand = sorted(bats, key=lambda x: x["conf"], reverse=True)

    selected = []
    centers = []

    for b in cand:
        if max_keep is not None and len(selected) >= max_keep:
            break

        bx1,by1,bx2,by2 = b["xyxy"]
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        w  = bx2 - bx1
        h  = by2 - by1
        radius = min(w, h) * 0.6   # The "radius" range of the same battery

        too_close = False
        for (scx, scy, sr) in centers:
            dx = cx - scx
            dy = cy - scy
            dist = (dx*dx + dy*dy) ** 0.5
            # If the distance between two centers is very small, consider it a duplicate box for the same battery
            if dist < min(radius, sr):
                too_close = True
                break

        if not too_close:
            selected.append(b)
            centers.append((cx, cy, radius))

    return selected


# ---------- Core: Full-image Battery Detection + Card-based Assignment ----------

def analyze_batch(bgr, card_det:CardDetector, bat_det:BatteryDetector,
                  expected:int, margin:int=6):
    H, W = bgr.shape[:2]

    # 1) Card Detection + Filtering
    cards_raw = card_det.detect(bgr)
    cards = filter_cards(cards_raw, W, H)

    # 2) Full-image Battery Detection (same calling method as Colab)
    bats_full = bat_det.detect_full(bgr)

    vis = bgr.copy()
    report = []
    idx = 0

    for c in cards:
        x1, y1, x2, y2 = c["xyxy"]
        x1e = max(0, x1 - margin)
        y1e = max(0, y1 - margin)
        x2e = min(W-1, x2 + margin)
        y2e = min(H-1, y2 + margin)

        # A. Pick out the batteries whose center point falls on this card.
        cand = []
        for b in bats_full:
            bx1, by1, bx2, by2 = b["xyxy"]
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            if (cx >= x1e) and (cx <= x2e) and (cy >= y1e) and (cy <= y2e):
                cand.append(b)

        # B. Simple size filtering: Discard too small false positives
        bb = []
        for b in cand:
            bx1, by1, bx2, by2 = b["xyxy"]
            bw, bh = bx2 - bx1, by2 - by1
            if bw >= 28 and bh >= 28:
                bb.append(b)

        # C. Merge duplicate boxes, but no longer limit the maximum number to expected
        #    Give a more relaxed upper limit, e.g., expected*4 (2 cards at most 8 candidates)
        bats = dedup_batteries(bb, max_keep=expected * 4 if expected > 0 else None)
        cnt = len(bats)

        ok = (cnt == expected)
        color = (0,255,0) if ok else (0,0,255)

        # draw card frame
        draw_box(vis, (x1, y1, x2, y2),
                 f"pack#{idx} cnt={cnt}/{expected}", color, 3)

        # draw battery boxes (using full-image coordinates)
        for b in bats:
            bx1, by1, bx2, by2 = b["xyxy"]
            draw_box(vis, (bx1, by1, bx2, by2),
                     None, color=(255,255,0), thick=2)

        report.append({
            "pack_index": idx,
            "card_box": [int(x1),int(y1),int(x2),int(y2)],
            "battery_count": cnt,
            "expected": expected,
            "ok": bool(ok),
            "card_conf": float(c["conf"])
        })
        idx += 1

    return report, vis


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("Batch Card→Battery Counting (Pi5 + YOLO)")
    ap.add_argument("--card_weights", default="/home/pi/models/card.pt")
    ap.add_argument("--bat_weights",  default="/home/pi/models/battery.pt")
    ap.add_argument("--img", help="Input the image path (process directly if available).")
    ap.add_argument("--expected", type=int, required=True,
                    help="The required number of batteries per pack, such as 1/2/4/8")
    ap.add_argument("--rotate", type=int, default=0, choices=[0,90,180,270])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--card_imgsz", type=int, default=640)
    ap.add_argument("--bat_imgsz", type=int, default=640)   # Reserved for compatibility
    ap.add_argument("--card_conf", type=float, default=0.50)
    ap.add_argument("--bat_conf", type=float, default=None) # None=Model Default 0.25
    ap.add_argument("--out_dir", default="/home/pi/batch_out")
    ap.add_argument("--save_name", default="auto")
    ap.add_argument("--trigger_pin", type=int, default=None)
    ap.add_argument("--buzzer_pin", type=int, default=None)
    ap.add_argument("--fallback_wait", type=float, default=0.0)
    ap.add_argument("--from_camera", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("ONNXRUNTIME_THREADING_FACTOR", "1")

    ensure_dir(args.out_dir)

    trig = Trigger(pin=args.trigger_pin, active_high=True) if args.from_camera else None
    buz  = Buzzer(pin=args.buzzer_pin, active_high=True) if args.buzzer_pin is not None else None

    card_det = CardDetector(args.card_weights, imgsz=args.card_imgsz,
                            conf=args.card_conf, threads=args.threads)
    bat_det  = BatteryDetector(args.bat_weights,
                               imgsz=args.bat_imgsz,
                               conf=args.bat_conf,
                               threads=args.threads)

    # Reading the picture
    if args.img:
        bgr = cv2.imread(args.img)
        if bgr is None:
            raise RuntimeError(f"Unable to read image: {args.img}")
    else:
        if trig is not None:
            print("⏳ Waiting for trigger signal…")
            trig.wait(fallback_seconds=args.fallback_wait)
        else:
            if args.fallback_wait > 0:
                time.sleep(args.fallback_wait)
            print("📸 Taking photo…")
        bgr = capture_from_camera()

    if args.rotate:
        bgr = auto_rotate_if_needed(bgr, args.rotate)

    t0 = time.time()
    report, vis = analyze_batch(bgr, card_det, bat_det, expected=args.expected)
    dt = (time.time() - t0) * 1000.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.save_name if args.save_name != "auto" else f"batch_{ts}"
    img_out  = os.path.join(args.out_dir, f"{base}.jpg")
    json_out = os.path.join(args.out_dir, f"{base}.json")
    csv_out  = os.path.join(args.out_dir, f"{base}.csv")

    cv2.imwrite(img_out, vis)
    with open(json_out, "w") as f:
        json.dump({"latency_ms": dt, "packs": report}, f, indent=2)
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pack_index","x1","y1","x2","y2",
                    "battery_count","expected","ok","card_conf"])
        for r in report:
            x1,y1,x2,y2 = r["card_box"]
            w.writerow([r["pack_index"],x1,y1,x2,y2,
                        r["battery_count"],r["expected"],
                        int(r["ok"]), f'{r["card_conf"]:.3f}'])

    bad = [p for p in report if not p["ok"]]
    print(f"\nDone. Cards: {len(report)} | NG: {len(bad)} | Time: {dt:.1f} ms")
    print(f"Image: {img_out}\nJSON:  {json_out}\nCSV:   {csv_out}")

    if bad and buz is not None:
        buz.beep(200); time.sleep(0.1); buz.beep(200)

if __name__ == "__main__":
    main()
