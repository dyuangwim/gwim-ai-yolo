# detect_batch.py
import os, cv2, json, time, csv, argparse
from datetime import datetime
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
        cv2.putText(img, label, (x1+4, y0+th+3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

def auto_rotate_if_needed(bgr, rotate:int):
    """rotate in degrees: 0/90/180/270"""
    if rotate==0: return bgr
    if rotate==90:  return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    if rotate==180: return cv2.rotate(bgr, cv2.ROTATE_180)
    if rotate==270: return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return bgr

def capture_from_camera(width=1920, height=1080):
    from picamera2 import Picamera2
    picam2 = Picamera2()
    cfg = picam2.create_still_configuration(main={"size": (width, height), "format":"RGB888"})
    picam2.configure(cfg)
    picam2.start(); time.sleep(0.6)
    arr = picam2.capture_array("main")
    picam2.stop(); picam2.close()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def analyze_batch(bgr, card_det:CardDetector, bat_det:BatteryDetector, expected:int, margin:int=6):
    """返回汇总数据与可视化渲染后的图像。"""
    H,W = bgr.shape[:2]
    cards = card_det.detect(bgr)
    vis = bgr.copy()
    report = []
    idx = 0
    for c in cards:
        x1,y1,x2,y2 = c["xyxy"]
        # 轻微扩一点边，避免切掉圆壳
        x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
        x2 = min(W-1, x2 + margin); y2 = min(H-1, y2 + margin)
        roi = bgr[y1:y2, x1:x2]
        bats = bat_det.detect(roi)
        cnt = len(bats)
        ok = (cnt == expected)
        color = (0,255,0) if ok else (0,0,255)

        # 画卡纸框
        draw_box(vis, (x1,y1,x2,y2), f"pack#{idx} cnt={cnt}/{expected}", color, 3)

        # 画电池框（换算为全图坐标）
        for b in bats:
            bx1,by1,bx2,by2 = b["xyxy"]
            draw_box(vis, (x1+bx1, y1+by1, x1+bx2, y1+by2), None, color=(255,255,0), thick=2)

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

def main():
    ap = argparse.ArgumentParser("Batch Card→Battery Counting (Pi5 + NCNN)")
    ap.add_argument("--card_weights", default="/home/pi/models/card_ncnn", help="NCNN folder")
    ap.add_argument("--bat_weights",  default="/home/pi/models/battery_ncnn", help="NCNN folder")
    ap.add_argument("--img", help="输入图片路径（有则直接处理）")
    ap.add_argument("--expected", type=int, required=True, help="每包应有的电池数量，如 1/2/4/8")
    ap.add_argument("--rotate", type=int, default=0, choices=[0,90,180,270], help="按需旋转")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--card_imgsz", type=int, default=640)
    ap.add_argument("--bat_imgsz", type=int, default=416)
    ap.add_argument("--card_conf", type=float, default=0.35)
    ap.add_argument("--bat_conf", type=float, default=0.35)
    ap.add_argument("--out_dir", default="/home/pi/batch_out")
    ap.add_argument("--save_name", default="auto")  # auto=按时间戳
    ap.add_argument("--trigger_pin", type=int, default=None, help="GPIO 触发拍照（BCM 编号）")
    ap.add_argument("--buzzer_pin", type=int, default=None, help="蜂鸣器 GPIO（BCM 编号）")
    ap.add_argument("--fallback_wait", type=float, default=0.0, help="无传感器时等待秒数再拍")
    ap.add_argument("--from_camera", action="store_true", help="从相机拍一张再分析")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("ONNXRUNTIME_THREADING_FACTOR", "1")

    ensure_dir(args.out_dir)

    # 硬件
    trig = Trigger(pin=args.trigger_pin, active_high=True) if args.from_camera else None
    buz  = Buzzer(pin=args.buzzer_pin, active_high=True) if args.buzzer_pin is not None else None

    # 模型
    card_det = CardDetector(args.card_weights, imgsz=args.card_imgsz, conf=args.card_conf, threads=args.threads)
    bat_det  = BatteryDetector(args.bat_weights,  imgsz=args.bat_imgsz,  conf=args.bat_conf,  threads=args.threads)

    # 获取图像
    if args.img:
        bgr = cv2.imread(args.img)
        if bgr is None: 
            raise RuntimeError(f"无法读取图像：{args.img}")
    else:
        if trig is not None:
            print("⏳ 等待触发信号…")
            trig.wait(fallback_seconds=args.fallback_wait)
        else:
            if args.fallback_wait>0:
                time.sleep(args.fallback_wait)
        print("📸 拍照中…")
        bgr = capture_from_camera()

    if args.rotate:
        bgr = auto_rotate_if_needed(bgr, args.rotate)

    t0 = time.time()
    report, vis = analyze_batch(bgr, card_det, bat_det, expected=args.expected)
    dt = (time.time() - t0)*1000.0

    # 保存输出
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = (args.save_name if args.save_name!="auto" else f"batch_{ts}")
    img_out  = os.path.join(args.out_dir, f"{base}.jpg")
    json_out = os.path.join(args.out_dir, f"{base}.json")
    csv_out  = os.path.join(args.out_dir, f"{base}.csv")

    cv2.imwrite(img_out, vis)
    with open(json_out, "w") as f:
        json.dump({"latency_ms": dt, "packs": report}, f, indent=2)
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pack_index","x1","y1","x2","y2","battery_count","expected","ok","card_conf"])
        for r in report:
            x1,y1,x2,y2 = r["card_box"]
            w.writerow([r["pack_index"],x1,y1,x2,y2,r["battery_count"],r["expected"],int(r["ok"]),f'{r["card_conf"]:.3f}'])

    # 统计/报警
    bad = [p for p in report if not p["ok"]]
    print(f"\nDone. Cards: {len(report)} | NG: {len(bad)} | Time: {dt:.1f} ms")
    print(f"Image: {img_out}\nJSON:  {json_out}\nCSV:   {csv_out}")

    if bad and buz is not None:
        buz.beep(200); time.sleep(0.1); buz.beep(200)

if __name__ == "__main__":
    main()
