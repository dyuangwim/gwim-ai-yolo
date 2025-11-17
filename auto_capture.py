# auto_capture.py
import os, time, cv2, json, glob, subprocess, warnings, argparse, threading, signal, atexit
from datetime import datetime
from picamera2 import Picamera2
from gpiozero import DistanceSensor
from gpiozero.input_devices import DistanceSensorNoEcho
from utils_hw import Buzzer

warnings.filterwarnings("ignore", category=DistanceSensorNoEcho)

CAM_W, CAM_H = 1280, 960
SAVE_DIR  = "/home/pi/batch_images"
OUT_DIR   = "/home/pi/batch_out_pt"
CARD_MODEL = "/home/pi/models/card.pt"
BAT_MODEL  = "/home/pi/models/battery.pt"

TRIG_PIN = 23
ECHO_PIN = 24
MAX_DISTANCE_M = 1.0

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN, max_distance=MAX_DISTANCE_M)

_BUZZER = None
_ALARM_THREAD = None
_ALARM_STOP_EVT = None

def _buzzer_off_safely():
    global _BUZZER, _ALARM_STOP_EVT, _ALARM_THREAD
    try:
        if _ALARM_STOP_EVT is not None:
            _ALARM_STOP_EVT.set()
        if _ALARM_THREAD is not None and _ALARM_THREAD.is_alive():
            _ALARM_THREAD.join(timeout=0.3)
    except Exception: pass
    try:
        if _BUZZER is not None:
            _BUZZER.close()  # inside close(): off()
    except Exception: pass

def _handle_signal(signum, frame):
    _buzzer_off_safely()
    raise SystemExit(0)

atexit.register(_buzzer_off_safely)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

def parse_args():
    ap = argparse.ArgumentParser("auto capture + batch detect")
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--buzzer_pin", type=int, default=21)
    ap.add_argument("--trigger_distance", type=float, default=0.12)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--keep_raw", type=int, default=200)
    ap.add_argument("--keep_out", type=int, default=500)
    return ap.parse_args()

def get_distance_cm():
    return sensor.distance * MAX_DISTANCE_M * 100.0

def capture_image():
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": (CAM_W, CAM_H), "format": "YUV420"}, controls={"FrameRate": 30})
    picam2.configure(cfg); picam2.start(); time.sleep(0.5)
    yuv = picam2.capture_array("main")
    try: bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    except Exception: bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    picam2.stop(); picam2.close()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"auto_{ts}"
    path = os.path.join(SAVE_DIR, f"{base}.jpg")
    cv2.imwrite(path, bgr)
    print(f"📸 Captured: {path} ({CAM_W}x{CAM_H})")
    return path, base

def run_detection(img_path, base_name, expected):
    json_path = os.path.join(OUT_DIR, f"{base_name}.json")
    cmd = [
        "python3", "/home/pi/battery_batch/detect_batch.py",
        "--card_weights", CARD_MODEL, "--bat_weights", BAT_MODEL,
        "--img", img_path, "--expected", str(expected),
        "--rotate", "0", "--out_dir", OUT_DIR, "--card_conf", "0.50",
        "--save_name", base_name,
    ]
    print("🚀 Running detection...")
    subprocess.run(cmd)
    print("✅ Detection finished.")
    ng = 0
    try:
        with open(json_path, "r") as f: data = json.load(f)
        packs = data.get("packs", [])
        ng = len([p for p in packs if not p.get("ok", True)])
        print(f"📊 NG packs in this image: {ng}")
    except Exception as e:
        print("⚠️ Failed to read JSON:", e)
    return ng, json_path

def alarm_beep_loop(bz: Buzzer, stop_evt: threading.Event):
    try:
        if hasattr(bz, "off"): bz.off()
    except Exception: pass
    while not stop_evt.is_set():
        try: bz.beep(200)
        except Exception: time.sleep(0.2)
        time.sleep(0.1)
    try:
        if hasattr(bz, "off"): bz.off()
    except Exception: pass

def cleanup_dir(path, pattern, keep_last):
    if keep_last <= 0: return
    files = glob.glob(os.path.join(path, pattern))
    if len(files) <= keep_last: return
    for f in sorted(files, key=os.path.getmtime, reverse=True)[keep_last:]:
        try: os.remove(f)
        except Exception: pass

def cleanup_outputs(keep_raw, keep_out):
    cleanup_dir(SAVE_DIR, "auto_*.jpg", keep_raw)
    cleanup_dir(OUT_DIR, "*.jpg", keep_out)
    cleanup_dir(OUT_DIR, "*.json", keep_out)
    cleanup_dir(OUT_DIR, "*.csv", keep_out)

def main():
    global _BUZZER, _ALARM_THREAD, _ALARM_STOP_EVT
    args = parse_args()
    expected = args.expected
    if expected <= 0:
        while True:
            s = input("Enter expected batteries per pack (1/2/3/4/...): ").strip()
            if s.isdigit() and int(s) > 0:
                expected = int(s); break
            print("❗ Invalid input.")
    print(f"🔢 Current packaging settings: {expected} batteries per pack")

    if args.buzzer_pin and args.buzzer_pin > 0:
        _BUZZER = Buzzer(pin=args.buzzer_pin, active_high=True)
        print(f"🔔 Buzzer GPIO (BCM): {args.buzzer_pin}（NG 时持续报警，按 Enter 停止）")
    else:
        print("🔔 No buzzer configured")

    trig_cm = args.trigger_distance * 100.0
    cooldown = args.cooldown
    print("🟢 System ready. Waiting for object...")

    try:
        while True:
            d = get_distance_cm()
            print(f"Distance: {d:5.1f} cm", end="\r")
            if d < trig_cm:
                print(f"\n📏 Object detected ({d:.1f} cm) → Capturing image...")
                img, base = capture_image()
                ng, _ = run_detection(img, base, expected)
                cleanup_outputs(args.keep_raw, args.keep_out)

                if ng > 0 and _BUZZER is not None:
                    print(f"❌ {ng} NG packages detected, continuous alerts initiated!")
                    _ALARM_STOP_EVT = threading.Event()
                    _ALARM_THREAD = threading.Thread(target=alarm_beep_loop, args=(_BUZZER, _ALARM_STOP_EVT), daemon=True)
                    _ALARM_THREAD.start()
                    try:
                        input("🔕 Press Enter to stop the buzzer and continue...\n")
                    except EOFError:
                        pass
                    finally:
                        if _ALARM_STOP_EVT: _ALARM_STOP_EVT.set()
                        if _ALARM_THREAD: _ALARM_THREAD.join(timeout=0.5)
                        _buzzer_off_safely()
                        print("🔕 The alarm has been stopped.")

                print(f"⏸ Cooling {cooldown:.1f} s before next trigger...\n")
                time.sleep(cooldown)
            else:
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n🛑 Exited by user.")
    except SystemExit:
        print("\n🛑 Terminated.")
    except Exception as e:
        print("\n❌ Error:", e)
    finally:
        _buzzer_off_safely()

if __name__ == "__main__":
    main()
