# auto_capture.py
import os
import time
import cv2
import json
import glob
import subprocess
import warnings
import argparse
import threading
import signal
import atexit
from datetime import datetime

from picamera2 import Picamera2
from gpiozero import DistanceSensor
from gpiozero.input_devices import DistanceSensorNoEcho

from utils_hw import Buzzer  # gpiozero版封装

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

# ---- 全局：蜂鸣与线程 ----
_BUZZER: Buzzer | None = None
_ALARM_THREAD: threading.Thread | None = None
_ALARM_STOP_EVT: threading.Event | None = None


def _alarm_quietly():
    """仅停止报警：停线程 + 拉低；不释放GPIO。"""
    global _ALARM_THREAD, _ALARM_STOP_EVT, _BUZZER
    try:
        if _ALARM_STOP_EVT is not None:
            _ALARM_STOP_EVT.set()
        if _ALARM_THREAD is not None and _ALARM_THREAD.is_alive():
            _ALARM_THREAD.join(timeout=0.5)
    except Exception:
        pass
    try:
        if _BUZZER is not None and hasattr(_BUZZER, "off"):
            _BUZZER.off()
    except Exception:
        pass

def _buzzer_close_all():
    """退出进程时调用：确保停报警并释放GPIO。"""
    _alarm_quietly()
    global _BUZZER
    try:
        if _BUZZER is not None:
            _BUZZER.close()
    except Exception:
        pass

def _handle_signal(signum, frame):
    _buzzer_close_all()
    raise SystemExit(0)

atexit.register(_buzzer_close_all)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def parse_args():
    ap = argparse.ArgumentParser("Ultrasonic-triggered auto capture + detect")
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--buzzer_pin", type=int, default=21)
    ap.add_argument("--trigger_distance", type=float, default=0.12)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--keep_raw", type=int, default=200)
    ap.add_argument("--keep_out", type=int, default=500)
    return ap.parse_args()


def get_distance_cm():
    d_m = sensor.distance * MAX_DISTANCE_M
    return d_m * 100.0


def capture_image():
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "YUV420"},
        controls={"FrameRate": 30},
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.5)

    yuv = picam2.capture_array("main")
    try:
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    except Exception:
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    picam2.stop()
    picam2.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"auto_{ts}"
    save_path = os.path.join(SAVE_DIR, f"{base_name}.jpg")
    cv2.imwrite(save_path, bgr)
    print(f"📸 Captured: {save_path} ({CAM_W}x{CAM_H})")
    return save_path, base_name


def run_detection(img_path, base_name, expected):
    json_path = os.path.join(OUT_DIR, f"{base_name}.json")
    cmd = [
        "python3", "/home/pi/battery_batch/detect_batch.py",
        "--card_weights", CARD_MODEL,
        "--bat_weights",  BAT_MODEL,
        "--img", img_path,
        "--expected", str(expected),
        "--rotate", "0",
        "--out_dir", OUT_DIR,
        "--card_conf", "0.50",
        "--save_name", base_name,
    ]
    print("🚀 Running detection...")
    subprocess.run(cmd)
    print("✅ Detection finished.")

    ng_count = 0
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        packs = data.get("packs", [])
        ng_count = len([p for p in packs if not p.get("ok", True)])
        print(f"📊 NG packs in this image: {ng_count}")
    except Exception as e:
        print("⚠️ Failed to read JSON:", e)

    return ng_count, json_path


def alarm_beep_loop(bz: Buzzer, stop_event: threading.Event):
    try:
        if hasattr(bz, "off"):
            bz.off()
    except Exception:
        pass

    while not stop_event.is_set():
        try:
            bz.beep(200)
        except Exception:
            time.sleep(0.2)
        time.sleep(0.1)

    try:
        if hasattr(bz, "off"):
            bz.off()
    except Exception:
        pass


def cleanup_dir(path, pattern, keep_last):
    if keep_last <= 0:
        return
    files = glob.glob(os.path.join(path, pattern))
    if len(files) <= keep_last:
        return
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    for f in files_sorted[keep_last:]:
        try:
            os.remove(f)
        except Exception:
            pass


def cleanup_outputs(keep_raw: int, keep_out: int):
    cleanup_dir(SAVE_DIR, "auto_*.jpg", keep_last=keep_raw)
    cleanup_dir(OUT_DIR, "*.jpg",  keep_last=keep_out)
    cleanup_dir(OUT_DIR, "*.json", keep_last=keep_out)
    cleanup_dir(OUT_DIR, "*.csv",  keep_last=keep_out)


def main():
    global _BUZZER, _ALARM_THREAD, _ALARM_STOP_EVT

    args = parse_args()

    expected = args.expected
    if expected <= 0:
        while True:
            s = input("Please enter expected batteries per pack (1/2/4/...): ").strip()
            if s.isdigit() and int(s) > 0:
                expected = int(s)
                break
            print("❗ Invalid input, try again.")
    print(f"🔢 Current packaging settings: {expected} batteries per pack")

    trigger_distance_m = float(args.trigger_distance)
    cooldown_s = float(args.cooldown)

    if args.buzzer_pin and args.buzzer_pin > 0:
        _BUZZER = Buzzer(pin=args.buzzer_pin, active_high=True)
        print(f"🔔 Buzzer GPIO (BCM): {args.buzzer_pin} (press Enter to stop when NG)")
    else:
        print("🔔 No buzzer GPIO set (no ringing)")

    print("🟢 System ready. Waiting for object...")

    try:
        while True:
            dist_cm = get_distance_cm()
            print(f"Distance: {dist_cm:5.1f} cm", end="\r")

            if dist_cm < trigger_distance_m * 100.0:
                print(f"\n📏 Object detected ({dist_cm:.1f} cm) → Capturing image...")
                img_path, base_name = capture_image()
                ng_count, _ = run_detection(img_path, base_name, expected)

                cleanup_outputs(args.keep_raw, args.keep_out)

                if ng_count > 0 and _BUZZER is not None:
                    print(f"❌ {ng_count} NG packages detected, continuous alerts initiated!")
                    _ALARM_STOP_EVT = threading.Event()
                    _ALARM_THREAD = threading.Thread(
                        target=alarm_beep_loop,
                        args=(_BUZZER, _ALARM_STOP_EVT),
                        daemon=True,
                    )
                    _ALARM_THREAD.start()
                    try:
                        input("🔕 Press Enter to stop the buzzer and continue...\n")
                    except EOFError:
                        # GUI 关闭了stdin
                        pass
                    finally:
                        # 仅静音，不关闭GPIO，以便后续还能响
                        _alarm_quietly()
                        print("🔕 The alarm has been stopped.")

                print(f"⏸ Cooling {cooldown_s:.1f} s before next trigger...\n")
                time.sleep(cooldown_s)
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 Exited by user.")
    except SystemExit:
        print("\n🛑 Terminated.")
    except Exception as e:
        print("\n❌ Error:", e)
    finally:
        # 仅在进程真正退出时才释放 GPIO
        _buzzer_close_all()


if __name__ == "__main__":
    main()
