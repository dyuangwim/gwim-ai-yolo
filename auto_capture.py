# auto_capture.py
import os, time, cv2, json, glob, subprocess, warnings, argparse, threading, signal, atexit, sys
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

TRIG_PIN, ECHO_PIN, MAX_DISTANCE_M = 23, 24, 1.0
os.makedirs(SAVE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN, max_distance=MAX_DISTANCE_M)

# ---------- 全局：蜂鸣器与报警线程 ----------
_BUZZER = None
_ALARM_THREAD = None
_ALARM_STOP_EVT = None
_SHUTDOWN_FLAG = False  # 新增：全局退出标志

def _alarm_quietly():
    """仅停止报警：停线程 + 拉低；不释放GPIO。"""
    global _ALARM_THREAD, _ALARM_STOP_EVT, _BUZZER
    try:
        if _ALARM_STOP_EVT: _ALARM_STOP_EVT.set()
        if _ALARM_THREAD and _ALARM_THREAD.is_alive(): _ALARM_THREAD.join(timeout=0.8)
    except Exception:
        pass
    try:
        if _BUZZER and hasattr(_BUZZER, "off"):
            _BUZZER.off()
    except Exception:
        pass
    _ALARM_THREAD = None
    _ALARM_STOP_EVT = None

def _buzzer_close_all():
    """真正退出进程时调用：确保停报警并释放 GPIO。"""
    global _SHUTDOWN_FLAG
    _SHUTDOWN_FLAG = True
    _alarm_quietly()
    global _BUZZER
    try:
        if _BUZZER: _BUZZER.close()
    except Exception:
        pass
    finally:
        _BUZZER = None

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
    return sensor.distance * MAX_DISTANCE_M * 100.0

def capture_image():
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": (CAM_W, CAM_H), "format": "YUV420"}, controls={"FrameRate": 30})
    picam2.configure(cfg); picam2.start(); time.sleep(0.5)
    yuv = picam2.capture_array("main")
    try:
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    except:
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    picam2.stop(); picam2.close()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"auto_{ts}"; path = os.path.join(SAVE_DIR, f"{base}.jpg")
    cv2.imwrite(path, bgr)
    print(f"Image: {path}", flush=True)  # 添加 flush
    return path, base

def run_detection(img_path, base, expected):
    json_path = os.path.join(OUT_DIR, f"{base}.json")
    cmd = [
        "python3", "/home/pi/battery_batch/detect_batch.py",
        "--card_weights", CARD_MODEL, "--bat_weights", BAT_MODEL,
        "--img", img_path, "--expected", str(expected),
        "--rotate", "0", "--out_dir", OUT_DIR, "--card_conf", "0.50",
        "--save_name", base,
    ]
    subprocess.run(cmd)
    print(f"JSON: {json_path}", flush=True)  # 添加 flush
    ng = 0
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        packs = data.get("packs", [])
        ng = len([p for p in packs if not p.get("ok", True)])
    except Exception as e:
        print("⚠️ Failed to read JSON:", e, flush=True)
    return ng, json_path

def _ensure_buzzer_ready(pin: int):
    """确保蜂鸣器可用：如已释放/异常，重新创建；做一次非常短的自检。"""
    global _BUZZER
    need_new = (_BUZZER is None)
    if not need_new:
        try:
            _BUZZER.off()
        except Exception:
            need_new = True
    if need_new:
        try:
            _BUZZER = Buzzer(pin=pin, active_high=True)
            try:
                _BUZZER.on(); time.sleep(0.02); _BUZZER.off()
            except Exception:
                pass
        except Exception as e:
            print("⚠️ Recreate buzzer failed:", e, flush=True)
            _BUZZER = None

def alarm_continuous_loop(bz: Buzzer, stop_evt: threading.Event):
    """
    持续报警：保持 ON，直到 stop_evt 置位；最后确保 OFF。
    """
    try:
        bz.on()
        while not stop_evt.is_set():
            time.sleep(0.02)
    except Exception:
        try: bz.off()
        except Exception: pass
        return
    try: bz.off()
    except Exception: pass

def wait_enter_from_gui(stop_evt: threading.Event):
    """
    改进版：通过持续读取 stdin 来等待 GUI 的停止信号
    - 读到 '\n' 则停止报警
    - 读到 EOF 或异常则继续等待（防止误触发）
    - 检查全局退出标志
    """
    global _SHUTDOWN_FLAG
    
    while not stop_evt.is_set() and not _SHUTDOWN_FLAG:
        try:
            # 非阻塞读取，超时 0.1 秒
            line = sys.stdin.readline()
            
            if not line:  # EOF
                # stdin 被关闭，继续等待 stop_evt（可能是 GUI Stop 触发）
                time.sleep(0.1)
                continue
                
            # 收到有效输入（通常是 '\n'）
            if '\n' in line:
                break
                
        except (IOError, OSError):
            # stdin 读取错误，继续等待
            time.sleep(0.1)
            continue
        except KeyboardInterrupt:
            # Ctrl+C，退出
            break

def _cleanup_dir(path, pattern, keep_last):
    if keep_last <= 0: return
    files = sorted(glob.glob(os.path.join(path, pattern)), key=os.path.getmtime, reverse=True)
    for f in files[keep_last:]:
        try: os.remove(f)
        except Exception: pass

def _cleanup_outputs(keep_raw, keep_out):
    _cleanup_dir(SAVE_DIR, "auto_*.jpg", keep_raw)
    for ext in ("*.jpg", "*.json", "*.csv"):
        _cleanup_dir(OUT_DIR, ext, keep_out)

def main():
    global _ALARM_THREAD, _ALARM_STOP_EVT
    args = parse_args()

    expected = args.expected
    if expected <= 0:
        while True:
            s = input("Please enter expected batteries per pack (1/2/4/...): ").strip()
            if s.isdigit() and int(s) > 0:
                expected = int(s); break
            print("● Invalid input, try again.")
    print(f"📢 Current packaging settings: {expected} batteries per pack", flush=True)

    trig_cm = args.trigger_distance * 100.0; cooldown = args.cooldown
    buzzer_pin = args.buzzer_pin if args.buzzer_pin and args.buzzer_pin > 0 else None
    if buzzer_pin: 
        print(f"🔔 Buzzer GPIO (BCM): {buzzer_pin}（有 NG 时将持续报警，GUI 点『Stop Alarm』静音）", flush=True)
    else:
        print("🔔 No buzzer GPIO set (no ringing)", flush=True)

    print("🟢 System ready. Waiting for object...", flush=True)
    
    try:
        while not _SHUTDOWN_FLAG:
            d = get_distance_cm()
            time.sleep(0.15)
            
            if d < trig_cm:
                img, base = capture_image()
                ng, _ = run_detection(img, base, expected)
                _cleanup_outputs(args.keep_raw, args.keep_out)

                if ng > 0 and buzzer_pin:
                    _ensure_buzzer_ready(buzzer_pin)
                    if _BUZZER is not None:
                        # 停止旧报警（如果有）
                        _alarm_quietly()
                        
                        # 启动新报警
                        _ALARM_STOP_EVT = threading.Event()
                        _ALARM_THREAD = threading.Thread(
                            target=alarm_continuous_loop,
                            args=(_BUZZER, _ALARM_STOP_EVT),
                            daemon=True
                        )
                        _ALARM_THREAD.start()
                        print("🔴 NG detected! Alarm started. Waiting for GUI stop signal...", flush=True)

                        # 等待 GUI 发送停止信号
                        wait_enter_from_gui(_ALARM_STOP_EVT)

                        # 收到信号 → 停报警
                        _alarm_quietly()
                        print("🔕 The alarm has been stopped.", flush=True)
                else:
                    # 无 NG：确保没有残留报警线程
                    _alarm_quietly()

                time.sleep(cooldown)
                
    except KeyboardInterrupt:
        print("\n🛑 Exited by user.", flush=True)
    except SystemExit:
        print("\n🛑 Terminated.", flush=True)
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
    finally:
        _buzzer_close_all()

if __name__ == "__main__":
    main()
