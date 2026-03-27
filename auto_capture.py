# auto_capture.py
import os, time, cv2, json, glob, subprocess, warnings, argparse, threading, signal, atexit, sys
from datetime import datetime
from picamera2 import Picamera2
from gpiozero import DistanceSensor
from gpiozero.input_devices import DistanceSensorNoEcho
from gpiozero import LED as StatusLED

from utils_hw import (
    Buzzer,
    init_stepper_dm556,
    stepper_move_degrees,
    close_stepper_dm556,
)

warnings.filterwarnings("ignore", category=DistanceSensorNoEcho)

CAM_W, CAM_H = 1280, 960
SAVE_DIR  = "/home/pi/batch_images"
OUT_DIR   = "/home/pi/batch_out_pt"
CARD_MODEL = "/home/pi/models/card.pt"
BAT_MODEL  = "/home/pi/models/battery.pt"

TRIG_PIN, ECHO_PIN, MAX_DISTANCE_M = 23, 24, 1.0
os.makedirs(SAVE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIG_PIN, max_distance=MAX_DISTANCE_M)

# ---------- Global: Buzzer / Alarm Thread / LED / Step Configuration ----------
_BUZZER = None
_ALARM_THREAD = None
_ALARM_STOP_EVT = None
_SHUTDOWN_FLAG = False
_STATUS_LED = None

_STEPPER_ENABLED = False
_STEPPER_CFG = dict(step_pin=5, dir_pin=6, microstep=8)


# ========== LED related ==========

def _init_status_led(pin: int = None):
    """
    Initialization status LED (blinks after taking a picture)
    pin = None or pin < 0 indicates it is not enabled.
    """
    global _STATUS_LED
    if pin is None or pin < 0:
        print("[LED] No status LED configured", flush=True)
        _STATUS_LED = None
        return

    try:
        _STATUS_LED = StatusLED(pin)
        _STATUS_LED.off()
        print(f"[LED] Status LED initialized on GPIO {pin}", flush=True)
    except Exception as e:
        print(f"[LED] Failed to init status LED on GPIO {pin}: {e}", flush=True)
        _STATUS_LED = None


def _led_blink(times: int = 2, on_ms: int = 120, off_ms: int = 120):
    """
    Let the status LED blink a few times, used for "photo taken" notification.
    """
    global _STATUS_LED
    if _STATUS_LED is None:
        return

    for _ in range(times):
        try:
            _STATUS_LED.on()
            time.sleep(on_ms / 1000.0)
            _STATUS_LED.off()
            time.sleep(off_ms / 1000.0)
        except Exception as e:
            print(f"[LED] Error while blinking: {e}", flush=True)
            break


def _led_close():
    """
    Turn off and release LED when process exits
    """
    global _STATUS_LED
    if _STATUS_LED is None:
        return

    try:
        _STATUS_LED.off()
    except Exception:
        pass

    try:
        _STATUS_LED.close()
    except Exception:
        pass

    _STATUS_LED = None


# ========== Stepper Motor Related ==========

def _init_stepper(args):
    """
    Initialize the stepper motor (DM556 + NEMA23).
    Here we only save the configuration and do a "warm-up test",
    The actual actions are initiated/closed again in each stepper_wiper_swing to avoid the first state being abnormal.
    """
    global _STEPPER_ENABLED, _STEPPER_CFG

    if getattr(args, "no_stepper", False):
        print("[Stepper] Disabled by --no_stepper flag.", flush=True)
        _STEPPER_ENABLED = False
        return

    _STEPPER_CFG = dict(
        step_pin=args.stepper_step_pin,
        dir_pin=args.stepper_dir_pin,
        microstep=args.stepper_microstep,
    )

    # Perform an initialization and shutdown test to confirm that wiring is working correctly.
    ok = init_stepper_dm556(**_STEPPER_CFG)
    if ok:
        print("[Stepper] Initial test OK, closing warm-up instance.", flush=True)
        close_stepper_dm556()
        _STEPPER_ENABLED = True
    else:
        print("[Stepper] Initial test FAILED, stepper disabled.", flush=True)
        _STEPPER_ENABLED = False


def _stepper_close():
    """Release the stepper motor GPIO when the process exits (to be on the safe side, turn it off again)."""
    global _STEPPER_ENABLED
    try:
        close_stepper_dm556()
    except Exception:
        pass
    _STEPPER_ENABLED = False


def stepper_wiper_swing(degrees: float = 45.0, speed_rps: float = 0.1):
    """
    The stepper motor performs one "windshield wiper" oscillation:
    - First, rotate counter-clockwise by degrees°
    - Then rotate clockwise back by degrees°

    ❗Note:
    To avoid the problem of "only rotating one side the first time, and then working correctly the second time,"
    each call here will:
    1. init_stepper_dm556(...)
    2. Rotate outwards
    3. Rotate backwards
    4. close_stepper_dm556()
    This ensures a clean state for each cycle.
    """
    global _STEPPER_ENABLED, _STEPPER_CFG
    if not _STEPPER_ENABLED:
        return

    try:
        # Each time, initialize first (create a new set of GPIOZero objects).
        if not init_stepper_dm556(**_STEPPER_CFG):
            print("[Stepper] init failed inside wiper_swing, skip.", flush=True)
            return

        print(f"[Stepper] Wiper swing start: {degrees}° ↔, speed={speed_rps} rev/s", flush=True)

        # 1）First, "reverse" the process.
        #    If you see that the direction is reversed, you can swap clockwise=True/False.
        stepper_move_degrees(degrees, clockwise=False, speed_rps=speed_rps)
        time.sleep(0.1)

        # 2）Then turn back "forward".
        stepper_move_degrees(degrees, clockwise=True, speed_rps=speed_rps)

        print("[Stepper] Wiper swing done.", flush=True)
    except Exception as e:
        print(f"[Stepper] Error during wiper swing: {e}", flush=True)
    finally:
        # Each time, release the GPIO
        try:
            close_stepper_dm556()
        except Exception:
            pass


# ========== Buzzer / Alarm ==========

def _alarm_quietly():
    """Stop alarms only: stop the thread + pull low; do not release GPIO."""
    global _ALARM_THREAD, _ALARM_STOP_EVT, _BUZZER
    try:
        if _ALARM_STOP_EVT: 
            _ALARM_STOP_EVT.set()
        if _ALARM_THREAD and _ALARM_THREAD.is_alive(): 
            _ALARM_THREAD.join(timeout=1.0)
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
    """Called when the process truly exits: Ensures alarms are stopped and GPIO is released.。"""
    global _SHUTDOWN_FLAG, _BUZZER
    _SHUTDOWN_FLAG = True
    _alarm_quietly()
    try:
        if _BUZZER: 
            _BUZZER.close()
    except Exception:
        pass
    finally:
        _BUZZER = None

    # Simultaneously release LED + stepper motor
    _led_close()
    _stepper_close()


def _handle_signal(signum, frame):
    print(f"\n[Signal {signum}] Shutting down.", flush=True)
    _buzzer_close_all()
    sys.exit(0)


atexit.register(_buzzer_close_all)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ========== Other existing logic (parameters, image capture, detection, alarm)==========

def parse_args():
    ap = argparse.ArgumentParser("Ultrasonic-triggered auto capture + detect")
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--buzzer_pin", type=int, default=21)
    ap.add_argument("--trigger_distance", type=float, default=0.12)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--keep_raw", type=int, default=200)
    ap.add_argument("--keep_out", type=int, default=500)
    # LED pin (BCM number), default 20; passing -1 indicates that the LED is not used.
    ap.add_argument("--led_pin", type=int, default=20,
                    help="BCM pin for status LED (blink after capture); set -1 to disable")

    # Stepper motor parameters (DM556 + NEMA23)
    ap.add_argument(
        "--stepper_step_pin",
        type=int,
        default=5,
        help="BCM pin for DM556 STEP (PUL-). Default=5",
    )
    ap.add_argument(
        "--stepper_dir_pin",
        type=int,
        default=6,
        help="BCM pin for DM556 DIR (DIR-). Default=6",
    )
    ap.add_argument(
        "--stepper_microstep",
        type=int,
        default=8,
        help="Microstep setting of DM556 (must match DIP switches). Default=8",
    )
    ap.add_argument(
        "--stepper_deg",
        type=float,
        default=50.0,   # 👈 Turn it slightly less, for example, 45 degrees, change it here.
        help="Wiper swing angle in degrees (one side). Default=90",
    )
    ap.add_argument(
        "--stepper_speed",
        type=float,
        default=0.1,
        help="Stepper speed in rev/s. Default=0.3",
    )
    ap.add_argument(
        "--no_stepper",
        action="store_true",
        help="Disable stepper motor control completely",
    )

    return ap.parse_args()


def get_distance_cm():
    return sensor.distance * MAX_DISTANCE_M * 100.0


def capture_image():
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "YUV420"}, 
        controls={"FrameRate": 30}
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
    base = f"auto_{ts}"
    path = os.path.join(SAVE_DIR, f"{base}.jpg")
    cv2.imwrite(path, bgr)
    print(f"Image: {path}", flush=True)
    return path, base


def run_detection(img_path, base, expected):
    json_path = os.path.join(OUT_DIR, f"{base}.json")
    cmd = [
        "python3", "/home/pi/battery_batch/detect_batch.py",
        "--card_weights", CARD_MODEL, 
        "--bat_weights", BAT_MODEL,
        "--img", img_path, 
        "--expected", str(expected),
        "--rotate", "0", 
        "--out_dir", OUT_DIR, 
        "--card_conf", "0.50",
        "--save_name", base,
    ]
    subprocess.run(cmd)
    print(f"JSON: {json_path}", flush=True)
    
    ng = 0
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        packs = data.get("packs", [])
        ng = len([p for p in packs if not p.get("ok", True)])
    except Exception as e:
        print(f"⚠️ Failed to read JSON: {e}", flush=True)
    return ng, json_path


def _create_fresh_buzzer(pin: int):
    """
    === Core Fix Function ===
    Creates a brand new buzzer instance on each call
    This is the key to resolving the buzzer not working after a restart!
    """
    global _BUZZER
    
    print(f"[BuzzerFix] === Starting fresh buzzer creation on GPIO {pin} ===", flush=True)
    
    # Step 1: Completely close the old instance
    if _BUZZER is not None:
        print("[BuzzerFix] Step 1: Closing old buzzer instance.", flush=True)
        try:
            _BUZZER.off()  # First pull low
            time.sleep(0.05)
            _BUZZER.close()  # Release GPIO
            print("[BuzzerFix] Old buzzer closed successfully", flush=True)
        except Exception as e:
            print(f"[BuzzerFix] Warning: Error closing old buzzer: {e}", flush=True)
        finally:
            _BUZZER = None
    else:
        print("[BuzzerFix] Step 1: No old buzzer to close", flush=True)
    
    # Step 2: Wait for GPIO to be fully released
    print("[BuzzerFix] Step 2: Waiting for GPIO to release.", flush=True)
    time.sleep(0.15)
    
    # Step 3: Create new instance
    print(f"[BuzzerFix] Step 3: Creating NEW buzzer on GPIO {pin}.", flush=True)
    try:
        _BUZZER = Buzzer(pin=pin, active_high=True)
        print("[BuzzerFix] New buzzer created successfully!", flush=True)
        
        # Step 4: Self-test
        print("[BuzzerFix] Step 4: Running self-test...", flush=True)
        try:
            _BUZZER.on()
            time.sleep(0.08)  # Slightly longer, ensure it's audible
            _BUZZER.off()
            print("[BuzzerFix] ✓ Self-test PASSED (you should have heard a short beep)", flush=True)
        except Exception as e:
            print(f"[BuzzerFix] ✗ Self-test FAILED: {e}", flush=True)
            
    except Exception as e:
        print(f"[BuzzerFix] ✗✗✗ CRITICAL ERROR: Failed to create buzzer: {e}", flush=True)
        _BUZZER = None
        
    print(f"[BuzzerFix] === Buzzer creation complete. Ready: {_BUZZER is not None} ===", flush=True)
    return _BUZZER is not None


def alarm_continuous_loop(bz: Buzzer, stop_evt: threading.Event):
    """Continuous alarm: Remain ON until stop_evt is set."""
    try:
        bz.on()
        print("[Alarm] 🔊🔊🔊 Buzzer ON - continuous alarm started 🔊🔊🔊", flush=True)
        while not stop_evt.is_set():
            time.sleep(0.05)
    except Exception as e:
        print(f"[Alarm] Error in alarm loop: {e}", flush=True)
    finally:
        try:
            bz.off()
            print("[Alarm] 🔇 Buzzer OFF - alarm stopped 🔇", flush=True)
        except Exception:
            pass


def wait_enter_from_gui(stop_evt: threading.Event, timeout_sec: float = 300.0):
    """Wait for stop signal from GUI"""
    global _SHUTDOWN_FLAG
    
    print("[WaitAlarm] Waiting for stop signal from GUI.", flush=True)
    start_time = time.time()
    
    import fcntl
    stdin_fd = sys.stdin.fileno()
    old_flags = fcntl.fcntl(stdin_fd, fcntl.F_GETFL)
    fcntl.fcntl(stdin_fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
    
    try:
        while not stop_evt.is_set() and not _SHUTDOWN_FLAG:
            if time.time() - start_time > timeout_sec:
                print("[WaitAlarm] Timeout reached, stopping alarm", flush=True)
                break
            
            try:
                data = sys.stdin.read(1024)
                if data:
                    print(f"[WaitAlarm] Received data from stdin: {repr(data)}", flush=True)
                    if '\n' in data:
                        print("[WaitAlarm] Stop signal received (newline)", flush=True)
                        break
                else:
                    time.sleep(0.1)
            except BlockingIOError:
                time.sleep(0.1)
            except (IOError, OSError) as e:
                print(f"[WaitAlarm] stdin read error: {e}, continuing.", flush=True)
                time.sleep(0.2)
            except Exception as e:
                print(f"[WaitAlarm] Unexpected error: {e}", flush=True)
                time.sleep(0.2)
                
    finally:
        try:
            fcntl.fcntl(stdin_fd, fcntl.F_SETFL, old_flags)
        except Exception:
            pass
    
    print("[WaitAlarm] Wait finished", flush=True)


def _cleanup_dir(path, pattern, keep_last):
    if keep_last <= 0: 
        return
    files = sorted(
        glob.glob(os.path.join(path, pattern)), 
        key=os.path.getmtime, 
        reverse=True
    )
    for f in files[keep_last:]:
        try: 
            os.remove(f)
        except Exception: 
            pass


def _cleanup_outputs(keep_raw, keep_out):
    _cleanup_dir(SAVE_DIR, "auto_*.jpg", keep_raw)
    for ext in ("*.jpg", "*.json", "*.csv"):
        _cleanup_dir(OUT_DIR, ext, keep_out)


def main():
    global _ALARM_THREAD, _ALARM_STOP_EVT
    
    print("=" * 60, flush=True)
    print("VERSION: 2024-11-17-v3-BUGFIX + STEPPER-WIPER-FIX", flush=True)
    print("=" * 60, flush=True)
    
    args = parse_args()

    # Initialization status LED + stepper motor configuration
    _init_status_led(args.led_pin)
    _init_stepper(args)

    expected = args.expected
    if expected <= 0:
        while True:
            s = input("Please enter expected batteries per pack (1/2/4/.): ").strip()
            if s.isdigit() and int(s) > 0:
                expected = int(s)
                break
            print("● Invalid input, try again.")
    
    print(f"📢 Current packaging settings: {expected} batteries per pack", flush=True)
    
    trig_cm = args.trigger_distance * 100.0
    cooldown = args.cooldown
    buzzer_pin = args.buzzer_pin if args.buzzer_pin and args.buzzer_pin > 0 else None
    
    if buzzer_pin:
        print(f"🔔 Buzzer GPIO (BCM): {buzzer_pin}（The alarm will continue to sound if there is an NG (Not Given) error. You can mute it by clicking 'Stop Alarm' in the GUI.）", flush=True)
    else:
        print("🔔 No buzzer GPIO set (no ringing)", flush=True)
    
    print("🟢 System ready. Waiting for object.", flush=True)
    
    try:
        while not _SHUTDOWN_FLAG:
            d = get_distance_cm()
            time.sleep(0.15)
            
            if d < trig_cm:
                print(f"[Main] Object detected at {d:.2f}cm, capturing.", flush=True)
                
                img, base = capture_image()
                
                # ✅ Photo taken → Status light flashes once
                _led_blink(times=2, on_ms=120, off_ms=120)

                # ✅ Photo taken → Stepper motor wiper operation
                stepper_wiper_swing(
                    degrees=args.stepper_deg,
                    speed_rps=args.stepper_speed,
                )
        
                ng, _ = run_detection(img, base, expected)
                _cleanup_outputs(args.keep_raw, args.keep_out)

                print(f"[Main] Detection complete: NG count = {ng}", flush=True)

                if ng > 0 and buzzer_pin:
                    print("[Main] ⚠️⚠️⚠️ NG detected, starting alarm. ⚠️⚠️⚠️", flush=True)
                    
                    # ============ Core fix: Recreate buzzer on each instance ============
                    buzzer_ready = _create_fresh_buzzer(buzzer_pin)
                    
                    if buzzer_ready and _BUZZER is not None:
                        print("[Main] Buzzer ready, starting alarm thread.", flush=True)
                        
                        # Stop old alarms (if any)
                        _alarm_quietly()
                        
                        # Activate new alarm
                        _ALARM_STOP_EVT = threading.Event()
                        _ALARM_THREAD = threading.Thread(
                            target=alarm_continuous_loop,
                            args=(_BUZZER, _ALARM_STOP_EVT),
                            daemon=True
                        )
                        _ALARM_THREAD.start()
                        print("🔴 Alarm thread started! You should hear continuous beeping now!", flush=True)

                        # Waiting for the GUI to send a stop signal
                        wait_enter_from_gui(_ALARM_STOP_EVT)

                        # Receive signal → Stop alarm
                        _alarm_quietly()
                        print("🔕 Alarm stopped by GUI", flush=True)
                    else:
                        print("⚠️⚠️⚠️ BUZZER NOT AVAILABLE - NO ALARM WILL SOUND ⚠️⚠️⚠️", flush=True)
                else:
                    # No NG: Ensures there are no residual alarm threads.
                    _alarm_quietly()
                    if ng == 0:
                        print("✅ All packs passed inspection", flush=True)

                time.sleep(cooldown)
                
    except KeyboardInterrupt:
        print("\n🛑 Exited by user (Ctrl+C)", flush=True)
    except SystemExit:
        print("\n🛑 System exit", flush=True)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        print("[Main] Cleanup.", flush=True)
        _buzzer_close_all()
        print("[Main] Exit complete", flush=True)


if __name__ == "__main__":
    main()
