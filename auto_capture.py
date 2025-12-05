# auto_capture.py
# VERSION: 2024-11-17-v3-BUGFIX + STEPPER-WIPER-FIX
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

# ---------- 全局：蜂鸣器 / 报警线程 / LED / 步进配置 ----------
_BUZZER = None
_ALARM_THREAD = None
_ALARM_STOP_EVT = None
_SHUTDOWN_FLAG = False
_STATUS_LED = None

_STEPPER_ENABLED = False
_STEPPER_CFG = dict(step_pin=5, dir_pin=6, microstep=8)


# ========== LED 相关 ==========

def _init_status_led(pin: int = None):
    """
    初始化状态 LED（拍照后闪烁用）
    pin = None 或 pin < 0 表示不启用
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
    让状态 LED 闪几下，用于“拍照完成”提示。
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
    进程退出时关闭并释放 LED
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


# ========== 步进电机相关 ==========

def _init_stepper(args):
    """
    初始化步进电机（DM556 + NEMA23）。
    这里只保存配置，并做一次“空初始化测试”，
    真正的动作在每次 stepper_wiper_swing 里重新 init/close，避免第一次状态异常。
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

    # 做一次初始化+关闭测试，确认 wiring 正常
    ok = init_stepper_dm556(**_STEPPER_CFG)
    if ok:
        print("[Stepper] Initial test OK, closing warm-up instance.", flush=True)
        close_stepper_dm556()
        _STEPPER_ENABLED = True
    else:
        print("[Stepper] Initial test FAILED, stepper disabled.", flush=True)
        _STEPPER_ENABLED = False


def _stepper_close():
    """进程退出时释放步进电机 GPIO（保险起见再关一下）"""
    global _STEPPER_ENABLED
    try:
        close_stepper_dm556()
    except Exception:
        pass
    _STEPPER_ENABLED = False


def stepper_wiper_swing(degrees: float = 90.0, speed_rps: float = 0.3):
    """
    步进电机做一次“雨刮器”摆动：
    - 先逆向转 degrees°
    - 再顺向转回来 degrees°

    ❗注意：
    为了避免你遇到的「第一次只转一边、第二次才正常」的问题，
    这里每一次调用都会：
      1. init_stepper_dm556(...)
      2. 转出去
      3. 再转回来
      4. close_stepper_dm556()
    这样每一轮状态都是干净的。
    """
    global _STEPPER_ENABLED, _STEPPER_CFG
    if not _STEPPER_ENABLED:
        return

    try:
        # 每次先初始化（新建一套 GPIOZero 对象）
        if not init_stepper_dm556(**_STEPPER_CFG):
            print("[Stepper] init failed inside wiper_swing, skip.", flush=True)
            return

        print(f"[Stepper] Wiper swing start: {degrees}° ↔, speed={speed_rps} rev/s", flush=True)

        # 1）先“逆向”
        #    如果你看到方向反了，可以把 clockwise=True / False 对调
        stepper_move_degrees(degrees, clockwise=False, speed_rps=speed_rps)
        time.sleep(0.1)

        # 2）再“顺向”转回来
        stepper_move_degrees(degrees, clockwise=True, speed_rps=speed_rps)

        print("[Stepper] Wiper swing done.", flush=True)
    except Exception as e:
        print(f"[Stepper] Error during wiper swing: {e}", flush=True)
    finally:
        # 每次用完都释放 GPIO
        try:
            close_stepper_dm556()
        except Exception:
            pass


# ========== 蜂鸣器 / 报警 ==========

def _alarm_quietly():
    """仅停止报警：停线程 + 拉低；不释放GPIO。"""
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
    """真正退出进程时调用：确保停报警并释放 GPIO。"""
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

    # 同时释放 LED + 步进电机
    _led_close()
    _stepper_close()


def _handle_signal(signum, frame):
    print(f"\n[Signal {signum}] Shutting down.", flush=True)
    _buzzer_close_all()
    sys.exit(0)


atexit.register(_buzzer_close_all)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ========== 其它原有逻辑（参数、拍照、检测、报警）==========

def parse_args():
    ap = argparse.ArgumentParser("Ultrasonic-triggered auto capture + detect")
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--buzzer_pin", type=int, default=21)
    ap.add_argument("--trigger_distance", type=float, default=0.12)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--keep_raw", type=int, default=200)
    ap.add_argument("--keep_out", type=int, default=500)
    # LED 引脚（BCM 号），默认 20；传 -1 表示不用 LED
    ap.add_argument("--led_pin", type=int, default=20,
                    help="BCM pin for status LED (blink after capture); set -1 to disable")

    # 步进电机参数（DM556 + NEMA23）
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
        default=90.0,   # 👈 想转少一点，比如 45°，就改这里
        help="Wiper swing angle in degrees (one side). Default=90",
    )
    ap.add_argument(
        "--stepper_speed",
        type=float,
        default=0.3,
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
    === 核心修复函数 ===
    每次调用都创建全新的蜂鸣器实例
    这是解决重启后蜂鸣器不响的关键！
    """
    global _BUZZER
    
    print(f"[BuzzerFix] === Starting fresh buzzer creation on GPIO {pin} ===", flush=True)
    
    # 步骤 1: 彻底关闭旧实例
    if _BUZZER is not None:
        print("[BuzzerFix] Step 1: Closing old buzzer instance.", flush=True)
        try:
            _BUZZER.off()  # 先拉低
            time.sleep(0.05)
            _BUZZER.close()  # 释放 GPIO
            print("[BuzzerFix] Old buzzer closed successfully", flush=True)
        except Exception as e:
            print(f"[BuzzerFix] Warning: Error closing old buzzer: {e}", flush=True)
        finally:
            _BUZZER = None
    else:
        print("[BuzzerFix] Step 1: No old buzzer to close", flush=True)
    
    # 步骤 2: 等待 GPIO 完全释放
    print("[BuzzerFix] Step 2: Waiting for GPIO to release.", flush=True)
    time.sleep(0.15)
    
    # 步骤 3: 创建新实例
    print(f"[BuzzerFix] Step 3: Creating NEW buzzer on GPIO {pin}.", flush=True)
    try:
        _BUZZER = Buzzer(pin=pin, active_high=True)
        print("[BuzzerFix] New buzzer created successfully!", flush=True)
        
        # 步骤 4: 自检测试
        print("[BuzzerFix] Step 4: Running self-test...", flush=True)
        try:
            _BUZZER.on()
            time.sleep(0.08)  # 稍微长一点，确保能听到
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
    """持续报警：保持 ON，直到 stop_evt 置位"""
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
    """等待 GUI 发送停止信号"""
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

    # 初始化状态 LED + 步进电机配置
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
        print(f"🔔 Buzzer GPIO (BCM): {buzzer_pin}（有 NG 时将持续报警，GUI 点『Stop Alarm』静音）", flush=True)
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
                
                # ✅ 拍照完成 → 闪一下状态灯
                _led_blink(times=2, on_ms=120, off_ms=120)

                # ✅ 拍照完成 → 步进电机雨刮器动作
                stepper_wiper_swing(
                    degrees=args.stepper_deg,
                    speed_rps=args.stepper_speed,
                )
        
                ng, _ = run_detection(img, base, expected)
                _cleanup_outputs(args.keep_raw, args.keep_out)

                print(f"[Main] Detection complete: NG count = {ng}", flush=True)

                if ng > 0 and buzzer_pin:
                    print("[Main] ⚠️⚠️⚠️ NG detected, starting alarm. ⚠️⚠️⚠️", flush=True)
                    
                    # ============ 核心修复：每次都重新创建蜂鸣器 ============
                    buzzer_ready = _create_fresh_buzzer(buzzer_pin)
                    
                    if buzzer_ready and _BUZZER is not None:
                        print("[Main] Buzzer ready, starting alarm thread.", flush=True)
                        
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
                        print("🔴 Alarm thread started! You should hear continuous beeping now!", flush=True)

                        # 等待 GUI 发送停止信号
                        wait_enter_from_gui(_ALARM_STOP_EVT)

                        # 收到信号 → 停报警
                        _alarm_quietly()
                        print("🔕 Alarm stopped by GUI", flush=True)
                    else:
                        print("⚠️⚠️⚠️ BUZZER NOT AVAILABLE - NO ALARM WILL SOUND ⚠️⚠️⚠️", flush=True)
                else:
                    # 无 NG：确保没有残留报警线程
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
