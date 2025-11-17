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

from utils_hw import Buzzer  # 我们用自己写的 gpiozero 版 Buzzer

# 忽略 DistanceSensor 的无回波警告（不影响实际测距逻辑）
warnings.filterwarnings("ignore", category=DistanceSensorNoEcho)

# 相机分辨率（保持与 run_async / yolo_detector 一致）
CAM_W, CAM_H = 1280, 960  # 4:3

SAVE_DIR  = "/home/pi/batch_images"
OUT_DIR   = "/home/pi/batch_out_pt"
CARD_MODEL = "/home/pi/models/card.pt"
BAT_MODEL  = "/home/pi/models/battery.pt"

# Ultrasonic 引脚配置（BCM 编号）
TRIG_PIN = 23
ECHO_PIN = 24
MAX_DISTANCE_M = 1.0  # 测距量程（1m）

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 初始化超声波传感器
sensor = DistanceSensor(
    echo=ECHO_PIN,
    trigger=TRIG_PIN,
    max_distance=MAX_DISTANCE_M,
)

# ---------- 全局对象，确保任何退出路径都能 OFF ----------
_BUZZER = None                # type: Buzzer | None
_ALARM_THREAD = None          # type: threading.Thread | None
_ALARM_STOP_EVT = None        # type: threading.Event | None


def _buzzer_off_safely():
    """无论何种退出都将蜂鸣器拉低。"""
    global _BUZZER, _ALARM_STOP_EVT, _ALARM_THREAD
    try:
        if _ALARM_STOP_EVT is not None:
            _ALARM_STOP_EVT.set()
        if _ALARM_THREAD is not None and _ALARM_THREAD.is_alive():
            _ALARM_THREAD.join(timeout=0.3)
    except Exception:
        pass
    try:
        if _BUZZER is not None:
            # utils_hw.Buzzer.off() 封装在 close() 里
            _BUZZER.close()
    except Exception:
        pass

def _handle_signal(signum, frame):
    # 收到 SIGINT / SIGTERM 时，确保蜂鸣器关闭，再优雅退出
    _buzzer_off_safely()
    # 直接退出整个进程
    raise SystemExit(0)

# 注册退出钩子与信号处理
atexit.register(_buzzer_off_safely)
signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def parse_args():
    ap = argparse.ArgumentParser("Ultrasonic-triggered auto capture + battery batch detect")
    ap.add_argument(
        "--expected", type=int, default=0,
        help="每包应有的电池数量，如 1/2/3/4（<=0 则启动时手动输入）",
    )
    # ⭐ 默认直接用 GPIO21，这样就不用每次都打 --buzzer_pin 21 了
    ap.add_argument(
        "--buzzer_pin", type=int, default=21,
        help="蜂鸣器 GPIO（BCM 编号），默认 21；设为 0 或负数则关闭蜂鸣器",
    )
    ap.add_argument(
        "--trigger_distance", type=float, default=0.12,
        help="触发距离（米），默认 0.12 = 12cm",
    )
    ap.add_argument(
        "--cooldown", type=float, default=5.0,
        help="每次触发后冷却时间（秒），默认 5s",
    )
    ap.add_argument(
        "--keep_raw", type=int, default=200,
        help="最多保留多少张原始自动照片 (SAVE_DIR) 默认 200 张",
    )
    ap.add_argument(
        "--keep_out", type=int, default=500,
        help="最多保留多少组输出文件 (OUT_DIR 下的 jpg/json/csv)，默认 500 组",
    )
    return ap.parse_args()


def get_distance_cm():
    """返回当前测得的距离（单位 cm）"""
    d_m = sensor.distance * MAX_DISTANCE_M
    return d_m * 100.0


def capture_image():
    """
    拍照并返回 (图片路径, base_name)
    - base_name 用来让 detect_batch.py 存同名的 jpg/json/csv
    """
    picam2 = Picamera2()

    cfg = picam2.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "YUV420"},
        controls={"FrameRate": 30},
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.5)  # 稍微等一下，让曝光稳定

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
    """
    调用 detect_batch.py 进行检测
    - 不再把 buzzer_pin 传给 detect_batch，让报警由本文件统一处理（持续响）
    - save_name 使用 base_name，这样 JSON 路径可以预先知道
    返回:
        (ng_count, json_path)
    """
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

    # 解析 JSON 判断 NG 数量
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
    """
    在独立线程里循环 beep，直到 stop_event 被 set。
    """
    # 额外保护：进入循环前先确保 OFF
    try:
        bz.off = getattr(bz, "off", None)
        if callable(bz.off):
            bz.off()
    except Exception:
        pass

    while not stop_event.is_set():
        try:
            bz.beep(200)        # 响 200ms
        except Exception:
            # 硬件异常也不中断主流程
            time.sleep(0.2)
        time.sleep(0.1)         # 间隔 100ms

    # 退出循环再确保一次 OFF
    try:
        if callable(getattr(bz, "off", None)):
            bz.off()
    except Exception:
        pass


def cleanup_dir(path, pattern, keep_last):
    """
    只保留最新 keep_last 个匹配 pattern 的文件，其余自动删除。
    """
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
    """
    清理 SAVE_DIR 和 OUT_DIR 里的旧文件，防止 sd card 用爆。
    """
    # 原始自动拍照图
    cleanup_dir(SAVE_DIR, "auto_*.jpg", keep_last=keep_raw)

    # 输出图 / json / csv
    cleanup_dir(OUT_DIR, "*.jpg",  keep_last=keep_out)
    cleanup_dir(OUT_DIR, "*.json", keep_last=keep_out)
    cleanup_dir(OUT_DIR, "*.csv",  keep_last=keep_out)


def main():
    global _BUZZER, _ALARM_THREAD, _ALARM_STOP_EVT

    args = parse_args()

    # 1) 处理 expected：<=0 时，让用户在启动时输入
    expected = args.expected
    if expected <= 0:
        while True:
            s = input(
                "Please enter the number of batteries that should be in each pack (1/2/3/4/...):"
            ).strip()
            if s.isdigit() and int(s) > 0:
                expected = int(s)
                break
            print("❗ Invalid input, please enter a positive integer again.")
    print(f"🔢 Current packaging settings: {expected} batteries per pack")

    trigger_distance_m = float(args.trigger_distance)
    cooldown_s = float(args.cooldown)

    # 2) 初始化蜂鸣器（统一在这里管理报警）
    if args.buzzer_pin is not None and args.buzzer_pin > 0:
        _BUZZER = Buzzer(pin=args.buzzer_pin, active_high=True)
        print(
            f"🔔 Buzzer GPIO (BCM): {args.buzzer_pin} "
            f"(An alarm will continuously sound if there is an NG (Not Okay) error; press Enter to stop.)"
        )
    else:
        print("🔔 No buzzer GPIO set (no ringing when detecting NG)")

    print("🟢 System ready. Waiting for object...")

    try:
        while True:
            dist_cm = get_distance_cm()
            print(f"Distance: {dist_cm:5.1f} cm", end="\r")

            if dist_cm < trigger_distance_m * 100.0:
                print(f"\n📏 Object detected ({dist_cm:.1f} cm) → Capturing image...")
                img_path, base_name = capture_image()
                ng_count, _ = run_detection(img_path, base_name, expected)

                # 自动清理旧文件
                cleanup_outputs(args.keep_raw, args.keep_out)

                # 如果有 NG → 启动持续报警，直到用户按 Enter
                if ng_count > 0 and _BUZZER is not None:
                    print(f"❌ {ng_count} NG packages detected, continuous alerts initiated!")
                    _ALARM_STOP_EVT = threading.Event()
                    _ALARM_THREAD = threading.Thread(
                        target=alarm_beep_loop, args=(_BUZZER, _ALARM_STOP_EVT), daemon=True
                    )
                    _ALARM_THREAD.start()
                    try:
                        input("🔕 Press Enter to stop the buzzer and continue...\n")
                    except EOFError:
                        # 如果上游（GUI）关闭了 stdin，也立刻停警报并优雅退出
                        pass
                    finally:
                        if _ALARM_STOP_EVT is not None:
                            _ALARM_STOP_EVT.set()
                        if _ALARM_THREAD is not None:
                            _ALARM_THREAD.join(timeout=0.5)
                        # 再确保一次 OFF
                        _buzzer_off_safely()
                        print("🔕 The alarm has been stopped.")

                print(f"⏸ Cooling {cooldown_s:.1f} s before next trigger...\n")
                time.sleep(cooldown_s)
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 Exited by user.")
    except SystemExit:
        # 来自信号处理的退出
        print("\n🛑 Terminated.")
    except Exception as e:
        print("\n❌ Error:", e)
    finally:
        # 任何退出路径都确保拉低蜂鸣器
        _buzzer_off_safely()


if __name__ == "__main__":
    main()
