# auto_capture.py
import os, time, cv2, subprocess, warnings, argparse
from datetime import datetime
from picamera2 import Picamera2
from gpiozero import DistanceSensor
from gpiozero.input_devices import DistanceSensorNoEcho

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
MAX_DISTANCE_M = 1.0  # 测距量程

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 初始化超声波传感器
sensor = DistanceSensor(
    echo=ECHO_PIN,
    trigger=TRIG_PIN,
    max_distance=MAX_DISTANCE_M
)

def parse_args():
    ap = argparse.ArgumentParser("Ultrasonic-triggered auto capture + battery batch detect")
    ap.add_argument(
        "--expected", type=int, default=0,
        help="每包应有的电池数量，如 1/2/3/4（<=0 则启动时手动输入）"
    )
    ap.add_argument(
        "--buzzer_pin", type=int, default=None,
        help="蜂鸣器 GPIO（BCM 编号，例如 21）；不设则不响铃"
    )
    ap.add_argument(
        "--trigger_distance", type=float, default=0.12,
        help="触发距离（米），默认 0.12 = 12cm"
    )
    ap.add_argument(
        "--cooldown", type=float, default=5.0,
        help="每次触发后冷却时间（秒），默认 5s"
    )
    return ap.parse_args()

def get_distance_cm():
    """返回当前测得的距离（单位 cm）"""
    d_m = sensor.distance * MAX_DISTANCE_M
    return d_m * 100.0

def capture_image():
    """
    拍照并返回保存路径
    使用与 run_async 相同的相机配置和颜色转换，
    确保视角和颜色一致。
    """
    picam2 = Picamera2()

    cfg = picam2.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "YUV420"},
        controls={"FrameRate": 30}
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
    save_path = os.path.join(SAVE_DIR, f"auto_{ts}.jpg")
    cv2.imwrite(save_path, bgr)
    print(f"📸 Captured: {save_path} ({CAM_W}x{CAM_H})")
    return save_path

def run_detection(img_path, expected, buzzer_pin):
    """
    调用 detect_batch.py 进行检测
    - expected: 每包电池数
    - buzzer_pin: 若非 None，则传给 detect_batch，让其在有 NG 时控制蜂鸣器
    """
    cmd = [
        "python3", "/home/pi/battery_batch/detect_batch.py",
        "--card_weights", CARD_MODEL,
        "--bat_weights",  BAT_MODEL,
        "--img", img_path,
        "--expected", str(expected),
        "--rotate", "0",
        "--out_dir", OUT_DIR,
        "--card_conf", "0.50"
    ]
    if buzzer_pin is not None:
        cmd += ["--buzzer_pin", str(buzzer_pin)]

    print("🚀 Running detection...")
    subprocess.run(cmd)
    print("✅ Detection finished.\n")

def main():
    args = parse_args()

    # 1) 处理 expected：<=0 时，让用户在启动时输入
    expected = args.expected
    if expected <= 0:
        while True:
            s = input("请输入每包应有的电池数量 (1/2/3/4/...): ").strip()
            if s.isdigit() and int(s) > 0:
                expected = int(s)
                break
            print("❗ 输入无效，请重新输入一个正整数。")
    print(f"🔢 当前包装设定: 每包 {expected} 颗电池")

    # 2) trigger 距离 & 冷却时间 & buzzer pin
    trigger_distance_m = float(args.trigger_distance)
    cooldown_s = float(args.cooldown)
    buzzer_pin = args.buzzer_pin
    if buzzer_pin is not None:
        print(f"🔔 蜂鸣器 GPIO (BCM): {buzzer_pin}（有 NG 时会响）")
    else:
        print("🔔 未设置蜂鸣器 GPIO（检测 NG 不会响铃）")

    print("🟢 System ready. Waiting for object...")

    try:
        while True:
            dist_cm = get_distance_cm()
            print(f"Distance: {dist_cm:5.1f} cm", end="\r")

            if dist_cm < trigger_distance_m * 100.0:
                print(f"\n📏 Object detected ({dist_cm:.1f} cm) → Capturing image...")
                img_path = capture_image()
                run_detection(img_path, expected, buzzer_pin)
                print(f"⏸ Cooling {cooldown_s:.1f} s before next trigger...\n")
                time.sleep(cooldown_s)
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 Exited by user.")
    except Exception as e:
        print("\n❌ Error:", e)

if __name__ == "__main__":
    main()
