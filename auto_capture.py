# auto_capture.py
import os, time, cv2, subprocess, warnings
from datetime import datetime
from picamera2 import Picamera2
from gpiozero import DistanceSensor
from gpiozero.input_devices import DistanceSensorNoEcho

# 忽略 DistanceSensor 的无回波警告
warnings.filterwarnings("ignore", category=DistanceSensorNoEcho)

# ----------------- 配置区 -----------------
TRIG_PIN = 23      # Ultrasonic Trigger pin (BCM 23)
ECHO_PIN = 24      # Ultrasonic Echo pin (BCM 24)
MAX_DISTANCE_M = 1.0       # 量程（单位: 米）
TRIGGER_DISTANCE_M = 0.12  # 触发阈值：0.12m = 12cm

SAVE_DIR = "/home/pi/batch_images"
OUT_DIR  = "/home/pi/batch_out_pt"
CARD_MODEL = "/home/pi/models/card.pt"
BAT_MODEL  = "/home/pi/models/battery.pt"
EXPECTED   = 2             # 每包期望电池数量
COOLDOWN_S = 5             # 每次触发后的冷却时间（防止连续多次触发）
CAM_WIDTH  = 1920
CAM_HEIGHT = 1080
# -----------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 初始化超声波传感器
sensor = DistanceSensor(
    echo=ECHO_PIN,
    trigger=TRIG_PIN,
    max_distance=MAX_DISTANCE_M
)

def get_distance_cm():
    """返回当前测得的距离（单位 cm）"""
    d_m = sensor.distance * MAX_DISTANCE_M
    return d_m * 100.0

def capture_image():
    """拍照并返回保存路径（颜色校正版本）"""
    picam2 = Picamera2()
    cfg = picam2.create_still_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "YUV420"},
        controls={"FrameDurationLimits": (33333, 33333)}  # 约 30FPS
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.5)  # 稍微等一下，让曝光稳定

    # ✅ 捕获 YUV 帧并手动转 BGR，避免颜色通道反转问题
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
    print(f"📸 Captured: {save_path}")
    return save_path

def run_detection(img_path):
    """调用 detect_batch.py 进行检测"""
    cmd = [
        "python3", "/home/pi/battery_batch/detect_batch.py",
        "--card_weights", CARD_MODEL,
        "--bat_weights", BAT_MODEL,
        "--img", img_path,
        "--expected", str(EXPECTED),
        "--rotate", "0",
        "--out_dir", OUT_DIR,
        "--card_conf", "0.50"
    ]
    print("🚀 Running detection...")
    subprocess.run(cmd)
    print("✅ Detection finished.\n")

def main_loop():
    print("🟢 System ready. Waiting for object...")

    try:
        while True:
            dist_cm = get_distance_cm()
            print(f"Distance: {dist_cm:5.1f} cm", end="\r")

            if dist_cm < TRIGGER_DISTANCE_M * 100.0:
                print(f"\n📏 Object detected ({dist_cm:.1f} cm) → Capturing image...")
                img_path = capture_image()
                run_detection(img_path)
                print(f"⏸ Cooling {COOLDOWN_S} s before next trigger...\n")
                time.sleep(COOLDOWN_S)
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 Exited by user.")
    except Exception as e:
        print("\n❌ Error:", e)

if __name__ == "__main__":
    main_loop()
