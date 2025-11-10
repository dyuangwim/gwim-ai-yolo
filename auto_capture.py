import os, time, cv2, subprocess
from datetime import datetime
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ----------------- 配置区 -----------------
TRIG_PIN = 23      # Ultrasonic Trigger pin (GPIO23)
ECHO_PIN = 24      # Ultrasonic Echo pin (GPIO24)
DIST_THRESHOLD = 12  # cm — 当距离小于这个值就触发拍照
SAVE_DIR = "/home/pi/batch_images"
OUT_DIR = "/home/pi/batch_out_pt"
CARD_MODEL = "/home/pi/models/card.pt"
BAT_MODEL  = "/home/pi/models/battery.pt"
EXPECTED = 2       # 每包期望电池数量
# -----------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 初始化 ultrasonic sensor
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)

def get_distance():
    """测距函数（单位 cm）"""
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    start = time.time()
    stop  = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        start = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        stop = time.time()
    elapsed = stop - start
    distance = (elapsed * 34300) / 2  # 声速 343m/s
    return distance

def capture_image():
    """拍照并返回保存路径"""
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (1920,1080), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)
    frame = picam2.capture_array("main")
    picam2.stop()
    picam2.close()

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
            dist = get_distance()
            if dist < DIST_THRESHOLD:
                print(f"📏 Object detected ({dist:.1f} cm) → Capturing image...")
                img_path = capture_image()
                run_detection(img_path)
                print("⏸ Cooling 5 s before next trigger...\n")
                time.sleep(5)
            time.sleep(0.2)
    except KeyboardInterrupt:
        GPIO.cleanup()
        print("\n🛑 Exited by user.")
    except Exception as e:
        GPIO.cleanup()
        print("❌ Error:", e)

if __name__ == "__main__":
    main_loop()
