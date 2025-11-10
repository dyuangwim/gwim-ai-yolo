from gpiozero import DistanceSensor
from time import sleep

TRIG_PIN = 23  # BCM
ECHO_PIN = 24  # BCM

sensor = DistanceSensor(echo=ECHO_PIN,
                        trigger=TRIG_PIN,
                        max_distance=1.0)

print("Start measuring... (Ctrl+C to exit)")

try:
    while True:
        d = sensor.distance * 100  # 转成 cm
        print(f"Distance: {d:5.1f} cm", end="\r")
        sleep(0.2)
except KeyboardInterrupt:
    print("\nBye")
