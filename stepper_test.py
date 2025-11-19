#!/usr/bin/env python3
import time
from gpiozero import DigitalOutputDevice

# 把这里改成你实际接到 L298N IN1~IN4 的 BCM 脚位
PINS = [5, 6, 13, 19]

devices = [DigitalOutputDevice(p, initial_value=False) for p in PINS]

print("开始轮流点亮 4 个输出，每个 1 秒 HIGH / 0.5 秒 LOW")
print("请观察 L298N 板子上的 IN/OUT LED 以及电机是否有吸住/抖动")
print("Ctrl + C 可以随时停止\n")

try:
    while True:
        for i, dev in enumerate(devices):
            print(f"GPIO {PINS[i]} -> HIGH")
            dev.on()
            time.sleep(1.0)

            print(f"GPIO {PINS[i]} -> LOW")
            dev.off()
            time.sleep(0.5)

except KeyboardInterrupt:
    print("\n用户中断，全部拉低输出。")
finally:
    for dev in devices:
        dev.off()
    print("结束。")
