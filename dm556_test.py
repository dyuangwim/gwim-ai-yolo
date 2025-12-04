#!/usr/bin/env python3
import time
from gpiozero import DigitalOutputDevice

# ========= GPIO 引脚设置（BCM 编号）=========
STEP_PIN = 5    # 接 DM556 的 PUL-
DIR_PIN  = 6    # 接 DM556 的 DIR-

# ========= 根据你 DM556 DIP 开关设置的微步 =========
# 如果你的微步设置是 8，比如 SW5-SW8 = 0100，就填 8
# 如果你设成 16，就填 16
MICROSTEP = 8

# 基本步距：1.8° → 200 steps / rev
STEPS_PER_REV = 200 * MICROSTEP

# 创建 GPIO 输出对象
step = DigitalOutputDevice(STEP_PIN, initial_value=False)
direction = DigitalOutputDevice(DIR_PIN, initial_value=False)

def pulse_step(delay):
    """发送一个 step 脉冲"""
    step.on()
    time.sleep(delay)
    step.off()
    time.sleep(delay)

def move_steps(steps, clockwise=True, speed_rps=0.2):
    """
    steps: 行走多少步（与 STEPS_PER_REV 同单位）
    clockwise: True=顺时针，False=逆时针
    speed_rps: 每秒多少圈（0.2 = 很慢，很适合第一次测试）
    """
    # 设置方向
    direction.value = 1 if clockwise else 0

    # 计算延迟
    steps_per_sec = STEPS_PER_REV * speed_rps
    delay = 1.0 / steps_per_sec / 2.0

    for _ in range(steps):
        pulse_step(delay)

def move_degrees(degrees, clockwise=True, speed_rps=0.2):
    """按角度转动（例如 90°, 45°）"""
    ratio = degrees / 360.0
    steps = int(STEPS_PER_REV * ratio)
    move_steps(steps, clockwise, speed_rps)

if __name__ == "__main__":
    print("=== DM556 + NEMA23 步进电机 测试程序 ===")
    print("5 秒后开始，请确保没有东西挡着电机！")
    time.sleep(5)

    try:
        print("1) 顺时针转 1 圈...")
        move_steps(STEPS_PER_REV, clockwise=True, speed_rps=0.2)

        time.sleep(1)

        print("2) 逆时针转 1 圈...")
        move_steps(STEPS_PER_REV, clockwise=False, speed_rps=0.2)

        time.sleep(1)

        print("3) 顺时针小角度摆动（45°）...")
        move_degrees(45, clockwise=True, speed_rps=0.3)
        time.sleep(0.5)

        print("4) 再逆时针摆回（45°）...")
        move_degrees(45, clockwise=False, speed_rps=0.3)

        print("测试完成！")

    except KeyboardInterrupt:
        print("\n用户中断。")

    finally:
        step.off()
        direction.off()
        print("GPIO 已释放，程序结束。")
