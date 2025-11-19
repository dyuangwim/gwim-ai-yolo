#!/usr/bin/env python3
import time
from gpiozero import DigitalOutputDevice

# ======== L298N IN1~IN4 所接的 BCM 脚位（请确认接在 32/36/38/40）========
PIN_IN1 = 12   # 物理 PIN 32
PIN_IN2 = 16   # 物理 PIN 36
PIN_IN3 = 20   # 物理 PIN 38
PIN_IN4 = 21   # 物理 PIN 40

# 创建 4 个输出
coil_pins = [
    DigitalOutputDevice(PIN_IN1, initial_value=False),
    DigitalOutputDevice(PIN_IN2, initial_value=False),
    DigitalOutputDevice(PIN_IN3, initial_value=False),
    DigitalOutputDevice(PIN_IN4, initial_value=False),
]

# 4 步全步进序列（适合 L298N 驱动 2 相步进）
SEQ_FULLSTEP = [
    (1, 0, 1, 0),  # A+ B+
    (0, 1, 1, 0),  # A- B+
    (0, 1, 0, 1),  # A- B-
    (1, 0, 0, 1),  # A+ B-
]

_seq_index = 0


def step_once(direction=1, delay=0.003):
    """
    direction = 1  顺时针
    direction = -1 逆时针
    delay 越小转得越快，先用 3ms 安全一点
    """
    global _seq_index
    _seq_index = (_seq_index + direction) % len(SEQ_FULLSTEP)
    pattern = SEQ_FULLSTEP[_seq_index]

    for dev, val in zip(coil_pins, pattern):
        dev.value = bool(val)

    time.sleep(delay)


def release_motor():
    """释放电机（全部拉低），减少发热"""
    for dev in coil_pins:
        dev.off()


def turn_steps(steps, direction=1, delay=0.003):
    for _ in range(steps):
        step_once(direction=direction, delay=delay)


if __name__ == "__main__":
    steps_per_rev = 200   # NEMA17 一般是 200 步一圈
    delay = 0.003

    try:
        print("先锁住线圈 2 秒钟，检查轴是否变紧...")
        # 先让一格 pattern 上电，感受一下轴有没有锁住
        for dev, val in zip(coil_pins, SEQ_FULLSTEP[0]):
            dev.value = bool(val)
        time.sleep(2)

        print("➡ 顺时针转 1 圈...")
        turn_steps(steps_per_rev, direction=1, delay=delay)

        time.sleep(0.5)

        print("⬅ 逆时针转 1/2 圈...")
        turn_steps(steps_per_rev // 2, direction=-1, delay=delay)

        print("保持 2 秒，然后释放电机")
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        release_motor()
        print("电机已释放，程序结束")
