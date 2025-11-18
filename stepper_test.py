#!/usr/bin/env python3
import time
from gpiozero import DigitalOutputDevice

# ============================
# 根据你实际接线修改这 4 个脚位（BCM 编号）
# ============================
PIN_IN1 = 5    # L298N IN1
PIN_IN2 = 6    # L298N IN2
PIN_IN3 = 13   # L298N IN3
PIN_IN4 = 19   # L298N IN4

# 创建 4 个输出管脚对象
coil_pins = [
    DigitalOutputDevice(PIN_IN1, initial_value=False),
    DigitalOutputDevice(PIN_IN2, initial_value=False),
    DigitalOutputDevice(PIN_IN3, initial_value=False),
    DigitalOutputDevice(PIN_IN4, initial_value=False),
]

# 简单 4 步全步进序列（适合 L298N 驱动 2 相步进）
# 顺时针方向（如果方向相反，改 direction = -1 或对调电机线）
SEQ_FULLSTEP = [
    (1, 0, 1, 0),  # A+ B+
    (0, 1, 1, 0),  # A- B+
    (0, 1, 0, 1),  # A- B-
    (1, 0, 0, 1),  # A+ B-
]

_seq_index = 0  # 当前在序列中的位置


def step_once(direction=1, delay=0.003):
    """
    direction = 1  顺时针
    direction = -1 逆时针
    delay: 每一步之间的延时，越小转得越快，先用 3ms 比较安全
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
    """转动指定步数"""
    for _ in range(steps):
        step_once(direction=direction, delay=delay)


if __name__ == "__main__":
    steps_per_rev = 200   # 大部分 NEMA17 是 1.8°/步 → 200 步 = 一圈
    delay = 0.003         # 可以慢慢再调小（变快）

    try:
        print("➡ 顺时针转 1 圈...")
        turn_steps(steps_per_rev, direction=1, delay=delay)

        time.sleep(0.5)

        print("⬅ 逆时针转 1/2 圈...")
        turn_steps(steps_per_rev // 2, direction=-1, delay=delay)

        print("停止保持 2 秒，然后释放线圈...")
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")

    finally:
        release_motor()
        print("电机已释放，程序结束。")
