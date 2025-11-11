# utils_hw.py
import time

# 使用 gpiozero，在 Raspberry Pi 5 + python3-lgpio 环境下工作良好
try:
    from gpiozero import DigitalInputDevice, Buzzer as GZBuzzer
    _HAS_GZ = True
except Exception:
    _HAS_GZ = False


class Trigger:
    """
    光电/接近传感器触发（低电平/高电平均可配置）。
    无硬件时退化为延时等待。
    """
    def __init__(self, pin: int = None, active_high: bool = True, debounce_ms: int = 60):
        self.pin = pin
        self.active_high = active_high
        self.debounce_ms = debounce_ms
        self.device = None

        if _HAS_GZ and pin is not None:
            # active_high=True → 使用下拉电阻（pull_up=False）
            # active_high=False → 使用上拉电阻（pull_up=True）
            pull_up = not active_high
            try:
                self.device = DigitalInputDevice(pin, pull_up=pull_up)
            except Exception:
                self.device = None

    def wait(self, fallback_seconds: float = 0.0):
        """
        阻塞等待一次“稳定触发”。
        - 如果没有硬件，就简单 sleep fallback_seconds。
        """
        if self.device is None:
            if fallback_seconds > 0:
                time.sleep(fallback_seconds)
            return True

        last = False
        stable_t = 0.0
        while True:
            v = bool(self.device.value)
            active = v if self.active_high else (not v)
            t = time.time()
            if active:
                if not last:
                    stable_t = t
                elif (t - stable_t) * 1000.0 >= self.debounce_ms:
                    return True
            last = active
            time.sleep(0.005)


class Buzzer:
    """
    简单蜂鸣器输出。
    - pin: BCM 引脚号（例如 21）
    - active_high: True 表示输出高电平时蜂鸣器响
    """
    def __init__(self, pin: int = None, active_high: bool = True):
        self.pin = pin
        self.active_high = active_high
        self.dev = None

        if _HAS_GZ and pin is not None:
            try:
                # gpiozero.Buzzer 默认 active_high=True，我们直接传进去
                self.dev = GZBuzzer(pin, active_high=active_high)
                self.dev.off()   # 确保上电时不响
            except Exception:
                self.dev = None

    def beep(self, ms: int = 120):
        """
        蜂鸣器响 ms 毫秒。
        如果没有硬件，则只 sleep，保证主流程不崩。
        """
        if self.dev is None:
            time.sleep(ms / 1000.0)
            return

        self.dev.on()
        time.sleep(ms / 1000.0)
        self.dev.off()

    def close(self):
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:
                pass
