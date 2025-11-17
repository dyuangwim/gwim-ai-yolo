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
            pull_up = not active_high
            try:
                self.device = DigitalInputDevice(pin, pull_up=pull_up)
            except Exception:
                self.device = None

    def wait(self, fallback_seconds: float = 0.0):
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
                self.dev = GZBuzzer(pin, active_high=active_high)
                self.off()   # 上电确保不响
            except Exception:
                self.dev = None

    # 显式 on/off，让外部可“持续响/立即静音”
    def on(self):
        if self.dev is not None:
            try: self.dev.on()
            except Exception: pass

    def off(self):
        if self.dev is not None:
            try: self.dev.off()
            except Exception: pass

    def beep(self, ms: int = 120):
        if self.dev is None:
            time.sleep(ms / 1000.0)
            return
        self.on()
        time.sleep(ms / 1000.0)
        self.off()

    def close(self):
        # close 前强制拉低，避免留下“响”的电平
        try:
            self.off()
        except Exception:
            pass
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:
                pass
            finally:
                self.dev = None
