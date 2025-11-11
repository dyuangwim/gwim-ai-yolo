# utils_hw.py
import time

# 我们优先使用 gpiozero，在所有 Pi 型号上都支持
try:
    from gpiozero import DigitalInputDevice, Buzzer as GZBuzzer
    _HAS_GZ = True
except Exception:
    _HAS_GZ = False


class Trigger:
    """
    光电/接近传感器触发（低电平/高电平均可配置）。
    - pin: BCM 引脚号
    - active_high: True 表示高电平为“有物体”
    """
    def __init__(self, pin: int = None, active_high: bool = True, debounce_ms: int = 60):
        self.pin = pin
        self.active_high = active_high
        self.debounce_ms = debounce_ms
        self.device = None

        if _HAS_GZ and pin is not None:
            # pull_up = 对应逻辑：active_high=True → 下拉；active_high=False → 上拉
            pull_up = not active_high
            try:
                self.device = DigitalInputDevice(pin, pull_up=pull_up)
            except Exception:
                self.device = None

    def wait(self, fallback_seconds: float = 0.0):
        """
        阻塞等待一次有效触发。
        如果没有硬件（device=None），则简单 sleep fallback_seconds。
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
    简单蜂鸣器输出（低/高电平有效）。
    - pin: BCM 引脚号
    - active_high: True = 输出高电平时蜂鸣器响
    """
    def __init__(self, pin: int = None, active_high: bool = True):
        self.pin = pin
        self.active_high = active_high
        self.dev = None

        if _HAS_GZ and pin is not None:
            try:
                self.dev = GZBuzzer(pin, active_high=active_high)
                # 确保默认是关闭状态
                self.dev.off()
            except Exception:
                self.dev = None

    def beep(self, ms: int = 120):
        """
        发声 ms 毫秒。如果没有硬件，就只 sleep。
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
