# utils_hw.py
import time

try:
    import RPi.GPIO as GPIO
    _HAS_GPIO = True
except Exception:
    _HAS_GPIO = False

class Trigger:
    """光电/接近传感器触发（低电平/高电平均可配置）。无硬件也可退化为延时。"""
    def __init__(self, pin:int=None, active_high:bool=True, debounce_ms:int=60):
        self.pin = pin
        self.active_high = active_high
        self.debounce_ms = debounce_ms
        if _HAS_GPIO and pin is not None:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN if active_high else GPIO.PUD_UP)

    def wait(self, fallback_seconds:float=0.0):
        if not _HAS_GPIO or self.pin is None:
            if fallback_seconds>0:
                time.sleep(fallback_seconds)
            return True
        last = False; stable_t = 0.0
        while True:
            v = GPIO.input(self.pin)
            active = bool(v) if self.active_high else (not bool(v))
            t = time.time()
            if active:
                if not last:
                    stable_t = t
                elif (t - stable_t)*1000 >= self.debounce_ms:
                    return True
            last = active
            time.sleep(0.005)

class Buzzer:
    """简单蜂鸣器输出（低/高电平有效）。"""
    def __init__(self, pin:int=None, active_high:bool=True):
        self.pin = pin; self.active_high = active_high
        if _HAS_GPIO and pin is not None:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, not active_high)

    def beep(self, ms:int=120):
        if not _HAS_GPIO or self.pin is None: 
            time.sleep(ms/1000.0); return
        GPIO.output(self.pin, self.active_high)
        time.sleep(ms/1000.0)
        GPIO.output(self.pin, not self.active_high)

    def close(self):
        if _HAS_GPIO and self.pin is not None:
            GPIO.cleanup(self.pin)
