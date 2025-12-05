# utils_hw.py
import time

# 使用 gpiozero，在 Raspberry Pi 5 + python3-lgpio 环境下工作良好
try:
    from gpiozero import (
        DigitalInputDevice,
        Buzzer as GZBuzzer,
        DigitalOutputDevice,
    )
    _HAS_GZ = True
except Exception:
    _HAS_GZ = False


class Trigger:
    """
    光电/接近传感器触发（低电平/高电平均可配置）。无硬件时退化为延时等待。
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

        def __repr__(self) -> str:
            return f"<Trigger pin={self.pin} active_high={self.active_high} debounce_ms={self.debounce_ms}>"

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
                self.off()   # 确保上电时不响
            except Exception:
                self.dev = None

    def on(self):
        if self.dev is not None:
            try:
                self.dev.on()
            except Exception:
                pass

    def off(self):
        if self.dev is not None:
            try:
                self.dev.off()
            except Exception:
                pass

    def beep(self, ms: int = 120):
        if self.dev is None:
            time.sleep(ms / 1000.0)
            return
        self.on()
        time.sleep(ms / 1000.0)
        self.off()

    def close(self):
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


class StepperDM556:
    """
    DM556 + NEMA23 步进电机控制封装（基于 gpiozero.DigitalOutputDevice）。
    默认：
      - STEP  → BCM5  (接 DM556 PUL-)
      - DIR   → BCM6  (接 DM556 DIR-)
      - PUL+ / DIR+ 接 3.3V
      - MICROSTEP = 8（和你测试脚本一致）

    用法示例：
        m = StepperDM556()
        m.move_degrees(90, clockwise=True, speed_rps=0.3)
        m.close()
    """

    def __init__(
        self,
        step_pin: int = 5,
        dir_pin: int = 6,
        microstep: int = 8,
    ):
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.microstep = microstep
        self.base_steps_per_rev = 200  # 1.8° → 200 steps / rev
        self.steps_per_rev = self.base_steps_per_rev * self.microstep

        self.step = None
        self.direction = None

        if not _HAS_GZ:
            print("[Stepper] gpiozero not available, stepper disabled.", flush=True)
            return

        try:
            self.step = DigitalOutputDevice(self.step_pin, initial_value=False)
            self.direction = DigitalOutputDevice(self.dir_pin, initial_value=False)
            print(
                f"[Stepper] DM556 initialized (STEP={self.step_pin}, DIR={self.dir_pin}, MICROSTEP={self.microstep})",
                flush=True,
            )
        except Exception as e:
            print(f"[Stepper] Failed to init stepper on GPIO {self.step_pin}/{self.dir_pin}: {e}", flush=True)
            self.step = None
            self.direction = None

    def _pulse_step(self, delay: float):
        """发送一个 step 脉冲"""
        if self.step is None:
            return
        self.step.on()
        time.sleep(delay)
        self.step.off()
        time.sleep(delay)

    def move_steps(self, steps: int, clockwise: bool = True, speed_rps: float = 0.2):
        """
        steps: 走多少步（与 steps_per_rev 同单位）
        clockwise: True=顺时针，False=逆时针
        speed_rps: 每秒多少圈（0.2 = 慢速度，适合测试）
        """
        if self.step is None or self.direction is None:
            print("[Stepper] No hardware, skip move_steps.", flush=True)
            return
        if steps <= 0:
            return
        if speed_rps <= 0:
            speed_rps = 0.2

        # 设置方向
        try:
            self.direction.value = 1 if clockwise else 0
        except Exception as e:
            print(f"[Stepper] Failed to set direction: {e}", flush=True)
            return

        # 计算每个脉冲的延迟
        steps_per_sec = self.steps_per_rev * speed_rps
        delay = 1.0 / steps_per_sec / 2.0  # 一个完整方波周期 = 2 * delay

        for _ in range(int(steps)):
            self._pulse_step(delay)

    def move_degrees(self, degrees: float, clockwise: bool = True, speed_rps: float = 0.2):
        """按角度转动（例如 90°, 45°）"""
        if degrees == 0:
            return
        ratio = float(degrees) / 360.0
        steps = int(self.steps_per_rev * ratio)
        self.move_steps(steps, clockwise=clockwise, speed_rps=speed_rps)

    def close(self):
        """释放 GPIO 资源"""
        try:
            if self.step is not None:
                try:
                    self.step.off()
                except Exception:
                    pass
                self.step.close()
        except Exception:
            pass
        finally:
            self.step = None

        try:
            if self.direction is not None:
                try:
                    self.direction.off()
                except Exception:
                    pass
                self.direction.close()
        except Exception:
            pass
        finally:
            self.direction = None

        print("[Stepper] Closed and GPIO released.", flush=True)
