# utils_hw.py
import time

# Using gpiozero, it works well in a Raspberry Pi 5 + python3-lgpio environment.
try:
    from gpiozero import (
        DigitalInputDevice,
        Buzzer as GZBuzzer,
        DigitalOutputDevice,
    )
    _HAS_GZ = True
except Exception:
    _HAS_GZ = False


# ========== Trigger (Keep your original logic)==========

class Trigger:
    """
    Photoelectric/proximity sensor trigger (configurable for both low and high levels). Degrades to a delay wait when no hardware is available.
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


# ========== Buzzer==========

class Buzzer:
    """
    Simple buzzer output.
    - pin: BCM pin number (e.g., 21)
    - active_high: True The buzzer sounds when the output is high.
    """

    def __init__(self, pin: int = None, active_high: bool = True):
        self.pin = pin
        self.active_high = active_high
        self.dev = None

        if _HAS_GZ and pin is not None:
            try:
                self.dev = GZBuzzer(pin, active_high=active_high)
                self.off()   # Make sure it doesn't make a sound when powered on.
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


# ========== DM556 + NEMA23 Stepper motor==========

_STEPPER_STEP_DEVICE = None
_STEPPER_DIR_DEVICE = None
_STEPPER_STEPS_PER_REV = None 
_STEPPER_MICROSTEP = None


def init_stepper_dm556(step_pin: int = 5, dir_pin: int = 6, microstep: int = 8):
    """
    Initialize the DM556 stepper motor control:

    - step_pin = BCM5  (connects to DM556 PUL-)
    - dir_pin  = BCM6  (connects to DM556 DIR-)
    - PUL+ / DIR+ connect to 3.3V
    - microstep = 8 must match the DM556 DIP switch settings
    """
    global _STEPPER_STEP_DEVICE, _STEPPER_DIR_DEVICE
    global _STEPPER_STEPS_PER_REV, _STEPPER_MICROSTEP

    if not _HAS_GZ:
        print("[Stepper] gpiozero not available, stepper disabled.", flush=True)
        _STEPPER_STEP_DEVICE = None
        _STEPPER_DIR_DEVICE = None
        _STEPPER_STEPS_PER_REV = None
        return False

    _STEPPER_MICROSTEP = int(microstep) if microstep and microstep > 0 else 8
    _STEPPER_STEPS_PER_REV = 200 * _STEPPER_MICROSTEP  # 1.8° → 200 steps / revolution

    try:
        _STEPPER_STEP_DEVICE = DigitalOutputDevice(step_pin, initial_value=False)
        _STEPPER_DIR_DEVICE = DigitalOutputDevice(dir_pin, initial_value=False)
        print(
            f"[Stepper] DM556 initialized (STEP=BCM{step_pin}, DIR=BCM{dir_pin}, MICROSTEP={_STEPPER_MICROSTEP})",
            flush=True,
        )
        return True
    except Exception as e:
        print(f"[Stepper] Failed to init stepper on GPIO {step_pin}/{dir_pin}: {e}", flush=True)
        _STEPPER_STEP_DEVICE = None
        _STEPPER_DIR_DEVICE = None
        _STEPPER_STEPS_PER_REV = None
        return False


def _stepper_pulse_step(delay: float):
    """Send a step pulse"""
    global _STEPPER_STEP_DEVICE
    if _STEPPER_STEP_DEVICE is None:
        return
    _STEPPER_STEP_DEVICE.on()
    time.sleep(delay)
    _STEPPER_STEP_DEVICE.off()
    time.sleep(delay)


def stepper_move_steps(steps: int, clockwise: bool = True, speed_rps: float = 0.2):
    """
    Move a specified number of steps:

    - steps: Number of steps to move (in the same units as STEPS_PER_REV)
    - clockwise: True=clockwise, False=counter-clockwise
    - speed_rps: Revolutions per second (0.2 = slow)
    """
    global _STEPPER_STEP_DEVICE, _STEPPER_DIR_DEVICE, _STEPPER_STEPS_PER_REV

    if (
        _STEPPER_STEP_DEVICE is None
        or _STEPPER_DIR_DEVICE is None
        or _STEPPER_STEPS_PER_REV is None
    ):
        print("[Stepper] No hardware, skip stepper_move_steps.", flush=True)
        return

    if steps <= 0:
        return
    if speed_rps <= 0:
        speed_rps = 0.2

    # Set direction
    try:
        _STEPPER_DIR_DEVICE.value = 1 if clockwise else 0
    except Exception as e:
        print(f"[Stepper] Failed to set direction: {e}", flush=True)
        return

    # Calculate delay
    steps_per_sec = _STEPPER_STEPS_PER_REV * speed_rps
    delay = 1.0 / steps_per_sec / 2.0

    for _ in range(int(steps)):
        _stepper_pulse_step(delay)


def stepper_move_degrees(degrees: float, clockwise: bool = True, speed_rps: float = 0.2):
    """
    Move by a specified number of degrees:

    - degrees: for example, 90, 45, 180
    """
    global _STEPPER_STEPS_PER_REV

    if _STEPPER_STEPS_PER_REV is None:
        print("[Stepper] Not initialized, skip stepper_move_degrees.", flush=True)
        return

    if degrees == 0:
        return

    ratio = abs(degrees) / 360.0
    steps = int(_STEPPER_STEPS_PER_REV * ratio)
    if steps <= 0:
        return

    # If the incoming angle is negative, then reverse it.
    cw = clockwise
    if degrees < 0:
        cw = not cw

    stepper_move_steps(steps, clockwise=cw, speed_rps=speed_rps)


def close_stepper_dm556():
    """Release stepper motor GPIO resources"""
    global _STEPPER_STEP_DEVICE, _STEPPER_DIR_DEVICE
    global _STEPPER_STEPS_PER_REV, _STEPPER_MICROSTEP

    try:
        if _STEPPER_STEP_DEVICE is not None:
            try:
                _STEPPER_STEP_DEVICE.off()
            except Exception:
                pass
            _STEPPER_STEP_DEVICE.close()
    except Exception:
        pass
    finally:
        _STEPPER_STEP_DEVICE = None

    try:
        if _STEPPER_DIR_DEVICE is not None:
            try:
                _STEPPER_DIR_DEVICE.off()
            except Exception:
                pass
            _STEPPER_DIR_DEVICE.close()
    except Exception:
        pass
    finally:
        _STEPPER_DIR_DEVICE = None

    _STEPPER_STEPS_PER_REV = None
    _STEPPER_MICROSTEP = None

    print("[Stepper] Closed and GPIO released.", flush=True)
