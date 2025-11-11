# test_buzzer.py
from utils_hw import Buzzer
import time

bz = Buzzer(pin=21, active_high=True)

print("Beep 1")
bz.beep(500)    # 0.5 秒
time.sleep(1.0)

print("Beep 2 (长一点)")
bz.beep(1000)   # 1 秒

print("Done")
