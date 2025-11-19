"""
This Raspberry Pi code was developed by newbiely.com
This Raspberry Pi code is made available for public use without any restriction
For comprehensive instructions and wiring diagrams, please visit:
https://newbiely.com/tutorials/raspberry-pi/raspberry-pi-stepper-motor
"""


import RPi.GPIO as GPIO
import time

# Define GPIO pins for L298N driver
IN1 = 12
IN2 = 16
IN3 = 20
IN4 = 21

# Set GPIO mode and configure pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# Constants for stepper motor control
DEG_PER_STEP = 1.8
STEP_PER_REVOLUTION = int(360 / DEG_PER_STEP)

# Function to move the stepper motor one step forward
def step_forward(delay, steps):
    for _ in range(steps):
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)
        time.sleep(delay)
        
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        time.sleep(delay)

# Function to move the stepper motor one step backward
def step_backward(delay, steps):
    for _ in range(steps):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.HIGH)
        time.sleep(delay)
        
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        time.sleep(delay)

try:
    # Set the delay between steps
    delay = 0.001
    while True:
        # Move the stepper motor one revolution in a clockwise direction
        step_forward(delay, STEP_PER_REVOLUTION)

        # Pause for 5 seconds
        time.sleep(5)

        # Move the stepper motor one revolution in an anticlockwise direction
        step_backward(delay, STEP_PER_REVOLUTION)

        # Halt for 5 seconds
        time.sleep(5)

except KeyboardInterrupt:
    print("\nExiting the script.")

finally:
    # Clean up GPIO settings
    GPIO.cleanup()
