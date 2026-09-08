import time

from arlo import Arlo

def measure_pwm(arlo, pwm, direction=1):
    input(f"Position robot at start line. Press Enter to run PWM={pwm}...")
    arlo.robot.go_diff(pwm, pwm, direction, direction)
    start = time.monotonic()
    input("Press Enter the instant the robot crosses the finish line...")
    elapsed = time.monotonic() - start
    arlo.robot.stop()
    print(f"PWM={pwm}: {elapsed:.2f}s -> {100/elapsed:.2f} cm/s")
    return 100 / elapsed

measure_pwm(arlo, 40, 1)
