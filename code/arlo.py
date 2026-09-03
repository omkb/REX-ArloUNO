import time
import threading

from enum import IntEnum
from typing import Optional
from robot import Robot
class Direction(IntEnum):
    FORWARD = 1
    BACKWARD = 0


class HaltSignal:
    """Like threading.Event, but carries a reason string for why it was set."""

    def __init__(self):
        self._event = threading.Event()
        self.reason = None

    def set(self, reason: str):
        self.reason = reason
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


class DriveHandle:
    """Reference to a single drive() call, running in its own thread.

    Lets you check on or cancel that specific drive without affecting
    any other drive() calls.
    """

    def __init__(self, thread: threading.Thread, halt_signal: HaltSignal):
        self._thread = thread
        self._halt_signal = halt_signal

    def halt(self, reason: str = "stopped"):
        """Stop the robot immediately, skipping any deceleration ramp,
        and terminate the drive thread."""
        self._halt_signal.set(reason)
        self._thread.join()

    def is_alive(self):
        return self._thread.is_alive()

    def wait(self):
        """Block until this drive finishes on its own (or is halted)."""
        self._thread.join()


class Arlo(object):
    """High(er) level API for the Arlo Robot with error checking"""

    # --- calibration constants: measure these on the real robot ---
    PWM_MIN = 40
    PWM_MAX = 127
    SPEED_AT_PWM_MIN_CMS = 5.0    # cm/s the robot actually moves at PWM 40
    SPEED_AT_PWM_MAX_CMS = 60.0   # cm/s the robot actually moves at PWM 127

    ACCELERATION_CMS2 = 30.0      # cm/s^2, tune to taste
    CONTROL_DT_S = 0.05           # seconds between speed updates

    def __init__(self, port=None):
        self.robot = Robot(port)
        self._current_handle: Optional[DriveHandle] = None

    def _speed_to_pwm(self, speed_cms):
        """Map a speed in cm/s to a PWM value in [PWM_MIN, PWM_MAX]."""
        speed_cms = max(self.SPEED_AT_PWM_MIN_CMS, min(speed_cms, self.SPEED_AT_PWM_MAX_CMS))
        span_speed = self.SPEED_AT_PWM_MAX_CMS - self.SPEED_AT_PWM_MIN_CMS
        span_pwm = self.PWM_MAX - self.PWM_MIN
        pwm = self.PWM_MIN + (speed_cms - self.SPEED_AT_PWM_MIN_CMS) * span_pwm / span_speed
        return int(round(pwm))

    def drive(self, direction: Direction, speed_cms: float,
              distance_cm: Optional[float] = None,
              debug: bool = True) -> DriveHandle:
        """
        Start driving in a background thread with a smooth speed ramp,
        and return immediately so the caller can keep doing other work.

        Only one drive can be active at a time: calling drive() again
        automatically interrupts whatever drive is currently running
        (immediately, no deceleration) before starting the new one.

        direction: Direction.FORWARD or Direction.BACKWARD
        speed_cms: target cruising speed in cm/s
        distance_cm: distance to travel in cm. If omitted (None), drives
            indefinitely at speed_cms until halted via .halt() or
            arlo.stop() — there's no target to decelerate toward, so it
            accelerates once and then cruises until interrupted.
        debug: if True (default), print each phase transition

        Returns a DriveHandle — call .halt() on it to stop this specific
        drive immediately, or .wait() to block until it finishes on its
        own (only returns on its own if distance_cm was given).
        You can also call arlo.stop() to halt whatever is currently driving.
        """
        # terminate any prior drive before starting a new one
        if self._current_handle is not None:
            self._current_handle.halt(reason="interrupted by new drive command")

        halt_signal = HaltSignal()
        thread = threading.Thread(
            target=self._drive_worker,
            args=(direction, speed_cms, distance_cm, debug, halt_signal),
            daemon=True,
        )
        handle = DriveHandle(thread, halt_signal)
        self._current_handle = handle
        thread.start()
        return handle

    def stop(self):
        """Immediately halt whatever drive is currently running (no
        deceleration ramp). Safe to call even if nothing is driving."""
        if self._current_handle is not None:
            self._current_handle.halt(reason="stopped")

    def _drive_worker(self, direction, speed_cms, distance_cm, debug, halt_signal):
        v0 = self.SPEED_AT_PWM_MIN_CMS
        a = self.ACCELERATION_CMS2
        target_speed = min(speed_cms, self.SPEED_AT_PWM_MAX_CMS)

        current_speed = v0
        traveled = 0.0
        t_prev = time.monotonic()
        last_pwm = None

        def send_pwm(pwm):
            nonlocal last_pwm
            if pwm != last_pwm:
                self.robot.go_diff(pwm, pwm, direction, direction)
                last_pwm = pwm

        def halted():
            if halt_signal.is_set():
                if debug:
                    print(f"drive: {halt_signal.reason}")
                self.robot.stop()
                return True
            return False

        indefinite = distance_cm is None

        if indefinite:
            d_accel = float("inf")  # accelerate freely, no distance budget
        else:
            d_ramp_total = (target_speed ** 2 - v0 ** 2) / a
            if d_ramp_total >= distance_cm:
                target_speed = (v0 ** 2 + a * distance_cm) ** 0.5
                d_accel = d_decel = distance_cm / 2.0
            else:
                d_accel = d_decel = (target_speed ** 2 - v0 ** 2) / (2 * a)

        # --- acceleration phase ---
        if debug:
            print("drive: acceleration")
        while current_speed < target_speed and traveled < d_accel:
            if halted():
                return
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now

            current_speed = min(target_speed, current_speed + a * dt)
            send_pwm(self._speed_to_pwm(current_speed))

            traveled += current_speed * dt
            time.sleep(self.CONTROL_DT_S)

        # --- cruise phase ---
        if debug:
            print("drive: cruise phase")
        send_pwm(self._speed_to_pwm(target_speed))

        if indefinite:
            # cruise forever — only .halt() / arlo.stop() ends this
            while True:
                if halted():
                    return
                time.sleep(self.CONTROL_DT_S)

        cruise_end = distance_cm - d_decel
        while traveled < cruise_end:
            if halted():
                return
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now

            traveled += target_speed * dt
            time.sleep(self.CONTROL_DT_S)

        # --- deceleration phase ---
        if debug:
            print("drive: deacceleration")
        while current_speed > v0 and traveled < distance_cm:
            if halted():
                return
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now

            current_speed = max(v0, current_speed - a * dt)
            send_pwm(self._speed_to_pwm(current_speed))

            traveled += current_speed * dt
            time.sleep(self.CONTROL_DT_S)

        if debug:
            print("drive: halt")
        self.robot.stop()