import time
import threading
import math

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
    PWM_MIN = 30
    PWM_MAX = 127
    SPEED_AT_PWM_MIN_CMS = 17    # cm/s the robot actually moves at PWM 30
    SPEED_AT_PWM_MAX_CMS = 79   # cm/s the robot actually moves at PWM 127

    # measured via calibration, all 4 points fit with linear regression
    SPEED_SLOPE = 0.6355      # cm/s per PWM unit
    SPEED_INTERCEPT = -1.99   # cm/s 

    ACCELERATION_CMS2 = 30.0      # cm/s^2, tune to taste
    CONTROL_DT_S = 0.05           # seconds between speed updates

    TRACK_WIDTH_CM = 39   # distance between wheel centers

    def __init__(self, port='/dev/ttyACM0'):
        print("Initializing arlo...")
        self.robot = Robot(port)
        self._current_handle: Optional[DriveHandle] = None
        print("Finished initializing arlo")

    def _speed_to_pwm(self, speed_cms):
        """Map a speed in cm/s to a PWM value in [PWM_MIN, PWM_MAX]."""
        speed_cms = max(self.SPEED_AT_PWM_MIN_CMS, min(speed_cms, self.SPEED_AT_PWM_MAX_CMS))
        pwm = (speed_cms - self.SPEED_INTERCEPT) / self.SPEED_SLOPE
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
                print(f"  -> pwm={pwm}")
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

    def _wheel_cms_to_angspeed_degs(self, wheel_speed_cms):
        """Per-wheel linear speed (cm/s) -> robot angular speed (deg/s),
        assuming both wheels spin at equal speed in opposite directions."""
        return math.degrees(2 * wheel_speed_cms / self.TRACK_WIDTH_CM)

    def _angspeed_degs_to_wheel_cms(self, angspeed_degs):
        """Robot angular speed (deg/s) -> required per-wheel linear speed (cm/s)."""
        return math.radians(angspeed_degs) * self.TRACK_WIDTH_CM / 2

    def turn(self, degrees: float, speed_degs: float = 60, debug: bool = True) -> DriveHandle:
        """
        Turn in place around the robot's own axis, with the same
        accel/cruise/decel ramp as drive().

        degrees: how far to turn. Positive = counter-clockwise (left),
            negative = clockwise (right).
        speed_degs: cruising angular speed in degrees/second (always positive).
        debug: if True (default), print each phase transition.

        Returns a DriveHandle — .halt() to stop early, .wait() to block
        until done. Calling turn()/drive() again auto-interrupts this one,
        same as drive().
        """
        if self._current_handle is not None:
            self._current_handle.halt(reason="interrupted by new turn command")

        halt_signal = HaltSignal()
        thread = threading.Thread(
            target=self._turn_worker,
            args=(degrees, speed_degs, debug, halt_signal),
        )
        handle = DriveHandle(thread, halt_signal)
        self._current_handle = handle
        thread.start()
        return handle

    def _turn_worker(self, degrees, speed_degs, debug, halt_signal):
        ccw = degrees >= 0
        total_degrees = abs(degrees)

        # Turning left (CCW): left wheel backward, right wheel forward.
        # If the robot turns the wrong way on real hardware, just swap these two.
        dir_right = Direction.BACKWARD if ccw else Direction.FORWARD
        dir_left  = Direction.FORWARD if ccw else Direction.BACKWARD

        v0_wheel = self.SPEED_AT_PWM_MIN_CMS
        a_wheel = self.ACCELERATION_CMS2

        v0_ang = self._wheel_cms_to_angspeed_degs(v0_wheel)
        a_ang = self._wheel_cms_to_angspeed_degs(a_wheel)  # conversion is linear/proportional, so this is valid
        target_ang = min(speed_degs, self._wheel_cms_to_angspeed_degs(self.SPEED_AT_PWM_MAX_CMS))

        current_ang = v0_ang
        turned = 0.0
        t_prev = time.monotonic()
        last_pwm = None

        def send_pwm(pwm):
            nonlocal last_pwm
            if pwm != last_pwm:
                print(f"  -> pwm={pwm}")
                self.robot.go_diff(pwm, pwm, dir_left, dir_right)
                last_pwm = pwm

        def halted():
            if halt_signal.is_set():
                if debug:
                    print(f"turn: {halt_signal.reason}")
                self.robot.stop()
                return True
            return False

        d_ramp_total = (target_ang ** 2 - v0_ang ** 2) / a_ang
        if d_ramp_total >= total_degrees:
            target_ang = (v0_ang ** 2 + a_ang * total_degrees) ** 0.5
            d_accel = d_decel = total_degrees / 2.0
        else:
            d_accel = d_decel = (target_ang ** 2 - v0_ang ** 2) / (2 * a_ang)

        # --- acceleration phase ---
        if debug:
            print("turn: acceleration")
        while current_ang < target_ang and turned < d_accel:
            if halted():
                return
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now

            current_ang = min(target_ang, current_ang + a_ang * dt)
            send_pwm(self._speed_to_pwm(self._angspeed_degs_to_wheel_cms(current_ang)))

            turned += current_ang * dt
            time.sleep(self.CONTROL_DT_S)

        # --- cruise phase ---
        if debug:
            print("turn: cruise phase")
        send_pwm(self._speed_to_pwm(self._angspeed_degs_to_wheel_cms(target_ang)))

        cruise_end = total_degrees - d_decel
        while turned < cruise_end:
            if halted():
                return
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now

            turned += target_ang * dt
            time.sleep(self.CONTROL_DT_S)

        # --- deceleration phase ---
        if debug:
            print("turn: deacceleration")
        while current_ang > v0_ang and turned < total_degrees:
            if halted():
                return
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now

            current_ang = max(v0_ang, current_ang - a_ang * dt)
            send_pwm(self._speed_to_pwm(self._angspeed_degs_to_wheel_cms(current_ang)))

            turned += current_ang * dt
            time.sleep(self.CONTROL_DT_S)

        if debug:
            print("turn: halt")
        self.robot.stop()