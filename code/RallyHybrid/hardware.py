"""Thin layer over the Arlo motors, sonars and camera. Units: cm and degrees, left positive.

The sonar safety check lives inside drive(), so no caller can drive without it.
"""
import math

import avoid
from arlo import Arlo, Direction
from camera import Camera

DRIVE_SPEED_CMS = 40
TURN_SPEED_DEGS = 60


class Hardware:
    def __init__(self):
        self.arlo = Arlo()
        self.camera = Camera(0, robottype='arlo', useCaptureThread=False)

    def close(self):
        self.arlo.stop()
        self.camera.terminateCaptureThread()
        capture = self.camera.get_capture()
        capture.close() if hasattr(capture, 'close') else capture.release()

    def sonar(self):
        """{'front', 'left', 'right', 'back': cm}. A failed reading counts as 0 (blocked)."""
        return {name: self._ping(name) for name in ('front', 'left', 'right', 'back')}

    def _ping(self, name):
        read = getattr(self.arlo.robot, f'read_{name}_ping_sensor')
        for _ in range(2):  # Retry once; single failed readings happen
            with self.arlo.io_lock:  # The drive thread shares the serial port
                value = read()
            if value is not None and value > 0:
                return value / 10
        return 0.0

    def markers(self):
        """{marker id: (distance_cm, bearing_rad)} for every ArUco marker in view."""
        ids, distances, angles = self.camera.detect_aruco_objects(self.camera.get_next_frame())
        seen = {}
        if ids is not None:
            for marker_id, distance, angle in zip(ids, distances, angles):
                marker_id = int(marker_id)
                if marker_id not in seen or distance < seen[marker_id][0]:
                    seen[marker_id] = (float(distance), float(angle))
        return seen

    def turn(self, degrees):
        """Turn on the spot; positive is left. Arlo is round, so this cannot hit anything."""
        # Arlo's turn() goes right for positive degrees
        self.arlo.turn(-degrees, speed_degs=TURN_SPEED_DEGS, debug=False).wait()

    def path_clear(self, forward):
        sonar = self.sonar()
        return avoid.clear_ahead(sonar) if forward else sonar['back'] > avoid.FRONT_STOP_CM

    def drive(self, distance_cm, watch=None):
        """Drive straight (negative is backwards) and return (cm driven, why it stopped early).

        The sonars in the driving direction stop the drive at any moment. watch() is
        called along the way and may stop it too by returning a reason. The reason is
        None when the full distance was driven.
        """
        forward = distance_cm > 0
        if not self.path_clear(forward):
            return 0.0, 'sonar'
        handle = self.arlo.drive(Direction.FORWARD if forward else Direction.BACKWARD,
                                 DRIVE_SPEED_CMS, distance_cm=abs(distance_cm), debug=False)
        reason = None
        while handle.is_alive() and reason is None:
            if not self.path_clear(forward):
                reason = 'sonar'
            elif watch is not None:
                reason = watch()
        if reason is not None:
            handle.halt(reason)
        handle.wait()
        driven = handle.estimated_distance_cm if reason else abs(distance_cm)
        return math.copysign(driven, distance_cm), reason
