"""Thin layer over the Arlo motors, sonars and camera. Units: cm and degrees, left positive."""
from arlo import Arlo, Direction
from camera import Camera

TURN_SPEED_DEGS = 50


class Hardware:
    def __init__(self):
        self.arlo = Arlo()
        self.camera = Camera(0, robottype='arlo', useCaptureThread=False)

    def close(self):
        self.arlo.stop()
        self.camera.terminateCaptureThread()
        capture = self.camera.get_capture()
        capture.close() if hasattr(capture, 'close') else capture.release()

    def sonar(self, *names):
        """{name: cm} for 'front', 'left', 'right', 'back'. A failed reading counts as 0 (blocked)."""
        return {name: self._ping(name) for name in names or ('front', 'left', 'right', 'back')}

    def _ping(self, name):
        read = getattr(self.arlo.robot, f'read_{name}_ping_sensor')
        for _ in range(2):  # Retry once; single failed readings happen
            with self.arlo.serial_lock:  # The drive thread shares the serial port
                value = read()
            if value is not None and value > 0:
                return value / 10
        return 0.0

    def markers(self):
        """{marker id: (distance_cm, bearing_rad)}, measured from the camera to the marker."""
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
        self.arlo.turn(-degrees, speed_degs=TURN_SPEED_DEGS, debug=False).wait()  # Arlo turns right for positive

    def drive(self, distance_cm, speed_cms):
        """Start driving (negative is backwards) and return the handle: .traveled, .turned
        (degrees, left positive), .steer(deg/s), .limit_speed(cm/s), .halt(), .is_alive()."""
        direction = Direction.FORWARD if distance_cm > 0 else Direction.BACKWARD
        return self.arlo.drive(direction, speed_cms, distance_cm=abs(distance_cm), debug=False)
