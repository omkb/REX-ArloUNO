"""RobotRally: find landmarks, localize with MCL, and drive the route.

Uses the existing mcl.py, navigation.py, arlo.py, camera.py.
The obstacle decisions live in obstacles.py (we will simplify that separately).
"""
import math
import time

import mcl
import navigation
import obstacles
from arlo import Arlo, Direction
from camera import Camera

# Marker IDs and coordinates in centimetres (check against the real setup).
LANDMARKS = {2: (0, 0), 6: (0, 300), 7: (400, 0), 5: (400, 300)}
ROUTE = [2, 6, 7, 5, 2]

MAX_SPREAD_CM = 20
TARGET_DISTANCE_CM = 50  # Approximation; calibrate against 40 cm from nearest box face
DRIVE_STEP_CM = 40
TURN_STEP_DEG = 10
CLEAR_TURN_DEG = 25
DRIVE_SPEED = 30
TURN_SPEED = 60
SLOW_DRIVE_SPEED = 10  # Speed near obstacles (cm/s)
SLOW_DISTANCE_MM = 500

def see(camera):
    """Return the closest detection of each known marker: {id: (cm, radians)}."""
    ids, distances, angles = camera.detect_aruco_objects(camera.get_next_frame())
    found = {}
    if ids is not None:
        for marker_id, distance, angle in zip(ids, distances, angles):
            marker_id = int(marker_id)
            if marker_id in LANDMARKS and (marker_id not in found or distance < found[marker_id][0]):
                found[marker_id] = (float(distance), float(angle))
    return found


def update(camera, particles, weights):
    obs = see(camera)

    if obs and mcl.measurement_update(particles, weights, obs, LANDMARKS):
        ess = 1 / (weights @ weights)
        print(f"MCL: landmarks={list(obs)}, ESS={ess:.0f}")

        particles, weights = mcl.resample(particles, weights)

    return obs, particles, weights

def sonar(arlo):
    with arlo.io_lock:
        return {
            name: getattr(arlo.robot, f"read_{name}_ping_sensor")()
            for name in ("front", "left", "right", "back")
        }


def turn(arlo, particles, degrees):
    """Positive angles are left/CCW in the MCL coordinate system."""
    if not obstacles.can_turn(**sonar(arlo)):
        return False
    arlo.turn(-degrees, speed_degs=TURN_SPEED, debug=False).wait()
    if particles is not None:
        mcl.motion_update(particles, delta_rot_rad=math.radians(degrees))
    return True


def drive(arlo, particles, distance_cm):
    """Drive a short step, interrupting if sonar becomes unsafe."""
    direction = "front" if distance_cm > 0 else "back"

    readings = sonar(arlo)
    sensor_value = readings[direction]

    if sensor_value is None or sensor_value <= obstacles.CLEARANCE_MM + abs(distance_cm) * 10:
        return 0.0

    # Also check diagonal sensors when driving forward
    if distance_cm > 0:
        if any(readings[s] is None or readings[s] <= obstacles.CLEARANCE_MM
               for s in ("left", "right")):
            return 0.0

    wheel_direction = Direction.FORWARD if distance_cm > 0 else Direction.BACKWARD
    speed = DRIVE_SPEED

    if distance_cm > 0:
        if any(
            readings[s] is not None and readings[s] < SLOW_DISTANCE_MM
            for s in ("front", "left", "right")
        ):
            speed = SLOW_DRIVE_SPEED

    handle = arlo.drive(wheel_direction, speed, distance_cm=abs(distance_cm), debug=False)

    halted = False
    while handle.is_alive():
        readings = sonar(arlo)

        blocked = readings[direction] is None or readings[direction] <= obstacles.CLEARANCE_MM

        if distance_cm > 0:
            blocked |= any(
                readings[s] is None or readings[s] <= obstacles.CLEARANCE_MM
                for s in ("left", "right")
            )

        if blocked:
            print(f"Obstacle detected: {readings}")
            handle.halt("obstacle")
            halted = True
            break

        time.sleep(0.05)

    handle.wait()

    distance = handle.estimated_distance_cm if halted else abs(distance_cm)
    distance = math.copysign(distance, distance_cm)

    if particles is not None and distance:
        mcl.motion_update(particles, delta_trans_cm=distance)

    return distance

def avoid(arlo, particles):
    """Back up if necessary, turn away, then drive around the obstacle."""

    readings = sonar(arlo)
    direction = 1 if readings["left"] > readings["right"] else -1

    # Back up until there is room to turn
    while not obstacles.can_turn(**sonar(arlo)):
        print("Too close to turn, backing up")

        if drive(arlo, particles, -10) >= 0:
            print("Backing failed:", sonar(arlo))
            return False

    # Turn away from the obstacle
    while True:
        if not turn(arlo, particles, direction * CLEAR_TURN_DEG):
            print("Turning failed:", sonar(arlo))
            return False

        # Only finish avoidance after a successful forward movement
        readings = sonar(arlo)
        if obstacles.choose_action(**readings, distance_cm=20) == "forward":
            if drive(arlo, particles, 20) >= 20:
                return True

def initialize(arlo, camera):
    """Start from L1, find a second landmark, then require a small spread."""
    start = ROUTE[0]
    while True:
        particles = weights = None
        seen_other = False

        for _ in range(360 // TURN_STEP_DEG):
            if particles is None:
                obs = see(camera)
                if start in obs:
                    particles, weights = mcl.initialize_from_landmark(LANDMARKS, start, *obs[start])
                    others = {i: value for i, value in obs.items() if i != start}
                    if others:
                        mcl.measurement_update(particles, weights, others, LANDMARKS)
                        particles, weights = mcl.resample(particles, weights)
                        seen_other = True
                    print(f"Found starting landmark L{start}")
            else:
                obs, particles, weights = update(camera, particles, weights)
                seen_other |= any(i != start for i in obs)

            if particles is not None and seen_other:
                spread = mcl.position_spread_cm(particles, weights)
                if spread <= MAX_SPREAD_CM:
                    print(f"Localized (spread {spread:.1f} cm)")
                    return particles, weights

            if not turn(arlo, particles, TURN_STEP_DEG):
                break
            time.sleep(0.2)

        # One full scan without success: move to get a new view, then reset MCL.
        print("Scanning insufficient; changing position")
        if drive(arlo, particles, DRIVE_STEP_CM) == 0 and drive(arlo, particles, -10) == 0:
            raise RuntimeError("Cannot reposition for initialization")


def relocalize(arlo, camera, particles, weights):
    """Turn and look for landmarks whenever MCL spread exceeds 15 cm."""
    while mcl.position_spread_cm(particles, weights) > MAX_SPREAD_CM:
        print("Localization uncertain; scanning")
        for _ in range(360 // TURN_STEP_DEG):
            if not turn(arlo, particles, TURN_STEP_DEG):
                if not avoid(arlo, particles):
                    raise RuntimeError("Cannot turn to relocalize")
                break
            time.sleep(0.25)
            obs, particles, weights = update(camera, particles, weights)
            spread = mcl.position_spread_cm(particles, weights)

            print(f"Scan: landmarks={list(obs.keys())}, spread={spread:.1f} cm")

            if spread <= MAX_SPREAD_CM:
                print("Localization recovered!")
                return particles, weights
        else:
            if not avoid(arlo, particles):
                raise RuntimeError("Cannot change viewpoint for localization")
    return particles, weights

def follow_route(arlo, camera, particles, weights):
    for target in ROUTE:
        print(f"Next target: L{target}")
        while True:
            _, particles, weights = update(camera, particles, weights)
            particles, weights = relocalize(arlo, camera, particles, weights)

            x, y, theta = mcl.pose_estimate(particles, weights)
            bearing, distance = navigation.direction_to_target(x, y, theta, *LANDMARKS[target])
            print(f"L{target}: distance={distance:.0f} cm, spread={mcl.position_spread_cm(particles, weights):.0f} cm")

            if distance <= TARGET_DISTANCE_CM:
                print(f"Reached L{target}")
                break

            if abs(math.degrees(bearing)) > 12:
                angle = max(-30, min(30, math.degrees(bearing)))
                if not turn(arlo, particles, angle):
                    if not avoid(arlo, particles):
                        raise RuntimeError("Cannot maneuver around obstacle")
                continue

            if distance > 150:
                max_step = 100
            else:
                max_step = 30

            step = min(max_step, distance - TARGET_DISTANCE_CM)
            if obstacles.choose_action(**sonar(arlo), distance_cm=step) == "forward":
                if drive(arlo, particles, step) > 0:
                    continue
            if not avoid(arlo, particles):
                raise RuntimeError("Cannot maneuver around obstacle")


def main():
    input("Clear the test area, then press Enter to start... ")
    arlo = Arlo()
    camera = Camera(0, robottype="arlo", useCaptureThread=False)
    try:
        particles, weights = initialize(arlo, camera)
        follow_route(arlo, camera, particles, weights)
        print("Route complete")
    finally:
        arlo.stop()
        camera.terminateCaptureThread()
        capture = camera.get_capture()
        capture.release() if hasattr(capture, "release") else capture.close()


if __name__ == "__main__":
    main()
