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
TARGET_DISTANCE_CM = 53  # Approximation; calibrate against 40 cm from nearest box face
DRIVE_STEP_CM = 40
TURN_STEP_DEG = 10
CLEAR_TURN_DEG = 65
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

def drive(arlo, particles, distance_cm, camera=None, target=None):
    """Drive while monitoring sonar and checking for the target landmark."""
    if distance_cm == 0:
        return 0.0

    direction = "front" if distance_cm > 0 else "back"
    readings = sonar(arlo)

    if (readings[direction] is None or
            readings[direction] <= obstacles.CLEARANCE_MM + abs(distance_cm) * 10):
        return 0.0

    if distance_cm > 0:
        if any(readings[s] is None or readings[s] <= obstacles.CLEARANCE_MM
               for s in ("left", "right")):
            return 0.0

    # Slow down if an obstacle is within 50 cm
    speed = DRIVE_SPEED
    if distance_cm > 0 and any(
        readings[s] is not None and readings[s] < SLOW_DISTANCE_MM
        for s in ("front", "left", "right")
    ):
        speed = SLOW_DRIVE_SPEED

    wheel_direction = Direction.FORWARD if distance_cm > 0 else Direction.BACKWARD
    handle = arlo.drive(
        wheel_direction, speed, distance_cm=abs(distance_cm), debug=False
    )

    halted = False
    reached_target = False
    last_camera_check = time.monotonic()
    last_log = time.monotonic()

    while handle.is_alive():
        readings = sonar(arlo)

        # Sonar safety
        blocked = (
            readings[direction] is None
            or readings[direction] <= obstacles.CLEARANCE_MM
        )

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

        # Print nearby obstacles occasionally
        now = time.monotonic()
        if now - last_log >= 0.3:
            if any(readings[s] is not None and readings[s] < 1000
                   for s in ("front", "left", "right")):
                print(
                    f"Sonar: F={readings['front']} "
                    f"L={readings['left']} R={readings['right']} mm"
                )
            last_log = now

        # Check camera every 0.5 seconds while driving forward
        if distance_cm > 0 and camera is not None and target is not None:
            if now - last_camera_check >= 0.5:
                last_camera_check = now

                ids, distances, angles = camera.detect_aruco_objects(
                    camera.get_next_frame()
                )

                if ids is not None:
                    for marker_id, marker_distance in zip(ids, distances):
                        if marker_id == target and marker_distance <= 40:
                            print(
                                f"Reached L{target} "
                                f"(camera: {marker_distance:.0f} cm)"
                            )
                            handle.halt("target reached")
                            halted = True
                            reached_target = True
                            break

                if reached_target:
                    break

        time.sleep(0.05)

    handle.wait()

    distance = handle.estimated_distance_cm if halted else abs(distance_cm)
    distance = math.copysign(distance, distance_cm)

    if particles is not None and distance:
        mcl.motion_update(particles, delta_trans_cm=distance)

    return distance

def avoid(arlo, particles):
    """Create space, turn away from obstacle, and continue."""
    readings = sonar(arlo)
    direction = 1 if (readings["left"] or 0) > (readings["right"] or 0) else -1

    print(
        f"Avoidance: left={readings['left']} mm, "
        f"right={readings['right']} mm "
        f"-> choosing {'LEFT' if direction == 1 else 'RIGHT'}"
    )

    for _ in range(8):
        readings = sonar(arlo)

        if obstacles.can_turn(**readings):
            if turn(arlo, particles, direction * CLEAR_TURN_DEG):
                readings = sonar(arlo)

                if obstacles.choose_action(**readings, distance_cm=20) == "forward":
                    if drive(arlo, particles, 40) >= 40:
                        return True
                continue

        # Cannot turn safely: create more room
        print("Cannot turn, backing up:", readings)

        if drive(arlo, particles, -10) >= 0:
            print("Backing blocked:", sonar(arlo))
            return False

    return False

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
            obs, particles, weights = update(camera, particles, weights)
            particles, weights = relocalize(arlo, camera, particles, weights)

            x, y, theta = mcl.pose_estimate(particles, weights)
            bearing, distance = navigation.direction_to_target(x, y, theta, *LANDMARKS[target])
            print(f"L{target}: distance={distance:.0f} cm, spread={mcl.position_spread_cm(particles, weights):.0f} cm")

            camera_distance = obs[target][0] if target in obs else float("inf")

            if distance <= TARGET_DISTANCE_CM or camera_distance <= 45:
                print(f"Reached L{target} (MCL={distance:.0f} cm, camera={camera_distance:.0f} cm)")
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
                if drive(arlo, particles, step, camera=camera, target=target) > 0:
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
