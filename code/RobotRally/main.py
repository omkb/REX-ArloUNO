"""RobotRally: sequential landmark MCL and reactive obstacle avoidance.

Run only after checking the landmark coordinates, turn sign and sonar directions.
All driving commands are open-loop estimates; test at low speed on a clear floor.
"""
import argparse
import math
import time

import mcl
import navigation
import obstacles
from arlo import Arlo, Direction
from camera import Camera

# Coordinates in cm. These describe the schematic square; correct them to
# the actual *observed ArUco marker centres* before relying on localization.
# Marker IDs in route order; the first marker is also the starting landmark.
LANDMARK_IDS = [7, 3, 6, 11]
LANDMARKS = dict(zip(LANDMARK_IDS, [
    (0.0, 0.0), (0.0, 300.0), (400.0, 0.0), (400.0, 300.0)
]))
START_LANDMARK = LANDMARK_IDS[0]
ROUTE = LANDMARK_IDS + [START_LANDMARK]

SCAN_STEP_DEG = 10
SCAN_SETTLE_S = 0.25
MAX_POSITION_SPREAD_CM = 55.0  # Experimental threshold, tune on the real robot
DRIVE_STEP_CM = 15.0
DRIVE_SPEED_CMS = 20.0
TURN_SPEED_DEG_S = 60.0
TURN_TOLERANCE_DEG = 12.0
TARGET_DISTANCE_CM = 45.0    # Reached when the estimated position is within this distance


def observations(camera):
    """Read one image and return known markers as {id: (cm, bearing_rad)}."""
    ids, distances, angles = camera.detect_aruco_objects(camera.get_next_frame())
    if ids is None:
        return {}
    obs = {}
    for marker_id, distance, angle in zip(ids, distances, angles):
        marker_id = int(marker_id)
        if marker_id in LANDMARKS:
            if marker_id not in obs or distance < obs[marker_id][0]:
                obs[marker_id] = (float(distance), float(angle))
    return obs


def sonar(arlo, direction):
    """Returns a sonar reading in mm; None for invalid/unavailable values."""
    method = getattr(arlo.robot, f"read_{direction}_ping_sensor")
    value = method()
    return float(value) if value is not None and value > 0 else None


def turn(arlo, particles, degrees):
    """Turn physically, then update particles using commanded (not measured) angle."""
    if abs(degrees) < 0.01:
        return
    # Arlo's wheel commands turn right for positive degrees; MCL uses left.
    arlo.turn(-degrees, speed_degs=TURN_SPEED_DEG_S, debug=False).wait()
    if particles is not None:
        mcl.motion_update(particles, delta_rot_rad=math.radians(degrees))


def drive(arlo, particles, distance_cm):
    """Drive in a short segment. Never drive without a valid sonar measurement."""
    if distance_cm == 0:
        return False
    sensor = 'front' if distance_cm > 0 else 'back'
    reading = sonar(arlo, sensor)
    if reading is None or reading <= obstacles.CLEARANCE_MM + abs(distance_cm) * 10:
        return False
    direction = Direction.FORWARD if distance_cm > 0 else Direction.BACKWARD
    arlo.drive(direction, DRIVE_SPEED_CMS, distance_cm=abs(distance_cm), debug=False).wait()
    if particles is not None:
        mcl.motion_update(particles, delta_trans_cm=distance_cm)
    return True


def update_localization(camera, particles, weights):
    obs = observations(camera)
    if obs and mcl.measurement_update(particles, weights, obs, LANDMARKS):
        particles, weights = mcl.resample(particles, weights)
    return obs, particles, weights


def choose_action(arlo, distance_cm=DRIVE_STEP_CM, failed_action=None):
    return obstacles.choose_action(
        sonar(arlo, 'front'), sonar(arlo, 'left'),
        sonar(arlo, 'right'), sonar(arlo, 'back'),
        distance_cm=distance_cm, failed_action=failed_action
    )


def avoid_obstacle(arlo, particles, action):
    """Perform just one movement; the caller reads sonar again afterwards."""
    if action == 'backward':
        return drive(arlo, particles, -10)
    turn(arlo, particles, -25 if action == 'right' else 25)
    return True


def reposition(arlo, particles, failed_action):
    """Choose one movement. Report translation so initialization can reset MCL."""
    action = choose_action(arlo, failed_action=failed_action)
    if action == 'forward':
        success = drive(arlo, particles, DRIVE_STEP_CM)
    else:
        success = avoid_obstacle(arlo, particles, action)
    moved = success and action in ('forward', 'backward')
    return moved, None if success else action


def initialize(arlo, camera):
    """Initialize from the starting landmark and another; restart after repositioning."""
    particles = weights = None
    seen_second = False
    failed_action = None
    while True:
        for _ in range(360 // SCAN_STEP_DEG):
            if particles is None:
                obs = observations(camera)
                print(f"Detected landmarks: {list(obs.keys())}")
                if START_LANDMARK in obs:
                    particles, weights = mcl.initialize_from_landmark(
                        LANDMARKS, START_LANDMARK, *obs[START_LANDMARK]
                    )
                    # The starting landmark is already included in initialization.
                    obs = {i: value for i, value in obs.items() if i != START_LANDMARK}
                    if obs:
                        mcl.measurement_update(particles, weights, obs, LANDMARKS)
                        particles, weights = mcl.resample(particles, weights)
            else:
                obs, particles, weights = update_localization(camera, particles, weights)

            if particles is not None:
                seen_second = seen_second or any(i != START_LANDMARK for i in obs)
                spread = mcl.position_spread_cm(particles, weights)
                if seen_second and obs and spread <= MAX_POSITION_SPREAD_CM:
                    print(f"Localization ready; spread={spread:.1f} cm")
                    return particles, weights
            turn(arlo, particles, SCAN_STEP_DEG)
            time.sleep(SCAN_SETTLE_S)

        print("Repositioning to improve localization")
        moved, failed_action = reposition(arlo, particles, failed_action)
        if moved:
            particles = weights = None
            seen_second = False


def visit_target(arlo, camera, particles, weights, target):
    """Drive towards the target using MCL; the marker need not be visible."""
    failed_action = None
    detouring = False
    while True:
        _, particles, weights = update_localization(camera, particles, weights)
        x, y, theta = mcl.pose_estimate(particles, weights)
        target_bearing, target_distance = navigation.direction_to_target(
            x, y, theta, *LANDMARKS[target]
        )
        spread = mcl.position_spread_cm(particles, weights)
        print(f"L{target}: pose=({x:.0f}, {y:.0f}, {math.degrees(theta):.0f}°), "
              f"dist={target_distance:.0f} cm, spread={spread:.0f} cm")

        if target_distance <= TARGET_DISTANCE_CM and spread <= MAX_POSITION_SPREAD_CM:
            print(f"Reached L{target}")
            return particles, weights

        action = choose_action(arlo, DRIVE_STEP_CM, failed_action)
        if action != 'forward':
            detouring = True
            success = avoid_obstacle(arlo, particles, action)
            failed_action = None if success else action
            continue

        if not detouring and abs(math.degrees(target_bearing)) > TURN_TOLERANCE_DEG:
            turn(arlo, particles, max(-30.0, min(30.0, math.degrees(target_bearing))))
            failed_action = None
        else:
            success = drive(arlo, particles, DRIVE_STEP_CM)
            failed_action = None if success else 'forward'
            if success:
                detouring = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Enable real robot motion')
    args = parser.parse_args()
    if not args.run:
        print('DRY RUN: no robot initialized or moved. Inspect LANDMARKS and constants,')
        print('then run: python3 -u main.py --run')
        return

    print('CHECK: marker centres, sonar directions, turn sign and camera calibration.')
    input(f'Clear the area, point robot at L{START_LANDMARK}, then press Enter to START...')
    arlo = Arlo()
    camera = None
    try:
        camera = Camera(0, robottype='arlo', useCaptureThread=False)
        print('Initialising localization...')
        particles, weights = initialize(arlo, camera)
        for landmark_id in ROUTE:
            print(f'Next target: L{landmark_id}')
            particles, weights = visit_target(
                arlo, camera, particles, weights, landmark_id
            )
        print('Route finished')
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        arlo.stop()
        if camera is not None:
            camera.terminateCaptureThread()
            camera.get_capture().close() if hasattr(camera.get_capture(), 'close') else camera.get_capture().release()


if __name__ == '__main__':
    main()
