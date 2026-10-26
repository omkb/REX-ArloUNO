"""RobotRally: sequential landmark MCL and reactive obstacle avoidance.

Run only after checking the landmark coordinates, turn sign and sonar directions.
All driving commands are open-loop estimates; test at low speed on a clear floor.
"""
import argparse
import math
import time

import cv2
import numpy as np

import mapstream
import mcl
import navigation
import obstacles
from arlo import Arlo, Direction
from camera import Camera

# Coordinates in cm. These describe the schematic square; correct them to
# the actual *observed ArUco marker centres* before relying on localization.
# Landmark name -> (ArUco marker ID, (x, y)). Route goes L1, L2, L3, L4, L1.
LANDMARKS = {
    'L1': (7, (0.0, 0.0)),
    'L2': (3, (0.0, 300.0)),
    'L3': (6, (400.0, 0.0)),
    'L4': (11, (400.0, 300.0)),
}
LANDMARK_BY_MARKER = {marker_id: name for name, (marker_id, _) in LANDMARKS.items()}
POSITIONS = {name: position for name, (_, position) in LANDMARKS.items()}
START_LANDMARK = 'L1'
ROUTE = list(LANDMARKS) + [START_LANDMARK]

SCAN_STEP_DEG = 10
SCAN_SETTLE_S = 0.25
MAX_POSITION_SPREAD_CM = 55.0  # Experimental threshold, tune on the real robot
DRIVE_STEP_CM = 15.0
DRIVE_SPEED_CMS = 20.0
TURN_SPEED_DEG_S = 60.0
TURN_TOLERANCE_DEG = 12.0
AVOID_TURN_DEG = 25.0       # Turn step when avoiding an obstacle
TARGET_DISTANCE_CM = 45.0    # Reached when the estimated position is within this distance

STREAM_MAP = True             # Stream the particle map to http://<robot>:STREAM_PORT/
STREAM_PORT = 8000
MAP_MARGIN_PX = 100           # 1 px per cm, plus a margin around the landmarks
MAP_MAX_DRAWN = 200           # Drawing all particles is slow on the Pi

streamer = None


def observations(camera):
    """Read one image and return (seen_landmarks, obstacle_markers).

    seen_landmarks is {name: (cm, bearing_rad)}; every other marker is an obstacle,
    returned as a list of (cm, bearing_rad).
    """
    ids, distances, angles = camera.detect_aruco_objects(camera.get_next_frame())
    if ids is None:
        return {}, []
    seen_landmarks, obstacle_markers = {}, []
    for marker_id, distance, angle in zip(ids, distances, angles):
        name = LANDMARK_BY_MARKER.get(int(marker_id))
        if name is None:
            obstacle_markers.append((float(distance), float(angle)))
        elif name not in seen_landmarks or distance < seen_landmarks[name][0]:
            seen_landmarks[name] = (float(distance), float(angle))
    return seen_landmarks, obstacle_markers


def draw_map(particles, weights, target=None):
    """Draw landmarks, particles and the pose estimate (BGR image, y-axis up)."""
    xs = [p[0] for p in POSITIONS.values()]
    ys = [p[1] for p in POSITIONS.values()]
    min_x, min_y = min(xs) - MAP_MARGIN_PX, min(ys) - MAP_MARGIN_PX
    width = int(max(xs) - min_x + MAP_MARGIN_PX)
    height = int(max(ys) - min_y + MAP_MARGIN_PX)
    world = np.full((height, width, 3), 255, dtype=np.uint8)

    def px(x, y):
        return int(x - min_x), int(height - (y - min_y))

    for name, (marker_id, (lx, ly)) in LANDMARKS.items():
        colour = (0, 0, 255) if name == target else (0, 160, 0)
        centre = px(lx, ly)
        cv2.circle(world, centre, 8, colour, 2)
        cv2.putText(world, f"{name} ({marker_id})", (centre[0] + 10, centre[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)

    if particles is None:
        return world

    drawn = particles
    if len(particles) > MAP_MAX_DRAWN:
        drawn = particles[np.random.choice(len(particles), MAP_MAX_DRAWN, replace=False)]
    for x, y, theta in drawn:
        start = px(x, y)
        cv2.circle(world, start, 2, (200, 120, 0), -1)
        cv2.line(world, start, px(x + 10 * math.cos(theta), y + 10 * math.sin(theta)),
                 (200, 120, 0), 1)

    x, y, theta = mcl.pose_estimate(particles, weights)
    start = px(x, y)
    cv2.circle(world, start, 6, (0, 0, 0), 2)
    cv2.line(world, start, px(x + 25 * math.cos(theta), y + 25 * math.sin(theta)),
             (0, 0, 0), 2)
    spread = mcl.position_spread_cm(particles, weights)
    cv2.putText(world, f"({x:.0f}, {y:.0f}, {math.degrees(theta):.0f} deg) spread={spread:.0f} cm",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return world


def publish_map(particles, weights, target=None):
    if streamer is not None:
        streamer.publish(draw_map(particles, weights, target))


def sonar(arlo, direction):
    """Returns a sonar reading in cm; None for invalid/unavailable values."""
    method = getattr(arlo.robot, f"read_{direction}_ping_sensor")
    value = method()
    return float(value) / 10 if value is not None and value > 0 else None


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
    if not obstacles.is_clear(sonar(arlo, sensor), distance_cm):
        return False
    direction = Direction.FORWARD if distance_cm > 0 else Direction.BACKWARD
    arlo.drive(direction, DRIVE_SPEED_CMS, distance_cm=abs(distance_cm), debug=False).wait()
    if particles is not None:
        mcl.motion_update(particles, delta_trans_cm=distance_cm)
    return True


def update_localization(particles, weights, seen_landmarks, target=None):
    """Weight particles by the seen landmarks, resample, and publish the map."""
    if mcl.measurement_update(particles, weights, seen_landmarks, POSITIONS):
        particles, weights = mcl.resample(particles, weights)
    publish_map(particles, weights, target)
    return particles, weights


def choose_action(arlo, obstacle_markers=(), failed_action=None):
    return obstacles.choose_action(
        sonar(arlo, 'front'), sonar(arlo, 'left'),
        sonar(arlo, 'right'), sonar(arlo, 'back'),
        obstacle_markers, distance_cm=DRIVE_STEP_CM, failed_action=failed_action
    )


def perform(arlo, particles, action):
    """Perform one action; return False if drive() refused to move."""
    if action == 'forward':
        return drive(arlo, particles, DRIVE_STEP_CM)
    if action == 'backward':
        return drive(arlo, particles, -obstacles.REVERSE_CM)
    turn(arlo, particles, -AVOID_TURN_DEG if action == 'right' else AVOID_TURN_DEG)
    return True


def reposition(arlo, particles, obstacle_markers, failed_action):
    """Choose one movement. Report translation so initialization can reset MCL."""
    action = choose_action(arlo, obstacle_markers, failed_action)
    success = perform(arlo, particles, action)
    moved = success and action in ('forward', 'backward')
    return moved, None if success else action


def initialize(arlo, camera):
    """Initialize from the starting landmark and another; restart after repositioning."""
    particles = weights = None
    seen_second = False
    failed_action = None
    while True:
        for _ in range(360 // SCAN_STEP_DEG):
            seen_landmarks, obstacle_markers = observations(camera)
            print(f"Detected landmarks: {list(seen_landmarks)}")
            if particles is None and START_LANDMARK in seen_landmarks:
                # The starting landmark is used up by initialization.
                particles, weights = mcl.initialize_from_landmark(
                    POSITIONS, START_LANDMARK, *seen_landmarks.pop(START_LANDMARK)
                )

            if particles is None:
                publish_map(None, None)
            else:
                particles, weights = update_localization(particles, weights, seen_landmarks)
                seen_second = seen_second or any(i != START_LANDMARK for i in seen_landmarks)
                spread = mcl.position_spread_cm(particles, weights)
                if seen_second and seen_landmarks and spread <= MAX_POSITION_SPREAD_CM:
                    print(f"Localization ready; spread={spread:.1f} cm")
                    return particles, weights
            turn(arlo, particles, SCAN_STEP_DEG)
            time.sleep(SCAN_SETTLE_S)

        print("Repositioning to improve localization")
        moved, failed_action = reposition(arlo, particles, obstacle_markers, failed_action)
        if moved:
            particles = weights = None
            seen_second = False


def visit_target(arlo, camera, particles, weights, target):
    """Drive towards the target using MCL; the marker need not be visible."""
    failed_action = None
    detouring = False
    while True:
        seen_landmarks, obstacle_markers = observations(camera)
        particles, weights = update_localization(particles, weights, seen_landmarks, target)
        x, y, theta = mcl.pose_estimate(particles, weights)
        target_bearing, target_distance = navigation.direction_to_target(
            x, y, theta, *POSITIONS[target]
        )
        spread = mcl.position_spread_cm(particles, weights)
        print(f"{target}: pose=({x:.0f}, {y:.0f}, {math.degrees(theta):.0f}°), "
              f"dist={target_distance:.0f} cm, spread={spread:.0f} cm")

        if target_distance <= TARGET_DISTANCE_CM and spread <= MAX_POSITION_SPREAD_CM:
            print(f"Reached {target}")
            return particles, weights

        action = choose_action(arlo, obstacle_markers, failed_action)
        if action != 'forward':
            detouring = True
        elif not detouring and abs(math.degrees(target_bearing)) > TURN_TOLERANCE_DEG:
            turn(arlo, particles, max(-30.0, min(30.0, math.degrees(target_bearing))))
            failed_action = None
            continue

        success = perform(arlo, particles, action)
        failed_action = None if success else action
        if success and action == 'forward':
            detouring = False


def main():
    global streamer
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Enable real robot motion')
    args = parser.parse_args()
    if not args.run:
        print('DRY RUN: no robot initialized or moved. Inspect LANDMARKS and constants,')
        print('then run: python3 -u main.py --run')
        return

    print('CHECK: marker centres, sonar directions, turn sign and camera calibration.')
    input(f'Clear the area, point robot at {START_LANDMARK}, then press Enter to START...')
    arlo = Arlo()
    camera = None
    if STREAM_MAP:
        streamer = mapstream.MapStreamer(STREAM_PORT)
        publish_map(None, None)
    try:
        camera = Camera(0, robottype='arlo', useCaptureThread=False)
        print('Initialising localization...')
        particles, weights = initialize(arlo, camera)
        for target in ROUTE:
            print(f'Next target: {target} (marker {LANDMARKS[target][0]})')
            particles, weights = visit_target(arlo, camera, particles, weights, target)
        print('Route finished')
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        arlo.stop()
        if camera is not None:
            camera.terminateCaptureThread()
            camera.get_capture().close() if hasattr(camera.get_capture(), 'close') else camera.get_capture().release()
        if streamer is not None:
            streamer.close()


if __name__ == '__main__':
    main()
