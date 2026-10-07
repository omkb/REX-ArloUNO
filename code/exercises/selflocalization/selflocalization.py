import time
import numpy as np

import particle
import camera

from arlo import Arlo, Direction

from mcl import (
    initialize_particles,
    move_particles,
    mcl_update,
    wrap_angle
)


# --------------------------------------------------
# MAP
# --------------------------------------------------

landmarks = {
    10: (0.0, 0.0),
    6: (170.0, 0.0)
}


# --------------------------------------------------
# ROBOT SETTINGS
# --------------------------------------------------

DRIVE_SPEED = 30       # cm/s
TURN_SPEED = 60        # degrees/s

GOAL_TOLERANCE = 10    # cm

BOX_LENGTH_CM = 27.5
BOX_WIDTH_CM = 21.5

# A marker on a long side is half the width from the center, while a
# marker on a short side is half the length from the center.
MARKER_TO_BOX_CENTER_OFFSETS_CM = (BOX_WIDTH_CM / 2, BOX_LENGTH_CM / 2)

ANGLE_TOLERANCE = np.deg2rad(5)

# Used when no landmark is visible
SEARCH_TURN = np.deg2rad(20)

MAX_STEPS = 30

INITIAL_POSITION_STD_CM = 15.0
INITIAL_THETA_STD = np.deg2rad(10)
CONFIDENT_FRAMES_REQUIRED = 3
MAX_INITIALIZATION_STEPS = 72


# --------------------------------------------------
# SETUP
# --------------------------------------------------

arlo = Arlo()

cam = camera.Camera(0, robottype="arlo", useCaptureThread=False)

particles = initialize_particles()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_goal():
    positions = list(landmarks.values())

    x1, y1 = positions[0]
    x2, y2 = positions[1]

    goal_x = (x1 + x2) / 2
    goal_y = (y1 + y2) / 2
    return goal_x, goal_y


def observe(particles):

    frame = cam.get_next_frame()

    object_ids, distances, angles = cam.detect_aruco_box_centers(frame, MARKER_TO_BOX_CENTER_OFFSETS_CM)

    if object_ids is None:
        print("No landmarks detected")
        return particles, set()

    observed_ids = {int(marker_id) for marker_id in object_ids if int(marker_id) in landmarks}

    if not observed_ids:
        print("No known landmarks detected")
        return particles, set()

    print("Measurements:")

    for marker_id, distance, angle in zip(object_ids, distances, angles):
        marker_id = int(marker_id)

        if marker_id in landmarks:
            print(
                f"  ID={marker_id}, "
                f"distance={distance:.1f} cm, "
                f"angle={np.degrees(angle):.1f} deg"
            )

    particles = mcl_update(particles, object_ids, distances, angles, landmarks)

    return particles, observed_ids


def particle_spread(particles):
    std_x = np.std([p.getX() for p in particles])
    std_y = np.std([p.getY() for p in particles])

    # Circular standard deviation handles angles around 0/360 degrees.
    angles = [p.getTheta() for p in particles]
    mean_cos = np.mean(np.cos(angles))
    mean_sin = np.mean(np.sin(angles))
    resultant_length = np.clip(np.hypot(mean_cos, mean_sin), 1e-12, 1.0)
    std_theta = np.sqrt(-2.0 * np.log(resultant_length))

    return std_x, std_y, std_theta


def turn_robot(particles, angle):

    if abs(angle) < ANGLE_TOLERANCE:
        return

    print(f"Turning {np.degrees(angle):.1f} degrees")

    # Our particle coordinate system:
    # positive = left / counter-clockwise
    #
    # Our physical Arlo has the opposite sign:
    # positive arlo.turn() = right
    arlo_angle = -np.degrees(angle)

    handle = arlo.turn(arlo_angle, speed_degs=TURN_SPEED, debug=False)

    handle.wait()

    # Update particle belief after physical movement
    move_particles(particles, delta_trans=0, delta_rot=angle)


def drive_robot(particles, distance):

    print(f"Driving {distance:.1f} cm")

    handle = arlo.drive(Direction.FORWARD, speed_cms=DRIVE_SPEED, distance_cm=distance, debug=False)

    handle.wait()

    # Update particle belief after physical movement
    move_particles(particles, delta_trans=distance, delta_rot=0)


def initialize_localization(particles):
    required_ids = set(landmarks)
    observed_ids = set()
    confident_frames = 0

    print("Initializing localization")

    for initialization_step in range(MAX_INITIALIZATION_STEPS):
        particles, current_ids = observe(particles)
        observed_ids.update(current_ids)

        std_x, std_y, std_theta = particle_spread(particles)
        all_landmarks_seen = required_ids.issubset(observed_ids)
        position_confident = std_x < INITIAL_POSITION_STD_CM and std_y < INITIAL_POSITION_STD_CM
        heading_confident = std_theta < INITIAL_THETA_STD

        # Count only new frames with a known landmark measurement.
        if current_ids and all_landmarks_seen and position_confident and heading_confident:
            confident_frames += 1
        else:
            confident_frames = 0

        print(
            f"Initialization {initialization_step + 1}: "
            f"seen={sorted(observed_ids)}, std_x={std_x:.1f} cm, "
            f"std_y={std_y:.1f} cm, std_theta={np.degrees(std_theta):.1f} deg, "
            f"confident_frames={confident_frames}/{CONFIDENT_FRAMES_REQUIRED}"
        )

        if confident_frames >= CONFIDENT_FRAMES_REQUIRED:
            print("Localization initialized")
            return particles

        # Stay still while confirming confidence with fresh frames.
        if confident_frames == 0:
            turn_robot(particles, SEARCH_TURN)
        time.sleep(0.2)

    raise RuntimeError("Could not initialize localization with sufficient confidence")


def navigation_step_size(distance_to_goal):
    if distance_to_goal > 100:
        return 30
    if distance_to_goal > 40:
        return 20
    return 10


# --------------------------------------------------
# MAIN
# --------------------------------------------------

goal_x, goal_y = get_goal()

print(
    f"Goal: ({goal_x:.1f}, {goal_y:.1f})"
)

try:

    particles = initialize_localization(particles)

    for step in range(MAX_STEPS):

        print(f"\n--- Step {step} ---")

        # --------------------------------------------------
        # 1. Observe landmarks and update MCL
        # --------------------------------------------------

        particles, observed_ids = observe(particles)

        # --------------------------------------------------
        # 3. Estimate current robot pose
        # --------------------------------------------------

        est_pose = particle.estimate_pose(particles)

        x = est_pose.getX()
        y = est_pose.getY()
        theta = est_pose.getTheta()

        print(f"Estimated pose: " f"x={x:.1f}, " f"y={y:.1f}, " f"theta={np.degrees(theta):.1f} deg")

        # --------------------------------------------------
        # 4. Calculate vector to midpoint
        # --------------------------------------------------

        dx = goal_x - x
        dy = goal_y - y

        distance_to_goal = np.sqrt(dx**2 + dy**2)

        print(f"Distance to goal: " f"{distance_to_goal:.1f} cm")

        # --------------------------------------------------
        # 5. Stop when midpoint is reached
        # --------------------------------------------------

        if distance_to_goal <= GOAL_TOLERANCE:
            print("Goal reached")
            break

        # --------------------------------------------------
        # 6. Search only after checking the propagated pose
        # --------------------------------------------------

        if not observed_ids and distance_to_goal > 50:
            print("Searching for landmark")
            turn_robot(particles, SEARCH_TURN)
            time.sleep(0.2)
            continue

        # --------------------------------------------------
        # 7. Calculate desired heading
        # --------------------------------------------------

        desired_heading = np.arctan2(dy, dx)

        turn_angle = wrap_angle(desired_heading - theta)

        # --------------------------------------------------
        # 8. Turn towards goal
        # --------------------------------------------------

        turn_robot(particles, turn_angle)

        # --------------------------------------------------
        # 9. Drive a short distance towards goal
        # --------------------------------------------------

        step_distance = navigation_step_size(distance_to_goal)
        drive_distance = min(step_distance, distance_to_goal)

        drive_robot(particles, drive_distance)

        time.sleep(0.2)

    else:
        print("Maximum number of steps reached")

finally:

    cam.terminateCaptureThread()
