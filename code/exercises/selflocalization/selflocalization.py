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

LANDMARK_A = 10
LANDMARK_B = 5

# Change this to the actual distance between the landmarks.
LANDMARK_DISTANCE = 170.0

landmarks = {
    LANDMARK_A: (0.0, 0.0),
    LANDMARK_B: (LANDMARK_DISTANCE, 0.0)
}


# --------------------------------------------------
# ROBOT SETTINGS
# --------------------------------------------------

DRIVE_SPEED = 30
TURN_SPEED = 60
INITIAL_SEARCH_TURN_SPEED = 20

STEP_DISTANCE = 30.0

GOAL_TOLERANCE = 10.0
ANGLE_TOLERANCE = np.deg2rad(5)

SEARCH_TURN = np.deg2rad(10)

MAX_STEPS = 40
MAX_INITIALIZATION_STEPS = 30


# --------------------------------------------------
# SETUP
# --------------------------------------------------

arlo = Arlo()

cam = camera.Camera(0, robottype="arlo", useCaptureThread=False)

particles = initialize_particles()


# --------------------------------------------------
# OBSERVATION
# --------------------------------------------------

def observe(particles, require_all_landmarks=False):

    frame = cam.get_next_frame()

    object_ids, distances, angles = cam.detect_aruco_objects(frame)

    if object_ids is None:
        print("No landmarks detected")
        return particles, set()

    observed_ids = {
        int(marker_id)
        for marker_id in object_ids
        if int(marker_id) in landmarks
    }

    if not observed_ids:
        print("No known landmarks detected")
        return particles, set()

    print("Measurements:")

    for marker_id, distance, angle in zip(object_ids, distances, angles):
        marker_id = int(marker_id)

        if marker_id in landmarks:
            print(f"  ID={marker_id}, " f"distance={distance:.1f} cm, " f"angle={np.degrees(angle):.1f} deg")

    # Initialization needs all landmarks in this frame before updating MCL.
    if require_all_landmarks and not set(landmarks).issubset(observed_ids):
        return particles, observed_ids

    particles = mcl_update(particles, object_ids, distances, angles, landmarks)

    return particles, observed_ids


# --------------------------------------------------
# ROBOT MOVEMENT
# --------------------------------------------------

def turn_robot(particles, angle, speed_degs=TURN_SPEED):

    if abs(angle) < ANGLE_TOLERANCE:
        return

    print(f"Turning {np.degrees(angle):.1f} degrees")

    # Particle coordinates:
    # positive angle = left / counter-clockwise.
    #
    # Physical Arlo:
    # positive arlo.turn() = right.
    arlo_angle = -np.degrees(angle)

    handle = arlo.turn(arlo_angle, speed_degs=speed_degs, debug=False)

    handle.wait()

    move_particles(particles, delta_trans=0, delta_rot=angle)


def drive_robot(particles, distance):

    print(f"Driving {distance:.1f} cm")

    handle = arlo.drive(Direction.FORWARD, speed_cms=DRIVE_SPEED, distance_cm=distance, debug=False)

    handle.wait()

    move_particles(particles, delta_trans=distance, delta_rot=0)


# --------------------------------------------------
# INITIAL LOCALIZATION
# --------------------------------------------------

def initialize_localization(particles):

    required_ids = set(landmarks)

    print("Initializing localization")

    for step in range(MAX_INITIALIZATION_STEPS):

        particles, observed_ids = observe(particles, require_all_landmarks=True)

        print(f"Landmarks in current frame: " f"{sorted(observed_ids)}")

        # observe has updated MCL once using both landmarks from this frame.
        if required_ids.issubset(observed_ids):
            print("Both landmarks seen in the same frame; localization initialized")
            return particles

        # Look for the other landmark.
        turn_robot(particles, SEARCH_TURN, speed_degs=INITIAL_SEARCH_TURN_SPEED)

        time.sleep(0.2)

    raise RuntimeError("Could not find both landmarks in the same frame")


# --------------------------------------------------
# GOAL
# --------------------------------------------------

def get_goal():

    x1, y1 = landmarks[LANDMARK_A]
    x2, y2 = landmarks[LANDMARK_B]

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

goal_x, goal_y = get_goal()

print(f"Goal: ({goal_x:.1f}, {goal_y:.1f})")

try:

    # First update MCL using both landmarks in the same frame.
    particles = initialize_localization(particles)

    for step in range(MAX_STEPS):

        print(f"\n--- Step {step} ---")

        # Observe and update particle filter.
        particles, observed_ids = observe(particles)

        # Estimate robot pose.
        pose = particle.estimate_pose(particles)

        x = pose.getX()
        y = pose.getY()
        theta = pose.getTheta()

        print(f"Estimated pose: " f"x={x:.1f}, " f"y={y:.1f}, " f"theta={np.degrees(theta):.1f} deg")

        # Vector from robot to goal.
        dx = goal_x - x
        dy = goal_y - y

        distance_to_goal = np.sqrt(dx**2 + dy**2)

        print(f"Distance to goal: " f"{distance_to_goal:.1f} cm")

        # Stop at midpoint.
        if distance_to_goal <= GOAL_TOLERANCE:
            print("Goal reached")
            break

        # Search without landmarks unless the goal is within one drive step.
        # For the final step, use the estimated pose to turn and drive to goal.
        if not observed_ids and distance_to_goal > STEP_DISTANCE:

            print("Searching for landmark")

            turn_robot(particles, SEARCH_TURN)

            time.sleep(0.2)

            continue

        # Direction from estimated pose to midpoint.
        desired_heading = np.arctan2(dy, dx)

        turn_angle = wrap_angle(desired_heading - theta)

        turn_robot(particles, turn_angle)

        # Drive a small step, then observe again.
        drive_distance = min(STEP_DISTANCE, distance_to_goal)

        drive_robot(particles, drive_distance)

        time.sleep(0.2)

    else:
        print("Maximum number of steps reached")

finally:

    cam.terminateCaptureThread()
