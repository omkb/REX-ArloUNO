import cv2
import numpy as np
import time
from picamera2 import Picamera2
import robot


# --------------------------------------------------
# ROBOT
# --------------------------------------------------

arlo = robot.Robot()

SEARCH_POWER = 45
TURN_POWER = 45
FORWARD_POWER = 50

LOOP_DELAY = 0.1

# How close we want to get to the landmark
STOP_DISTANCE_CM = 35

# If abs(X) is smaller than this, we consider
# the robot sufficiently aligned
FORWARD_STEP_TIME = 0.75

TARGET_IDS = [7]

# --------------------------------------------------
# CAMERA
# --------------------------------------------------

IMAGE_SIZE = (1640, 1232)

cam = Picamera2()

config = cam.create_video_configuration(
    main={
        "size": IMAGE_SIZE,
        "format": "RGB888"
    }
)

cam.configure(config)
cam.start()

time.sleep(1)


# --------------------------------------------------
# CAMERA CALIBRATION
# --------------------------------------------------

f = 1288.37

cx = IMAGE_SIZE[0] / 2
cy = IMAGE_SIZE[1] / 2

camera_matrix = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
], dtype=np.float32)

# We have not measured lens distortion
dist_coeffs = np.zeros((5, 1), dtype=np.float32)


# --------------------------------------------------
# ARUCO
# --------------------------------------------------

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

parameters = cv2.aruco.DetectorParameters_create()

# Physical side length of the ArUco marker
# Use the actual black marker size
MARKER_SIZE_CM = 14.4


# --------------------------------------------------
# MOTOR FUNCTIONS
# --------------------------------------------------

def stop():
    arlo.stop()


def search_turn():
    """
    Rotate clockwise around the robot's center.
    """
    arlo.go_diff(
        SEARCH_POWER,
        SEARCH_POWER,
        1,
        0
    )


def turn_left():
    arlo.go_diff(
        TURN_POWER,
        TURN_POWER,
        0,
        1
    )


def turn_right():
    arlo.go_diff(
        TURN_POWER,
        TURN_POWER,
        1,
        0
    )


def forward():
    arlo.go_diff(
        FORWARD_POWER,
        FORWARD_POWER,
        1,
        1
    )

def get_tolerance(z):
     return max(20, 0.20*z)

# --------------------------------------------------
# ARUCO DETECTION
# --------------------------------------------------

def detect_landmark(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    corners, ids, _  = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    if ids is None:
        return None

    for i, marker_id in enumerate(ids.flatten()):

        if marker_id not in TARGET_IDS:
            continue

        marker_corners = corners[i]

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners,
            MARKER_SIZE_CM,
            camera_matrix,
            dist_coeffs
        )

        return marker_id, rvecs[0][0], tvecs[0][0]

    return None

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

state = "SEARCH"

try:
    while True:

        # ------------------------------------------
        # SEARCH
        # ------------------------------------------
        if state == "SEARCH":

            search_turn()
            time.sleep(0.15)

            stop()
            time.sleep(0.2)

            frame = cam.capture_array()
            result = detect_landmark(frame)

            if result is None:
                print("SEARCH: no landmark")
                continue

            marker_id, rvec, tvec = result

            print("Found marker:", marker_id)
            print("tvec:", tvec)

            state = "ALIGN"


        # ------------------------------------------
        # ALIGN
        # ------------------------------------------
        elif state == "ALIGN":

            # Robot should be stationary here
            stop()
            time.sleep(0.2)

            frame = cam.capture_array()
            result = detect_landmark(frame)

            if result is None:
                print("Lost landmark")
                state = "SEARCH"
                continue

            marker_id, rvec, tvec = result

            x = tvec[0]
            z = tvec[2]

            tolerance = get_tolerance(z)

            print(
                f"ALIGN: ID={marker_id}, "
                f"X={x:.1f} cm, "
                f"Z={z:.1f} cm"
            )

            if x > tolerance:
                print("Landmark is right -> turn right")

                turn_right()
                time.sleep(0.1)
                stop()

            elif x < -tolerance:
                print("Landmark is left -> turn left")

                turn_left()
                time.sleep(0.1)
                stop()

            else:
                print("Aligned")
                stop()

                state = "APPROACH"


        # ------------------------------------------
        # APPROACH
        # ------------------------------------------
        elif state == "APPROACH":

            stop()
            time.sleep(0.2)

            frame = cam.capture_array()
            result = detect_landmark(frame)

            if result is None:
                print("Lost landmark while approaching")

                state = "SEARCH"
                continue

            marker_id, rvec, tvec = result

            x = tvec[0]
            z = tvec[2]

            tolerance = get_tolerance(z)

            print(
                f"APPROACH: "
                f"X={x:.1f} cm, "
                f"Z={z:.1f} cm"
            )

            if z <= STOP_DISTANCE_CM:
                stop()

                print("Reached landmark")
                break


            # Only realign if we are clearly off course
            if abs(x) > tolerance:
                print("Too far off center -> realign")
                state = "ALIGN"
                continue

            # Otherwise just drive forward a little
            print("Forward")
            forward()
            time.sleep(FORWARD_STEP_TIME)
            stop()

finally:
    stop()
    cam.stop()

    print("Finished")
