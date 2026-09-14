import cv2
import numpy as np
import time
from picamera2 import Picamera2

IMAGE_SIZE = (1640, 1232)

f = 1288.37

cx = IMAGE_SIZE[0] / 2
cy = IMAGE_SIZE[1] / 2

camera_matrix = np.array([
    [f, 0, cx],
    [0, f, cy],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1), dtype=np.float32)

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

parameters = cv2.aruco.DetectorParameters_create()

MARKER_SIZE_CM = 14.4


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


def detect_all_landmarks(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

    if ids is None:
        return []

    _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        MARKER_SIZE_CM,
        camera_matrix,
        dist_coeffs
    )

    landmarks = []

    for i in range(len(ids)):

        marker_id = ids[i][0]
        tvec = tvecs[i][0]

        x = tvec[0]
        z = tvec[2]

        landmarks.append((marker_id, x, z))

    return landmarks


try:

    frame = cam.capture_array()

    landmarks = detect_all_landmarks(frame)

    print("Detected landmarks:")

    for marker_id, x, z in landmarks:
        print(
            f"ID={marker_id}, "
            f"X={x:.1f} cm, "
            f"Z={z:.1f} cm"
        )

finally:
    cam.stop()
