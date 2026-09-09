import cv2
import os
import numpy as np
from datetime import datetime
from picamera2 import Picamera2
import time

output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

# --- Capture image ---
picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)
picam2.start()
time.sleep(1)  # let auto-exposure/white-balance settle

frame = picam2.capture_array()
picam2.stop()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"capture_{timestamp}.png")

# picamera2 gives RGB; convert to BGR so colors save correctly via OpenCV
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imwrite(filename, frame_bgr)
print(f"Saved image to {filename}")

# --- Detect ArUco marker and measure pixel height ---
gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()

corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

if ids is not None:
    for corner_set, marker_id in zip(corners, ids.flatten()):
        pts = corner_set[0]  # top-left, top-right, bottom-right, bottom-left

        left_edge = np.linalg.norm(pts[0] - pts[3])
        right_edge = np.linalg.norm(pts[1] - pts[2])
        pixel_height = (left_edge + right_edge) / 2

        print(f"Marker {marker_id}: {pixel_height:.2f} px tall")
else:
    print("No markers detected.")
