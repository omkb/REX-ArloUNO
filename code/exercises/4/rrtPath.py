import cv2
import numpy as np
import time
from picamera2 import Picamera2
import matplotlib.pyplot as plt

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

BOX_LENGTH_CM = 28
BOX_WIDTH_CM = 23
ROBOT_RADIUS_CM = 24

BOX_RADIUS_CM = np.sqrt(
    (BOX_LENGTH_CM / 2) ** 2 +
    (BOX_WIDTH_CM / 2) ** 2
)

COLLISION_RADIUS_CM = BOX_RADIUS_CM + ROBOT_RADIUS_CM

EXPAND_DISTANCE = 30

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

frame = cam.capture_array()

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


def build_obstacle_map_with_radius(landmarks):
    obstacles = []

    for marker_id, x, z in landmarks:
        obstacles.append(
            (marker_id, x, z, COLLISION_RADIUS_CM)
        )

    return obstacles


def collision(position, obstacles):
    x, z = position

    for marker_id, ox, oz, radius in obstacles:

        distance_squared = (
            (x - ox) ** 2 +
            (z - oz) ** 2
        )

        if distance_squared <= radius ** 2:
            return True

    return False

def collision_in_segment(p1, p2, obstacles, step_cm=10):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    distance = np.linalg.norm(p2 - p1)

    num_steps = int(distance / step_cm) + 1

    for t in np.linspace(0, 1, num_steps):
        point = p1 + t * (p2 - p1)

        if collision(point, obstacles):
            return True

    return False

def sample_point(x_min, x_max, z_min, z_max, goal):

    if np.random.rand() < 0.1:
        return np.array(goal, dtype=float)

    return np.array([
        np.random.uniform(x_min, x_max),
        np.random.uniform(z_min, z_max)
    ])

def find_nearest(tree, sample):

    distances = [
        np.linalg.norm(node["point"] - sample)
        for node in tree
    ]

    index = np.argmin(distances)

    return index

def expand(from_point, to_point):

    direction = to_point - from_point
    distance = np.linalg.norm(direction)

    if distance <= EXPAND_DISTANCE:
        return to_point

    direction = direction / distance

    return from_point + EXPAND_DISTANCE * direction

def get_sampling_bounds(goal, margin=200):
    x_min = min(0, goal[0]) - margin
    x_max = max(0, goal[0]) + margin

    z_min = 0
    z_max = goal[1] + margin

    return x_min, x_max, z_min, z_max


def rrt(start, goal, obstacles, max_iterations=5000):

    x_min, x_max, z_min, z_max = get_sampling_bounds(goal)

    tree = [{"point": np.array(start, dtype=float), "parent": None}]

    goal = np.array(goal)

    for _ in range(max_iterations):

        sample = sample_point(
            x_min, x_max,
            z_min, z_max,
            goal
        )

        nearest_index = find_nearest(tree, sample)
        nearest_point = tree[nearest_index]["point"]

        new_point = expand(nearest_point, sample)

        # New node itself is invalid
        if collision(new_point, obstacles):
            continue

        # Edge to new node is invalid
        if collision_in_segment(nearest_point, new_point, obstacles):
            continue

        tree.append({"point": new_point, "parent": nearest_index})

        new_index = len(tree) - 1

        # Are we close enough to goal
        if np.linalg.norm(new_point - goal) <= EXPAND_DISTANCE:

            # Can we connect the final segment
            if not collision_in_segment(new_point, goal, obstacles):

                tree.append({"point": goal, "parent": new_index})

                return tree

    return None

def extract_path(tree):

    path = []

    index = len(tree) - 1

    while index is not None:

        node = tree[index]

        path.append(node["point"])

        index = node["parent"]

    path.reverse()

    return path


landmarks = detect_all_landmarks(frame)

obstacles = build_obstacle_map_with_radius(landmarks)

start = (0, 0)
goal = (0, 300)

tree = rrt(start, goal, obstacles)

path = extract_path(tree)

print(path)

