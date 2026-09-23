import cv2
import numpy as np
import time
from picamera2 import Picamera2
import matplotlib.pyplot as plt
from arlo import Arlo, Direction

arlo = Arlo()


# --------------------------------------------------
# CAMERA
# --------------------------------------------------

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


# --------------------------------------------------
# ROBOT / OBSTACLE GEOMETRY
# --------------------------------------------------

CAMERA_OFFSET_CM = 22.5

BOX_LENGTH_CM = 35
BOX_WIDTH_CM = 30
ROBOT_RADIUS_CM = 25

# Because the ArUco marker is on the SIDE of the box,
# not at its center, use a conservative radius measured
# from a face center to the furthest box corner.

RADIUS_FROM_LENGTH_FACE = np.sqrt(
    BOX_WIDTH_CM ** 2 +
    (BOX_LENGTH_CM / 2) ** 2
)

RADIUS_FROM_WIDTH_FACE = np.sqrt(
    BOX_LENGTH_CM ** 2 +
    (BOX_WIDTH_CM / 2) ** 2
)

BOX_RADIUS_CM = max(
    RADIUS_FROM_LENGTH_FACE,
    RADIUS_FROM_WIDTH_FACE
)

COLLISION_RADIUS_CM = BOX_RADIUS_CM + ROBOT_RADIUS_CM

print("Box radius:", BOX_RADIUS_CM)
print("Collision radius:", COLLISION_RADIUS_CM)


# --------------------------------------------------
# RRT
# --------------------------------------------------

EXPAND_DISTANCE = 30


# --------------------------------------------------
# CAMERA SETUP
# --------------------------------------------------

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


# --------------------------------------------------
# LANDMARK DETECTION
# --------------------------------------------------

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

        # Position relative to camera
        x = tvec[0]
        z = tvec[2]

        # Convert to robot-center coordinate system

        landmarks.append(
            (marker_id, x, z)
        )

    return landmarks


# --------------------------------------------------
# OBSTACLE MAP
# --------------------------------------------------

def build_obstacle_map_with_radius(landmarks):

    obstacles = []

    for marker_id, x, z in landmarks:

        obstacles.append(
            (
                marker_id,
                x,
                z,
                COLLISION_RADIUS_CM
            )
        )

    return obstacles


# --------------------------------------------------
# COLLISION CHECK
# --------------------------------------------------

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

    p1 = np.array(p1)
    p2 = np.array(p2)

    distance = np.linalg.norm(p2 - p1)

    num_steps = int(distance / step_cm) + 1

    for t in np.linspace(0, 1, num_steps):

        point = p1 + t * (p2 - p1)

        if collision(point, obstacles):
            return True

    return False


# --------------------------------------------------
# RRT HELPERS
# --------------------------------------------------

def sample_point(x_min, x_max, z_min, z_max, goal):

    # Goal bias
    if np.random.rand() < 0.1:
        return np.array(goal)

    return np.array([
        np.random.uniform(x_min, x_max),
        np.random.uniform(z_min, z_max)
    ])


def find_nearest(tree, sample):

    distances = [
        np.linalg.norm(node["point"] - sample)
        for node in tree
    ]

    return np.argmin(distances)


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

# --------------------------------------------------
# RRT
# --------------------------------------------------

def rrt(start, goal, obstacles, max_iterations=5000):

    x_min, x_max, z_min, z_max = get_sampling_bounds(goal)

    tree = [
        {
            "point": np.array(start),
            "parent": None
        }
    ]

    goal = np.array(goal)

    for _ in range(max_iterations):

        sample = sample_point(
            x_min,
            x_max,
            z_min,
            z_max,
            goal
        )

        nearest_index = find_nearest(tree, sample)
        nearest_point = tree[nearest_index]["point"]

        new_point = expand(
            nearest_point,
            sample
        )

        if collision(new_point, obstacles):
            continue

        if collision_in_segment(
            nearest_point,
            new_point,
            obstacles
        ):
            continue

        tree.append({
            "point": new_point,
            "parent": nearest_index
        })

        new_index = len(tree) - 1

        if np.linalg.norm(new_point - goal) <= EXPAND_DISTANCE:

            if not collision_in_segment(
                new_point,
                goal,
                obstacles
            ):

                tree.append({
                    "point": goal,
                    "parent": new_index
                })

                return tree

    return None


# --------------------------------------------------
# EXTRACT PATH
# --------------------------------------------------

def extract_path(tree):

    path = []

    index = len(tree) - 1

    while index is not None:

        node = tree[index]

        path.append(node["point"])

        index = node["parent"]

    path.reverse()

    return path


# --------------------------------------------------
# FOLLOW PATH
# --------------------------------------------------

def follow_path(path, drive_speed=30, turn_speed=60):

    # Heading relative to ORIGINAL coordinate system.
    # 0 degrees = +z
    current_heading = 0.0

    for i in range(len(path) - 1):

        current = path[i]
        target = path[i + 1]

        dx = target[0] - current[0]
        dz = target[1] - current[1]

        distance = np.sqrt(
            dx ** 2 +
            dz ** 2
        )

        # +x is right
        desired_heading = np.degrees(
            np.arctan2(dx, dz)
        )

        turn_angle = (
            desired_heading -
            current_heading
        )

        # Normalize to [-180, 180]
        turn_angle = (
            (turn_angle + 180) % 360
        ) - 180

        print(
            f"Segment {i}: "
            f"{current} -> {target}, "
            f"turn={turn_angle:.1f} deg, "
            f"drive={distance:.1f} cm"
        )

        # Turn around ROBOT CENTER
        turn_handle = arlo.turn(
            turn_angle,
            speed_degs=turn_speed,
            debug = False)

        turn_handle.wait()

        # Move ROBOT CENTER towards next waypoint
        drive_handle = arlo.drive(
            Direction.FORWARD,
            speed_cms=drive_speed,
            distance_cm=distance,
            debug = False)

        drive_handle.wait()

        current_heading = desired_heading

    print("Path completed")

def simplify_path(path, obstacles):
    simplified_path = [path[0]]

    current_index = 0

    while current_index < len(path) - 1:

        # Start by trying to go directly to the goal
        next_index = len(path) - 1

        # Move backwards until we find a reachable point
        while next_index > current_index + 1:
            if not collision_in_segment(
                path[current_index],
                path[next_index],
                obstacles
            ):
                break

            next_index -= 1

        simplified_path.append(path[next_index])
        current_index = next_index

    return simplified_path

# --------------------------------------------------
# MAIN
# --------------------------------------------------

landmarks = detect_all_landmarks(frame)

print("Landmarks relative to robot center:")
print(landmarks)

obstacles = build_obstacle_map_with_radius(
    landmarks
)

print("Obstacles:")
print(obstacles)

start = (0, -CAMERA_OFFSET_CM)
goal = (0, 400)

tree = rrt(
    start,
    goal,
    obstacles
)

test_path = [
    np.array([0.0, -CAMERA_OFFSET_CM]),
    np.array([0.0, 50.0]),
    np.array([70.0, 120.0])
]


path = extract_path(tree)

print("Original path:")
for point in path:
    print(point)

path = simplify_path(path, obstacles)

print("Simplified path:")
for point in path:
    print(point)

follow_path(path)
