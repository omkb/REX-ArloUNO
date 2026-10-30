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
import rrt
from arlo import Arlo, Direction
from camera import Camera

# Coordinates in cm. These describe the schematic square; correct them to
# the actual *observed ArUco marker centres* before relying on localization.
# Landmark name -> (ArUco marker ID, (x, y)). Route goes L1, L2, L3, L4, L1.
LANDMARKS = {
    'L1': (1, (0.0, 0.0)),
    'L2': (2, (0.0, 300.0)),
    'L3': (3, (400.0, 0.0)),
    'L4': (4, (400.0, 300.0)),
}
LANDMARK_BY_MARKER = {marker_id: name for name, (marker_id, _) in LANDMARKS.items()}
POSITIONS = {name: position for name, (_, position) in LANDMARKS.items()}
START_LANDMARK = 'L1'
ROUTE = list(LANDMARKS) + [START_LANDMARK]

SCAN_STEP_DEG = 10
SCAN_SETTLE_S = 0.25
SURVEY_FRAMES = 3            # Frames taken facing a new target landmark before planning its leg
MAX_POSITION_SPREAD_CM = 55.0  # Experimental threshold, tune on the real robot
DRIVE_STEP_CM = 30.0         # Forward step while detouring, and the shortest segment drive
MEASURE_MIN_CM = 10.0        # While driving, update MCL from landmarks at most once per this distance
DRIVE_SPEED_CMS = 40.0
APPROACH_SPEED_CMS = 20.0    # Slow down to this within APPROACH_DISTANCE_CM of the target or anything ahead
APPROACH_DISTANCE_CM = 100.0
TURN_SPEED_DEG_S = 50.0     # Slowest real turn: wheels at SPEED_AT_PWM_MIN_CMS (lower is clamped)
TURN_TOLERANCE_DEG = 12.0
MAX_TURN_STEP_DEG = 180.0   # Longest single turn towards a target; 180 turns the full bearing at once
FRONT_STOP_CM = 25          # Sonar: stop for anything this close ahead (or behind, reversing)
SIDE_STOP_CM = 12           # ... or this close to the diagonal sonars
ESCAPE_CM = 60              # Drive this far away from what the sonars found
BACKUP_CM = 20
ESCAPE_TURN_DEG = 45
ROBOT_RADIUS_CM = 22.0
TOUCH_CM = 42.0             # No sonar bump this close to a landmark or a mapped obstacle
SONAR_BEARINGS = {'front': 0.0, 'left': math.radians(45), 'right': math.radians(-45)}
TARGET_DISTANCE_CM = 50.0    # Reached when the estimated position is within this distance
GOAL_RADIUS_CM = TARGET_DISTANCE_CM / 2  # RRT paths end this far from the landmark, on its free side
WAYPOINT_DISTANCE_CM = 20.0  # Advance to the next RRT waypoint within this distance
STEER_GAIN = 1.0             # Turn rate (deg/s) per degree of bearing error while driving a segment
MAX_STEER_DEG_S = 20.0       # Sharpest arc while driving; 20 deg/s at 40 cm/s is a ~115 cm radius
STEER_MIN_DISTANCE_CM = 40.0  # Drive straight this close to the waypoint, where the bearing swings
COURT_WIDTH_CM = 400.0       # Court spans x in [0, 400] and y in [0, 300], L1 at the origin
COURT_HEIGHT_CM = 300.0
COURT_MARGIN_CM = 90.0       # Drivable margin around the court; driving beyond it is not allowed
RRT_BOUNDS = (-COURT_MARGIN_CM, COURT_WIDTH_CM + COURT_MARGIN_CM,
              -COURT_MARGIN_CM, COURT_HEIGHT_CM + COURT_MARGIN_CM)

STREAM_MAP = True             # Stream the particle map to http://<robot>:STREAM_PORT/
STREAM_PORT = 8000
MAP_MARGIN_PX = 100           # 1 px per cm, plus a margin around the landmarks
MAP_MAX_DRAWN = 200           # Drawing all particles is slow on the Pi
MAP_PUBLISH_INTERVAL_S = 1.0  # Map updates while driving a segment; each one pauses sensing briefly
MAP_MAX_SIGHTINGS = 10        # Later sightings still move a mapped obstacle by at least 1/10
TRAIL_MIN_STEP_CM = 5.0       # Only extend the trail once the pose estimate moves this far

streamer = None
obstacle_map = {}  # Marker ID -> (x, y, sightings): obstacle centres in world cm
trail = []         # Pose estimates (x, y) along the route, in world cm
bumps = []         # (x, y) of obstacles found by the sonars this leg, in world cm


def observations(camera):
    """Read one image and return (seen_landmarks, obstacle_markers).

    seen_landmarks is {name: (cm, bearing_rad)}; every other marker is an obstacle,
    returned as {marker_id: (cm, bearing_rad)}.
    """
    ids, distances, angles = camera.detect_aruco_objects(camera.get_next_frame())
    if ids is None:
        return {}, {}
    seen_landmarks, obstacle_markers = {}, {}
    for marker_id, distance, angle in zip(ids, distances, angles):
        marker_id = int(marker_id)
        name = LANDMARK_BY_MARKER.get(marker_id)
        seen = seen_landmarks if name is not None else obstacle_markers
        key = name if name is not None else marker_id
        if key not in seen or distance < seen[key][0]:
            seen[key] = (float(distance), float(angle))
    return seen_landmarks, obstacle_markers


def map_obstacles(obstacle_markers, x, y, theta):
    """Place seen obstacle markers on the map, averaging repeated sightings."""
    for marker_id, (distance, bearing) in obstacle_markers.items():
        reach = distance + obstacles.OBSTACLE_RADIUS_CM
        ox = x + reach * math.cos(theta + bearing)
        oy = y + reach * math.sin(theta + bearing)
        if marker_id in obstacle_map:
            mx, my, n = obstacle_map[marker_id]
            n = min(n + 1, MAP_MAX_SIGHTINGS)
            ox, oy = mx + (ox - mx) / n, my + (oy - my) / n
        else:
            n = 1
            print(f"Mapped obstacle {marker_id} at ({ox:.0f}, {oy:.0f})")
        obstacle_map[marker_id] = (ox, oy, n)


def record_trail(x, y):
    if not trail or math.hypot(x - trail[-1][0], y - trail[-1][1]) >= TRAIL_MIN_STEP_CM:
        trail.append((x, y))


def draw_map(particles, weights, target=None, path=None):
    """Draw landmarks, obstacles, particles, the pose estimate and the RRT path planned for
    the current leg (BGR image, y-axis up). ``path`` runs from the leg start to the target."""
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

    for marker_id, (ox, oy, _) in obstacle_map.items():
        centre = px(ox, oy)
        cv2.circle(world, centre, int(obstacles.OBSTACLE_RADIUS_CM), (90, 90, 90), -1)
        cv2.circle(world, centre, int(obstacles.MAP_CLEARANCE_CM), (170, 170, 170), 1)
        cv2.putText(world, str(marker_id), (centre[0] - 8, centre[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    for bx, by in bumps:
        centre = px(bx, by)
        cv2.circle(world, centre, int(obstacles.OBSTACLE_RADIUS_CM), (0, 0, 160), -1)
        cv2.circle(world, centre, int(obstacles.MAP_CLEARANCE_CM), (150, 150, 230), 1)

    if len(trail) > 1:
        points = np.array([px(*point) for point in trail], dtype=np.int32)
        cv2.polylines(world, [points], False, (0, 140, 255), 2, cv2.LINE_AA)

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
    if path:
        points = np.array([px(*point) for point in path], dtype=np.int32)
        cv2.polylines(world, [points], False, (200, 0, 200), 2, cv2.LINE_AA)
        for point in path[1:-1]:  # The ends are the leg start and the target landmark
            cv2.drawMarker(world, px(*point), (200, 0, 200), cv2.MARKER_CROSS, 12, 2)
    spread = mcl.position_spread_cm(particles, weights)
    cv2.putText(world, f"({x:.0f}, {y:.0f}, {math.degrees(theta):.0f} deg) spread={spread:.0f} cm",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return world


def publish_map(particles, weights, target=None, path=None):
    if streamer is not None:
        streamer.publish(draw_map(particles, weights, target, path))


def sonar(arlo, direction):
    """Returns a sonar reading in cm; None for invalid/unavailable values."""
    method = getattr(arlo.robot, f"read_{direction}_ping_sensor")
    with arlo.serial_lock:  # The drive thread may be sending motor commands
        value = method()
    return float(value) / 10 if value is not None and value > 0 else None


def clamp_turn(degrees):
    return max(-MAX_TURN_STEP_DEG, min(MAX_TURN_STEP_DEG, degrees))


def turn(arlo, particles, degrees):
    """Turn physically, then update particles using commanded (not measured) angle."""
    if abs(degrees) < 0.01:
        return
    # Arlo's wheel commands turn right for positive degrees; MCL uses left.
    arlo.turn(-degrees, speed_degs=TURN_SPEED_DEG_S, debug=False).wait()
    if particles is not None:
        mcl.motion_update(particles, delta_rot_rad=math.radians(degrees))


def sonars(arlo, *names):
    """{name: cm} for the front and diagonal sonars. A failed reading (twice) counts as 0."""
    readings = {}
    for name in names or ('front', 'left', 'right'):
        value = sonar(arlo, name)
        if value is None:
            value = sonar(arlo, name)
        readings[name] = value or 0.0
    return readings


def blocked(readings):
    return any(cm < (SIDE_STOP_CM if name in ('left', 'right') else FRONT_STOP_CM)
               for name, cm in readings.items())


def escape_drive(arlo, particles, distance_cm):
    """Drive straight, halting if the sonars find something close in the driving direction."""
    names = ('front', 'left', 'right') if distance_cm > 0 else ('back',)
    direction = Direction.FORWARD if distance_cm > 0 else Direction.BACKWARD
    handle = arlo.drive(direction, DRIVE_SPEED_CMS, distance_cm=abs(distance_cm), debug=False)
    try:
        while handle.is_alive():
            if blocked(sonars(arlo, *names)):
                print("Escape drive halted: sonar")
                break
    finally:
        if handle.is_alive():
            handle.halt("sonar")
    if particles is not None:
        mcl.motion_update(particles, math.copysign(handle.traveled, distance_cm),
                          math.radians(handle.turned))


def known_pose(particles, weights):
    if particles is None or mcl.position_spread_cm(particles, weights) > MAX_POSITION_SPREAD_CM:
        return None
    return mcl.pose_estimate(particles, weights)


def room_ahead(arlo, particles, weights):
    """Room to drive ESCAPE_CM ahead: by the sonars, and by the map if the pose is known
    (the end must be inside the bounds and outside the no-go zones we are not in already)."""
    readings = sonars(arlo)
    if readings['front'] < ESCAPE_CM + FRONT_STOP_CM or blocked(readings):
        return False
    pose = known_pose(particles, weights)
    if pose is None:
        return True
    x, y, theta = pose
    end = (x + ESCAPE_CM * math.cos(theta), y + ESCAPE_CM * math.sin(theta))
    x_min, x_max, y_min, y_max = RRT_BOUNDS
    return (x_min <= end[0] <= x_max and y_min <= end[1] <= y_max
            and all(math.dist(end, (ox, oy)) > r or math.dist((x, y), (ox, oy)) <= r
                    for ox, oy, r in no_go_zones()))


def remember_bump(particles, weights, readings):
    """Put what the closest sonar sees on the map, unless it is a landmark or an obstacle
    the camera has mapped already."""
    pose = known_pose(particles, weights)
    if pose is None:
        return
    x, y, theta = pose
    name = min(readings, key=readings.get)
    reach = ROBOT_RADIUS_CM + readings[name] + obstacles.OBSTACLE_RADIUS_CM
    bump = (x + reach * math.cos(theta + SONAR_BEARINGS[name]),
            y + reach * math.sin(theta + SONAR_BEARINGS[name]))
    known = list(POSITIONS.values()) + [(ox, oy) for ox, oy, _ in obstacle_map.values()]
    if all(math.dist(bump, point) > TOUCH_CM for point in known):
        print(f"Sonar bump at ({bump[0]:.0f}, {bump[1]:.0f})")
        bumps.append(bump)


def escape(arlo, particles, weights, side=None):
    """Something close: turn away from it, to the side with the most room (or the given
    side, +1 left / -1 right), in 45 degree steps until there is room ahead, then drive
    ESCAPE_CM. Also used to look from elsewhere when localization fails."""
    readings = sonars(arlo)
    if blocked(readings):
        remember_bump(particles, weights, readings)
        print(f"Escaping: {readings}")
        if readings['front'] < FRONT_STOP_CM and min(readings['left'], readings['right']) < SIDE_STOP_CM:
            escape_drive(arlo, particles, -BACKUP_CM)
    if side is None:
        side = 1 if readings['left'] > readings['right'] else -1
    for _ in range(360 // ESCAPE_TURN_DEG):
        if room_ahead(arlo, particles, weights):
            escape_drive(arlo, particles, ESCAPE_CM)
            return
        turn(arlo, particles, side * ESCAPE_TURN_DEG)
    escape_drive(arlo, particles, -BACKUP_CM)  # No room anywhere: back up and look again


def update_localization(particles, weights, seen_landmarks):
    """Weight particles by the seen landmarks and resample."""
    if mcl.measurement_update(particles, weights, seen_landmarks, POSITIONS):
        particles, weights = mcl.resample(particles, weights)
    return particles, weights


def segment_distance(waypoint_distance):
    """Distance to drive towards a waypoint in one continuous segment. The last
    waypoint already stops GOAL_RADIUS_CM short of the landmark."""
    return max(DRIVE_STEP_CM, waypoint_distance)


def predict(particles, moved_cm, turned_rad):
    """Particles moved along an arc of moved_cm that turns by turned_rad, without noise."""
    shown = particles.copy()
    heading = shown[:, 2] + turned_rad / 2  # Mean heading along the arc
    shown[:, 0] += moved_cm * np.cos(heading)
    shown[:, 1] += moved_cm * np.sin(heading)
    shown[:, 2] = mcl.wrap_angle(shown[:, 2] + turned_rad)
    return shown


def no_go_zones():
    """Mapped obstacles and sonar bumps as (x, y, radius) circles the robot centre must
    stay out of."""
    return ([(ox, oy, obstacles.MAP_CLEARANCE_CM) for ox, oy, _ in obstacle_map.values()]
            + [(bx, by, obstacles.MAP_CLEARANCE_CM) for bx, by in bumps])


def path_blocked(path, waypoint_index, x, y):
    """True when the rest of the path crosses a no-go zone: the planned segments from
    the current one on, or the line from the robot to its waypoint, where it steers.
    Obstacles are mapped after planning, and repeated sightings move them."""
    zones = no_go_zones()
    rest = path[waypoint_index - 1:]
    if rrt.path_in_collision(rest, zones):
        return True
    # From inside a zone every line is blocked; leaving it is the planned segment's job
    return (not any(math.dist((x, y), (ox, oy)) <= r for ox, oy, r in zones)
            and rrt.path_in_collision([(x, y), path[waypoint_index]], zones))


def drive_segment(arlo, camera, particles, weights, distance_cm, target, path,
                  waypoint_index):
    """Drive distance_cm without stopping, watching the front sonar and the camera.

    Steers in an arc towards path[waypoint_index] along the way by giving one wheel
    more power than the other, so a heading error is corrected without stopping to
    turn. Halts early for an obstacle ahead or when the rest of the path runs through
    a no-go zone (see path_blocked), and slows to
    APPROACH_SPEED_CMS within APPROACH_DISTANCE_CM of the target landmark (by pose
    estimate or camera) or of whatever the front sonar sees. Landmarks seen on the
    way update MCL, and the map is published every MAP_PUBLISH_INTERVAL_S. Returns
    (particles, weights, replan); replan is True when the path ran into a no-go zone.
    """
    waypoint = path[waypoint_index]
    handle = arlo.drive(Direction.FORWARD, DRIVE_SPEED_CMS, distance_cm=distance_cm, debug=False)
    applied = 0.0          # Distance already applied to the particles
    applied_turn = 0.0     # Heading change (rad) already applied to the particles
    replan = False
    last_publish = time.monotonic()
    try:
        while handle.is_alive():
            readings = sonars(arlo)
            if blocked(readings):
                print(f"Segment halted: sonar {readings}")
                break
            front = readings['front']
            seen_landmarks, obstacle_markers = observations(camera)
            # The camera measures the target directly; the pose estimate may lag behind
            nearest = min(front, seen_landmarks.get(target, (math.inf,))[0])
            if obstacles.marker_in_path(obstacle_markers.values()) is not None:
                print("Segment halted: obstacle marker in path")
                break

            # Only apply motion before a measurement: every motion update adds fixed noise.
            moved = handle.traveled - applied
            turned = math.radians(handle.turned) - applied_turn
            if seen_landmarks and moved >= MEASURE_MIN_CM:
                mcl.motion_update(particles, moved, turned)
                applied += moved
                applied_turn += turned
                particles, weights = update_localization(particles, weights, seen_landmarks)
                moved = turned = 0.0

            # The particles with the motion not yet applied to them, for steering and the map
            shown = predict(particles, moved, turned)
            if mcl.position_spread_cm(particles, weights) <= MAX_POSITION_SPREAD_CM:
                x, y, theta = mcl.pose_estimate(shown, weights)
                nearest = min(nearest, math.dist((x, y), POSITIONS[target]))
                record_trail(x, y)
                map_obstacles(obstacle_markers, x, y, theta)
                if obstacle_markers and path_blocked(path, waypoint_index, x, y):
                    print("Segment halted: path runs through a no-go zone")
                    replan = True
                    break
                bearing, distance = navigation.direction_to_target(x, y, theta, *waypoint)
                steer = 0.0
                if distance > STEER_MIN_DISTANCE_CM:
                    steer = max(-MAX_STEER_DEG_S,
                                min(MAX_STEER_DEG_S, STEER_GAIN * math.degrees(bearing)))
                handle.steer(steer)
            else:
                handle.steer(0.0)  # Do not steer on an unreliable pose
            if nearest <= APPROACH_DISTANCE_CM and handle.speed_limit_cms > APPROACH_SPEED_CMS:
                print(f"Segment slowing: {nearest:.0f} cm from {target} or an obstacle")
                handle.limit_speed(APPROACH_SPEED_CMS)

            if streamer is not None and time.monotonic() - last_publish >= MAP_PUBLISH_INTERVAL_S:
                publish_map(shown, weights, target, path)
                last_publish = time.monotonic()
    finally:
        if handle.is_alive():
            handle.halt("segment interrupted")
        mcl.motion_update(particles, handle.traveled - applied,
                          math.radians(handle.turned) - applied_turn)
    return particles, weights, replan


def initialize(arlo, camera):
    """Initialize from the starting landmark and another; restart after repositioning."""
    particles = weights = None
    seen_second = False
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
                publish_map(particles, weights)
                seen_second = seen_second or any(i != START_LANDMARK for i in seen_landmarks)
                spread = mcl.position_spread_cm(particles, weights)
                if seen_second and seen_landmarks and spread <= MAX_POSITION_SPREAD_CM:
                    print(f"Localization ready; spread={spread:.1f} cm")
                    return particles, weights
            turn(arlo, particles, SCAN_STEP_DEG)
            time.sleep(SCAN_SETTLE_S)

        print("Repositioning to improve localization")
        escape(arlo, None, None)
        particles = weights = None
        seen_second = False


def survey_leg(arlo, camera, particles, weights, target):
    """Face the target landmark and map the obstacles in view before planning the leg."""
    for _ in range(int(180 // MAX_TURN_STEP_DEG) + 2):
        x, y, theta = mcl.pose_estimate(particles, weights)
        bearing, _ = navigation.direction_to_target(x, y, theta, *POSITIONS[target])
        if abs(math.degrees(bearing)) <= TURN_TOLERANCE_DEG:
            break
        turn(arlo, particles, clamp_turn(math.degrees(bearing)))
        time.sleep(SCAN_SETTLE_S)
        seen_landmarks, _ = observations(camera)
        particles, weights = update_localization(particles, weights, seen_landmarks)
    for _ in range(SURVEY_FRAMES):
        seen_landmarks, obstacle_markers = observations(camera)
        particles, weights = update_localization(particles, weights, seen_landmarks)
        if mcl.position_spread_cm(particles, weights) <= MAX_POSITION_SPREAD_CM:
            map_obstacles(obstacle_markers, *mcl.pose_estimate(particles, weights))
    print(f"Surveyed leg to {target}: {len(obstacle_map)} obstacle(s) mapped")
    publish_map(particles, weights, target)
    return particles, weights


def visit_target(arlo, camera, particles, weights, target):
    """Survey the leg, then plan and follow an RRT path to the target, replanning
    whenever the rest of the path runs through a no-go zone."""
    bumps.clear()
    particles, weights = survey_leg(arlo, camera, particles, weights, target)
    path = None
    waypoint_index = 1
    while True:
        seen_landmarks, obstacle_markers = observations(camera)
        particles, weights = update_localization(particles, weights, seen_landmarks)
        x, y, theta = mcl.pose_estimate(particles, weights)
        spread = mcl.position_spread_cm(particles, weights)
        if spread <= MAX_POSITION_SPREAD_CM:  # Only trust the pose for mapping when converged
            record_trail(x, y)
            map_obstacles(obstacle_markers, x, y, theta)

        target_x, target_y = POSITIONS[target]
        target_distance = math.hypot(target_x - x, target_y - y)
        if target_distance <= TARGET_DISTANCE_CM and spread <= MAX_POSITION_SPREAD_CM:
            print(f"Reached {target}")
            return particles, weights

        if path is None or (spread <= MAX_POSITION_SPREAD_CM
                            and path_blocked(path, waypoint_index, x, y)):
            path = rrt.plan_path((x, y), (target_x, target_y), no_go_zones(),
                                 RRT_BOUNDS, goal_radius=GOAL_RADIUS_CM)
            if path is None:
                raise RuntimeError(f"RRT could not find a path to {target}")
            waypoint_index = 1
            print(f"RRT planned {len(path) - 1} segment(s) to {target}")

        waypoint = path[waypoint_index]
        _, waypoint_distance = navigation.direction_to_target(x, y, theta, *waypoint)
        if waypoint_index < len(path) - 1 and waypoint_distance <= WAYPOINT_DISTANCE_CM:
            waypoint_index += 1
            continue

        target_bearing, _ = navigation.direction_to_target(x, y, theta, *waypoint)
        publish_map(particles, weights, target, path)
        print(f"{target}: pose=({x:.0f}, {y:.0f}, {math.degrees(theta):.0f}°), "
              f"dist={target_distance:.0f} cm, spread={spread:.0f} cm, "
              f"RRT waypoint=({waypoint[0]:.0f}, {waypoint[1]:.0f})")

        if blocked(sonars(arlo)):
            escape(arlo, particles, weights)
            path = None
            continue

        if abs(math.degrees(target_bearing)) > TURN_TOLERANCE_DEG:
            turn(arlo, particles, clamp_turn(math.degrees(target_bearing)))
            continue

        marker_bearing = obstacles.marker_in_path(obstacle_markers.values())
        if marker_bearing is not None:  # Too close to an obstacle to plan around it
            escape(arlo, particles, weights, side=-1 if marker_bearing > 0 else 1)
            path = None
            continue

        particles, weights, replan = drive_segment(
            arlo, camera, particles, weights,
            segment_distance(waypoint_distance),
            target, path, waypoint_index,
        )
        if replan:
            path = None


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
