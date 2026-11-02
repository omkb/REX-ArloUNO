"""RoboRally: visit the landmarks L1 -> L2 -> L3 -> L4 -> L1 without touching any box.

The robot tracks its pose with a particle filter (MCL, mcl.py), updated from the
landmarks the camera sees. Every other ArUco marker is a box: boxes are put on a map
and avoided by planning an RRT path (rrt.py) around them.

  1. Localize   Turn on the spot until L1 and one more landmark give a sure pose.
  2. For each landmark on the route:
       Survey   Face the landmark and map the boxes in view.
       Plan     An RRT path keeping MAP_CLEARANCE_CM from every mapped box.
       Drive    Follow the path in smooth arcs, updating MCL and the map on the way.
                Replan when a newly mapped box blocks the path.
  3. Safety nets, checked all the time:
       Sonar    Something too close: escape (turn towards room, drive ESCAPE_CM) and
                remember it as a box ("bump") for the rest of the leg.
       Spread   Pose too uncertain: stop and relocalize (turn until a landmark).

Run on the robot:  python3 -u main2.py --run
Live map:          http://<robot>:8000/
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

# --- Course (cm). Landmark name -> (ArUco marker id, position of the marker)
LANDMARKS = {
    'L1': (1, (0.0, 0.0)),
    'L2': (2, (0.0, 300.0)),
    'L3': (3, (400.0, 0.0)),
    'L4': (4, (400.0, 300.0)),
}
LANDMARK_BY_MARKER = {marker_id: name for name, (marker_id, _) in LANDMARKS.items()}
POSITIONS = {name: position for name, (_, position) in LANDMARKS.items()}
START_LANDMARK = 'L1'
ROUTE = ['L1', 'L2', 'L3', 'L4', 'L1']

COURT_WIDTH_CM = 400.0
COURT_HEIGHT_CM = 300.0
COURT_MARGIN_CM = 90.0       # How far outside the landmarks the robot may drive
BOUNDS = (-COURT_MARGIN_CM, COURT_WIDTH_CM + COURT_MARGIN_CM,
          -COURT_MARGIN_CM, COURT_HEIGHT_CM + COURT_MARGIN_CM)

# --- Localization
MAX_SPREAD_CM = 55.0         # Trust the pose only while the particle spread is below this
SCAN_STEP_DEG = 10           # Turn step while scanning for landmarks
SCAN_SETTLE_S = 0.25         # Let the camera settle after a turn
MEASURE_MIN_CM = 10.0        # While driving, measure landmarks at most once per this distance

# --- Visiting a landmark
TARGET_DISTANCE_CM = 60.0    # Reached when the pose is this close to the landmark
GOAL_RADIUS_CM = TARGET_DISTANCE_CM / 2  # RRT paths end this far from the landmark
SURVEY_FRAMES = 3            # Frames taken facing the next landmark before planning
RRT_MAX_ITERATIONS = 2000    # A failing plan takes ~40 s on the Pi at 5000

# --- Driving
DRIVE_SPEED_CMS = 40.0
APPROACH_SPEED_CMS = 20.0    # Slow down to this ...
APPROACH_DISTANCE_CM = 100.0  # ... this close to the target or anything ahead
TURN_SPEED_DEG_S = 50.0
TURN_TOLERANCE_DEG = 12.0    # Turn towards the waypoint first if it is further off than this
MIN_SEGMENT_CM = 30.0        # Shortest drive towards a waypoint
WAYPOINT_DISTANCE_CM = 20.0  # Move on to the next waypoint this close to it
STEER_GAIN = 1.0             # Turn rate (deg/s) per degree off the waypoint while driving
MAX_STEER_DEG_S = 20.0       # Sharpest arc: 20 deg/s at 40 cm/s is a ~115 cm radius
STEER_MIN_DISTANCE_CM = 40.0  # Drive straight this close to the waypoint

# --- Sonar safety net
FRONT_STOP_CM = 25           # Stop for anything this close ahead (or behind, reversing)
SIDE_STOP_CM = 12            # ... or this close to the diagonal sonars
ESCAPE_CM = 60               # Drive this far away from what the sonars found
ESCAPE_TURN_DEG = 45
BACKUP_CM = 20
ROBOT_RADIUS_CM = 22.0
TOUCH_CM = 42.0              # No bump this close to a landmark or a mapped box
SONAR_BEARINGS = {'front': 0.0, 'left': math.radians(45), 'right': math.radians(-45)}
LEFT, RIGHT = 1, -1

# --- Live map
STREAM_PORT = 8000
MAP_MARGIN_PX = 100          # 1 px per cm, plus a margin around the landmarks
MAP_MAX_DRAWN = 200          # Drawing all particles is slow on the Pi
MAP_PUBLISH_INTERVAL_S = 1.0  # While driving; each publish pauses sensing briefly
MAP_MAX_SIGHTINGS = 10       # Later sightings still move a mapped box by at least 1/10
TRAIL_MIN_STEP_CM = 5.0


def blocked(readings):
    """True when a sonar reading ({name: cm}) is too close."""
    return any(cm < (SIDE_STOP_CM if name in ('left', 'right') else FRONT_STOP_CM)
               for name, cm in readings.items())


def goal_reachable(goal, zones):
    """True when some point within GOAL_RADIUS_CM of the goal is outside every zone."""
    for radius in (0.0, GOAL_RADIUS_CM / 2, GOAL_RADIUS_CM):
        for angle in np.radians(np.arange(0, 360, 10)):
            point = (goal[0] + radius * math.cos(angle), goal[1] + radius * math.sin(angle))
            if all(math.dist(point, (x, y)) > r for x, y, r in zones):
                return True
    return False


class Racer:
    def __init__(self, arlo, camera, streamer=None):
        self.arlo = arlo
        self.camera = camera
        self.streamer = streamer
        self.particles = None  # MCL particles (x, y, theta); None before localization
        self.weights = None
        self.boxes = {}        # Marker id -> (x, y, sightings): box centres seen by the camera
        self.bumps = []        # (x, y): boxes found by the sonars, this leg only
        self.trail = []        # Pose estimates along the route
        self.relaxed = None    # (target, reach): ignore no-go zones this close to the target

    # ------------------------------------------------------------------ Route
    def run(self):
        self.localize()
        for target in ROUTE:
            print(f"Next target: {target} (marker {LANDMARKS[target][0]})")
            self.visit(target)
        print("Route finished")

    def visit(self, target):
        """Survey the leg, then plan and follow an RRT path to the target."""
        self.bumps.clear()
        self.survey(target)
        path, waypoint = None, 1
        while True:
            landmarks, boxes = self.look()
            self.measure(landmarks)
            if not self.localized():
                self.relocalize(target)
                path = None
                continue
            x, y, theta = self.pose()
            self.record_trail(x, y)
            self.map_boxes(boxes, (x, y, theta))

            distance = math.dist((x, y), POSITIONS[target])
            if distance <= TARGET_DISTANCE_CM:
                print(f"Reached {target}")
                return

            if path is None or self.path_blocked(path, waypoint, x, y):
                path, waypoint = self.plan((x, y), target), 1
                if path is None:
                    print(f"RRT could not find a path to {target}; moving and trying again")
                    self.escape()
                    continue
                print(f"RRT planned {len(path) - 1} segment(s) to {target}")

            bearing, waypoint_distance = navigation.direction_to_target(x, y, theta, *path[waypoint])
            if waypoint < len(path) - 1 and waypoint_distance <= WAYPOINT_DISTANCE_CM:
                waypoint += 1
                continue

            self.publish(target, path)
            print(f"{target}: pose=({x:.0f}, {y:.0f}, {math.degrees(theta):.0f}°), "
                  f"dist={distance:.0f} cm, spread={self.spread():.0f} cm, "
                  f"waypoint=({path[waypoint][0]:.0f}, {path[waypoint][1]:.0f})")

            if blocked(self.sonars()):
                self.escape()
                path = None
            elif abs(math.degrees(bearing)) > TURN_TOLERANCE_DEG:
                self.turn(math.degrees(bearing))
            elif (box_bearing := obstacles.marker_in_path(boxes.values())) is not None:
                # Too close to a box to plan around it: escape away from it
                self.escape(side=RIGHT if box_bearing > 0 else LEFT)
                path = None
            elif self.drive_segment(target, path, waypoint,
                                    max(MIN_SEGMENT_CM, waypoint_distance)):
                path = None

    def drive_segment(self, target, path, waypoint, distance_cm):
        """Drive distance_cm without stopping, steering in an arc towards path[waypoint].
        Returns True when a newly mapped box blocks the path, so it must be replanned.

        Halts for a sonar too close, a box marker right ahead or a too large spread, and
        slows down near the target or anything ahead. The driven motion is only given to
        MCL right before a landmark measurement, since every motion update adds noise;
        `now` is a copy with all the motion so far, for the spread, steering and the map.
        """
        handle = self.arlo.drive(Direction.FORWARD, DRIVE_SPEED_CMS,
                                 distance_cm=distance_cm, debug=False)
        applied_cm = applied_rad = 0.0  # Motion already given to the particles
        last_publish = time.monotonic()
        try:
            while handle.is_alive():
                readings = self.sonars()
                if blocked(readings):
                    print(f"Segment halted: sonar {readings}")
                    return False
                landmarks, boxes = self.look()
                if obstacles.marker_in_path(boxes.values()) is not None:
                    print("Segment halted: box marker in path")
                    return False

                moved = handle.traveled - applied_cm
                turned = math.radians(handle.turned) - applied_rad
                if landmarks and moved >= MEASURE_MIN_CM:
                    mcl.motion_update(self.particles, moved, turned)
                    applied_cm, applied_rad = applied_cm + moved, applied_rad + turned
                    moved = turned = 0.0
                    self.measure(landmarks)

                now = self.particles.copy()
                mcl.motion_update(now, moved, turned)
                if self.spread(now) > MAX_SPREAD_CM:
                    print("Segment halted: position spread too large")
                    return False
                x, y, theta = mcl.pose_estimate(now, self.weights)
                self.record_trail(x, y)
                self.map_boxes(boxes, (x, y, theta))
                if boxes and self.path_blocked(path, waypoint, x, y):
                    print("Segment halted: path runs through a no-go zone")
                    return True

                bearing, distance = navigation.direction_to_target(x, y, theta, *path[waypoint])
                steer = STEER_GAIN * math.degrees(bearing) if distance > STEER_MIN_DISTANCE_CM else 0.0
                handle.steer(max(-MAX_STEER_DEG_S, min(MAX_STEER_DEG_S, steer)))

                # The camera measures the target directly; the pose may lag behind
                nearest = min(readings['front'], landmarks.get(target, (math.inf,))[0],
                              math.dist((x, y), POSITIONS[target]))
                if nearest <= APPROACH_DISTANCE_CM and handle.speed_limit_cms > APPROACH_SPEED_CMS:
                    print(f"Segment slowing: {nearest:.0f} cm from {target} or an obstacle")
                    handle.limit_speed(APPROACH_SPEED_CMS)

                if time.monotonic() - last_publish >= MAP_PUBLISH_INTERVAL_S:
                    self.publish(target, path, now)
                    last_publish = time.monotonic()
        finally:
            if handle.is_alive():
                handle.halt("segment interrupted")
            mcl.motion_update(self.particles, handle.traveled - applied_cm,
                              math.radians(handle.turned) - applied_rad)
        return False

    # ----------------------------------------------------------- Localization
    def localize(self):
        """Turn on the spot until MCL is sure from L1 plus one more landmark; MCL starts at
        the first sighting of L1. After a full turn without that, move and start over."""
        print("Localizing")
        while True:
            self.particles = self.weights = None
            seen_second = False
            for _ in range(360 // SCAN_STEP_DEG):
                landmarks, _ = self.look()
                print(f"Detected landmarks: {list(landmarks)}")
                if self.particles is None and START_LANDMARK in landmarks:
                    self.particles, self.weights = mcl.initialize_from_landmark(
                        POSITIONS, START_LANDMARK, *landmarks.pop(START_LANDMARK))
                if self.particles is not None:
                    self.measure(landmarks)
                    seen_second |= any(name != START_LANDMARK for name in landmarks)
                self.publish()
                if seen_second and landmarks and self.localized():
                    print(f"Localization ready; spread={self.spread():.1f} cm")
                    return
                self.turn(SCAN_STEP_DEG)
                time.sleep(SCAN_SETTLE_S)
            print("Repositioning to improve localization")
            self.particles = self.weights = None
            self.escape()

    def relocalize(self, target):
        """Turn on the spot until a landmark brings the spread under MAX_SPREAD_CM.
        After a full turn without that, move elsewhere and turn again."""
        print(f"Relocalizing; spread={self.spread():.0f} cm")
        while True:
            for _ in range(360 // SCAN_STEP_DEG):
                self.turn(SCAN_STEP_DEG)  # Turn first: the current view was used already
                time.sleep(SCAN_SETTLE_S)
                landmarks, _ = self.look()
                self.measure(landmarks)
                self.publish(target)
                if landmarks and self.localized():
                    print(f"Relocalized from {list(landmarks)}; spread={self.spread():.0f} cm")
                    return
            print("Relocalizing: no good landmark in a full turn; moving")
            self.escape()

    def measure(self, landmarks):
        """Weight the particles by the seen landmarks and resample."""
        if mcl.measurement_update(self.particles, self.weights, landmarks, POSITIONS):
            self.particles, self.weights = mcl.resample(self.particles, self.weights)

    def spread(self, particles=None):
        return mcl.position_spread_cm(self.particles if particles is None else particles,
                                      self.weights)

    def localized(self):
        return self.particles is not None and self.spread() <= MAX_SPREAD_CM

    def pose(self):
        return mcl.pose_estimate(self.particles, self.weights)

    # ------------------------------------------------------- Mapping, planning
    def survey(self, target):
        """Face the target landmark and map the boxes in view before planning the leg."""
        for _ in range(3):  # Turns are open loop: correct the heading up to three times
            x, y, theta = self.pose()
            bearing, _ = navigation.direction_to_target(x, y, theta, *POSITIONS[target])
            if abs(math.degrees(bearing)) <= TURN_TOLERANCE_DEG:
                break
            self.turn(math.degrees(bearing))
            time.sleep(SCAN_SETTLE_S)
            self.measure(self.look()[0])
        for _ in range(SURVEY_FRAMES):
            landmarks, boxes = self.look()
            self.measure(landmarks)
            if self.localized():
                self.map_boxes(boxes, self.pose())
        print(f"Surveyed leg to {target}: {len(self.boxes)} box(es) mapped")
        self.publish(target)

    def map_boxes(self, boxes, pose):
        """Put the seen box markers on the map, averaging repeated sightings."""
        x, y, theta = pose
        for marker_id, (distance, bearing) in boxes.items():
            reach = distance + obstacles.OBSTACLE_RADIUS_CM  # Marker face -> box centre
            bx = x + reach * math.cos(theta + bearing)
            by = y + reach * math.sin(theta + bearing)
            if marker_id in self.boxes:
                old_x, old_y, n = self.boxes[marker_id]
                n = min(n + 1, MAP_MAX_SIGHTINGS)
                bx, by = old_x + (bx - old_x) / n, old_y + (by - old_y) / n
            else:
                n = 1
                print(f"Mapped box {marker_id} at ({bx:.0f}, {by:.0f})")
            self.boxes[marker_id] = (bx, by, n)

    def remember_bump(self, readings):
        """Put what the closest sonar sees on the map, unless it is a landmark or a box
        the camera has mapped already."""
        if not self.localized():
            return
        x, y, theta = self.pose()
        name = min(readings, key=readings.get)
        reach = ROBOT_RADIUS_CM + readings[name] + obstacles.OBSTACLE_RADIUS_CM
        bump = (x + reach * math.cos(theta + SONAR_BEARINGS[name]),
                y + reach * math.sin(theta + SONAR_BEARINGS[name]))
        known = list(POSITIONS.values()) + [(bx, by) for bx, by, _ in self.boxes.values()]
        if all(math.dist(bump, point) > TOUCH_CM for point in known):
            print(f"Sonar bump at ({bump[0]:.0f}, {bump[1]:.0f})")
            self.bumps.append(bump)

    def record_trail(self, x, y):
        if not self.trail or math.dist((x, y), self.trail[-1]) >= TRAIL_MIN_STEP_CM:
            self.trail.append((x, y))

    def no_go_zones(self):
        """Boxes and bumps as (x, y, radius) circles the robot centre must stay out of."""
        centres = [(x, y) for x, y, _ in self.boxes.values()] + self.bumps
        zones = [(x, y, obstacles.MAP_CLEARANCE_CM) for x, y in centres]
        if self.relaxed is not None:
            target, reach = self.relaxed
            zones = [(x, y, r) for x, y, r in zones
                     if math.dist(POSITIONS[target], (x, y)) > r + reach]
        return zones

    def plan(self, start, target):
        """RRT path to the target, or None. When the no-go zones cover the whole goal area
        (a box mapped next to the landmark), ignore the zones covering the landmark, then
        those touching the goal area; the sonars still guard the last stretch."""
        goal = POSITIONS[target]
        self.relaxed = None
        count = len(self.no_go_zones())
        if not goal_reachable(goal, self.no_go_zones()):
            self.relaxed = (target, 0.0)
            if not goal_reachable(goal, self.no_go_zones()):
                self.relaxed = (target, GOAL_RADIUS_CM)
            print(f"Ignoring {count - len(self.no_go_zones())} no-go zone(s) covering {target}")
        return rrt.plan_path(start, goal, self.no_go_zones(), BOUNDS,
                             goal_radius=GOAL_RADIUS_CM, max_iterations=RRT_MAX_ITERATIONS)

    def path_blocked(self, path, waypoint, x, y):
        """True when the rest of the path, or the line from the robot to its waypoint,
        crosses a no-go zone. From inside a zone every line is blocked; leaving it is the
        planned segment's job, so then only the planned segments count."""
        zones = self.no_go_zones()
        if rrt.path_in_collision(path[waypoint - 1:], zones):
            return True
        inside_zone = any(math.dist((x, y), (zx, zy)) <= r for zx, zy, r in zones)
        return not inside_zone and rrt.path_in_collision([(x, y), path[waypoint]], zones)

    # ------------------------------------------------------------ Sonar escape
    def escape(self, side=None):
        """Get away from something close: back up if boxed in, then turn (towards the side
        with the most room, or the given side) in 45 degree steps until there is room
        ahead, and drive ESCAPE_CM. Also used to look from elsewhere when lost."""
        readings = self.sonars()
        if blocked(readings):
            self.remember_bump(readings)
            print(f"Escaping: {readings}")
            if readings['front'] < FRONT_STOP_CM and min(readings['left'], readings['right']) < SIDE_STOP_CM:
                self.drive_straight(-BACKUP_CM)
        if side is None:
            side = LEFT if readings['left'] > readings['right'] else RIGHT
        for _ in range(360 // ESCAPE_TURN_DEG):
            if self.room_ahead():
                self.drive_straight(ESCAPE_CM)
                return
            self.turn(side * ESCAPE_TURN_DEG)
        self.drive_straight(-BACKUP_CM)  # No room anywhere: back up and look again

    def room_ahead(self):
        """Room to drive ESCAPE_CM ahead by the sonars and, when localized, by the map: the
        end must be inside the bounds and outside the no-go zones we are not in already."""
        readings = self.sonars()
        if readings['front'] < ESCAPE_CM + FRONT_STOP_CM or blocked(readings):
            return False
        if not self.localized():
            return True
        x, y, theta = self.pose()
        end = (x + ESCAPE_CM * math.cos(theta), y + ESCAPE_CM * math.sin(theta))
        x_min, x_max, y_min, y_max = BOUNDS
        return (x_min <= end[0] <= x_max and y_min <= end[1] <= y_max
                and all(math.dist(end, (zx, zy)) > r or math.dist((x, y), (zx, zy)) <= r
                        for zx, zy, r in self.no_go_zones()))

    # ------------------------------------------------------- Sensors and motors
    def look(self):
        """One camera frame -> ({landmark name: (cm, bearing)}, {box marker id: (cm, bearing)}).
        Bearings are in radians, positive to the left; the nearest of duplicate markers wins."""
        ids, distances, angles = self.camera.detect_aruco_objects(self.camera.get_next_frame())
        landmarks, boxes = {}, {}
        if ids is None:
            return landmarks, boxes
        for marker_id, distance, angle in zip(ids, distances, angles):
            marker_id = int(marker_id)
            name = LANDMARK_BY_MARKER.get(marker_id)
            seen, key = (landmarks, name) if name is not None else (boxes, marker_id)
            if key not in seen or distance < seen[key][0]:
                seen[key] = (float(distance), float(angle))
        return landmarks, boxes

    def sonars(self, *names):
        """{name: cm} for the given sonars, by default the front and both diagonals.
        A sonar that fails twice reads 0, so it counts as blocked."""
        return {name: self.ping(name) or self.ping(name) or 0.0
                for name in names or ('front', 'left', 'right')}

    def ping(self, name):
        """One sonar reading in cm, or None when invalid."""
        read = getattr(self.arlo.robot, f"read_{name}_ping_sensor")
        with self.arlo.serial_lock:  # A drive thread may be sending motor commands
            value = read()
        return value / 10 if value is not None and value > 0 else None

    def turn(self, degrees):
        """Turn on the spot, positive to the left (MCL's convention)."""
        if abs(degrees) < 0.01:
            return
        self.arlo.turn(-degrees, speed_degs=TURN_SPEED_DEG_S, debug=False).wait()  # Arlo: + is right
        if self.particles is not None:
            mcl.motion_update(self.particles, delta_rot_rad=math.radians(degrees))

    def drive_straight(self, distance_cm):
        """Drive straight (negative is backwards), halting if the sonars in the driving
        direction find something close."""
        forward = distance_cm > 0
        handle = self.arlo.drive(Direction.FORWARD if forward else Direction.BACKWARD,
                                 DRIVE_SPEED_CMS, distance_cm=abs(distance_cm), debug=False)
        try:
            while handle.is_alive():
                if blocked(self.sonars() if forward else self.sonars('back')):
                    print("Drive halted: sonar")
                    break
        finally:
            if handle.is_alive():
                handle.halt("sonar")
        if self.particles is not None:
            mcl.motion_update(self.particles, math.copysign(handle.traveled, distance_cm),
                              math.radians(handle.turned))

    # ---------------------------------------------------------------- Live map
    def publish(self, target=None, path=None, particles=None):
        if self.streamer is not None:
            particles = self.particles if particles is None else particles
            self.streamer.publish(self.draw_map(particles, target, path))

    def draw_map(self, particles, target=None, path=None):
        """Landmarks, boxes, bumps, trail, particles, pose and RRT path (BGR, y-axis up)."""
        xs = [x for x, _ in POSITIONS.values()]
        ys = [y for _, y in POSITIONS.values()]
        min_x, min_y = min(xs) - MAP_MARGIN_PX, min(ys) - MAP_MARGIN_PX
        width = int(max(xs) - min_x + MAP_MARGIN_PX)
        height = int(max(ys) - min_y + MAP_MARGIN_PX)
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        def px(x, y):
            return int(x - min_x), int(height - (y - min_y))

        def text(label, at, scale, colour):
            cv2.putText(image, label, at, font, scale, colour, 1, cv2.LINE_AA)

        for name, (marker_id, (lx, ly)) in LANDMARKS.items():
            colour = (0, 0, 255) if name == target else (0, 160, 0)
            cx, cy = px(lx, ly)
            cv2.circle(image, (cx, cy), 8, colour, 2)
            text(f"{name} ({marker_id})", (cx + 10, cy - 10), 0.6, colour)

        for marker_id, (bx, by, _) in self.boxes.items():
            cx, cy = px(bx, by)
            cv2.circle(image, (cx, cy), int(obstacles.OBSTACLE_RADIUS_CM), (90, 90, 90), -1)
            cv2.circle(image, (cx, cy), int(obstacles.MAP_CLEARANCE_CM), (170, 170, 170), 1)
            text(str(marker_id), (cx - 8, cy + 5), 0.45, (255, 255, 255))

        for bx, by in self.bumps:
            centre = px(bx, by)
            cv2.circle(image, centre, int(obstacles.OBSTACLE_RADIUS_CM), (0, 0, 160), -1)
            cv2.circle(image, centre, int(obstacles.MAP_CLEARANCE_CM), (150, 150, 230), 1)

        if len(self.trail) > 1:
            points = np.array([px(*point) for point in self.trail], dtype=np.int32)
            cv2.polylines(image, [points], False, (0, 140, 255), 2, cv2.LINE_AA)

        if particles is None:
            return image

        drawn = particles
        if len(particles) > MAP_MAX_DRAWN:
            drawn = particles[np.random.choice(len(particles), MAP_MAX_DRAWN, replace=False)]
        for x, y, theta in drawn:
            cv2.circle(image, px(x, y), 2, (200, 120, 0), -1)
            cv2.line(image, px(x, y), px(x + 10 * math.cos(theta), y + 10 * math.sin(theta)),
                     (200, 120, 0), 1)

        x, y, theta = mcl.pose_estimate(particles, self.weights)
        cv2.circle(image, px(x, y), 6, (0, 0, 0), 2)
        cv2.line(image, px(x, y), px(x + 25 * math.cos(theta), y + 25 * math.sin(theta)),
                 (0, 0, 0), 2)
        if path:
            points = np.array([px(*point) for point in path], dtype=np.int32)
            cv2.polylines(image, [points], False, (200, 0, 200), 2, cv2.LINE_AA)
            for point in path[1:-1]:
                cv2.drawMarker(image, px(*point), (200, 0, 200), cv2.MARKER_CROSS, 12, 2)
        text(f"({x:.0f}, {y:.0f}, {math.degrees(theta):.0f} deg) "
             f"spread={self.spread(particles):.0f} cm", (10, 20), 0.5, (0, 0, 0))
        return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='Enable real robot motion')
    if not parser.parse_args().run:
        print('DRY RUN: nothing moved. Check LANDMARKS and the constants, then run:')
        print('  python3 -u main2.py --run')
        return

    input(f'Clear the area, point the robot at {START_LANDMARK}, then press Enter to start...')
    arlo = Arlo()
    streamer = mapstream.MapStreamer(STREAM_PORT)
    camera = None
    try:
        camera = Camera(0, robottype='arlo', useCaptureThread=False)
        Racer(arlo, camera, streamer).run()
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        arlo.stop()
        if camera is not None:
            camera.terminateCaptureThread()
            capture = camera.get_capture()
            capture.close() if hasattr(capture, 'close') else capture.release()
        streamer.close()


if __name__ == '__main__':
    main()
