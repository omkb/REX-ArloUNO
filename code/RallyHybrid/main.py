"""DIKU RoboRally: visit L1, L2, L3, L4 and L1 again, fast and without touching anything.

The camera and MCL plan the route; the sonars are only the safety net.

  Localize  Turn on the spot until two landmarks fix the pose (MCL). Obstacle
            markers seen meanwhile go on the map.
  Plan      Face the next landmark, map the obstacles in view, and plan an RRT path
            (rrt.py) to any free point within REACH_CM of it. So a landmark with
            boxes in front of it is visited from the side or from behind. No-go zones
            are the mapped boxes, what the sonars found, and the other landmarks.
  Follow    Drive the path in arcs, updating MCL from landmarks seen on the way.
            Replan when a newly seen box blocks the rest of the path.

Every time the robot stands still it decides in this fixed order:
  1. Reached?  MCL or the camera puts us within REACH_CM of the target -> next one.
  2. Lost?     MCL is unsure -> turn on the spot until it is sure again.
  3. Sonar.    A close box the camera missed (at the start, or when MCL is off)
               -> escape(): turn away from it until there is room, drive ESCAPE_CM.
  4. Path.     Turn towards the next waypoint and drive to it.

What the sonars found goes on the map, so the next RRT path goes around it
instead of straight back into it. Nothing gives up: dead ends lead to another
escape or scan.

Run on the robot:  python3 -u main.py   and watch http://protoarlo.local:8000/
"""
import math
import time

import rrt
from geometry import absolute, relative
from hardware import Hardware
from localization import Localizer
from mapview import MapView

LANDMARKS = {1: (0, 0), 2: (0, 300), 3: (400, 0), 4: (400, 300)}  # ArUco id -> centre (x, y) cm
ROUTE = [1, 2, 3, 4, 1]
TRACK_MARGIN_CM = 120  # The track ends this far outside the landmarks

# Sizes in cm, between centres unless said otherwise
ROBOT_RADIUS_CM = 22
BOX_RADIUS_CM = 20
VISIT_CM = 40       # Race rule: a visit counts within this of the landmark box
TOUCH_CM = ROBOT_RADIUS_CM + BOX_RADIUS_CM  # Touching a box
REACH_CM = TOUCH_CM + VISIT_CM - 10         # Visited, with 10 cm to spare
MARGIN_CM = 18      # Room kept to boxes and to other landmarks' visit zones
TIGHT_MARGIN_CM = 8  # ... when there is no path with MARGIN_CM

# Sonar safety net (sonar to whatever it sees)
FRONT_STOP_CM = 25  # Stop for anything this close ahead (or behind, reversing)
SIDE_STOP_CM = 12   # ... or this close to the diagonal sonars
ESCAPE_CM = 60      # Drive this far away from what the sonars found
BACKUP_CM = 20

SPEED_CMS = 40
SLOW_SPEED_CMS = 20  # Within 100 cm of the target or of anything ahead
SCAN_STEP_DEG = 20
AIM_DEG = 12         # Turn on the spot first when the waypoint is further off than this

SONAR_BEARINGS = {'front': 0.0, 'left': math.radians(45), 'right': math.radians(-45)}
_xs, _ys = [p[0] for p in LANDMARKS.values()], [p[1] for p in LANDMARKS.values()]
_inset = TRACK_MARGIN_CM - ROBOT_RADIUS_CM  # How far outside the landmarks the robot centre may go
BOUNDS = (min(_xs) - _inset, max(_xs) + _inset, min(_ys) - _inset, max(_ys) + _inset)


def from_centre(distance, bearing):
    """Camera reading (front of robot to box face) -> (distance, bearing) centre to centre."""
    x = ROBOT_RADIUS_CM + (distance + BOX_RADIUS_CM) * math.cos(bearing)
    y = (distance + BOX_RADIUS_CM) * math.sin(bearing)
    return math.hypot(x, y), math.atan2(y, x)


def blocked(sonar):
    return any(cm < (SIDE_STOP_CM if name in ('left', 'right') else FRONT_STOP_CM)
               for name, cm in sonar.items())


def inside(point):
    x_min, x_max, y_min, y_max = BOUNDS
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


class Racer:
    def __init__(self, hardware, view=None):
        self.hw = hardware
        self.view = view
        self.localizer = Localizer(LANDMARKS)
        self.target = ROUTE[0]
        self.boxes = {}      # Obstacle marker id -> (x, y, sightings), box centres seen by the camera
        self.bumps = []      # (x, y) of boxes found by the sonars, this leg
        self.path = None     # RRT path [(x, y), ...] from where it was planned to the target
        self.waypoint = 1    # Index in path we are heading for
        self.margin = MARGIN_CM  # The margin the path was planned with
        # Boxes seen before MCL knows the pose, while turning on the spot: (id, distance,
        # bearing, self.turned at the time). Placed on the map once we know the pose.
        self.sightings = []
        self.turned = 0.0

    def run(self):
        start = time.monotonic()
        self.localize()
        for target in ROUTE:
            self.target = target
            self.log(f"--- Next target: L{target}")
            self.visit()
            self.log(f"Reached L{target} after {time.monotonic() - start:.0f} s")

    def visit(self):
        self.bumps, self.path = [], None
        self.survey()
        while True:
            markers = self.look()
            if self.reached(markers):
                return
            if not self.localizer.known():
                self.localize()
                continue
            sonar = self.hw.sonar('front', 'left', 'right')
            if blocked(sonar):
                self.escape(sonar)
                continue

            pose = self.localizer.pose()
            if self.path is None or self.path_blocked(pose):
                self.plan(pose)
            distance, bearing = relative(pose, self.path[self.waypoint])
            if distance < 20:
                if self.waypoint < len(self.path) - 1:
                    self.waypoint += 1
                    continue
                # At the end of the path, yet not reached: edge towards the landmark itself
                distance, bearing = relative(pose, LANDMARKS[self.target])
                distance -= REACH_CM - 10
            if abs(math.degrees(bearing)) > AIM_DEG:
                self.turn(math.degrees(bearing))
                continue
            stopped = self.drive(distance, on_path=True)
            if stopped == 'replan':
                self.path = None

    def survey(self):
        """Face the target and look a few times, to map the boxes on the way before planning."""
        if self.localizer.known():
            self.turn(math.degrees(relative(self.localizer.pose(), LANDMARKS[self.target])[1]))
            for _ in range(3):
                self.look()

    def reached(self, markers):
        if self.target in markers and from_centre(*markers[self.target])[0] <= REACH_CM:
            return True
        return (self.localizer.known()
                and relative(self.localizer.pose(), LANDMARKS[self.target])[0] <= REACH_CM)

    # --- Planning
    def zones(self, margin=MARGIN_CM):
        """No-go circles (x, y, radius) for the robot centre."""
        zones = [(x, y, TOUCH_CM + margin) for x, y, _ in self.boxes.values()]
        zones += [(x, y, TOUCH_CM + margin) for x, y in self.bumps]
        zones += [(*point, TOUCH_CM + margin + (0 if i == self.target else VISIT_CM))
                  for i, point in LANDMARKS.items()]
        return zones

    def plan(self, pose):
        """RRT path to any free point within REACH_CM of the target. rrt allows starting
        inside a zone (e.g. the landmark we just visited) as long as we head out of it.
        If that fails, try with a tight margin, then also without the zones we are in."""
        for self.margin, leave in ((MARGIN_CM, True), (TIGHT_MARGIN_CM, True), (TIGHT_MARGIN_CM, False)):
            zones = [(x, y, r) for x, y, r in self.zones(self.margin)
                     if leave or math.dist(pose[:2], (x, y)) > r]
            path = rrt.plan_path(pose[:2], LANDMARKS[self.target], zones, BOUNDS,
                                 goal_radius=REACH_CM - 5)
            if path is not None:
                break
        else:  # Should not happen; the sonars still watch
            path = [pose[:2], LANDMARKS[self.target]]
        self.path, self.waypoint = path, 1
        self.log(f"Planned {len(path) - 1} segment(s) to L{self.target} with margin {self.margin}")

    def path_blocked(self, pose):
        """True when the rest of the path, or the line from us to the waypoint, crosses a
        no-go zone. Boxes are mapped after planning, and later sightings move them.
        Zones the path started in do not count; plan() already dealt with those."""
        zones = [(x, y, r) for x, y, r in self.zones(self.margin)
                 if math.dist(self.path[0], (x, y)) > r]
        if rrt.path_in_collision(self.path[self.waypoint - 1:], zones):
            return True
        # From inside a zone every line is blocked; leaving it is the planned segment's job
        return (not any(math.dist(pose[:2], (x, y)) <= r for x, y, r in zones)
                and rrt.path_in_collision([pose[:2], self.path[self.waypoint]], zones))

    # --- Sonar safety net
    def escape(self, sonar):
        """Something close that the camera missed: turn away from it, to the side with the
        most room, in 45 degree steps until there is room ahead, then drive ESCAPE_CM.
        Also used to look from elsewhere when localizing fails."""
        if blocked(sonar):
            self.remember_bump(sonar)
            self.log(f"Escaping: {sonar}")
            if sonar['front'] < FRONT_STOP_CM and min(sonar['left'], sonar['right']) < SIDE_STOP_CM:
                self.drive(-BACKUP_CM)
        side = 1 if sonar['left'] > sonar['right'] else -1
        for _ in range(8):
            if self.room_ahead():
                self.drive(ESCAPE_CM)
                self.path = None
                return
            self.turn(side * 45)
        self.drive(-BACKUP_CM)  # No room anywhere: back up and look again

    def room_ahead(self):
        """Room to drive ESCAPE_CM ahead: by the sonars, and by the map if we know where we
        are (the end must be outside the no-go zones, except those we are in already)."""
        sonar = self.hw.sonar('front', 'left', 'right')
        if sonar['front'] < ESCAPE_CM + FRONT_STOP_CM or blocked(sonar):
            return False
        if not self.localizer.known():
            return True
        pose = self.localizer.pose()
        end = absolute(pose, ESCAPE_CM, 0.0)
        return inside(end) and all(math.dist(end, (x, y)) > r or math.dist(pose[:2], (x, y)) <= r
                                   for x, y, r in self.zones(TIGHT_MARGIN_CM))

    def remember_bump(self, sonar):
        """Put what the closest sonar sees on the map, unless it is a landmark or a box
        the camera has mapped already."""
        if not self.localizer.known():
            return
        name = min(sonar, key=sonar.get)
        bump = absolute(self.localizer.pose(), ROBOT_RADIUS_CM + sonar[name] + BOX_RADIUS_CM,
                        SONAR_BEARINGS[name])
        known = list(LANDMARKS.values()) + [(x, y) for x, y, _ in self.boxes.values()]
        if all(math.dist(bump, point) > TOUCH_CM for point in known):
            self.bumps.append(bump)

    # --- Sensing
    def localize(self):
        """Turn on the spot until MCL knows where we are; if a full turn is not enough,
        move somewhere else and try again."""
        self.log("Localizing")
        while True:
            for _ in range(360 // SCAN_STEP_DEG):
                self.look()
                if self.localizer.known():
                    self.survey()
                    return
                self.turn(SCAN_STEP_DEG)
                time.sleep(0.25)  # Let the camera deliver a sharp picture
            self.log("No pose after a full turn; looking from elsewhere")
            self.escape(self.hw.sonar('front', 'left', 'right'))

    def look(self):
        """Take a picture: landmarks update MCL, other markers go on the map."""
        markers = self.hw.markers()
        seen = {i: from_centre(*m) for i, m in markers.items()}
        self.localizer.observe(seen)
        boxes = [(i, *s) for i, s in seen.items() if i not in LANDMARKS]
        if self.localizer.known():
            pose = self.localizer.pose()
            for i, distance, bearing, turned in self.sightings:
                self.map_box(i, absolute(pose, distance, bearing - (self.turned - turned)))
            self.sightings = []
            for i, distance, bearing in boxes:
                self.map_box(i, absolute(pose, distance, bearing))
        else:
            self.sightings += [(*box, self.turned) for box in boxes]
        self.show()
        return markers

    def map_box(self, i, point):
        if i not in self.boxes:
            self.log(f"Mapped box {i} at ({point[0]:.0f}, {point[1]:.0f})")
            self.boxes[i] = (*point, 1)
            return
        x, y, n = self.boxes[i]
        n = min(n + 1, 10)  # Average the sightings, but let new ones still move it
        self.boxes[i] = (x + (point[0] - x) / n, y + (point[1] - y) / n, n)

    # --- Moving. Everything goes through turn() and drive(), so MCL knows how we moved.
    def turn(self, degrees):
        self.hw.turn(degrees)
        self.localizer.moved(turn_rad=math.radians(degrees))
        self.turned += math.radians(degrees)

    def drive(self, distance_cm, on_path=False):
        """Drive (negative: backwards) and return why it stopped early, or None.

        The sonars in the driving direction can stop it at any moment ('sonar'). On the
        path it also looks while driving: it steers in an arc towards the waypoint,
        slows down near the target, stops there ('reached'), and stops when a newly
        seen box blocks the path ('replan').
        """
        sonars = ('front', 'left', 'right') if distance_cm > 0 else ('back',)
        if blocked(self.hw.sonar(*sonars)):
            return 'sonar'
        handle = self.hw.drive(distance_cm, SPEED_CMS)
        done = [0.0, 0.0]  # cm and degrees already given to MCL
        reason = None
        try:
            while reason is None and handle.is_alive():
                sonar = self.hw.sonar(*sonars)
                if blocked(sonar):
                    reason = 'sonar'
                elif on_path:
                    reason = self.follow(handle, done, sonar['front'])
        finally:
            if handle.is_alive():
                handle.halt(reason or 'interrupted')
                handle.wait()
            self.account(handle, done, math.copysign(1, distance_cm))
        if reason not in (None, 'reached'):
            self.log(f"Drive stopped: {reason}")
        return reason

    def follow(self, handle, done, front_cm):
        """One look while driving along the path."""
        if handle.traveled - done[0] >= 10:  # Not every time: each update adds noise
            self.account(handle, done)
        markers = self.look()
        if self.reached(markers):
            return 'reached'
        if not self.localizer.known():
            handle.steer(0.0)  # Do not steer on a pose we do not trust
            return None
        pose = self.localizer.pose()
        if self.path_blocked(pose):
            return 'replan'
        distance, bearing = relative(pose, self.path[self.waypoint])
        # 20 deg/s at 40 cm/s is a 115 cm radius. Close to the waypoint the bearing swings.
        handle.steer(max(-20.0, min(20.0, math.degrees(bearing))) if distance > 40 else 0.0)
        target_distance = relative(pose, LANDMARKS[self.target])[0]
        if min(target_distance - REACH_CM, front_cm) < 100:
            handle.limit_speed(SLOW_SPEED_CMS)
        return None

    def account(self, handle, done, sign=1.0):
        """Give MCL the motion of the drive so far that it has not had yet."""
        cm, degrees = handle.traveled - done[0], handle.turned - done[1]
        done[0], done[1] = handle.traveled, handle.turned
        self.localizer.moved(sign * cm, math.radians(degrees))
        self.turned += math.radians(degrees)
        if cm > 0:
            self.sightings = []  # They were seen from somewhere else

    def log(self, message):
        print(message)
        self.show(message, force=True)

    def show(self, status='', force=False):
        if self.view is not None:
            self.view.draw(self, status, force)


def main():
    view = MapView(LANDMARKS, BOUNDS)  # Start the stream first, so the browser can connect
    input("Point the robot at L1 and press Enter to start the race... ")
    hardware = Hardware()
    try:
        Racer(hardware, view).run()
        print("Race complete")
        input("Press Enter to stop the map stream... ")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        hardware.close()
        view.close()


if __name__ == '__main__':
    main()
