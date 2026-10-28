"""DIKU RoboRally: visit L1, L2, L3, L4 and L1 again, fast and without touching anything.

Before the race the robot turns on the spot until it has seen two landmarks and
knows where it is (localize). Obstacles it sees meanwhile go on the map.

Then, every time the robot stands still, it decides in this fixed order:

  1. Reached?  The camera sees the target within VISIT_CM, or MCL puts us on the
               ring around it -> go for the next one.
  2. Sonar.    Something close that the camera missed -> escape(): turn away from
               it until there is room, then drive a good bit away from it.
  3. Goal.     Where the target is, from the camera if we see it, else from MCL.
               Neither -> search(): turn on the spot until we have one.
               We drive to the closest point on a ring around the target that is
               clear of obstacles, so a landmark blocked from the front is visited
               from the side or from behind.
  4. Camera.   Obstacles (seen now, or remembered on the map) bend the bearing to
               the closest heading that passes all of them (avoid.choose_heading).
  5. Drive.    Turn to the heading and drive a step. The sonars can stop the drive
               at any moment, and the camera stops it for a new obstacle in the way.

Two things keep the layers from working against each other:
  * What the sonar stopped for goes on the same map as the camera's obstacles,
    so step 4 plans around the obstacle we just escaped from instead of driving
    back into it.
  * One side (left/right) is shared by escaping and planning and kept until we
    have made a clean step towards the goal, so obstacles are passed on one side.

Nothing gives up: every dead end leads to another escape, search or turn.

Run on the robot:  python3 -u main.py
and watch the map at http://protoarlo.local:8000/ (mapview.py).
"""
import math
import time

import avoid
from avoid import VISIT_CM, BOX_RADIUS_CM, PASS_CM, LANDMARK_PASS_CM, RING_CM
from hardware import Hardware
from localization import Localizer
from mapview import MapView

# ArUco ID -> (x, y) in cm. L1 is ID 1 at the origin, L2 is ID 2 straight ahead of it, etc.
LANDMARKS = {1: (0, 0), 2: (0, 300), 3: (400, 0), 4: (400, 300)}
ROUTE = [1, 2, 3, 4, 1]

STEP_CM = 80         # Longest drive before stopping to look again
COMMIT_CM = 60       # Drive this far away from an obstacle after escaping it
BACKUP_CM = 20
SCAN_STEP_DEG = 30   # Turn between pictures when searching
ESCAPE_STEP_DEG = 45 # Turn away from an obstacle in steps this big
LOOKAHEAD_CM = 150   # Obstacles further away than this do not matter yet


class Racer:
    def __init__(self, hardware, view=None):
        self.hw = hardware
        self.view = view
        self.target = ROUTE[0]
        self.heading = None     # World direction of the last chosen heading, for the map
        self.localizer = Localizer(LANDMARKS)
        self.obstacle_map = {}  # Obstacle marker id -> world (x, y) of the box centre
        self.bumps = []         # World (x, y) of what the sonar stopped us for, this leg
        self.side = 0           # +1 passing obstacles on the left, -1 on the right, 0 free
        # Before MCL knows where we are, obstacles are kept relative to where we started,
        # tracked by odometry, and moved onto the map once we know our pose.
        self.odometry = (0.0, 0.0, 0.0)
        self.unplaced = {}

    def run(self):
        start = time.monotonic()
        self.localize()
        for target in ROUTE:
            self.target = target
            self.log(f"--- Next target: L{target}")
            self.go_to(target)
            self.log(f"Reached L{target} after {time.monotonic() - start:.0f} s")

    def go_to(self, target):
        self.side = 0
        self.bumps = []
        while True:
            markers = self.look()
            sonar = self.hw.sonar()

            if self.reached(target, markers, sonar) or self.on_ring(target, markers):
                return

            if not avoid.clear_ahead(sonar):
                self.escape(sonar, target)
                continue

            centre = self.goal(target, markers)
            if centre is None:
                self.search(target)
                continue

            obstacles = self.obstacles(markers, target)
            distance, bearing = avoid.ring_goal(centre, obstacles)
            reach = min(LOOKAHEAD_CM, distance)
            nearby = [o for o in obstacles if o[0] - o[2] < reach]
            heading = avoid.choose_heading(bearing, nearby, self.side)
            detour = heading != bearing
            if detour:
                self.side = 1 if avoid.wrap(heading - bearing) > 0 else -1
            pose = self.localizer.pose()
            self.heading = None if pose is None else pose[2] + heading
            self.log(f"L{target}: {distance:.0f} cm at {math.degrees(bearing):.0f} deg, "
                     f"heading {math.degrees(heading):.0f} deg")

            self.turn(math.degrees(heading))
            known = set(markers) | set(self.obstacle_map)
            _, stopped = self.drive(max(5, min(STEP_CM, distance)),
                                    watch=lambda: self.watch(target, known))
            if stopped is None and not detour:
                self.side = 0  # A clean step towards the goal: obstacles are behind us
            elif stopped is not None:
                self.log(f"Drive stopped: {stopped}")

    def reached(self, target, markers, sonar):
        """Within VISIT_CM of the target, seen by the camera or by the front sonar.

        The camera measures from the robot's front to the box face, like the race rule.
        Up close the front sonar is more precise. We trust it when the camera sees the
        target ahead within 2 * VISIT_CM: then no box fits between the sonar and the
        landmark, so the sonar sees the landmark.
        """
        if target not in markers:
            return False
        distance, bearing = markers[target]
        if distance <= VISIT_CM:
            return True
        return (sonar['front'] <= VISIT_CM and abs(bearing) < math.radians(25)
                and distance <= 2 * VISIT_CM)

    def on_ring(self, target, markers):
        """MCL says we are on the ring around the target: turn to it and check with the camera.

        If the camera then sees it further away than VISIT_CM, MCL was wrong (and is now
        corrected by that sighting). If the camera does not see it at all, we trust MCL:
        that close, no box fits between us and the landmark.
        """
        pose = self.localizer.pose()
        if target in markers or pose is None:
            return False
        distance, bearing = avoid.relative(pose, LANDMARKS[target])
        if distance > RING_CM + 10:
            return False
        self.turn(math.degrees(bearing))
        time.sleep(0.2)
        markers = self.look()
        self.log(f"On the ring around L{target} by MCL; camera sees {markers.get(target)}")
        return target not in markers or self.reached(target, markers, self.hw.sonar())

    def goal(self, target, markers):
        """(distance, bearing) to the target centre: from the camera if we see it, else MCL."""
        if target in markers:
            return avoid.from_centre(*markers[target])
        pose = self.localizer.pose()
        if pose is not None:
            return avoid.relative(pose, LANDMARKS[target])
        return None

    def obstacles(self, markers, target):
        """Everything to steer around, as (distance, bearing, clearance) from the robot.

        The target itself is in it too, so we do not drive through it to the far side
        of its ring. Its clearance is smaller than the ring, so it never blocks the ring.
        """
        boxes = [avoid.from_centre(*m) for i, m in markers.items()
                 if i not in LANDMARKS or i == target]
        pose = self.localizer.pose()
        if pose is not None:
            boxes += [avoid.relative(pose, point) for i, point in self.obstacle_map.items()
                      if i not in markers]
            boxes += [avoid.relative(pose, point) for point in self.bumps]
            if target not in markers:
                boxes.append(avoid.relative(pose, LANDMARKS[target]))

        return [(d, b, PASS_CM) for d, b in boxes] + self.zones(target, markers)

    def zones(self, target, markers):
        """The visit zones of the other landmarks, as obstacles (a revisit costs 14 minutes).

        If we are already inside one (just visited it), we only must not get any closer.
        """
        seen = {i: avoid.from_centre(*m) for i, m in markers.items()
                if i in LANDMARKS and i != target}
        pose = self.localizer.pose()
        if pose is not None:
            for i, point in LANDMARKS.items():
                if i != target and i not in seen:
                    seen[i] = avoid.relative(pose, point)
        return [(d, b, max(PASS_CM, min(LANDMARK_PASS_CM, d))) for d, b in seen.values()]

    def watch(self, target, known):
        """Checked while driving: stop at the target, or for a new obstacle in the way."""
        markers = self.hw.markers()
        if target in markers and markers[target][0] <= VISIT_CM:
            return 'target reached'
        for i, (distance, bearing) in markers.items():
            if (i != target and i not in known and distance < LOOKAHEAD_CM
                    and avoid.blocks(0.0, (*avoid.from_centre(distance, bearing), PASS_CM))):
                return f'new obstacle {i}'
        return None

    def escape(self, sonar, target):
        """Something close that the camera missed: turn away from it, then drive away.

        We turn to the side whose sonar has the most room, in ESCAPE_STEP_DEG steps until
        there is room for COMMIT_CM ahead (and it is not into another landmark's visit
        zone), and then drive COMMIT_CM. Afterwards we keep passing on that side.
        """
        self.side = avoid.open_side(sonar)
        self.remember_bump(sonar)
        self.log(f"Escape to the {'left' if self.side > 0 else 'right'}: {sonar}")

        if avoid.all_blocked(sonar):
            self.drive(-BACKUP_CM)
        for _ in range(360 // ESCAPE_STEP_DEG):
            self.turn(self.side * ESCAPE_STEP_DEG)
            zones = [z for z in self.zones(target, {}) if z[0] - z[2] < COMMIT_CM]
            if (avoid.clear_ahead(self.hw.sonar(), room_cm=COMMIT_CM)
                    and not any(avoid.blocks(0.0, z) for z in zones)):
                self.drive(COMMIT_CM)
                return
        # A full turn without room: back up and look again
        self.drive(-BACKUP_CM)

    def remember_bump(self, sonar):
        """Put what the closest front sonar sees on the map, so we plan around it later.

        Not when it is a landmark: we know where those are, and must not plan around
        the target we are going to.
        """
        pose = self.localizer.pose()
        if pose is None:
            return
        name = min(avoid.SONAR_ANGLES, key=lambda s: sonar[s])
        distance = avoid.ROBOT_RADIUS_CM + sonar[name] + BOX_RADIUS_CM
        bump = avoid.absolute(pose, distance, avoid.SONAR_ANGLES[name])
        if all(math.dist(bump, point) > PASS_CM for point in LANDMARKS.values()):
            self.bumps.append(bump)

    def localize(self):
        """Before the race: turn on the spot until two landmarks have fixed our pose."""
        self.log("Localizing")
        self.look()
        for _ in range(360 // SCAN_STEP_DEG):
            if self.localizer.pose() is not None:
                return
            self.turn(SCAN_STEP_DEG)
            time.sleep(0.2)  # Let the camera deliver a sharp picture
            self.look()
        # Not sure yet: the race loop will search on the way

    def search(self, target):
        """We neither see the target nor know where we are: turn and look until we do."""
        self.log(f"Searching for L{target}")
        for _ in range(360 // SCAN_STEP_DEG):
            self.turn(SCAN_STEP_DEG)
            time.sleep(0.2)  # Let the camera deliver a sharp picture
            if self.goal(target, self.look()) is not None:
                return
        # A full turn without enough landmarks: look from somewhere else
        self.drive(STEP_CM)

    def look(self):
        """Take a picture: landmarks update MCL, obstacle markers go on the map."""
        markers = self.hw.markers()
        self.localizer.observe({i: avoid.from_centre(*m) for i, m in markers.items()
                                if i in LANDMARKS})
        obstacles = {i: avoid.from_centre(*m) for i, m in markers.items() if i not in LANDMARKS}
        pose = self.localizer.pose()
        if pose is None:
            for i, seen in obstacles.items():
                self.unplaced[i] = avoid.absolute(self.odometry, *seen)
            self.show()
            return markers
        for i, point in self.unplaced.items():
            self.obstacle_map[i] = avoid.absolute(pose, *avoid.relative(self.odometry, point))
        self.unplaced = {}
        for i, seen in obstacles.items():
            self.obstacle_map[i] = avoid.absolute(pose, *seen)
        self.show()
        return markers

    def log(self, message):
        print(message)
        self.show(message)

    def show(self, status=''):
        if self.view is not None:
            self.view.draw(self, self.target, status)

    # Every movement goes through these two, so MCL always knows how we moved
    def turn(self, degrees):
        self.hw.turn(degrees)
        self.localizer.moved(turn_deg=degrees)
        x, y, theta = self.odometry
        self.odometry = (x, y, theta + math.radians(degrees))

    def drive(self, distance_cm, watch=None):
        driven, stopped = self.hw.drive(distance_cm, watch)
        self.localizer.moved(drive_cm=driven)
        self.odometry = (*avoid.absolute(self.odometry, driven, 0.0), self.odometry[2])
        return driven, stopped


def main():
    view = MapView(LANDMARKS)  # Start the stream first, so the browser can connect
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
