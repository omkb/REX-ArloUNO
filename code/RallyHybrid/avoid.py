"""Obstacle decisions as pure functions, so they can be tested without the robot.

Distances are in cm and angles in radians, positive to the left. An obstacle is
(distance, bearing, clearance): where its centre is seen from the robot centre,
and how close the robot centre may pass it.
"""
import math

FRONT_STOP_CM = 30   # Stop when the front (or back) sonar reads less than this
SIDE_STOP_CM = 20    # Stop when a diagonal sonar reads less than this
SONAR_ANGLES = {'front': 0.0, 'left': math.radians(45), 'right': math.radians(-45)}

ROBOT_RADIUS_CM = 22  # Robot centre to its edge, where the sonars sit
BOX_RADIUS_CM = 20    # Marker on a box face to the box centre
MARGIN_CM = 10
VISIT_CM = 40         # A landmark is visited within this distance (race rules)

PASS_CM = ROBOT_RADIUS_CM + BOX_RADIUS_CM + MARGIN_CM
# Drive to a ring this far from a landmark's centre: the robot's edge is then
# VISIT_CM / 2 from the box face, so half the visit distance is left for MCL errors.
RING_CM = ROBOT_RADIUS_CM + BOX_RADIUS_CM + VISIT_CM // 2
# Stay out of the visit zone of landmarks that are not the target. An extra margin,
# because we only know where they are from MCL, and box corners stick out.
LANDMARK_PASS_CM = PASS_CM + VISIT_CM + 2 * MARGIN_CM


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def clear_ahead(sonar, room_cm=0):
    """True when the front has room_cm beyond the stop distance and both diagonals are clear."""
    return (sonar['front'] > FRONT_STOP_CM + room_cm
            and sonar['left'] > SIDE_STOP_CM and sonar['right'] > SIDE_STOP_CM)


def all_blocked(sonar):
    return (sonar['front'] <= FRONT_STOP_CM
            and sonar['left'] <= SIDE_STOP_CM and sonar['right'] <= SIDE_STOP_CM)


def open_side(sonar):
    """+1 (left) or -1 (right): the diagonal with the most room."""
    return 1 if sonar['left'] >= sonar['right'] else -1


def blocks(heading, obstacle):
    """True when driving along heading passes the obstacle closer than its clearance."""
    distance, bearing, clearance = obstacle
    ahead = distance * math.cos(bearing - heading)
    beside = distance * math.sin(bearing - heading)
    return ahead > 0 and abs(beside) < clearance


def choose_heading(goal, obstacles, side=0):
    """Return the heading closest to goal that passes all obstacles.

    The candidates are the goal itself and the two tangents past each obstacle, so
    a gap between two obstacles is only used when both clearances fit through it.
    side (+1 left, -1 right, 0 none) is the side we are already passing on; those
    candidates win, so jittery camera estimates cannot make the robot swap sides.
    If every candidate is blocked, the goal is returned and the sonars take over.
    """
    candidates = [goal]
    for distance, bearing, clearance in obstacles:
        angle = math.asin(min(1.0, clearance / distance)) + math.radians(3)
        candidates += [wrap(bearing + angle), wrap(bearing - angle)]

    free = [h for h in candidates if not any(blocks(h, o) for o in obstacles)]
    if not free:
        return goal

    def cost(heading):
        turn = wrap(heading - goal)
        return (side * turn < 0, abs(turn))  # Wrong side first, then the smallest turn
    return min(free, key=cost)


def from_centre(distance, bearing):
    """Camera sighting -> (distance, bearing) from the robot centre to the box centre.

    The camera sits at the front edge of the robot and sees the near face of a box.
    """
    x = ROBOT_RADIUS_CM + (distance + BOX_RADIUS_CM) * math.cos(bearing)
    y = (distance + BOX_RADIUS_CM) * math.sin(bearing)
    return math.hypot(x, y), math.atan2(y, x)


def ring_goal(centre, obstacles, radius=RING_CM):
    """The closest point on a ring around a landmark that keeps clear of all obstacles.

    centre is the landmark's (distance, bearing) from the robot. Returns the point's
    (distance, bearing). So when obstacles stand in front of a landmark, we go round
    and visit it from the side or from behind.
    """
    cx, cy = centre[0] * math.cos(centre[1]), centre[0] * math.sin(centre[1])
    towards_us = math.atan2(-cy, -cx)
    points = []
    for k in range(12):
        angle = towards_us + math.radians(30 * k)
        x, y = cx + radius * math.cos(angle), cy + radius * math.sin(angle)
        points.append((math.hypot(x, y), math.atan2(y, x)))

    def free(point):
        x, y = point[0] * math.cos(point[1]), point[0] * math.sin(point[1])
        return all(math.hypot(x - d * math.cos(b), y - d * math.sin(b)) >= clearance
                   for d, b, clearance in obstacles)
    return min([p for p in points if free(p)] or points, key=lambda p: p[0])


def relative(pose, point):
    """(distance, bearing) from a pose (x, y, theta) to a world point (x, y)."""
    x, y, theta = pose
    dx, dy = point[0] - x, point[1] - y
    return math.hypot(dx, dy), wrap(math.atan2(dy, dx) - theta)


def absolute(pose, distance, bearing):
    """World point (x, y) at distance and bearing from a pose (x, y, theta)."""
    x, y, theta = pose
    return x + distance * math.cos(theta + bearing), y + distance * math.sin(theta + bearing)
