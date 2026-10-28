"""World coordinates are cm with angles in radians, left (counter-clockwise) positive.
A pose is (x, y, theta)."""
import math


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def relative(pose, point):
    """(distance, bearing) from the pose to a world point."""
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    return math.hypot(dx, dy), wrap(math.atan2(dy, dx) - pose[2])


def absolute(pose, distance, bearing):
    """The world point at (distance, bearing) from the pose."""
    angle = pose[2] + bearing
    return pose[0] + distance * math.cos(angle), pose[1] + distance * math.sin(angle)
