import math


def direction_to_target(x, y, theta, target_x, target_y):
    """Return (turn_radians, distance_cm) from the robot to the target."""
    dx = target_x - x
    dy = target_y - y
    distance = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    turn = (target_angle - theta + math.pi) % (2 * math.pi) - math.pi
    return turn, distance


def target_reached(distance_cm, threshold_cm):
    """True when the estimated distance is within the chosen threshold."""
    return distance_cm <= threshold_cm