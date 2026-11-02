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

def detour_waypoint(x, y, target_x, target_y, obstacles_xy, clearance_cm):
    """Return the point to aim for so the path passes mapped obstacles.

    An obstacle blocks when it lies between the robot and the target, closer than
    clearance_cm to the straight line. The waypoint sits clearance_cm beside the
    nearest blocking obstacle, on the side away from it. Returns the target itself
    when nothing blocks.
    """
    length = math.hypot(target_x - x, target_y - y)
    if length == 0:
        return target_x, target_y
    ux, uy = (target_x - x) / length, (target_y - y) / length
    nearest = None
    for ox, oy in obstacles_xy:
        along = (ox - x) * ux + (oy - y) * uy      # Distance along the path
        offset = (oy - y) * ux - (ox - x) * uy     # Positive: obstacle left of path
        if 0 < along < length and abs(offset) < clearance_cm:
            if nearest is None or along < nearest[0]:
                nearest = (along, ox, oy, offset)
    if nearest is None:
        return target_x, target_y
    _, ox, oy, offset = nearest
    side = -1 if offset >= 0 else 1                # Pass on the side away from it
    return ox - side * uy * clearance_cm, oy + side * ux * clearance_cm
