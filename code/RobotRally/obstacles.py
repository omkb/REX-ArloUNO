"""Simple sonar decisions, inspired by Exercise 2. Distances are in mm."""

CLEARANCE_MM = 250
TURN_CLEARANCE_MM = 100

def can_turn(front, left, right, back):
    """Allow rotation unless an obstacle is very close."""
    return all(
        d is not None and d > TURN_CLEARANCE_MM
        for d in (front, left, right, back))

def choose_action(front, left, right, back=None, distance_cm=15, avoiding=False):
    """Choose a safe movement based on sonar readings."""

    can_drive = (
        front is not None and front > CLEARANCE_MM + distance_cm * 10
        and left is not None and left > CLEARANCE_MM
        and right is not None and right > CLEARANCE_MM
    )

    # Normal navigation: drive forward when clear
    if can_drive and not avoiding:
        return "forward"

    # During avoidance, try turning before driving
    if can_turn(front, left, right, back):
        return "left" if (left or 0) > (right or 0) else "right"

    # Too close to turn: back up to create space
    if back is not None and back > CLEARANCE_MM + 100:
        return "backward"

    # Already turned into a clear direction
    if can_drive:
        return "forward"

    return "wait"
