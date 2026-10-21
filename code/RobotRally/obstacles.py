CLEARANCE_MM = 250  # Minimum clearance after a forward or backward segment


def choose_action(front, left, right, back=None, distance_cm=15, failed_action=None):
    if failed_action != "forward" and front is not None and front > CLEARANCE_MM + distance_cm * 10:
        return "forward"

    options = []

    if left is not None and left > CLEARANCE_MM:
        options.append((left, "left"))

    if right is not None and right > CLEARANCE_MM:
        options.append((right, "right"))

    if options:
        return max(options)[1]

    if failed_action != "backward" and back is not None and back > CLEARANCE_MM + 100:  # Room for a 10 cm reverse
        return "backward"

    # No clear translation: keep turning and check the sensors again.
    return "right" if (right or 0) > (left or 0) else "left"
