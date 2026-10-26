import math

# All distances in cm. Sonar readings are None when invalid.
CLEARANCE_CM = 35.0  # Minimum clearance after a forward or backward segment
REVERSE_CM = 10.0    # Length of one reverse step

# Camera-detected obstacles: every ArUco marker that is not a landmark.
# Distances are to the marker on the obstacle's surface. Tune on the real robot.
CAMERA_LOOKAHEAD_CM = 50     # React to obstacle markers closer than this
PATH_HALF_WIDTH_CM = 30      # Half robot width plus obstacle radius and margin

# Mapped obstacles: centres in world cm, placed from camera sightings.
OBSTACLE_RADIUS_CM = 20      # Marker face to obstacle centre
MAP_CLEARANCE_CM = 50        # Robot centre to obstacle centre when passing one


def is_clear(reading_cm, distance_cm=0.0):
    """True when moving distance_cm towards a sonar reading still leaves CLEARANCE_CM."""
    return reading_cm is not None and reading_cm > CLEARANCE_CM + abs(distance_cm)


def marker_in_path(markers, lookahead_cm=CAMERA_LOOKAHEAD_CM):
    """Return the bearing of the nearest obstacle marker in the path, else None.

    markers is a list of (distance_cm, bearing_rad) with positive bearing to the left.
    """
    blocking = [(distance, bearing) for distance, bearing in markers
                if distance < lookahead_cm and math.cos(bearing) > 0
                and abs(distance * math.sin(bearing)) < PATH_HALF_WIDTH_CM]
    return min(blocking)[1] if blocking else None


def choose_action(front, left, right, back=None, markers=(), distance_cm=15, failed_action=None):
    """Return 'forward', 'left', 'right' or 'backward' from sonar (cm) and obstacle markers."""
    if failed_action != "forward" and is_clear(front, distance_cm):
        bearing = marker_in_path(markers)
        if bearing is None:
            return "forward"
        return "right" if bearing > 0 else "left"  # Turn away from the marker

    options = [(reading, side) for reading, side in ((left, "left"), (right, "right"))
               if is_clear(reading)]
    if options:
        return max(options)[1]

    if failed_action != "backward" and is_clear(back, REVERSE_CM):
        return "backward"

    # No clear translation: keep turning and check the sensors again.
    return "right" if (right or 0) > (left or 0) else "left"
