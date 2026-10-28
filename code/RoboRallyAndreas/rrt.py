"""RRT path planning for circular obstacles in the robot's world map."""
import math
import random


EXPAND_DISTANCE_CM = 30.0
GOAL_SAMPLE_PROBABILITY = 0.1


def _distance(point_a, point_b):
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def _in_collision(point, obstacles):
    return any(
        _distance(point, (obstacle_x, obstacle_y)) <= radius
        for obstacle_x, obstacle_y, radius in obstacles
    )


def _segment_in_collision(start, end, obstacles):
    segment_x = end[0] - start[0]
    segment_y = end[1] - start[1]
    segment_length_squared = segment_x * segment_x + segment_y * segment_y

    for obstacle_x, obstacle_y, radius in obstacles:
        if segment_length_squared == 0:
            closest_x, closest_y = start
        else:
            fraction = (
                (obstacle_x - start[0]) * segment_x
                + (obstacle_y - start[1]) * segment_y
            ) / segment_length_squared
            fraction = max(0.0, min(1.0, fraction))
            closest_x = start[0] + fraction * segment_x
            closest_y = start[1] + fraction * segment_y
        if _distance((closest_x, closest_y), (obstacle_x, obstacle_y)) <= radius:
            return True
    return False


def path_in_collision(points, obstacles):
    """True when any segment between consecutive points crosses an obstacle circle. As
    in plan_path, the first segment may leave obstacles that contain its start."""
    if len(points) < 2:
        return False
    start_obstacles = tuple(o for o in obstacles if _in_collision(points[0], (o,)))
    if start_obstacles:
        other_obstacles = tuple(o for o in obstacles if o not in start_obstacles)
        if not _leaves_obstacles(points[0], points[1], start_obstacles) or (
            _segment_in_collision(points[0], points[1], other_obstacles)
        ):
            return True
        points = points[1:]
    return any(_segment_in_collision(a, b, obstacles) for a, b in zip(points, points[1:]))


def _leaves_obstacles(start, end, start_obstacles):
    """True when the segment from start heads straight away from the centre of every
    obstacle containing start, so it gets no closer to any of them."""
    return all(
        (end[0] - start[0]) * (start[0] - x) + (end[1] - start[1]) * (start[1] - y) >= 0
        for x, y, _ in start_obstacles
    )


def _simplify_path(path, obstacles):
    simplified = [path[0]]
    current_index = 0

    while current_index < len(path) - 1:
        next_index = len(path) - 1
        while next_index > current_index + 1:
            if not _segment_in_collision(path[current_index], path[next_index], obstacles):
                break
            next_index -= 1
        simplified.append(path[next_index])
        current_index = next_index

    return simplified


def _towards(point, target, distance):
    """The point distance cm from target on the straight line towards point."""
    length = _distance(point, target)
    return (
        target[0] + (point[0] - target[0]) * distance / length,
        target[1] + (point[1] - target[1]) * distance / length,
    )


def _enter_circle(start, end, centre, radius):
    """Where the segment from start (outside the circle) first enters it, given that
    end lies inside."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    fx, fy = start[0] - centre[0], start[1] - centre[1]
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    fraction = (-b - math.sqrt(max(0.0, b * b - 4 * a * c))) / (2 * a)
    return (start[0] + fraction * dx, start[1] + fraction * dy)


def _sample_goal_area(goal, goal_radius):
    """A uniformly random point within goal_radius of goal."""
    angle = random.uniform(0.0, 2 * math.pi)
    radius = goal_radius * math.sqrt(random.random())
    return (goal[0] + radius * math.cos(angle), goal[1] + radius * math.sin(angle))


def _finish(path, goal, goal_radius, obstacles):
    """Simplify the path, and end it where it enters the goal area, goal_radius from goal,
    instead of driving on towards the goal itself."""
    path = _simplify_path(path, obstacles)
    if goal_radius > 0 and _distance(path[-2], goal) > goal_radius:
        path[-1] = _enter_circle(path[-2], path[-1], goal, goal_radius)
    return path


def plan_path(start, goal, obstacles, bounds, goal_radius=0.0, max_iterations=5000):
    """Return a simplified collision-free path, or None if planning fails.

    Obstacles are ``(x, y, radius_cm)`` circles. ``bounds`` is the drivable rectangle
    ``(x_min, x_max, y_min, y_max)``; samples are drawn only inside it, and since the
    rectangle is convex every edge between them stays inside too. Coordinates and radii use cm.

    With a ``goal_radius``, any free point that close to the goal will do, so the goal
    itself may lie inside an obstacle (e.g. a landmark with an obstacle next to it). The
    path then ends goal_radius short of the goal, or at the first point found within it.

    A start inside obstacles (e.g. one mapped onto the robot) is allowed: the first edge
    must then head away from their centres and end outside every obstacle.
    """
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    obstacles = tuple(
        (float(x), float(y), float(radius)) for x, y, radius in obstacles
    )

    goal_free = not _in_collision(goal, obstacles)
    if goal_radius <= 0 and not goal_free:
        return None
    start_obstacles = tuple(
        obstacle for obstacle in obstacles if _in_collision(start, (obstacle,))
    )
    # Obstacles that each tree edge is checked against: the start's own obstacles only
    # count for edges that do not start at the start, see _leaves_obstacles.
    other_obstacles = tuple(o for o in obstacles if o not in start_obstacles)
    if _distance(start, goal) <= goal_radius:
        return [start, start]
    if goal_free and not _segment_in_collision(start, goal, obstacles):
        return _finish([start, goal], goal, goal_radius, obstacles)
    if goal_radius > 0:
        approach = _towards(start, goal, goal_radius)
        if not _in_collision(approach, obstacles) and not _segment_in_collision(
            start, approach, obstacles
        ):
            return [start, approach]

    x_min, x_max, y_min, y_max = (float(limit) for limit in bounds)

    tree = [(start, None)]
    for _ in range(max_iterations):
        if random.random() < GOAL_SAMPLE_PROBABILITY:
            sample = goal if goal_free else _sample_goal_area(goal, goal_radius)
        else:
            sample = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))

        nearest_index = min(
            range(len(tree)),
            key=lambda index: _distance(tree[index][0], sample),
        )
        nearest = tree[nearest_index][0]
        distance = _distance(nearest, sample)
        if distance == 0:
            continue
        scale = min(1.0, EXPAND_DISTANCE_CM / distance)
        new_point = (
            nearest[0] + (sample[0] - nearest[0]) * scale,
            nearest[1] + (sample[1] - nearest[1]) * scale,
        )
        if nearest_index == 0 and start_obstacles:
            blocked = not _leaves_obstacles(start, new_point, start_obstacles) or (
                _segment_in_collision(nearest, new_point, other_obstacles)
            )
        else:
            blocked = _segment_in_collision(nearest, new_point, obstacles)
        if blocked or _in_collision(new_point, obstacles):
            continue

        tree.append((new_point, nearest_index))
        new_index = len(tree) - 1
        if _distance(new_point, goal) <= goal_radius:
            pass  # Close enough: the path ends at new_point
        elif goal_free and _distance(new_point, goal) <= EXPAND_DISTANCE_CM and not (
            _segment_in_collision(new_point, goal, obstacles)
        ):
            tree.append((goal, new_index))
        else:
            continue
        path = []
        index = len(tree) - 1
        while index is not None:
            point, index = tree[index]
            path.append(point)
        path.reverse()
        return _finish(path, goal, goal_radius, obstacles)

    return None
