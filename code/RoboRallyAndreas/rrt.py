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


def plan_path(start, goal, obstacles, bounds, max_iterations=5000):
    """Return a simplified collision-free path, or None if planning fails.

    Obstacles are ``(x, y, radius_cm)`` circles. ``bounds`` is the drivable rectangle
    ``(x_min, x_max, y_min, y_max)``; samples are drawn only inside it, and since the
    rectangle is convex every edge between them stays inside too. Coordinates and radii use cm.
    """
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    obstacles = tuple(
        (float(x), float(y), float(radius)) for x, y, radius in obstacles
    )

    if _in_collision(start, obstacles) or _in_collision(goal, obstacles):
        return None
    if not _segment_in_collision(start, goal, obstacles):
        return [start, goal]

    x_min, x_max, y_min, y_max = (float(limit) for limit in bounds)

    tree = [(start, None)]
    for _ in range(max_iterations):
        if random.random() < GOAL_SAMPLE_PROBABILITY:
            sample = goal
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
        if _in_collision(new_point, obstacles) or _segment_in_collision(
            nearest, new_point, obstacles
        ):
            continue

        tree.append((new_point, nearest_index))
        new_index = len(tree) - 1
        if _distance(new_point, goal) <= EXPAND_DISTANCE_CM and not _segment_in_collision(
            new_point, goal, obstacles
        ):
            tree.append((goal, new_index))
            path = []
            index = len(tree) - 1
            while index is not None:
                point, index = tree[index]
                path.append(point)
            path.reverse()
            return _simplify_path(path, obstacles)

    return None
