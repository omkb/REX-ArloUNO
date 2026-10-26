"""Simple MCL using an (N, 3) array of x/y/theta and an array of weights.

Distances are in centimetres and angles in radians. Typical use::

    particles, weights = initialize_from_landmark(landmarks, marker_id, distance, bearing)
    motion_update(particles, delta_trans_cm=20)
    measurement_update(particles, weights, observations, landmarks)
    pose = pose_estimate(particles, weights)
    spread = position_spread_cm(particles, weights)
    particles, weights = resample(particles, weights)

The initialization already includes the first observation; do not apply it again.
Only resample after new measurements. Position spread is a simple localization
check, but cannot guarantee that the heading is known.
"""

import numpy as np


NUM_PARTICLES = 1000
SIGMA_DISTANCE = 10.0
SIGMA_BEARING = np.deg2rad(5.0)
SIGMA_TRANSLATION = 4.0
SIGMA_ROTATION = np.deg2rad(4.0)


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def initialize_from_landmark(landmarks, landmark_id, distance_cm, bearing_rad, num_particles=NUM_PARTICLES):
    """Sample poses on a ring around one observed landmark."""
    lx, ly = landmarks[landmark_id]
    phi = np.random.uniform(-np.pi, np.pi, num_particles)
    radius = np.maximum(1.0, np.random.normal(distance_cm, SIGMA_DISTANCE, num_particles))
    x = lx + radius * np.cos(phi)
    y = ly + radius * np.sin(phi)
    theta = wrap_angle(
        np.arctan2(ly - y, lx - x) - bearing_rad
        + np.random.normal(0, SIGMA_BEARING, num_particles)
    )
    return np.column_stack((x, y, theta)), np.full(num_particles, 1.0 / num_particles)


def motion_update(particles, delta_trans_cm=0.0, delta_rot_rad=0.0):
    """Rotate, then translate. Update particles in place."""
    if delta_trans_cm == 0 and delta_rot_rad == 0:
        return

    n = len(particles)
    theta = wrap_angle(
        particles[:, 2] + delta_rot_rad + np.random.normal(0, SIGMA_ROTATION, n)
    )
    particles[:, 2] = theta

    # Turning in place must not introduce translation noise.
    if delta_trans_cm != 0:
        distance = delta_trans_cm + np.random.normal(0, SIGMA_TRANSLATION, n)
        particles[:, 0] += distance * np.cos(theta)
        particles[:, 1] += distance * np.sin(theta)


def measurement_update(particles, weights, observations, landmarks):
    """Update weights in place; observations are {id: (distance_cm, bearing_rad)}.

    Multiply existing weights by new likelihoods to combine observations over
    time. Log weights keep small likelihoods numerically stable. Return whether
    a known landmark was observed; unknown IDs are ignored.
    """
    observations = {key: value for key, value in observations.items() if key in landmarks}
    if not observations:
        return False

    x, y, theta = particles.T
    log_weights = np.full(len(weights), -np.inf)
    np.log(weights, out=log_weights, where=weights > 0)

    for marker_id, (distance_cm, bearing_rad) in observations.items():
        lx, ly = landmarks[marker_id]
        dx, dy = lx - x, ly - y
        expected_bearing = wrap_angle(np.arctan2(dy, dx) - theta)
        distance_error = (distance_cm - np.hypot(dx, dy)) / SIGMA_DISTANCE
        bearing_error = wrap_angle(bearing_rad - expected_bearing) / SIGMA_BEARING
        log_weights -= 0.5 * (distance_error ** 2 + bearing_error ** 2)

    new_weights = np.exp(log_weights - np.max(log_weights))
    weights[:] = new_weights / new_weights.sum()
    return True


def resample(particles, weights):
    """Return sampled particles and uniform weights after new measurements."""
    n = len(particles)
    indices = np.random.choice(n, size=n, replace=True, p=weights)
    return particles[indices].copy(), np.full(n, 1.0 / n)


def pose_estimate(particles, weights):
    """Weighted mean (x_cm, y_cm, theta_rad), using a circular heading mean."""
    x, y = weights @ particles[:, :2]
    theta = np.arctan2(weights @ np.sin(particles[:, 2]),
                       weights @ np.cos(particles[:, 2]))
    return float(x), float(y), float(theta)


def position_spread_cm(particles, weights):
    """Weighted RMS distance from the mean position; smaller means less spread."""
    x, y, _ = pose_estimate(particles, weights)
    distances_squared = (particles[:, 0] - x) ** 2 + (particles[:, 1] - y) ** 2
    return float(np.sqrt(weights @ distances_squared))
