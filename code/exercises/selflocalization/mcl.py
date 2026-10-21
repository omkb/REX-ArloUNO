import numpy as np
import particle


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

NUM_PARTICLES = 1000

# Possible initial robot area
X_MIN = -100.0
X_MAX = 400.0

Y_MIN = -300.0
Y_MAX = 300.0


# Measurement noise
# Starting values - should be tuned experimentally
SIGMA_DISTANCE = 10.0
SIGMA_BEARING = np.deg2rad(5.0)


# Motion noise
# Starting values - should be tuned experimentally
SIGMA_TRANSLATION = 2.0       # cm
SIGMA_ROTATION = np.deg2rad(2.0)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

def initialize_particles(num_particles=NUM_PARTICLES):
    particles = []

    for _ in range(num_particles):
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        theta = np.random.uniform(-np.pi, np.pi)

        p = particle.Particle(
            x,
            y,
            theta,
            1.0 / num_particles
        )

        particles.append(p)

    return particles


# --------------------------------------------------
# MOTION MODEL
# --------------------------------------------------

def move_particles(particles, delta_trans, delta_rot):

    for p in particles:

        noisy_trans = delta_trans + np.random.normal(
            0,
            SIGMA_TRANSLATION
        )

        noisy_rot = delta_rot + np.random.normal(
            0,
            SIGMA_ROTATION
        )

        new_theta = wrap_angle(
            p.getTheta() + noisy_rot
        )

        new_x = (
            p.getX()
            + noisy_trans * np.cos(new_theta)
        )

        new_y = (
            p.getY()
            + noisy_trans * np.sin(new_theta)
        )

        p.setX(new_x)
        p.setY(new_y)
        p.setTheta(new_theta)


# --------------------------------------------------
# EXPECTED MEASUREMENT
# --------------------------------------------------

def expected_measurement(p, landmark):

    landmark_x, landmark_y = landmark

    dx = landmark_x - p.getX()
    dy = landmark_y - p.getY()

    distance = np.sqrt(
        dx**2 + dy**2
    )

    direction_to_landmark = np.arctan2(
        dy,
        dx
    )

    bearing = wrap_angle(
        direction_to_landmark - p.getTheta()
    )

    return distance, bearing


# --------------------------------------------------
# CHOOSE OBSERVATIONS
# --------------------------------------------------

def get_observations(object_ids, distances, angles, landmarks):

    observations = {}

    if object_ids is None:
        return observations

    for marker_id, distance, angle in zip(
        object_ids,
        distances,
        angles
    ):

        marker_id = int(marker_id)

        if marker_id not in landmarks:
            continue

        # If the same landmark ID is seen multiple times,
        # keep the closest detection.
        if (
            marker_id not in observations
            or distance < observations[marker_id][0]
        ):
            observations[marker_id] = (
                distance,
                angle
            )

    return observations


# --------------------------------------------------
# MEASUREMENT MODEL
# --------------------------------------------------

def measurement_update(
    particles,
    object_ids,
    distances,
    angles,
    landmarks
):

    observations = get_observations(
        object_ids,
        distances,
        angles,
        landmarks
    )

    if len(observations) == 0:
        return False

    for p in particles:

        weight = 1.0

        for marker_id, observation in observations.items():

            measured_distance, measured_bearing = observation

            expected_distance, expected_bearing = expected_measurement(
                p,
                landmarks[marker_id]
            )

            distance_error = (
                measured_distance
                - expected_distance
            )

            bearing_error = wrap_angle(
                measured_bearing
                - expected_bearing
            )

            distance_weight = np.exp(
                -0.5
                * (distance_error / SIGMA_DISTANCE) ** 2
            )

            bearing_weight = np.exp(
                -0.5
                * (bearing_error / SIGMA_BEARING) ** 2
            )

            weight *= (
                distance_weight
                * bearing_weight
            )

        p.setWeight(weight)

    total_weight = sum(
        p.getWeight()
        for p in particles
    )

    if total_weight == 0:
        return False

    for p in particles:
        p.setWeight(
            p.getWeight() / total_weight
        )

    return True


# --------------------------------------------------
# RESAMPLING
# --------------------------------------------------

def resample_particles(particles):

    weights = [
        p.getWeight()
        for p in particles
    ]

    indices = np.random.choice(
        len(particles),
        size=len(particles),
        replace=True,
        p=weights
    )

    new_particles = []

    for index in indices:

        old = particles[index]

        new_particle = particle.Particle(
            old.getX(),
            old.getY(),
            old.getTheta(),
            1.0 / len(particles)
        )

        new_particles.append(
            new_particle
        )

    return new_particles


# --------------------------------------------------
# ONE MCL UPDATE
# --------------------------------------------------

def mcl_update(
    particles,
    object_ids,
    distances,
    angles,
    landmarks
):

    measurement_available = measurement_update(
        particles,
        object_ids,
        distances,
        angles,
        landmarks
    )

    if measurement_available:
        particles = resample_particles(
            particles
        )

    return particles
