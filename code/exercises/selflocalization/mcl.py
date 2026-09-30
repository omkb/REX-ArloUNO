import numpy as np
import particle


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

NUM_PARTICLES = 1000

# Area where the robot may initially be
X_MIN = -100.0
X_MAX = 500.0
Y_MIN = -250.0
Y_MAX = 350.0

# Camera is mounted 22.5 cm in front of robot center
CAMERA_OFFSET = 22.5

# Measurement noise
SIGMA_DISTANCE = 4.0
SIGMA_BEARING = np.deg2rad(3.0)

# Motion noise
ALPHA1 = 0.002       # rotation -> rotation noise
ALPHA2 = 0.000001    # translation -> rotation noise
ALPHA3 = 0.01        # translation -> translation noise
ALPHA4 = 1.0         # rotation -> translation noise


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------
# INITIALIZE PARTICLES
# --------------------------------------------------

def initialize_particles(num_particles=NUM_PARTICLES):
    particles = []

    for _ in range(num_particles):
        x = np.random.uniform(X_MIN, X_MAX)
        y = np.random.uniform(Y_MIN, Y_MAX)
        theta = np.random.uniform(0, 2 * np.pi)

        p = particle.Particle(x, y, theta, 1.0 / num_particles)

        particles.append(p)

    return particles


# --------------------------------------------------
# MOTION MODEL
# --------------------------------------------------

def move_particles(particles, delta_trans, delta_rot):

    # Variance of rotational and translational error
    rot_variance = (
        ALPHA1 * abs(delta_rot) +
        ALPHA2 * abs(delta_trans)
    )

    trans_variance = (
        ALPHA3 * abs(delta_trans) +
        ALPHA4 * abs(delta_rot)
    )

    # Standard deviation = sqrt(variance)
    sigma_rot = np.sqrt(rot_variance)
    sigma_trans = np.sqrt(trans_variance)

    for p in particles:

        noisy_rot = delta_rot + np.random.normal(0, sigma_rot)
        noisy_trans = delta_trans + np.random.normal(0, sigma_trans)

        new_theta = wrap_angle(p.getTheta() + noisy_rot)

        new_x = p.getX() + noisy_trans * np.cos(new_theta)
        new_y = p.getY() + noisy_trans * np.sin(new_theta)

        p.setX(new_x)
        p.setY(new_y)
        p.setTheta(new_theta)


# --------------------------------------------------
# EXPECTED MEASUREMENT
# --------------------------------------------------

def expected_measurement(p, landmark):
    theta = p.getTheta()

    # Camera position
    camera_x = p.getX() + CAMERA_OFFSET * np.cos(theta)
    camera_y = p.getY() + CAMERA_OFFSET * np.sin(theta)

    landmark_x, landmark_y = landmark

    dx = landmark_x - camera_x
    dy = landmark_y - camera_y

    distance = np.sqrt(dx**2 + dy**2)

    landmark_angle = np.arctan2(dy, dx)

    bearing = wrap_angle(landmark_angle - theta)

    return distance, bearing


# --------------------------------------------------
# MEASUREMENT MODEL
# --------------------------------------------------

def measurement_update(particles, object_ids, distances, angles, landmarks):

    if object_ids is None:
        return False

    # Store one observation for each landmark
    observations = {}

    for i in range(len(object_ids)):
        marker_id = int(object_ids[i])

        if marker_id in landmarks and marker_id not in observations:
            observations[marker_id] = (
                distances[i],
                angles[i]
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

            distance_error = measured_distance - expected_distance
            bearing_error = wrap_angle(measured_bearing - expected_bearing)

            distance_weight = np.exp(
                -0.5 * (distance_error / SIGMA_DISTANCE) ** 2
            )

            bearing_weight = np.exp(
                -0.5 * (bearing_error / SIGMA_BEARING) ** 2
            )

            weight *= distance_weight * bearing_weight

        p.setWeight(weight)

    # Normalize weights
    total_weight = sum(p.getWeight() for p in particles)

    if total_weight == 0:
        return False

    for p in particles:
        p.setWeight(p.getWeight() / total_weight)

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

        new_particles.append(new_particle)

    return new_particles


# --------------------------------------------------
# ONE MCL UPDATE
# --------------------------------------------------

def mcl_update(particles, object_ids, distances, angles, landmarks):

    measurement_available = measurement_update(particles, object_ids, distances, angles, landmarks)

    if measurement_available:
        particles = resample_particles(particles)

    return particles
