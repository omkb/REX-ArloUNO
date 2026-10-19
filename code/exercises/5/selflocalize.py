import cv2
import particle
import camera
import mapstream
import numpy as np
import time
from timeit import default_timer as timer
import sys
import os
import socket
import select
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "lib"))

import robot
from arlo import Arlo

def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def get_state(particles):
    """Particle poses as numpy arrays (x, y, theta)."""
    x = np.array([p.getX() for p in particles])
    y = np.array([p.getY() for p in particles])
    th = np.array([p.getTheta() for p in particles])
    return x, y, th


def set_state(particles, x, y, th):
    """Write numpy arrays back into the particle objects."""
    th = np.mod(th, 2.0 * np.pi)
    for p, xi, yi, ti in zip(particles, x, y, th):
        p.x, p.y, p.theta = xi, yi, ti


def move_particles(particles, dist, dtheta,
                   alpha_d=0.1, alpha_th=0.1, min_sigma=0.2, min_sigma_th=0.005):
    """Motion model: move every particle dist cm along its own heading and turn it
    dtheta rad, then add noise proportional to how far the robot moved."""
    n = len(particles)
    x, y, th = get_state(particles)
    x = x + dist * np.cos(th)
    y = y + dist * np.sin(th)
    th = th + dtheta
    sigma = min_sigma + alpha_d * abs(dist)
    # Take distance into account when calculating heading because robot does not drive straight
    sigma_theta = min_sigma_th + alpha_th * abs(dtheta) + 0.01 * abs(dist)
    x += np.random.normal(0.0, sigma, n)
    y += np.random.normal(0.0, sigma, n)
    th += np.random.normal(0.0, sigma_theta, n)
    set_state(particles, x, y, th)


def compute_weights(particles, objectIDs, dists, angles, sigma_d=15.0, sigma_phi=0.3):
    """Measurement model: Gaussian likelihood of distance and bearing per landmark,
    multiplied over all observations (done in log space to avoid underflow)."""
    x, y, th = get_state(particles)
    log_w = np.zeros(len(particles))
    for ID, d, phi in zip(objectIDs, dists, angles):
        if ID not in landmarks:
            continue  # unknown box, ignore
        lx, ly = landmarks[ID]
        dx, dy = lx - x, ly - y
        d_exp = np.hypot(dx, dy)
        phi_exp = normalize_angle(np.arctan2(dy, dx) - th)
        log_w += -0.5 * ((d - d_exp) / sigma_d) ** 2
        log_w += -0.5 * (normalize_angle(phi - phi_exp) / sigma_phi) ** 2
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    for p, wi in zip(particles, w):
        p.setWeight(wi)


def resample(particles, sigma=0.5, sigma_theta=0.01):
    """Draw N new particles proportionally to weight, reset weights to uniform,
    and add a small jitter to keep diversity."""
    n = len(particles)
    x, y, th = get_state(particles)
    w = np.array([p.getWeight() for p in particles])
    w /= w.sum()
    idx = np.random.choice(n, size=n, p=w)
    x = x[idx] + np.random.normal(0.0, sigma, n)
    y = y[idx] + np.random.normal(0.0, sigma, n)
    th = th[idx] + np.random.normal(0.0, sigma_theta, n)
    return [particle.Particle(xi, yi, ti, 1.0 / n) for xi, yi, ti in zip(x, y, th)]


def is_localized(particles, pos_tol=20.0, heading_tol=0.3):
    """True when the particle cloud has collapsed: position spread below pos_tol cm
    and circular heading spread below heading_tol rad."""
    x, y, th = get_state(particles)
    r = np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th)))
    heading_std = np.sqrt(-2.0 * np.log(max(r, 1e-12)))
    return max(np.std(x), np.std(y)) < pos_tol and heading_std < heading_tol


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return socket.gethostname() == 'protoarlo'

showCamera = not isRunningOnArlo()  # Camera stream is too heavy for SSH X forwarding
showWorld = not isRunningOnArlo()  # World view window on the laptop
streamWorld = isRunningOnArlo()  # On Arlo, stream the world view to a browser instead
STREAM_PORT = 8000

# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [6, 10]
landmarks = {
    landmarkIDs[0]: (0.0, 0.0),  # Coordinates for landmark 1
    landmarkIDs[1]: (170.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks





def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world, max_drawn=200):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles (a random subset, drawing all of them is slow on the Pi)
    if len(particles) > max_drawn:
        idx = np.random.choice(len(particles), size=max_drawn, replace=False)
        particles = [particles[i] for i in idx]
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)
        cv2.putText(world, str(ID), (lm[0] + 8, lm[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, landmark_colors[i], 1, cv2.LINE_AA)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
         ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)



def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles

def pwm_to_cms(pwm):
    """Wheel speed in cm/s for a given motor PWM, from Arlo's calibration."""
    return Arlo.SPEED_SLOPE * pwm + Arlo.SPEED_INTERCEPT


TURN_POWER = 50
DRIVE_POWER = 60
SETTLE_TIME = 0.3  # s to wait after stopping so the next camera frame is sharp


def turn_burst(arlo, angle, power=TURN_POWER):
    """Turn in place by roughly angle rad (positive = left), then stop.
    Blocks for the duration of the turn. Returns the commanded turn in rad."""
    omega = 2.0 * pwm_to_cms(power) / Arlo.TRACK_WIDTH_CM  # wheels spin opposite ways
    if angle > 0:   # turn left (counter-clockwise)
        arlo.go_diff(power, power, 0, 1)
    else:           # turn right
        arlo.go_diff(power, power, 1, 0)
    time.sleep(abs(angle) / omega)
    arlo.stop()
    time.sleep(SETTLE_TIME)
    return angle


def drive_burst(arlo, dist, power=DRIVE_POWER):
    """Drive forward roughly dist cm, then stop. Blocks for the duration of the
    drive. Returns the commanded distance in cm."""
    arlo.go_diff(power, power, 1, 1)
    time.sleep(dist / pwm_to_cms(power))
    arlo.stop()
    time.sleep(SETTLE_TIME)
    return dist


def search_step(arlo, step=np.radians(20)):
    """Not localized yet: turn a little to look for landmarks.
    Returns (dist, dtheta) moved, for the motion model."""
    return 0.0, turn_burst(arlo, step)


def drive_step(arlo, pose, target,
               ang_tol=np.radians(10), dist_tol=15,
               max_turn=np.radians(30), max_drive=30.0):
    """One stop-and-go control step: turn towards the target or drive a short
    distance towards it, then stop so the next measurement is taken standing still.
    Call it every time you have a fresh pose estimate.
    Returns (reached, dist, dtheta): whether the target is reached, and the
    commanded motion in cm and rad for the motion model.
    dist_tol is in your landmark units."""
    x, y, theta = pose
    dx, dy = target[0] - x, target[1] - y
    remaining = np.hypot(dx, dy)

    # Arrived?
    if remaining < dist_tol:
        arlo.stop()
        return True, 0.0, 0.0

    # Signed heading error, wrapped to [-pi, pi]
    err = normalize_angle(np.arctan2(dy, dx) - theta)

    if abs(err) > ang_tol:
        return False, 0.0, turn_burst(arlo, np.clip(err, -max_turn, max_turn))
    else:
        return False, drive_burst(arlo, min(remaining, max_drive)), 0.0

# Main program #
cam = None
streamer = None
try:
    # Open windows
    if showCamera:
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

    if streamWorld:
        streamer = mapstream.MapStreamer(STREAM_PORT)

    if showWorld:
        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)


    # Initialize particles
    num_particles = 1000
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec
    moved = (0.0, 0.0)  # (cm, rad) commanded on Arlo since the last prediction

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map, and stream it right away so the browser shows it during start-up
    draw_world(est_pose, particles, world)
    if streamWorld:
        streamer.publish(world)

    # Initialize the robot (XXX: You do this) TODO
    if isRunningOnArlo():
        arlo = robot.Robot()

    # Don't drive until the user presses Enter; the filter and map keep running meanwhile
    start_driving = False
    if isRunningOnArlo():
        print("Press Enter to start driving...")

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        #cam = camera.Camera(1, robottype='macbookpro', useCaptureThread=False)

    os.makedirs('/dev/shm/view', exist_ok=True)

    last_time = timer()

    while True:

        # Start driving once Enter has been pressed (non-blocking check of stdin)
        if isRunningOnArlo() and not start_driving:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                sys.stdin.readline()
                start_driving = True
                print("Driving")

        # Move the robot according to user input (only for testing)
        if showCamera or showWorld:
            action = cv2.waitKey(10)
            if action == ord('q'): # Quit
                break
        else:
            action = -1

        if not isRunningOnArlo():
            if action == ord('w'): # Forward
                velocity += 4.0
            elif action == ord('x'): # Backwards
                velocity -= 4.0
            elif action == ord('s'): # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'): # Left
                angular_velocity += 0.2
            elif action == ord('d'): # Right
                angular_velocity -= 0.2


        # --- Predict ---
        now = timer()
        dt = now - last_time
        last_time = now
        if isRunningOnArlo():
            move_particles(particles, *moved)
            moved = (0.0, 0.0)
        else:
            move_particles(particles, velocity * dt, angular_velocity * dt)

        colour = cam.get_next_frame()
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])

            # --- Weight (all observations at once) ---
            compute_weights(particles, objectIDs, dists, angles)

            # --- Resample ---
            particles = resample(particles)

            cam.draw_aruco_objects(colour)
        else:
            for p in particles:
                p.setWeight(1.0 / num_particles)

    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

            
        if showCamera:
            # Show frame
            cv2.imshow(WIN_RF1, colour)

        if showWorld or streamWorld:
            # Draw map
            draw_world(est_pose, particles, world)

        if showWorld:
            # Show world
            cv2.imshow(WIN_World, world)

        if streamWorld:
            streamer.publish(world)
        
        # Move robot in between the two landmarks
        lm1, lm2 = landmarks[landmarkIDs[0]], landmarks[landmarkIDs[1]]
        target = ((lm1[0] + lm2[0]) / 2, (lm1[1] + lm2[1]) / 2)        
        pose = (est_pose.getX(), est_pose.getY(), est_pose.getTheta())
        
        if isRunningOnArlo() and start_driving:
            if not is_localized(particles):
                moved = search_step(arlo)
            else:
                reached, *moved = drive_step(arlo, pose, target)
                if reached:
                    print("Reached the target")
  
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    if cam is not None:
        cam.terminateCaptureThread()

    if streamer is not None:
        streamer.close()

