import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
import os
import socket
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "lib"))

import robot

def normalize_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def move_particles(particles, velocity, angular_velocity, dt, sigma=2.0, sigma_theta=0.05):
    """Motion model: move every particle along its own heading, then add noise."""
    dist = velocity * dt
    dtheta = angular_velocity * dt
    for p in particles:
        th = p.getTheta()
        particle.move_particle(p, dist * np.cos(th), dist * np.sin(th), dtheta)
    particle.add_uncertainty(particles, sigma, sigma_theta)


def compute_weights(particles, objectIDs, dists, angles, sigma_d=15.0, sigma_phi=0.3):
    """Measurement model: Gaussian likelihood of distance and bearing per landmark,
    multiplied over all observations (done in log space to avoid underflow)."""
    log_w = np.zeros(len(particles))
    for k, p in enumerate(particles):
        for ID, d, phi in zip(objectIDs, dists, angles):
            if ID not in landmarks:
                continue  # unknown box, ignore
            lx, ly = landmarks[ID]
            dx, dy = lx - p.getX(), ly - p.getY()
            d_exp = np.hypot(dx, dy)
            phi_exp = normalize_angle(np.arctan2(dy, dx) - p.getTheta())
            log_w[k] += -0.5 * ((d - d_exp) / sigma_d) ** 2
            log_w[k] += -0.5 * (normalize_angle(phi - phi_exp) / sigma_phi) ** 2
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    for p, wi in zip(particles, w):
        p.setWeight(wi)


def resample(particles):
    """Draw N new particles proportionally to weight, reset weights to uniform."""
    n = len(particles)
    w = np.array([p.getWeight() for p in particles])
    w /= w.sum()
    idx = np.random.choice(n, size=n, p=w)
    return [particle.Particle(particles[i].getX(), particles[i].getY(),
                              particles[i].getTheta(), 1.0 / n) for i in idx]

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return socket.gethostname() == 'protoarlo'

showGUI = not isRunningOnArlo()  # Only open GUI windows when not on the robot

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
landmarkIDs = [9, 11]
landmarks = {
    landmarkIDs[0]: (0.0, 0.0),  # Coordinates for landmark 1
    landmarkIDs[1]: (100.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks





def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
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

    # Draw particles
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

def drive_step(arlo, pose, target,
               turn_power=50, drive_power=60,
               ang_tol=np.radians(10), dist_tol=150):
    """One control step. Call it every time you have a fresh pose estimate.
    Returns True when the target is reached. dist_tol is in your landmark units."""
    x, y, theta = pose
    dx, dy = target[0] - x, target[1] - y

    # Arrived?
    if np.hypot(dx, dy) < dist_tol:
        arlo.stop()
        return True

    # Signed heading error, wrapped to [-pi, pi]
    err = np.arctan2(dy, dx) - theta
    err = (err + np.pi) % (2 * np.pi) - np.pi

    if abs(err) > ang_tol:
        if err > 0:   # target is to the left -> turn left (counter-clockwise)
            arlo.go_diff(turn_power, turn_power, 0, 1)
        else:         # target is to the right -> turn right
            arlo.go_diff(turn_power, turn_power, 1, 0)
    else:
        arlo.go_diff(drive_power, drive_power, 1, 1)  # facing it, go forward
    return False

# Main program #
cam = None
try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

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

    # Initialize the robot (XXX: You do this) TODO
    if isRunningOnArlo():
        arlo = robot.Robot()

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

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

        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break

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
        move_particles(particles, velocity, angular_velocity, dt)

        colour = cam.get_next_frame()
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])

            # --- Weight (all observations at once) ---
            compute_weights(particles, objectIDs, dists, angles)

            # --- Resample ---
            particles = resample(particles)
            particle.add_uncertainty(particles, 1.0, 0.02)  # small jitter to keep diversity

            cam.draw_aruco_objects(colour)
        else:
            for p in particles:
                p.setWeight(1.0 / num_particles)

    
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        # Draw map
        draw_world(est_pose, particles, world)
            
        if showGUI:
    
            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)
        
        # Move robot in between the two landmarks
        lm1, lm2 = landmarks[landmarkIDs[0]], landmarks[landmarkIDs[1]]
        target = ((lm1[0] + lm2[0]) / 2, (lm1[1] + lm2[1]) / 2)        
        pose = (est_pose.getX(), est_pose.getY(), est_pose.getTheta())
        
        if isRunningOnArlo():
            if drive_step(arlo, pose, target):
                print("Reached the target")
  
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    if cam is not None:
        cam.terminateCaptureThread()

