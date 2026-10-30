"""Draw what the robot believes about the world, streamed to a browser for debugging.

Same map as RoboRallyAndreas. Open http://protoarlo.local:8000/ while main.py runs:
  circles        landmarks (red: the target)
  dark grey      boxes seen by the camera; orange: what the sonars found
  light grey     no-go zones for the robot centre; grey frame: the track margin
  blue           MCL particles; black: the pose estimate
  magenta        the RRT path; orange line: our trail
"""
import math
import time

import cv2
import numpy as np

import mapstream
import mcl

BOX_RADIUS_CM = 20
MAX_DRAWN = 200       # Drawing all particles is slow on the Pi
MIN_INTERVAL_S = 0.5  # Drawing takes time from the robot; do it at most this often
BLACK, GREY, LIGHT_GREY = (0, 0, 0), (90, 90, 90), (170, 170, 170)
RED, GREEN, BLUE = (0, 0, 255), (0, 160, 0), (200, 120, 0)
MAGENTA, ORANGE, WHITE = (200, 0, 200), (0, 140, 255), (255, 255, 255)


class MapView:
    def __init__(self, landmarks, bounds, port=8000):
        self.landmarks = landmarks
        self.bounds = bounds
        x_min, x_max, y_min, y_max = bounds
        self.x0, self.y1 = x_min - 30, y_max + 30
        self.size = (int(y_max - y_min + 60), int(x_max - x_min + 60))
        self.trail = []
        self.last = 0.0
        self.streamer = mapstream.MapStreamer(port)

    def px(self, x, y):
        """World (x, y) in cm -> image pixel (column, row), 1 px per cm, y up."""
        return int(x - self.x0), int(self.y1 - y)

    def text(self, img, text, at, colour, scale=0.5):
        cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)

    def draw(self, racer, status='', force=False):
        if not force and time.monotonic() - self.last < MIN_INTERVAL_S:
            return
        self.last = time.monotonic()
        img = np.full((*self.size, 3), 255, np.uint8)
        x_min, x_max, y_min, y_max = self.bounds
        cv2.rectangle(img, self.px(x_min, y_max), self.px(x_max, y_min), LIGHT_GREY, 1)

        for x, y, r in racer.zones():
            cv2.circle(img, self.px(x, y), int(r), LIGHT_GREY, 1)
        for n, (i, (x, y)) in enumerate(self.landmarks.items(), 1):
            colour = RED if i == racer.target else GREEN
            centre = self.px(x, y)
            cv2.circle(img, centre, 8, colour, 2)
            self.text(img, f"L{n} ({i})", (centre[0] + 10, centre[1] - 10), colour, 0.6)
        for i, (x, y, _) in racer.boxes.items():
            centre = self.px(x, y)
            cv2.circle(img, centre, BOX_RADIUS_CM, GREY, -1)
            self.text(img, str(i), (centre[0] - 8, centre[1] + 5), WHITE, 0.45)
        for x, y in racer.bumps:
            cv2.circle(img, self.px(x, y), BOX_RADIUS_CM, ORANGE, -1)

        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([self.px(*p) for p in self.trail], np.int32)],
                          False, ORANGE, 2, cv2.LINE_AA)

        localizer = racer.localizer
        known = 'known' if localizer.known() else 'unknown'
        header = f"Target {racer.target}  pose {known}"
        if localizer.particles is not None:
            particles, weights = localizer.particles, localizer.weights
            drawn = particles
            if len(particles) > MAX_DRAWN:
                drawn = particles[np.random.choice(len(particles), MAX_DRAWN, replace=False)]
            for x, y, theta in drawn:
                cv2.circle(img, self.px(x, y), 2, BLUE, -1)
                cv2.line(img, self.px(x, y),
                         self.px(x + 10 * math.cos(theta), y + 10 * math.sin(theta)), BLUE, 1)

            x, y, theta = localizer.pose()
            cv2.circle(img, self.px(x, y), 6, BLACK, 2)
            cv2.line(img, self.px(x, y),
                     self.px(x + 25 * math.cos(theta), y + 25 * math.sin(theta)), BLACK, 2)
            if localizer.known() and (not self.trail or math.dist(self.trail[-1], (x, y)) >= 5):
                self.trail.append((x, y))
            spread = mcl.position_spread_cm(particles, weights)
            header += f"  ({x:.0f}, {y:.0f}, {math.degrees(theta):.0f} deg) spread={spread:.0f} cm"

        if racer.path:
            cv2.polylines(img, [np.array([self.px(*p) for p in racer.path], np.int32)],
                          False, MAGENTA, 2, cv2.LINE_AA)
            for point in racer.path[1:-1]:
                cv2.drawMarker(img, self.px(*point), MAGENTA, cv2.MARKER_CROSS, 12, 2)
            cv2.circle(img, self.px(*racer.path[racer.waypoint]), 4, MAGENTA, -1)

        self.text(img, header, (10, 20), BLACK)
        if status:
            self.text(img, status, (10, 40), BLACK)
        self.streamer.publish(img)
        return img

    def close(self):
        self.streamer.close()
