"""Draw what the robot believes about the world, streamed to a browser for debugging.

Open http://protoarlo.local:8000/ while main.py runs. The map shows:
  grey frame     how far the robot centre may go (track margin)
  blue boxes     landmarks (green: the target); red boxes: boxes seen by the camera
  orange circles what the sonars found; grey circles: no-go zones for the robot centre
  grey dots      MCL particles; white: the robot with its heading
  magenta line   the RRT path; yellow line: our trail
"""
import math
import time

import cv2
import numpy as np

import mapstream

PX_PER_CM = 1.0
BOX_CM, ROBOT_CM = 20, 22  # Half the box size and the robot radius, only for drawing
MIN_INTERVAL_S = 0.5  # Drawing takes time from the robot; do it at most this often
GREY, YELLOW, GREEN, MAGENTA = (130, 130, 130), (0, 220, 220), (0, 200, 0), (200, 0, 200)
BLUE, RED, ORANGE, WHITE = (220, 120, 0), (0, 0, 220), (0, 140, 255), (255, 255, 255)


class MapView:
    def __init__(self, landmarks, bounds, port=8000):
        self.landmarks = landmarks
        self.bounds = bounds
        x_min, x_max, y_min, y_max = bounds
        self.x0, self.y1 = x_min - 30, y_max + 30
        self.size = (int((y_max - y_min + 60) * PX_PER_CM), int((x_max - x_min + 60) * PX_PER_CM))
        self.trail = []
        self.last = 0.0
        self.streamer = mapstream.MapStreamer(port)

    def px(self, point):
        """World (x, y) in cm -> image pixel (column, row), with y pointing up."""
        return int((point[0] - self.x0) * PX_PER_CM), int((self.y1 - point[1]) * PX_PER_CM)

    def circle(self, img, centre, radius_cm, colour, thickness=1):
        cv2.circle(img, self.px(centre), int(radius_cm * PX_PER_CM), colour, thickness)

    def box(self, img, centre, colour, label):
        r = BOX_CM
        cv2.rectangle(img, self.px((centre[0] - r, centre[1] + r)),
                      self.px((centre[0] + r, centre[1] - r)), colour, -1)
        cv2.putText(img, label, self.px((centre[0] - r, centre[1] + r + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    def draw(self, racer, status='', force=False):
        if not force and time.monotonic() - self.last < MIN_INTERVAL_S:
            return
        self.last = time.monotonic()
        img = np.full((*self.size, 3), 40, np.uint8)
        x_min, x_max, y_min, y_max = self.bounds
        cv2.rectangle(img, self.px((x_min, y_max)), self.px((x_max, y_min)), GREY, 1)

        for x, y, r in racer.zones():
            self.circle(img, (x, y), r, GREY)
        for i, point in self.landmarks.items():
            self.box(img, point, GREEN if i == racer.target else BLUE, f"L{i}")
        for i, (x, y, _) in racer.boxes.items():
            self.box(img, (x, y), RED, str(i))
        for point in racer.bumps:
            self.circle(img, point, BOX_CM, ORANGE, 2)
        if racer.path:
            cv2.polylines(img, [np.array([self.px(p) for p in racer.path])], False, MAGENTA, 2)
            self.circle(img, racer.path[racer.waypoint], 4, MAGENTA, -1)

        localizer = racer.localizer
        if localizer.particles is not None:
            for x, y, _ in localizer.particles[::5]:
                cv2.circle(img, self.px((x, y)), 1, GREY, -1)
            if localizer.known():
                x, y, theta = localizer.pose()
                if not self.trail or math.dist(self.trail[-1], (x, y)) > 3:
                    self.trail.append((x, y))
                self.circle(img, (x, y), ROBOT_CM, WHITE, 2)
                cv2.line(img, self.px((x, y)),
                         self.px((x + 40 * math.cos(theta), y + 40 * math.sin(theta))), WHITE, 2)
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([self.px(p) for p in self.trail])], False, YELLOW, 1)

        known = 'known' if localizer.known() else 'unknown'
        cv2.putText(img, f"Target L{racer.target}  pose {known}  {status}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        self.streamer.publish(img)
        return img

    def close(self):
        self.streamer.close()
