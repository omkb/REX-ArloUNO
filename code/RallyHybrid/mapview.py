"""Draw what the robot believes about the world, streamed to a browser for debugging.

Open http://protoarlo.local:8000/ while main.py runs. The map shows:
  blue boxes     landmarks (green: the current target), thin ring: where we drive to
  red boxes      obstacles on the map (camera), with the clearance we keep
  orange crosses what the sonar stopped us for
  grey dots      MCL particles; the robot is the circle with a heading line
  green line     the heading we chose last (world); yellow line: our trail
"""
import math

import cv2
import numpy as np

import avoid
import mapstream

PX_PER_CM = 1.5
BORDER_CM = 150  # Space around the landmarks, since we may drive outside the square
GREY, YELLOW, GREEN = (130, 130, 130), (0, 220, 220), (0, 200, 0)
BLUE, RED, ORANGE, WHITE = (220, 120, 0), (0, 0, 220), (0, 140, 255), (255, 255, 255)


class MapView:
    def __init__(self, landmarks, port=8000):
        self.landmarks = landmarks
        xs = [p[0] for p in landmarks.values()]
        ys = [p[1] for p in landmarks.values()]
        self.x0, self.y1 = min(xs) - BORDER_CM, max(ys) + BORDER_CM
        self.size = (int((max(ys) - min(ys) + 2 * BORDER_CM) * PX_PER_CM),
                     int((max(xs) - min(xs) + 2 * BORDER_CM) * PX_PER_CM))
        self.trail = []
        self.streamer = mapstream.MapStreamer(port)

    def px(self, point):
        """World (x, y) in cm -> image pixel (column, row), with y pointing up."""
        return (int((point[0] - self.x0) * PX_PER_CM), int((self.y1 - point[1]) * PX_PER_CM))

    def box(self, img, centre, colour, label):
        r = avoid.BOX_RADIUS_CM
        cv2.rectangle(img, self.px((centre[0] - r, centre[1] + r)),
                      self.px((centre[0] + r, centre[1] - r)), colour, -1)
        cv2.putText(img, label, self.px((centre[0] - r, centre[1] + r + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    def circle(self, img, centre, radius_cm, colour, thickness=1):
        cv2.circle(img, self.px(centre), int(radius_cm * PX_PER_CM), colour, thickness)

    def draw(self, racer, target, status=''):
        img = np.full((*self.size, 3), 40, np.uint8)

        for i, point in self.landmarks.items():
            colour = GREEN if i == target else BLUE
            self.box(img, point, colour, f"L{i}")
            self.circle(img, point, avoid.RING_CM, colour)
        for i, point in racer.obstacle_map.items():
            self.box(img, point, RED, str(i))
            self.circle(img, point, avoid.PASS_CM, RED)
        for point in racer.bumps:
            cv2.drawMarker(img, self.px(point), ORANGE, cv2.MARKER_TILTED_CROSS, 14, 2)

        localizer = racer.localizer
        if localizer.particles is not None:
            for x, y, _ in localizer.particles[::5]:
                cv2.circle(img, self.px((x, y)), 1, GREY, -1)

        pose = localizer.pose()
        if pose is not None:
            if not self.trail or math.dist(self.trail[-1], pose[:2]) > 2:
                self.trail.append(pose[:2])
            self.circle(img, pose[:2], avoid.ROBOT_RADIUS_CM, WHITE, 2)
            cv2.line(img, self.px(pose[:2]), self.px(avoid.absolute(pose, 40, 0.0)), WHITE, 2)
            if racer.heading is not None:
                cv2.line(img, self.px(pose[:2]), self.px(avoid.absolute((*pose[:2], racer.heading), 120, 0.0)),
                         GREEN, 1)
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array([self.px(p) for p in self.trail])], False, YELLOW, 1)

        text = f"Target L{target}  pose {'unknown' if pose is None else 'known'}  {status}"
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        self.streamer.publish(img)
        return img

    def close(self):
        self.streamer.close()
