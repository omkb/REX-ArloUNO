"""Particle filter (mcl.py) wrapped so the rest only asks: where am I, if we know?"""
import math

import mcl
from avoid import relative

GOOD_SPREAD_CM = 40  # Trust the pose when the particles are this close together
LOST_CM = 100        # Restart when a landmark is this much further or closer than expected


class Localizer:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.particles = self.weights = None

    def moved(self, turn_deg=0.0, drive_cm=0.0):
        if self.particles is not None:
            mcl.motion_update(self.particles, drive_cm, math.radians(turn_deg))

    def observe(self, seen):
        """Update from landmarks seen by the camera, {id: (distance, bearing)}."""
        seen = {i: value for i, value in seen.items() if i in self.landmarks}
        if not seen:
            return
        if self.particles is None or self._lost(seen):
            # Start from a ring around the closest landmark; the next one fixes the pose
            first = min(seen, key=lambda i: seen[i][0])
            self.particles, self.weights = mcl.initialize_from_landmark(
                self.landmarks, first, *seen.pop(first))
        if seen:
            mcl.measurement_update(self.particles, self.weights, seen, self.landmarks)
            self.particles, self.weights = mcl.resample(self.particles, self.weights)

    def pose(self):
        """(x, y, theta) when the particles agree, otherwise None."""
        if self.particles is None:
            return None
        if mcl.position_spread_cm(self.particles, self.weights) > GOOD_SPREAD_CM:
            return None
        return mcl.pose_estimate(self.particles, self.weights)

    def _lost(self, seen):
        pose = self.pose()
        if pose is None:
            return False
        for i, (distance, _) in seen.items():
            if abs(relative(pose, self.landmarks[i])[0] - distance) > LOST_CM:
                print(f"MCL disagrees with landmark {i}; restarting localization")
                return True
        return False
