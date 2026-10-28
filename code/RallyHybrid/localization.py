"""The particle filter (mcl.py) wrapped, so the rest only asks: where am I, and do I know?"""
import math

import mcl
from geometry import relative

GOOD_SPREAD_CM = 55  # Trust the pose when the particles are this close together
LOST_CM = 100        # Restart when a landmark is this much further or closer than expected


class Localizer:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.particles = self.weights = None

    def moved(self, drive_cm=0.0, turn_rad=0.0):
        """Straight drive, turn on the spot, or both at once for an arc."""
        if self.particles is not None:
            mcl.motion_update(self.particles, drive_cm, turn_rad)

    def observe(self, seen):
        """Update from landmarks {id: (centre distance, bearing)} seen by the camera."""
        seen = {i: value for i, value in seen.items() if i in self.landmarks}
        if not seen:
            return
        if self.particles is None or self._disagrees(seen):
            # Start on a ring around the closest landmark; the next one fixes the pose
            first = min(seen, key=lambda i: seen[i][0])
            self.particles, self.weights = mcl.initialize_from_landmark(
                self.landmarks, first, *seen.pop(first))
        if mcl.measurement_update(self.particles, self.weights, seen, self.landmarks):
            self.particles, self.weights = mcl.resample(self.particles, self.weights)

    def known(self):
        return (self.particles is not None
                and mcl.position_spread_cm(self.particles, self.weights) <= GOOD_SPREAD_CM)

    def pose(self):
        """Best guess (x, y, theta); only trust it when known()."""
        return mcl.pose_estimate(self.particles, self.weights)

    def _disagrees(self, seen):
        if not self.known():
            return False
        pose = self.pose()
        for i, (distance, _) in seen.items():
            if abs(relative(pose, self.landmarks[i])[0] - distance) > LOST_CM:
                print(f"MCL disagrees with landmark {i}; restarting localization")
                return True
        return False
