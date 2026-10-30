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
        self.seen = set()     # Landmarks seen since the particles were (re)started
        self.moved_since = False

    def moved(self, drive_cm=0.0, turn_rad=0.0):
        """Straight drive, turn on the spot, or both at once for an arc."""
        if self.particles is not None:
            mcl.motion_update(self.particles, drive_cm, turn_rad)
            self.moved_since = True

    def observe(self, seen):
        """Update from landmarks {id: (centre distance, bearing)} seen by the camera."""
        seen = {i: value for i, value in seen.items() if i in self.landmarks}
        if not seen:
            return
        was_known = self.known()
        if self.particles is None or self._disagrees(seen):
            # Start on a ring around the closest landmark; another landmark fixes the pose
            first = min(seen, key=lambda i: seen[i][0])
            self.particles, self.weights = mcl.initialize_from_landmark(
                self.landmarks, first, *seen.pop(first))
            self.seen = {first}
            self.moved_since = True
        # Resampling the same view again and again, standing still, would shrink a ring
        # around one landmark to a random spot on it. So only update after moving.
        if self.moved_since and mcl.measurement_update(self.particles, self.weights, seen, self.landmarks):
            self.particles, self.weights = mcl.resample(self.particles, self.weights)
            self.seen |= set(seen)
            self.moved_since = False
        if self.known() and not was_known:
            x, y, theta = self.pose()
            print(f"Pose known: ({x:.0f}, {y:.0f}, {math.degrees(theta):.0f} deg) "
                  f"spread={self.spread():.0f} cm from landmarks {sorted(self.seen)}")

    def known(self):
        """One landmark only gives a ring around it, so it takes two."""
        return (self.particles is not None and len(self.seen) >= 2
                and self.spread() <= GOOD_SPREAD_CM)

    def spread(self):
        return mcl.position_spread_cm(self.particles, self.weights)

    def pose(self):
        """Best guess (x, y, theta); only trust it when known()."""
        return mcl.pose_estimate(self.particles, self.weights)

    def _disagrees(self, seen):
        if not self.known():
            return False
        pose = self.pose()
        for i, (distance, bearing) in seen.items():
            expected, expected_bearing = relative(pose, self.landmarks[i])
            if abs(expected - distance) > LOST_CM:
                print(f"MCL disagrees with landmark {i}: seen {distance:.0f} cm at "
                      f"{math.degrees(bearing):.0f} deg, expected {expected:.0f} cm at "
                      f"{math.degrees(expected_bearing):.0f} deg from pose "
                      f"({pose[0]:.0f}, {pose[1]:.0f}, {math.degrees(pose[2]):.0f} deg); restarting")
                return True
        return False
