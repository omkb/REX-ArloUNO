"""Test MCL localization from two ArUco landmarks observed at different headings.

Required local files: arlo.py, robot.py, mcl.py, particle.py.
Additional libraries: numpy, opencv with aruco, matplotlib, picamera2 (or USB camera).

IMPORTANT: This is an experiment, not an autonomous navigation controller.
The robot only turns in place. The code assumes the turn angle is approximately
correct and requires camera calibration at the selected image resolution.

Example (replace coordinates and camera values with your calibrated values):
 python3 -u test_sequential_mcl.py --landmark 9:0:0 --landmark 10:170:0 \
     --first-id 9 --marker-cm 14.4 --fx 1288 --step 10 --save-every 5

Camera geometry: x=right, z=forward; MCL bearing=positive toward robot's left.
Landmark coordinates must refer to the SAME physical point as the ArUco marker.
"""

import argparse
import math
from pathlib import Path
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mcl
import particle
from arlo import Arlo


class Camera:
    def __init__(self, index, width, height):
        self.picam = None
        self.cap = None
        self.width = width
        self.height = height
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            config = self.picam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            self.mode = "picamera2"
        except (ImportError, RuntimeError, OSError):
            if self.picam is not None:
                try:
                    self.picam.close()
                except Exception:
                    pass
                self.picam = None
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError("Camera unavailable; adapt Camera to your existing camera code")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.mode = "opencv"
        time.sleep(0.8)

    def gray(self):
        if self.picam is not None:
            frame = self.picam.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("No frame from camera")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def close(self):
        if self.picam is not None:
            self.picam.stop()
            self.picam.close()
        if self.cap is not None:
            self.cap.release()


class MarkerObserver:
    def __init__(self, fx, fy, cx, cy, marker_cm):
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.dist = np.zeros((5, 1), dtype=np.float64)  # Replace with calibrated distortion if available.
        half = marker_cm / 2
        # Corner order OpenCV ArUco: top left, top right, bottom right, bottom left
        self.object_corners = np.array([
            [-half, half, 0], [half, half, 0],
            [half, -half, 0], [-half, -half, 0]
        ], dtype=np.float64)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
            self.detect = detector.detectMarkers
        else:
            params = cv2.aruco.DetectorParameters_create()
            self.detect = lambda img: cv2.aruco.detectMarkers(img, dictionary, parameters=params)

    def observe(self, gray):
        corners, ids, _ = self.detect(gray)
        results = {}
        if ids is None:
            return results
        for polygon, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            pts = np.asarray(polygon, dtype=np.float64).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(
                self.object_corners, pts, self.K, self.dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue
            tx, _, tz = tvec.reshape(3)
            if tz <= 0:
                continue
            # Horizontal distance, and bearing measured positive to the LEFT.
            distance = float(np.hypot(tx, tz))
            bearing = float(-np.arctan2(tx, tz))
            if marker_id not in results or distance < results[marker_id][0]:
                results[marker_id] = (distance, bearing)
        return results


def parse_landmark(s):
    try:
        ident, x, y = s.split(":")
        return int(ident), (float(x), float(y))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "Use ID:X_CM:Y_CM, e.g. 9:0:0"
        ) from exc

def initialize_ring(landmark_xy, measured_distance, measured_bearing, n, sigma_dist, sigma_bearing):
    """Draw particles conditioned on the FIRST observation of one landmark.

    For any azimuth phi around the landmark, pose x,y is set on a noisy ring.
    Robot heading theta is chosen to make the observed bearing consistent.
    This avoids degeneracy from rejection of a small uniform initial sample.
    """
    lx, ly = landmark_xy
    phi = np.random.uniform(-np.pi, np.pi, n)
    radius = np.maximum(1.0, np.random.normal(measured_distance, sigma_dist, n))
    xx = lx + radius * np.cos(phi)
    yy = ly + radius * np.sin(phi)
    direction_to_landmark = np.arctan2(ly - yy, lx - xx)
    theta = mcl.wrap_angle(direction_to_landmark - measured_bearing +
                           np.random.normal(0, sigma_bearing, n))
    return [particle.Particle(float(x), float(y), float(t), 1.0/n)
            for x, y, t in zip(xx, yy, theta)]


def rotate_only(particles, commanded_degrees, extra_sigma_deg):
    """Rotate headings, not x/y. Uncertainty models open-loop turning error."""
    delta = np.deg2rad(commanded_degrees)
    noise = np.deg2rad(extra_sigma_deg)
    for p in particles:
        p.setTheta(float(mcl.wrap_angle(p.getTheta() + delta +
                                        np.random.normal(0, noise))))


def snapshot(particles, landmarks, name, folder):
    """Store positions, headings, and relative weights; never mutate particles."""
    dest = Path(folder)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.png"
    arr = np.array([[p.getX(), p.getY(), p.getTheta()] for p in particles])
    weights = np.array([p.getWeight() for p in particles])
    weights = weights / weights.sum()
    ess = 1.0 / np.sum(weights**2)
    fig, ax = plt.subplots(figsize=(7, 6))
    indices = np.linspace(0, len(arr)-1, min(len(arr), 8000), dtype=int)
    sc = ax.scatter(arr[indices, 0], arr[indices, 1], s=3.5,
                    c=weights[indices]/weights.max(), cmap="viridis",
                    vmin=0, vmax=1, alpha=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, label="Relative weight")
    arrows = indices[::max(1, len(indices)//70)]
    ax.quiver(arr[arrows, 0], arr[arrows, 1],
              12*np.cos(arr[arrows, 2]), 12*np.sin(arr[arrows, 2]),
              angles="xy", scale_units="xy", scale=1, width=0.0028, color="tab:orange")
    for ident, (x, y) in landmarks.items():
        ax.plot(x, y, "ks", markersize=9)
        ax.annotate(str(ident), (x, y), xytext=(6, 7), textcoords="offset points")
    ax.set(title=f"{name} | N={len(particles)}, ESS={ess:.0f}", xlabel="x (cm)", ylabel="y (cm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print(f"  snapshot: {path}", flush=True)


def capture_observations(camera, observer, frames=3, pause=0.10):
    """Return detections from just ONE frame, never merge IDs across frames."""
    obs = {}
    for _ in range(frames):
        obs = observer.observe(camera.gray())
        time.sleep(pause)
        if obs:
            break
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmark", action="append", type=parse_landmark, required=True,
                        help="Known marker coordinates ID:X:Y in cm (repeat at least twice)")
    parser.add_argument("--first-id", type=int, required=True)
    parser.add_argument("--marker-cm", type=float, required=True,
                        help="Actual printed ArUco square side length in cm")
    parser.add_argument("--fx", type=float, required=True, help="Calibrated focal length in px at image resolution")
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--particles", type=int, default=12000)
    parser.add_argument("--step", type=float, default=10.0)
    parser.add_argument("--speed", type=float, default=60)
    parser.add_argument("--max-turn", type=float, default=360.0)
    parser.add_argument("--turn-sign", type=int, choices=(-1, 1), default=1,
                        help="Set -1 if positive arlo.turn() physically rotates clockwise")
    parser.add_argument("--turn-scale", type=float, default=1.0,
                        help="Multiplier if actual rotation is systematically larger/smaller than commanded")
    parser.add_argument("--turn-noise-deg", type=float, default=3.0,
                        help="Heading uncertainty added at EACH turn command")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--folder", default="pf_test")
    args = parser.parse_args()

    landmarks = dict(args.landmark)
    if len(landmarks) < 2 or args.first_id not in landmarks:
        parser.error("Need at least two DIFFERENT known landmarks including --first-id")
    if min(args.step, args.speed, args.marker_cm, args.fx) <= 0 or args.particles <= 0:
        parser.error("step, speed, marker size, fx and particle count must be positive")
    if args.save_every <= 0:
        parser.error("--save-every must be >= 1")
    cx = args.cx if args.cx is not None else args.width/2
    cy = args.cy if args.cy is not None else args.height/2

    observer = MarkerObserver(args.fx, args.fy or args.fx, cx, cy, args.marker_cm)
    camera = Camera(args.camera_index, args.width, args.height)
    arlo = None
    try:
        print(f"Camera: {camera.mode} | landmarks={landmarks}", flush=True)
        input(f"Point robot at marker {args.first_id}, with clear space for turning. Press Enter... ")
        first = None
        for _ in range(15):
            obs = capture_observations(camera, observer, frames=1, pause=0.1)
            if args.first_id in obs:
                first = obs[args.first_id]
                break
        if first is None:
            print(f"Could not detect first marker {args.first_id}. Check calibration, marker visibility, lighting.")
            return
        d1, b1 = first
        print(f"FIRST marker={args.first_id}: distance={d1:.1f}cm, bearing={np.rad2deg(b1):.1f}deg", flush=True)
        particles = initialize_ring(
            landmarks[args.first_id], d1, b1, args.particles,
            mcl.SIGMA_DISTANCE, mcl.SIGMA_BEARING
        )
        snapshot(particles, landmarks, "01_after_first_marker", args.folder)

        arlo = Arlo()
        moved = 0.0
        count = 0
        second = None
        while moved < args.max_turn - 1e-6:
            degrees = min(args.step, args.max_turn - moved)
            print(f"TURN {count+1}: command {degrees:.1f} deg; total {moved+degrees:.1f} deg", flush=True)
            arlo.turn(degrees, speed_degs=args.speed, debug=False).wait()
            # Motion update assumes actual rotation sign is --turn-sign * commanded.
            rotate_only(particles, args.turn_sign*args.turn_scale*degrees, args.turn_noise_deg)
            moved += degrees
            count += 1
            time.sleep(0.35)   # capture only when stationary
            obs = capture_observations(camera, observer, frames=3)
            visible = sorted(set(obs) & set(landmarks))
            print(f"  Visible: {visible}", flush=True)
            candidates = [mid for mid in visible if mid != args.first_id]
            if candidates:
                second_id = candidates[0]
                second = (second_id, obs[second_id])
                snapshot(particles, landmarks, "02_before_second_update", args.folder)
                break
            if count == 1 or count % args.save_every == 0:
                snapshot(particles, landmarks, f"turn_{count:02d}_{moved:.0f}deg", args.folder)

        if second is None:
            print("No second landmark detected after scanning. Saved earlier snapshots.")
            return

        second_id, (d2, b2) = second
        print(f"SECOND marker={second_id}: distance={d2:.1f}cm, bearing={np.rad2deg(b2):.1f}deg", flush=True)
        # The first observation is already built into the ring initialization.
        # Apply likelihood ONLY for second observation. DO NOT resample beforehand.
        ok = mcl.measurement_update(particles, [second_id], [d2], [b2], landmarks)
        if not ok:
            print("Second measurement update returned False (all particle weights zero).")
            print("Possible causes: incorrect rotation sign/angle, camera calibration, landmark coords, or too few particles.")
            return
        snapshot(particles, landmarks, "03_after_second_weighting", args.folder)
        weights = np.array([p.getWeight() for p in particles], dtype=float)
        ess = 1.0 / np.sum(weights**2)
        imax = int(np.argmax(weights))
        best = particles[imax]
        print(f"ESS={ess:.1f}/{len(particles)} | max-weight pose BEFORE resampling: "
              f"x={best.getX():.1f}cm y={best.getY():.1f}cm "
              f"theta={np.rad2deg(best.getTheta()):.1f}deg", flush=True)
        particles = mcl.resample_particles(particles)
        snapshot(particles, landmarks, "04_after_resampling", args.folder)
        print(f"Done. View PNG files in: {Path(args.folder).resolve()}")
    except KeyboardInterrupt:
        print("Interrupted by user", flush=True)
    finally:
        if arlo is not None:
            arlo.stop()
        camera.close()


if __name__ == "__main__":
   main()

