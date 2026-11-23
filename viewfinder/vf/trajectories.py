"""Camera-path presets for the "imagine" stage, written in Lyra 2.0's trajectory format.

Lyra expects an .npz with:
  w2c          (N, 4, 4) world-to-camera, OpenCV axes (x right, y down, z forward)
  intrinsics   (N, 3, 3) pixels, for an image of image_height x image_width
The first pose should be the identity (the anchor photo's own camera). Translations are
in metres relative to the anchor, because Lyra scales the anchor's depth to metric with MoGe.

Frame counts must be 1 + 80k (81, 161, 241, ...) to match Lyra's autoregressive chunks.
"""

from __future__ import annotations

import numpy as np

PRESETS = {
    "look_around": "Stand still and turn your head left, then right (reveals the sides of the room)",
    "turn_around": "Slow 360° turn while stepping around a small circle (reveals what's behind you)",
    "walk_forward": "Walk forward into the scene, swaying slightly",
    "pull_back": "Step backwards to reveal more of the surroundings",
    "truck_right": "Slide sideways to the right while facing forward",
    "orbit": "Circle around whatever is in the middle of the photo",
}


def _ease(t: np.ndarray) -> np.ndarray:
    """Smooth start and stop (cosine ease-in-out), t in [0, 1]."""
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _yaw_rotation(theta: float) -> np.ndarray:
    """Camera-to-world rotation for turning right by theta radians about the camera's up axis.

    Columns are the camera's right, down and forward axes in anchor coordinates.
    """
    c, s = np.cos(theta), np.sin(theta)
    right = np.array([c, 0.0, -s])
    down = np.array([0.0, 1.0, 0.0])
    forward = np.array([s, 0.0, c])
    return np.stack([right, down, forward], axis=1)


def _c2w(position: np.ndarray, yaw: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _yaw_rotation(yaw)
    m[:3, 3] = position
    return m


def valid_frame_count(n: int) -> int:
    """Round up to the nearest 1 + 80k."""
    k = max(1, int(np.ceil((n - 1) / 80)))
    return 1 + 80 * k


def make_trajectory(
    preset: str,
    num_frames: int,
    K: np.ndarray,
    width: int,
    height: int,
    distance: float = 1.0,
    degrees: float = 60.0,
    orbit_depth: float = 2.0,
) -> dict:
    """Build a camera path.

    distance    metres moved (walk/pull/truck) or circle radius (turn_around)
    degrees     how far to turn (look_around: each side; orbit: total arc)
    orbit_depth how far in front of the anchor camera the orbit centre is (metres)
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    n = valid_frame_count(num_frames)
    t = np.linspace(0.0, 1.0, n)
    e = _ease(t)
    rad = np.radians(degrees)
    poses = []

    for i in range(n):
        if preset == "look_around":
            # 0 -> +rad -> -rad -> 0, as one smooth sine wave
            yaw = rad * np.sin(2 * np.pi * t[i])
            pos = np.array([0.0, 0.0, 0.05 * np.sin(np.pi * t[i])])
        elif preset == "turn_around":
            # Walk a circle of radius `distance` that starts and ends at the anchor
            # position, always looking outward, for a full 360°.
            yaw = 2 * np.pi * e[i]
            centre = np.array([0.0, 0.0, -distance])  # behind the anchor
            pos = centre + distance * np.array([np.sin(yaw), 0.0, np.cos(yaw)])
        elif preset == "walk_forward":
            yaw = np.radians(8.0) * np.sin(2 * np.pi * t[i])
            pos = np.array([0.0, 0.0, distance * e[i]])
        elif preset == "pull_back":
            yaw = 0.0
            pos = np.array([0.0, 0.0, -distance * e[i]])
        elif preset == "truck_right":
            yaw = 0.0
            pos = np.array([distance * e[i], 0.0, 0.0])
        elif preset == "orbit":
            # Circle around a point orbit_depth metres straight ahead, keep looking at it.
            ang = rad * e[i]  # from the anchor (0) round to +rad
            centre = np.array([0.0, 0.0, orbit_depth])
            pos = centre + orbit_depth * np.array([-np.sin(ang), 0.0, -np.cos(ang)])
            yaw = ang
        poses.append(_c2w(pos, yaw))

    c2w = np.stack(poses)
    c2w[0] = np.eye(4)  # exactly the anchor camera
    w2c = np.linalg.inv(c2w).astype(np.float32)
    intr = np.repeat(np.asarray(K, np.float32)[None], n, axis=0)
    return {
        "w2c": w2c,
        "intrinsics": intr,
        "image_height": np.int64(height),
        "image_width": np.int64(width),
    }


def save_trajectory(path, traj: dict) -> None:
    np.savez(path, **traj)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write a Lyra 2.0 camera path (.npz)")
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=161)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=70.0, help="horizontal field of view (deg) if no K")
    ap.add_argument("--distance", type=float, default=1.0)
    ap.add_argument("--degrees", type=float, default=60.0)
    a = ap.parse_args()
    f = 0.5 * a.width / np.tan(np.radians(a.fov) / 2)
    K = np.array([[f, 0, a.width / 2], [0, f, a.height / 2], [0, 0, 1]])
    tr = make_trajectory(a.preset, a.frames, K, a.width, a.height, a.distance, a.degrees)
    save_trajectory(a.out, tr)
    print(f"wrote {a.out}: {len(tr['w2c'])} frames")
