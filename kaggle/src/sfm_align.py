"""Camera poses with COLMAP, then put the room in a game-friendly frame.

Usage:
    python sfm_align.py --images data/images --work work --out data/sparse/0
        [--matcher sequential|exhaustive] [--camera-height 1.4] [--known-scale S]

Output world frame (so splats, colliders and engines agree):
  * +Y is up, the floor is at y = 0
  * units are approximately metres. Scale is estimated by assuming you held the
    phone about --camera-height metres above the floor. Pass --known-scale to override.
  * walls are roughly aligned with the X and Z axes
Writes align.json next to --out with a seed point for the collider/player spawn.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pycolmap


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:  # opposite: rotate 180 deg about any perpendicular axis
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 0, 1.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def estimate_floor(y: np.ndarray, cam_y: float) -> float:
    """Floor height: a dense horizontal layer of points well below the cameras.

    If the floor was filmed, its points form a clear peak at the bottom of the
    height histogram. If not, fall back to a low percentile.
    """
    lo = np.percentile(y, 0.5)
    below = y[(y >= lo) & (y < cam_y - 0.4 * (cam_y - lo))]
    if len(below) > 200:
        hist, edges = np.histogram(below, bins=40)
        peak = int(np.argmax(hist))
        if hist[peak] > 4 * max(np.median(hist), 1):
            c = 0.5 * (edges[peak] + edges[peak + 1])
            w = edges[1] - edges[0]
            print("Floor: found a dense floor layer")
            return float(np.median(below[np.abs(below - c) <= w]))
    print("Floor: no clear floor layer (was the floor filmed?), using the lowest points")
    return float(np.percentile(y, 1))


def run_sfm(images: Path, work: Path, matcher: str) -> pycolmap.Reconstruction:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    pycolmap.extract_features(
        db, images, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader
    )
    if matcher == "sequential":
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = 20
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing)
    else:
        pycolmap.match_exhaustive(db)

    recs = pycolmap.incremental_mapping(db, images, work / "sparse")
    if not recs:
        raise SystemExit(
            "COLMAP could not reconstruct the scene. Usual causes: motion blur, blank "
            "walls, moving too fast, or too few overlapping frames."
        )
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    n_img = len(list(images.iterdir()))
    print(
        f"Registered {best.num_reg_images()}/{n_img} images, {best.num_points3D()} points "
        f"({len(recs)} model(s))"
    )
    if best.num_reg_images() < 0.5 * n_img:
        print("WARNING: under half the frames registered. The splat will be missing parts of the room.")
    return best


def align(rec: pycolmap.Reconstruction, camera_height: float, known_scale):
    imgs = [rec.image(i) for i in rec.reg_image_ids()]
    centers = np.array([im.projection_center() for im in imgs])
    # OpenCV camera: +Y points down in the image, so camera "up" in world = R^T @ (0,-1,0)
    ups = np.array([im.cam_from_world().rotation.matrix().T @ np.array([0, -1.0, 0]) for im in imgs])
    up = ups.mean(0)
    R1 = rotation_between(up, np.array([0, 1.0, 0]))

    pts = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3])
    if len(pts) < 100:
        pts = np.array([p.xyz for p in rec.points3D.values()])
    pts_r, cams_r = pts @ R1.T, centers @ R1.T

    floor_y = estimate_floor(pts_r[:, 1], np.median(cams_r[:, 1]))
    cam_h = np.median(cams_r[:, 1]) - floor_y
    if known_scale:
        scale = float(known_scale)
    elif cam_h > 1e-6:
        scale = camera_height / cam_h
    else:
        print("WARNING: could not find the floor below the cameras; leaving scale at 1.")
        scale = 1.0

    # Yaw: line the dominant horizontal direction of the points up with +X.
    xz = pts_r[:, [0, 2]]
    lo, hi = np.percentile(xz, 5, axis=0), np.percentile(xz, 95, axis=0)
    inner = xz[np.all((xz >= lo) & (xz <= hi), axis=1)]
    evals, evecs = np.linalg.eigh(np.cov((inner - inner.mean(0)).T))
    major = evecs[:, np.argmax(evals)]
    yaw = -np.arctan2(major[1], major[0])  # rotate major axis (x,z) onto +X
    c, s = np.cos(yaw), np.sin(yaw)
    R2 = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    R = R2 @ R1

    cams_R = centers @ R.T
    center_xz = np.median(cams_R[:, [0, 2]], axis=0)
    t = -scale * np.array([center_xz[0], floor_y, center_xz[1]])
    rec.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(R), t))
    sim3 = {"scale": float(scale), "R": R.tolist(), "t": t.tolist()}  # new = scale * R @ old + t

    # Spawn/seed: the camera position nearest the middle of the capture path, at 1 m.
    cams = np.array([rec.image(i).projection_center() for i in rec.reg_image_ids()])
    mid = np.median(cams[:, [0, 2]], axis=0)
    nearest = cams[np.argmin(np.linalg.norm(cams[:, [0, 2]] - mid, axis=1))]
    pts_new = np.array([p.xyz for p in rec.points3D.values()])
    lo, hi = np.percentile(pts_new, 2, axis=0), np.percentile(pts_new, 98, axis=0)
    return {
        "scale_applied": scale,
        "scale_source": "known_scale" if known_scale else f"camera_height={camera_height}m",
        "seed": [float(nearest[0]), 1.0, float(nearest[2])],
        "camera_height_median": float(np.median(cams[:, 1])),
        "bounds_min": lo.tolist(),
        "bounds_max": hi.tolist(),
        "num_images": rec.num_reg_images(),
        "num_points": rec.num_points3D(),
        "sim3": sim3,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    ap.add_argument("--camera-height", type=float, default=1.4)
    ap.add_argument("--known-scale", type=float, default=None)
    a = ap.parse_args()

    rec = run_sfm(Path(a.images), Path(a.work), a.matcher)
    info = align(rec, a.camera_height, a.known_scale)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rec.write(out)
    (out.parent.parent / "align.json").write_text(json.dumps(info, indent=2))
    size = np.array(info["bounds_max"]) - np.array(info["bounds_min"])
    print(f"Room is roughly {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m (W x H x D)")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
