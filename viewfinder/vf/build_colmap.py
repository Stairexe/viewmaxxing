"""Build a gsplat-ready COLMAP dataset from wm_export.py output, levelled and in metres.

Usage (vfcore env, pycolmap 4.x):
    python build_colmap.py EXPORT_DIR DATA_DIR [--camera-height 1.4] [--known-scale S]
        [--max-points 300000]

DATA_DIR gets images/, sparse/0/ (binary COLMAP) and align.json, in the same world
frame as the Kaggle pipeline: +Y up, floor at y = 0, roughly metres.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pycolmap
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kaggle" / "src"))
from sfm_align import align  # noqa: E402  (shared levelling + metric-scale logic)


def load_points(ply: Path, max_points: int, seed: int = 0):
    if not ply.exists():
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    v = PlyData.read(str(ply))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    names = v.dtype.names
    if {"red", "green", "blue"} <= set(names):
        rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 128, np.uint8)
    ok = np.all(np.isfinite(xyz), 1)
    xyz, rgb = xyz[ok], rgb[ok]
    if len(xyz) > max_points:
        idx = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def build(export_dir: Path, max_points: int) -> pycolmap.Reconstruction:
    cams = json.loads((export_dir / "cameras.json").read_text())
    rec = pycolmap.Reconstruction()
    for i, c in enumerate(cams, start=1):
        K = np.asarray(c["K"], np.float64)
        cam = pycolmap.Camera(
            model="PINHOLE", width=int(c["width"]), height=int(c["height"]),
            params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]], camera_id=i,
        )
        rec.add_camera_with_trivial_rig(cam)
        w2c = np.linalg.inv(np.asarray(c["c2w"], np.float64))
        image = pycolmap.Image(name=c["name"], camera_id=i, image_id=i)
        rec.add_image_with_trivial_frame(image, pycolmap.Rigid3d(w2c[:3, :]))
    xyz, rgb = load_points(export_dir / "points.ply", max_points)
    for p, col in zip(xyz, rgb):
        rec.add_point3D(p, pycolmap.Track(), col)
    print(f"COLMAP model: {rec.num_reg_images()} images, {rec.num_points3D()} points")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--camera-height", type=float, default=1.4)
    ap.add_argument("--known-scale", type=float, default=None)
    ap.add_argument("--max-points", type=int, default=300_000)
    a = ap.parse_args()

    export_dir, data = Path(a.export_dir), Path(a.data_dir)
    rec = build(export_dir, a.max_points)
    info = align(rec, a.camera_height, a.known_scale)

    if data.exists():
        shutil.rmtree(data)
    (data / "sparse" / "0").mkdir(parents=True)
    shutil.copytree(export_dir / "images", data / "images")
    rec.write(data / "sparse" / "0")
    (data / "align.json").write_text(json.dumps(info, indent=2))
    size = np.array(info["bounds_max"]) - np.array(info["bounds_min"])
    print(f"Scene is roughly {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m (W x H x D)")


if __name__ == "__main__":
    main()
