"""Build a low-poly collision mesh (GLB) from a Gaussian-splat PLY.

Usage:
    python collider.py scene.ply collision.glb [--voxel 0.05] [--faces 30000]
        [--no-floor] [--align align.json]

Assumes the splat is in the frame written by sfm_align.py (+Y up, floor at y=0,
metres). It voxelises the solid Gaussians, runs marching cubes, decimates, and
adds a flat floor slab under the room, because floors are often textureless and
full of holes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData
from scipy import ndimage
from skimage import measure


def load_solid_gaussians(path: Path, min_opacity: float, max_size: float) -> np.ndarray:
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    size = np.exp(np.max(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1), 1))
    keep = (opacity >= min_opacity) & (size <= max_size) & np.all(np.isfinite(xyz), 1)
    print(f"{keep.sum():,} of {len(xyz):,} Gaussians are solid enough for collision")
    return xyz[keep]


def build(xyz, voxel, min_count, bounds=None):
    if bounds is None:
        lo, hi = np.percentile(xyz, 0.5, 0), np.percentile(xyz, 99.5, 0)
    else:
        lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
    pad = 3 * voxel
    lo, hi = lo - pad, hi + pad
    xyz = xyz[np.all((xyz >= lo) & (xyz <= hi), 1)]
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    idx = np.floor((xyz - lo) / voxel).astype(int)
    grid = np.zeros(dims, np.int32)
    np.add.at(grid, tuple(idx.T), 1)
    occ = grid >= min_count
    if occ.sum() < 0.002 * occ.size and min_count > 1:
        print("Sparse splat: counting every voxel with at least one Gaussian as solid")
        occ = grid >= 1
    occ = ndimage.binary_closing(occ, iterations=1)  # seal pinholes
    labels, n = ndimage.label(occ)
    if n > 1:  # drop tiny floating islands
        sizes = ndimage.sum(occ, labels, range(1, n + 1))
        big = 1 + np.flatnonzero(sizes >= 20)
        if len(big):
            occ = np.isin(labels, big)
    if not occ.any():
        print("WARNING: no solid voxels found; the collider will only contain the floor")
        return None, lo, hi
    padded = np.pad(occ, 1).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(padded, 0.5)
    verts = (verts - 1) * voxel + lo
    return trimesh.Trimesh(verts, faces, process=True), lo, hi


def floor_slab(lo, hi, thickness=0.1):
    size = [hi[0] - lo[0], thickness, hi[2] - lo[2]]
    box = trimesh.creation.box(extents=size)
    box.apply_translation([(lo[0] + hi[0]) / 2, -thickness / 2, (lo[2] + hi[2]) / 2])
    return box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.05, help="voxel size in metres")
    ap.add_argument("--min-count", type=int, default=2, help="Gaussians per voxel to count as solid")
    ap.add_argument("--min-opacity", type=float, default=0.3)
    ap.add_argument("--max-size", type=float, default=0.25, help="ignore Gaussians bigger than this (m)")
    ap.add_argument("--faces", type=int, default=30000, help="target triangle count")
    ap.add_argument("--no-floor", action="store_true")
    ap.add_argument("--align", default=None, help="align.json from sfm_align.py (room bounds)")
    a = ap.parse_args()

    xyz = load_solid_gaussians(Path(a.ply), a.min_opacity, a.max_size)
    bounds = None
    if a.align and Path(a.align).exists():
        info = json.loads(Path(a.align).read_text())
        lo, hi = np.array(info["bounds_min"]), np.array(info["bounds_max"])
        margin = 0.15 * (hi - lo)
        bounds = (lo - margin, hi + margin)
    mesh, lo, hi = build(xyz, a.voxel, a.min_count, bounds)
    parts = []
    if mesh is not None:
        print(f"Marching cubes: {len(mesh.faces):,} triangles")
        if len(mesh.faces) > a.faces:
            mesh = mesh.simplify_quadric_decimation(face_count=a.faces)
            print(f"Decimated to {len(mesh.faces):,} triangles")
        parts.append(mesh)
    if not a.no_floor or not parts:
        parts.append(floor_slab(lo, hi))
    out = trimesh.util.concatenate(parts)
    out.export(a.out)
    print(f"Wrote {a.out}: {len(out.faces):,} triangles, extent {np.round(out.extents, 2)} m")


if __name__ == "__main__":
    main()
