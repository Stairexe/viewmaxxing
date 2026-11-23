"""Small Gaussian-splat PLY helpers (numpy only)."""

from __future__ import annotations

import numpy as np
from numpy.lib import recfunctions as rfn
from plyfile import PlyData, PlyElement


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, quaternions as (w, x, y, z)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, np.float64)
    return q / np.linalg.norm(q)


def transform_splat(src: str, dst: str, scale: float, R, t) -> int:
    """Apply x' = scale * R @ x + t to a 3DGS PLY (positions, rotations, log-scales).

    Only valid for degree-0 colour (f_dc_*). Higher SH bands would also need rotating,
    so they are dropped with a warning.
    """
    R = np.asarray(R, np.float64)
    t = np.asarray(t, np.float64)
    ply = PlyData.read(src)
    v = ply["vertex"].data.copy()
    names = v.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    xyz = scale * xyz @ R.T + t
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if all(f"rot_{i}" in names for i in range(4)):
        q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        q = _quat_mul(_matrix_to_quat(R)[None], q)
        for i in range(4):
            v[f"rot_{i}"] = q[:, i]
    for i in range(3):
        if f"scale_{i}" in names:
            v[f"scale_{i}"] = v[f"scale_{i}"] + np.log(scale)
    rest = [n for n in names if n.startswith("f_rest_")]
    if rest:
        print(f"note: dropping {len(rest)} higher-order SH coefficients (not rotated)")
        keep = [n for n in names if n not in rest]
        v = rfn.repack_fields(v[keep])
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(dst)
    return len(v)
