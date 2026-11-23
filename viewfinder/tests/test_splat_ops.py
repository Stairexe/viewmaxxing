import numpy as np
from plyfile import PlyData, PlyElement

from vf.splat_ops import transform_splat


def quat_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def cov(v, i):
    S = np.diag(np.exp([v[f"scale_{k}"][i] for k in range(3)])) ** 2
    Rq = quat_to_matrix(np.array([v[f"rot_{k}"][i] for k in range(4)], float))
    return Rq @ S @ Rq.T


def test_similarity_transform_moves_and_rotates_gaussians(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(n, dtype=[(f, "f4") for f in fields])
    for f in ("x", "y", "z"):
        v[f] = rng.normal(size=n)
    for k in range(3):
        v[f"scale_{k}"] = np.log(rng.uniform(0.01, 0.2, n))
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    for k in range(4):
        v[f"rot_{k}"] = q[:, k]
    src, dst = tmp_path / "a.ply", tmp_path / "b.ply"
    PlyData([PlyElement.describe(v, "vertex")]).write(str(src))

    ang = np.radians(37)
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]) @ \
        np.array([[1, 0, 0], [0, np.cos(1.1), -np.sin(1.1)], [0, np.sin(1.1), np.cos(1.1)]])
    s, t = 2.5, np.array([1.0, -2.0, 0.5])
    transform_splat(str(src), str(dst), s, R, t)
    w = PlyData.read(str(dst))["vertex"].data

    assert "f_rest_0" not in w.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    xyz2 = np.stack([w["x"], w["y"], w["z"]], 1)
    assert np.allclose(xyz2, s * xyz @ R.T + t, atol=1e-4)
    for i in range(n):  # covariance must transform as s^2 R C R^T
        assert np.allclose(cov(w, i), s * s * R @ cov(v, i) @ R.T, atol=1e-5)
