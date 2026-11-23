"""Fake WorldMirror 2.0: cameras on a circle in a 4x2.5x3 m box room (arbitrary scale/frame)."""
import argparse, glob, json, os
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ap = argparse.ArgumentParser()
ap.add_argument("--input_path"); ap.add_argument("--strict_output_path"); ap.add_argument("--target_size", type=int)
ap.add_argument("--no_interactive", action="store_true"); ap.add_argument("--no_save_normal", action="store_true")
ap.add_argument("--enable_bf16", action="store_true")
a = ap.parse_args()
paths = sorted(sum((glob.glob(os.path.join(a.input_path, e)) for e in ["*.jpeg", "*.jpg", "*.png", "*.webp"]), []))
w, h = Image.open(paths[0]).size
eff = (min(max(w, h), a.target_size) // 14) * 14
if w >= h: W, H = eff, round(h * eff / w / 14) * 14
else: H, W = eff, round(w * eff / h / 14) * 14
S, rng = 0.3, np.random.default_rng(len(paths))   # WorldMirror's frame: unknown scale
ex, inx = [], []
for i, _ in enumerate(paths):
    a_ = 2 * np.pi * i / len(paths)
    C = np.array([0.8 * np.cos(a_), -1.4, 0.6 * np.sin(a_)])     # y down = OpenCV-ish world
    fwd = np.array([np.cos(a_ + 0.5), 0.1, np.sin(a_ + 0.5)]); fwd /= np.linalg.norm(fwd)
    right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right); down = np.cross(fwd, right)
    c2w = np.eye(4); c2w[:3, :3] = np.stack([right, down, fwd], 1); c2w[:3, 3] = C * S
    ex.append({"camera_id": i, "matrix": c2w.tolist()})
    f = 0.8 * W
    inx.append({"camera_id": i, "matrix": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]})
os.makedirs(a.strict_output_path, exist_ok=True)
json.dump({"num_cameras": len(paths), "extrinsics": ex, "intrinsics": inx},
          open(os.path.join(a.strict_output_path, "camera_params.json"), "w"))
# Points/gaussians on the room box (floor y=0 -> y=-0 in "down" world, ceiling at -2.5)
P = []
for _ in range(6000):
    face = rng.integers(6); p = rng.uniform([-2, -2.5, -1.5], [2, 0, 1.5])
    if face == 0: p[1] = 0
    elif face == 1: p[1] = -2.5
    elif face == 2: p[0] = -2
    elif face == 3: p[0] = 2
    elif face == 4: p[2] = -1.5
    else: p[2] = 1.5
    P.append(p * S)
P = np.array(P)
v = np.zeros(len(P), dtype=[(n, "f4") for n in ["x", "y", "z"]] + [(n, "u1") for n in ["red", "green", "blue"]])
v["x"], v["y"], v["z"] = P.T; v["red"] = 200
PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(a.strict_output_path, "points.ply"))
g = np.zeros(len(P), dtype=[(n, "f4") for n in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]])
g["x"], g["y"], g["z"] = P.T; g["opacity"] = 3; g["scale_0"] = g["scale_1"] = g["scale_2"] = np.log(0.02 * S); g["rot_0"] = 1
PlyData([PlyElement.describe(g, "vertex")]).write(os.path.join(a.strict_output_path, "gaussians.ply"))
print(f"fake WorldMirror: {len(paths)} views at {W}x{H}")
