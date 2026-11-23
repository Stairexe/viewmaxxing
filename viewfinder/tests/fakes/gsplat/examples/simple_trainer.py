"""Fake gsplat trainer: checks the dataset loads like gsplat would, then emits a splat."""
import argparse, os, shutil, sys
import numpy as np, pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("strategy"); ap.add_argument("--data_dir"); ap.add_argument("--data_factor"); ap.add_argument("--result_dir")
ap.add_argument("--no-normalize_world_space", action="store_true"); ap.add_argument("--save_ply", action="store_true")
ap.add_argument("--disable_viewer", action="store_true"); ap.add_argument("--steps_scaler"); ap.add_argument("--test_every")
a = ap.parse_args()
rec = pycolmap.Reconstruction(os.path.join(a.data_dir, "sparse/0"))
names = {im.name for im in rec.images.values()}
assert names == set(os.listdir(os.path.join(a.data_dir, "images"))), "images/ and COLMAP disagree"
os.makedirs(os.path.join(a.result_dir, "ply"), exist_ok=True)
src = os.environ["FAKE_SPLAT"]
shutil.copy(src, os.path.join(a.result_dir, "ply", "point_cloud_7499.ply"))
print("fake gsplat:", len(names), "views")
