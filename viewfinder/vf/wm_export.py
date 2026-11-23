"""Turn a WorldMirror 2.0 result into full-resolution training views for splat refinement.

WorldMirror resizes and center-crops every image to a multiple of 14 px before
inference, and its camera_params.json intrinsics refer to that small cropped
image. This maps them back onto a high-resolution crop of each original photo,
so gsplat can refine the splat with sharp pixels.

Usage (any env with numpy + pillow; runs inside the hyworld2 env on the GPU box):
    python wm_export.py --images IMG_DIR --recon RECON_DIR --target-size 952 --out OUT_DIR

Writes OUT_DIR/images/<name>.jpg and OUT_DIR/cameras.json:
    [{"name", "width", "height", "K": 3x3, "c2w": 4x4 (OpenCV)}, ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

PATCH = 14


# --- Must match hyworld2/worldrecon/hyworldmirror/utils/inference_utils.py -------------
def list_images(image_dir: str) -> list[str]:
    paths = []
    for ext in ["*.jpeg", "*.jpg", "*.png", "*.webp"]:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def adaptive_target_size(first_w: int, first_h: int, max_target: int) -> int:
    eff = min(max(first_w, first_h), max_target)
    eff = (eff // PATCH) * PATCH
    return max(eff, PATCH * 2)


def resize_dims(orig_w: int, orig_h: int, target: int) -> tuple[int, int]:
    if orig_w >= orig_h:
        new_w = target
        new_h = round(orig_h * (new_w / orig_w) / PATCH) * PATCH
    else:
        new_h = target
        new_w = round(orig_w * (new_h / orig_h) / PATCH) * PATCH
    return new_w, new_h


def crop_transform(orig_w: int, orig_h: int, target: int) -> dict:
    new_w, new_h = resize_dims(orig_w, orig_h, target)
    crop_x = (new_w - target) // 2 if new_w > target else 0
    crop_y = (new_h - target) // 2 if new_h > target else 0
    return {
        "sx": new_w / orig_w, "sy": new_h / orig_h,
        "crop_x": crop_x, "crop_y": crop_y,
        "W": min(new_w, target), "H": min(new_h, target),
    }
# -----------------------------------------------------------------------------------------


def to_full_res(K_inf: np.ndarray, t: dict, max_size: int):
    """Map inference-crop intrinsics to a crop of the original image.

    Inference pixel u_inf = sx * u_orig - crop_x, so the matching region of the original
    starts at crop_x / sx and the intrinsics just divide by the resize factors.
    Returns (crop box in original pixels, K for that crop, final output size).
    """
    x0, y0 = t["crop_x"] / t["sx"], t["crop_y"] / t["sy"]
    w, h = t["W"] / t["sx"], t["H"] / t["sy"]
    K = K_inf.astype(np.float64).copy()
    K[0, 0] /= t["sx"]; K[0, 2] /= t["sx"]
    K[1, 1] /= t["sy"]; K[1, 2] /= t["sy"]
    scale = min(1.0, max_size / max(w, h))
    out_w, out_h = int(round(w * scale)), int(round(h * scale))
    K[0, 0] *= out_w / w; K[0, 2] *= out_w / w
    K[1, 1] *= out_h / h; K[1, 2] *= out_h / h
    return (x0, y0, x0 + w, y0 + h), K, (out_w, out_h)


def check_against_worldmirror(paths, target):
    """If the HY-World code is importable, make sure our crop math matches theirs."""
    try:
        from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
            compute_preprocessing_transform,
        )
    except Exception:
        return
    theirs = compute_preprocessing_transform(paths, target)
    w, h = Image.open(paths[0]).size
    ours = crop_transform(w, h, target)
    assert (ours["crop_x"], ours["crop_y"], ours["W"], ours["H"]) == (
        theirs["crop_x"], theirs["crop_y"], theirs["final_w"], theirs["final_h"]
    ), f"crop mismatch: ours={ours} theirs={theirs}"
    print("crop math matches WorldMirror's preprocessing")


def export(image_dir: str, recon_dir: str, target_size: int, out_dir: str, max_size: int = 1600) -> list:
    paths = list_images(image_dir)
    cams = json.loads((Path(recon_dir) / "camera_params.json").read_text())
    extr = {int(e["camera_id"]): np.array(e["matrix"]) for e in cams["extrinsics"]}
    intr = {int(e["camera_id"]): np.array(e["matrix"]) for e in cams["intrinsics"]}
    if len(paths) != len(extr):
        raise SystemExit(f"{len(paths)} images but {len(extr)} cameras: inputs changed since inference?")

    w0, h0 = Image.open(paths[0]).size
    target = adaptive_target_size(w0, h0, target_size)
    check_against_worldmirror(paths, target_size)

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    records = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        t = crop_transform(img.width, img.height, target)
        K_inf = intr[i]
        # Sanity: predicted principal point should sit near the middle of the inference crop.
        if abs(K_inf[0, 2] - t["W"] / 2) > 0.15 * t["W"] or abs(K_inf[1, 2] - t["H"] / 2) > 0.15 * t["H"]:
            print(f"WARNING {Path(p).name}: principal point {K_inf[:2, 2]} far from crop centre "
                  f"{t['W'] / 2, t['H'] / 2}. Intrinsics may not be in inference pixels.")
        box, K, size = to_full_res(K_inf, t, max_size)
        name = Path(p).stem + ".jpg"
        img.crop(tuple(int(round(v)) for v in box)).resize(size, Image.Resampling.LANCZOS).save(
            out / "images" / name, quality=95)
        records.append({"name": name, "width": size[0], "height": size[1],
                        "K": K.tolist(), "c2w": extr[i].tolist()})
    (out / "cameras.json").write_text(json.dumps(records, indent=1))
    pts = Path(recon_dir) / "points.ply"
    if pts.exists():
        shutil.copy(pts, out / "points.ply")
    print(f"exported {len(records)} views ({records[0]['width']}x{records[0]['height']}) to {out}")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--target-size", type=int, default=952)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-size", type=int, default=1600)
    a = ap.parse_args()
    export(a.images, a.recon, a.target_size, a.out, a.max_size)
