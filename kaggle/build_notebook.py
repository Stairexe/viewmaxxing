"""Builds room_to_splat.ipynb from the scripts in src/ (run: python build_notebook.py).

The notebook has to be self-contained (Kaggle can't clone a private repo), so the
helper scripts are embedded as %%writefile cells.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


cells = [
    md(r"""
# viewmaxxing: room video → walkable 3D (Gaussian splat + collision mesh)

**What you get** (in `/kaggle/working/output/`, downloadable as `viewmaxxing_<name>.zip`):

| File | Use it for |
|---|---|
| `scene.html` | Double-click it: a walk-around 3D viewer of your room in any browser |
| `scene.spz` | **Blender 5.3** (File → Import → PLY/SPZ), Scaniverse/SuperSplat, Godot add-on |
| `scene.sog` | Web engines (PlayCanvas, Spark/three.js), the smallest file |
| `scene.ply` | Raw splat, works everywhere (big) |
| `collision.glb` | Invisible low-poly walls and floor for physics / walking (import into Blender or Godot) |
| `align.json` | Spawn point, room bounds, scale info |

World frame: **+Y up, floor at y = 0, roughly metres**, walls lined up with the X/Z axes.

## Setup (once)
1. Right panel → **Settings → Accelerator → GPU T4 x2**. Don't pick P100: current PyTorch doesn't support it.
2. **Settings → Internet → On**. This needs a phone-verified Kaggle account.
3. **Add Input → Upload** your room video (or a folder of photos) as a new Dataset. Then put its path in `INPUT_PATH` below.
4. **Run All.** Expect about **25–40 min with `QUALITY="fast"`** and **60–90 min with `"full"`**. That uses about 1–1.5 of your 30 weekly GPU hours.

## How to film (this matters more than any setting)
- Use the **normal camera app in Video mode**, 1×, 4K or 1080p, with **HDR Video OFF** (Settings → Camera → Record Video).
- Light the room evenly. **Move slowly**, because blur kills it. Make two loops around the room: one at chest height looking outward, and one lower **pointing at the floor and furniture**.
- Spend 1–3 minutes per room. Cover every surface from 2–3 angles. Avoid filming mirrors and windows head-on.
"""),
    code(r"""
# ===== SETTINGS: edit these =====
INPUT_PATH      = "/kaggle/input/room-scan"  # a video file, a folder with one video, or a folder of photos
SCENE_NAME      = "my_room"
QUALITY         = "fast"     # "fast" (~7.5k steps) or "full" (30k steps, sharper, ~3x longer)
TARGET_FRAMES   = 200        # frames picked from the video (100-300). More = slower but more complete
MAX_IMAGE_SIZE  = 1600       # long side of frames in px. Lower (1280) if you run out of GPU memory
CAMERA_HEIGHT_M = 1.4        # how high you held the phone; used to guess real-world scale
KNOWN_SCALE     = None       # set a number to override the scale guess (see the scale note at the end)
SEED_POS        = None       # spawn/collider seed [x, y, z]; None = auto (middle of your walking path)
# =================================
"""),
    code(r"""
import os, sys, subprocess, json, shutil, glob, time
from pathlib import Path

W = Path("/kaggle/working"); VM = W / "vm"; DATA = W / "data"; RES = W / "results"; OUT = W / "output"
for p in (VM,): p.mkdir(parents=True, exist_ok=True)

def sh(cmd, cwd=None, env=None, tail=40):
    # Run a shell command, stream output, and fail loudly with the last lines.
    print(f"$ {cmd}")
    t = time.time()
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env={**os.environ, **(env or {})},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if not line.startswith(("I20", "W20")):  # hide COLMAP's chatty glog lines
            print(line, end="")
    if p.wait() != 0:
        raise RuntimeError(f"Command failed ({p.returncode}). Last output:\n" + "".join(lines[-tail:]))
    print(f"  [done in {time.time() - t:.0f}s]")

sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv")
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| GPU OK:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU. Set Settings > Accelerator > GPU T4 x2"
cc = torch.cuda.get_device_capability(0)
assert cc >= (7, 0), f"GPU compute capability {cc} is too old for current PyTorch. Pick GPU T4 x2, not P100."
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}"  # compile CUDA only for this GPU (faster)
"""),
    md("## 1. Install (≈10 min, mostly compiling gsplat's CUDA kernels once)"),
    code(r"""
# gsplat trainer (pinned release) and its example-script dependencies
sh("git clone -q --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git " + str(W / "gsplat")
   if not (W / "gsplat").exists() else "echo gsplat already cloned")
sh("pip install -q gsplat==1.5.3 viser tyro 'torchmetrics[image]' tensorly splines plyfile trimesh "
   "fast-simplification scikit-image imageio[ffmpeg] pillow-heif "
   "'git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787' "
   "'git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5'")

# gsplat's COLMAP reader is a small pure-python 'pycolmap' fork. It clashes by name with the official
# pycolmap (used for SfM), so the official one goes into its own folder and runs in a subprocess.
sh("pip install -q --no-deps 'git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e'")
import importlib.util
sm = Path(importlib.util.find_spec("pycolmap").submodule_search_locations[0]) / "scene_manager.py"
sm.write_text(sm.read_text().replace("np.uint64(-1)", "np.uint64(np.iinfo(np.uint64).max)"))  # numpy 2 fix

SFM_LIBS = W / "sfm_libs"
if not SFM_LIBS.exists():
    try:  # GPU build (fast SIFT + matching); fall back to CPU build if it won't load
        sh(f"pip install -q --target {SFM_LIBS} pycolmap-cuda12==4.2.1")
        sh(f"PYTHONPATH={SFM_LIBS} python -c 'import pycolmap; print(\"pycolmap\", pycolmap.__version__, \"CUDA:\", pycolmap.has_cuda)'")
    except RuntimeError as e:
        print("CUDA pycolmap unavailable, using CPU build:", str(e)[-300:])
        shutil.rmtree(SFM_LIBS, ignore_errors=True)
        sh(f"pip install -q --target {SFM_LIBS} pycolmap==4.2.1")

# Node.js + PlayCanvas splat-transform (compression and HTML viewer)
def node_ok():
    try:
        return int(subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip()[1:].split(".")[0]) >= 18
    except Exception:
        return False
if not node_ok():
    sh("curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | tar -xJ -C /kaggle/working")
    os.environ["PATH"] = "/kaggle/working/node-v22.11.0-linux-x64/bin:" + os.environ["PATH"]
sh("npm install -g -s @playcanvas/splat-transform@3 && splat-transform --version")

# Compile gsplat's CUDA kernels now, so the training cell doesn't look frozen
sh("python -c 'from gsplat.cuda._backend import _C; assert _C is not None, \"gsplat CUDA build failed (is nvcc installed?)\"; print(\"gsplat CUDA kernels ready\")'", tail=80)
"""),
]

for name in ("frames.py", "sfm_align.py", "collider.py"):
    src = (HERE / "src" / name).read_text()
    cells.append(code(f"%%writefile /kaggle/working/vm/{name}\n" + src))

cells += [
    md("## 2. Video → sharp frames"),
    code(r"""
sh(f"python {VM}/frames.py '{INPUT_PATH}' {DATA}/images --target {TARGET_FRAMES} --max-size {MAX_IMAGE_SIZE}")
from PIL import Image
imgs = sorted((DATA / "images").glob("*.jpg"))
strip = Image.new("RGB", (6 * 320, 240))
for i, p in enumerate(imgs[:: max(1, len(imgs) // 6)][:6]):
    strip.paste(Image.open(p).resize((320, 240)), (i * 320, 0))
display(strip)
"""),
    md("## 3. Camera poses (COLMAP) + level the floor + metric scale"),
    code(r"""
is_video = Path(INPUT_PATH).is_file() or not any(
    p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"} for p in Path(INPUT_PATH).rglob("*"))
matcher = "sequential" if is_video else "exhaustive"
extra = f"--known-scale {KNOWN_SCALE}" if KNOWN_SCALE else ""
sh(f"python {VM}/sfm_align.py --images {DATA}/images --work {W}/colmap --out {DATA}/sparse/0 "
   f"--matcher {matcher} --camera-height {CAMERA_HEIGHT_M} {extra}", env={"PYTHONPATH": str(SFM_LIBS)})
ALIGN = json.loads((DATA / "align.json").read_text())
"""),
    md("## 4. Train the Gaussian splat"),
    code(r"""
scaler = 1.0 if QUALITY == "full" else 0.25
shutil.rmtree(RES, ignore_errors=True)
sh(f"python simple_trainer.py default --data_dir {DATA} --data_factor 1 --result_dir {RES} "
   f"--no-normalize_world_space --save_ply --disable_viewer --steps_scaler {scaler} --test_every 100000",
   cwd=W / "gsplat" / "examples", env={"CUDA_VISIBLE_DEVICES": "0"}, tail=60)
plys = sorted(glob.glob(f"{RES}/ply/point_cloud_*.ply"), key=lambda p: int(Path(p).stem.split("_")[-1]))
PLY = plys[-1]
print("Final splat:", PLY, f"{os.path.getsize(PLY) / 1e6:.0f} MB")
renders = sorted(glob.glob(f"{RES}/renders/*.png"))
if renders:
    display(Image.open(renders[-1]))  # left: render, right: real photo
"""),
    md("## 5. Export: compressed splats, HTML viewer, collision mesh"),
    code(r"""
shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir()
shutil.copy(PLY, OUT / "scene.ply")
# Drop broken / near-invisible Gaussians, then write the compressed formats.
sh(f"splat-transform -w {OUT}/scene.ply --filter-nan --filter-value opacity,gt,0.02 {OUT}/clean.ply")
for ext in ("spz", "sog", "html"):  # falls back to the CPU encoder automatically when there's no WebGPU
    sh(f"splat-transform -w {OUT}/clean.ply {OUT}/scene.{ext}")
shutil.move(OUT / "clean.ply", OUT / "scene.ply")

seed = SEED_POS or ALIGN["seed"]
ALIGN["seed"] = seed
(OUT / "align.json").write_text(json.dumps(ALIGN, indent=2))
sh(f"python {VM}/collider.py {OUT}/scene.ply {OUT}/collision.glb --align {OUT}/align.json")

# Optional higher-quality collider from PlayCanvas (needs WebGPU/Vulkan; often missing on Kaggle)
try:
    sh(f"splat-transform -w {OUT}/scene.ply --seed-pos {seed[0]},{seed[1]},{seed[2]} "
       f"--voxel-external-fill --voxel-carve --collision-mesh {OUT}/scene.voxel.json", tail=8)
    print("Bonus: scene.collision.glb (PlayCanvas carve) written too")
except RuntimeError:
    print("(PlayCanvas voxel collider skipped: no WebGPU here. collision.glb from collider.py is fine.)")

zip_path = shutil.make_archive(str(W / f"viewmaxxing_{SCENE_NAME}"), "zip", OUT)
for f in sorted(OUT.iterdir()):
    print(f"{f.name:28s} {f.stat().st_size / 1e6:8.1f} MB")
print("\nDownload:", zip_path, "(right panel → Output, or the file browser)")
"""),
    md(r"""
## 6. What to do with it
- **See it now:** unzip, then double-click `scene.html` and drag to look around.
- **Blender 5.3+:** File → Import → **PLY** (`scene.ply`) or the **SPZ** importer (`scene.spz`). Set **Up axis = Y** in the import options, because this file is Y-up. Then File → Import → glTF → `collision.glb`, set it to invisible in renders, and use it for physics or nav.
- **Godot:** the `godot-gaussian-splatting` add-on loads the PLY. Add `collision.glb` as a `StaticBody3D` (or let the add-on generate collision itself).
- **Your character:** put `character.glb` at `align.json → seed` (it's in metres, on the floor, Y-up).

### Scale note
Scale is a guess based on `CAMERA_HEIGHT_M`. Check `align.json`: if a door that's really 2.0 m tall measures 2.3 m in Blender, set `KNOWN_SCALE = 2.0/2.3 * scale_applied` and re-run steps 3–5.

### If it looks bad
| Symptom | Fix |
|---|---|
| "Registered 40/200 images" | Blur or fast motion. Re-film slower with more overlap, or lower `TARGET_FRAMES` |
| Floating fog/blobs | Use `QUALITY="full"`, film more angles, crop in SuperSplat (free, superspl.at/editor) |
| CUDA out of memory | `MAX_IMAGE_SIZE = 1280` or `TARGET_FRAMES = 150` |
| Floor tilted / room sideways | You filmed mostly up or down. Film one loop holding the phone level |
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = HERE / "room_to_splat.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
