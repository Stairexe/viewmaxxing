"""viewfinder-open: photos → (imagined views) → levelled, game-ready 3D Gaussian splat.

Run one job:   python -m vf.pipeline JOB_DIR
JOB_DIR must contain input/ (photos) and job.json (options, see DEFAULTS).

Stages (each writes status.json so the web UI can follow along):
  prepare   normalise photos to one aspect ratio
  recon     WorldMirror 2.0 feed-forward reconstruction of the photos  → instant preview
  imagine   Lyra 2.0 generates new views along a camera path from anchor photo(s)   [optional]
  fuse      WorldMirror 2.0 again on photos + generated views → one shared 3D frame
  views     map cameras back to full-resolution crops (wm_export)
  colmap    build a levelled, metric COLMAP dataset (+Y up, floor y=0)
  refine    gsplat optimisation on all views                                          [optional]
  export    SPZ / SOG / HTML viewer / collision mesh / zip
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KAGGLE_SRC = REPO / "kaggle" / "src"

# Where the setup script put things. Override with environment variables.
WORK = Path(os.environ.get("VF_WORK", "/workspace"))
LYRA_DIR = Path(os.environ.get("VF_LYRA_DIR", WORK / "lyra" / "Lyra-2"))
HYW_DIR = Path(os.environ.get("VF_HYW_DIR", WORK / "HY-World-2.0"))
GSPLAT_DIR = Path(os.environ.get("VF_GSPLAT_DIR", WORK / "gsplat"))
CONDA_SH = os.environ.get("VF_CONDA_SH", str(WORK / "miniconda3" / "etc" / "profile.d" / "conda.sh"))
ENV_LYRA = os.environ.get("VF_ENV_LYRA", "lyra2")
ENV_HYW = os.environ.get("VF_ENV_HYW", "hyworld2")
ENV_CORE = os.environ.get("VF_ENV_CORE", "vfcore")
# Official pycolmap 4 lives in its own folder: gsplat's example code imports a different
# package that is also called "pycolmap", so the two can't share site-packages.
SFM_LIBS = Path(os.environ.get("VF_SFM_LIBS", WORK / "sfm_libs"))
LYRA_LIBS = (
    'SITE=$CONDA_PREFIX/lib/python3.10/site-packages; export CUDA_HOME=$CONDA_PREFIX; '
    'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:'
    '$SITE/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
)

DEFAULTS = {
    "name": "scene",
    "max_photo_size": 1920,
    "recon_target_size": 952,      # WorldMirror inference resolution (multiple of 14)
    "fuse_target_size": 644,       # lower when fusing many views to fit in memory
    "imagine": {
        "enabled": False,
        "anchors": [0],            # which photos to "walk away from" (indices after sorting)
        "preset": "turn_around",   # see vf/trajectories.py PRESETS
        "frames": 161,             # 1 + 80k
        "distance": 0.6,           # metres
        "degrees": 60.0,
        "prompt": "",              # describe the scene; empty = generic prompt
        "fast": True,              # Lyra --use_dmd (~15x faster, slightly lower quality)
        "stride": 4,               # keep every Nth generated frame for fusion
        "pose_scale": 1.0,
    },
    "refine": {"enabled": True, "steps_scaler": 0.25},
    "camera_height": 1.4,
    "known_scale": None,
    "collider_voxel": 0.05,
}

GENERIC_PROMPT = (
    "A photorealistic, steady first-person walkthrough of the same place, consistent lighting, "
    "materials and furniture, sharp details, no people, no text overlays."
)


# ----------------------------------------------------------------------------- utilities
def merge(defaults: dict, given: dict) -> dict:
    out = dict(defaults)
    for k, v in (given or {}).items():
        out[k] = merge(defaults[k], v) if isinstance(defaults.get(k), dict) and isinstance(v, dict) else v
    return out


class Job:
    def __init__(self, root: Path):
        self.root = root
        self.opts = merge(DEFAULTS, json.loads((root / "job.json").read_text()) if (root / "job.json").exists() else {})
        self.log_path = root / "log.txt"
        self.status = {"state": "running", "stage": None, "done": [], "message": "", "outputs": [],
                       "started": time.time()}

    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    def save_status(self, **kw):
        self.status.update(kw, updated=time.time())
        tmp = self.p("status.json.tmp")
        tmp.write_text(json.dumps(self.status, indent=1))
        tmp.replace(self.p("status.json"))

    def stage(self, name: str, message: str = ""):
        if self.status["stage"] and self.status["stage"] not in self.status["done"]:
            self.status["done"].append(self.status["stage"])
        self.save_status(stage=name, message=message)
        self.log(f"\n===== {name}: {message}")

    def add_output(self, rel: str):
        if rel not in self.status["outputs"]:
            self.status["outputs"].append(rel)
            self.save_status()

    def log(self, text: str):
        with open(self.log_path, "a") as f:
            f.write(text.rstrip("\n") + "\n")
        print(text, flush=True)

    def run(self, cmd: str, env: str | None = None, cwd: Path | None = None, pre: str = ""):
        """Run a shell command (optionally inside a conda env), streaming into log.txt."""
        parts = []
        if env:
            parts += [f"source {shlex.quote(CONDA_SH)}", f"conda activate {env}"]
        if env == ENV_CORE:  # gsplat compiles/loads its CUDA kernels from here
            parts.append(f'export CUDA_HOME="${{CONDA_PREFIX:-}}" TORCH_EXTENSIONS_DIR={shlex.quote(str(WORK / "torch_extensions"))}')
        if pre:
            parts.append(pre)
        if cwd:
            parts.append(f"cd {shlex.quote(str(cwd))}")
        parts.append(cmd)
        full = " && ".join(parts)
        self.log(f"$ {cmd}")
        t = time.time()
        with open(self.log_path, "a") as f:
            proc = subprocess.run(["bash", "-c", full], stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = self.log_path.read_text().splitlines()[-25:]
            raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\n" + "\n".join(tail))
        self.log(f"  [ok in {time.time() - t:.0f}s]")


def q(path) -> str:
    return shlex.quote(str(path))


# ----------------------------------------------------------------------------- stages
def prepare(job: Job) -> list[Path]:
    """Copy photos to photos/ with one shared aspect ratio (WorldMirror requires it)."""
    from PIL import Image, ImageOps

    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
    src = sorted(p for p in job.p("input").rglob("*") if p.suffix.lower() in exts)
    if not src:
        raise RuntimeError("no photos in input/")
    imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in src]
    aspects = np.array([im.width / im.height for im in imgs])
    target_aspect = float(np.median(aspects))
    out = job.p("photos")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    paths = []
    for i, im in enumerate(imgs):
        w, h = im.size
        if w / h > target_aspect:  # too wide: crop sides
            nw = int(round(h * target_aspect)); box = ((w - nw) // 2, 0, (w - nw) // 2 + nw, h)
        else:                      # too tall: crop top/bottom
            nh = int(round(w / target_aspect)); box = (0, (h - nh) // 2, w, (h - nh) // 2 + nh)
        im = im.crop(box)
        s = min(1.0, job.opts["max_photo_size"] / max(im.size))
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
        p = out / f"a_{i:04d}.jpg"  # "a_" sorts before generated "b_" frames
        im.save(p, quality=95)
        paths.append(p)
    job.log(f"{len(paths)} photos, {paths[0].name} is {Image.open(paths[0]).size}, aspect {target_aspect:.3f}")
    return paths


def worldmirror(job: Job, images: Path, out: Path, target_size: int):
    shutil.rmtree(out, ignore_errors=True)
    job.run(
        f"python -m hyworld2.worldrecon.pipeline --input_path {q(images)} --strict_output_path {q(out)} "
        f"--target_size {target_size} --no_interactive --no_save_normal --enable_bf16",
        env=ENV_HYW, cwd=HYW_DIR,
    )
    if not (out / "camera_params.json").exists():
        raise RuntimeError(f"WorldMirror wrote no camera_params.json in {out}")


def splat_transform(job: Job, args: str):
    job.run(f"splat-transform -w {args}", env=ENV_CORE)


def recon(job: Job):
    worldmirror(job, job.p("photos"), job.p("recon"), job.opts["recon_target_size"])
    job.p("output").mkdir(exist_ok=True)
    splat_transform(job, f"{q(job.p('recon', 'gaussians.ply'))} {q(job.p('output', 'preview.html'))}")
    job.add_output("output/preview.html")


def imagine(job: Job) -> list[Path]:
    """Lyra 2.0: generate a walkthrough video from each anchor photo, keep every Nth frame."""
    import cv2
    from PIL import Image

    sys.path.insert(0, str(HERE.parent))
    from vf.trajectories import make_trajectory, save_trajectory, valid_frame_count
    from vf.wm_export import adaptive_target_size, crop_transform, list_images

    o = job.opts["imagine"]
    photos = list_images(str(job.p("photos")))
    cams = json.loads(job.p("recon", "camera_params.json").read_text())
    intr = {int(e["camera_id"]): np.array(e["matrix"]) for e in cams["intrinsics"]}
    pw, ph = Image.open(photos[0]).size
    tgt = adaptive_target_size(pw, ph, job.opts["recon_target_size"])
    t = crop_transform(pw, ph, tgt)
    frames_out = job.p("gen_frames")
    shutil.rmtree(frames_out, ignore_errors=True)
    frames_out.mkdir()
    prompt = o["prompt"].strip() or GENERIC_PROMPT
    n_frames = valid_frame_count(o["frames"])
    kept = []
    for a in o["anchors"]:
        if a >= len(photos):
            job.log(f"skipping anchor {a}: only {len(photos)} photos")
            continue
        gdir = job.p("gen", f"anchor_{a}")
        shutil.rmtree(gdir, ignore_errors=True)
        gdir.mkdir(parents=True)
        anchor_png = gdir / f"anchor_{a}.png"
        Image.open(photos[a]).save(anchor_png)
        # WorldMirror intrinsics are in inference pixels; scale them to the full photo.
        K = intr[a].copy()
        K[0, 0] /= t["sx"]; K[0, 2] = (K[0, 2] + t["crop_x"]) / t["sx"]
        K[1, 1] /= t["sy"]; K[1, 2] = (K[1, 2] + t["crop_y"]) / t["sy"]
        traj = make_trajectory(o["preset"], n_frames, K, pw, ph, o["distance"], o["degrees"])
        save_trajectory(gdir / "trajectory.npz", traj)
        job.run(
            "PYTHONPATH=. python -m lyra_2._src.inference.lyra2_custom_traj_inference "
            f"--input_image_path {q(anchor_png)} --trajectory_path {q(gdir / 'trajectory.npz')} "
            f"--experiment lyra2 --checkpoint_dir checkpoints/model --prompt {q(prompt)} "
            f"--num_frames {n_frames} --pose_scale {o['pose_scale']} --num_samples 1 "
            f"--output_path {q(gdir)}" + (" --use_dmd" if o["fast"] else ""),
            env=ENV_LYRA, cwd=LYRA_DIR,
            pre=f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && {LYRA_LIBS}",
        )
        video = gdir / f"anchor_{a}.mp4"
        if not video.exists():
            found = sorted(gdir.rglob("*.mp4"))
            if not found:
                raise RuntimeError(f"Lyra produced no video in {gdir}")
            video = found[-1]
        shutil.copy(video, job.p("output", f"imagined_anchor_{a}.mp4"))
        job.add_output(f"output/imagined_anchor_{a}.mp4")

        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i > 0 and i % o["stride"] == 0:  # frame 0 is the anchor photo itself
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Crop the generated frame to the photos' aspect ratio and size.
                w, h = im.size
                target_aspect = pw / ph
                if w / h > target_aspect:
                    nw = int(round(h * target_aspect)); im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
                else:
                    nh = int(round(w / target_aspect)); im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
                p = frames_out / f"b_a{a:02d}_{i:04d}.jpg"
                im.resize((pw, ph), Image.Resampling.LANCZOS).save(p, quality=95)
                kept.append(p)
            i += 1
        cap.release()
        job.log(f"anchor {a}: {i} generated frames, kept {len(kept)} so far")
    return kept


def fuse(job: Job, generated: list[Path]) -> Path:
    if not generated:
        return job.p("recon")
    fin = job.p("fused_in")
    shutil.rmtree(fin, ignore_errors=True)
    fin.mkdir()
    for p in list(job.p("photos").glob("*.jpg")) + generated:
        try:
            os.link(p, fin / p.name)
        except OSError:
            shutil.copy(p, fin / p.name)
    worldmirror(job, fin, job.p("fused"), job.opts["fuse_target_size"])
    return job.p("fused")


def views_and_colmap(job: Job, recon_dir: Path, image_dir: Path, target_size: int) -> dict:
    job.run(f"python {q(HERE / 'wm_export.py')} --images {q(image_dir)} --recon {q(recon_dir)} "
            f"--target-size {target_size} --out {q(job.p('views'))}", env=ENV_HYW, cwd=HYW_DIR)
    extra = f" --known-scale {job.opts['known_scale']}" if job.opts["known_scale"] else ""
    job.run(f"python {q(HERE / 'build_colmap.py')} {q(job.p('views'))} {q(job.p('data'))} "
            f"--camera-height {job.opts['camera_height']}{extra}", env=ENV_CORE,
            pre=f"export PYTHONPATH={q(SFM_LIBS)}" if SFM_LIBS.exists() else "")
    return json.loads(job.p("data", "align.json").read_text())


def refine(job: Job) -> Path:
    res = job.p("results")
    shutil.rmtree(res, ignore_errors=True)
    job.run(
        f"python simple_trainer.py default --data_dir {q(job.p('data'))} --data_factor 1 "
        f"--result_dir {q(res)} --no-normalize_world_space --save_ply --disable_viewer "
        f"--steps_scaler {job.opts['refine']['steps_scaler']} --test_every 100000",
        env=ENV_CORE, cwd=GSPLAT_DIR / "examples",
    )
    plys = sorted(res.glob("ply/point_cloud_*.ply"), key=lambda p: int(p.stem.split("_")[-1]))
    if not plys:
        raise RuntimeError("gsplat wrote no PLY")
    return plys[-1]


def export(job: Job, ply: Path, align_info: dict):
    out = job.p("output")
    out.mkdir(exist_ok=True)
    shutil.copy(ply, out / "raw.ply")
    splat_transform(job, f"{q(out / 'raw.ply')} --filter-nan --filter-value opacity,gt,0.02 {q(out / 'scene.ply')}")
    (out / "raw.ply").unlink()
    for ext in ("spz", "sog", "html"):
        splat_transform(job, f"{q(out / 'scene.ply')} {q(out / ('scene.' + ext))}")
    (out / "align.json").write_text(json.dumps(align_info, indent=2))
    job.run(f"python {q(KAGGLE_SRC / 'collider.py')} {q(out / 'scene.ply')} {q(out / 'collision.glb')} "
            f"--align {q(out / 'align.json')} --voxel {job.opts['collider_voxel']}", env=ENV_CORE)
    for f in ("scene.html", "scene.spz", "scene.sog", "scene.ply", "collision.glb", "align.json"):
        job.add_output(f"output/{f}")
    zip_base = job.p(f"viewfinder_{job.opts['name']}")
    shutil.make_archive(str(zip_base), "zip", out)
    job.add_output(zip_base.name + ".zip")


def main(job_dir: str):
    job = Job(Path(job_dir).resolve())
    try:
        job.stage("prepare", "normalising photos")
        prepare(job)
        job.stage("recon", "WorldMirror 2.0 reconstruction")
        recon(job)
        generated = []
        if job.opts["imagine"]["enabled"]:
            job.stage("imagine", f"Lyra 2.0 '{job.opts['imagine']['preset']}' from anchors {job.opts['imagine']['anchors']}")
            generated = imagine(job)
            job.stage("fuse", f"WorldMirror 2.0 on photos + {len(generated)} imagined views")
        recon_dir = fuse(job, generated)
        image_dir = job.p("fused_in") if generated else job.p("photos")
        target = job.opts["fuse_target_size"] if generated else job.opts["recon_target_size"]
        job.stage("colmap", "levelling, metric scale, COLMAP dataset")
        info = views_and_colmap(job, recon_dir, image_dir, target)
        if job.opts["refine"]["enabled"]:
            job.stage("refine", "gsplat optimisation")
            final = refine(job)
        else:
            sys.path.insert(0, str(HERE.parent))
            from vf.splat_ops import transform_splat
            final = job.p("aligned_gaussians.ply")
            s = info["sim3"]
            transform_splat(str(recon_dir / "gaussians.ply"), str(final), s["scale"], s["R"], s["t"])
        job.stage("export", "SPZ / SOG / HTML / collider")
        export(job, final, info)
        job.status["done"].append("export")
        job.save_status(state="done", stage=None, message="finished",
                        seconds=round(time.time() - job.status["started"]))
    except Exception as e:  # report everything to the UI
        job.log(traceback.format_exc())
        job.save_status(state="error", message=str(e)[-2000:])
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])
