"""End-to-end run of vf.pipeline with fake WorldMirror / Lyra / gsplat (CPU only).

Exercises all the glue between stages: file layouts, flags, alignment, export.
Needs splat-transform on PATH; skipped otherwise.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).parent
FAKES = HERE / "fakes"
pytestmark = pytest.mark.skipif(shutil.which("splat-transform") is None, reason="splat-transform not installed")


def make_job(tmp_path, opts, n=8, size=(800, 600)):
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(job / "input" / f"{i}.jpg")
    (job / "job.json").write_text(json.dumps(opts))
    return job


def run(job, tmp_path):
    fake_splat = tmp_path / "fake_splat.ply"
    # A refined splat already in the levelled frame: reuse the fake WorldMirror gaussians, then shift.
    env = dict(os.environ,
               VF_CONDA_SH=str(FAKES / "conda.sh"), VF_HYW_DIR=str(FAKES / "hyw"),
               VF_LYRA_DIR=str(FAKES / "lyra"), VF_GSPLAT_DIR=str(FAKES / "gsplat"),
               PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}", FAKE_SPLAT=str(fake_splat))
    # build the fake refined splat from a WorldMirror-like room
    subprocess.run([sys.executable, str(FAKES / "hyw/hyworld2/worldrecon/pipeline.py"), "--input_path",
                    str(job / "input"), "--strict_output_path", str(tmp_path / "fs"), "--target_size", "518"],
                   check=True, env=env, capture_output=True)
    shutil.copy(tmp_path / "fs" / "gaussians.ply", fake_splat)
    p = subprocess.run([sys.executable, "-m", "vf.pipeline", str(job)], cwd=HERE.parent, env=env,
                       capture_output=True, text=True)
    status = json.loads((job / "status.json").read_text())
    assert p.returncode == 0, (job / "log.txt").read_text()[-3000:]
    return status


def test_reconstruct_only_without_refine(tmp_path):
    job = make_job(tmp_path, {"name": "room", "refine": {"enabled": False}})
    st = run(job, tmp_path)
    assert st["state"] == "done"
    assert st["done"] == ["prepare", "recon", "colmap", "export"]
    for f in ("output/preview.html", "output/scene.html", "output/scene.spz", "output/collision.glb", "viewfinder_room.zip"):
        assert f in st["outputs"] and (job / f).exists(), f
    info = json.loads((job / "output/align.json").read_text())
    # fake cameras sit 1.4 m above the floor of a 2.5 m room -> scale recovered, floor at 0
    assert info["camera_height_median"] == pytest.approx(1.4, abs=0.05)
    assert info["bounds_max"][1] == pytest.approx(2.5, abs=0.2)
    assert info["bounds_min"][1] == pytest.approx(0.0, abs=0.1)


def test_imagine_fuse_refine(tmp_path):
    job = make_job(tmp_path, {"name": "room", "imagine": {"enabled": True, "anchors": [0, 3], "frames": 81, "stride": 8},
                              "refine": {"enabled": True}})
    st = run(job, tmp_path)
    assert st["done"] == ["prepare", "recon", "imagine", "fuse", "colmap", "refine", "export"]
    gen = sorted((job / "gen_frames").glob("*.jpg"))
    assert len(gen) == 2 * 10  # 81 frames, every 8th after the first, per anchor
    assert Image.open(gen[0]).size == Image.open(next((job / "photos").glob("*.jpg"))).size
    assert len(list((job / "data/images").glob("*.jpg"))) == 8 + 20
    assert "output/imagined_anchor_3.mp4" in st["outputs"]
