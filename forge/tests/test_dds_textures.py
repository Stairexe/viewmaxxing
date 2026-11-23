import json
from pathlib import Path

import numpy as np
from PIL import Image

from forge.building import build
from forge.dds import read_header, write_dds
from forge.textures import generate

SPEC = json.loads((Path(__file__).parents[1] / "specs" / "building_001.json").read_text())


def _psnr(a, b):
    mse = ((a.astype(float) - b.astype(float)) ** 2).mean()
    return 10 * np.log10(255 ** 2 / max(mse, 1e-9))


def test_dxt1_roundtrip_and_mips(tmp_path):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:256, 0:128] / 256
    img = np.stack([200 * xx + 30, 150 * yy + 40, 90 + 0 * xx], -1).astype(np.uint8)
    p = tmp_path / "a.dds"
    info = write_dds(Image.fromarray(img), p, "DXT1")
    h = read_header(p)
    assert (h["width"], h["height"], h["format"]) == (128, 256, "DXT1")
    assert h["mips"] == info["mips"] == 6          # 128 → 4 on the short edge: log2(128) - 1
    back = np.asarray(Image.open(p).convert("RGB"))
    assert _psnr(back, img) > 35


def test_dxt1_flat_grey_stays_opaque(tmp_path):
    p = tmp_path / "g.dds"
    write_dds(Image.new("RGB", (16, 16), (128, 128, 128)), p, "DXT1")
    back = np.asarray(Image.open(p).convert("RGBA"))
    assert back[..., 3].min() == 255 and abs(int(back[..., 0].mean()) - 128) <= 4


def test_dxt5_keeps_alpha(tmp_path):
    a = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[None, :], (64, 1))
    img = np.dstack([np.full((64, 64, 3), 200, np.uint8), a])
    p = tmp_path / "t.dds"
    write_dds(Image.fromarray(img, "RGBA"), p, "DXT5")
    back = np.asarray(Image.open(p).convert("RGBA"))
    assert _psnr(back[..., 3], img[..., 3]) > 40


def test_rejects_non_power_of_two(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        write_dds(Image.new("RGB", (100, 64)), tmp_path / "x.dds")


def test_every_material_texture_is_generated(tmp_path):
    m = build(SPEC)
    man = generate(SPEC, m.materials, tmp_path)
    wanted = {t for mat in m.materials.values() for t in (mat.diffuse, mat.normal, mat.spec) if t}
    assert wanted == set(man)
    for name, info in man.items():
        mat = m.materials[info["material"]]
        assert info["format"] == ("DXT5" if (mat.alpha and info["map"] == "d") else "DXT1"), name
        assert (tmp_path / info["path"]).exists()
    assert sum(i["bytes"] for i in man.values()) < 16e6
