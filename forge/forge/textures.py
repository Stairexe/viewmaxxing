"""Procedural, tileable PBR-ish textures for a building, written as GTA-ready DDS.

Each material gets diffuse (_d), normal (_n) and spec (_s) maps unless it is emissive.
This is the *baseline* look: AI-generated swatches (step ② of docs/MLO_FORGE.md) replace
any of these by dropping `<texture_name>.png` into an override folder.

Spec map convention for GTA normal_spec: intensity in RGB (brighter = shinier).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .dds import write_dds

SIZE = 512


# ---------------------------------------------------------------------- noise helpers
def tile_noise(size: int, scale: float, seed: int, octaves: int = 4) -> np.ndarray:
    """Seamlessly tiling fBm noise in 0..1 (band-limited white noise via FFT, so it wraps)."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2) + 1e-6
    out = np.zeros((size, size))
    amp = 1.0
    for o in range(octaves):
        cutoff = scale * (2 ** o) / size
        spec = np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-(f / cutoff) ** 2)
        layer = np.real(np.fft.ifft2(spec))
        layer = (layer - layer.mean()) / (layer.std() + 1e-9)
        out += amp * layer
        amp *= 0.5
    out = (out - out.min()) / (np.ptp(out) + 1e-9)
    return out


def hex_or_rgb(c) -> np.ndarray:
    return np.asarray(c, dtype=np.float64)


def colourise(base_rgb, n: np.ndarray, var: float) -> np.ndarray:
    base = hex_or_rgb(base_rgb)[None, None, :]
    return np.clip(base * (1 + var * (n[..., None] - 0.5) * 2), 0, 255)


def normal_from_height(h: np.ndarray, strength: float) -> np.ndarray:
    """Tangent-space normal map from a height field (wraps, so it tiles)."""
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * strength
    n = np.stack([-dx, dy, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return (n * 0.5 + 0.5) * 255


def cracks(size: int, count: int, seed: int) -> np.ndarray:
    """Thin random-walk crack mask 0..1 (wraps around edges)."""
    rng = np.random.default_rng(seed)
    m = np.zeros((size, size))
    for _ in range(count):
        y, x = rng.integers(0, size, 2)
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(40, 160))):
            ang += rng.normal(0, 0.35)
            y = (y + np.sin(ang) * 1.5) % size
            x = (x + np.cos(ang) * 1.5) % size
            m[int(y), int(x)] = 1.0
    img = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.6))
    return np.asarray(img, np.float64) / 255


def fit_font(draw, text: str, max_w: float, start: int):
    size = start
    f = font(size)
    while draw.textlength(text, font=f) > max_w and size > 8:
        size -= 2
        f = font(size)
    return f


def font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def neon_text(text: str, w: int, h: int, colour, bg=(8, 8, 12), box=None) -> Image.Image:
    """Glowing neon lettering for emissive signs."""
    img = Image.new("RGB", (w, h), bg)
    if box:
        ImageDraw.Draw(img).rectangle([4, 4, w - 5, h - 5], fill=box)
    layer = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    size = int(h * 0.62)
    f = font(size)
    while d.textlength(text, font=f) > w * 0.92 and size > 8:
        size -= 2
        f = font(size)
    tw = d.textlength(text, font=f)
    if box:  # printed lightbox: plain coloured letters with a soft halo, no neon core
        ImageDraw.Draw(img).text(((w - tw) / 2, (h - size) / 2 - size * 0.08), text, font=f, fill=tuple(colour))
        return img
    d.text(((w - tw) / 2, (h - size) / 2 - size * 0.08), text, font=f, fill=tuple(colour))
    glow = layer.filter(ImageFilter.GaussianBlur(h * 0.06))
    core = Image.eval(layer, lambda v: min(255, int(v * 0.6 + (255 if v > 0 else 0) * 0.4)))
    out = np.asarray(img, np.float64) + np.asarray(glow, np.float64) * 1.6 + np.asarray(core, np.float64)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------- materials
def _rgb(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
def _grey(a): return _rgb(np.repeat(np.clip(a, 0, 255)[..., None], 3, -1))


def make_material_maps(name: str, palette: dict, seed: int, wear: float) -> dict[str, Image.Image]:
    S = SIZE
    n1, n2 = tile_noise(S, 6, seed), tile_noise(S, 40, seed + 1)
    flat_n = _rgb(np.full((S, S, 3), (128, 128, 255), np.float64))
    yy, xx = np.mgrid[0:S, 0:S] / S

    if name == "stucco":
        cr = cracks(S, 10, seed + 2) * wear
        bleach = tile_noise(S, 3, seed + 3)
        d = colourise(palette["walls"], n1 * 0.6 + n2 * 0.4, 0.10)
        d = d * (1 - 0.18 * wear * bleach[..., None]) + 255 * 0.18 * wear * bleach[..., None]
        d *= (1 - 0.45 * cr[..., None])
        h = n2 * 0.8 + n1 * 0.2 - cr * 0.6
        return {"d": _rgb(d), "n": _rgb(normal_from_height(h, 3.0)), "s": _grey(30 + 25 * n2)}
    if name in ("trim", "plinth", "concrete", "int_wall"):
        base = {"trim": palette["trim"], "plinth": (150, 146, 140), "concrete": (138, 136, 132),
                "int_wall": (226, 221, 208)}[name]
        var = {"trim": 0.05, "plinth": 0.18, "concrete": 0.22, "int_wall": 0.04}[name]
        d = colourise(base, n1 * 0.5 + n2 * 0.5, var)
        if name in ("plinth", "concrete"):
            d *= 1 - 0.3 * cracks(S, 6, seed + 4)[..., None] * wear
        return {"d": _rgb(d), "n": _rgb(normal_from_height(n2, 2.0 if name != "int_wall" else 0.6)),
                "s": _grey(20 + 30 * n2)}
    if name == "roof":
        stains = (tile_noise(S, 4, seed + 5) > 0.62).astype(np.float64)
        d = colourise((120, 122, 124), n2, 0.15) * (1 - 0.25 * stains[..., None])
        return {"d": _rgb(d), "n": _rgb(normal_from_height(n2, 1.5)), "s": _grey(15 + 60 * stains)}
    if name == "metal":
        brushed = tile_noise(S, 60, seed + 6)
        brushed = np.repeat(brushed.mean(1, keepdims=True), S, 1) * 0.7 + brushed * 0.3
        d = colourise((180, 184, 188), brushed, 0.08)
        return {"d": _rgb(d), "n": flat_n, "s": _grey(140 + 60 * brushed)}
    if name == "shutter":
        slats = (np.sin(yy * 2 * np.pi * 12) * 0.5 + 0.5)
        d = colourise(palette["shutters"], n2 * 0.4 + slats * 0.6, 0.12) * (1 - 0.15 * wear * n1[..., None])
        return {"d": _rgb(d), "n": _rgb(normal_from_height(slats, 6.0)), "s": _grey(40 + 40 * slats)}
    if name == "awning":
        stripe = ((xx * 8) % 2 < 1).astype(np.float64)
        pink, white = hex_or_rgb(palette["neon"][0] if isinstance(palette["neon"][0], (list, tuple)) else (230, 120, 150)), hex_or_rgb((240, 236, 226))
        d = stripe[..., None] * pink + (1 - stripe[..., None]) * white
        d = d * (1 - 0.12 * n2[..., None]) * (1 - 0.2 * wear * tile_noise(S, 3, seed + 7)[..., None])
        return {"d": _rgb(d), "n": _rgb(normal_from_height(n2, 1.0)), "s": _grey(20 + 10 * n2)}
    if name == "glass":
        d = np.zeros((S, S, 4))
        d[..., :3] = colourise((150, 190, 200), n1, 0.05)
        d[..., 3] = 70 + 25 * n1
        return {"d": Image.fromarray(np.clip(d, 0, 255).astype(np.uint8), "RGBA"), "n": flat_n, "s": _grey(np.full((S, S), 230.0))}
    if name == "glass_dark":
        refl = np.clip(yy * 0.6 + n1 * 0.4, 0, 1)
        d = colourise((40, 62, 70), refl, 0.4)
        return {"d": _rgb(d), "n": flat_n, "s": _grey(np.full((S, S), 220.0))}
    if name == "door_steel":
        panel = ((np.abs(xx - 0.5) < 0.42) & (np.abs(yy - 0.5) < 0.45)).astype(np.float64)
        rust = (tile_noise(S, 8, seed + 8) > 0.7) * wear
        d = colourise((92, 98, 102), n2, 0.1) * (1 - 0.1 * panel[..., None])
        d = d * (1 - rust[..., None]) + rust[..., None] * np.array([120, 70, 40])
        return {"d": _rgb(d), "n": _rgb(normal_from_height(panel, 4.0)), "s": _grey(70 + 50 * (1 - rust))}
    if name == "terrazzo":
        rng = np.random.default_rng(seed + 9)
        im = Image.fromarray(colourise((232, 226, 214), n2, 0.03).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        chips = [(226, 140, 160), (110, 110, 112), (60, 60, 62), (190, 196, 200)]
        for _ in range(2600):
            x, y = rng.integers(0, S, 2)
            r = int(rng.integers(1, 5))
            c = chips[int(rng.integers(0, len(chips)))]
            for ox in (-S, 0, S):            # draw wrapped so it tiles
                for oy in (-S, 0, S):
                    dr.ellipse([x - r + ox, y - r + oy, x + r + ox, y + r + oy], fill=c)
        return {"d": im.convert("RGB"), "n": _rgb(normal_from_height(n2, 0.4)), "s": _grey(np.full((S, S), 150.0))}
    if name == "ceiling":
        grid = ((xx % 1 < 0.015) | (yy % 1 < 0.015)).astype(np.float64)
        d = colourise((236, 234, 228), n2, 0.03) * (1 - 0.35 * grid[..., None])
        return {"d": _rgb(d), "n": _rgb(normal_from_height(-grid, 4.0)), "s": _grey(20 + 0 * n2)}
    if name == "wood":
        grain = np.sin((xx * 30 + n1 * 6) * np.pi) * 0.5 + 0.5
        d = colourise((128, 84, 52), grain * 0.6 + n2 * 0.4, 0.25)
        return {"d": _rgb(d), "n": _rgb(normal_from_height(grain, 1.0)), "s": _grey(50 + 40 * grain)}
    if name == "ac_unit":
        grille = (np.sin(yy * 2 * np.pi * 24) > 0.3).astype(np.float64)
        rust = (tile_noise(S, 6, seed + 10) > 0.72) * wear
        d = colourise((206, 200, 184), n2, 0.06) * (1 - 0.35 * grille[..., None])
        d = d * (1 - 0.6 * rust[..., None]) + 0.6 * rust[..., None] * np.array([130, 80, 50])
        return {"d": _rgb(d), "n": _rgb(normal_from_height(grille, 3.0)), "s": _grey(60 + 40 * grille)}
    raise KeyError(name)


def make_sign_maps(name: str, spec: dict) -> dict[str, Image.Image]:
    signs = {s["where"]: s["text"] for s in spec.get("signage", [])}
    neon = spec["style"]["palette"].get("neon", ["hot pink", "cyan"])
    pink, cyan = (255, 70, 170), (60, 230, 255)
    if name == "sign_main":
        return {"d": neon_text(signs.get("front parapet", "STORE"), 1024, 128, pink)}
    if name == "sign_window":
        return {"d": neon_text(signs.get("storefront glass", "OPEN"), 512, 128, cyan)}
    if name == "sign_open":
        return {"d": neon_text(signs.get("corner", "OPEN 24 HRS"), 512, 128, (255, 60, 60), bg=(30, 30, 30), box=(245, 240, 230))}
    if name == "fridge_glow":
        rng = np.random.default_rng(3)
        im = Image.new("RGB", (512, 256), (230, 240, 250))
        d = ImageDraw.Draw(im)
        for shelf in range(4):
            y0 = 20 + shelf * 58
            d.rectangle([0, y0 + 46, 512, y0 + 50], fill=(150, 160, 170))
            x = 6
            while x < 500:
                w = int(rng.integers(10, 18))
                c = tuple(int(v) for v in rng.integers(30, 230, 3))
                d.rectangle([x, y0 + 8, x + w, y0 + 46], fill=c)
                x += w + 4
        for xdiv in (170, 340):
            d.rectangle([xdiv - 3, 0, xdiv + 3, 256], fill=(90, 95, 100))
        return {"d": im}
    raise KeyError(name)


def make_decal_maps(name: str, spec: dict, seed: int) -> dict[str, Image.Image]:
    flat_n = _rgb(np.full((256, 256, 3), (128, 128, 255), np.float64))
    if name == "poster":
        W, H = 1024, 512
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        posters = [("NEON LAGOON", "LIVE FRI", (40, 20, 70), (255, 90, 200)),
                   ("LOTTO", "JACKPOT $$$", (250, 220, 60), (200, 30, 30)),
                   ("PALM KNGZ", "", (0, 0, 0), (60, 230, 120))]
        for k, (t1, t2, bg, fg) in enumerate(posters):
            x0 = k * W // 3 + 10
            x1 = (k + 1) * W // 3 - 10
            if t1 == "PALM KNGZ":   # graffiti tag: no paper, just paint
                f = fit_font(d, t1, (x1 - x0) - 30, 110)
                d.text((x0 + 8, 170), t1, font=f, fill=fg + (235,), stroke_width=5, stroke_fill=(20, 20, 20, 235))
                continue
            d.rectangle([x0, 30, x1, H - 30], fill=bg + (240,))
            d.text((x0 + 20, 70), t1, font=fit_font(d, t1, (x1 - x0) - 40, 56), fill=fg + (255,))
            d.text((x0 + 20, 170), t2, font=fit_font(d, t2, (x1 - x0) - 40, 40), fill=(255, 255, 255, 255))
            # torn corner
            d.polygon([(x1, H - 30), (x1 - 70, H - 30), (x1, H - 120)], fill=(0, 0, 0, 0))
        arr = np.asarray(im, np.float64)
        age = tile_noise(512, 8, seed)[:, :, None]
        age = np.asarray(Image.fromarray((age[..., 0] * 255).astype(np.uint8)).resize((W, H)), np.float64)[..., None] / 255
        arr[..., :3] *= 0.75 + 0.25 * age
        return {"d": Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA"),
                "n": flat_n, "s": _grey(np.full((256, 256), 20.0))}
    if name == "grime":
        W, H = 512, 256
        n = np.asarray(Image.fromarray((tile_noise(512, 10, seed) * 255).astype(np.uint8)).resize((W, H)), np.float64) / 255
        streak = np.asarray(Image.fromarray((tile_noise(512, 30, seed + 1) * 255).astype(np.uint8)).resize((W, H)), np.float64) / 255
        streak = np.repeat(streak.mean(0, keepdims=True), H, 0)
        yy = np.linspace(0, 1, H)[:, None]      # image row 0 = top of the decal (UV v=1): fade out upwards
        a = np.clip((yy ** 1.6) * (0.55 + 0.45 * n) + 0.25 * streak * yy, 0, 1)
        rgb = np.stack([40 + 20 * n, 38 + 18 * n, 30 + 15 * n], -1)
        arr = np.concatenate([rgb, (a * 200)[..., None]], -1)
        return {"d": Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA"),
                "n": flat_n, "s": _grey(np.full((256, 256), 10.0))}
    raise KeyError(name)


PALETTE_RGB = {
    "faded flamingo pink": (226, 156, 160), "off-white": (238, 233, 220), "mint green": (170, 224, 196),
    "sea-foam teal": (110, 190, 182),
}


def resolve_palette(spec: dict) -> dict:
    p = spec["style"]["palette"]
    def rgb(v, default):
        if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(c, (int, float)) for c in v):
            return tuple(v)
        return PALETTE_RGB.get(str(v).lower(), default)
    return {"walls": rgb(p.get("walls"), (226, 156, 160)), "trim": rgb(p.get("trim"), (238, 233, 220)),
            "accent": rgb(p.get("accent"), (170, 224, 196)), "shutters": rgb(p.get("shutters"), (110, 190, 182)),
            "neon": [(255, 70, 170), (60, 230, 255)]}


def generate(spec: dict, materials: dict, out_dir: Path, overrides: Path | None = None) -> dict:
    """Write every texture the materials reference. Returns a manifest {texture_name: info}."""
    out_dir = Path(out_dir)
    (out_dir / "dds").mkdir(parents=True, exist_ok=True)
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    palette = resolve_palette(spec)
    wear = float(spec["style"].get("wear_level", 0.5))
    manifest = {}
    for i, (mname, mat) in enumerate(sorted(materials.items())):
        if mat.emissive:
            maps = make_sign_maps(mname, spec)
        elif mname in ("poster", "grime"):
            maps = make_decal_maps(mname, spec, 100 + i)
        else:
            maps = make_material_maps(mname, palette, 10 * i + 1, wear)
        for key, tex_name in (("d", mat.diffuse), ("n", mat.normal), ("s", mat.spec)):
            if not tex_name or key not in maps:
                continue
            img = maps[key]
            if overrides is not None and (overrides / f"{tex_name}.png").exists():
                img = Image.open(overrides / f"{tex_name}.png")
                w, h = img.size
                img = img.resize((1 << (max(w, 4).bit_length() - 1), 1 << (max(h, 4).bit_length() - 1)),
                                 Image.Resampling.LANCZOS)
            fmt = "DXT5" if (key == "d" and mat.alpha) else "DXT1"
            img = img.convert("RGBA" if fmt == "DXT5" else "RGB")
            info = write_dds(img, out_dir / "dds" / f"{tex_name}.dds", fmt)
            img.save(out_dir / "png" / f"{tex_name}.png")
            info.update(material=mname, map=key, path=f"dds/{tex_name}.dds",
                        colorspace="sRGB" if key == "d" else "Non-Color")
            manifest[tex_name] = info
    (out_dir / "textures.json").write_text(json.dumps(manifest, indent=1))
    return manifest
