"""DXT1 (BC1) / DXT5 (BC3) DDS writer with a full mip chain, pure numpy.

Why our own encoder: GTA V only takes DDS (Sollumz embeds/packs the DDS bytes as-is), it
needs power-of-two sizes and a mip chain down to 4x4, and Pillow's DDS writer emits a single
level. Rules from muto-atlas trunk/gta-fundamentals.md §5 (measured on vanilla files):
  - DXT1 for opaque maps (diffuse/normal/spec), DXT5 when alpha is needed
  - mip count = log2(short edge) - 1 (ends at 4x4)
  - in DXT1, colour0 must be > colour1 (else 3-colour mode turns index 3 transparent black)
"""

from __future__ import annotations

import struct

import numpy as np
from PIL import Image

DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PIXELFORMAT = 0x1, 0x2, 0x4, 0x1000
DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE = 0x20000, 0x80000
DDPF_FOURCC = 0x4
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000


def is_pow2(n: int) -> bool:
    return n >= 4 and (n & (n - 1)) == 0


def _to_565(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.rint(c), 0, 255).astype(np.uint32)
    return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)


def _from_565(v: np.ndarray) -> np.ndarray:
    r = (v >> 11) & 31
    g = (v >> 5) & 63
    b = v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.float64)


def _blocks(img: np.ndarray) -> np.ndarray:
    """(H, W, C) → (H/4 * W/4, 16, C) in row-major block order, row-major pixels."""
    h, w, c = img.shape
    return img.reshape(h // 4, 4, w // 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(-1, 16, c)


def encode_bc1_blocks(rgb: np.ndarray, force_four_colour: bool = True) -> np.ndarray:
    """rgb: (N, 16, 3) float 0-255 → (N, 8) uint8 BC1 blocks."""
    n = rgb.shape[0]
    mean = rgb.mean(1, keepdims=True)
    centred = rgb - mean
    # principal axis by a few power iterations, seeded with the bbox diagonal
    axis = rgb.max(1) - rgb.min(1) + 1e-3
    cov = np.einsum("npi,npj->nij", centred, centred)
    for _ in range(6):
        axis = np.einsum("nij,nj->ni", cov, axis)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
    proj = np.einsum("npi,ni->np", centred, axis)
    e0 = mean[:, 0] + axis * proj.max(1, keepdims=True)
    e1 = mean[:, 0] + axis * proj.min(1, keepdims=True)
    c0, c1 = _to_565(e0), _to_565(e1)
    if force_four_colour:  # c0 must be strictly greater for 4-colour mode
        swap = c0 < c1
        c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    p0, p1 = _from_565(c0), _from_565(c1)
    palette = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], 1)          # (N, 4, 3)
    d = ((rgb[:, :, None, :] - palette[:, None, :, :]) ** 2).sum(-1)              # (N, 16, 4)
    idx = d.argmin(-1).astype(np.uint32)
    idx[c0 == c1] = 0                                                              # flat block
    bits = (idx << (2 * np.arange(16, dtype=np.uint32))).sum(1).astype(np.uint32)
    out = np.zeros((n, 8), np.uint8)
    out[:, 0:2] = c0.astype("<u2").view(np.uint8).reshape(n, 2)
    out[:, 2:4] = c1.astype("<u2").view(np.uint8).reshape(n, 2)
    out[:, 4:8] = bits.astype("<u4").view(np.uint8).reshape(n, 4)
    return out


def encode_bc3_alpha_blocks(a: np.ndarray) -> np.ndarray:
    """a: (N, 16) float 0-255 → (N, 8) uint8 BC3 alpha blocks (8-value mode, a0 > a1)."""
    n = a.shape[0]
    a0 = np.clip(np.rint(a.max(1)), 0, 255).astype(np.int64)
    a1 = np.clip(np.rint(a.min(1)), 0, 255).astype(np.int64)
    pal = np.stack([a0, a1] + [((7 - k) * a0 + k * a1) / 7 for k in range(1, 7)], 1).astype(np.float64)  # (N, 8)
    idx = np.abs(a[:, :, None] - pal[:, None, :]).argmin(-1).astype(np.uint64)
    idx[a0 == a1] = 0
    bits = (idx << (3 * np.arange(16, dtype=np.uint64))).sum(1).astype(np.uint64)
    out = np.zeros((n, 8), np.uint8)
    out[:, 0] = a0
    out[:, 1] = a1
    for k in range(6):
        out[:, 2 + k] = ((bits >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8)
    return out


def encode_level(img: Image.Image, fmt: str) -> bytes:
    arr = np.asarray(img.convert("RGBA"), dtype=np.float64)
    blocks = _blocks(arr)
    colour = encode_bc1_blocks(blocks[:, :, :3])
    if fmt == "DXT1":
        return colour.tobytes()
    alpha = encode_bc3_alpha_blocks(blocks[:, :, 3])
    return np.concatenate([alpha, colour], 1).tobytes()


def mip_chain(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    levels = [img]
    while min(w, h) > 4:
        w, h = max(4, w // 2), max(4, h // 2)
        levels.append(img.resize((w, h), Image.Resampling.LANCZOS))
    return levels


def header(width: int, height: int, fourcc: str, mips: int, top_size: int) -> bytes:
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT | DDSD_LINEARSIZE
    caps = DDSCAPS_TEXTURE | DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    h = struct.pack("<4sI", b"DDS ", 124)
    h += struct.pack("<IIIIII", flags, height, width, top_size, 0, mips)
    h += b"\0" * 44
    h += struct.pack("<II4sIIIII", 32, DDPF_FOURCC, fourcc.encode(), 0, 0, 0, 0, 0)
    h += struct.pack("<IIIII", caps, 0, 0, 0, 0)
    assert len(h) == 128
    return h


def write_dds(img: Image.Image, path, fmt: str = "DXT1") -> dict:
    w, h = img.size
    if not (is_pow2(w) and is_pow2(h)):
        raise ValueError(f"{path}: {w}x{h} is not power-of-two (GTA textures flicker otherwise)")
    if fmt not in ("DXT1", "DXT5"):
        raise ValueError(fmt)
    levels = mip_chain(img)
    data = [encode_level(lv, fmt) for lv in levels]
    with open(path, "wb") as f:
        f.write(header(w, h, fmt, len(levels), len(data[0])))
        for d in data:
            f.write(d)
    return {"width": w, "height": h, "format": fmt, "mips": len(levels), "bytes": 128 + sum(map(len, data))}


def read_header(path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(128)
    magic, size, flags, height, width, linear, depth, mips = struct.unpack("<4sIIIIIII", raw[:32])
    fourcc = raw[84:88].decode()
    return {"magic": magic, "width": width, "height": height, "mips": mips, "format": fourcc}
