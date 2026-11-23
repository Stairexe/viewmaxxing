"""Engine-neutral mesh building blocks (pure Python + numpy, no Blender).

Conventions (match GTA V): metres, Z-up, right-handed. Faces are quads or triangles
with counter-clockwise winding seen from outside (normal = right-hand rule).

Every face carries a material name, per-corner UVs and a per-corner colour (RGBA 0-1).
UVs are world-scale box projections, so a material tiles at a fixed texel density
(`tile_m` metres per texture repeat) and adjacent walls line up without unwrapping.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

Vec = np.ndarray


def v3(x, y, z) -> Vec:
    return np.array([x, y, z], dtype=np.float64)


def normalize(v: Vec) -> Vec:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class Material:
    """How a surface looks. `shader` is a GTA .sps preset; textures are logical names
    resolved by textures.py (diffuse/normal/spec)."""

    name: str
    shader: str = "normal_spec.sps"
    diffuse: str | None = None
    normal: str | None = None
    spec: str | None = None
    tile_m: float = 1.0          # metres covered by one texture repeat
    alpha: bool = False          # needs DXT5 + alpha-capable shader
    emissive: bool = False


@dataclass
class Mesh:
    name: str
    verts: list = field(default_factory=list)       # [x, y, z]
    faces: list = field(default_factory=list)       # [i0, i1, i2(, i3)]
    face_mats: list = field(default_factory=list)   # material name per face
    face_uvs: list = field(default_factory=list)    # per face: [[u, v], ...] one per corner
    face_cols: list = field(default_factory=list)   # per face: [[r, g, b, a], ...]

    # ------------------------------------------------------------------ primitives
    def add_face(self, pts, mat: Material, color=(1, 1, 1, 1), uvs=None):
        pts = [np.asarray(p, dtype=np.float64) for p in pts]
        if uvs is None:
            uvs = box_uvs(pts, mat.tile_m)
        base = len(self.verts)
        self.verts.extend(p.tolist() for p in pts)
        self.faces.append(list(range(base, base + len(pts))))
        self.face_mats.append(mat.name)
        self.face_uvs.append([list(map(float, uv)) for uv in uvs])
        self.face_cols.append([list(color)] * len(pts))

    def add_quad(self, a, b, c, d, mat, color=(1, 1, 1, 1), uvs=None):
        self.add_face([a, b, c, d], mat, color, uvs)

    def add_quad_facing(self, quad, facing, mat, color=(1, 1, 1, 1)):
        """Add a planar quad, reversing its winding if needed so its normal points along `facing`."""
        q = [np.asarray(p, float) for p in quad]
        n = np.cross(q[1] - q[0], q[2] - q[1])
        if np.dot(n, np.asarray(facing, float)) < 0:
            q = q[::-1]
        self.add_quad(*q, mat, color)

    def add_box(self, lo, hi, mat, color=(1, 1, 1, 1), skip=(), mats=None):
        """Axis-aligned box. skip: subset of {'-x','+x','-y','+y','-z','+z'} faces to omit.
        mats: optional per-face material overrides."""
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        P = {
            "000": (x0, y0, z0), "100": (x1, y0, z0), "110": (x1, y1, z0), "010": (x0, y1, z0),
            "001": (x0, y0, z1), "101": (x1, y0, z1), "111": (x1, y1, z1), "011": (x0, y1, z1),
        }
        faces = {
            "-z": ("000", "010", "110", "100"),
            "+z": ("001", "101", "111", "011"),
            "-y": ("000", "100", "101", "001"),
            "+y": ("110", "010", "011", "111"),
            "-x": ("010", "000", "001", "011"),
            "+x": ("100", "110", "111", "101"),
        }
        for key, corners in faces.items():
            if key in skip:
                continue
            m = (mats or {}).get(key, mat)
            self.add_quad(*[P[c] for c in corners], m, color)

    def add_oriented_box(self, origin, u, v, w, size, mat, color=(1, 1, 1, 1), skip=()):
        """Box spanned from `origin` along unit axes u, v, w with extents size=(su, sv, sw)."""
        u, v, w = (normalize(np.asarray(a, float)) for a in (u, v, w))
        su, sv, sw = size
        o = np.asarray(origin, float)
        p = lambda a, b, c: o + u * a * su + v * b * sv + w * c * sw  # noqa: E731
        faces = {
            "-w": (p(0, 0, 0), p(0, 1, 0), p(1, 1, 0), p(1, 0, 0)),
            "+w": (p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1)),
            "-v": (p(0, 0, 0), p(1, 0, 0), p(1, 0, 1), p(0, 0, 1)),
            "+v": (p(1, 1, 0), p(0, 1, 0), p(0, 1, 1), p(1, 1, 1)),
            "-u": (p(0, 1, 0), p(0, 0, 0), p(0, 0, 1), p(0, 1, 1)),
            "+u": (p(1, 0, 0), p(1, 1, 0), p(1, 1, 1), p(1, 0, 1)),
        }
        # Keep outward winding even if (u, v, w) is left-handed.
        flip = np.dot(np.cross(u, v), w) < 0
        for key, quad in faces.items():
            if key in skip:
                continue
            self.add_quad(*(quad[::-1] if flip else quad), mat, color)

    def add_cylinder(self, base, radius, height, mat, segments=12, color=(1, 1, 1, 1), caps=True):
        bx, by, bz = base
        ring = [(bx + radius * math.cos(2 * math.pi * i / segments),
                 by + radius * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
        circ = 2 * math.pi * radius
        for i in range(segments):
            (x0, y0), (x1, y1) = ring[i], ring[(i + 1) % segments]
            u0, u1 = circ * i / segments / mat.tile_m, circ * (i + 1) / segments / mat.tile_m
            h = height / mat.tile_m
            self.add_quad((x0, y0, bz), (x1, y1, bz), (x1, y1, bz + height), (x0, y0, bz + height), mat, color,
                          uvs=[(u0, 0), (u1, 0), (u1, h), (u0, h)])
        if caps:
            self.add_face([(x, y, bz + height) for x, y in ring], mat, color)
            self.add_face([(x, y, bz) for x, y in ring[::-1]], mat, color)

    def extend(self, other: "Mesh"):
        base = len(self.verts)
        self.verts.extend(other.verts)
        self.faces.extend([[i + base for i in f] for f in other.faces])
        self.face_mats.extend(other.face_mats)
        self.face_uvs.extend(other.face_uvs)
        self.face_cols.extend(other.face_cols)

    # ------------------------------------------------------------------ queries
    @property
    def triangle_count(self) -> int:
        return sum(len(f) - 2 for f in self.faces)

    def bounds(self):
        a = np.asarray(self.verts)
        return a.min(0), a.max(0)

    def face_normal(self, i: int) -> Vec:
        p = np.asarray([self.verts[j] for j in self.faces[i]])
        n = np.zeros(3)
        for k in range(len(p)):  # Newell's method (works for any planar polygon)
            a, b = p[k], p[(k + 1) % len(p)]
            n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                           (a[0] - b[0]) * (a[1] + b[1])])
        return normalize(n)

    def face_center(self, i: int) -> Vec:
        return np.mean([self.verts[j] for j in self.faces[i]], axis=0)

    def face_area(self, i: int) -> float:
        p = np.asarray([self.verts[j] for j in self.faces[i]])
        n = np.zeros(3)
        for k in range(1, len(p) - 1):
            n += np.cross(p[k] - p[0], p[k + 1] - p[0])
        return 0.5 * float(np.linalg.norm(n))

    def materials(self) -> list[str]:
        return sorted(set(self.face_mats))

    def to_dict(self) -> dict:
        return {"name": self.name, "verts": self.verts, "faces": self.faces, "face_mats": self.face_mats,
                "face_uvs": self.face_uvs, "face_cols": self.face_cols}


def box_uvs(pts, tile_m: float):
    """World-space box projection: project onto the plane most aligned with the face."""
    pts = [np.asarray(p, float) for p in pts]
    n = np.zeros(3)
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        n += np.array([(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])])
    ax = int(np.argmax(np.abs(n)))
    if ax == 2:      # floor / ceiling: (x, y)
        uv = [(p[0], p[1] * (1 if n[2] >= 0 else -1)) for p in pts]
    elif ax == 0:    # wall facing ±x: (y, z)
        uv = [(p[1] * (1 if n[0] >= 0 else -1), p[2]) for p in pts]
    else:            # wall facing ±y: (x, z)
        uv = [(p[0] * (-1 if n[1] >= 0 else 1), p[2]) for p in pts]
    return [(u / tile_m, v / tile_m) for u, v in uv]


# ---------------------------------------------------------------------- walls
@dataclass
class Opening:
    """Rectangular hole in a wall, in wall-local coordinates (u along the wall, z up)."""

    u0: float
    u1: float
    z0: float
    z1: float
    kind: str = "window"     # window | door | storefront | vent
    name: str = ""


def wall_with_openings(mesh: Mesh, origin, direction, length: float, z0: float, z1: float,
                       thickness: float, openings: list[Opening], outer: Material, reveal: Material,
                       inner: Material | None = None, color=(1, 1, 1, 1), inner_color=None,
                       outward=None):
    """A straight wall segment from `origin` along `direction` (horizontal), between heights
    z0..z1, with rectangular openings cut through it.

    The outer face is on the `outward` side (default: right of direction, i.e. direction x +Z),
    the inner face is `thickness` inside. Openings get reveal faces (jambs/sill/head).
    If `inner` is None no inner face is produced (exterior-only shell).
    """
    o = np.asarray(origin, float)
    d = normalize(np.asarray([direction[0], direction[1], 0.0]))
    out = normalize(np.asarray(outward, float)) if outward is not None else np.array([d[1], -d[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    ops = sorted([op for op in openings if op.u1 > op.u0 and op.z1 > op.z0], key=lambda op: op.u0)

    us = sorted({0.0, length, *[c for op in ops for c in (op.u0, op.u1)]})
    zs = sorted({z0, z1, *[c for op in ops for c in (op.z0, op.z1)]})
    us = [u for u in us if 0.0 <= u <= length]
    zs = [z for z in zs if z0 <= z <= z1]

    def inside_opening(uc, zc):
        return any(op.u0 < uc < op.u1 and op.z0 < zc < op.z1 for op in ops)

    def P(u, z, depth=0.0):
        return o + d * u + up * z - out * depth

    for i in range(len(us) - 1):
        for j in range(len(zs) - 1):
            ua, ub, za, zb = us[i], us[i + 1], zs[j], zs[j + 1]
            if inside_opening((ua + ub) / 2, (za + zb) / 2):
                continue
            mesh.add_quad_facing([P(ua, za), P(ub, za), P(ub, zb), P(ua, zb)], out, outer, color)
            if inner is not None:
                mesh.add_quad_facing([P(ua, za, thickness), P(ub, za, thickness), P(ub, zb, thickness),
                                      P(ua, zb, thickness)], -out, inner, inner_color or color)

    # Reveals: four faces lining each opening, facing into the hole.
    for op in (ops if thickness > 1e-6 else []):
        a0, a1 = P(op.u0, op.z0), P(op.u1, op.z0)
        b0, b1 = P(op.u0, op.z1), P(op.u1, op.z1)
        t = -out * thickness
        mesh.add_quad_facing([a0, a1, a1 + t, a0 + t], up, reveal, color)      # sill
        mesh.add_quad_facing([b0, b1, b1 + t, b0 + t], -up, reveal, color)     # head
        mesh.add_quad_facing([a0, a0 + t, b0 + t, b0], d, reveal, color)       # left jamb
        mesh.add_quad_facing([a1, a1 + t, b1 + t, b1], -d, reveal, color)      # right jamb


def arc_points(center, radius, a0, a1, segments):
    cx, cy = center
    return [(cx + radius * math.cos(a0 + (a1 - a0) * k / segments),
             cy + radius * math.sin(a0 + (a1 - a0) * k / segments)) for k in range(segments + 1)]


def band_along_polyline(mesh: Mesh, pts2d, z0, z1, depth, mat, color=(1, 1, 1, 1), outward_sign=+1, closed=False):
    """A horizontal trim band that sticks out `depth` from a wall polyline (speed lines,
    cornices). outward_sign picks which side of the polyline is 'outside' (+1 = right)."""
    pts = [np.array([x, y, 0.0]) for x, y in pts2d]
    n = len(pts)
    segs = range(n if closed else n - 1)
    # Per-vertex outward normals (averaged) so corners mitre cleanly.
    seg_out = []
    for i in segs:
        a, b = pts[i], pts[(i + 1) % n]
        dd = normalize(b - a)
        seg_out.append(np.array([dd[1], -dd[0], 0.0]) * outward_sign)
    vout = []
    for k in range(n):
        ns = []
        if closed or k > 0:
            ns.append(seg_out[(k - 1) % len(seg_out)])
        if closed or k < n - 1:
            ns.append(seg_out[k % len(seg_out)])
        m = normalize(sum(ns))
        cosang = max(0.3, float(np.dot(m, ns[0])))
        vout.append(m / cosang)
    up = np.array([0, 0, 1.0])
    for i in segs:
        j = (i + 1) % n
        a, b = pts[i], pts[j]
        ao, bo = a + vout[i] * depth, b + vout[j] * depth
        so = seg_out[i % len(seg_out)]
        mesh.add_quad_facing([ao + up * z0, bo + up * z0, bo + up * z1, ao + up * z1], so, mat, color)
        mesh.add_quad_facing([ao + up * z1, bo + up * z1, b + up * z1, a + up * z1], up, mat, color)
        mesh.add_quad_facing([a + up * z0, b + up * z0, bo + up * z0, ao + up * z0], -up, mat, color)
    if not closed:  # end caps
        for k, sgn in ((0, -1), (n - 1, 1)):
            a, ao = pts[k], pts[k] + vout[k] * depth
            dd = normalize(pts[min(k + 1, n - 1)] - pts[max(k - 1, 0)])
            mesh.add_quad_facing([a + up * z0, ao + up * z0, ao + up * z1, a + up * z1], dd * sgn, mat, color)


def inset_polygon(pts2d, inset: float):
    """Inset a CCW polygon by `inset` with mitred corners (intersect adjacent offset edges)."""
    P = [np.asarray(p, float) for p in pts2d]
    n = len(P)
    out = []
    for i in range(n):
        a0, a1 = P[i - 1], P[i]          # incoming edge
        b0, b1 = P[i], P[(i + 1) % n]    # outgoing edge
        def off(p0, p1):
            d = normalize(p1 - p0)
            inward = np.array([-d[1], d[0]])  # left of CCW edge = inside
            return p0 + inward * inset, d
        pa, da = off(a0, a1)
        pb, db = off(b0, b1)
        den = da[0] * db[1] - da[1] * db[0]
        if abs(den) < 1e-9:              # collinear: just offset the vertex
            out.append(pb)
            continue
        tpar = ((pb[0] - pa[0]) * db[1] - (pb[1] - pa[1]) * db[0]) / den
        out.append(pa + da * tpar)
    return [tuple(map(float, q)) for q in out]


def save_meshes(path, meshes: dict[str, Mesh]):
    with open(path, "w") as f:
        json.dump({k: m.to_dict() for k, m in meshes.items()}, f)
