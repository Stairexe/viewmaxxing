"""Spec (building_XXX.json) → game-ready building model: meshes, materials, collision,
MLO rooms/portals/entities and placement. Pure Python; Blender only converts the result.

Building-local frame: metres, Z-up, origin at the centre of the footprint at ground level.
The front facade faces -Y. The rounded corner (if any) is front-left (-X, -Y).

Layout of the result (what Blender/Sollumz needs):
    drawables   name → {mesh, collision mesh (optional), lod meshes, kind}
    mlo         rooms, portals, entities (MLO-local = building-local)
    ymap        exterior entities + the MLO instance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import (Material, Mesh, Opening, arc_points, band_along_polyline, inset_polygon,
                       wall_with_openings)

EXT_COLOR = (1.0, 1.0, 1.0, 1.0)          # exterior: natural + artificial ambient on
INT_COLOR = (0.0, 0.25, 1.0, 1.0)         # interior: R=0 isolates from sky/weather (muto-atlas, vanilla-measured)
MLO_ENTITY_FLAGS = 18350080               # measured on vanilla MLO entities (muto-atlas map branch)
YMAP_ORPHAN_FLAGS = 1572864               # cast static + dynamic shadows, no LOD parent


@dataclass
class Drawable:
    name: str
    mesh: Mesh
    kind: str                       # exterior | interior | prop
    collision: Mesh | None = None   # embedded collision (exterior)
    lod_low: Mesh | None = None     # embedded low LOD
    lod_dist_high: float = 120.0
    lod_dist_low: float = 400.0
    room: str | None = None         # MLO room for interior drawables


@dataclass
class Room:
    name: str
    bb_min: list
    bb_max: list
    timecycle: str = "int_gasstation"
    floor_id: int = 0


@dataclass
class Portal:
    room_from: str
    room_to: str
    corners: list                   # 4 points, normal (right-hand) points from room_from to room_to
    kind: str = "door"


@dataclass
class Entity:
    archetype: str
    position: list
    room: str | None = None         # MLO entities only
    rotation_z_deg: float = 0.0


@dataclass
class BuildingModel:
    id: str
    prefix: str
    materials: dict[str, Material]
    drawables: dict[str, Drawable] = field(default_factory=dict)
    mlo_name: str = ""
    mlo_collision: Mesh | None = None
    rooms: list[Room] = field(default_factory=list)
    portals: list[Portal] = field(default_factory=list)
    mlo_entities: list[Entity] = field(default_factory=list)
    ymap_entities: list[Entity] = field(default_factory=list)
    placement: dict = field(default_factory=dict)
    texture_dictionary: str = ""
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------- materials
def make_materials(prefix: str) -> dict[str, Material]:
    def M(name, **kw):
        kw.setdefault("diffuse", f"{prefix}_{name}_d")
        kw.setdefault("normal", f"{prefix}_{name}_n")
        kw.setdefault("spec", f"{prefix}_{name}_s")
        return Material(name=name, **kw)

    return {
        "stucco": M("stucco", tile_m=2.0),
        "trim": M("trim", tile_m=1.0),
        "plinth": M("plinth", tile_m=1.0),
        "roof": M("roof", tile_m=2.0),
        "metal": M("metal", tile_m=1.0),
        "shutter": M("shutter", tile_m=1.0),
        "awning": M("awning", tile_m=2.0),
        "glass": M("glass", shader="normal_spec_alpha.sps", alpha=True, tile_m=2.0),
        "glass_dark": M("glass_dark", tile_m=2.0),
        "door_steel": M("door_steel", tile_m=1.0),
        "terrazzo": M("terrazzo", tile_m=2.0),
        "concrete": M("concrete", tile_m=2.0),
        "int_wall": M("int_wall", tile_m=2.0),
        "ceiling": M("ceiling", tile_m=1.2),
        "wood": M("wood", tile_m=1.0),
        "ac_unit": M("ac_unit", tile_m=0.8),
        "sign_main": Material("sign_main", shader="emissive.sps", diffuse=f"{prefix}_sign_main_d", emissive=True, tile_m=1.0),
        "sign_window": Material("sign_window", shader="emissive.sps", diffuse=f"{prefix}_sign_window_d", emissive=True, tile_m=1.0),
        "sign_open": Material("sign_open", shader="emissive.sps", diffuse=f"{prefix}_sign_open_d", emissive=True, tile_m=1.0),
        "fridge_glow": Material("fridge_glow", shader="emissive.sps", diffuse=f"{prefix}_fridge_glow_d", emissive=True, tile_m=1.0),
        "poster": M("poster", shader="normal_spec_alpha.sps", alpha=True, tile_m=1.0),
        "grime": M("grime", shader="normal_spec_alpha.sps", alpha=True, tile_m=1.0),
    }


def _uv_rect(mesh_face_pts, u0=0.0, v0=0.0, u1=1.0, v1=1.0):
    """Fit a whole texture (0..1) onto a quad, corner order a,b,c,d."""
    return [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]


# ---------------------------------------------------------------------- footprint
@dataclass
class Segment:
    name: str        # front / arc_N / left / right / back
    a: np.ndarray    # start (x, y)
    b: np.ndarray    # end (x, y)

    @property
    def length(self):
        return float(np.linalg.norm(self.b - self.a))

    @property
    def direction(self):
        d = self.b - self.a
        return d / np.linalg.norm(d)

    @property
    def outward(self):
        d = self.direction
        return np.array([d[1], -d[0]])   # right of a CCW outline = outside


def footprint_segments(W, D, r, arc_segments=6) -> list[Segment]:
    """CCW outline. Front edge y=-D/2, rounded corner at (-W/2, -D/2) with radius r."""
    x0, x1, y0, y1 = -W / 2, W / 2, -D / 2, D / 2
    segs = [
        Segment("front", np.array([x0 + r, y0]), np.array([x1, y0])),
        Segment("right", np.array([x1, y0]), np.array([x1, y1])),
        Segment("back", np.array([x1, y1]), np.array([x0, y1])),
        Segment("left", np.array([x0, y1]), np.array([x0, y0 + r])),
    ]
    if r > 0:
        pts = arc_points((x0 + r, y0 + r), r, math.pi, 1.5 * math.pi, arc_segments)
        for k in range(arc_segments):
            segs.append(Segment(f"arc_{k}", np.array(pts[k]), np.array(pts[k + 1])))
    return segs


def offset_segment(seg: Segment, inset: float) -> Segment:
    """The same wall moved `inset` metres inwards (for the interior face)."""
    n = -seg.outward * inset
    return Segment(seg.name, seg.a + n, seg.b + n)


# ---------------------------------------------------------------------- the generator
def build(spec: dict) -> BuildingModel:
    sid = spec["id"]
    prefix = spec.get("prefix") or ("vm_" + sid.replace("building_", "b"))
    prefix = prefix.lower()
    mats = make_materials(prefix)
    m = BuildingModel(id=sid, prefix=prefix, materials=mats, texture_dictionary=f"{prefix}_txd",
                      mlo_name=f"{prefix}_mlo")

    mass = spec["massing"]
    W, D = mass["footprint_m"]["width"], mass["footprint_m"]["depth"]
    r = 2.5 if "rounded" in mass.get("corner", "") else 0.0
    h0, h1 = mass["floor_height_m"][:2]
    t = 0.3                                  # wall thickness
    z_roof = h0 + h1
    z_par = z_roof + mass.get("parapet_height_m", 0.9)
    z_ceil = h0 - 0.3                        # ground-floor ceiling (slab above)
    segs = footprint_segments(W, D, r)
    seg = {s.name: s for s in segs}
    arcs = [s for s in segs if s.name.startswith("arc_")]

    # ----- openings per wall (u = metres along the wall from its start) -----------
    fl = seg["front"].length
    front_ops = [
        Opening(0.2, 3.4, 0.45, 3.2, "storefront", "front_glass_l"),
        Opening(3.6, 5.3, 0.0, 3.2, "entry", "front_entry"),
        Opening(5.5, 8.6, 0.45, 3.2, "storefront", "front_glass_r"),
    ]
    front_up = [Opening(u, u + 1.4, h0 + 0.8, h0 + 2.3, "window", f"front_win_{i}")
                for i, u in enumerate((0.9, 3.7, 6.5, 9.3)) if u + 1.4 < fl - 0.3]
    ll = seg["left"].length
    left_ops = [
        Opening(1.2, 2.3, 0.0, 2.3, "side_door", "left_door"),
        Opening(ll - 2.6, ll - 0.2, 0.45, 3.2, "storefront", "left_glass"),
    ]
    left_up = [Opening(u, u + 1.2, h0 + 0.8, h0 + 2.3, "window", f"left_win_{i}")
               for i, u in enumerate((1.4, 3.8, 6.2)) if u + 1.2 < ll - 0.3]
    bl = seg["back"].length
    back_ops = [
        Opening(2.0, 3.1, 0.0, 2.3, "back_door", "back_door"),
        Opening(bl - 4.0, bl - 3.0, 1.6, 2.4, "barred", "back_win"),
    ]
    back_up = [Opening(u, u + 1.0, h0 + 1.0, h0 + 2.0, "window", f"back_win_up_{i}") for i, u in enumerate((4.0, 9.0))]
    right_ops: list[Opening] = []
    ops_ground = {"front": front_ops, "left": left_ops, "back": back_ops, "right": right_ops}
    ops_upper = {"front": front_up, "left": left_up, "back": back_up, "right": []}
    for a in arcs:
        ops_ground[a.name] = [Opening(0.08, a.length - 0.08, 0.45, 3.2, "storefront", f"{a.name}_glass")]
        ops_upper[a.name] = []

    # ================================================================== EXTERIOR
    ext = Mesh(f"{prefix}_ext")
    for s in segs:
        o3 = (s.a[0], s.a[1], 0.0)
        out3 = (s.outward[0], s.outward[1], 0.0)
        dir3 = (s.direction[0], s.direction[1], 0.0)
        wall_with_openings(ext, o3, dir3, s.length, 0.0, h0, t, ops_ground[s.name], mats["stucco"], mats["trim"],
                           color=EXT_COLOR, outward=out3)
        wall_with_openings(ext, (s.a[0], s.a[1], 0.0), dir3, s.length, h0, z_par, t, ops_upper[s.name],
                           mats["stucco"], mats["trim"], color=EXT_COLOR, outward=out3)

    # CCW outline: front.a → right.a → back.a → left.a → arc points (arc_0.a == left.b)
    outline_closed = [seg["front"].a, seg["right"].a, seg["back"].a, seg["left"].a] + [a.a for a in arcs]
    outline_xy = [tuple(p) for p in outline_closed]

    # Plinth, speed lines, cornice (wrap all sides; speed lines only front+arc+left like streamline moderne)
    band_along_polyline(ext, outline_xy, 0.0, 0.45, 0.06, mats["plinth"], EXT_COLOR, closed=True)
    show_side = [seg["left"].a, seg["left"].b] + [a.b for a in arcs] + [seg["front"].b]
    show_xy = [tuple(p) for p in show_side]
    for zc in (h0 - 0.05, h0 + 0.25, z_roof - 0.15):
        band_along_polyline(ext, show_xy, zc - 0.07, zc + 0.07, 0.07, mats["trim"], EXT_COLOR)
    band_along_polyline(ext, outline_xy, z_par - 0.25, z_par, 0.22, mats["trim"], EXT_COLOR, closed=True)

    # Roof deck + parapet inner faces + cap
    ordered = [seg["front"], seg["right"], seg["back"], seg["left"], *arcs]
    roof_poly = inset_polygon([tuple(s_.a) for s_ in ordered], t)
    inner_outline = [Segment(s_.name, np.array(roof_poly[i]), np.array(roof_poly[(i + 1) % len(roof_poly)]))
                     for i, s_ in enumerate(ordered)]
    inner_by_name = {s_.name: s_ for s_ in inner_outline}
    ext.add_face([(x, y, z_roof) for x, y in roof_poly], mats["roof"], EXT_COLOR)
    for s_out, s_in in zip([seg["front"], seg["right"], seg["back"], seg["left"], *arcs], inner_outline):
        a, b = s_in.a, s_in.b
        ext.add_quad_facing([(a[0], a[1], z_roof), (b[0], b[1], z_roof), (b[0], b[1], z_par), (a[0], a[1], z_par)],
                            (-s_out.outward[0], -s_out.outward[1], 0), mats["stucco"], EXT_COLOR)
        ao, bo = s_out.a, s_out.b
        ext.add_quad_facing([(ao[0], ao[1], z_par), (bo[0], bo[1], z_par), (b[0], b[1], z_par), (a[0], a[1], z_par)],
                            (0, 0, 1), mats["trim"], EXT_COLOR)

    # Upper-floor windows: dark glass + shutters + sills; storefront mullions + glass
    def on_wall(s: Segment, u, z, depth=0.0):
        p = s.a + s.direction * u - s.outward * depth
        return np.array([p[0], p[1], z])

    for s in segs:
        out3 = np.array([s.outward[0], s.outward[1], 0.0])
        d3 = np.array([s.direction[0], s.direction[1], 0.0])
        for op in ops_upper[s.name]:
            q = [on_wall(s, op.u0, op.z0, 0.18), on_wall(s, op.u1, op.z0, 0.18),
                 on_wall(s, op.u1, op.z1, 0.18), on_wall(s, op.u0, op.z1, 0.18)]
            ext.add_quad_facing(q, out3, mats["glass_dark"], EXT_COLOR)
            # sill
            sill_o = on_wall(s, op.u0 - 0.08, op.z0 - 0.08, -0.08)
            ext.add_oriented_box(sill_o, d3, -out3, (0, 0, 1), (op.u1 - op.u0 + 0.16, 0.16, 0.08), mats["trim"], EXT_COLOR)
            # open shutters either side, angled 20° off the wall
            sw = (op.u1 - op.u0) / 2
            if s.name in ("front", "left"):
                for side, u_edge in ((-1, op.u0), (1, op.u1)):
                    base = on_wall(s, u_edge + (side < 0) * (-sw), op.z0, -0.02)
                    ext.add_oriented_box(base, d3, out3, (0, 0, 1), (sw, 0.04, op.z1 - op.z0), mats["shutter"], EXT_COLOR)
        for op in ops_ground[s.name]:
            if op.kind in ("storefront", "entry"):
                # frame posts every ~1.6 m + transom bar
                n_posts = max(1, int(round((op.u1 - op.u0) / 1.6)))
                for k in range(n_posts + 1):
                    u = op.u0 + (op.u1 - op.u0) * k / n_posts
                    ext.add_oriented_box(on_wall(s, u - 0.04, op.z0, 0.12), d3, -out3, (0, 0, 1),
                                         (0.08, 0.08, op.z1 - op.z0), mats["metal"], EXT_COLOR)
                if op.kind == "storefront":
                    ext.add_oriented_box(on_wall(s, op.u0, 2.55, 0.12), d3, -out3, (0, 0, 1),
                                         (op.u1 - op.u0, 0.08, 0.08), mats["metal"], EXT_COLOR)
                    q = [on_wall(s, op.u0, op.z0, 0.16), on_wall(s, op.u1, op.z0, 0.16),
                         on_wall(s, op.u1, op.z1, 0.16), on_wall(s, op.u0, op.z1, 0.16)]
                    ext.add_quad_facing(q, out3, mats["glass"], EXT_COLOR)
                    ext.add_quad_facing(q, -out3, mats["glass"], EXT_COLOR)
            elif op.kind in ("side_door", "back_door"):
                q = [on_wall(s, op.u0, op.z0, 0.2), on_wall(s, op.u1, op.z0, 0.2),
                     on_wall(s, op.u1, op.z1, 0.2), on_wall(s, op.u0, op.z1, 0.2)]
                ext.add_quad_facing(q, out3, mats["door_steel"], EXT_COLOR)
            elif op.kind == "barred":
                q = [on_wall(s, op.u0, op.z0, 0.2), on_wall(s, op.u1, op.z0, 0.2),
                     on_wall(s, op.u1, op.z1, 0.2), on_wall(s, op.u0, op.z1, 0.2)]
                ext.add_quad_facing(q, out3, mats["glass_dark"], EXT_COLOR)
                for k in range(5):
                    u = op.u0 + 0.1 + (op.u1 - op.u0 - 0.2) * k / 4
                    ext.add_oriented_box(on_wall(s, u - 0.015, op.z0, 0.04), d3, -out3, (0, 0, 1),
                                         (0.03, 0.03, op.z1 - op.z0), mats["metal"], EXT_COLOR)

    # Awning over the storefront, balcony over the entry, signs
    f = seg["front"]
    fd3 = np.array([f.direction[0], f.direction[1], 0])
    fo3 = np.array([f.outward[0], f.outward[1], 0])
    a0 = on_wall(f, 0.0, 3.35)
    ext.add_oriented_box(a0, fd3, fo3, (0, 0, 1), (8.8, 1.3, 0.12), mats["awning"], EXT_COLOR)
    bal = on_wall(f, 3.2, h0 - 0.05)
    ext.add_oriented_box(bal, fd3, fo3, (0, 0, 1), (2.5, 1.0, 0.15), mats["trim"], EXT_COLOR)
    for k in range(6):  # railing posts + top rail
        ext.add_oriented_box(bal + fd3 * (0.05 + 2.35 * k / 5) + fo3 * 0.9 + np.array([0, 0, 0.15]), fd3, fo3,
                             (0, 0, 1), (0.04, 0.04, 1.0), mats["metal"], EXT_COLOR)
    ext.add_oriented_box(bal + fo3 * 0.9 + np.array([0, 0, 1.12]), fd3, fo3, (0, 0, 1), (2.5, 0.05, 0.05),
                         mats["metal"], EXT_COLOR)
    # Main neon sign board above the parapet line (front), lightbox on the corner, window neon
    sign_o = on_wall(f, 1.0, z_roof + 0.05, -0.25)
    ext.add_oriented_box(sign_o, fd3, fo3, (0, 0, 1), (8.0, 0.12, 1.1), mats["metal"], EXT_COLOR, skip=("+v",))
    face = [sign_o + fo3 * 0.125, sign_o + fd3 * 8.0 + fo3 * 0.125,
            sign_o + fd3 * 8.0 + fo3 * 0.125 + np.array([0, 0, 1.1]), sign_o + fo3 * 0.125 + np.array([0, 0, 1.1])]
    ext.add_face(face, mats["sign_main"], EXT_COLOR, uvs=_uv_rect(face))
    win = on_wall(f, 0.5, 1.4, 0.1)
    face = [win, win + fd3 * 2.4, win + fd3 * 2.4 + np.array([0, 0, 0.6]), win + np.array([0, 0, 0.6])]
    ext.add_face(face, mats["sign_window"], EXT_COLOR, uvs=_uv_rect(face))
    lb = on_wall(f, fl - 2.2, 2.6, -0.3)
    ext.add_oriented_box(lb, fd3, fo3, (0, 0, 1), (1.6, 0.25, 0.6), mats["metal"], EXT_COLOR, skip=("+v",))
    face = [lb + fo3 * 0.255, lb + fd3 * 1.6 + fo3 * 0.255, lb + fd3 * 1.6 + fo3 * 0.255 + np.array([0, 0, 0.6]),
            lb + fo3 * 0.255 + np.array([0, 0, 0.6])]
    ext.add_face(face, mats["sign_open"], EXT_COLOR, uvs=_uv_rect(face))

    # Posters + grime on the right (party) wall, grime strip along the base everywhere
    rs = seg["right"]
    rd3, ro3 = np.array([*rs.direction, 0]), np.array([*rs.outward, 0])
    for k, (u, z, w, h) in enumerate(((1.5, 1.0, 1.0, 1.4), (2.7, 1.1, 0.9, 1.2), (6.0, 0.8, 2.4, 1.6))):
        p = on_wall(rs, u, z, -0.012)
        face = [p, p + rd3 * w, p + rd3 * w + np.array([0, 0, h]), p + np.array([0, 0, h])]
        ext.add_face(face, mats["poster"], EXT_COLOR, uvs=_uv_rect(face, u0=k / 3, u1=(k + 1) / 3))
    for s in segs:
        if s.length < 0.5:
            continue
        o3 = np.array([*s.outward, 0])
        d3 = np.array([*s.direction, 0])
        p = on_wall(s, 0.0, 0.45, -0.075)
        face = [p, p + d3 * s.length, p + d3 * s.length + np.array([0, 0, 1.0]), p + np.array([0, 0, 1.0])]
        ext.add_face(face, mats["grime"], EXT_COLOR, uvs=_uv_rect(face, u1=max(1.0, s.length / 2)))

    # ----- exterior collision: solid walls, gaps only where people walk in -----------
    col = Mesh(f"{prefix}_ext_col")
    walk_in = {"entry"}
    for s in segs:
        o3, d3 = (s.a[0], s.a[1], 0.0), (s.direction[0], s.direction[1], 0.0)
        gaps = [op for op in ops_ground[s.name] if op.kind in walk_in]
        wall_with_openings(col, o3, d3, s.length, 0.0, z_par, t, gaps, mats["concrete"], mats["concrete"],
                           inner=mats["concrete"], outward=(s.outward[0], s.outward[1], 0.0))
    col.add_face([(x, y, z_roof) for x, y in roof_poly], mats["concrete"])
    col.add_face([(x, y, z_ceil) for x, y in roof_poly[::-1]], mats["concrete"])

    # ----- embedded low LOD: the massing as a box-ish prism --------------------------
    low = Mesh(f"{prefix}_ext_low")
    for s in [seg["front"], seg["right"], seg["back"], seg["left"], *arcs]:
        low.add_quad_facing([(s.a[0], s.a[1], 0), (s.b[0], s.b[1], 0), (s.b[0], s.b[1], z_par), (s.a[0], s.a[1], z_par)],
                            (s.outward[0], s.outward[1], 0), mats["stucco"], EXT_COLOR)
    low.add_face([(x, y, z_par) for x, y in outline_xy], mats["roof"], EXT_COLOR)

    m.drawables[ext.name] = Drawable(ext.name, ext, "exterior", collision=col, lod_low=low,
                                     lod_dist_high=120.0, lod_dist_low=500.0)

    # ----- exterior props (separate drawables so they can be reused) -----------------
    ac = Mesh(f"{prefix}_ac_unit")
    ac.add_box((-0.35, 0.0, 0.0), (0.35, 0.45, 0.45), mats["ac_unit"], EXT_COLOR)
    m.drawables[ac.name] = Drawable(ac.name, ac, "prop", collision=bbox_collision(ac, mats["concrete"]),
                                    lod_dist_high=60, lod_dist_low=150)
    tank = Mesh(f"{prefix}_water_tank")
    tank.add_cylinder((0, 0, 0), 0.45, 1.3, mats["metal"], segments=14, color=EXT_COLOR)
    m.drawables[tank.name] = Drawable(tank.name, tank, "prop", collision=bbox_collision(tank, mats["concrete"]),
                                      lod_dist_high=80, lod_dist_low=200)

    for s_name, ops in (("front", front_up), ("left", left_up)):
        s = seg[s_name]
        for op in ops:
            p = on_wall(s, (op.u0 + op.u1) / 2, op.z0 - 0.7, 0.0)
            yaw = math.degrees(math.atan2(s.outward[1], s.outward[0])) - 90.0
            m.ymap_entities.append(Entity(ac.name, p.tolist(), rotation_z_deg=yaw))
    m.ymap_entities.append(Entity(tank.name, [W / 4, D / 4, z_roof], rotation_z_deg=0.0))
    m.ymap_entities.append(Entity(ac.name, [-W / 4, D / 4, z_roof], rotation_z_deg=0.0))
    m.ymap_entities.insert(0, Entity(ext.name, [0.0, 0.0, 0.0]))

    # ================================================================== INTERIOR (MLO)
    y_part = D / 2 - 3.2                    # partition between shop and storage
    xi0, xi1, yi0, yi1 = -W / 2 + t, W / 2 - t, -D / 2 + t, D / 2 - t
    shop = Mesh(f"{prefix}_int_shop")
    store = Mesh(f"{prefix}_int_store")
    for s in segs:
        si = inner_by_name[s.name]
        L = si.length
        # map each outer opening onto the inner wall by projecting its ends
        def to_inner_u(u_outer):
            p = s.a + s.direction * u_outer
            return float(np.dot(p - si.a, si.direction))
        mapped = []
        for op in ops_ground[s.name]:
            u0, u1 = max(0.0, to_inner_u(op.u0)), min(L, to_inner_u(op.u1))
            if u1 - u0 > 0.05:
                mapped.append(Opening(u0, u1, op.z0, min(op.z1, z_ceil - 0.01), op.kind, op.name))
        bins = [(0.0, L)]
        if s.name in ("left", "right") and abs(si.b[1] - si.a[1]) > 1e-6:
            cut = (y_part - si.a[1]) / (si.b[1] - si.a[1]) * L
            if 0 < cut < L:
                bins = [(0.0, cut), (cut, L)]
        for u0, u1 in bins:
            mid = si.a + si.direction * (u0 + u1) / 2
            mesh = store if mid[1] > y_part else shop
            ops = [Opening(op.u0 - u0, op.u1 - u0, op.z0, op.z1, op.kind, op.name)
                   for op in mapped if op.u1 > u0 and op.u0 < u1]
            start = si.a + si.direction * u0
            # interior face only: its 'outer' side points into the room
            wall_with_openings(mesh, (start[0], start[1], 0.0), (si.direction[0], si.direction[1], 0.0), u1 - u0,
                               0.0, z_ceil, 0.0, ops, mats["int_wall"], mats["int_wall"], color=INT_COLOR,
                               outward=(-s.outward[0], -s.outward[1], 0.0))
    # floors + ceilings (shop floor polygon includes the rounded corner)
    shop_poly = [p for p in roof_poly]
    shop_floor = [(x, min(y, y_part)) for x, y in shop_poly]
    shop.add_face([(x, y, 0.01) for x, y in shop_floor], mats["terrazzo"], INT_COLOR)
    shop.add_face([(x, y, z_ceil) for x, y in shop_floor[::-1]], mats["ceiling"], INT_COLOR)
    store.add_quad_facing([(xi0, y_part, 0.01), (xi1, y_part, 0.01), (xi1, yi1, 0.01), (xi0, yi1, 0.01)], (0, 0, 1),
                          mats["concrete"], INT_COLOR)
    store.add_quad_facing([(xi0, y_part, z_ceil), (xi1, y_part, z_ceil), (xi1, yi1, z_ceil), (xi0, yi1, z_ceil)],
                          (0, 0, -1), mats["ceiling"], INT_COLOR)
    # partition wall with an open doorway near the right side
    door_u0, door_u1 = W - t * 2 - 2.0, W - t * 2 - 1.0
    part_ops = [Opening(door_u0, door_u1, 0.0, 2.2, "doorway", "inner_door")]
    wall_with_openings(shop, (xi0, y_part - 0.08, 0.0), (1, 0, 0), xi1 - xi0, 0.0, z_ceil, 0.0, part_ops,
                       mats["int_wall"], mats["int_wall"], color=INT_COLOR, outward=(0, -1, 0))
    wall_with_openings(store, (xi0, y_part + 0.08, 0.0), (1, 0, 0), xi1 - xi0, 0.0, z_ceil, 0.0, part_ops,
                       mats["int_wall"], mats["int_wall"], color=INT_COLOR, outward=(0, 1, 0))
    m.drawables[shop.name] = Drawable(shop.name, shop, "interior", room="shop", lod_dist_high=60, lod_dist_low=0)
    m.drawables[store.name] = Drawable(store.name, store, "interior", room="storage", lod_dist_high=60, lod_dist_low=0)

    # interior props
    counter = Mesh(f"{prefix}_counter")
    counter.add_box((-1.5, -0.35, 0.0), (1.5, 0.35, 1.05), mats["wood"], INT_COLOR)
    counter.add_box((-1.55, -0.4, 1.05), (1.55, 0.4, 1.1), mats["trim"], INT_COLOR)
    shelf = Mesh(f"{prefix}_shelf")
    shelf.add_box((-1.2, -0.3, 0.0), (1.2, 0.3, 0.12), mats["metal"], INT_COLOR)
    for z in (0.5, 0.95, 1.4):
        shelf.add_box((-1.2, -0.3, z), (1.2, 0.3, z + 0.03), mats["metal"], INT_COLOR)
    shelf.add_box((-0.03, -0.3, 0.0), (0.03, 0.3, 1.7), mats["metal"], INT_COLOR)
    fridge = Mesh(f"{prefix}_fridge")
    fridge.add_box((-1.5, -0.4, 0.0), (1.5, 0.4, 2.1), mats["metal"], INT_COLOR, skip=("-y",))
    face = [(-1.45, -0.41, 0.1), (1.45, -0.41, 0.1), (1.45, -0.41, 2.0), (-1.45, -0.41, 2.0)]
    fridge.add_face(face, mats["fridge_glow"], INT_COLOR, uvs=_uv_rect(face, u1=3.0))
    for prop in (counter, shelf, fridge):
        m.drawables[prop.name] = Drawable(prop.name, prop, "prop", collision=bbox_collision(prop, mats["concrete"]),
                                          lod_dist_high=40, lod_dist_low=0)

    m.mlo_entities = [
        Entity(shop.name, [0.0, 0.0, 0.0], room="shop"),
        Entity(store.name, [0.0, 0.0, 0.0], room="storage"),
        Entity(counter.name, [W / 2 - 2.5, -D / 2 + 2.2, 0.0], room="shop", rotation_z_deg=90.0),
        Entity(shelf.name, [-1.5, -0.6, 0.0], room="shop"),
        Entity(shelf.name, [-1.5, 1.2, 0.0], room="shop"),
        Entity(shelf.name, [1.8, -0.6, 0.0], room="shop"),
        Entity(fridge.name, [-1.0, y_part - 0.5, 0.0], room="shop"),
    ]

    # rooms (index 0 must be limbo = the outside world)
    m.rooms = [
        Room("limbo", [0, 0, 0], [0, 0, 0]),
        Room("shop", [xi0, yi0, 0.0], [xi1, y_part, z_ceil]),
        Room("storage", [xi0, y_part, 0.0], [xi1, yi1, z_ceil]),
    ]

    # portals: every ground-floor opening the player can see/walk through, plus the inner doorway
    def portal_quad(s: Segment, op: Opening, room_from: str, room_to: str, toward):
        z1 = min(op.z1, z_ceil - 0.01)
        q = [on_wall(s, op.u0, op.z0, t / 2), on_wall(s, op.u1, op.z0, t / 2),
             on_wall(s, op.u1, z1, t / 2), on_wall(s, op.u0, z1, t / 2)]
        n = np.cross(q[1] - q[0], q[2] - q[1])
        if np.dot(n, toward) < 0:
            q = q[::-1]
        return Portal(room_from, room_to, [p.tolist() for p in q], op.kind)

    for s in segs:
        out3 = np.array([s.outward[0], s.outward[1], 0.0])
        for op in ops_ground[s.name]:
            if op.kind in ("storefront", "entry", "barred"):
                mid = on_wall(s, (op.u0 + op.u1) / 2, 1.0, t)
                room = "storage" if mid[1] > y_part else "shop"
                m.portals.append(portal_quad(s, op, room, "limbo", out3))
    q = [np.array([xi0 + door_u0, y_part, 0.0]), np.array([xi0 + door_u1, y_part, 0.0]),
         np.array([xi0 + door_u1, y_part, 2.2]), np.array([xi0 + door_u0, y_part, 2.2])]
    if np.dot(np.cross(q[1] - q[0], q[2] - q[1]), (0, 1, 0)) < 0:
        q = q[::-1]
    m.portals.append(Portal("shop", "storage", [p.tolist() for p in q], "doorway"))

    # MLO collision: floors, ceilings, partition (outer walls come from the exterior collision)
    mcol = Mesh(f"{prefix}_mlo")
    mcol.add_quad_facing([(xi0, yi0, 0.0), (xi1, yi0, 0.0), (xi1, yi1, 0.0), (xi0, yi1, 0.0)], (0, 0, 1), mats["concrete"])
    wall_with_openings(mcol, (xi0, y_part - 0.08, 0.0), (1, 0, 0), xi1 - xi0, 0.0, z_ceil, 0.16, part_ops,
                       mats["concrete"], mats["concrete"], inner=mats["concrete"], outward=(0, -1, 0))
    m.mlo_collision = mcol

    m.placement = resolve_placement(spec.get("placement") or {}, D)
    m.notes.append(f"footprint {W} x {D} m, {mass['floors']} floors, parapet top {z_par:.2f} m")
    return m


def bbox_collision(mesh: Mesh, mat: Material) -> Mesh:
    lo, hi = mesh.bounds()
    col = Mesh(f"{mesh.name}_col")
    col.add_box(tuple(lo), tuple(hi), mat)
    return col


def resolve_placement(p: dict, depth: float) -> dict:
    """World placement of the building origin (footprint centre at ground level).

    anchor "front_centre" (what /vm_here prints): `position` is the middle of the front wall on
    the ground and `heading_deg` is the direction you faced (looking out at the street). The
    building's front faces local -Y, so its heading is that +180°, and its centre sits D/2 behind.
    Without an anchor, position/heading are taken as the origin's own transform.
    """
    pos = list(p.get("position", [1700.0, 3260.0, 41.0]))
    h = float(p.get("heading_deg", 0.0))
    if p.get("anchor") == "front_centre":
        H = (h + 180.0) % 360.0
        a = math.radians(H)
        cx = pos[0] - math.sin(a) * (depth / 2)   # rotate (0, D/2) by H
        cy = pos[1] + math.cos(a) * (depth / 2)
        return {"position": [cx, cy, pos[2]], "heading_deg": H, "source": dict(p)}
    return {"position": pos, "heading_deg": h, "source": dict(p)}


def summary(m: BuildingModel) -> dict:
    tris = {n: d.mesh.triangle_count for n, d in m.drawables.items()}
    return {
        "prefix": m.prefix,
        "drawables": tris,
        "materials": sorted({mat for d in m.drawables.values() for mat in d.mesh.materials()}),
        "rooms": [r.name for r in m.rooms],
        "portals": len(m.portals),
        "mlo_entities": len(m.mlo_entities),
        "ymap_entities": len(m.ymap_entities),
    }
