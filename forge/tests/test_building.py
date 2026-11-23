import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from forge.building import EXT_COLOR, INT_COLOR, build
from forge.geometry import Mesh, inset_polygon

SPEC = json.loads((Path(__file__).parents[1] / "specs" / "building_001.json").read_text())


@pytest.fixture(scope="module")
def model():
    return build(SPEC)


def test_inset_polygon_square():
    sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
    assert np.allclose(inset_polygon(sq, 0.5), [(0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)])


def test_names_are_lowercase_unique_and_short(model):
    names = list(model.drawables) + [model.mlo_name, model.texture_dictionary]
    assert len(names) == len(set(names))
    for n in names:
        assert re.fullmatch(r"[a-z0-9_]{1,40}", n), n


def test_budgets(model):
    b = SPEC["budgets"]
    assert model.drawables[f"{model.prefix}_ext"].mesh.triangle_count <= b["exterior_hd_tris"]
    assert model.drawables[f"{model.prefix}_ext"].lod_low.triangle_count <= b["exterior_lod_tris"]
    assert len(model.drawables[f"{model.prefix}_ext"].mesh.materials()) <= b["max_materials_hd"] + 8


def _outer_wall_faces(mesh: Mesh, W, D, r, tol=0.02):
    """Faces lying on the outer footprint planes (straight walls only)."""
    out = []
    for i, f in enumerate(mesh.faces):
        c = mesh.face_center(i)
        n = mesh.face_normal(i)
        if abs(n[2]) > 0.1:
            continue
        if abs(c[1] + D / 2) < tol and c[0] > -W / 2 + r:      # front plane
            out.append((i, np.array([0, -1, 0])))
        elif abs(c[0] - W / 2) < tol:                            # right
            out.append((i, np.array([1, 0, 0])))
        elif abs(c[1] - D / 2) < tol:                            # back
            out.append((i, np.array([0, 1, 0])))
        elif abs(c[0] + W / 2) < tol and c[1] > -D / 2 + r:      # left
            out.append((i, np.array([-1, 0, 0])))
    return out


def test_exterior_walls_face_outwards(model):
    ext = model.drawables[f"{model.prefix}_ext"].mesh
    W, D = SPEC["massing"]["footprint_m"]["width"], SPEC["massing"]["footprint_m"]["depth"]
    faces = _outer_wall_faces(ext, W, D, 2.5)
    assert len(faces) > 50
    bad = [i for i, want in faces if np.dot(ext.face_normal(i), want) < 0.99 and ext.face_mats[i] == "stucco"]
    assert not bad, f"{len(bad)} outer wall faces point inwards"


def test_interior_walls_face_into_their_room(model):
    for name, room in (("int_shop", "shop"), ("int_store", "storage")):
        mesh = model.drawables[f"{model.prefix}_{name}"].mesh
        r = next(x for x in model.rooms if x.name == room)
        centre = (np.array(r.bb_min) + np.array(r.bb_max)) / 2
        for i in range(len(mesh.faces)):
            n = mesh.face_normal(i)
            c = mesh.face_center(i)
            if mesh.face_area(i) < 1e-6:
                continue
            if abs(n[2]) > 0.9:   # floor up, ceiling down
                assert (n[2] > 0) == (c[2] < 1.0), (name, i)
                continue
            to_centre = centre - c
            to_centre[2] = 0
            assert np.dot(n, to_centre) > 0, f"{name} face {i} faces away from the room"


def test_interior_surfaces_have_r0_vertex_colour(model):
    for d in model.drawables.values():
        cols = {tuple(c) for f in d.mesh.face_cols for c in f}
        expected = INT_COLOR if d.kind in ("interior",) or d.name.endswith(("counter", "shelf", "fridge")) else EXT_COLOR
        assert cols == {expected}, d.name


def test_no_degenerate_faces(model):
    for d in model.drawables.values():
        for i in range(len(d.mesh.faces)):
            assert d.mesh.face_area(i) > 1e-5, (d.name, i, d.mesh.face_mats[i])


def test_portals_point_from_room_to_room(model):
    rooms = {r.name: r for r in model.rooms}
    centre_xy = np.zeros(2)
    for p in model.portals:
        q = np.asarray(p.corners)
        n = np.cross(q[1] - q[0], q[2] - q[1])
        n /= np.linalg.norm(n)
        c = q.mean(0)
        rf = rooms[p.room_from]
        from_c = (np.array(rf.bb_min) + np.array(rf.bb_max)) / 2
        assert np.dot(n, c - from_c) > 0, f"{p.kind} portal faces back into {p.room_from}"
        if p.room_to == "limbo":   # outward = away from the building centre
            assert np.dot(n[:2], c[:2] - centre_xy) > 0
        # planar, non-degenerate rectangle-ish
        area = 0.5 * np.linalg.norm(np.cross(q[2] - q[0], q[3] - q[1]))
        assert area > 0.5


def test_entry_is_walkable_and_glass_is_not(model):
    col = model.drawables[f"{model.prefix}_ext"].collision
    D = SPEC["massing"]["footprint_m"]["depth"]

    def blocked(x, z):
        """Is there a collision face on the front wall plane covering (x, z)?"""
        for i, f in enumerate(col.faces):
            pts = np.asarray([col.verts[j] for j in f])
            if np.ptp(pts[:, 1]) > 1e-6 or abs(pts[0, 1] + D / 2) > 1e-6:
                continue
            if pts[:, 0].min() <= x <= pts[:, 0].max() and pts[:, 2].min() <= z <= pts[:, 2].max():
                return True
        return False

    entry = next(p for p in model.portals if p.kind == "entry")
    ex = np.asarray(entry.corners)[:, 0].mean()
    assert not blocked(ex, 1.0), "front entry must be open in the collision"
    glass = next(p for p in model.portals if p.kind == "storefront")
    gx = np.asarray(glass.corners)[:, 0].mean()
    assert blocked(gx, 1.5), "storefront glass must be solid"


def test_mlo_entities_sit_inside_their_rooms(model):
    rooms = {r.name: r for r in model.rooms}
    for e in model.mlo_entities:
        r = rooms[e.room]
        d = model.drawables[e.archetype]
        lo, hi = d.mesh.bounds()
        a = math.radians(e.rotation_z_deg)
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = corners @ R.T + np.asarray(e.position)
        assert np.all(world >= np.array(r.bb_min) - 0.02) and np.all(world <= np.array(r.bb_max) + 0.02), e.archetype


def test_front_centre_anchor_puts_front_wall_on_the_spot():
    import math
    from forge.building import resolve_placement
    D = 11.0
    for h in (0.0, 90.0, 213.0):
        pl = resolve_placement({"position": [100.0, 200.0, 30.0], "heading_deg": h, "anchor": "front_centre"}, D)
        H = math.radians(pl["heading_deg"])
        # local front-centre (0, -D/2) rotated by H plus origin must equal the anchor
        fx = pl["position"][0] + (0 * math.cos(H) - (-D / 2) * math.sin(H))
        fy = pl["position"][1] + (0 * math.sin(H) + (-D / 2) * math.cos(H))
        assert (round(fx, 6), round(fy, 6)) == (100.0, 200.0)
        # the front (local -Y) must face where the player was looking: GTA heading h → (-sin h, cos h)
        out = (math.sin(H), -math.cos(H))
        look = (-math.sin(math.radians(h)), math.cos(math.radians(h)))
        assert abs(out[0] - look[0]) < 1e-9 and abs(out[1] - look[1]) < 1e-9


def test_props_have_collision(model):
    for d in model.drawables.values():
        if d.kind == "prop":
            assert d.collision is not None and d.collision.triangle_count == 12, d.name
