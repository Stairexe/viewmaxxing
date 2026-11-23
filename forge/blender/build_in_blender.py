"""Build the Sollumz scene from model.json and export GTA V / FiveM assets.

Run (Windows, from a terminal):
  "C:\\Program Files\\Blender Foundation\\Blender 5.3\\blender.exe" -b --python build_in_blender.py -- \
      --model build\\building_001\\model.json --textures build\\building_001\\textures --out build\\building_001\\export

Needs the Sollumz extension (2.9.x) installed and enabled in that Blender. With PyMateria installed
(Sollumz preferences → Dependencies, Windows only) the export is native binary (.ydr/.ybn/.ytyp/.ymap/.ytd);
without it, CodeWalker XML (.ydr.xml …), which CodeWalker can convert.

Rules applied (muto-atlas, measured): object transforms are identity (geometry is built at its final
coordinates); every drawable has 'UVMap 0' and 'Color 1'; names are lowercase; images are DDS on disk;
normal/spec images are Non-Color; MLO entities use flags 18350080, lodDist -1 and a room.
"""

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import addon_utils
import bpy
from mathutils import Quaternion, Vector


# ------------------------------------------------------------------------------ Sollumz
def find_sollumz() -> str:
    """Sollumz' module name depends on how it was installed (muto-atlas tool-pitfalls §2)."""
    for mod in addon_utils.modules():
        name = mod.__name__
        if name.split(".")[-1].lower().startswith("sollumz"):
            addon_utils.enable(name, default_set=True)
            return name
    raise SystemExit("Sollumz is not installed in this Blender (Edit → Preferences → Get Extensions / Install from Disk)")


SZ = find_sollumz()
sz = lambda sub: importlib.import_module(f"{SZ}.{sub}")  # noqa: E731
SollumType = sz("sollumz_properties").SollumType
ArchetypeType = sz("sollumz_properties").ArchetypeType
AssetType = sz("sollumz_properties").AssetType
create_shader = sz("ydr.shader_materials").create_shader
convert_obj_to_drawable = sz("tools.drawablehelper").convert_obj_to_drawable
convert_obj_to_composite = sz("tools.boundhelper").convert_obj_to_composite
create_collision_material_from_index = sz("ybn.collision_materials").create_collision_material_from_index
get_maps = sz("ymap_next.properties.map").get_maps
get_addon_preferences = sz("sollumz_preferences").get_addon_preferences

CONCRETE = 1  # collision material index (muto-atlas data/collision_materials.tsv)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--textures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", choices=["auto", "native", "cwxml"], default="auto")
    ap.add_argument("--save-blend", default="")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------ helpers
def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def make_mesh_object(md: dict, name: str, materials: dict):
    """Mesh object from the engine-neutral mesh dict: per-corner UVs ('UVMap 0') and
    vertex colours ('Color 1', written with color_srgb as Sollumz/GTA expect)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in md["verts"]], [], [tuple(f) for f in md["faces"]])
    names = sorted(set(md["face_mats"]))
    for n in names:
        me.materials.append(materials[n])
    index = {n: i for i, n in enumerate(names)}
    for poly, mname in zip(me.polygons, md["face_mats"]):
        poly.material_index = index[mname]
    uv = me.uv_layers.new(name="UVMap 0")
    col = me.color_attributes.new("Color 1", "BYTE_COLOR", "CORNER")
    for poly, fuv, fcol in zip(me.polygons, md["face_uvs"], md["face_cols"]):
        for k, li in enumerate(poly.loop_indices):
            uv.data[li].uv = fuv[k]
            col.data[li].color_srgb = fcol[k]
    me.validate(clean_customdata=False)
    me.update()
    obj = link(bpy.data.objects.new(name, me))
    return obj


def make_materials(model: dict, tex_dir: Path) -> dict:
    images = {}

    def image(tex_name: str, colorspace: str):
        if tex_name in images:
            return images[tex_name]
        path = tex_dir / "dds" / f"{tex_name}.dds"
        if not path.exists():
            raise SystemExit(f"missing texture {path}")
        img = bpy.data.images.load(str(path), check_existing=True)
        img.name = tex_name
        try:
            img.colorspace_settings.name = "sRGB" if colorspace == "sRGB" else "Non-Color"
        except TypeError:
            pass
        images[tex_name] = img
        return img

    mats = {}
    for name, m in model["materials"].items():
        mat = create_shader(m["shader"])
        mat.name = f"{model['prefix']}_{name}"
        nodes = {n.name: n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"}
        for slot, key, cs in (("DiffuseSampler", "diffuse", "sRGB"), ("BumpSampler", "normal", "Non-Color"),
                              ("SpecSampler", "spec", "Non-Color")):
            if slot in nodes and m.get(key):
                node = nodes[slot]
                node.image = image(m[key], cs)
                node.texture_properties.embedded = False   # shared external .ytd
        mats[name] = mat
    return mats, images


def collision_object(md: dict, name: str):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in md["verts"]], [], [tuple(f) for f in md["faces"]])
    me.materials.append(create_collision_material_from_index(CONCRETE))
    me.update()
    obj = link(bpy.data.objects.new(name, me))
    return convert_obj_to_composite(obj, SollumType.BOUND_GEOMETRYBVH, "General (Default)")


def yaw_quat(deg: float) -> Quaternion:
    return Quaternion((0.0, 0.0, 1.0), math.radians(deg))


# ------------------------------------------------------------------------------ build
def build(model: dict, tex_dir: Path):
    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    mats, images = make_materials(model, tex_dir)
    prefix = model["prefix"]

    # one shared texture dictionary for the whole building (like vanilla map models)
    txd = scene.sz_txds.new_texture_dictionary(model["texture_dictionary"])
    for img in images.values():
        txd.new_texture(img)

    drawables = {}
    for name, d in model["drawables"].items():
        obj = make_mesh_object(d["mesh"], name, mats)
        if d.get("lod_low"):
            low = make_mesh_object(d["lod_low"], f"{name}_low", mats)
            low_mesh = low.data
            bpy.data.objects.remove(low)
        else:
            low_mesh = None
        drawable = convert_obj_to_drawable(obj)
        drawable.name = name
        model_obj = next(c for c in drawable.children if c.sollum_type == SollumType.DRAWABLE_MODEL)
        model_obj.sz_lods.high.mesh = model_obj.data            # export skips drawables without it
        if low_mesh is not None:
            model_obj.sz_lods.low.mesh = low_mesh
        dp = drawable.drawable_properties
        dp.lod_dist_high = d["lod_dist_high"]
        dp.lod_dist_med = d["lod_dist_high"]
        dp.lod_dist_low = d["lod_dist_low"] or d["lod_dist_high"]
        dp.lod_dist_vlow = d["lod_dist_low"] or d["lod_dist_high"]
        if d.get("collision"):
            comp = collision_object(d["collision"], f"{name}_col")
            comp.parent = drawable                                   # embedded collision
        drawables[name] = drawable

    mlo = model["mlo"]
    mlo_col = collision_object(mlo["collision"], mlo["name"])
    mlo_col.name = mlo["name"]

    # --------------------------------------------------------------- ytyp
    ytyp = scene.ytyps.add()
    ytyp.name = prefix
    scene.ytyp_index = len(scene.ytyps) - 1
    for name, obj in drawables.items():
        d = model["drawables"][name]
        a = ytyp.new_archetype(ArchetypeType.BASE)
        a.name = name
        a.asset = obj
        a.asset_type = AssetType.DRAWABLE
        a.texture_dictionary = model["texture_dictionary"]
        a.physics_dictionary = name if d.get("collision") else ""
        a.lod_dist = max(d["lod_dist_low"], d["lod_dist_high"]) if d["kind"] != "interior" else 100.0
        a.hd_texture_dist = min(60.0, a.lod_dist)

    arch = ytyp.new_archetype(ArchetypeType.MLO)
    arch.name = mlo["name"]
    arch.asset = mlo_col
    arch.asset_type = AssetType.ASSETLESS
    arch.lod_dist = 150.0
    room_ids = {}
    for r in mlo["rooms"]:
        room = arch.new_room()
        room.name = r["name"]
        room.bb_min = r["bb_min"]
        room.bb_max = r["bb_max"]
        if r["name"] != "limbo":
            room.timecycle = r["timecycle"]
        room_ids[r["name"]] = str(room.id)
    for p in mlo["portals"]:
        portal = arch.new_portal()
        portal.room_from_id = room_ids[p["room_from"]]
        portal.room_to_id = room_ids[p["room_to"]]
        portal.corner1, portal.corner2, portal.corner3, portal.corner4 = [Vector(c) for c in p["corners"]]
    for e in mlo["entities"]:
        ent = arch.new_entity()
        ent.archetype_name = e["archetype"]
        ent.position = e["position"]
        ent.rotation = yaw_quat(e["rotation_z_deg"])
        ent.attached_room_id = room_ids[e["room"]]
        ent.flags.total = str(18350080) if isinstance(ent.flags.total, str) else 18350080
        ent.lod_dist = -1.0
    ArchetypeProperties = sz("ytyp.properties.ytyp").ArchetypeProperties
    ArchetypeProperties.update_cached_room_enum_items(arch.uuid)
    ArchetypeProperties.update_cached_portal_enum_items(arch.uuid)

    # --------------------------------------------------------------- ymap
    maps = get_maps(bpy.context, create_if_missing=True)
    group = maps.new_group()
    group.name = prefix
    md = group.new_map()
    md.name = prefix
    place = model["placement"]
    origin = Vector(place["position"])
    heading = yaw_quat(place.get("heading_deg", 0.0))
    lo, hi = Vector((1e9, 1e9, 1e9)), Vector((-1e9, -1e9, -1e9))

    def add_entity(archetype, local_pos, yaw_deg, lod_dist, is_mlo=False, radius=10.0):
        nonlocal lo, hi
        e = group.new_entity()
        e.map_data_uuid = md.uuid
        e.archetype_name = archetype
        world = origin + heading @ Vector(local_pos)
        e.position = world
        e.rotation = heading @ yaw_quat(yaw_deg)
        e.lod_dist = lod_dist
        e.lod_level = "HD"
        e.flags.total = str(1572864) if isinstance(e.flags.total, str) else 1572864
        e.is_mlo = is_mlo
        lo = Vector(map(min, lo, world - Vector((radius,) * 3)))
        hi = Vector(map(max, hi, world + Vector((radius,) * 3)))
        return e

    for e in model["ymap"]["entities"]:
        d = model["drawables"][e["archetype"]]
        add_entity(e["archetype"], e["position"], e["rotation_z_deg"], max(d["lod_dist_low"], d["lod_dist_high"]),
                   radius=12.0)
    add_entity(mlo["name"], (0.0, 0.0, 0.0), 0.0, 150.0, is_mlo=True, radius=12.0)
    md.extents_manual = True
    md.entities_extents_min, md.entities_extents_max = lo, hi
    pad = Vector((500.0, 500.0, 200.0))     # stream in before the far LOD distance
    md.streaming_extents_min, md.streaming_extents_max = lo - pad, hi + pad
    return drawables, mlo_col


def export(out_dir: Path, fmt: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefs = get_addon_preferences(bpy.context).export_settings
    from szio.gta5 import AssetFormat, is_provider_available
    native_ok = is_provider_available(AssetFormat.NATIVE)
    if fmt == "native" and not native_ok:
        raise SystemExit("Native export needs PyMateria (Sollumz preferences → Dependencies, Windows only)")
    use_native = native_ok if fmt == "auto" else fmt == "native"
    prefs.target_formats = {"NATIVE"} if use_native else {"CWXML"}
    prefs.target_versions = {"GEN8"}      # FiveM runs the legacy (gen8) game
    prefs.limit_to_selected = False
    prefs.export_ytyps = True
    prefs.export_ymaps = True
    prefs.export_ytds = True
    for attr in ("export_ytyps_include", "export_ymaps_include", "export_ytds_include"):
        if hasattr(prefs, attr):
            setattr(prefs, attr, "ALL")
    for o in bpy.context.scene.objects:
        o.hide_set(False)
        o.hide_select = False
    with bpy.context.temp_override(selected_objects=list(bpy.context.scene.objects)):
        res = bpy.ops.sollumz.export_assets(directory=str(out_dir), direct_export=True)
    print("EXPORT", res, "native" if use_native else "cwxml")
    return "native" if use_native else "cwxml"


def main():
    a = parse_args()
    model = json.loads(Path(a.model).read_text())
    build(model, Path(a.textures))
    used = export(Path(a.out), a.format)
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.save_blend).resolve()))
    files = sorted(str(p.relative_to(a.out)) for p in Path(a.out).rglob("*") if p.is_file())
    (Path(a.out) / "export_manifest.json").write_text(json.dumps({"format": used, "files": files}, indent=1))
    print("FILES", len(files))
    for f in files:
        print("  ", f)


if __name__ == "__main__":
    main()
