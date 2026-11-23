"""Render preview images of a built .blend (Workbench, textured) for the rating gallery.

blender -b <building>.blend --python render_preview.py -- --out previews/ [--size 1280]
"""
import argparse, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=1280)
a = ap.parse_args(argv)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 24
scene.cycles.use_denoising = False
scene.render.resolution_x, scene.render.resolution_y = a.size, int(a.size * 0.625)
world = bpy.data.worlds.new("sky"); scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.7, 0.95, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 4.0; sun.rotation_euler = (math.radians(50), 0, math.radians(35))
scene.collection.objects.link(sun)
ground = bpy.data.meshes.new("ground"); ground.from_pydata([(-60,-60,-0.01),(60,-60,-0.01),(60,60,-0.01),(-60,60,-0.01)],[],[(0,1,2,3)])
gobj = bpy.data.objects.new("ground", ground); scene.collection.objects.link(gobj)
gm = bpy.data.materials.new("ground"); gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.34, 0.32, 1)
ground.materials.append(gm)

# GTA shader node groups don't render outside the game: swap in simple preview materials
for mat in list(bpy.data.materials):
    if not mat.node_tree:
        continue
    img = next((n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.name == "DiffuseSampler" and n.image), None)
    if img is None:
        continue
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = "UVMap 0"
    nt.links.new(uv.outputs[0], tex.inputs[0])
    if "sign" in mat.name or "glow" in mat.name:
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs[1].default_value = 3.0
        nt.links.new(tex.outputs[0], em.inputs[0]); nt.links.new(em.outputs[0], out_n.inputs[0])
    else:
        bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bs.inputs["Roughness"].default_value = 0.1 if "glass" in mat.name else 0.7
        nt.links.new(tex.outputs[0], bs.inputs["Base Color"])
        if any(k in mat.name for k in ("glass", "poster", "grime")):
            nt.links.new(tex.outputs[1], bs.inputs["Alpha"])
        nt.links.new(bs.outputs[0], out_n.inputs[0])

# hide collision / helper objects
for o in scene.objects:
    st = getattr(o, "sollum_type", "")
    if "bound" in str(st) or o.name.endswith(("_col", ".bvh", ".poly_mesh")) or o.type == "EMPTY" and "bound" in str(st):
        o.hide_render = True
cam_data = bpy.data.cameras.new("cam"); cam_data.lens = 24
cam = bpy.data.objects.new("cam", cam_data); scene.collection.objects.link(cam); scene.camera = cam

def shoot(name, loc, target, hide=()):
    for o in scene.objects:
        if any(h in o.name for h in hide): o.hide_render = True
    cam.location = Vector(loc)
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    for o in scene.objects:
        if any(h in o.name for h in hide): o.hide_render = False

shoot("front_34", (-14, -20, 4.5), (0, 0, 3.5))
shoot("back_34", (14, 18, 6), (0, 0, 3))
shoot("roof", (0, -10, 26), (0, 0, 7))
# interior: stand inside the shop by the corner window, look towards the fridges and the counter
shoot("interior", (-4.8, -3.6, 1.65), (2.5, 1.5, 1.0))
print("PREVIEWS", sorted(p.name for p in out.glob("*.png")))
