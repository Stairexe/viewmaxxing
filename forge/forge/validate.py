"""Read the export back and check the things that fail *silently* in GTA/FiveM.

Every check here comes from a measured failure mode in muto-atlas (trunk/gta-fundamentals.md,
branches/map/*): unregistered/mismatched names, entity outside a room, wrong MLO flags, PNG or
non-power-of-two textures, empty dictionaries, missing physics dictionary, sentinel extents.

With a CodeWalker-XML export (Linux/CI) all checks run. With a native binary export (Windows +
PyMateria) the XML is not there, so only file-level checks run; use CodeWalker to inspect further.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .dds import is_pow2, read_header

NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _val(el):
    if el is None:
        return None
    return el.text if (el.text and el.text.strip()) else el.attrib.get("value")


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def err(self, m): self.errors.append(m)
    def warn(self, m): self.warnings.append(m)
    def ok(self, m): self.info.append(m)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self):
        return {"passed": self.passed, "errors": self.errors, "warnings": self.warnings, "info": self.info}


def validate_export(export_dir: Path, model: dict) -> Report:
    r = Report()
    export_dir = Path(export_dir)
    files = {p.name: p for p in export_dir.rglob("*") if p.is_file()}
    xml_mode = any(n.endswith(".ytyp.xml") for n in files)
    native_mode = any(n.endswith(".ytyp") for n in files)
    if not (xml_mode or native_mode):
        r.err("no .ytyp / .ytyp.xml in the export")
        return r
    ext = ".xml" if xml_mode else ""
    r.ok(f"export format: {'CodeWalker XML' if xml_mode else 'native binary'}")

    prefix = model["prefix"]
    expected_drawables = set(model["drawables"])
    for name in expected_drawables:
        if f"{name}.ydr{ext}" not in files:
            r.err(f"drawable {name}.ydr{ext} missing")
    if f"{model['mlo']['name']}.ybn{ext}" not in files:
        r.err(f"MLO collision {model['mlo']['name']}.ybn{ext} missing")
    if f"{model['texture_dictionary']}.ytd{ext}" not in files:
        r.err(f"texture dictionary {model['texture_dictionary']}.ytd{ext} missing")
    for n in files:
        stem = n.split(".")[0]
        if n.endswith((".ydr", ".ybn", ".ytd", ".ytyp", ".ymap", ".xml")) and not NAME_RE.match(stem):
            r.err(f"asset name not lowercase/simple: {n}")

    # textures: DDS on disk, power of two, mips, alpha formats where needed
    alpha_maps = {t for t, info in model["textures"].items() if info["format"] == "DXT5"}
    for tname, info in model["textures"].items():
        cand = [p for n, p in files.items() if n == f"{tname}.dds"]
        if xml_mode and not cand:
            r.err(f"texture {tname}.dds not exported next to the .ytd.xml")
            continue
        if cand:
            h = read_header(cand[0])
            if h["magic"] != b"DDS ":
                r.err(f"{tname}: not a DDS file")
            if not (is_pow2(h["width"]) and is_pow2(h["height"])):
                r.err(f"{tname}: {h['width']}x{h['height']} not power of two")
            want_mips = max(1, min(h["width"], h["height"]).bit_length() - 1 - 1)
            if h["mips"] < want_mips:
                r.err(f"{tname}: {h['mips']} mips, want {want_mips} (down to 4x4)")
            if tname in alpha_maps and h["format"] != "DXT5":
                r.err(f"{tname}: needs alpha but is {h['format']}")
    total_mb = sum(i["bytes"] for i in model["textures"].values()) / 1e6
    (r.warn if total_mb > 16 else r.ok)(f"texture dictionary ≈ {total_mb:.1f} MB (FiveM flags > 16 MB as oversized)")

    if not xml_mode:
        r.warn("native export: deep checks need the XML (rebuild with --format cwxml once to run them)")
        return r

    # ------------------------------------------------------------------ ytyp
    ytyp = ET.parse(files[f"{prefix}.ytyp.xml"]).getroot()
    archetypes = {}
    for a in ytyp.find("archetypes").findall("Item"):
        archetypes[_val(a.find("name"))] = a
    for name in expected_drawables:
        a = archetypes.get(name)
        if a is None:
            r.err(f"archetype {name} missing from the ytyp")
            continue
        if _val(a.find("textureDictionary")) != model["texture_dictionary"]:
            r.err(f"{name}: textureDictionary is not {model['texture_dictionary']} (textures would never load)")
        has_col = model["drawables"][name].get("collision") is not None
        if has_col and _val(a.find("physicsDictionary")) != name:
            r.err(f"{name}: embedded collision but physicsDictionary != own name (you'd walk through it)")
        bmin = [float(a.find("bbMin").attrib[k]) for k in "xyz"]
        bmax = [float(a.find("bbMax").attrib[k]) for k in "xyz"]
        if any(hi - lo < 1e-3 for lo, hi in zip(bmin, bmax)) and model["drawables"][name]["kind"] != "decal":
            r.warn(f"{name}: degenerate bounding box {bmin} {bmax}")
    mlo = archetypes.get(model["mlo"]["name"])
    if mlo is None or mlo.attrib.get("type") != "CMloArchetypeDef":
        r.err("MLO archetype missing")
        return r
    rooms = mlo.find("rooms").findall("Item")
    if _val(rooms[0].find("name")) != "limbo":
        r.err("room 0 must be limbo")
    ents = mlo.find("entities").findall("Item")
    attached = set()
    for rm in rooms[1:]:
        attached |= {int(x) for x in (rm.find("attachedObjects").text or "").split()}
    for i, e in enumerate(ents):
        an = _val(e.find("archetypeName"))
        if an not in archetypes:
            r.err(f"MLO entity {i} uses unknown archetype {an} (silently never spawns)")
        if int(_val(e.find("flags"))) != 18350080:
            r.err(f"MLO entity {an}: flags {_val(e.find('flags'))}, vanilla MLO entities use 18350080")
        if float(_val(e.find("lodDist"))) != -1.0:
            r.warn(f"MLO entity {an}: lodDist {_val(e.find('lodDist'))}, vanilla uses -1")
        if i not in attached:
            r.err(f"MLO entity {an} (#{i}) is not attached to any room (invisible, can break the MLO)")
    portals = mlo.find("portals").findall("Item")
    if not portals:
        r.err("MLO has no portals")
    for p in portals:
        corners = p.find("corners").findall("Item")
        if len(corners) != 4:
            r.err(f"portal with {len(corners)} corners")
    r.ok(f"ytyp: {len(archetypes)} archetypes, MLO {len(rooms)} rooms / {len(portals)} portals / {len(ents)} entities")

    # ------------------------------------------------------------------ ymap
    ymap_name = next((n for n in files if n.endswith(".ymap.xml")), None)
    if ymap_name is None:
        r.err("no ymap exported")
        return r
    ym = ET.parse(files[ymap_name]).getroot()
    items = ym.find("entities").findall("Item")
    if not items:
        r.err("ymap has no entities (Sollumz can export an empty ymap silently)")
    mins = [float(ym.find("streamingExtentsMin").attrib[k]) for k in "xyz"]
    if any(abs(v) > 1e30 for v in mins):
        r.err("ymap streaming extents are the empty sentinel")
    n_mlo = sum(1 for it in items if it.attrib.get("type") == "CMloInstanceDef")
    if n_mlo != 1:
        r.err(f"expected exactly one MLO instance in the ymap, found {n_mlo}")
    for it in items:
        an = _val(it.find("archetypeName"))
        if an not in archetypes:
            r.err(f"ymap entity uses unknown archetype {an}")
        if it.attrib.get("type") == "CEntityDef" and int(_val(it.find("flags"))) & 8:
            r.err(f"ymap entity {an} has 'LOD in parent map' but no parent (vanilla never does this)")
    r.ok(f"ymap: {len(items)} entities incl. {n_mlo} MLO instance")

    # ------------------------------------------------------------------ drawables
    for name in expected_drawables:
        d = ET.parse(files[f"{name}.ydr.xml"]).getroot()
        sg = d.find("ShaderGroup")
        if sg.find("TextureDictionary") is not None and len(sg.find("TextureDictionary")):
            r.warn(f"{name}: has embedded textures (expected the shared .ytd)")
        refs = {_val(p.find("Name")) for s in sg.find("Shaders").findall("Item")
                for p in s.find("Parameters").findall("Item") if p.attrib.get("type") == "Texture"}
        missing = sorted(t for t in refs if t and t not in model["textures"])
        if missing:
            r.err(f"{name}: references textures not in the dictionary: {missing[:5]}")
        layout = d.find(".//VertexBuffer/Layout")
        if layout is None or layout.find("Colour0") is None:
            r.err(f"{name}: no vertex colour (Color 1) in the vertex layout")
        if model["drawables"][name].get("collision") is not None and d.find("Bounds") is None:
            r.err(f"{name}: collision expected but no embedded Bounds")
    r.ok(f"{len(expected_drawables)} drawables checked")
    return r
