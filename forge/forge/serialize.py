"""BuildingModel → model.json (the only thing the Blender side reads)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .building import BuildingModel


def model_to_dict(m: BuildingModel, textures: dict) -> dict:
    return {
        "id": m.id,
        "prefix": m.prefix,
        "texture_dictionary": m.texture_dictionary,
        "materials": {k: asdict(v) for k, v in m.materials.items()},
        "textures": textures,
        "drawables": {
            name: {
                "kind": d.kind,
                "mesh": d.mesh.to_dict(),
                "collision": d.collision.to_dict() if d.collision else None,
                "lod_low": d.lod_low.to_dict() if d.lod_low else None,
                "lod_dist_high": d.lod_dist_high,
                "lod_dist_low": d.lod_dist_low,
                "room": d.room,
            }
            for name, d in m.drawables.items()
        },
        "mlo": {
            "name": m.mlo_name,
            "collision": m.mlo_collision.to_dict() if m.mlo_collision else None,
            "rooms": [asdict(r) for r in m.rooms],
            "portals": [asdict(p) for p in m.portals],
            "entities": [asdict(e) for e in m.mlo_entities],
        },
        "ymap": {"entities": [asdict(e) for e in m.ymap_entities]},
        "placement": m.placement,
    }


def write_model(m: BuildingModel, textures: dict, path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(model_to_dict(m, textures)))
    return path
