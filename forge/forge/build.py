"""MLO Forge: building spec → FiveM resource, in one command.

    python -m forge.build specs/building_001.json --blender "C:/Program Files/Blender Foundation/Blender 5.3/blender.exe"

Steps: geometry + rooms/portals (pure Python) → procedural/AI textures as DDS → Blender + Sollumz
scene and export → read-back validation → FiveM resource → run record for rating/learning.
Outputs go to build/<building_id>/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .building import build as build_model
from .building import summary
from .package import make_resource
from .runs import record_run
from .serialize import model_to_dict, write_model
from .textures import generate as generate_textures
from .validate import validate_export

HERE = Path(__file__).resolve().parent
BLENDER_SCRIPT = HERE.parent / "blender" / "build_in_blender.py"

DEFAULT_BLENDER = [
    r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "blender",
]


def find_blender(explicit: str | None) -> str:
    for c in ([explicit] if explicit else []) + [os.environ.get("BLENDER", "")] + DEFAULT_BLENDER:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    raise SystemExit("Blender not found: pass --blender <path to blender.exe> or set BLENDER")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m forge.build")
    ap.add_argument("spec")
    ap.add_argument("--out", default="build")
    ap.add_argument("--blender", default=None)
    ap.add_argument("--format", choices=["auto", "native", "cwxml"], default="auto",
                    help="native = binary GTA files (needs PyMateria in Sollumz, Windows); cwxml = CodeWalker XML")
    ap.add_argument("--overrides", default=None, help="folder of <texture_name>.png to use instead of procedural ones")
    ap.add_argument("--skip-blender", action="store_true", help="only geometry + textures + model.json")
    ap.add_argument("--factory", action="store_true",
                    help="run Blender with --factory-startup (ignores your other add-ons; Sollumz must still be installed)")
    a = ap.parse_args(argv)

    t0 = time.time()
    spec_path = Path(a.spec)
    spec = json.loads(spec_path.read_text())
    out = Path(a.out) / spec["id"]
    out.mkdir(parents=True, exist_ok=True)

    print("1/5 geometry, rooms, portals")
    model = build_model(spec)
    print("   ", json.dumps(summary(model)))

    print("2/5 textures (DDS, mip chains)")
    tex = generate_textures(spec, model.materials, out / "textures", Path(a.overrides) if a.overrides else None)
    model_json = write_model(model, tex, out / "model.json")

    report = {"passed": None}
    resource = None
    if not a.skip_blender:
        print("3/5 Blender + Sollumz build/export")
        blender = find_blender(a.blender)
        export_dir = out / "export"
        if export_dir.exists():
            shutil.rmtree(export_dir)
        cmd = [blender, "-b", *(["--factory-startup"] if a.factory else []), "--python", str(BLENDER_SCRIPT), "--",
               "--model", str(model_json), "--textures", str(out / "textures"), "--out", str(export_dir),
               "--format", a.format, "--save-blend", str(out / f"{spec['id']}.blend")]
        env = dict(os.environ)
        log = out / "blender.log"
        with open(log, "w") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=1800)
        tail = log.read_text(errors="replace").splitlines()[-30:]
        if p.returncode != 0 or not (export_dir / "export_manifest.json").exists():
            print("\n".join(tail))
            raise SystemExit(f"Blender step failed (see {log})")
        print("4/5 validate (read-back)")
        rep = validate_export(export_dir, model_to_dict(model, tex))
        report = rep.to_dict()
        for m in rep.info:
            print("    ok  ", m)
        for m in rep.warnings:
            print("    warn", m)
        for m in rep.errors:
            print("    FAIL", m)
        print("5/5 FiveM resource")
        fmt = json.loads((export_dir / "export_manifest.json").read_text())["format"]
        if fmt == "native":
            resource = make_resource(json.loads(model_json.read_text()), export_dir, out / "fivem",
                                     title=spec.get("title", spec["id"]))
            print(f"    resource ready: {resource}")
        else:
            print("    export is CodeWalker XML: install PyMateria in Sollumz (Windows) for a ready FiveM resource")
    (out / "validation.json").write_text(json.dumps(report, indent=1))
    run_id = record_run(Path(a.out) / "runs.sqlite", spec, hashlib.sha256(spec_path.read_bytes()).hexdigest()[:12],
                        summary(model), tex, report, str(resource or ""), time.time() - t0)
    print(f"done in {time.time() - t0:.0f}s · run #{run_id} recorded (rate it: python -m forge.runs rate {run_id} good|bad --tags ...)")
    if report.get("passed") is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
