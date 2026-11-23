"""Print what this Blender can do for the forge: Sollumz version, szio, native (PyMateria) export.

blender -b --python check_sollumz.py
"""
import importlib
import sys

import addon_utils

name = next((m.__name__ for m in addon_utils.modules() if m.__name__.split(".")[-1].lower().startswith("sollumz")), None)
if name is None:
    print("CHECK sollumz=MISSING  → install Sollumz 2.9.x (github.com/Sollumz/Sollumz/releases) in this Blender")
    sys.exit(1)
addon_utils.enable(name, default_set=True)
info = addon_utils.module_bl_info(sys.modules[name])
try:
    from szio.gta5 import AssetFormat, is_provider_available
    native = is_provider_available(AssetFormat.NATIVE)
    szio_ok = True
except Exception as e:  # noqa: BLE001
    native, szio_ok = False, f"error: {e}"
try:
    importlib.import_module(name + ".ydr.shader_materials").create_shader("normal_spec.sps")
    shaders = True
except Exception as e:  # noqa: BLE001
    shaders = f"error: {e}"
print(f"CHECK sollumz={info.get('version')} module={name} szio={szio_ok} shaders={shaders} native_export={native}")
if not native:
    print("      → for binary .ydr/.ytyp/.ymap: Blender → Edit → Preferences → Add-ons → Sollumz → Dependencies → install PyMateria")
