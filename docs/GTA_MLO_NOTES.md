# How GTA V's world is built, and the MLO rules the forge follows

Research notes from reading **muto-atlas** (MIT; rules measured on vanilla files and in-game tests), the **Sollumz** 2.9.1 source and tests, **CodeWalker** source and the **FiveM** docs. Each rule names its source. "Measured" means muto-atlas measured it on vanilla data, not a guess.

## 1. How Rockstar structures the world (RAGE engine)
- **Three separate layers** (muto `trunk/gta-fundamentals.md` §1):
  - **archetype**, in a `.ytyp`: *what* an object is (model name, bounds, lodDist, flags, dictionaries)
  - **entity**, in a `.ymap`: *where* it is
  - **drawable**, in a `.ydr/.ydd/.yft`: *how it looks* (mesh, shaders, LODs, embedded collision)

  Change an archetype and every placement of it changes.
- **Everything binds by joaat hash of the lowercase name.** A name that doesn't match fails silently: the entity never spawns. Names must be **lowercase and unique** across everything streamed.
- **Interiors (MLOs)** are archetypes of type `CMloArchetypeDef` with **rooms**, **portals** and their **own entity list**. Room 0 is **limbo**, the outside world. A `CMloInstanceDef` in a ymap places the interior.
  - Portals are quads between two rooms. The engine only renders through them, which is how GTA culls interiors.
  - A prop placed inside an interior *with a ymap* is culled: it must go in the MLO's entity list and be **attached to a room**.
- **LODs** (muto `branches/map/lod.md`, measured over 3.1 M vanilla entities):
  - Most vanilla entities are **ORPHANHD**: no LOD chain, a single drawable with embedded high/medium/low models.
  - LOD chains exist (`HD → LOD → SLOD1–4`), but aren't a simple ladder.
  - The entity flag `1572872` means "my LOD parent is in the parent ymap". Using it without a parent is a defect Rockstar never ships.
- **Vanilla interiors** (muto `branches/map/vanilla-interiors.md`, v_coroner / v_abattoir):
  - 3–600 k triangles, 120–250 lights, the plainest shaders (`default`, `spec`, `normal_spec`)
  - corners chamfered ~1.5–2 cm
  - shadows come from a separate low-poly black shadow mesh
  - dirt is separate overlay meshes
  - the shell is split per room group
- **Vertex colour `Color 1` is a lighting gate, not paint.** Interior surfaces have **R = 0** (cuts off sky and weather) and **B ≈ 1**. A Blender round trip breaks this and the room looks "wet" (muto `branches/look/vertex-color.md`).

## 2. File rules (muto `trunk/gta-fundamentals.md` §2, §5, §6)
| Rule | What the forge does |
|---|---|
| Textures must be **DDS** (PNG is silently skipped), power-of-two, mips down to 4×4, DXT1 for opaque, DXT5 for alpha. Normal and spec maps are Non-Color | `forge/dds.py` writes BC1/BC3 + mips itself; checked by decoding with Pillow |
| Sollumz embeds/packs the **DDS from disk** as-is; the texture name comes from the **file name** | Files are named `<texture_name>.dds` |
| Every drawable needs **`UVMap 0`** and **`Color 1`** (written with `color_srgb`) | `blender/build_in_blender.py` |
| **Object transforms bake into the export** | All geometry is built at its final coordinates; objects stay at identity |
| Map models use an **external texture dictionary**, and the archetype must name it or the textures never load | One shared `<prefix>_txd.ytd`, set on every archetype; validated |
| Embedded collision binds only if **`physicsDictionary` = the model's own name** | Set and validated |
| MLO entities: **flags 18350080**, **lodDist −1**, **attached to a room** (any of these wrong means invisible) | Set and validated |
| `.ytyp` must be declared with **`data_file 'DLC_ITYP_REQUEST'`**; stream files auto-load from `stream/` | `forge/package.py` writes `fxmanifest.lua` |
| Two stream files with the same name: FiveM silently uses one | Packager refuses duplicates |
| After changing an asset, **reconnect fully**; `restart` isn't enough | Printed by `install_resource.ps1` |
| A broken stream file silently drops the **whole** resource | Validation runs before packaging |
| The hard limit is 65,535 stream files per server; ymap entities use the Building/EntityDesc pools, not the Object pool | One ymap per building |

## 3. Tools and what each is for (all free)
- **Sollumz** (Blender add-on, GPL): builds and exports `.ydr/.ybn/.ytd/.ytyp/.ymap`.
  - Native binary needs **PyMateria** (Windows).
  - Its Python API (`create_shader`, `convert_obj_to_drawable`, `convert_obj_to_composite`, `ytyps.new_archetype`, `new_room/new_portal/new_entity`, `ymap_next` map groups, `sz_txds`) is what `build_in_blender.py` drives headlessly.
  - The module name depends on how it was installed, so it's found dynamically.
- **CodeWalker** (MIT): map viewer and editor, ytyp/ymap XML⇄binary conversion, game data lookups. When importing texture XML, it takes size, format and mips from the DDS itself.
- **muto-atlas** (MIT, Claude Code plugin):
  - builds 24 data layers from your own GTA install (archetypes, doors, LOD chains, lights, timecycles)
  - has a FiveM **native database**, which caught `GetGroundZFor_3dCoord`, a name that doesn't exist; the right one is `GetGroundZFor3dCoord`
  - has a Lua linter and the measured rules above
- **FiveM / txAdmin:** the local test server.

## 4. GTA 6
GTA 6 launched on 19 Nov 2026 on PS5/Xbox only, with no PC version or mod tools, so there's nothing to target yet. The forge keeps an engine-neutral master (`model.json`: meshes, materials, rooms, portals, placement), so a future GTA 6 exporter would only replace the Blender/Sollumz step.
