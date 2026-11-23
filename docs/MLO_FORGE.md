# MLO Forge: AI → game-ready buildings in a "next-gen Vice City" style

**Goal:** describe a building → AI makes reference images → our pipeline turns them into a **heavily detailed, game-optimised building** (exterior first, then interior MLO), with textures, decals and props. Every run gets rated, and the ratings train the system to get better.

**All-free rule:** every tool below is free and open source, or free to use. The one optional paid item is a rented GPU for the biggest model (Lyra 2.0), which you already have scripts for in `viewfinder/`.

---

## 0. Reality checks first

| Question | Answer (Nov 2026) | What we do |
|---|---|---|
| Can we put buildings into **GTA 6**? | GTA 6 launched **19 Nov 2026 on PS5/Xbox only**. There's no PC version announced, no mod tools, and console games can't be modded. | Build for **GTA V / FiveM MLOs** now, where the tooling is mature and free (Blender + **Sollumz** + CodeWalker). Keep every building in an **engine-neutral master format** (glTF + JSON), so it can be re-exported if GTA 6 PC modding arrives. |
| "GTA 6 style" | We can match the **look** (South Florida / "Leonida": art deco, pastel stucco, neon, sun-bleached grime, palms). | Original buildings, original brand names on signs. **No Rockstar logos, brands or copied in-game buildings.** That keeps it legal and keeps FiveM servers from pulling it. |
| Can splats become a game building directly? | Not well. Splat→mesh gives a blobby, millions-of-triangles mesh with lighting baked in. GTA-quality buildings are **clean low-poly shells + tiling materials + trim sheets + decal layers + separate props**. | Use splats and image-to-3D as a **3D reference**, then **rebuild** the building cleanly (procedural modules fitted to that reference). The detail comes from textures, decals and props, not from a dense mesh. This is the key design decision. |

---

## 1. The pipeline (exterior first)

```
 ① BRIEF ─────────▶ ② REFERENCE PACK ─────────▶ ③ 3D PROXY ─────────▶ ④ PARSE
 building.json       hero shot, 4 ortho          WorldMirror splat +     massing boxes,
 (style, size,       elevations, roof, details,   points; TRELLIS.2       floor heights,
 floors, use)        material swatches, decals    mesh of the hero         window/door grid,
                     (ChatGPT / Qwen / FLUX)      (rough 3D truth)         signs, props list
                                                                              │
 ⑨ LEARN ◀──── ⑧ QA + RATE ◀──── ⑦ OPTIMISE + EXPORT ◀──── ⑥ DRESS ◀──── ⑤ REBUILD
 reward model,   auto metrics +    LODs, collision, budget    textures,    Blender Geometry
 LoRA on the     your 👍/👎 +       checks → glTF master →     trims,       Nodes: walls, modular
 winners,        failure tags      Sollumz → .ydr/.ytd/       decals,      windows, cornices,
 param tuning                      .ybn/.ytyp/.ymap (FiveM)   dirt, props  balconies, roof kit
```

### ① Brief: `forge/specs/<building>.json`
A small spec: footprint (m), floors, floor height, use (corner store, motel, laundromat…), style tags, colour palette, signage text (original names), prop wishlist, and budgets. An LLM can draft it from one sentence. Everything downstream reads it. See `forge/specs/building_001.json`.

### ② Reference pack (AI images)
Generated **in this order, each from the previous image** (image editing keeps them consistent):

1. **Hero shot:** a 3/4 street view in the mood you want.
2. **Orthographic elevations** (front, left, right, back): flat, even light, **no shadows**, no perspective. These become the texture source and the measurement source.
3. **Roof plan:** top-down.
4. **Detail sheets:** close-ups of each window, door, cornice, awning and sign type.
5. **Material swatches:** seamless top-down tiles (stucco, terrazzo, brick, roof membrane, tile).
6. **Decal sheets:** stains, cracks, graffiti tags, posters, rust streaks, puddle marks, on a flat background.

Image tools (all free):

| Tool | Free how | Use for |
|---|---|---|
| **ChatGPT / Gemini (web)** | Free-tier daily image limits | Hero shots and edits when you're at your desk |
| **Qwen-Image-2512 / Qwen-Image-Edit-2511** | Apache 2.0, self-host (needs ~24 GB+; a rented GPU or a quantised version) | Best open model for **readable sign text** and multi-angle edits. **Qwen-Image-Layered** outputs transparent layers, which is ideal for decals |
| **FLUX.2 [klein] 4B** | Apache 2.0, fits ~8–13 GB, so it **runs on free Kaggle T4s** | Bulk generation and the **LoRA we train later** (step ⑨) |

**Rule that matters:** texture references must be **flat-lit** (overcast, no hard shadows), or the shadows get baked into the walls. Hero shots can be moody; elevations and swatches can't.

### ③ 3D proxy (a rough 3D truth to measure from)
- **WorldMirror 2.0** (already in `viewfinder/`) on the hero + elevations + detail views gives camera poses, depth, a point cloud and a splat. The poses tell us how the views relate.
- **TRELLIS.2** (Microsoft, **MIT**) on the hero gives a textured mesh of the whole building. That's good for overall massing: overhangs, balcony depth, roof shape.
- Optional: **Lyra 2.0** orbit from the hero (rented H100) for more generated views around the corner. This is the viewfinder "imagine" stage.

We **never ship** these meshes. They're measuring tapes.

### ④ Parse: turn pictures into numbers
- **Massing:** fit boxes and planes to the proxy point cloud (RANSAC planes, then rectilinear boxes) → footprint, storey heights, setbacks, parapets.
- **Facade grid:** on each orthographic elevation, segment windows, doors, signs, AC units, pipes and awnings with **SAM 2/3** (open), prompted by **Florence-2 / Grounding-DINO** (open) → bounding boxes in metres (elevation pixel → metre via the spec's width).
- **Output:** `parsed.json`, i.e. walls, openings with sizes, trim lines, sign rectangles, prop slots. An LLM sanity-checks it against the spec, e.g. "3 floors but only 2 window rows?".

### ⑤ Rebuild: clean, game-ready geometry (Blender, headless `bpy`)
A **Geometry Nodes building generator** driven by `parsed.json`:
- Wall panels with real **openings** cut where windows and doors are (needed later for interiors and MLO portals).
- **Modular kit:** window frames, doors, cornice and trim profiles swept along edges, balconies, railings, awnings, parapets, roof kit (AC units, vents, water tank, satellite dish). Kit pieces are AI-generated once (TRELLIS.2 from detail sheets), cleaned and then **reused** across every building.
- Result: low-poly, UV'd and instanced by construction. Optimisation is built in, not bolted on.

### ⑥ Dress: where "heavily detailed" comes from
| Layer | How | Tools (free) |
|---|---|---|
| **Tiling materials** | AI swatch → seamless → derive normal/height/roughness | Qwen/FLUX for albedo; **DeepBump** (Blender add-on) or **Material Maker** for normal/height; or the CC0 libraries **ambientCG / Poly Haven** |
| **Trim sheet** | One atlas strip of mouldings, edges and frames that all trim geometry maps onto | Blender bake |
| **Unique facade detail** | Project the flat-lit **orthographic elevation** onto the facade (it lines up because it's orthographic), then mask it so tiling materials show at full res | Blender projection + bake |
| **Decals** | Separate alpha planes: grime, cracks, graffiti, posters, rust, sign faces. Maps to GTA decal shaders | Qwen-Image-Layered / AI sheets → cut with SAM |
| **Dirt & wear** | AO + curvature + "rain streak" masks baked to **vertex colours** (GTA uses vertex colour for dirt/AO) | Blender bakes |
| **Emissive** | Neon tubes as geometry + emissive material; window glow cards | Kit + materials |
| **Props** | AC units, dumpsters, newspaper boxes, palms, bins, signs. Segment them in references → **TRELLIS.2** → retopo/decimate → UV (**xatlas**) → bake → **prop library** (tagged, reused) | TRELLIS.2, Blender, xatlas, Instant Meshes / QuadriFlow |

### ⑦ Optimise + export
- **LODs:** for the exterior HD → LOD → SLOD chain, automatically decimate, then bake to a single atlas for the far LODs (Blender, or **meshoptimizer/gltfpack**).
- **Collision:** simplified boxes from the massing, not from the render mesh.
- **Budget checks** (our targets, tune later): exterior HD ≤ ~60k tris, ≤ 12 materials, texel density ~512 px/m on hero walls, and each texture dictionary small enough that FiveM doesn't flag it as oversized (aim ≲ 16 MB).
- **Master:** `building.glb` + `building.meta.json` (materials, decals, LODs, collision, rooms, portals). Engine-neutral.
- **FiveM / GTA V:** headless Blender + **Sollumz** (open source; 2.9.x has native binary export, Blender 5.x) → `.ydr` (models), `.ytd` (textures), `.ybn` (collision), `.ytyp` (archetypes, MLO rooms/portals), `.ymap` (placement) + `fxmanifest.lua`. Check placement in **CodeWalker** (free, Windows), then test on a **local FiveM server** (free).
- **Other engines:** the same `.glb` drops into Godot, Unreal or Unity.

### ⑧ QA + rate
Every run is saved to a **run database** (SQLite + files): spec, prompts, seeds, model versions, every intermediate output and the final export.

- **Auto-metrics:**
  - **Faithfulness:** CLIP/SigLIP + DINOv2 similarity between renders and the reference pack.
  - **Style score:** similarity to *your* approved reference set.
  - **Mesh health:** non-manifold edges, flipped normals, tri and material budgets, UV overlap, texel density.
  - **Walkability:** Recast navmesh over the collision, door widths ≥ 1 m.
  - **Export validity:** Sollumz export succeeds, and the files load.
- **You rate it** in a small web gallery with turntable renders, a day/night toggle and a "walk-in" splat/mesh viewer. Give 👍/👎 or 1–5, plus **failure tags** like `warped_facade`, `floating_props`, `baked_shadows`, `wrong_scale`, `texture_seams`, `not_on_style`, `sign_text_garbled`, `too_clean`.

### ⑨ Learn from good vs bad results
Three loops, from cheap to expensive:
1. **Reward model** (immediately useful). A small model (logistic regression or LightGBM on CLIP/DINOv2 embeddings + metrics) predicts your rating. The pipeline then generates **N candidates per step** and keeps the top-scoring ones, so you only review the best. It trains in seconds on a CPU.
2. **Parameter tuning.** **Optuna** searches pipeline knobs (decimation ratios, decal density, dirt amount, voxel sizes, prompt templates) to maximise the reward model's score.
3. **Style LoRA.** Fine-tune **FLUX.2 [klein] 4B** on your **approved** references (free on Kaggle T4s; Qwen-Image on a rented GPU), so every new building comes out on-style. Later, preference-tune (Diffusion-DPO) on 👍/👎 pairs, and train a LoRA for "flat-lit orthographic elevation" so step ② stops needing retries.

The **failure tags** tell us which stage to fix: `baked_shadows` → prompts or the lighting rule; `floating_props` → placement; `warped_facade` → parsing.

---

## 2. Interiors (phase 2, after exteriors work)
1. **Floor plan:** an LLM turns the spec into a room graph (store floor, back room, stairs, apartment) → a plan that **reuses the exterior's real door and window openings**, so inside and outside line up.
2. **Room shells:** the same Geometry Nodes kit (walls, floors, ceilings, skirting, door frames) → clean MLO-ready rooms.
3. **Room references:** render each empty shell, then have the image model "furnish and dress" that exact view (image edit, conditioned on the shell render) → one reference per room.
4. **Layout from references:** WorldMirror on the room views gives a point cloud. **SpatialLM** (open, NeurIPS 2025; point cloud → walls, doors, windows and **object boxes with categories**) gives us where the furniture goes.
5. **Props:** match each box to the prop library; generate any missing ones (TRELLIS.2). Add decals (posters, stains), clutter scattering and light sources.
6. **MLO setup:** one Sollumz **room** per room, **portals** generated automatically from door openings, entity sets, timecycle modifiers and light emitters.
7. **The same QA:** walk the navmesh through every door and check portal visibility, plus the interior budget and ratings.

---

## 3. Tool list (all free)

| Job | Tool | Licence / cost | Runs on |
|---|---|---|---|
| Reference images | ChatGPT/Gemini web (free tier), **Qwen-Image-2512/Edit-2511/Layered**, **FLUX.2 klein 4B** | Free tier / Apache 2.0 / Apache 2.0 | Browser / rented GPU / **Kaggle T4** |
| Multi-view → 3D proxy | **WorldMirror 2.0** | Tencent community (not EU/UK/KR) | Kaggle T4 or rented |
| Image → 3D (massing, props, kit) | **TRELLIS.2** | MIT | Rented GPU (24 GB+) or free HF Space quota |
| Imagined extra views | **Lyra 2.0** (optional) | NVIDIA research-only | Rented H100 (only paid part) |
| Segmentation / detection | **SAM 2/3**, **Florence-2**, Grounding-DINO | Apache / MIT | Kaggle T4 |
| Indoor layout | **SpatialLM** | Open (check per model) | Kaggle T4 |
| Splats | gsplat, splat-transform (already in repo) | Apache / MIT | Kaggle / rented |
| Building generator, bakes, LODs | **Blender 5.3** (headless) + Geometry Nodes | GPL, free | CPU is fine; GPU speeds up bakes |
| Normal/height from albedo | **DeepBump**, **Material Maker** | GPL / MIT | CPU |
| Materials | **ambientCG**, **Poly Haven** | CC0 | n/a |
| UVs / retopo / LOD | **xatlas**, Instant Meshes, QuadriFlow, **meshoptimizer** | MIT / BSD | CPU |
| GTA V / FiveM export | **Sollumz**, **CodeWalker**, local **FiveM** server | Open source / free | Windows for CodeWalker/FiveM |
| Learning loop | scikit-learn / LightGBM, **Optuna**, open_clip, DINOv2, ai-toolkit / diffusers (LoRA) | Open | CPU + Kaggle |
| Run DB + rating UI | SQLite + FastAPI page (like `viewfinder/vf/server.py`) | Ours | Anywhere |

**Free compute plan:** Kaggle (30 GPU-h/week) for image generation, WorldMirror, SAM and the LoRA. Your PC's CPU, or Kaggle CPU, for Blender. ChatGPT/Gemini free tiers for hero images. Rent a GPU only if you want Lyra or full-size Qwen-Image.

---

## 4. How this reuses what's already in the repo
- `viewfinder/` → the **3D proxy** stage (WorldMirror, optional Lyra) and splat viewing.
- `kaggle/` → the free-GPU runner pattern (one notebook per heavy stage).
- `kaggle/src/collider.py`, alignment → collision and world-frame conventions (+Y up, metres; Sollumz converts to GTA's Z-up on export).
- `viewfinder/vf/server.py` → becomes the **rating gallery + run database** server.

---

## 5. Roadmap (milestones you can see)

| # | Milestone | Done when |
|---|---|---|
| M0 | **Reference pack for building #1** with ChatGPT + `forge/prompts/building_001.md` | You have hero + 4 flat elevations + swatches + decal sheet you like |
| M1 | **Blockout from refs:** spec + parsed elevations → Blender shell with real openings, correct size | Shell matches elevations within ~5% |
| M2 | **Dressed exterior:** materials, trim, projected facade, decals, dirt, 10+ props | Turntable render looks on-style; you rate it 👍 |
| M3 | **In FiveM:** LODs, collision, Sollumz export, standing in game on a local server | You can walk up to it in FiveM |
| M4 | **Rating loop:** run DB, gallery, reward model, best-of-N | Pipeline auto-picks the candidates you'd have picked |
| M5 | **Interior MLO** of building #1: rooms, portals, props | Walk in through the front door in FiveM |
| M6 | **Style LoRA** trained on approved refs; building #2 made with fewer retries | Fewer 👎 than building #1 |

## 6. Open decisions
1. **Target:** FiveM (GTA V) MLOs now, with the glTF master for later? *(Assumed yes.)*
2. **Your machine:** Windows PC (needed for CodeWalker/FiveM testing)? GPU?
3. **Building #1:** the default is the pastel art-deco **corner store with an apartment above** (`forge/specs/building_001.json`). It's small, iconic and decal-heavy, and its interior is simple.
