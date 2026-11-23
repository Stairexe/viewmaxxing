# viewmaxxing: game plan

**Goal:** Scan a real room with an iPhone 14 Pro, or start from ordinary photos. AI rebuilds the room in 3D, you clean it up in our own editor, then walk around it in the browser or take it into Blender or a game engine as an optimised splat plus a collision mesh.

Personal use only. Free and open-source tools where they exist, with paid tools as optional add-ons.

Research date: **23 Nov 2026**. This field changes month to month, so recheck versions before each phase.

---

## 0. Things to get straight first

| Thing | Reality |
|---|---|
| **TrueDepth vs LiDAR** | TrueDepth is the *front* Face ID camera. It only works out to about 1 m, so it's good for faces and small objects but useless for a room. For rooms, use the **rear LiDAR scanner** (the 14 Pro has one, range about 5 m). Our app uses LiDAR for rooms and TrueDepth only for an optional face or object mode. |
| **"Atlas"** | **World Labs Atlas**, announced 1 Sep 2026 and opened to everyone on 12 Nov 2026. It's a world model that reconstructs a scene from **1 to 64 photos**, *imagines* the parts you didn't photograph, and exports **3D Gaussian splats** and meshes (Splats-to-Mesh tool). Its sibling **Marble** has a public API, low-res collider mesh export ($20/mo Standard tier) and high-res textured mesh export ($35/mo Pro). No pricing has been published for Atlas itself, and the developer API covers **Marble, not Atlas**. It isn't free, so it's an *optional* "fill in the gaps" backend, not the core. |
| **Gaussian splats vs meshes** | A splat is millions of soft, coloured, semi-transparent blobs. It looks photoreal but has **no surfaces**, so you can't walk on it or collide with it. A mesh is triangles: worse looking, but physics and game engines understand it. **A walkable game level needs both:** the splat for looks and an invisible low-poly mesh for collision. Every serious pipeline in 2026 works this way. |
| **Blender** | **Blender 5.3 (Nov 2026) imports splats natively** (PLY, SPZ, glTF) and renders them in EEVEE and Cycles. It's **import-only**: there's no splat export yet. The free **KIRI 3DGS Render** add-on adds editing and splat-to-mesh. |

---

## 1. The pipeline at a glance

```
 ┌──────────── CAPTURE ────────────┐    ┌──────────────── BACKEND (GPU worker) ───────────────┐    ┌──── EDIT / PLAY ────┐
 │ iPhone 14 Pro                   │    │                                                     │    │                     │
 │  A) LiDAR video: RGB + depth +  │──▶ │ 1. Poses:    ARKit poses (A) or VGGT-Ω/MapAnything  │    │ Web editor (three.js│
 │     ARKit camera poses          │    │ 2. Depth:    LiDAR (A) or Depth Anything 3 / MoGe   │──▶ │  + Spark): crop,    │
 │  B) RoomPlan: walls/doors/      │    │ 3. Splat:    gsplat / Brush, depth-supervised       │    │  delete floaters,   │
 │     furniture boxes (USDZ)      │    │ 4. Clean:    floater removal, SAM object masks      │    │  AI select, relight │
 │  C) Plain photos (any camera)   │    │ 5. Mesh:     collision voxel mesh + MILo/2DGS mesh  │    │ Walk mode (Rapier   │
 │  D) One single photo            │    │ 6. Compress: SPZ / SOG, LOD, decimate               │    │  physics)           │
 └─────────────────────────────────┘    │ (opt) World Labs Atlas/Marble: hallucinate unseen   │    │ Export → Blender,   │
                                        └─────────────────────────────────────────────────────┘    │  Godot, Unity, UE   │
                                                                                                   └─────────────────────┘
```

The four input modes trade quality for convenience:

| Mode | Input | Quality | Main tech |
|---|---|---|---|
| **A. LiDAR scan** (best) | 1–3 min walk-through video with LiDAR depth and poses | ★★★★★ | ARKit poses + LiDAR depth → depth-supervised 3DGS |
| **B. RoomPlan** | Same walk-through | Clean, blocky geometry | Apple RoomPlan → perfect walls/floor collider and blockout |
| **C. Photos** | 10–200 photos from any camera | ★★★★ | VGGT-Ω / MapAnything → poses + points → 3DGS |
| **D. Single photo** | 1 image | ★★ (works from the photo's viewpoint, breaks when you walk away) | Depth Anything 3 Metric / MoGe-2 → depth mesh; Apple SHARP → splat; Atlas for a full 360° room |

Running A and B together on the same scan is the trick: RoomPlan gives a **clean, watertight floor and walls for collision**, and the splat gives the **photoreal look**.

---

## 2. Tool research: what to use and why

### 2.1 Capture (iPhone)

| Tool | Cost | What it gives you | Use |
|---|---|---|---|
| **Scaniverse** (Niantic) | Free on-device "Classic" mode | Unlimited on-device splats (SPZ) + mesh export | Phase 0 baseline to compare our results against |
| **Polycam** | Free tier exports GLTF only | LiDAR room mesh, splats in the cloud | Optional comparison only |
| **KIRI Engine** | Freemium | Splats, splat-to-mesh 3.0 (paid) | Optional |
| **Stray Scanner** / **Record3D** | Free / cheap | **Raw RGB video + LiDAR depth + ARKit poses**. Nerfstudio has a Record3D importer, so no COLMAP needed | **Phase 1 capture** before our own app exists |
| **Splatcam** | Free | Seeds the splat from LiDAR depth at capture time | Reference for our own capture app |
| **Our own Swift app** (Phase 3) | Free (Apple dev account needed for sideloading beyond 7 days) | ARKit `sceneDepth` + confidence + poses + RoomPlan + TrueDepth object mode, uploads straight to our backend | Final product |

### 2.2 AI depth and geometry (the "predicts depth from a photo" part)

| Model | What it does | License | Role |
|---|---|---|---|
| **Depth Anything 3** (ByteDance, ICLR 2026) | Depth + camera poses from 1 to many views; the Giant model also outputs splats | `DA3METRIC-LARGE`, `DA3-BASE`, `DA3-SMALL` are **Apache 2.0**; Large/Giant are CC BY-NC (fine for personal use) | Single-photo metric depth; LiDAR depth completion and cleanup |
| **VGGT-Ω** (Meta/Oxford, May 2026) | Feed-forward: poses + depth + point maps from hundreds of images in seconds; beats DA3 on benchmarks | Check repo | **Photo mode (C): replaces slow COLMAP** |
| **MapAnything** (Meta, 3DV 2026) | Metric 3D from images *plus* optional intrinsics/poses/depth; exports COLMAP format; one interface over VGGT, DA3, MASt3R, π³, MoGe | Apache-ish (check per model) | **Our backend's model switchboard**: feed it LiDAR depth + ARKit poses for best results |
| **MoGe-2** | Monocular metric geometry + normals | MIT | Single-photo fallback |
| **Apple SHARP** (`ml-sharp`) | **One photo → 3D Gaussian splat in under 1 s** | Apple research license (personal use OK) | Mode D "instant splat" |
| **SAM 2/3** | Click-to-segment objects | Apache | Editor: select a chair → delete or move it |

### 2.3 Splat training and processing

| Tool | Why |
|---|---|
| **gsplat** (Nerfstudio) | Fast CUDA trainer; supports depth loss, so LiDAR depth makes walls flat and kills floaters |
| **Brush** (Rust/WebGPU) | Trains on **Mac/AMD/Intel/NVIDIA and even in the browser**; takes COLMAP or Nerfstudio format. Use it if you don't have an NVIDIA GPU |
| **PlayCanvas `splat-transform`** (MIT, CLI) | Crop, filter, convert PLY↔SPZ↔SOG, **and generate a collision mesh from a splat** (`--collision-mesh` voxel carve with a player-capsule size). This is the key game-ready step |
| **SuperSplat** (PlayCanvas, free web editor) | Manual cleanup while our editor is still being built |
| **MILo** (SIGGRAPH Asia 2025) | Highest-quality splat→mesh with far fewer vertices; needs the original photos + poses, which we have |
| **2DGS / PGSR / GOF** | Alternative mesh extractors (research licenses) |
| **3DGS-to-PC** (Apache 2.0) | Quick splat → point cloud → Poisson mesh |

### 2.4 Viewing, editing and game engines

| Target | Tool |
|---|---|
| **Web (our editor + walk mode)** | **Spark 2.0** (World Labs, MIT, three.js): streams 100M+ splats, mixes splats with meshes, WebXR. Physics: **Rapier** (WASM) |
| **Blender** | **5.3 native import** (PLY/SPZ); **KIRI 3DGS Render** add-on (free, Apache 2.0) for editing, relighting and splat→mesh |
| **Godot** | `godot-gaussian-splatting` add-on (open source): mobile/VR, **generates `StaticBody3D` collision from splats in the editor** |
| **Unity** | `aras-p/UnityGaussianSplatting` (MIT) |
| **Unreal** | XScene (open source) or Volinga (paid) |
| **PlayCanvas** | Most complete web support: collisions, relighting, WebXR |

### 2.5 Free GPU compute (if there's no local NVIDIA card)

| Option | Free allowance |
|---|---|
| **Kaggle Notebooks** | 30 GPU-h/week (T4/P100, 16 GB), 9 h sessions |
| **Google Colab** | About 15–30 h/week of T4, not guaranteed |
| **Lightning AI** | Free GPU hours on signup |
| **Modal** | **$30/month free credit**, serverless. **Best fit for our backend worker** because it's a real API with no notebook babysitting |
| **Hugging Face Spaces (ZeroGPU)** | Good for the single-photo models (DA3, SHARP) |

A typical room (2 min video, about 300 frames, 30k iterations) trains in about 10–20 min on a T4 and about 5 min on an RTX 4090.

---

## 3. App architecture

One repo, three parts:

```
viewmaxxing/
├── ios/                 # Phase 3: SwiftUI capture app (ARKit + RoomPlan + TrueDepth)
├── backend/
│   ├── api/             # FastAPI: projects, uploads, jobs, artifacts
│   ├── worker/          # GPU pipeline steps (run locally, or on Modal)
│   │   ├── ingest.py        # unpack Record3D / Stray / photos / our app's format
│   │   ├── poses.py         # ARKit poses passthrough | VGGT-Ω | MapAnything
│   │   ├── depth.py         # LiDAR + DA3 depth completion; single-photo depth
│   │   ├── train_splat.py   # gsplat (CUDA) or Brush (anything else)
│   │   ├── clean.py         # floater removal, SAM masks, cropping
│   │   ├── mesh.py          # splat-transform collision mesh, MILo, decimation
│   │   ├── compress.py      # SPZ / SOG / LOD
│   │   └── worldlabs.py     # optional Atlas/Marble API call
│   └── storage/         # local disk or S3-compatible (MinIO / Cloudflare R2 free tier)
├── web/                 # React + three.js + Spark + Rapier: viewer, editor, walk mode
└── docs/
```

**Backend stack:** Python 3.12, FastAPI, SQLite (personal use, so no Postgres needed), and a simple job queue (`arq` + Redis, or a SQLite-backed queue). The worker is the same code whether it runs on your PC's GPU or as a Modal function.

**Main API:**
- `POST /projects` → `POST /projects/{id}/upload` (video+depth zip, photos, or a single image)
- `POST /projects/{id}/jobs {pipeline: "lidar" | "photos" | "single" | "atlas", options}`
- `GET /jobs/{id}` (progress stream over SSE)
- `GET /projects/{id}/artifacts` → `scene.spz`, `scene.sog`, `collision.glb`, `mesh.glb`, `roomplan.usdz`, `preview.mp4`
- `POST /projects/{id}/edits` (the editor's edit list, re-applied non-destructively)

**Editor features ("tweaks inside the app"):**
1. Crop box and sphere, lasso-delete floaters, opacity/size filters
2. **AI select**: click an object, SAM segments it in 2D, the selection is lifted into 3D splats, then delete, move or duplicate it
3. Transform, level the floor (auto-detect from RoomPlan or RANSAC on the floor plane), set real-world scale
4. Colour grade and exposure; relight later (PlayCanvas/KIRI-style)
5. Place a spawn point, regenerate the collision mesh, toggle the collider visualisation
6. **Walk mode**: WASD + mouse, capsule controller, gravity, step height; WebXR on a headset
7. **Fill gaps**: send the scan's photos to Atlas (optional, paid) and merge the hallucinated parts
8. Export bundle: `.spz` + `collision.glb` + `scene.json` (spawn point, scale, up axis) + Blender/Godot import notes

---

## 4. Getting a scene into Blender

1. Download the export bundle from the app.
2. Blender 5.3: **File → Import → PLY** (or the SPZ node) loads `scene.ply/.spz` as a native Gaussian-splat point cloud. Watch the axes: most splats are Z-forward, −Y-up, so our exporter pre-rotates to Blender's Z-up.
3. **File → Import → glTF** loads `collision.glb`. Hide it in renders and use it as the physics or nav proxy.
4. Optional: import `mesh.glb` (MILo) or `roomplan.usdz` when you want real geometry to model on top of.
5. Build the game: Blender → Godot (glTF), or skip Blender and load the bundle straight into Godot with the splat add-on.

**Why "optimised" matters:** a raw room splat is 1–5 M Gaussians and a 200–1000 MB PLY. After pruning and **SPZ** compression it's about 10× smaller; **SOG** is smaller still for the web. Keep the collision mesh under about 20k triangles.

---

## 5. Phased roadmap

| Phase | Deliverable | Est. effort |
|---|---|---|
| **0. Baseline (1 evening)** | Scan your room with Scaniverse (free) → export SPZ → open in Blender 5.3 and SuperSplat. This shows the quality bar and costs nothing. | 1–2 h |
| **1. Backend MVP** | FastAPI + worker. Pipeline **A** using **Record3D/Stray Scanner** data → gsplat with LiDAR depth loss → splat-transform collision mesh → SPZ. CLI first, then API. | 1–2 weekends |
| **2. Web viewer + walk mode** | React + Spark: load SPZ + collider, first-person walk with Rapier. Upload page and job progress. | 1 weekend |
| **3. AI modes** | Pipeline **C** (photos → VGGT-Ω/MapAnything → splat) and **D** (single photo → DA3 Metric depth mesh + SHARP splat). | 1–2 weekends |
| **4. Editor** | Crop, delete, SAM AI select, floor levelling, scale, spawn point, re-run collider, export bundle. | 2–3 weekends |
| **5. iOS capture app** | SwiftUI + ARKit: record RGB + `sceneDepth` + confidence + poses, RoomPlan pass, upload to the backend. TrueDepth object mode. | 2–3 weekends |
| **6. Extras** | Atlas/Marble "fill the gaps", MILo high-quality mesh, multi-room stitching, WebXR, Godot export template. | Ongoing |

---

## 6. Costs (personal use)

| Item | Cost |
|---|---|
| Everything in §2 except World Labs | **$0** |
| GPU | $0 with a local NVIDIA card or Kaggle/Colab/Modal free tiers |
| Apple developer account (only to keep our iOS app installed for more than 7 days) | $99/yr, optional; free sideloading works with weekly re-signing |
| World Labs Marble / Atlas (optional "imagine the unseen") | From $20/mo; Atlas pricing not published yet |

---

## 7. Capture tips that matter more than any model

- Light the room evenly. Turn off auto-exposure flicker (our app will lock exposure and white balance).
- Walk slowly in **two loops**: one at chest height pointing outward, one lower pointing at the floor and furniture.
- Cover every surface from 2–3 angles. Splats can't invent what was never seen (that's what Atlas is for).
- Mirrors, glass and TV screens will be messy in every method, so cover them or mask them in the editor.
- Keep each scan to about 1–3 min per room. Stitch rooms together in Phase 6 rather than doing one giant scan.

---

## 8. Open decisions

1. **Your computer:** NVIDIA GPU PC, Mac (Apple Silicon), or neither? This decides gsplat-local vs Brush-local vs Modal.
2. **Final game engine:** browser (our own), Godot, Unity or Unreal? The default is browser plus a Godot export.
3. **Use Atlas?** The paid World Labs credits are optional.
