# viewfinder-open

An open, self-hosted equivalent of World Labs' **Atlas Viewfinder**: photos in → an explorable, game-ready 3D Gaussian splat out. It fills in parts of the place you never photographed. It runs on a GPU you rent by the hour.

> **Why not "reverse engineer" Viewfinder itself?** Atlas runs on World Labs' servers. Their website only uploads photos and streams results back, so there's no model in any app or web bundle to extract. Copying the service would also break their terms. This is a **clean-room rebuild** instead: we took what World Labs has published about how Viewfinder behaves and assembled the same capabilities from openly released models.

## What Viewfinder does (from World Labs' public descriptions) → what we use

| Viewfinder behaviour (World Labs, Sep–Nov 2026) | viewfinder-open |
|---|---|
| Reconstructs a real place from 1–64 photos; "faithful with as few as two or three" | **WorldMirror 2.0** (Tencent HY-World 2.0, Apr 2026): one forward pass → camera poses, depth, points, 3D Gaussians |
| "Fills in regions that no camera ever saw … the more it sees, the less it imagines" | **Lyra 2.0** (NVIDIA, SIGGRAPH Asia 2026) generates a camera-controlled walkthrough from an anchor photo. Those frames are added as extra views, so real photos dominate wherever they exist |
| "Pixel-perfect camera control", render camera paths | Camera-path presets in `vf/trajectories.py` (turn around, look around, walk, orbit …), passed to Lyra as exact poses |
| "Turns the point cloud into a complete splat scene that renders on-device" | WorldMirror again on photos + imagined views (one shared 3D frame), then **gsplat** refinement and **splat-transform** compression (SPZ/SOG/HTML) |
| Splats-to-mesh for games | Collision mesh (`collision.glb`) from the splat, plus the level / metric / floor-at-0 world frame from the Kaggle pipeline |

**What Atlas does that we don't:** it's one unified model trained to generate and reconstruct jointly. Ours is three models stitched together. It can also produce 1440p video up to a minute long, and its text-to-world mode corresponds roughly to HY-World 2.0's world-generation pipeline (not wired in yet; see Next steps).

## Pipeline

```
photos ─▶ prepare ─▶ recon (WorldMirror 2.0) ──▶ preview.html  (≈1 min, "instant" result)
                         │
                         ├─(imagine on)─▶ Lyra 2.0 per anchor photo + camera path ─▶ generated frames
                         ▼
                      fuse: WorldMirror 2.0 on photos + generated frames (one 3D frame)
                         ▼
                      views: map cameras back to full-res crops (wm_export.py)
                         ▼
                      colmap: level it, +Y up, floor y=0, metric scale (build_colmap.py)
                         ▼
                      refine: gsplat (optional) ─▶ export: scene.html/.spz/.sog/.ply, collision.glb, zip
```

Everything runs on one GPU box. A small web page (`vf/server.py`) lets you upload photos, pick a camera path, watch progress and download results.

## Rent a GPU and run it

**GPU:** H100 80 GB (best) or A100 80 GB. Lyra 2.0 is tested on 80 GB H100s; WorldMirror is small. Late-2026 prices: roughly **$1.50–3/h** for an H100 on RunPod, Vast.ai or Lambda; an A100 80 GB is about $0.70–1.50/h. A 24 GB card (4090) can run reconstruction and refinement but **not** Lyra.

1. **Rent a pod.** Choose an Ubuntu 22.04 + CUDA 12.x template and attach a **≥ 300 GB persistent volume** at `/workspace` (environments plus about 150 GB of weights). Make sure SSH is enabled.
2. **Install (once, about 1.5–2 h, mostly compiling flash-attn and downloading weights):**
   ```bash
   cd /workspace && git clone <your viewmaxxing repo url> viewmaxxing
   bash viewmaxxing/viewfinder/setup_gpu.sh
   ```
   You can run parts separately: `setup_gpu.sh base core hyw` first (reconstruction only), then `lyra` later. If Hugging Face asks for a login, run `huggingface-cli login` first.
3. **Start the server** (the command is printed at the end of setup) inside `tmux`, bound to localhost.
4. **Connect from your computer** through an SSH tunnel, so nothing is exposed to the internet:
   ```bash
   ssh -L 8000:localhost:8000 root@<pod-ip> -p <ssh-port>
   ```
   Open http://localhost:8000. If you expose the port directly instead, set `VF_TOKEN=<long random string>` before starting the server and paste it into the page.
5. **Stop the pod when you're done.** The volume keeps everything, so the next start skips setup.

### Rough time and cost per scene (H100)

| Job | Time | Cost at ~$2.50/h |
|---|---|---|
| 20–60 photos, reconstruct + short refine | ~8–12 min | ~$0.40 |
| + imagine (fast mode, 161 frames, 1 anchor) | +~5 min (incl. model load) | +~$0.20 |
| + imagine (quality mode) | +~20 min per anchor | +~$0.80 |

These are estimates from the projects' published timings (Lyra: about 9 min per 80 frames, or about 35 s with `--use_dmd`; WorldMirror and Lyra's splat reconstruction: about 1 min). I haven't measured them myself.

## Options (web form or `job.json`)

| Option | Default | What it does |
|---|---|---|
| `imagine.enabled` | off | Run Lyra 2.0 to invent unseen areas |
| `imagine.preset` | `turn_around` | `turn_around`, `look_around`, `walk_forward`, `pull_back`, `truck_right`, `orbit` |
| `imagine.anchors` | `[0]` | Which photos to start walking from |
| `imagine.distance` / `degrees` / `frames` | 0.6 m / 60° / 161 | Size and length of the path (frames = 1 + 80k) |
| `imagine.prompt` | generic | Describe the place; this helps Lyra stay on-topic |
| `imagine.fast` | on | Lyra's DMD distillation: about 15× faster |
| `refine.enabled` / `steps_scaler` | on / 0.25 | gsplat polish: 0.25 = 7.5k steps, 1 = 30k |
| `camera_height` / `known_scale` | 1.4 / – | How metric scale is guessed, or a manual override |

## Licences: personal use only

- **Lyra 2.0 weights:** NVIDIA *Internal Scientific Research and Development Model Licence*. Research use only, nothing produced for sale or distribution.
- **HY-World 2.0 / WorldMirror:** Tencent community licence. **Not licensed in the EU, UK or South Korea**, has acceptable-use rules, and needs a separate licence above 1M monthly users.
- Our code, gsplat (Apache 2.0) and splat-transform (MIT) are permissive. Don't publish or sell scenes made with this stack without checking those model licences.

## What's tested and what isn't

**Tested here (CPU sandbox, `pytest tests/` → 17 passed):**
- **Camera paths:** valid rotations, start exactly at the anchor, smooth, correct shapes.
- **WorldMirror → full-res views:** the crop/intrinsics mapping recovers the original focal length exactly (448.88 px at 640×480).
- **Levelling and metric scale:** on a synthetic room with known geometry, 2.5% scale error, 0.02° floor tilt, floor at y = 0. gsplat's own COLMAP loader reads the dataset.
- **Splat transform:** positions and full 3D covariances transform correctly.
- **Job server:** auth, upload validation, only published files downloadable, path traversal blocked, failures reported.
- **Whole pipeline end to end**, with fake WorldMirror, Lyra and gsplat processes that follow the real CLIs: every stage, file hand-off and output.

**Not tested:** the real models on a real GPU, and `setup_gpu.sh` on a fresh pod. Their flags and file formats were checked against the projects' source (WorldMirror `pipeline.py` and `inference_utils.py`; Lyra `lyra2_custom_traj_inference.py`; gsplat v1.5.3 `simple_trainer.py`). The first real run is still the real test. If a stage fails, the web page shows the last log lines, and the full log is `$VF_WORK/vf_jobs/<id>/log.txt`.

## Next steps
- Text / single image → full 360° world: wire in HY-World 2.0's WorldNav + WorldStereo 2.0 pipeline (it needs the 80 GB GPU too).
- Feed iPhone LiDAR depth and ARKit poses to WorldMirror as priors (`--prior_cam_path`, `--prior_depth_path`).
- Automatically pick anchors facing the least-covered directions.
