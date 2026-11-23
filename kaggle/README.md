# Kaggle: room video → walkable splat

`room_to_splat.ipynb` turns a phone video of a room into:
- a Gaussian splat (`scene.ply`, `scene.spz`, `scene.sog`)
- a double-click HTML viewer (`scene.html`)
- a collision mesh (`collision.glb`)

The world is +Y up, the floor is at y = 0, and units are roughly metres.

**Run it:** Kaggle → New Notebook → File → Import Notebook → upload `room_to_splat.ipynb`. Then:
1. Settings: **GPU T4 x2**, Internet **On**.
2. Add Input → upload your video as a dataset.
3. Edit `INPUT_PATH` and click Run All.

A fast run takes about 25–40 min of your 30 weekly GPU hours; `QUALITY="full"` takes about 60–90 min.

| Step | Tool |
|---|---|
| Frames | ffmpeg; keeps the sharpest frame in each window |
| Camera poses | COLMAP (`pycolmap`, the GPU build when available) |
| Level + scale | `src/sfm_align.py`: finds up from the camera orientations, floor from a point-height histogram, metric scale from your phone's height above the floor |
| Training | gsplat v1.5.3 `simple_trainer.py` |
| Export | PlayCanvas `splat-transform` (SPZ/SOG/HTML) |
| Collider | `src/collider.py` (voxels → marching cubes → decimate, plus a floor slab) |

**Editing:** change the scripts in `src/`, then run `python build_notebook.py` to regenerate the notebook. The notebook embeds the scripts because Kaggle can't clone a private repo.

**Tested** (CPU, synthetic room with known geometry): frames → COLMAP → alignment gave 2.5% scale error, 0.02° floor tilt, and the floor at y = 0. Export and collider ran end to end. Training was **not** tested here because there's no GPU in this environment. The first real run on Kaggle is the test for that step.
