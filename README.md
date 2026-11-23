# viewmaxxing

AI pipeline for building game-ready 3D environments: real-world scans **and** AI-designed buildings (GTA-style MLOs).

**Current direction: [MLO Forge](docs/MLO_FORGE.md).** Describe a building → AI reference images → 3D proxy → clean rebuilt, textured, decal-heavy, game-optimised building → FiveM/GTA V MLO (engine-neutral glTF master). Every result gets rated, and the ratings train the pipeline.
- **[forge/](forge/README.md): working generator.** `python -m forge.build forge/specs/building_001.json` gives a FiveM resource (exterior + MLO interior). GTA/MLO rules it follows: [docs/GTA_MLO_NOTES.md](docs/GTA_MLO_NOTES.md)
- Building #1: spec [forge/specs/building_001.json](forge/specs/building_001.json), reference prompts [forge/prompts/building_001.md](forge/prompts/building_001.md)

Components reused by the forge:
- [viewfinder/](viewfinder/README.md): self-hosted photos → (imagined views) → splat on rented GPUs (WorldMirror 2.0 + Lyra 2.0 + gsplat)
- [kaggle/](kaggle/README.md): free-GPU room video → splat + collision notebook
- [docs/GAMEPLAN.md](docs/GAMEPLAN.md): original scan-to-game research · [docs/CHARACTER.md](docs/CHARACTER.md): your face → rigged character
