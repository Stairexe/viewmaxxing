# Your face → a rigged 3D game character

The goal is **your real face** on a body you build with other tools, rigged so it can walk around the scanned room.

Researched 23 Nov 2026.

---

## Pick a route

| Route | Likeness | Effort | Cost | Needs a PC? |
|---|---|---|---|---|
| **A. Selfie → avatar** (Avaturn or MetaPerson) | ★★★ | 15 min | Free (MetaPerson: first avatar free) | No, it runs in the browser |
| **B. MetaHuman** (iPhone TrueDepth → Unreal MetaHuman → Blender) | ★★★★★ | 2–4 h | Free | Yes: Unreal Engine 5.6+ (Windows, or Apple Silicon Mac) |
| **C. Blender-native** (FaceBuilder head + MPFB2 body) | ★★★★ | 3–6 h | FaceBuilder is paid (has a trial); MPFB2 is free | Yes: Blender |

**Recommendation:** do **A tonight** to get a playable character in 15 minutes, then do **B** for the version you keep. MetaHumans have been licensed for use in **any engine or creative software** since 5.6 (metahuman.com/license), so you can export one to Blender or Godot.

---

## Route A: selfie → rigged full-body avatar (fast)

1. Open **avaturn.me** or **MetaPerson Creator** (avatarsdk.com) on your phone or PC.
2. Take the photos it asks for (shot list below; Avaturn wants front, left and right).
3. Pick a body type, clothes and hair in the editor. The body is generated, so you don't need to photograph it.
4. Export as **GLB** (or FBX). It comes rigged and is **Mixamo-compatible**.
5. Upload the FBX to **mixamo.com** (free) and download walk, run and idle animations.
6. In Blender: **File → Import → glTF/FBX**.

## Route B: MetaHuman from your iPhone 14 Pro (best quality)

The 14 Pro's front **TrueDepth** camera is exactly what this needs.

1. On your PC, install the **Epic Games Launcher** and **Unreal Engine 5.6 or newer**, and enable the **MetaHuman** plugins.
2. On your iPhone, install **Live Link Face** (free, by Epic). Select the **MetaHuman Animator** capture mode, which records TrueDepth depth and video.
3. Record a short take (see "TrueDepth capture" below) and import it into Unreal with the Capture Manager.
4. Create a **MetaHuman Identity** from the footage. Use **promote frame** to pick your neutral, front, left and right frames, then **Identity Solve** to fit the MetaHuman topology to your face.
5. Open it in **MetaHuman Creator**. Keep the face, then choose body type, height, skin, hair and clothes.
6. Export to **Blender** (FBX/glTF export, or a community DNA importer add-on). You get a full face rig with blendshapes, so your character can blink, talk and emote.

> If you'd rather not record depth, Route B also works from a regular **photogrammetry head scan**. Use **Mesh to MetaHuman**: import the scanned head mesh instead of the footage. See the photogrammetry shot list below.

## Route C: Blender-only

1. **KeenTools FaceBuilder** (Blender add-on) builds a head from about 5–15 photos (shot list below).
2. **MPFB2** (the free MakeHuman add-on for Blender) generates the body with sliders and a skeleton.
3. Join the head onto the MPFB2 body at the neck. Rig it with Rigify or upload to Mixamo.

---

## The photos

Whichever route you choose, the photos go **into the tool**, not to me. Don't send face photos to Claude or anywhere else you don't need to.

### Before you shoot (this matters more than which app)

- **Soft, even light.** Face a window on a cloudy day, or stand outside in open shade. **No flash, no overhead spotlight, no harsh sun.** Shadows get baked into the skin texture.
- **Neutral face.** Mouth closed, teeth together, eyes open, looking straight ahead. Expressions come later from the rig.
- **Hair off your forehead and ears.** Tie it back or slick it down. Hair can't be scanned well, so every route replaces it with generated hair.
- **Remove** glasses, earrings and hats. Clean-shave or keep your usual beard, but don't change it between shots.
- **Plain background** in a contrasting colour to your skin and hair.
- **No wide-angle selfie distortion.** The selfie camera held close makes noses look big. Have someone else shoot with the **rear camera at 2×**, from about 1–1.5 m away.

### Shot list for selfie avatars (Route A) and FaceBuilder (Route C): 5–9 photos

| # | Angle | Notes |
|---|---|---|
| 1 | **Front**, eye level | The most important shot. Face fills about 60% of the frame |
| 2 | ¾ left (45°) | Turn your head, not the camera |
| 3 | ¾ right (45°) | |
| 4 | Left profile (90°) | Ear visible |
| 5 | Right profile (90°) | |
| 6 | Front, slightly from **below** (chin up 15°) | Jaw and under-chin shape |
| 7 | Front, slightly from **above** | Forehead and top of head |
| 8–9 | Optional: ¾ angles from below | Help FaceBuilder fit the jaw |

### Photogrammetry head scan (Mesh to MetaHuman, KIRI, RealityScan): 60–100 photos

Use this if you want a true 3D scan rather than a fitted model. **You need a helper**, because you have to stay perfectly still.

- Sit on a chair and stare at one fixed point. Don't move your eyes or face at all.
- The helper walks a full circle around you with the **rear camera** (or KIRI Engine / RealityScan in photo mode), taking a photo every **10–15°**:
  - **Ring 1:** eye level, about 30 shots
  - **Ring 2:** about 30° above, looking down at the top of your head, about 25 shots
  - **Ring 3:** about 20° below, looking up at the jaw and under the chin, about 25 shots
  - plus 5–10 close-ups of the ears, nose and eyes
- Every photo should overlap the previous one by about 70%, be in focus, and use the same exposure.
- Process it in **KIRI Engine** (free photo scan, exports OBJ/GLB) or **RealityScan**. Then crop it to the head in Blender before Mesh to MetaHuman.

### TrueDepth capture (Live Link Face for Route B, or Heges for a quick mesh)

- Mount the phone at eye level, about 30–40 cm away (a tripod or a stack of books works).
- Use the soft-light setup above. Hair back.
- Record about 10 s: **neutral facing front (hold 2 s)** → slowly turn your head **left to about 60°** → back → **right to about 60°** → back → tilt **up**, then **down**. Keep your expression neutral the whole time. You can also add one clip of a big smile and one with your teeth showing, for the identity's teeth fit.
- Note that **Heges** (a TrueDepth scanner app) gives a raw coloured head mesh (OBJ/PLY/GLB). It's fine as a sculpting reference or for Mesh to MetaHuman, but it isn't rigged.

---

## Getting it into the game

```
Face (any route) ─▶ full-body rigged character (GLB/FBX)
                          │
                          ├─▶ mixamo.com: walk / run / idle / jump clips (free)
                          ▼
                     Blender 5.3: merge animations, check scale (1 unit = 1 m),
                     apply transforms, export GLB
                          │
                          ▼
     Our web walk mode (three.js + Spark), or Godot, loaded into the scanned room splat
     with the collision mesh
```

**Game-ready targets:** about 30–70k triangles for a third-person character (a MetaHuman can be exported at a lower LOD), textures of 2k or less, a single skeleton, and a height that matches the room scan's real-world scale.

When the viewmaxxing app reaches Phase 2 (walk mode), it will load `character.glb` and use it as the player instead of the first-person camera.
