# Reference pack prompts: building_001 (Sunray Mart)

Use **ChatGPT** (or Gemini, or Qwen-Image-Edit). Do the steps **in order, in one chat**, and always attach the previous image you liked, so every view shows the *same* building. Save the images as the filenames shown, in `forge/refs/building_001/`.

Rules for every prompt:
- Original names only: no real brands, no Rockstar or GTA logos.
- From step 2 on, **flat even overcast light, no cast shadows**. Those images become textures.
- If a result drifts (a different building, wrong floor count), re-roll it. Don't continue from a bad image.

---

### 1. Hero shot → `01_hero.png`
> Photorealistic next-gen open-world game screenshot. A two-storey streamline-moderne art deco corner store in a fictional South Florida coastal city. Faded flamingo-pink stucco walls with off-white horizontal speed-line trim and mint-green accents. Rounded front-left corner, full-width aluminium storefront glass, recessed entry, striped awning, and red-pink and cyan neon channel letters on the parapet reading "SUNRAY MART". Upper floor has four windows with sea-foam teal hurricane shutters and a small balcony over the entrance. A lightbox reading "OPEN 24 HRS" on the corner, window neon "LIQUOR · LOTTO · ICE". Sun-bleached paint, salt grime, rust streaks under wall AC units, posters and tags on the side wall, palms, newspaper boxes, an ice freezer chest, cracked sidewalk. Late-afternoon light, 3/4 view from across the street at eye level, 24 mm lens, 16:9.

### 2. Front elevation → `02_elev_front.png` (attach 01)
> Same building as the attached image. Exact **orthographic front elevation**: camera perfectly straight on, no perspective, no vanishing points. Flat even overcast daylight, **no cast shadows**, no people, no cars, no trees in front. The whole facade fills the frame edge to edge, from the ground line to the top of the parapet, on a plain light-grey background. Keep every detail: signs, windows, shutters, awning, grime, posters. Architectural reference photo, ultra sharp, 16:9.

### 3–5. Other elevations → `03_elev_left.png`, `04_elev_right.png`, `05_elev_back.png` (attach 01 + 02)
> Same building as the attached images. Exact orthographic **[LEFT side with the rounded corner | RIGHT side party wall | BACK service side]** elevation, same flat overcast light, no shadows, no perspective, facade fills the frame on plain light-grey background.
> - **Left:** storefront glass wraps the rounded corner for 3 m, then a plain wall with a faded mural, a side door to the stairs, and 3 upper windows with teal shutters.
> - **Right:** mostly blank stucco party wall, layered posters, graffiti tags, electrical meter box, conduit and pipes, water stains.
> - **Back:** steel back door, dumpster bay, two AC condensers, an external metal stair to the apartment, small barred windows, drain pipes, heavy grime.

### 6. Roof → `06_roof.png` (attach 01)
> Same building, exact top-down orthographic roof plan. Flat roof with a low parapet, grey weathered membrane, two AC condensers, a water heater tank, a satellite dish, vents, puddle stains, and the back of the neon sign. Flat overcast light, no shadows.

### 7. Detail sheet → `07_details.png` (attach 01 + 02)
> Reference sheet of close-up details from the attached building, arranged in a 3×3 grid on plain light-grey: storefront window frame, recessed entry door, upper window with teal shutters, speed-line cornice profile, rounded-corner glass block, balcony railing, striped awning, wall AC unit, neon letter "S" (tube detail). Each straight-on, flat even light, no shadows, crisp.

### 8. Material swatches → `08_swatch_<name>.png` (one per material, square)
> Seamless tileable texture, perfectly top-down/straight-on, flat even light, no shadows, no perspective, fills the square: **[faded flamingo-pink stucco with hairline cracks and sun bleaching | off-white painted concrete trim | terrazzo floor with pink and grey chips | weathered grey roof membrane | cracked concrete sidewalk | sea-foam teal painted aluminium shutters]**.

### 9. Decal sheet → `09_decals.png`
> Sprite sheet of 12 separate grunge decals for a game, each isolated with lots of empty space, on **pure flat white**: rust streak, water stain drip, black mould patch at a wall base, crack network, chewing-gum spots, faded torn poster (fictional band "NEON LAGOON"), torn lottery poster, two graffiti tags (fictional crew "PALM KNGZ"), bird droppings, oil stain, footprint scuffs. Flat lighting, no shadows, high detail.
> (With **Qwen-Image-Layered** you can ask for transparent layers instead. That's even better.)

---

**Check before moving on:** windows line up between the hero and the elevations, the floor count is 2, the corner is on the left, the sign text is spelled right, and there are no shadows on 02–08. These images feed stage ③ (3D proxy) and stage ④ (parse) in `docs/MLO_FORGE.md`.
