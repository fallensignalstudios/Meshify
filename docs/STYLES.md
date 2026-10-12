# Style packs (faction × asset type × condition)

A style pack is a TOML file that teaches Meshify what your world looks like.
The built-in one is `meshify/styles/sovereign_call.toml`.

The three settings are **independent**, so any combination works:

```bash
meshify generate "heavy assault rifle" -f dominion -t weapon -c frontier --name DomRifle
meshify generate "sliding security door" -f reformation -t modular_architecture -c pristine
meshify generate "corrupted command console" -f eclipse -t industrial_prop -c derelict
meshify styles     # list everything in the installed packs
```

In the web UI, they're the **Faction / Asset type / Condition** dropdowns.

## What each setting controls

| | Prompt | Materials | Mesh |
|---|---|---|---|
| **Faction** | Design language and palette words, plus things to avoid | Metal swatches (→ metallic map), energy colour (→ emissive mask), roughness bias | — |
| **Asset type** | Framing template + aspect ratio (wide for weapons) | Texture size | Triangle budget, real-world size, size axis, pivot, grid snapping |
| **Condition** | Wear words (polished … torn open) | Roughness offset, AO strength | — |

Anything you set explicitly (texture, triangle budget, size) overrides the preset.

## Framing: whole objects, not close-ups

Text-to-image models love zooming into a cool detail, and a cropped reference
gives a partial 3D model. Three things guard against that:

- **Framing first.** Each asset type's `prompt` is a template, and `{subject}`
  is replaced by what you typed. For example, weapons become "full-length side
  profile of an entire heavy assault rifle, whole weapon from end to end...". Words
  at the start of a prompt carry the most weight.
- **Aspect ratio.** `aspect = "wide"` renders 1344×768 for long things
  (weapons, ship parts), `"tall"` is 768×1344, and the default is square.
- **Crop check.** After background removal, if the object touches the edge of
  the frame, Meshify retries with the next seed (up to 3 tries, set by
  `MESHIFY_REFERENCE_ATTEMPTS`) and keeps the least-cropped reference.

## How the faction materials work

- **Metallic.** Texture pixels close in colour to one of the faction's `metals`
  swatches become metallic. If a faction lists `nonmetals` (enamel, leather,
  paint), each pixel goes to whichever family it's *closer* to. Without that,
  dark oxblood enamel reads as "near" blackened steel. Dominion bronze and blackened steel turn into real
  metal in Unreal, while red cloth or black stone stays non-metal. Eclipse has
  no metals: corruption is organic.
- **Emissive.** Pixels near the faction's `emissive` colour that are also
  *brighter than their surroundings* (thin glowing channels, not flat panels)
  go into `T_<name>_EmissiveMask.png`, and the glow colour into
  `T_<name>_Emissive.png`. Multiply the mask by an intensity parameter in your
  material and the glow is fully controllable. If no glow is found, or the
  "glow" would cover more than 20% of the texture (probably just a bright
  surface), no emissive map is written.
- **Condition** shifts roughness (pristine is glossier, derelict rougher) and
  scales ambient occlusion (more grime in the crevices).

The Sovereign Call colours are sampled from the faction style boards and
reference art, and tests check them on real art: Dominion bronze is metal,
oxblood is enamel and the amber strips glow; Reformation and Ancient white or
ivory armour doesn't glow; Eclipse's violet glows are found (glow brightness
uses the brightest channel, so vivid violet counts as bright). Non-metal
swatches also veto glow, which stops edge highlights on pale armour being
read as light.

These are colour heuristics, so they work best when the generated texture
actually uses the faction palette. A faction prompt helps a lot with that.

## Editing or adding a faction

```toml
[factions.my_faction]
label = "My Faction"
prompt = "short design-language words, most important first"
negative = "things this faction never looks like"
metals = ["#8c6239", "#2b2b2e"]   # colours that should read as metal
nonmetals = ["#4c281d"]            # look-alikes that must NOT (enamel, leather, paint)
emissive = ["#ffaa33", "#ffe08a"]  # glow colours: include the white-hot core
roughness_bias = -0.05             # negative = glossier overall
```

Keep `prompt` short. On 12 GB cards the text-to-image model (SDXL) reads about
75 tokens per encoder. Meshify sends subject + faction + asset type to one
encoder and condition + framing to the other, but it's still worth staying tight.

## Style reference library (the strongest lever)

Words only get you so far: "bronze trim outlining every panel" means many
things to a model. A **style reference** shows it. On SDXL (the default on
12 GB cards), Meshify feeds the references through IP-Adapter in *style-only*
mode (InstantStyle): the generated image borrows their colours, materials and
finish, while your prompt and the asset type still decide what the object is.

References are stored per faction and asset type, and **picked up
automatically**:

```
meshify/styles/references/dominion/          <- every Dominion asset
meshify/styles/references/dominion/weapon/   <- Dominion weapons (used first)
```

- **Easiest:** in the web UI, choose a faction and asset type, then use
  **+ Add for ...** under *Style references*. Thumbnails show exactly what the
  next job will use; × removes one.
- Up to 4 images per job, most specific first. Several images are blended, so
  the faction's shared look wins over one image's quirks.
- **Style strength** (0-1, default 0.6): raise it if the colours don't come
  through, lower it if results copy too much.
- **One-off references** (UI upload, or `--style-image`, repeatable) replace the
  library for a single job.
- Commit the folder so the team shares one faction look, or keep it elsewhere
  with `MESHIFY_REFERENCE_DIR`. `meshify styles` shows how many references each
  faction/type has. More in `meshify/styles/references/README.md`.

The first use downloads the IP-Adapter weights (~1.5 GB, Apache-2.0). Style
references apply to text prompts. When you upload a reference image *as the
input*, it already defines the look.

## Structure rules and candidates (one scope, sane proportions)

Text-to-image is a dice roll: one seed gives a clean rifle, the next two
scopes and a stock the size of a door. Two layers handle that:

1. **Structure rules in the prompt.** An asset type can carry `structure`
   (added to the prompt, e.g. "one integrated optic, single barrel,
   believable proportions") and failure modes in its `negative` ("two scopes,
   duplicate parts, extra barrels, malformed").
2. **Candidates + judge.** For text prompts Meshify generates several
   references (default 3; UI *Candidates*, `--candidates`, or
   `MESHIFY_CANDIDATES`) and a CLIP judge (`openai/clip-vit-large-patch14`, MIT,
   ~0.9 GB on first use) scores each against the asset type's `good` and `bad`
   descriptions:

   ```toml
   [categories.weapon]
   structure = "one integrated optic, single barrel, believable proportions"
   good = ["a single well-proportioned {subject} with one scope"]
   bad = ["a gun with two scopes", "a weapon with duplicated parts", "a malformed disproportionate weapon"]
   ```

   The pack's `[pack]` section holds defaults for every type (single complete
   object, not a cropped close-up, no text). Cropped candidates are never
   chosen. All candidates are saved in `candidates/`, and the job list shows
   their scores with a ★ on the winner, so you can pick another by hand.

CLIP is great at "one clean, complete object" and decent at obvious
duplicates, but it isn't a precise counter, which is why the prompt carries
the same rules. More candidates mean better odds but proportionally more time
(SDXL takes a few seconds per image on an RTX 5070).

## Silhouette guides (guaranteed proportions)

The judge picks the best candidate. A guide makes every candidate follow the
right outline in the first place. For text prompts on SDXL, a silhouette of
the archetype becomes an edge map for a canny ControlNet
(`diffusers/controlnet-canny-sdxl-1.0`, downloaded on first use; override with
`MESHIFY_CONTROLNET`).

- Guides live in `meshify/styles/guides/<faction>/<asset type>/` (bundled:
  rifles, sidearms, heavy weapons and blades from the faction boards). See
  that folder's README.
- **Auto** (default) picks the guide whose name matches the prompt ("heavy
  assault rifle" -> `assault_rifle`, "pistol" -> `sidearm`). No match, no guide.
- In the UI, the *Silhouette guide* panel shows which one will be used. Pick
  another, choose None, or upload a one-off silhouette. CLI: `--guide auto|none|<name>|path.png`.
- **Guide strength** (0-1, default 0.6): higher follows the outline more
  strictly. The guide lets go for the last 20% of steps so details can form.

Guides combine with style references (materials) and the judge (best of N):
the outline comes from the guide, the look from the references.

### Variants (a sniper keeps its scope, a sword has no barrel)

An asset type can have variants chosen by prompt words. The first match
replaces the type's `structure`, `negative`, `good` and `bad`:

```toml
[categories.weapon.variants.marksman]
match = ["sniper", "marksman", "dmr", "precision"]
structure = "one precision scope, long single barrel, believable proportions"
```

Weapons default to a compact holographic sight (and negate "sniper scope").
`marksman` keeps a scope, `sidearm` is a compact handgun with no scope or
stock, and `melee` (blade, sword, axe, spear, staff...) drops gun parts
entirely. Guides follow the same split: `assault_rifle` has a compact sight,
`marksman_rifle` the scope.

### Faction accents (colours that must survive)

`accents` are the faction colours that tend to get lost (Dominion oxblood
enamel and amber light strips). They lead SDXL's *second* encoder, while the
faction prompt sits on the first, so they're stated twice. That's the
strongest emphasis plain prompts allow. Dominion also negates "all black,
monochrome, unpainted". For even more colour, raise *Style strength*.

### Per-asset-type faction language (overrides)

Generic faction words lose to strong nouns: "Dominion... assault rifle" still
draws a real-world AK, because that's what "assault rifle" means in photos. So
a faction can describe itself differently per asset type:

```toml
[factions.dominion.overrides.weapon]
prompt = "Dominion imperial state-issued service weapon, monolithic armored chassis, ..."
negative = "neon, cyberpunk, fantasy filigree, gold plated"
```

The override's `prompt` replaces the faction prompt for that asset type, and
its `negative` is added. Weapon asset types also carry their own negatives
(`AK-47, AR-15, wood furniture, picatinny rails...`) for every faction.

### Token budget

SDXL reads 75 tokens per encoder. Meshify splits both the prompt and the
negative across its two encoders:

| | Encoder 1 | Encoder 2 |
|---|---|---|
| Prompt | framing + your subject + faction | condition + render style + studio framing |
| Negative | asset type + faction "is not" terms | pack-wide, quality and framing terms |

A test (`test_every_styled_prompt_fits_sdxl_token_limits`) checks every
faction × type × condition against the limit, so run `pytest` after editing.
Meshify also logs a warning at generation time if a prompt half may be cut off.

Changes are picked up on the next `meshify serve` / `meshify generate`.

## Your own packs

Put more `*.toml` files in a folder and point `MESHIFY_STYLE_DIR` at it. If
more than one pack is installed, choose one with `--pack <id>`.
