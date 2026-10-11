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
| **Asset type** | Framing ("side view, full length" for weapons...) | Texture size | Triangle budget, real-world size, size axis, pivot, grid snapping |
| **Condition** | Wear words (polished … torn open) | Roughness offset, AO strength | — |

Anything you set explicitly (texture, triangle budget, size) overrides the preset.

## How the faction materials work

- **Metallic.** Texture pixels close in colour to one of the faction's `metals`
  swatches become metallic. Dominion bronze and blackened steel turn into real
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

These are colour heuristics, so they work best when the generated texture
actually uses the faction palette. A faction prompt helps a lot with that.

## Editing or adding a faction

```toml
[factions.my_faction]
label = "My Faction"
prompt = "short design-language words, most important first"
negative = "things this faction never looks like"
metals = ["#8c6239", "#2b2b2e"]   # colours that should read as metal
emissive = "#ffaa33"               # energy/glow colour
roughness_bias = -0.05             # negative = glossier overall
```

Keep `prompt` short. On 12 GB cards the text-to-image model (SDXL) reads about
75 tokens per encoder. Meshify sends subject + faction + asset type to one
encoder and condition + framing to the other, but it's still worth staying tight.

Changes are picked up on the next `meshify serve` / `meshify generate`.

## Your own packs

Put more `*.toml` files in a folder and point `MESHIFY_STYLE_DIR` at it. If
more than one pack is installed, choose one with `--pack <id>`.
