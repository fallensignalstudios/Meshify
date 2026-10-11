# Meshify

**Free, open-source, realism-first 3D model generation for game developers.**

Meshify is a local alternative to hosted tools like Meshy and Tripo. Give it a
prompt or a photo and it produces an Unreal-ready asset: a textured mesh with
a full PBR texture set, at real-world scale, with the pivot on the ground.
Everything runs on your own GPU. No credits, no uploads, no subscriptions.

```
prompt ──► photoreal reference ──► background removal ──► image-to-3D model
                                                              │
   Unreal-ready GLB + textures ◄── scale & pivot ◄── PBR pass ◄┘
```

## Why "realism-first"?

Most generated 3D assets look stylized or "AI-ish" for three reasons. Meshify
targets each one:

1. **Stylized input.** Image-to-3D models copy the look of their input image.
   Meshify steers text-to-image toward evenly lit, photoreal product shots
   (it backs off if you explicitly ask for a style).
2. **Baked-in lighting.** Generated textures have shadows and highlights painted
   in, which look wrong under your game's lights. Meshify *delights* the albedo.
3. **No material response.** A single colour texture can't react to light.
   Meshify derives normal, roughness and AO maps and packs an Unreal-standard
   ORM texture.

It also exports at **real-world scale** (set a height in metres), because an
asset at the wrong size never looks real.

## Models

Meshify wraps open models behind one interface, so new ones can be dropped in
as the field moves.

| Backend | License | Commercial use | Notes |
|---|---|---|---|
| `trellis` (Microsoft TRELLIS) | MIT | ✅ | Default. Clean geometry. |
| `hunyuan3d` (Tencent Hunyuan3D 2) | Tencent community license | ⚠️ restricted | Excellent detail. Excludes EU/UK/South Korea, limits large-scale use. |
| `dummy` | MIT | ✅ | CPU test placeholder (no GPU needed). |

Text prompts use **FLUX.1-schnell** (Apache-2.0) by default, or **RealVisXL**
(SDXL-based, OpenRAIL++) on 12 GB cards. Any diffusers text-to-image model
can be swapped in.

Meshify's own code is MIT. The assets you generate are yours, subject to the
license of the backend that made them.

## Quick start

You need an NVIDIA GPU: 16 GB+ for everything, or 12 GB in the automatic
low-VRAM mode. See **[docs/INSTALL.md](docs/INSTALL.md)** for the full setup,
including Windows via WSL2 and RTX 50-series cards.

```bash
meshify backends                     # what's installed, with licenses
meshify serve                        # web UI at http://127.0.0.1:7860

meshify generate "weathered oak barrel with iron bands" --name Barrel --size 0.9
meshify generate --image photo.jpg --name Lantern --size 0.35 --faces 20000
```

Each job writes a folder in `outputs/`:

```
SM_Barrel.glb                 # drag into Unreal
SM_Barrel.obj
textures/T_Barrel_BaseColor.png
textures/T_Barrel_Normal_DX.png   # Unreal / DirectX
textures/T_Barrel_Normal_GL.png   # Blender / glTF
textures/T_Barrel_ORM.png         # R=AO, G=Roughness, B=Metallic
textures/T_Barrel_Roughness.png, T_Barrel_AO.png, T_Barrel_Metallic.png
textures/T_Barrel_Emissive.png, T_Barrel_EmissiveMask.png   # when glow is found
reference.png, cutout.png, meshify.json
```

Then see **[docs/UNREAL.md](docs/UNREAL.md)** for importing.

## Style packs: faction × asset type × condition

Meshify ships with a **Sovereign Call** style pack (Dominion, Reformation,
Ancient Sovereign, Aurelion Enforcer, Eclipse × 9 asset types × 4 conditions).
It sets the prompt's design language, faction-correct metallic and emissive
materials, and per-asset-type triangle budgets, sizes and pivots:

```bash
meshify generate "heavy assault rifle" --faction dominion --type weapon --condition frontier
```

It's all one editable TOML file. See **[docs/STYLES.md](docs/STYLES.md)**.

## Options

| Option | Default | |
|---|---|---|
| `--backend` | `trellis` | Which 3D model to use |
| `--faction` / `--type` / `--condition` | none | Style preset (see `meshify styles`) |
| `--size` | preset / auto | Real-world size in metres (height, or length for weapons) |
| `--faces` | preset / 50000 | Triangle budget, reduced *before* texture baking. `0` = full density for Nanite |
| `--texture-size` | preset / 2048 | 512 / 1024 / 2048 / 4096 |
| `--delight` | 0.6 | 0 keeps baked lighting, 1 flattens it fully |
| `--no-realism` | off | Skip the photoreal prompt and PBR pass |
| `--seed` | 42 | Same seed + same input = same result |

## Project layout

```
meshify/
  backends/       # one file per 3D model; add yours by subclassing ImageTo3DBackend
  pipeline/
    images.py     # text-to-image, realism prompt, background removal, framing
    materials.py  # delighting, normal/roughness/AO, ORM packing
    mesh.py       # floater removal, decimation, scale, grid snap and pivot
  styles/         # style packs (sovereign_call.toml)
    run.py        # the end-to-end pipeline
  server.py       # local web API + job queue
  web/            # browser UI (offline, with a vendored 3D viewer)
  cli.py
```

## Roadmap

- [ ] Learned PBR materials (replace the image-space colour heuristics)
- [ ] Multi-view input (front/side/back photos) for better geometry
- [ ] Automatic retopology and LOD generation
- [ ] FBX export
- [ ] Unreal Editor plugin: generate and import without leaving the editor
- [ ] Batch generation from a prompt list

## Development

```bash
pip install -e ".[dev]"
pytest
```
