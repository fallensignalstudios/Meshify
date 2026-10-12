# Caelis handoff: Dominion rifle materials

Tested upstream commit: [9195b0a](https://github.com/fallensignalstudios/Meshify/commit/9195b0a77481bb983deec6b99d64a35b4557c496).
Date: 2026-10-11. Context: [PR #1](https://github.com/fallensignalstudios/Meshify/pull/1).
These are verified results and requested follow-up fixes from the owner's Windows GPU test session.

## What is working

Reference projection now preserves the bright amber handguard channel and clearer oxblood panels. The exported GLB embeds the 2048 x 2048 emissive PNG pixel-for-pixel, with emissive factor [1, 1, 1]. The glow is visibly present in Meshify's viewer. Preserve this behavior.

The latest run completed in 206.3 seconds with 20,000 triangles, 12,724 vertices, 2048 textures, a 1 m longest dimension, and a centered pivot. Finite geometry, framing, normal maps, and metallic/ORM blue-channel consistency passed. The locally adapted checkout passed all 131 tests; its Windows/Blackwell, fp16, model-offloading, and prompt-budget accommodations were retained for the run.

## Controlled comparison and recipe

Use these settings when checking the next change:

| Setting | Value |
|---|---|
| Prompt | heavy assault rifle |
| Backend | hunyuan3d; mini shape + real texture painting |
| Pack / faction / category / condition | sovereign_call / dominion / weapon / maintained |
| Starting seed / candidates | 42 / 3; seeds 42, 43, 44 |
| Style references | Automatic Dominion weapon library; four images |
| Style strength / guide strength | 0.6 / 0.6 |
| Guide | auto; assault_rifle |
| Realism / delight strength | true / 0.6 |
| Reference projection strength | 0.85 |
| Triangle / texture / size / pivot | Category defaults: 20,000 / 2048 / 1 m longest / center |

Machine: native Windows, RTX 5070 12 GB, PyTorch 2.7.1+cu128, RealVisXL fp16 with IP-Adapter and SDXL Canny ControlNet, Hunyuan3D mini with low-memory offloading.

Previous job: 9107cb31aa88 (01da12d). Latest job: a14cff8173f4 (9195b0a). Both selected seed 43. All three candidate images, the reference, cutout, guide, mesh vertices/faces/UVs, and bounds are identical across these runs. This isolates the material changes.

| Area-weighted surface measure | Previous | Latest |
|---|---:|---:|
| Metallic value > 0.5 | 72.42% | 78.83% |
| Mean roughness | 0.4879 | 0.4763 |
| Mean sRGB luminance | 0.10727 | 0.14737 |
| Emissive-mask value > 0.5 | 0% | 1.780% |

Method: 300,000 uniformly distributed surface samples, triangles selected proportional to area, seed 42, nearest UV texel sampling. Luminance is measured in sRGB channel space, not linear light. These are map measurements, not a ground-truth material classification.

Application metadata reports projection IoU 0.84, trusted projection atlas coverage 45.7%, metallic coverage 77.2%, and glow coverage 1.17%. Its coverage denominator is a heuristic texture mask; do not equate it with surface area. Rasterized occupied UV coverage is about 60.17%, while the inferred texture mask covers 91.3% of the atlas.

## Priority 1: make the projection slider scale projected emission

Location: `meshify/pipeline/materials.py`, projected-glow block in `build_pbr_maps`.

RGB blending uses `projection.weight * projection_strength`. Glow currently uses only `projection.sample(ref_glow) * (projection.weight > 0.3)`. Every nonzero projection strength therefore transfers the same full-strength glow mask. A controlled synthetic amber-strip check at strengths 0.01 and 0.85 produced identical masks; strength 0 disabled projection.

Requested fix:

- Scale the projected emission contribution by the requested strength, preferably using the continuous projection weight while retaining the visibility/trust gate.
- Preserve glow detected independently in the baked texture.
- Replace the later `(emissive_mask > 0.5).any()` existence check as needed. A scaled contribution must not disappear merely because its intensity is below 0.5. Use a small nonzero threshold appropriate for 8-bit export.
- Keep detection/coverage statistics distinct from emission intensity so the UI does not imply that a dim but valid glow is absent.

Regression acceptance:

1. Use a dull bake with no independently detected glow and a reference with a known amber strip. Compare the projected emission at strengths 0, 0.01, 0.25, 0.85, and 1.
2. The transferred contribution is absent at 0 and increases monotonically in pixel intensity with strength. Low strengths remain represented when they are above the chosen export threshold. Check intensity, not only coverage above 0.5.
3. Existing bake glow survives with projection disabled. Hidden and edge-on texels receive no transferred glow.
4. The exported GLB retains the emissive image and factor. Default strength 0.85 still visibly preserves the rifle's amber channel.

## Priority 2: stop treating uniform real panels as UV padding

Location: `texture_mask` in `meshify/pipeline/materials.py`.

The heuristic removes a dominant exact neutral color as atlas fill. A synthetic texture containing 60% Dominion dark steel (#1a1a1c) and 40% gray, all valid model surface, retained only 40%. The dark panel's metallic classifier was 1.0 before masking and 0 in the exported metallic map.

Requested fix:

- Prefer occupancy rasterized from all mesh UV triangles when a mesh is available; retain a conservative fallback for texture-only callers.
- Do not use projection weight as occupancy. A zero weight also describes valid hidden or edge-on surfaces.
- Exclude actual unused atlas space from brightness/blur statistics without excluding legitimate flat neutral material solely because it is common.

Regression acceptance: cover the fully occupied dark-panel fixture, genuine neutral/black padding around UV islands (including padding identical in color to a real panel), and valid hidden/edge-on UV islands. Repeat with projection disabled and with projection skipped for a poor silhouette match. Occupied panel metalness and color correction must survive; actual padding must stay excluded.

## Lower-priority visual cleanup and evidence

The projected rifle still has rough surface detail and irregular edges for Maintained. Compare the same fixed reference before changing prompts, and render once with the derived normal map disabled to distinguish normal-map noise from actual geometry. Smooth color noise and derived bump separately; avoid blanket mesh smoothing that rounds the rifle's edges. Flat painted panels should have approximately (128, 128, 255) tangent-space normals. Keep the current glow, faction colors, scale, pivot, and UV integrity while refining this.

The guide contains two forward tube-like protrusions. Review whether the lower one is intentional; changing the guide is the appropriate first step if a single front tube is desired.

After the fixes, run pytest and repeat the recipe above. Record projection activation, emission pixels and GLB embedding, finite geometry, normal/ORM packing, and scale/pivot. Inspect both sides and UV seams. The full neural pipeline and local preview have been tested; Unreal import remains unverified.

Detailed comparison JSON, job metadata, reference/candidate images, and the 47,004,684-byte latest asset bundle are retained on the owner's test machine. Local paths, logs, model files, and compatibility commits are not part of this documentation handoff.
