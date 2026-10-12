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

## Response from Caelis

Thank you for the controlled comparison and the exact acceptance criteria. All three items are addressed, each with tests that follow your criteria (`tests/test_handoff.py`).

**Priority 1: the projection slider now scales projected emission.**
- Projected glow = reference glow × continuous trust (`clip(projection.weight / 0.5, 0, 1)`: square-on texels count fully, glancing ones partly, hidden or edge-on ones not at all) × `projection_strength`.
- Glow the bake kept is detected on the bake itself (colour-matched, before projection blending), so projection can't re-detect its own colours at full strength. The two are combined with `max`.
- The `> 0.5` existence check is gone. Emission is kept when any texel exceeds `EMISSIVE_EPS = 2/255`, the smallest value that survives 8-bit export.
- The stats now separate `emissive_coverage` (share of surface emitting above the export threshold) from `emissive_intensity` (mean emission where lit).
- Measured on the synthetic amber-strip fixture, strip intensity at strengths 0 / 0.01 / 0.25 / 0.85 / 1 is 0 / 0.009 / 0.19 / 0.65 / 0.76.
- Tests cover: strictly increasing intensity, absent at 0, 0.01 above the export threshold, no glow on hidden or edge-on texels, bake glow surviving with projection disabled, and the GLB keeping the emissive texture with factor [1, 1, 1].

**Priority 2: occupancy comes from the mesh's UV triangles.**
- New `materials.uv_occupancy(mesh, size)` rasterizes every UV triangle, with a 2 px dilation for seam filtering. `run.py` passes it to `build_pbr_maps(occupancy=...)`.
- The texture-only fallback no longer uses colour frequency at all. Only near-black texels count as empty.
- Projection weight is never used as occupancy.
- Tests cover:
  - the fully occupied 60% #1a1a1c / 40% grey fixture keeps its dark-steel metalness, both with occupancy and in the fallback
  - padding the same colour as a real panel stays excluded
  - hidden and edge-on islands count as surface
  - occupancy works with projection disabled and when projection is skipped for a poor silhouette, with colour matching still running

**Surface noise.**
- Derived detail is now soft-thresholded at a robust noise floor (MAD estimate), so film grain goes to exactly zero. Flat painted panels get flat (128, 128, 255) normals, and panel lines keep their relief (tested).
- New `normal_strength` setting (`--normal-strength`, UI slider *Surface detail*). Set it to 0 for a perfectly flat normal map, to separate normal-map noise from real geometry as you suggested.
- No mesh smoothing was added, so edges aren't rounded.

**Guide's twin front tubes.** The Dominion `assault_rifle` guide comes from V's officer-rifle concept, which has twin muzzles by design, so it's left as is for now. If a single front tube is wanted for standard-issue rifles, removing the lower tube from `meshify/styles/guides/dominion/weapon/assault_rifle.png` is a one-line change. Just say so.

**Your local compatibility changes** (Windows/Blackwell, fp16, offloading, prompt budget): if they're generally useful, please push them to this branch so everyone gets them and the test count lines up.

To verify: run `pytest`, then repeat your recipe above and compare against job a14cff8173f4. Expect unchanged geometry and reference; emissive intensity that follows the slider; dark-steel panels keeping their metal; and smoother normals. Rendering with `--normal-strength 0` shows the bare geometry.
