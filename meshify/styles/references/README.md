# Style reference library

Drop faction concept art into folders here and Meshify uses it automatically
for **text prompts** with that faction:

```
references/
  dominion/                 <- used for every Dominion asset type
    bronze_trim.png
    oxblood_panels.png
    weapon/                 <- used first for Dominion weapons
      service_rifle.png
      officer_rifle.png
    modular_architecture/
      corridor.png
  reformation/
    weapon/
      ...
```

Folder names are the ids from `sovereign_call.toml`: factions `dominion`,
`reformation`, `ancient`, `aurelion`, `eclipse`; asset types `small_prop`,
`medium_prop`, `industrial_prop`, `modular_architecture`, `architecture_hero`,
`weapon`, `hero_weapon`, `armor`, `ship_component`.

You don't have to touch the folders yourself. In the web UI, pick a faction
(and asset type) and use **+ Add for ...** under *Style references*. The
thumbnails show what the next job will use, and × removes one.

How it's used:
- Up to 4 images per job (set by `MESHIFY_MAX_STYLE_REFS`), asset-type images
  first, then faction-wide ones.
- SDXL looks at them through IP-Adapter in style-only mode: it borrows colours,
  materials and finish and blends several images, so the *shared* faction
  look wins. Your prompt and asset type still decide what the object is.
- One-off uploads in the UI (or `--style-image`, repeatable) replace the
  library for a single job.

Tips:
- 2-4 images per folder: one clear object or material, plain background.
- Pick images that show the faction's materials, not just shapes.
- Avoid lettering, or it can creep into generations.
- Commit this folder so the whole team shares the same faction look.
  To keep it elsewhere, set `MESHIFY_REFERENCE_DIR`.
