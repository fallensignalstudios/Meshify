# Style reference images

Drop faction concept art here and point the style pack at it, e.g. in
`sovereign_call.toml`:

```toml
[factions.dominion]
reference = "references/dominion.png"

[factions.dominion.overrides.weapon]
reference = "references/dominion_weapon.png"
```

When you generate from a **text prompt**, SDXL looks at the reference through
IP-Adapter and borrows its colours and materials (bronze trim, oxblood enamel,
amber glow...) while your prompt and the asset type still decide what the
object is. Only the style is borrowed, not the exact shape.

Tips:
- One clear object on a plain background works best.
- Pick art with the faction's materials, not just its shapes.
- Avoid lettering in references, or it may show up in generations.
- In the web UI, uploading a "Style reference" overrides the pack's reference
  for that job. "Style strength" sets how strongly it applies (0.6 default).
