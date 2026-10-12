# Silhouette guides

A guide is a **black silhouette on white** (or clean dark line art) of an
archetype: an assault rifle in side view, a sidearm, a blade. For text
prompts, SDXL follows its outline through an edge ControlNet, so the
reference gets one scope, a sane stock and the right length. The prompt and
style references still decide materials and detail.

```
guides/
  dominion/weapon/assault_rifle.png    <- faction-specific (checked first)
  dominion/hero_weapon/ceremonial_blade.png
  weapon/sidearm.png                   <- any faction (optional)
```

**Auto** matches file names to the prompt, with synonyms: "pistol"/"handgun"
-> `sidearm`, "sniper" -> `marksman`, "cannon"/"minigun"/"rotary" -> `heavy`,
"sword"/"greatsword"/"dagger" -> `blade`. If nothing matches, no guide is used.

The bundled guides were extracted from the faction style boards' weapons.

Making your own:
- One object, side view, filled black on white. A quick blockout render or a
  paint-bucket fill over concept art works well.
- Name it after the archetype (`marksman_rifle.png`, `heavy_cannon.png`).
- Upright guides (blades) are laid on their side automatically in wide frames.
