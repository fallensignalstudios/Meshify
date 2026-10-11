"""Style packs: faction x asset type x condition presets loaded from TOML files.

Built-in packs live next to this file. Add your own by pointing
MESHIFY_STYLE_DIR at a folder of *.toml files (same format).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # TRELLIS's default environment is Python 3.10
    import tomli as tomllib

BUILTIN_DIR = Path(__file__).parent


@dataclass
class MaterialStyle:
    """What the PBR pass needs to know about a faction + condition."""

    metals: list[tuple[int, int, int]] = field(default_factory=list)
    emissive: tuple[int, int, int] | None = None
    roughness_bias: float = 0.0
    ao_strength: float = 1.0


@dataclass
class ResolvedStyle:
    prompt_parts: list[str]  # what it is: faction look
    look_parts: list[str]  # how worn it is: condition
    negative: str  # everything this style avoids (pack + asset type + faction)
    render: str | None
    material: MaterialStyle
    tris: int | None = None
    texture: int | None = None
    size_m: float | None = None
    size_axis: str = "height"
    pivot: str = "bottom_center"
    grid_m: float | None = None
    identity_negative: str = ""  # asset type + faction only (SDXL encoder 1)
    pack_negative: str = ""  # pack-wide (no text, no cartoon...) (SDXL encoder 2)
    framing: str | None = None  # asset-type template, "{subject}" = the user's prompt
    aspect: str = "square"  # text-to-image frame: square, wide or tall

    def subject(self, prompt: str) -> str:
        """The user's prompt wrapped in the asset-type framing (framing first)."""
        prompt = prompt.strip().rstrip(".")
        if not self.framing:
            return prompt
        if "{subject}" in self.framing:
            return self.framing.replace("{subject}", prompt)
        return f"{prompt}, {self.framing}"

    def apply_to_prompt(self, prompt: str) -> str:
        parts = [self.subject(prompt), *self.prompt_parts, *self.look_parts]
        return ", ".join(p for p in parts if p)


def _hex(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Colours must be #rrggbb, got {color!r}")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _style_dirs() -> list[Path]:
    dirs = [BUILTIN_DIR]
    if extra := os.environ.get("MESHIFY_STYLE_DIR"):
        dirs.append(Path(extra))
    return dirs


@lru_cache(maxsize=1)
def load_packs() -> dict[str, dict]:
    packs = {}
    for folder in _style_dirs():
        for path in sorted(folder.glob("*.toml")):
            with path.open("rb") as fh:
                data = tomllib.load(fh)
            pack_id = data.get("pack", {}).get("id", path.stem)
            for section in ("factions", "categories", "conditions"):
                data.setdefault(section, {})
            packs[pack_id] = data
    return packs


def list_packs() -> list[dict]:
    """Labels and defaults for the UI."""
    out = []
    for pack_id, data in load_packs().items():
        out.append({
            "id": pack_id,
            "name": data.get("pack", {}).get("name", pack_id),
            "factions": {k: v.get("label", k) for k, v in data["factions"].items()},
            "conditions": {k: v.get("label", k) for k, v in data["conditions"].items()},
            "categories": {
                k: {"label": v.get("label", k), "tris": v.get("tris"), "texture": v.get("texture"),
                    "size_m": v.get("size_m"), "size_axis": v.get("size_axis", "height")}
                for k, v in data["categories"].items()
            },
        })
    return out


def resolve(pack: str | None, faction: str | None, category: str | None,
            condition: str | None) -> ResolvedStyle | None:
    if not (pack or faction or category or condition):
        return None
    packs = load_packs()
    if not pack:
        if len(packs) != 1:
            raise ValueError("Several style packs are installed; pick one with the pack setting.")
        pack = next(iter(packs))
    if pack not in packs:
        raise ValueError(f"Unknown style pack {pack!r}. Available: {', '.join(packs) or 'none'}")
    data = packs[pack]

    def pick(section: str, key: str | None) -> dict:
        if not key:
            return {}
        if key not in data[section]:
            raise ValueError(f"Unknown {section[:-1] if section != 'categories' else 'asset type'} "
                             f"{key!r}. Choose from: {', '.join(data[section])}")
        return data[section][key]

    fac, cat, cond = pick("factions", faction), pick("categories", category), pick("conditions", condition)
    # A faction can speak differently per asset type (e.g. Dominion weapons).
    override = fac.get("overrides", {}).get(category, {}) if category else {}
    faction_prompt = override.get("prompt", fac.get("prompt"))
    info = data.get("pack", {})
    negative = ", ".join(n for n in [info.get("negative"), cat.get("negative"),
                                     fac.get("negative"), override.get("negative")] if n)

    material = MaterialStyle(
        metals=[_hex(c) for c in fac.get("metals", [])],
        emissive=_hex(fac["emissive"]) if fac.get("emissive") else None,
        roughness_bias=fac.get("roughness_bias", 0.0) + cond.get("roughness_offset", 0.0),
        ao_strength=cond.get("ao_strength", 1.0),
    )
    return ResolvedStyle(
        # Order matters for 77-token CLIP models: faction look first, then framing.
        prompt_parts=[faction_prompt] if faction_prompt else [],
        look_parts=[cond["prompt"]] if cond.get("prompt") else [],
        negative=negative,
        identity_negative=", ".join(n for n in [cat.get("negative"), fac.get("negative"),
                                                override.get("negative")] if n),
        pack_negative=info.get("negative", ""),
        render=info.get("render"),
        material=material,
        tris=cat.get("tris"),
        texture=cat.get("texture"),
        size_m=cat.get("size_m"),
        size_axis=cat.get("size_axis", "height"),
        pivot=cat.get("pivot", "bottom_center"),
        grid_m=cat.get("grid_m"),
        framing=cat.get("prompt"),
        aspect=cat.get("aspect", "square"),
    )
