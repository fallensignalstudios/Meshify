"""Style packs: faction x asset type x condition presets loaded from TOML files.

Built-in packs live next to this file. Add your own by pointing
MESHIFY_STYLE_DIR at a folder of *.toml files (same format).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # TRELLIS's default environment is Python 3.10
    import tomli as tomllib

from ..pipeline.guides import list_guides  # noqa: E402

BUILTIN_DIR = Path(__file__).parent
log = logging.getLogger(__name__)


@dataclass
class MaterialStyle:
    """What the PBR pass needs to know about a faction + condition."""

    metals: list[tuple[int, int, int]] = field(default_factory=list)
    nonmetals: list[tuple[int, int, int]] = field(default_factory=list)  # paint, enamel, leather
    emissive: list[tuple[int, int, int]] = field(default_factory=list)
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
    references: list[Path] = field(default_factory=list)  # style images, most specific first
    guides: list[Path] = field(default_factory=list)  # silhouette guides for this faction + type
    good: list[str] = field(default_factory=list)  # judge: what a good candidate looks like
    bad: list[str] = field(default_factory=list)  # judge: failure modes to avoid

    def subject(self, prompt: str) -> str:
        """The user's prompt wrapped in the asset-type framing (framing first)."""
        prompt = prompt.strip().rstrip(".")
        if not self.framing:
            return prompt
        if "{subject}" in self.framing:
            return self.framing.replace("{subject}", prompt)
        return f"{prompt}, {self.framing}"

    def checks(self, prompt: str) -> tuple[list[str], list[str]]:
        """Judge texts with {subject} filled in."""
        subject = prompt.strip().rstrip(".") or "object"
        fill = lambda texts: [t.replace("{subject}", subject) for t in texts]
        return fill(self.good), fill(self.bad)

    def apply_to_prompt(self, prompt: str) -> str:
        parts = [self.subject(prompt), *self.prompt_parts, *self.look_parts]
        return ", ".join(p for p in parts if p)


def _hex(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Colours must be #rrggbb, got {color!r}")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_STYLE_REFERENCES = int(os.environ.get("MESHIFY_MAX_STYLE_REFS", "4"))


def _pack_references(data: dict, rel) -> list[Path]:
    """`reference = "x.png"` or a list, relative to the pack file. Missing files
    are skipped (with a warning) so a pack still works before its art exists."""
    out = []
    for item in [rel] if isinstance(rel, str) else (rel or []):
        path = (data["_dir"] / item).resolve()
        if path.is_file():
            out.append(path)
        else:
            log.warning("Style reference %s not found; continuing without it.", path)
    return out


def reference_root(data: dict) -> Path:
    """The reference library: <root>/<faction>/<asset type>/ and <root>/<faction>/."""
    if env := os.environ.get("MESHIFY_REFERENCE_DIR"):
        return Path(env)
    return data["_dir"] / "references"


def _images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def library_references(data: dict, faction: str | None, category: str | None,
                       limit: int | None = None) -> list[Path]:
    """Images dropped into the library folders, most specific first.

    With a `limit`, one slot is kept for faction-wide images when there are
    any: those usually carry the faction's materials (bronze, oxblood...),
    which a full set of asset-type images would otherwise crowd out.
    """
    if not faction:
        return []
    root = reference_root(data) / faction
    specific = _images_in(root / category) if category else []
    general = _images_in(root)
    if limit is not None and specific and general:
        specific = specific[:max(1, limit - 1)]
    return specific + general


def _dedupe(paths: list[Path]) -> list[Path]:
    seen, out = set(), []
    for path in paths:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


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
            data["_dir"] = path.parent
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
            "references": _library_counts(data),
            "guides": {
                f"{f}/{c}": [g.stem for g in list_guides(data, f, c)]
                for f in data["factions"] for c in data["categories"]
                if list_guides(data, f, c)
            },
        })
    return out


def _library_counts(data: dict) -> dict[str, int]:
    """{"dominion": 2, "dominion/weapon": 3} for the folders that have images."""
    counts = {}
    for faction in data["factions"]:
        root = reference_root(data) / faction
        if n := len(_images_in(root)):
            counts[faction] = n
        for category in data["categories"]:
            if n := len(_images_in(root / category)):
                counts[f"{faction}/{category}"] = n
    return counts


def _apply_variant(cat: dict, prompt: str | None) -> dict:
    """Asset-type variants: the first whose `match` words appear in the prompt
    replaces the type's structure/negative/good/bad ("sniper" keeps a scope)."""
    words = set(re.findall(r"[a-z]+", (prompt or "").lower()))
    for variant in cat.get("variants", {}).values():
        if words & {m.lower() for m in variant.get("match", [])}:
            return {**cat, **{k: v for k, v in variant.items() if k != "match"}}
    return cat


def resolve(pack: str | None, faction: str | None, category: str | None,
            condition: str | None, prompt: str | None = None) -> ResolvedStyle | None:
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
    cat = _apply_variant(cat, prompt)
    # A faction can speak differently per asset type (e.g. Dominion weapons).
    override = fac.get("overrides", {}).get(category, {}) if category else {}
    faction_prompt = override.get("prompt", fac.get("prompt"))
    info = data.get("pack", {})
    negative = ", ".join(n for n in [info.get("negative"), cat.get("negative"),
                                     fac.get("negative"), override.get("negative")] if n)

    material = MaterialStyle(
        metals=[_hex(c) for c in fac.get("metals", [])],
        nonmetals=[_hex(c) for c in fac.get("nonmetals", [])],
        emissive=[_hex(c) for c in ([fac["emissive"]] if isinstance(fac.get("emissive"), str)
                                    else fac.get("emissive", []))],
        roughness_bias=fac.get("roughness_bias", 0.0) + cond.get("roughness_offset", 0.0),
        ao_strength=cond.get("ao_strength", 1.0),
    )
    return ResolvedStyle(
        # Order matters for 77-token CLIP models: faction look first, then framing.
        prompt_parts=[faction_prompt] if faction_prompt else [],
        # Faction accents lead SDXL's second encoder too (the faction prompt is on
        # the first), so key colours like oxblood and amber are stated twice.
        # Then condition and asset-type structure rules ("compact sight").
        look_parts=[p for p in (override.get("accents", fac.get("accents")),
                                cond.get("prompt"), cat.get("structure")) if p],
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
        guides=list_guides(data, faction, category),
        good=[*info.get("good", []), *cat.get("good", [])],
        bad=[*info.get("bad", []), *cat.get("bad", [])],
        references=_dedupe([
            *_pack_references(data, override.get("reference")),
            *library_references(data, faction, category, MAX_STYLE_REFERENCES),
            *_pack_references(data, fac.get("reference")),
        ])[:MAX_STYLE_REFERENCES],
        aspect=cat.get("aspect", "square"),
    )


# ------------------------------------------------------------ library edits

def _library_folder(pack: str, faction: str, category: str | None) -> Path:
    packs = load_packs()
    if pack not in packs:
        raise ValueError(f"Unknown style pack {pack!r}")
    data = packs[pack]
    # Only known names become folder names, so nothing can escape the library.
    if faction not in data["factions"]:
        raise ValueError(f"Unknown faction {faction!r}")
    if category and category not in data["categories"]:
        raise ValueError(f"Unknown asset type {category!r}")
    folder = reference_root(data) / faction
    return folder / category if category else folder


def list_library(pack: str, faction: str, category: str | None) -> list[dict]:
    """What a job with this faction + asset type would use, with where it comes from."""
    style = resolve(pack, faction, category, None)
    folder_specific = _library_folder(pack, faction, category) if category else None
    folder_faction = _library_folder(pack, faction, None)
    out = []
    for path in style.references:
        if folder_specific and path.parent == folder_specific.resolve():
            scope = "type"
        elif path.parent == folder_faction.resolve():
            scope = "faction"
        else:
            scope = "pack"  # set in the TOML file; edit it there
        out.append({"name": path.name, "scope": scope, "path": str(path)})
    return out


def add_reference(pack: str, faction: str, category: str | None, filename: str,
                  content: bytes) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise ValueError(f"Reference images must be one of {sorted(IMAGE_EXTS)}")
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(filename).stem)[:60]
    folder = _library_folder(pack, faction, category)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{stem or 'reference'}{suffix}"
    n = 2
    while target.exists():
        target = folder / f"{stem or 'reference'}_{n}{suffix}"
        n += 1
    target.write_bytes(content)
    return target


def delete_reference(pack: str, faction: str, category: str | None, name: str) -> None:
    folder = _library_folder(pack, faction, category)
    target = folder / Path(name).name
    if target.suffix.lower() not in IMAGE_EXTS or not target.is_file():
        raise ValueError(f"No reference named {name!r} here")
    target.unlink()
