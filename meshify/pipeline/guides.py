"""Silhouette guides: keep text-to-image proportions on model.

A guide is a silhouette (black shape on white) or clean line art of an
archetype: an assault rifle side view, a sidearm, a blade. It becomes an edge
map for an SDXL ControlNet, so the generated reference follows that outline
(one scope, a sane stock, the right length) while the prompt and style
references still decide materials and detail.

Guides live next to the style pack:

    guides/<faction>/<asset type>/assault_rifle.png   (faction-specific, first)
    guides/<asset type>/sidearm.png                    (any faction)

"Auto" picks the guide whose file name best matches the prompt ("heavy
assault rifle" -> assault_rifle, "pistol" -> sidearm). No match, no guide:
a rifle outline is never forced onto a grenade.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Prompt words -> the archetype word used in guide file names.
SYNONYMS = {
    "pistol": "sidearm", "handgun": "sidearm", "revolver": "sidearm", "sidearm": "sidearm",
    "rifle": "rifle", "carbine": "rifle", "gun": "rifle",
    "sniper": "marksman", "marksman": "marksman", "dmr": "marksman",
    "shotgun": "shotgun", "scattergun": "shotgun",
    "heavy": "heavy", "cannon": "heavy", "minigun": "heavy", "rotary": "heavy",
    "launcher": "heavy", "gatling": "heavy",
    "blade": "blade", "sword": "blade", "greatblade": "blade", "greatsword": "blade",
    "knife": "blade", "dagger": "blade", "saber": "blade", "sabre": "blade",
}
# Name words that say nothing about shape.
IGNORED = {"weapon", "the", "a", "of", "with", "and"}


def guide_root(data: dict) -> Path:
    if env := os.environ.get("MESHIFY_GUIDE_DIR"):
        return Path(env)
    return data["_dir"] / "guides"


def list_guides(data: dict, faction: str | None, category: str | None) -> list[Path]:
    if not category:
        return []
    root = guide_root(data)
    folders = ([root / faction / category] if faction else []) + [root / category]
    out = []
    for folder in folders:
        if folder.is_dir():
            out += sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return out


def _words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {SYNONYMS.get(w, w) for w in words if w not in IGNORED}


def choose(guides: list[Path], prompt: str | None) -> Path | None:
    """Best name match for the prompt, or None if nothing matches at all."""
    wanted = _words(prompt or "")
    best, best_key = None, None
    for i, path in enumerate(guides):
        name = _words(path.stem.replace("_", " "))
        overlap = len(wanted & name)
        if overlap == 0:
            continue
        # More shared words wins; then the more specific (shorter) name; then
        # faction-specific folders, which list_guides returns first.
        key = (overlap, -len(name), -i)
        if best_key is None or key > best_key:
            best, best_key = path, key
    return best


def find(guides: list[Path], name: str) -> Path | None:
    return next((p for p in guides if p.stem == name or p.name == name), None)


def control_image(guide: Image.Image, width: int, height: int, fill: float = 0.85) -> Image.Image:
    """Edge map for a canny ControlNet: white outline on black, the shape fitted
    into the generation frame with a margin (matching the framing prompt)."""
    gray = np.asarray(guide.convert("L"), dtype=np.float32) / 255
    shape = gray < 0.5  # dark = object (silhouette or line art)
    if not shape.any():
        raise ValueError("The guide image has no dark shape to follow.")
    ys, xs = np.nonzero(shape)
    shape = shape[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = shape.shape
    # A tall guide (an upright blade) in a wide frame would come out tiny:
    # lay it on its side, which is also the side profile image-to-3D wants.
    if (h > 1.5 * w and width > height) or (w > 1.5 * h and height > width):
        shape = np.rot90(shape)
        h, w = shape.shape
    scale = min(width * fill / w, height * fill / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = np.asarray(Image.fromarray(shape.astype(np.uint8) * 255)
                         .resize((new_w, new_h), Image.LANCZOS)) > 127
    canvas = np.zeros((height, width), dtype=bool)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    thickness = max(1, round(min(width, height) / 384))
    outline = canvas & ~ndimage.binary_erosion(canvas, iterations=thickness)
    return Image.fromarray((outline * 255).astype(np.uint8)).convert("RGB")
