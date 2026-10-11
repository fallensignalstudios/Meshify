"""Generation settings shared by the CLI, the server and the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

OUTPUT_ROOT = Path(os.environ.get("MESHIFY_OUTPUT_DIR", "outputs"))
DEFAULT_BACKEND = os.environ.get("MESHIFY_BACKEND", "trellis")

# Used when neither the request nor a style preset says otherwise.
DEFAULT_TEXTURE_SIZE = 2048
DEFAULT_TARGET_FACES = 50_000

TEXTURE_SIZES = (512, 1024, 2048, 4096)
SIZE_AXES = ("height", "longest")
PIVOTS = ("bottom_center", "bottom_corner", "center")


@dataclass
class GenerationRequest:
    """Everything needed to turn a prompt or image into an Unreal-ready asset.

    Fields left as None fall back to the style preset (if any), then defaults.
    """

    prompt: str | None = None
    image_path: str | None = None
    backend: str = DEFAULT_BACKEND
    seed: int = 42

    # Style preset (see meshify/styles/*.toml)
    style_pack: str | None = None
    faction: str | None = None
    category: str | None = None
    condition: str | None = None
    style_image_path: str | None = None  # overrides the pack's faction reference
    style_strength: float = 0.6  # how strongly the reference's look is applied

    # Realism pipeline
    realism: bool = True  # photoreal prompt shaping + delighting + PBR maps
    delight_strength: float = 0.6  # 0 = keep baked lighting, 1 = flatten fully
    texture_size: int | None = None

    # Mesh / export
    target_faces: int | None = None  # 0 = full density (e.g. for Nanite)
    real_world_size_m: float | None = None
    size_axis: str | None = None  # "height" or "longest"
    pivot: str | None = None
    asset_name: str = "asset"

    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt and not self.image_path:
            raise ValueError("Provide a prompt, an image, or both.")
        if not 0.0 <= self.style_strength <= 1.0:
            raise ValueError("style_strength must be between 0 and 1.")
        if not 0.0 <= self.delight_strength <= 1.0:
            raise ValueError("delight_strength must be between 0 and 1.")
        if self.texture_size is not None and self.texture_size not in TEXTURE_SIZES:
            raise ValueError("texture_size must be 512, 1024, 2048 or 4096.")
        if self.target_faces is not None and 0 < self.target_faces < 500:
            raise ValueError("target_faces must be at least 500 (or 0 for full density).")
        if self.real_world_size_m is not None and self.real_world_size_m <= 0:
            raise ValueError("real_world_size_m must be positive.")
        if self.size_axis is not None and self.size_axis not in SIZE_AXES:
            raise ValueError(f"size_axis must be one of {SIZE_AXES}.")
        if self.pivot is not None and self.pivot not in PIVOTS:
            raise ValueError(f"pivot must be one of {PIVOTS}.")
        from .styles import resolve

        resolve(self.style_pack, self.faction, self.category, self.condition)

    def to_dict(self) -> dict:
        return asdict(self)
