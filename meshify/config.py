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
    style_image_paths: list[str] = field(default_factory=list)  # replace the library refs
    style_strength: float = 0.6  # how strongly the reference's look is applied
    candidates: int | None = None  # text-to-image candidates to judge (None = default 3)
    guide: str | None = None  # silhouette: None/"auto" = match prompt, "none", a guide name or file path
    guide_strength: float = 0.6  # how strictly the reference follows the silhouette

    # Realism pipeline
    realism: bool = True  # photoreal prompt shaping + delighting + PBR maps
    delight_strength: float = 0.6  # 0 = keep baked lighting, 1 = flatten fully
    projection_strength: float = 0.85  # reference projected onto camera-facing surfaces (0 = off)
    normal_strength: float = 1.0  # derived normal map strength (0 = flat, for diagnosis)
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
        if self.candidates is not None and not 1 <= self.candidates <= 8:
            raise ValueError("candidates must be between 1 and 8.")
        if not 0.0 <= self.guide_strength <= 1.0:
            raise ValueError("guide_strength must be between 0 and 1.")
        if not 0.0 <= self.style_strength <= 1.0:
            raise ValueError("style_strength must be between 0 and 1.")
        if not 0.0 <= self.normal_strength <= 3.0:
            raise ValueError("normal_strength must be between 0 and 3.")
        if not 0.0 <= self.projection_strength <= 1.0:
            raise ValueError("projection_strength must be between 0 and 1.")
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
