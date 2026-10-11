"""PBR material stage: delight the albedo and derive normal / roughness / AO maps.

Generated textures usually have lighting "painted in" (shadows, highlights),
which looks fake as soon as Unreal lights the asset differently. This stage:

1. Delights: divides out low-frequency shading so the engine can light it.
2. Derives a tangent-space normal map from fine surface detail.
3. Estimates roughness and ambient occlusion.
4. Packs Occlusion/Roughness/Metallic into one ORM texture. glTF's
   metallicRoughness channels (G=roughness, B=metallic) plus R=occlusion is
   exactly Unreal's ORM convention, so the same image works in both.

These are image-space heuristics, not a learned material model. They are a
big step up from a flat baked texture, and the backend can be swapped for a
learned PBR model later without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class PBRMaps:
    albedo: Image.Image
    normal_gl: Image.Image  # OpenGL / glTF convention (+Y)
    normal_dx: Image.Image  # DirectX / Unreal convention (-Y)
    roughness: Image.Image
    ao: Image.Image
    orm: Image.Image  # R=AO, G=roughness, B=metallic


def _to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray, mode: str = "RGB") -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), mode)


def delight(rgb: np.ndarray, strength: float) -> np.ndarray:
    """Remove large-scale shading while keeping colour and fine detail."""
    if strength <= 0:
        return rgb
    luma = rgb @ LUMA
    sigma = max(rgb.shape[:2]) / 32
    shading = ndimage.gaussian_filter(luma, sigma) + 1e-3
    shading /= np.median(shading)
    flattened = rgb / shading[..., None] ** strength
    # Also pull blown highlights and crushed shadows toward the middle.
    flattened = flattened / (1 + strength * 0.35 * np.maximum(flattened - 0.85, 0))
    return np.clip(flattened, 0, 1)


def height_from_albedo(rgb: np.ndarray) -> np.ndarray:
    """Fine surface detail as a 0..1 height field (high-pass luminance)."""
    luma = rgb @ LUMA
    detail = luma - ndimage.gaussian_filter(luma, 6)
    detail = ndimage.gaussian_filter(detail, 0.8)
    scale = np.percentile(np.abs(detail), 99) + 1e-6
    return np.clip(0.5 + 0.5 * detail / scale, 0, 1)


def normal_from_height(height: np.ndarray, strength: float = 2.0) -> np.ndarray:
    dx = ndimage.sobel(height, axis=1)
    dy = ndimage.sobel(height, axis=0)
    # Image rows go down; flip so +Y points up in texture space (OpenGL).
    n = np.dstack([-dx * strength, dy * strength, np.ones_like(height)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def roughness_from_albedo(rgb: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Busy, high-contrast surfaces read as rough; smooth uniform areas as glossier."""
    local_mean = ndimage.uniform_filter(height, 9)
    local_var = ndimage.uniform_filter(height ** 2, 9) - local_mean ** 2
    busy = np.sqrt(np.maximum(local_var, 0))
    busy /= np.percentile(busy, 98) + 1e-6
    # Very bright, unsaturated pixels are often specular highlights -> glossier.
    sat = rgb.max(axis=-1) - rgb.min(axis=-1)
    glint = np.clip((rgb @ LUMA - 0.8) * 4, 0, 1) * (1 - sat)
    rough = 0.55 + 0.35 * np.clip(busy, 0, 1) - 0.25 * glint
    return np.clip(ndimage.gaussian_filter(rough, 1.5), 0.15, 1.0)


def ao_from_height(height: np.ndarray) -> np.ndarray:
    cavity = ndimage.gaussian_filter(height, 4) - height
    return np.clip(1.0 - np.maximum(cavity, 0) * 2.5, 0.4, 1.0)


def build_pbr_maps(base_color: Image.Image, delight_strength: float) -> PBRMaps:
    rgb = _to_float(base_color)
    albedo = delight(rgb, delight_strength)
    height = height_from_albedo(albedo)
    normal_gl = normal_from_height(height)
    normal_dx = normal_gl.copy()
    normal_dx[..., 1] = 1.0 - normal_dx[..., 1]
    rough = roughness_from_albedo(albedo, height)
    ao = ao_from_height(height)
    metal = np.zeros_like(rough)  # dielectric by default; metals need a learned model
    orm = np.dstack([ao, rough, metal])
    return PBRMaps(
        albedo=_to_image(albedo),
        normal_gl=_to_image(normal_gl),
        normal_dx=_to_image(normal_dx),
        roughness=_to_image(rough, "L"),
        ao=_to_image(ao, "L"),
        orm=_to_image(orm),
    )


def extract_base_color(mesh: trimesh.Trimesh) -> Image.Image | None:
    visual = mesh.visual
    if not isinstance(visual, trimesh.visual.TextureVisuals) or visual.material is None:
        return None
    material = visual.material
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    return image.convert("RGB") if image is not None else None


def apply_pbr_maps(mesh: trimesh.Trimesh, maps: PBRMaps) -> None:
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        name="meshify_pbr",
        baseColorTexture=maps.albedo,
        normalTexture=maps.normal_gl,
        metallicRoughnessTexture=maps.orm,
        occlusionTexture=maps.orm,
        metallicFactor=1.0,
        roughnessFactor=1.0,
    )
