"""PBR material stage: delight the albedo and derive normal / roughness / AO maps.

Generated textures usually have lighting "painted in" (shadows, highlights),
which looks fake as soon as Unreal lights the asset differently. This stage:

1. Delights: divides out low-frequency shading so the engine can light it.
2. Derives a tangent-space normal map from fine surface detail.
3. Estimates roughness and ambient occlusion.
4. Packs Occlusion/Roughness/Metallic into one ORM texture. glTF's
   metallicRoughness channels (G=roughness, B=metallic) plus R=occlusion is
   exactly Unreal's ORM convention, so the same image works in both.

With a style pack, the faction's metal swatches drive the metallic map, its
energy colour is pulled out into an isolated emissive mask, and the
condition (pristine ... derelict) shifts roughness and AO.

These are image-space heuristics, not a learned material model. They are a
big step up from a flat baked texture, and the backend can be swapped for a
learned PBR model later without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

if TYPE_CHECKING:
    from ..styles import MaterialStyle

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class PBRMaps:
    albedo: Image.Image
    normal_gl: Image.Image  # OpenGL / glTF convention (+Y)
    normal_dx: Image.Image  # DirectX / Unreal convention (-Y)
    roughness: Image.Image
    ao: Image.Image
    orm: Image.Image  # R=AO, G=roughness, B=metallic
    metallic: Image.Image | None = None
    emissive: Image.Image | None = None  # glow colour, black elsewhere
    emissive_mask: Image.Image | None = None  # 0..1, for an intensity parameter in Unreal


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


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..1) -> CIE Lab, so colour distances roughly match perception."""
    rgb = rgb.astype(np.float32, copy=False)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.2126, 0.0193],
                          [0.3576, 0.7152, 0.1192],
                          [0.1805, 0.0722, 0.9505]], dtype=np.float32)
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def _distance_to(lab: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    swatches = srgb_to_lab(np.array(colors, dtype=np.float32) / 255.0)
    # One swatch at a time keeps memory flat on 4K textures.
    best = np.full(lab.shape[:2], np.inf, dtype=np.float32)
    for swatch in swatches:
        np.minimum(best, np.linalg.norm(lab - swatch, axis=-1), out=best)
    return best


def metallic_from_palette(albedo: np.ndarray, metals: list[tuple[int, int, int]],
                          nonmetals: list[tuple[int, int, int]] | None = None) -> np.ndarray:
    """Pixels close to the faction's metal swatches become metallic (soft edge).

    Dark colours all sit close together in Lab, so oxblood enamel is "near"
    blackened steel. With `nonmetals` (enamel, leather, paint), each pixel goes
    to whichever swatch family it is actually closer to.
    """
    if not metals:
        return np.zeros(albedo.shape[:2], dtype=np.float32)
    lab = srgb_to_lab(albedo)
    dist = _distance_to(lab, metals)
    metal = np.clip(1.0 - (dist - 12.0) / 18.0, 0, 1)  # full under dE 12, none past 30
    if nonmetals:
        margin = _distance_to(lab, nonmetals) - dist  # > 0: closer to a metal swatch
        metal *= np.clip(0.5 + margin / 8.0, 0, 1)
    return ndimage.gaussian_filter(metal, 1.0).astype(np.float32)


def emissive_from_palette(rgb: np.ndarray, colors: list[tuple[int, int, int]],
                          max_coverage: float = 0.2) -> np.ndarray | None:
    """Find glowing channels: near one of the faction's energy colours AND brighter than
    their surroundings. The local-contrast test stops flat ivory or white panels
    from being mistaken for white-gold glow."""
    if not colors:
        return None
    luma = rgb @ LUMA
    lifted = luma - ndimage.gaussian_filter(luma, 8)
    near = np.clip(1.0 - (_distance_to(srgb_to_lab(rgb), colors) - 15.0) / 20.0, 0, 1)
    mask = near * np.clip((lifted - 0.06) / 0.12, 0, 1) * (luma > 0.45)
    mask = ndimage.gaussian_filter(mask.astype(np.float32), 0.8)
    coverage = float((mask > 0.5).mean())
    if coverage < 0.0005:
        return None  # no glow found
    if coverage > max_coverage:
        return None  # implausibly large: probably just a brightly coloured surface
    return mask


def build_pbr_maps(base_color: Image.Image, delight_strength: float,
                   style: "MaterialStyle | None" = None) -> PBRMaps:
    rgb = _to_float(base_color)
    albedo = delight(rgb, delight_strength)
    height = height_from_albedo(albedo)
    normal_gl = normal_from_height(height)
    normal_dx = normal_gl.copy()
    normal_dx[..., 1] = 1.0 - normal_dx[..., 1]
    rough = roughness_from_albedo(albedo, height)
    ao = ao_from_height(height)

    metal = np.zeros_like(rough)  # dielectric unless the style says which colours are metal
    emissive_mask = None
    if style is not None:
        metal = metallic_from_palette(albedo, style.metals, style.nonmetals)
        # Machined metal reads smoother than paint or stone.
        rough = rough - 0.15 * metal + style.roughness_bias
        ao = 1.0 - (1.0 - ao) * style.ao_strength
        # Detect glow on the original texture, where emissive pixels are brightest.
        emissive_mask = emissive_from_palette(rgb, style.emissive)
    rough = np.clip(rough, 0.05, 1.0)
    ao = np.clip(ao, 0.0, 1.0)

    orm = np.dstack([ao, rough, metal])
    maps = PBRMaps(
        albedo=_to_image(albedo),
        normal_gl=_to_image(normal_gl),
        normal_dx=_to_image(normal_dx),
        roughness=_to_image(rough, "L"),
        ao=_to_image(ao, "L"),
        orm=_to_image(orm),
        metallic=_to_image(metal, "L"),
    )
    if emissive_mask is not None:
        maps.emissive = _to_image(rgb * emissive_mask[..., None])
        maps.emissive_mask = _to_image(emissive_mask, "L")
    return maps


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
        emissiveTexture=maps.emissive,
        emissiveFactor=[1.0, 1.0, 1.0] if maps.emissive is not None else None,
    )
