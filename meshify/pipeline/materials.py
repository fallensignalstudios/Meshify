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
    from .projection import Projection

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
    stats: dict | None = None  # coverage numbers for debugging/tuning


def _to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray, mode: str = "RGB") -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), mode)


def texture_mask(rgb: np.ndarray) -> np.ndarray:
    """Texels that belong to the model. UV atlases (Hunyuan3D, TRELLIS) leave
    empty space between islands, usually black or one flat fill colour; it must
    not take part in brightness statistics or blurs."""
    used = rgb.max(axis=-1) > 0.02
    flat = rgb.reshape(-1, 3)
    sample = flat[np.random.default_rng(0).integers(0, len(flat), min(len(flat), 200_000))]
    values, counts = np.unique((sample * 255).astype(np.uint8), axis=0, return_counts=True)
    share = counts.max() / len(sample)
    fill = values[counts.argmax()].astype(np.float32) / 255
    # Padding is one exact, neutral (grey/black) colour over a big chunk of the
    # atlas; a saturated flat colour is a real surface (e.g. a painted panel).
    if 0.15 < share < 0.9 and fill.max() - fill.min() < 0.06:
        used &= np.abs(rgb - fill).max(axis=-1) > 1.5 / 255
    if used.mean() < 0.01:  # nothing left? then treat the whole image as texture
        return np.ones(rgb.shape[:2], dtype=bool)
    return used


def _blur(x: np.ndarray, sigma: float, mask: np.ndarray | None) -> np.ndarray:
    """Gaussian blur that only averages texels inside `mask` (normalized
    convolution), so empty atlas space doesn't drag island edges down."""
    if mask is None:
        return ndimage.gaussian_filter(x, sigma)
    m = mask.astype(np.float32)
    num = ndimage.gaussian_filter(x * m, sigma)
    den = ndimage.gaussian_filter(m, sigma)
    return np.where(den > 1e-4, num / np.maximum(den, 1e-4), x)


def delight(rgb: np.ndarray, strength: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Remove large-scale shading while keeping colour and fine detail."""
    if strength <= 0:
        return rgb
    luma = rgb @ LUMA
    sigma = max(rgb.shape[:2]) / 32
    shading = _blur(luma, sigma, mask) + 1e-3
    shading /= np.median(shading[mask] if mask is not None else shading)
    flattened = rgb / shading[..., None] ** strength
    # Also pull blown highlights and crushed shadows toward the middle.
    flattened = flattened / (1 + strength * 0.35 * np.maximum(flattened - 0.85, 0))
    if mask is not None:
        flattened = np.where(mask[..., None], flattened, rgb)
    return np.clip(flattened, 0, 1)


def height_from_albedo(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Fine surface detail as a 0..1 height field (high-pass luminance)."""
    luma = rgb @ LUMA
    detail = luma - _blur(luma, 6, mask)
    # Generated textures are grainy; smooth before deriving normals or the
    # grain reads as a rough, pitted surface.
    detail = ndimage.gaussian_filter(detail, 1.4)
    if mask is not None:
        detail *= mask
    scale = np.percentile(np.abs(detail[mask] if mask is not None else detail), 99) + 1e-6
    return np.clip(0.5 + 0.5 * detail / scale, 0, 1)


def normal_from_height(height: np.ndarray, strength: float = 1.2) -> np.ndarray:
    dx = ndimage.sobel(height, axis=1)
    dy = ndimage.sobel(height, axis=0)
    # Image rows go down; flip so +Y points up in texture space (OpenGL).
    n = np.dstack([-dx * strength, dy * strength, np.ones_like(height)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def roughness_from_albedo(rgb: np.ndarray, height: np.ndarray,
                          mask: np.ndarray | None = None) -> np.ndarray:
    """Busy, high-contrast surfaces read as rough; smooth uniform areas as glossier."""
    local_mean = ndimage.uniform_filter(height, 9)
    local_var = ndimage.uniform_filter(height ** 2, 9) - local_mean ** 2
    busy = np.sqrt(np.maximum(local_var, 0))
    busy /= np.percentile(busy[mask] if mask is not None else busy, 98) + 1e-6
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


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > 0.206893, f ** 3, (f - 16 / 116) / 7.787)
    xyz *= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    lin = xyz @ np.array([[3.2406, -0.9689, 0.0557],
                          [-1.5372, 1.8758, -0.2040],
                          [-0.4986, 0.0415, 1.0570]], dtype=np.float32)
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def match_reference(rgb: np.ndarray, mask: np.ndarray, ref_rgb: np.ndarray,
                    ref_mask: np.ndarray, strength: float = 0.85) -> np.ndarray:
    """Pull the texture's colour statistics (Lab mean/spread) toward the
    reference image. Baked textures come out darker and muddier than the
    reference the user approved; this restores its brightness and saturation
    without moving any detail."""
    lab = srgb_to_lab(rgb)
    ref_lab = srgb_to_lab(ref_rgb)[ref_mask]
    tex = lab[mask]
    mu_t, sd_t = tex.mean(0), tex.std(0) + 1e-3
    mu_r, sd_r = ref_lab.mean(0), ref_lab.std(0) + 1e-3
    ratio = np.clip(sd_r / sd_t, 0.5, 2.5)
    matched = (lab - mu_t) * ratio + mu_r
    out = lab_to_srgb(lab + strength * (matched - lab))
    return np.where(mask[..., None], out, rgb).astype(np.float32)


def _kmeans_colors(rgb: np.ndarray, k: int = 6, iters: int = 8) -> list[tuple[int, int, int]]:
    """A few representative colours of a pixel set (tiny k-means in Lab)."""
    if len(rgb) == 0:
        return []
    rng = np.random.default_rng(0)
    pts = srgb_to_lab(rgb[rng.integers(0, len(rgb), min(len(rgb), 20_000))])
    k = min(k, len(pts))
    # Spread initial centres across lightness so dark and bright parts both get one.
    centres = pts[np.argsort(pts[:, 0])[np.linspace(0, len(pts) - 1, k).astype(int)]]
    for _ in range(iters):
        labels = np.argmin(((pts[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(labels == j):
                centres[j] = pts[labels == j].mean(0)
    out = (lab_to_srgb(centres) * 255).round().astype(int)
    return [tuple(int(c) for c in col) for col in out]


def learn_palette(ref_rgb: np.ndarray, ref_mask: np.ndarray, style: "MaterialStyle"):
    """Classify the reference with the faction swatches (this works well on a
    clean studio render), then sample its actual metal, non-metal and glow
    colours. The texture is classified against these, so a darker bake of the
    same bronze still matches bronze instead of drifting toward oxblood."""
    metal = metallic_from_palette(ref_rgb, style.metals, style.nonmetals)
    glow = emissive_from_palette(ref_rgb, style.emissive, style.nonmetals, mask=ref_mask)
    glow = glow if glow is not None else np.zeros(ref_mask.shape, np.float32)
    metals = _kmeans_colors(ref_rgb[ref_mask & (metal > 0.6) & (glow < 0.3)])
    nonmetals = _kmeans_colors(ref_rgb[ref_mask & (metal < 0.2) & (glow < 0.3)])
    glows = _kmeans_colors(ref_rgb[ref_mask & (glow > 0.5)], k=3)
    return metals, nonmetals, glows


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


def _hue_saturation(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = (np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) + 360) % 360
    return hue, sat, mx


def _vivid_glow(rgb: np.ndarray, mask: np.ndarray, hues: list[float]) -> np.ndarray:
    """Glow in a vivid colour family (amber, cyan, violet), found as regions:

    - seeds: pixels that look like light (very bright, vivid, in the hue family,
      clearly brighter than their surroundings), in clusters of real size;
    - growth: from the seeds into connected, vivid pixels of the same family,
      which picks up the darker, fiery parts of a channel (hysteresis);
    - holes inside a light are filled.

    Specular glints on bronze or red trim never seed, so they don't leak in."""
    k = min(rgb.shape[:2]) / 768  # tuned at 768 px; scale structuring sizes
    it = lambda n: max(1, round(n * k))
    hue, sat, mx = _hue_saturation(rgb)
    hd = np.min([np.minimum(np.abs(hue - h), 360 - np.abs(hue - h)) for h in hues], axis=0)
    luma = rgb @ LUMA
    lifted = luma - _blur(luma, 25 * k, mask)
    seeds = (hd < 30) & (sat > 0.45) & (mx > 0.8) & (lifted > 0.12) & mask
    labels, n = ndimage.label(seeds)
    if n == 0:
        return np.zeros(mask.shape, bool)
    sizes = ndimage.sum(seeds, labels, range(1, n + 1))
    seeds = np.isin(labels, 1 + np.flatnonzero(sizes >= 30 * k * k))
    # Bronze shares amber's hue, so growth also needs to be a little brighter
    # than its surroundings and must stay near a seed (no flooding a panel).
    near_seed = ndimage.binary_dilation(seeds, iterations=it(14))
    family = ((hd < 40) & (sat > 0.5) & (mx > 0.4) & (lifted > 0.03) & near_seed & mask) | seeds
    family = ndimage.binary_closing(family, iterations=it(2))
    labels, _ = ndimage.label(family)
    keep = np.unique(labels[seeds])
    grown = np.isin(labels, keep[keep > 0])
    return ndimage.binary_fill_holes(ndimage.binary_closing(grown, iterations=it(4))) & mask


def _pale_glow(rgb: np.ndarray, mask: np.ndarray | None, colors, nonmetals) -> np.ndarray:
    """Glow in a pale colour (white-gold): near the colour AND brighter than its
    surroundings; `nonmetals` (ivory, pearl) veto bright edge highlights."""
    luma = rgb @ LUMA
    lifted = luma - _blur(luma, 8, mask)
    lab = srgb_to_lab(rgb)
    dist = _distance_to(lab, colors)
    near = np.clip(1.0 - (dist - 15.0) / 20.0, 0, 1)
    glow = near * np.clip((lifted - 0.04) / 0.12, 0, 1) * (rgb.max(axis=-1) > 0.6)
    if nonmetals:
        glow *= np.clip(0.5 + (_distance_to(lab, nonmetals) - dist) / 8.0, 0, 1)
    return ndimage.gaussian_filter(glow.astype(np.float32), 0.8)


def emissive_from_palette(rgb: np.ndarray, colors: list[tuple[int, int, int]],
                          nonmetals: list[tuple[int, int, int]] | None = None,
                          max_coverage: float = 0.2,
                          mask: np.ndarray | None = None) -> np.ndarray | None:
    """Glow mask (0..1) for a faction's energy colours, or None if there's none.

    Vivid colours are found by hue family as connected light regions; pale
    colours (white-gold) by colour distance plus local brightness."""
    if not colors:
        return None
    area_mask = mask if mask is not None else np.ones(rgb.shape[:2], bool)
    arr = np.array(colors, dtype=np.float32) / 255
    hue, sat, _ = _hue_saturation(arr[None])
    vivid = [float(h) for h, s_ in zip(hue[0], sat[0]) if s_ >= 0.35]
    pale = [c for c, s_ in zip(colors, sat[0]) if s_ < 0.35]
    glow = np.zeros(rgb.shape[:2], np.float32)
    if vivid:
        glow = np.maximum(glow, ndimage.gaussian_filter(
            _vivid_glow(rgb, area_mask, vivid).astype(np.float32), 0.8))
    if pale:
        glow = np.maximum(glow, _pale_glow(rgb, mask, pale, nonmetals))
    glow *= area_mask
    coverage = float((glow > 0.5).sum() / max(area_mask.sum(), 1))
    if coverage < 0.0005:
        return None  # no glow found
    if coverage > max_coverage:
        return None  # implausibly large: probably just a brightly coloured surface
    return glow


def build_pbr_maps(base_color: Image.Image, delight_strength: float,
                   style: "MaterialStyle | None" = None,
                   reference: Image.Image | None = None,
                   projection: "Projection | None" = None,
                   projection_strength: float = 0.85) -> PBRMaps:
    """`reference` is the RGBA cut-out the mesh was made from. When given, the
    texture is colour-matched to it and materials are learned from it.
    `projection` (from projection.project) maps texels to reference pixels:
    surfaces facing the reference view take its colours directly, and its
    glow is projected onto the mesh."""
    rgb = _to_float(base_color)
    mask = texture_mask(rgb)
    stats = {"texture_coverage": round(float(mask.mean()), 3), "colour_matched": False}
    ref_rgb = ref_mask = None
    if reference is not None:
        ref = np.asarray(reference.convert("RGBA"), dtype=np.float32) / 255.0
        ref_rgb, ref_mask = ref[..., :3], ref[..., 3] > 0.5

    proj_weight = None
    if projection is not None and ref_rgb is not None and projection_strength > 0:
        proj_weight = projection.weight * projection_strength
        if proj_weight.shape == mask.shape:
            projected = projection.sample(ref_rgb)
            rgb = rgb * (1 - proj_weight[..., None]) + projected * proj_weight[..., None]
            mask = mask | (proj_weight > 0.3)
            stats["projected"] = round(float((projection.weight > 0.3).mean()), 3)
            stats["projection_iou"] = round(projection.iou, 2)
        else:
            proj_weight = None

    metals = list(style.metals) if style else []
    nonmetals = list(style.nonmetals) if style else []
    glows = list(style.emissive) if style else []
    learned_glows: list[tuple[int, int, int]] = []
    if ref_rgb is not None:
        if ref_mask.sum() > 500 and mask.sum() > 500:
            rgb = match_reference(rgb, mask, ref_rgb, ref_mask)
            stats["colour_matched"] = True
            if style is not None:
                m, n, g = learn_palette(ref_rgb, ref_mask, style)
                metals, nonmetals, learned_glows = metals + m, nonmetals + n, g
                stats["learned_colours"] = {"metal": len(m), "nonmetal": len(n), "glow": len(g)}

    albedo = delight(rgb, delight_strength, mask)
    height = height_from_albedo(albedo, mask)
    normal_gl = normal_from_height(height)
    normal_dx = normal_gl.copy()
    normal_dx[..., 1] = 1.0 - normal_dx[..., 1]
    rough = roughness_from_albedo(albedo, height, mask)
    ao = ao_from_height(height)

    metal = np.zeros_like(rough)  # dielectric unless the style says which colours are metal
    emissive_mask = None
    if style is not None:
        # Classify the colour-matched (not delit) texture: it's comparable to
        # the reference the palette was learned from.
        metal = metallic_from_palette(rgb, metals, nonmetals) * mask
        # Machined metal reads smoother than paint or stone.
        rough = rough - 0.15 * metal + style.roughness_bias
        ao = 1.0 - (1.0 - ao) * style.ao_strength
        emissive_mask = emissive_from_palette(rgb, glows, nonmetals, mask=mask)
        if learned_glows:
            # A bake often dulls glow into paint; the reference's own glow
            # colours still find it by colour distance + local brightness.
            dull = _pale_glow(rgb, mask, learned_glows, nonmetals) * mask
            if 0.0005 < (dull > 0.5).sum() / max(mask.sum(), 1) < 0.2:
                emissive_mask = dull if emissive_mask is None else np.maximum(emissive_mask, dull)
        if proj_weight is not None:
            # Texture painters turn glowing channels into dull paint; find the
            # glow on the clean reference and project it instead.
            ref_glow = emissive_from_palette(ref_rgb, glows, list(style.nonmetals), mask=ref_mask)
            if ref_glow is not None:
                projected_glow = projection.sample(ref_glow) * (projection.weight > 0.3)
                emissive_mask = projected_glow if emissive_mask is None else \
                    np.maximum(emissive_mask, projected_glow)
                if not (emissive_mask > 0.5).any():
                    emissive_mask = None
    rough = np.clip(rough, 0.05, 1.0)
    ao = np.clip(ao, 0.0, 1.0)
    stats["metallic_coverage"] = round(float((metal[mask] > 0.5).mean()), 3)
    stats["emissive_coverage"] = (round(float((emissive_mask[mask] > 0.5).mean()), 4)
                                  if emissive_mask is not None else 0.0)

    orm = np.dstack([ao, rough, metal])
    maps = PBRMaps(
        albedo=_to_image(albedo),
        normal_gl=_to_image(normal_gl),
        normal_dx=_to_image(normal_dx),
        roughness=_to_image(rough, "L"),
        ao=_to_image(ao, "L"),
        orm=_to_image(orm),
        metallic=_to_image(metal, "L"),
        stats=stats,
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
