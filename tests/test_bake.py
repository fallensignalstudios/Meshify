"""Baked UV-atlas textures: empty space between islands, darker and duller than
the reference. The material stage must not darken them further, and must find
the same metal/enamel/glow as on the reference."""
import numpy as np
from PIL import Image

from meshify.pipeline import images, materials
from meshify.styles import resolve

REF = "meshify/styles/references/dominion/hero_weapon/officer_rifle_target.png"


def _fake_bake(cutout: Image.Image) -> Image.Image:
    a = np.asarray(cutout.convert("RGBA"), dtype=np.float32) / 255
    rgb, alpha = a[..., :3], a[..., 3] > 0.5
    ys, xs = np.nonzero(alpha)
    obj, m = rgb[ys.min():ys.max(), xs.min():xs.max()], alpha[ys.min():ys.max(), xs.min():xs.max()]
    atlas = np.zeros((900, 900, 3), np.float32)
    w = obj.shape[1]
    for i, (oy, ox) in enumerate([(20, 20), (330, 60), (620, 120)]):  # three islands with gaps
        part, pm = obj[:, i * w // 3:(i + 1) * w // 3], m[:, i * w // 3:(i + 1) * w // 3]
        part, pm = part[:250, :700], pm[:250, :700]
        region = atlas[oy:oy + part.shape[0], ox:ox + part.shape[1]]
        region[pm] = part[pm]
    lum = (atlas @ materials.LUMA)[..., None]
    return Image.fromarray((np.clip(0.55 * (0.6 * atlas + 0.4 * lum), 0, 1) * 255).astype(np.uint8))


def test_delight_does_not_darken_atlas_islands():
    tex = np.zeros((256, 256, 3), np.float32)
    tex[20:150, 20:230] = (0.55, 0.40, 0.25)
    tex[170:240, 30:150] = (0.35, 0.10, 0.08)
    mask = materials.texture_mask(tex)
    out = materials.delight(tex, 0.6, mask)
    assert out[mask].mean() / tex[mask].mean() > 0.9  # used to drop to ~0.63


def test_neutral_padding_is_masked_but_flat_paint_is_not():
    padded = np.full((100, 100, 3), 0.5, np.float32)  # grey padding fill
    padded[10:60, 10:60] = np.random.default_rng(0).random((50, 50, 3))
    assert not materials.texture_mask(padded)[80, 80] and materials.texture_mask(padded)[30, 30]
    panel = np.zeros((100, 100, 3), np.float32) + (0.55, 0.38, 0.22)  # flat bronze paint
    assert materials.texture_mask(panel).all()


def test_baked_texture_recovers_reference_look():
    ref = Image.open(REF).convert("RGB")
    cutout = images.remove_background(ref)
    alpha = np.asarray(cutout)[..., 3] > 128
    baked = _fake_bake(cutout)
    style = resolve("sovereign_call", "dominion", "weapon", "maintained").material

    ref_maps = materials.build_pbr_maps(ref, 0.0, style)
    ref_metal = (np.asarray(ref_maps.metallic)[alpha] > 127).mean()
    maps = materials.build_pbr_maps(baked, 0.6, style, reference=cutout)

    b = np.asarray(baked) / 255
    used = b.max(-1) > 0.02
    albedo = np.asarray(maps.albedo) / 255
    ref_brightness = (np.asarray(ref) / 255)[alpha].mean()
    assert abs(albedo[used].mean() - ref_brightness) < 0.25 * ref_brightness  # not dark & muddy
    assert maps.stats["colour_matched"]
    assert abs(maps.stats["metallic_coverage"] - ref_metal) < 0.2  # used to collapse to ~0
    assert maps.emissive_mask is not None  # the amber strips survive the bake
