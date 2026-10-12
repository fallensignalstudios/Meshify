"""Acceptance tests from docs/CAELIS_HANDOFF.md (owner's GPU test session)."""
import io
import json
import os
import struct
import sys

import numpy as np
import pytest
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from test_projection import reference, slab  # noqa: E402

from meshify.pipeline import materials, projection  # noqa: E402
from meshify.styles import resolve  # noqa: E402

DOMINION = resolve("sovereign_call", "dominion", "weapon", "maintained").material
SIZE = 512


def texel(u, v):
    return int((1 - v) * (SIZE - 1)), int(u * (SIZE - 1))


def dull_bake(glow_in_bake=False):
    """Slab texture: dark steel everywhere the UVs reach, black padding
    elsewhere, no glow (the painter dulled it) unless asked."""
    mesh = slab()
    occ = materials.uv_occupancy(mesh, (SIZE, SIZE), dilate=0)
    tex = np.zeros((SIZE, SIZE, 3), np.float32)
    tex[occ] = (0.10, 0.10, 0.11)
    if glow_in_bake:  # a real glow the bake kept, on the back face
        r0, c0 = texel(0.3, 0.30)
        r1, c1 = texel(0.7, 0.24)
        tex[r0:r1, c0:c1] = (1.0, 0.67, 0.2)
    return mesh, Image.fromarray((tex * 255).astype(np.uint8)), occ


def build(strength, glow_in_bake=False, project=True):
    mesh, baked, occ = dull_bake(glow_in_bake)
    ref = reference()
    proj = projection.project(mesh, (SIZE, SIZE), ref) if project and strength > 0 else None
    return materials.build_pbr_maps(baked, 0.6, DOMINION, reference=ref, projection=proj,
                                    projection_strength=strength,
                                    occupancy=materials.uv_occupancy(mesh, (SIZE, SIZE))), proj


def strip_intensity(maps):
    """Mean emissive intensity over the front-face texels under the reference's amber strip."""
    if maps.emissive_mask is None:
        return 0.0
    e = np.asarray(maps.emissive_mask, np.float32) / 255
    r0, c0 = texel(0.32, 0.79)
    r1, c1 = texel(0.68, 0.74)
    return float(e[r0:r1, c0:c1].mean())


# ---------------------------------------------------------------- Priority 1

def test_projected_glow_scales_monotonically_with_strength():
    levels = [strip_intensity(build(s)[0]) for s in (0.0, 0.01, 0.25, 0.85, 1.0)]
    assert levels[0] == 0.0  # absent at 0
    assert all(b > a for a, b in zip(levels, levels[1:])), levels  # strictly increasing
    assert levels[1] >= materials.EMISSIVE_EPS  # 0.01 still survives 8-bit export


def test_low_strength_glow_is_kept_and_reported():
    maps, _ = build(0.01)
    assert maps.emissive_mask is not None  # not dropped for being below 0.5
    assert maps.stats["emissive_coverage"] > 0 and 0 < maps.stats["emissive_intensity"] < 0.05


def test_hidden_and_edge_on_texels_get_no_projected_glow():
    maps, proj = build(1.0)
    e = np.asarray(maps.emissive_mask, np.float32) / 255
    edges = e[texel(0, 0.03)[0]:, :]  # the thin edge strip at the bottom of the atlas
    assert edges.max() <= materials.EMISSIVE_EPS


def test_bake_glow_survives_with_projection_disabled():
    maps, _ = build(0.0, glow_in_bake=True)
    assert maps.emissive_mask is not None
    e = np.asarray(maps.emissive_mask, np.float32) / 255
    r, c = texel(0.5, 0.27)
    assert e[r, c] > 0.5


def test_glb_keeps_emissive_texture_and_factor(tmp_path):
    maps, _ = build(0.85)
    mesh = slab()
    materials.apply_pbr_maps(mesh, maps)
    path = tmp_path / "rifle.glb"
    mesh.export(path)
    data = path.read_bytes()
    gltf = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
    mat = gltf["materials"][0]
    assert "emissiveTexture" in mat and mat["emissiveFactor"] == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------- Priority 2

def test_fully_occupied_dark_steel_panel_keeps_metal():
    """60% Dominion dark steel + 40% grey, all real surface: nothing is padding."""
    tex = np.zeros((200, 200, 3), np.uint8)
    tex[:120] = (0x1a, 0x1a, 0x1c)
    tex[120:] = (128, 128, 128)
    img = Image.fromarray(tex)
    full = np.ones((200, 200), bool)
    for occ in (full, None):  # with UV occupancy, and the texture-only fallback
        maps = materials.build_pbr_maps(img, 0.6, DOMINION, occupancy=occ)
        metal = np.asarray(maps.metallic, np.float32) / 255
        assert metal[20:100, 20:180].mean() > 0.8, "dark steel panel lost its metalness"


def test_padding_identical_to_a_panel_is_still_excluded():
    mesh = slab()
    occ = materials.uv_occupancy(mesh, (SIZE, SIZE))
    tex = np.full((SIZE, SIZE, 3), 0.10, np.float32)  # padding == panel colour everywhere
    mask = materials.texture_mask(tex, occ)
    assert mask[texel(0.5, 0.76)]  # on the front island
    assert not mask[texel(0.5, 0.50)]  # in the gap between front and back islands


def test_hidden_and_edge_on_islands_count_as_surface():
    mesh = slab()
    occ = materials.uv_occupancy(mesh, (SIZE, SIZE))
    proj = projection.project(mesh, (SIZE, SIZE), reference())
    r, c = texel(0.1, 0.015)  # an edge strip: real surface, zero projection weight
    assert occ[r, c] and proj.weight[r, c] < 0.1
    tex = np.zeros((SIZE, SIZE, 3), np.float32)
    assert materials.texture_mask(tex, occ)[r, c]


def steel_reference() -> Image.Image:
    """Same side view as reference(), painted Dominion dark steel."""
    arr = np.asarray(reference()).copy()
    solid = arr[..., 3] > 0
    arr[solid, :3] = (0x1c, 0x1c, 0x1e)
    arr[solid, :3] = np.clip(arr[solid, :3] + np.random.default_rng(0).integers(-3, 4, (solid.sum(), 3)), 0, 255)
    return Image.fromarray(arr, "RGBA")


@pytest.mark.parametrize("mode", ["disabled", "skipped"])
def test_occupancy_works_without_projection(mode):
    mesh, baked, _ = dull_bake()
    occ = materials.uv_occupancy(mesh, (SIZE, SIZE))
    ref = steel_reference()
    if mode == "skipped":  # poor silhouette match -> project() returns None
        ball = trimesh.creation.icosphere(subdivisions=2)
        ball.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((len(ball.vertices), 2)))
        assert projection.project(ball, (SIZE, SIZE), Image.new("RGBA", (900, 100), (200, 0, 0, 255))) is None
    maps = materials.build_pbr_maps(baked, 0.6, DOMINION, reference=ref, projection=None,
                                    occupancy=occ)
    metal = np.asarray(maps.metallic, np.float32) / 255
    assert metal[texel(0.5, 0.76)] > 0.5  # dark steel island stays metal
    assert metal[texel(0.5, 0.50)] == 0  # padding gets nothing
    assert maps.stats["colour_matched"]  # colour correction still runs


# ------------------------------------------------------------- Normal maps

def test_flat_painted_panels_get_flat_normals():
    rng = np.random.default_rng(0)
    tex = np.full((256, 256, 3), 0.35, np.float32) + rng.normal(0, 0.015, (256, 256, 3))  # grainy flat paint
    tex[100:104, :] = 0.05  # one real panel line
    maps = materials.build_pbr_maps(Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8)), 0.6, DOMINION)
    n = np.asarray(maps.normal_gl).astype(int)
    flat = n[20:80, 20:236]
    assert np.abs(flat - (128, 128, 255)).max(axis=-1).mean() < 3  # grain gone
    assert np.abs(n[96:108, 20:236] - (128, 128, 255)).max() > 20  # panel line kept


def test_normal_strength_zero_gives_a_flat_normal_map():
    tex = (np.random.default_rng(1).random((128, 128, 3)) * 255).astype(np.uint8)
    maps = materials.build_pbr_maps(Image.fromarray(tex), 0.6, DOMINION, normal_strength=0.0)
    n = np.asarray(maps.normal_gl).astype(int)
    assert np.abs(n - (128, 128, 255)).max() <= 1
