"""Procedural hard-surface backend: modelled geometry from a silhouette."""

import numpy as np
import pytest
import trimesh
from PIL import Image

from meshify import hardsurface as hs
from meshify.backends import BACKENDS, get_backend
from meshify.config import GenerationRequest
from meshify.pipeline import mesh as mesh_prep
from meshify.pipeline import projection
from meshify.pipeline.run import generate


def rifle_mask(w=900, h=320):
    """Stock + receiver + magazine + a long thin barrel."""
    m = np.zeros((h, w), bool)
    m[80:200, 60:620] = True  # stock + receiver
    m[200:290, 380:450] = True  # magazine
    m[120:140, 620:860] = True  # barrel
    m[60:80, 300:360] = True  # sight block
    return m


def rifle_image(tmp_path):
    m = rifle_mask()
    rgb = np.zeros(m.shape + (3,), np.uint8) + 160  # plain grey studio background
    rgb[m] = (45, 45, 48)
    rgb[m & (np.arange(m.shape[1])[None] < 300)] = (110, 20, 22)  # oxblood stock
    rgb[135:140, 100:600] = (255, 170, 50)  # amber strip
    path = tmp_path / "rifle.png"
    Image.fromarray(rgb).save(path)
    return path


def test_registered_and_available():
    assert "procedural" in BACKENDS
    ok, reason = BACKENDS["procedural"].is_available()
    assert ok, reason
    assert BACKENDS["procedural"].info.commercial_ok


def test_build_splits_body_barrel_and_plates():
    style = hs.STYLES["dominion"]
    sil = hs.trace_silhouette(rifle_mask(), style)
    mesh, parts = hs.build(sil, style)
    kinds = {p.kind for p in parts}
    assert {"body", "thin", "plate"} <= kinds
    barrel = [p for p in parts if p.kind == "thin"]
    assert any(p.polygon.bounds[2] > 0.95 for p in barrel)  # the barrel reaches the muzzle
    # The barrel is slimmer than the receiver; plates stand proud of it.
    body_depth = max(p.z1 - p.z0 for p in parts if p.kind == "body")
    assert max(p.z1 - p.z0 for p in barrel) < body_depth
    assert max(p.z1 for p in parts if p.kind == "plate") > body_depth / 2
    assert len(mesh.faces) < 15000  # game-ready, not a scan
    assert mesh.bounds[1][0] == pytest.approx(1.0, abs=0.01)


def test_uvs_in_atlas_and_mesh_matches_silhouette():
    style = hs.STYLES["dominion"]
    mask = rifle_mask()
    sil = hs.trace_silhouette(mask, style)
    mesh, _ = hs.build(sil, style)
    uv = mesh.visual.uv
    assert uv.min() >= 0 and uv.max() <= 1
    # Seen from the front, the model is the silhouette it was built from.
    iou, view, _ = projection.find_view(mesh, mask)
    assert iou > 0.9


def test_paint_uses_reference_colours():
    style = hs.STYLES["dominion"]
    mask = rifle_mask()
    rgba = np.zeros(mask.shape + (4,), np.uint8)
    rgba[mask] = (200, 30, 30, 255)
    sil = hs.trace_silhouette(mask, style)
    mesh, _ = hs.build(sil, style)
    tex = np.asarray(hs.paint_from_reference(mesh, sil, rgba, 512)).astype(int)
    uv = mesh.visual.uv
    px = np.clip((uv[:, 0] * 511).astype(int), 0, 511)
    py = np.clip(((1 - uv[:, 1]) * 511).astype(int), 0, 511)
    sampled = tex[py, px]
    assert (np.abs(sampled - (200, 30, 30)).max(1) < 40).mean() > 0.9


def test_clean_meshes_skip_floater_removal():
    style = hs.STYLES["dominion"]
    mesh, _ = hs.build(hs.trace_silhouette(rifle_mask(), style), style)
    assert mesh_prep.remove_floaters(mesh) is mesh  # plates are separate pieces on purpose


def test_end_to_end_procedural(tmp_path):
    req = GenerationRequest(image_path=str(rifle_image(tmp_path)), backend="procedural",
                            style_pack="sovereign_call", faction="dominion", category="weapon",
                            texture_size=512, asset_name="Proc Rifle")
    result = generate(req, output_root=tmp_path / "out")
    scene = trimesh.load(result.output_dir / result.files["glb"])
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    assert result.stats["faces"] < 15000
    assert max(mesh.extents) == pytest.approx(result.stats["size_m"]["x"], rel=0.01)
    assert result.stats["materials"]["projection_iou"] > 0.85
    assert get_backend("procedural").style_name == "dominion"
