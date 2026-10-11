import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw

from meshify.config import GenerationRequest
from meshify.pipeline import images, materials
from meshify.pipeline.mesh import prepare_mesh
from meshify.pipeline.run import generate
from meshify.styles import list_packs, resolve

FACTIONS = ["dominion", "reformation", "ancient", "aurelion", "eclipse"]
TYPES = ["small_prop", "medium_prop", "industrial_prop", "modular_architecture",
         "architecture_hero", "weapon", "hero_weapon", "armor", "ship_component"]
CONDITIONS = ["pristine", "maintained", "frontier", "derelict"]


def test_sovereign_call_pack_is_complete():
    pack = next(p for p in list_packs() if p["id"] == "sovereign_call")
    assert set(pack["factions"]) == set(FACTIONS)
    assert set(pack["categories"]) == set(TYPES)
    assert set(pack["conditions"]) == set(CONDITIONS)


@pytest.mark.parametrize("faction", FACTIONS)
@pytest.mark.parametrize("category", TYPES)
def test_every_combination_resolves(faction, category):
    for condition in CONDITIONS:
        style = resolve("sovereign_call", faction, category, condition)
        assert style.tris and style.texture in (1024, 2048, 4096)
        assert style.material.emissive is not None


def test_budgets_follow_the_brief():
    assert resolve(None, None, "small_prop", None).tris == 3000
    assert resolve(None, None, "hero_weapon", None).texture == 4096
    modular = resolve(None, None, "modular_architecture", None)
    assert modular.pivot == "bottom_corner" and modular.grid_m == 1.0
    assert resolve(None, None, "weapon", None).size_axis == "longest"


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="faction"):
        resolve("sovereign_call", "rebels", None, None)
    with pytest.raises(ValueError, match="asset type"):
        resolve("sovereign_call", None, "spaceship", None)
    with pytest.raises(ValueError):
        GenerationRequest(prompt="x", faction="rebels").validate()


def test_styled_prompt_keeps_faction_and_no_text_negative():
    style = resolve(None, "reformation", "weapon", "frontier")
    subject, look = images.build_prompt_pair("sidearm", True, style)
    assert subject.startswith("sidearm, Reformation")
    assert "frontier weathering" in look and "grey background" in look
    negative = images.build_negative(style)
    assert "text" in negative and "gold" in negative
    assert "stylized" not in negative.split(", ")  # the brief asks for stylized realism


def _panel(base, channel=None):
    img = Image.new("RGB", (256, 256), base)
    if channel:
        ImageDraw.Draw(img).rectangle([40, 120, 216, 128], fill=channel)
    return img


def test_dominion_bronze_is_metallic_and_amber_channel_glows():
    style = resolve(None, "dominion", None, None).material
    maps = materials.build_pbr_maps(_panel((140, 98, 57), (255, 170, 51)), 0.0, style)
    metal = np.asarray(maps.metallic) / 255
    assert metal[30:90, 30:90].mean() > 0.8  # bronze panel
    assert maps.emissive_mask is not None
    mask = np.asarray(maps.emissive_mask) / 255
    assert mask[124, 128] > 0.5 and mask[30, 30] < 0.05  # channel only


def test_flat_ivory_is_not_mistaken_for_white_gold_glow():
    style = resolve(None, "ancient", None, None).material
    maps = materials.build_pbr_maps(_panel((250, 240, 215)), 0.0, style)
    assert maps.emissive_mask is None


def test_eclipse_has_no_metal_and_condition_changes_roughness():
    tex = _panel((25, 20, 30), (155, 77, 255))
    eclipse = materials.build_pbr_maps(tex, 0.0, resolve(None, "eclipse", None, None).material)
    assert np.asarray(eclipse.metallic).max() == 0
    pristine = materials.build_pbr_maps(tex, 0.0, resolve(None, "eclipse", None, "pristine").material)
    derelict = materials.build_pbr_maps(tex, 0.0, resolve(None, "eclipse", None, "derelict").material)
    assert np.asarray(derelict.roughness).mean() > np.asarray(pristine.roughness).mean() + 40


def test_modular_piece_snaps_to_grid_with_corner_pivot():
    mesh = trimesh.creation.box(extents=[2.3, 1.0, 0.4])
    mesh = prepare_mesh(mesh, None, 4.0, "height", "bottom_corner", grid_m=1.0)
    width = max(mesh.extents[0], mesh.extents[2])
    assert width == pytest.approx(round(width), abs=1e-6)
    assert np.allclose(mesh.bounds[0], 0, atol=1e-6)


def test_weapon_is_sized_by_length_and_centred():
    mesh = trimesh.creation.box(extents=[0.2, 0.1, 3.0])
    mesh = prepare_mesh(mesh, None, 1.1, "longest", "center")
    assert mesh.extents.max() == pytest.approx(1.1)
    assert np.allclose((mesh.bounds[0] + mesh.bounds[1]) / 2, 0, atol=1e-6)


def test_end_to_end_dominion_weapon(tmp_path):
    photo = tmp_path / "rifle.png"
    img = Image.new("RGB", (400, 400), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 170, 340, 230], fill=(140, 98, 57))
    draw.rectangle([90, 196, 310, 202], fill=(255, 170, 51))
    img.save(photo)
    req = GenerationRequest(image_path=str(photo), backend="dummy", faction="dominion",
                            category="weapon", condition="frontier", asset_name="DomRifle")
    result = generate(req, output_root=tmp_path / "out")
    assert result.stats["texture_size"] == 2048  # from the weapon preset
    for key in ["metallic", "orm", "normal_dx"]:
        assert (result.output_dir / result.files[key]).is_file()
    scene = trimesh.load(result.output_dir / result.files["glb"])
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    assert mesh.extents.max() == pytest.approx(1.0, rel=1e-3)  # weapon preset length
