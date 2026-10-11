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
    # Framing comes first so SDXL shows the whole weapon, not a close-up.
    assert subject.startswith("full-length side profile of a complete fictional sci-fi sidearm")
    assert "Reformation" in subject
    assert "campaign-worn" in look and "grey background" in look
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


def test_weapons_get_a_wide_frame_and_anti_closeup_negative():
    assert resolve(None, None, "weapon", None).aspect == "wide"
    assert resolve(None, None, "armor", None).aspect == "square"
    assert images.ASPECT_SIZES["wide"] == (1344, 768)
    assert "close-up" in images.build_negative(resolve(None, None, "weapon", None))


def _cutout(box):
    img = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle(box, fill=(120, 90, 60, 255))
    return img


def test_border_contact_detects_cropped_objects():
    assert images.border_contact(_cutout([60, 60, 340, 240])) == 0.0  # fully inside
    assert images.border_contact(_cutout([0, 40, 399, 299])) > 0.3  # runs off 3 edges


def test_cropped_reference_is_retried_with_next_seed(tmp_path, monkeypatch):
    seeds = []

    def fake_t2i(prompt, seed, realism, style=None, **_):
        seeds.append(seed)
        img = Image.new("RGB", (400, 300), (128, 128, 128))
        box = [0, 0, 399, 299] if len(seeds) == 1 else [80, 80, 320, 220]  # 1st cropped
        ImageDraw.Draw(img).rectangle(box, fill=(140, 98, 57))
        return img

    def fake_bg(img):
        rgba = np.asarray(img.convert("RGBA")).copy()
        rgba[..., 3] = np.where((rgba[..., :3] == 128).all(-1), 0, 255)
        return Image.fromarray(rgba, "RGBA")

    monkeypatch.setattr(images, "text_to_image", fake_t2i)
    monkeypatch.setattr(images, "remove_background", fake_bg)
    req = GenerationRequest(prompt="heavy assault rifle", backend="dummy", seed=7,
                            faction="dominion", category="weapon", texture_size=512)
    result = generate(req, output_root=tmp_path)
    assert seeds == [7, 8]
    assert result.stats["reference_attempts"] == 2


def test_dominion_weapons_use_their_own_language_and_avoid_real_guns():
    weapon = resolve(None, "dominion", "weapon", None)
    text = weapon.apply_to_prompt("heavy assault rifle")
    for word in ["blackened steel", "bronze trim", "oxblood", "amber", "monolithic"]:
        assert word in text
    # Colours lead the faction text so SDXL doesn't drop them.
    faction_text = weapon.prompt_parts[0]
    assert faction_text.index("bronze") < faction_text.index("monolithic")
    assert "AK-47" in weapon.negative and "wood furniture" in weapon.negative
    # The override only applies to weapons; Dominion architecture keeps the base look.
    arch = resolve(None, "dominion", "modular_architecture", None).apply_to_prompt("doorway")
    assert "Roman-Byzantine" in arch and "service weapon" not in arch


def test_frontier_is_campaign_worn_not_junkyard():
    frontier = resolve(None, None, None, "frontier").look_parts[0]
    assert "well maintained" in frontier and "repair welds" not in frontier
    assert "repair welds" in resolve(None, None, None, "derelict").look_parts[0]


def test_every_styled_prompt_fits_sdxl_token_limits():
    pack = next(p for p in list_packs() if p["id"] == "sovereign_call")
    for faction in pack["factions"]:
        for category in pack["categories"]:
            for condition in pack["conditions"]:
                style = resolve("sovereign_call", faction, category, condition)
                halves = [*images.build_prompt_pair("heavy assault rifle", True, style),
                          *images.build_negative_pair(style)]
                for half in halves:
                    assert images.approx_clip_tokens(half) <= images.CLIP_TOKEN_LIMIT, \
                        (faction, category, condition, half)


def test_negative_split_keeps_framing_negatives_on_second_encoder():
    first, second = images.build_negative_pair(resolve(None, "dominion", "weapon", "frontier"))
    assert "AK-47" in first and "close-up" in second and "text" in second
    assert not set(first.split(", ")) & set(second.split(", "))


def test_sdxl_negatives_stay_split_between_encoders():
    # Regression: a misplaced for/else used to overwrite the split negative.
    kwargs = images.build_t2i_kwargs("SG161222/RealVisXL_V5.0", "heavy assault rifle", True,
                                     resolve(None, "dominion", "weapon", "frontier"))
    assert "AK-47" in kwargs["negative_prompt"] and "close-up" not in kwargs["negative_prompt"]
    assert "close-up" in kwargs["negative_prompt_2"]
    assert (kwargs["width"], kwargs["height"]) == (1344, 768)


def test_flux_gets_one_prompt_and_no_negative():
    kwargs = images.build_t2i_kwargs("black-forest-labs/FLUX.1-schnell", "rifle", True,
                                     resolve(None, "dominion", "weapon", None))
    assert "prompt_2" not in kwargs and "negative_prompt" not in kwargs
    assert kwargs["num_inference_steps"] == 4


def test_pack_reference_image_resolves_relative_to_pack(tmp_path, monkeypatch):
    from meshify import styles

    (tmp_path / "references").mkdir()
    Image.new("RGB", (8, 8), (140, 98, 57)).save(tmp_path / "references" / "dom.png")
    (tmp_path / "test.toml").write_text(
        '[pack]\nid = "t"\n[factions.dom]\nprompt = "x"\nreference = "references/dom.png"\n'
        '[factions.dom.overrides.weapon]\nreference = "references/missing.png"\n'
        '[categories.weapon]\nprompt = "{subject}"\n[categories.prop]\nprompt = "{subject}"\n')
    monkeypatch.setenv("MESHIFY_STYLE_DIR", str(tmp_path))
    styles.load_packs.cache_clear()
    try:
        assert resolve("t", "dom", "prop", None).reference == tmp_path / "references" / "dom.png"
        assert resolve("t", "dom", "weapon", None).reference is None  # missing file: skipped
    finally:
        styles.load_packs.cache_clear()


def test_style_reference_is_passed_to_text_to_image(tmp_path, monkeypatch):
    seen = {}
    ref = tmp_path / "concept.png"
    Image.new("RGB", (64, 64), (140, 98, 57)).save(ref)

    def fake_t2i(prompt, seed, realism, style=None, style_image=None, style_strength=0.6):
        seen["image"], seen["strength"] = style_image, style_strength
        img = Image.new("RGB", (400, 300), (128, 128, 128))
        ImageDraw.Draw(img).rectangle([80, 80, 320, 220], fill=(140, 98, 57))
        return img

    monkeypatch.setattr(images, "text_to_image", fake_t2i)
    req = GenerationRequest(prompt="rifle", backend="dummy", faction="dominion", category="weapon",
                            style_image_path=str(ref), style_strength=0.8, texture_size=512)
    result = generate(req, output_root=tmp_path / "out")
    assert seen["image"].size == (64, 64) and seen["strength"] == 0.8
    assert (result.output_dir / result.files["style_reference"]).is_file()
