import numpy as np
import pytest
from PIL import Image, ImageDraw

from meshify.config import GenerationRequest
from meshify.pipeline import guides, images
from meshify.pipeline.run import generate
from meshify.styles import load_packs, resolve


def _paths(names):
    from pathlib import Path
    return [Path(f"/lib/{n}.png") for n in names]


@pytest.mark.parametrize("prompt,expected", [
    ("heavy assault rifle", "assault_rifle"),
    ("imperial pistol", "sidearm"),
    ("heavy rotary cannon", "heavy_weapon"),
    ("sniper rifle", "marksman_rifle"),
    ("frag grenade", None),
])
def test_prompt_picks_the_matching_archetype(prompt, expected):
    lib = _paths(["assault_rifle", "heavy_weapon", "marksman_rifle", "sidearm"])
    chosen = guides.choose(lib, prompt)
    assert (chosen.stem if chosen else None) == expected


def test_bundled_guides_cover_every_faction():
    data = load_packs()["sovereign_call"]
    for faction in ["dominion", "reformation", "ancient", "aurelion", "eclipse"]:
        rifles = guides.list_guides(data, faction, "weapon")
        assert guides.choose(rifles, "rifle") is not None, faction
    assert guides.choose(resolve(None, "dominion", "hero_weapon", None).guides, "greatsword")


def test_control_image_is_a_fitted_outline():
    sil = Image.new("L", (300, 100), 255)
    ImageDraw.Draw(sil).rectangle([20, 30, 280, 70], fill=0)
    ctrl = np.asarray(guides.control_image(sil, 1344, 768).convert("L")) > 127
    assert ctrl.shape == (768, 1344)
    ys, xs = np.nonzero(ctrl)
    assert 0.8 < (xs.max() - xs.min()) / 1344 < 0.9  # fills ~85% of the frame width
    assert ctrl[384, 672] == 0  # an outline, not a filled shape


def test_tall_guides_lie_down_in_wide_frames():
    blade = Image.new("L", (60, 400), 255)
    ImageDraw.Draw(blade).rectangle([20, 10, 40, 390], fill=0)
    ctrl = np.asarray(guides.control_image(blade, 1344, 768).convert("L")) > 127
    ys, xs = np.nonzero(ctrl)
    assert (xs.max() - xs.min()) > 3 * (ys.max() - ys.min())


def _fake_text_to_image(seen):
    def fake(prompt, seed, realism, style=None, guide=None, guide_strength=0.6, **_):
        seen["guide"], seen["strength"] = guide, guide_strength
        img = Image.new("RGB", (400, 300), (128, 128, 128))
        ImageDraw.Draw(img).rectangle([80, 110, 320, 190], fill=(140, 98, 57))
        return img
    return fake


def test_auto_guide_reaches_the_generator(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(images, "text_to_image", _fake_text_to_image(seen))
    req = GenerationRequest(prompt="heavy assault rifle", backend="dummy", faction="dominion",
                            category="weapon", texture_size=512, guide_strength=0.7)
    result = generate(req, output_root=tmp_path)
    assert seen["guide"].size == (1344, 768) and seen["strength"] == 0.7
    assert result.stats["guide"] == "assault_rifle"
    assert (result.output_dir / result.files["guide"]).is_file()


def test_guide_can_be_turned_off_or_named(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(images, "text_to_image", _fake_text_to_image(seen))
    base = dict(prompt="heavy assault rifle", backend="dummy", faction="dominion",
                category="weapon", texture_size=512)
    generate(GenerationRequest(**base, guide="none"), output_root=tmp_path)
    assert seen["guide"] is None
    result = generate(GenerationRequest(**base, guide="heavy_weapon"), output_root=tmp_path)
    assert result.stats["guide"] == "heavy_weapon"
    with pytest.raises(ValueError, match="Unknown silhouette guide"):
        generate(GenerationRequest(**base, guide="laser_sword"), output_root=tmp_path)
