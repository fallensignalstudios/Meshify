"""Reference-image stage: text -> photoreal image, background removal, framing.

Realism starts here. Image-to-3D models copy the style of their input, so a
stylized reference gives a stylized mesh. We push the text-to-image model
toward neutral, evenly lit "product photo" images, which reconstruct well
and carry less baked-in lighting into the texture.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import re
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ..gpu import free_cuda, low_vram

if TYPE_CHECKING:
    from ..styles import ResolvedStyle

# FLUX.1-schnell (Apache-2.0) is the best open photoreal model, but its 12B
# transformer doesn't fit in under ~16 GB. On smaller cards we default to
# RealVisXL (SDXL-based, CreativeML OpenRAIL++-M), which is very photoreal and fast.
log = logging.getLogger(__name__)

T2I_MODEL_FULL = "black-forest-labs/FLUX.1-schnell"
T2I_MODEL_LOW_VRAM = "SG161222/RealVisXL_V5.0"


def t2i_model() -> str:
    return os.environ.get("MESHIFY_T2I_MODEL") or (
        T2I_MODEL_LOW_VRAM if low_vram() else T2I_MODEL_FULL)

REALISM_SUFFIX = (
    "photorealistic studio product photograph, single object, whole object in frame, "
    "centered, three-quarter view, soft even diffuse lighting, no harsh shadows, "
    "plain neutral grey background, physically accurate materials, real-world wear "
    "and surface detail, sharp focus, 85mm lens, high resolution"
)
STYLIZED_TERMS = (
    "cartoon", "stylized", "anime", "low poly", "lowpoly", "toon", "chibi", "pixar",
    "cel shaded", "cel-shaded", "illustration", "clay render",
)
NEGATIVE_PROMPT = (
    "cartoon, stylized, anime, illustration, painting, drawing, cel shading, low poly, "
    "toy, plastic look, multiple objects, cropped, cut off, text, letters, logo, watermark, "
    "dramatic lighting, hard shadows, busy background, close-up, macro, detail shot, "
    "partial view, out of frame"
)

# SDXL/FLUX-native resolutions. Long objects get a wide frame so they aren't cropped.
ASPECT_SIZES = {"square": (1024, 1024), "wide": (1344, 768), "tall": (768, 1344)}


# Shorter framing used with style packs, leaving token room for faction detail.
STYLE_FRAMING = (
    "single object centered, entire object in frame with empty margin around it, "
    "plain neutral grey background, "
    "soft even studio lighting, physically based materials, sharp focus"
)


def build_prompt(prompt: str, realism: bool, style: "ResolvedStyle | None" = None) -> str:
    prompt = prompt.strip().rstrip(".")
    if style is not None:
        parts = [style.apply_to_prompt(prompt)]
        if realism:
            parts += [style.render, STYLE_FRAMING]
        return ", ".join(p for p in parts if p)
    if not realism:
        return prompt
    lowered = prompt.lower()
    if any(term in lowered for term in STYLIZED_TERMS):
        # The user explicitly asked for a style; don't fight them.
        return prompt
    return f"{prompt}, {REALISM_SUFFIX}"


CLIP_TOKEN_LIMIT = 75  # 77 minus start/end tokens


def approx_clip_tokens(text: str) -> int:
    """Conservative CLIP token estimate (never under-counted the real BPE
    tokenizer on any Sovereign Call prompt). Used to warn before SDXL truncates."""
    words = re.findall(r"[A-Za-z]+|\d|[^\sA-Za-z\d]", text)
    return sum(max(1, math.ceil(len(w) / 7)) if w[0].isalpha() else 1 for w in words)


def build_prompt_pair(prompt: str, realism: bool, style: "ResolvedStyle") -> tuple[str, str]:
    """Split a styled prompt across SDXL's two text encoders (77 tokens each).

    Encoder 1 gets what the object *is* (subject, faction, asset type); encoder
    2 gets how it looks and is framed (condition, render style, studio framing).
    """
    subject = ", ".join(p for p in [style.subject(prompt), *style.prompt_parts] if p)
    look = list(style.look_parts)
    if realism:
        look += [style.render, STYLE_FRAMING]
    return subject, ", ".join(p for p in look if p)


def _merge_terms(*chunks: str | None, exclude: str = "") -> str:
    terms: list[str] = []
    for chunk in chunks:
        for term in (chunk or "").split(","):
            term = term.strip()
            if term and term not in terms and term.lower() not in exclude:
                terms.append(term)
    return ", ".join(terms)


def build_negative(style: "ResolvedStyle | None" = None) -> str:
    # Never negate something the style asks for (e.g. "stylized" in "stylized realism").
    wanted = (style.render or "").lower() if style else ""
    return _merge_terms(style.negative if style else None, NEGATIVE_PROMPT, exclude=wanted)


def build_negative_pair(style: "ResolvedStyle") -> tuple[str, str]:
    """Split negatives across SDXL's encoders like the positive prompt: what this
    faction/asset type is NOT on encoder 1; pack-wide, quality and framing
    negatives on encoder 2. Together they're too long for one 77-token encoder."""
    wanted = (style.render or "").lower()
    first = _merge_terms(style.identity_negative, exclude=wanted)
    taken = {t.strip() for t in first.split(",")}
    rest = [t for t in f"{style.pack_negative},{NEGATIVE_PROMPT}".split(",")
            if t.strip() not in taken]
    return first, _merge_terms(*rest, exclude=wanted)


# IP-Adapter lets SDXL copy materials/colours from a style reference image.
# The ViT-H variant keeps the image encoder small (~1.2 GB) for 12 GB cards.
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHTS = os.environ.get("MESHIFY_IP_ADAPTER", "ip-adapter_sdxl_vit-h.safetensors")

_t2i_pipe = None
_t2i_key: tuple | None = None


def build_t2i_kwargs(model: str, prompt: str, realism: bool,
                     style: "ResolvedStyle | None" = None) -> dict:
    """Prompt, negative and size arguments for the diffusers call (no torch needed)."""
    width, height = ASPECT_SIZES.get(style.aspect if style else "square", ASPECT_SIZES["square"])
    kwargs: dict = {"width": width, "height": height}
    is_sdxl = "xl" in model.lower()
    if style is not None and is_sdxl:
        kwargs["prompt"], kwargs["prompt_2"] = build_prompt_pair(prompt, realism, style)
    else:
        # FLUX's T5 encoder reads up to 512 tokens, so it gets everything.
        kwargs["prompt"] = build_prompt(prompt, realism, style)

    if "schnell" in model.lower():
        # FLUX schnell is distilled: 4 steps, no CFG, no negative prompt.
        kwargs.update(num_inference_steps=4, guidance_scale=0.0)
        return kwargs

    kwargs.update(num_inference_steps=30, guidance_scale=5.0)
    if "prompt_2" in kwargs:
        kwargs["negative_prompt"], kwargs["negative_prompt_2"] = build_negative_pair(style)
    else:
        kwargs["negative_prompt"] = build_negative(style)
    if is_sdxl:
        for key in ("prompt", "prompt_2", "negative_prompt", "negative_prompt_2"):
            if key in kwargs and approx_clip_tokens(kwargs[key]) > CLIP_TOKEN_LIMIT:
                log.warning("%s may exceed SDXL's 77-token limit and get cut off; "
                            "shorten the prompt or the style pack text: %s", key, kwargs[key])
    return kwargs


def _load_pipeline(model: str, with_ip_adapter: bool):
    import torch
    from diffusers import AutoPipelineForText2Image

    is_flux = "flux" in model.lower()
    dtype = torch.bfloat16 if is_flux else torch.float16
    extra = {}
    if with_ip_adapter:
        from transformers import CLIPVisionModelWithProjection

        extra["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            IP_ADAPTER_REPO, subfolder="models/image_encoder", torch_dtype=dtype)
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype, **extra)
    if with_ip_adapter:
        pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="sdxl_models",
                             weight_name=IP_ADAPTER_WEIGHTS)
    if is_flux and low_vram():
        # Streams weights layer by layer: slow, but FLUX runs on 12 GB.
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def text_to_image(prompt: str, seed: int, realism: bool,
                  style: "ResolvedStyle | None" = None,
                  style_images: list[Image.Image] | None = None,
                  style_strength: float = 0.6) -> Image.Image:
    """Generate a reference image. Needs `pip install meshify[text]` and a GPU.

    `style_images` (SDXL only) lend their colours and materials via IP-Adapter;
    several images are blended, so the faction's shared look wins over any one
    image's quirks. `style_strength` 0..1 sets how strongly.
    """
    global _t2i_pipe, _t2i_key
    if importlib.util.find_spec("diffusers") is None:
        raise RuntimeError(
            "Text prompts need the text-to-image extra: pip install -e .[text] "
            "(or supply an image instead)."
        )
    import torch

    model = t2i_model()
    use_ip = bool(style_images)
    if use_ip and "xl" not in model.lower():
        log.warning("Style reference images need an SDXL model (current: %s); ignoring it.", model)
        use_ip = False
    key = (model, use_ip)
    if _t2i_pipe is None or _t2i_key != key:
        release_text_to_image()
        _t2i_pipe = _load_pipeline(model, use_ip)
        _t2i_key = key

    kwargs = build_t2i_kwargs(model, prompt, realism, style)
    kwargs["generator"] = torch.Generator("cpu").manual_seed(seed)
    if use_ip:
        # InstantStyle: inject the reference only into the style-carrying
        # attention block, so it lends colours/materials, not its exact shape.
        scale = {"up": {"block_0": [0.0, style_strength, 0.0]}}
        _t2i_pipe.set_ip_adapter_scale(scale)
        # One IP-Adapter, several images: [[img, img, ...]].
        kwargs["ip_adapter_image"] = [[img.convert("RGB") for img in style_images]]
    return _t2i_pipe(**kwargs).images[0]


def release_text_to_image() -> None:
    """Unload the text-to-image model so the 3D model gets the whole GPU."""
    global _t2i_pipe, _t2i_key
    if _t2i_pipe is not None:
        _t2i_pipe = None
        _t2i_key = None
        free_cuda()


def remove_background(image: Image.Image) -> Image.Image:
    """Return RGBA with the background removed. Keeps existing alpha if present."""
    image = image.convert("RGBA")
    alpha = np.asarray(image)[..., 3]
    if alpha.min() < 250:
        return image  # already has transparency
    if importlib.util.find_spec("rembg") is not None:
        from rembg import remove

        return remove(image).convert("RGBA")
    return _remove_flat_background(image)


def _remove_flat_background(image: Image.Image, tolerance: float = 28.0) -> Image.Image:
    """Fallback for plain studio backgrounds: key out the border colour."""
    rgba = np.asarray(image).astype(np.float32)
    rgb = rgba[..., :3]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb - bg, axis=-1)
    alpha = np.clip((dist - tolerance) / tolerance, 0, 1) * 255
    rgba[..., 3] = alpha
    return Image.fromarray(rgba.astype(np.uint8), "RGBA")


def border_contact(image: Image.Image, band: float = 0.01) -> float:
    """Fraction of the image border covered by the object (0 = fully inside the frame).

    A high value means the object runs off the edge, i.e. the reference is a
    cropped close-up and would give a partial 3D model.
    """
    alpha = np.asarray(image.convert("RGBA"))[..., 3] > 128
    h, w = alpha.shape
    b = max(1, int(round(min(h, w) * band)))
    edges = np.concatenate([alpha[:b].ravel(), alpha[-b:].ravel(),
                            alpha[:, :b].ravel(), alpha[:, -b:].ravel()])
    return float(edges.mean())


def frame_object(image: Image.Image, size: int = 518, fill: float = 0.85) -> Image.Image:
    """Crop to the object, pad to a square with margin, resize for the 3D model."""
    image = image.convert("RGBA")
    alpha = np.asarray(image)[..., 3]
    ys, xs = np.nonzero(alpha > 16)
    if len(xs) == 0:
        raise ValueError("Couldn't find an object in the image after background removal.")
    image = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = int(max(image.size) / fill)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)
