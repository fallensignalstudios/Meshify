"""Reference-image stage: text -> photoreal image, background removal, framing.

Realism starts here. Image-to-3D models copy the style of their input, so a
stylized reference gives a stylized mesh. We push the text-to-image model
toward neutral, evenly lit "product photo" images, which reconstruct well
and carry less baked-in lighting into the texture.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
from PIL import Image

T2I_MODEL = os.environ.get("MESHIFY_T2I_MODEL", "black-forest-labs/FLUX.1-schnell")

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
    "toy, plastic look, multiple objects, cropped, cut off, text, watermark, "
    "dramatic lighting, hard shadows, busy background"
)


def build_prompt(prompt: str, realism: bool) -> str:
    prompt = prompt.strip().rstrip(".")
    if not realism:
        return prompt
    lowered = prompt.lower()
    if any(term in lowered for term in STYLIZED_TERMS):
        # The user explicitly asked for a style; don't fight them.
        return prompt
    return f"{prompt}, {REALISM_SUFFIX}"


_t2i_pipe = None


def text_to_image(prompt: str, seed: int, realism: bool) -> Image.Image:
    """Generate a reference image. Needs `pip install meshify[text]` and a GPU."""
    global _t2i_pipe
    if importlib.util.find_spec("diffusers") is None:
        raise RuntimeError(
            "Text prompts need the text-to-image extra: pip install -e .[text] "
            "(or supply an image instead)."
        )
    import torch
    from diffusers import AutoPipelineForText2Image

    if _t2i_pipe is None:
        _t2i_pipe = AutoPipelineForText2Image.from_pretrained(T2I_MODEL, torch_dtype=torch.bfloat16)
        # Offloading keeps FLUX usable next to a 3D model on a single 24GB card.
        _t2i_pipe.enable_model_cpu_offload()

    full_prompt = build_prompt(prompt, realism)
    kwargs = {"prompt": full_prompt, "generator": torch.Generator("cpu").manual_seed(seed),
              "width": 1024, "height": 1024}
    if "schnell" in T2I_MODEL.lower():
        # FLUX schnell is distilled: 4 steps, no CFG, no negative prompt.
        kwargs.update(num_inference_steps=4, guidance_scale=0.0)
    else:
        kwargs.update(num_inference_steps=30, guidance_scale=5.0, negative_prompt=NEGATIVE_PROMPT)
    return _t2i_pipe(**kwargs).images[0]


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
