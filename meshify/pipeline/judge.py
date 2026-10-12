"""Candidate judge: pick the best of several text-to-image references.

Text-to-image is a dice roll: one seed gives a clean rifle, the next gives two
scopes. Meshify generates a few candidates and asks CLIP which one best
matches "a single well-proportioned rifle with one scope" rather than "a gun
with two scopes" or "a malformed weapon". The checks come from the style
pack's asset types (`good` / `bad`), with sensible defaults otherwise.

CLIP is good at "is this a clean, complete, single object" and decent at
obvious duplicates. It is not a precise counter, which is why the prompt and
negative prompt carry the same rules; the judge is the safety net.
"""

from __future__ import annotations

import importlib.util
import logging
import os

from PIL import Image

from ..gpu import free_cuda

log = logging.getLogger(__name__)

JUDGE_MODEL = os.environ.get("MESHIFY_JUDGE_MODEL", "openai/clip-vit-large-patch14")
DEFAULT_GOOD = ["a single complete {subject} on a plain background"]
DEFAULT_BAD = ["a cropped close-up of part of an object", "several objects", "text and logos"]

_model = None
_processor = None


def available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def _load():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _model = CLIPModel.from_pretrained(JUDGE_MODEL, torch_dtype=dtype).to(device).eval()
        _processor = CLIPProcessor.from_pretrained(JUDGE_MODEL)
    return _model, _processor


def score(images: list[Image.Image], good: list[str], bad: list[str]) -> list[float] | None:
    """Probability mass each image puts on the `good` descriptions (0..1).

    Returns None when the judge can't run (no torch/transformers), so callers
    fall back to the first acceptable candidate.
    """
    if not images or not good or not available():
        return None
    import torch

    try:
        model, processor = _load()
    except Exception as exc:  # e.g. no network for the first download
        log.warning("Candidate judge unavailable (%s); using the first good candidate.", exc)
        return None
    texts = [*good, *bad]
    inputs = processor(text=texts, images=[im.convert("RGB") for im in images],
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.no_grad():
        probs = model(**inputs).logits_per_image.float().softmax(dim=-1)
    return probs[:, :len(good)].sum(dim=-1).tolist()


def release() -> None:
    global _model, _processor
    if _model is not None:
        _model = _processor = None
        free_cuda()
