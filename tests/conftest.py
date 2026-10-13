import pytest


@pytest.fixture(autouse=True)
def no_real_judge(monkeypatch):
    """Keep tests independent of whether torch/transformers are installed: the
    CLIP judge is off unless a test switches it on with its own fake."""
    from meshify.pipeline import judge

    monkeypatch.setattr(judge, "available", lambda: False)


@pytest.fixture(autouse=True)
def flat_background(monkeypatch):
    """Use the built-in background keyer so tests don't depend on (or wait for) rembg."""
    monkeypatch.setenv("MESHIFY_BACKGROUND", "flat")
