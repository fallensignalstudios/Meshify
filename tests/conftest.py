import pytest


@pytest.fixture(autouse=True)
def no_real_judge(monkeypatch):
    """Keep tests independent of whether torch/transformers are installed: the
    CLIP judge is off unless a test switches it on with its own fake."""
    from meshify.pipeline import judge

    monkeypatch.setattr(judge, "available", lambda: False)
