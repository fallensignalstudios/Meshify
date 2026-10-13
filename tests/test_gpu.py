from meshify import gpu
from meshify.pipeline import images

RTX_5070 = {"name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "compute_capability": "12.0",
            "torch": "2.7.0", "cuda": "12.8"}


def test_low_vram_detected_on_12gb_card(monkeypatch):
    monkeypatch.delenv("MESHIFY_LOW_VRAM", raising=False)
    monkeypatch.setattr(gpu, "gpu_info", lambda: RTX_5070)
    assert gpu.low_vram()
    monkeypatch.setattr(gpu, "gpu_info", lambda: {**RTX_5070, "vram_gb": 24.0})
    assert not gpu.low_vram()


def test_low_vram_env_override(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_info", lambda: None)
    monkeypatch.setenv("MESHIFY_LOW_VRAM", "1")
    assert gpu.low_vram()
    monkeypatch.setenv("MESHIFY_LOW_VRAM", "0")
    assert not gpu.low_vram()


def test_blackwell_needs_cuda_12_8(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_info", lambda: RTX_5070)
    assert gpu.blackwell_torch_problem() is None
    monkeypatch.setattr(gpu, "gpu_info", lambda: {**RTX_5070, "cuda": "12.1"})
    assert "CUDA 12.8" in gpu.blackwell_torch_problem()
    monkeypatch.setattr(gpu, "gpu_info", lambda: {**RTX_5070, "compute_capability": "8.9",
                                                   "cuda": "12.1"})
    assert gpu.blackwell_torch_problem() is None  # RTX 40-series is fine on 12.1


def test_text_to_image_model_follows_vram(monkeypatch):
    monkeypatch.delenv("MESHIFY_T2I_MODEL", raising=False)
    monkeypatch.setenv("MESHIFY_LOW_VRAM", "1")
    assert images.t2i_model() == images.T2I_MODEL_LOW_VRAM
    monkeypatch.setenv("MESHIFY_LOW_VRAM", "0")
    assert images.t2i_model() == images.T2I_MODEL_FULL
    monkeypatch.setenv("MESHIFY_T2I_MODEL", "my/model")
    assert images.t2i_model() == "my/model"
