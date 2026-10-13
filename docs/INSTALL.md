# Installing Meshify on a GPU machine

Meshify itself is light. The 3D models it drives are the heavy part.

## Hardware

| Setup | VRAM | Notes |
|---|---|---|
| TRELLIS | ~16 GB | Default backend, MIT license |
| Hunyuan3D 2 (shape + texture) | ~16 GB | Shape-only needs ~6 GB |
| + FLUX.1-schnell for text prompts | +8-12 GB | Uses CPU offload, so it shares a 24 GB card fine |

An RTX 3090 / 4090 / 5090 (24 GB+) runs everything. A 16 GB card runs either
3D backend from a reference image. Run `nvidia-smi` to see what you have.

**12 GB cards (RTX 5070, 4070, 3060 12GB...)** work in **low-VRAM mode**, which
switches on automatically below ~15 GB (force it with `MESHIFY_LOW_VRAM=1`):

- Text-to-image uses RealVisXL (SDXL-based, photoreal) instead of FLUX, and is
  unloaded before the 3D model loads.
- Hunyuan3D uses its 0.6B "mini" shape model and offloads the texture painter
  to system RAM. This is the most reliable 12 GB setup.
- TRELLIS has no low-VRAM variant. It may fit at 12 GB, but expect
  out-of-memory errors on some inputs. Try `--texture-size 1024` if it does.
- Have 32 GB+ of system RAM; offloading parks models there.

**RTX 50-series (Blackwell)** cards need PyTorch built for **CUDA 12.8 or newer**
(PyTorch 2.7+). Older builds fail with `no kernel image is available`. Follow
the [RTX 50-series](#rtx-50-series-blackwell) steps below instead of the
default ones. `meshify backends` warns you if the versions don't match.

## Windows: use WSL2 (recommended)

TRELLIS builds several CUDA extensions that are painful on native Windows.
WSL2 gives you real Linux with full GPU access:

1. In an admin PowerShell: `wsl --install -d Ubuntu`, then reboot.
2. Install the normal **Windows** NVIDIA driver (do *not* install a driver inside WSL).
3. In Ubuntu, check the GPU is visible: `nvidia-smi`.
4. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

Everything below runs inside the Ubuntu terminal. `meshify serve` opens on
http://127.0.0.1:7860, and you can open that from your normal Windows browser.

## 1. Meshify

```bash
git clone https://github.com/fallensignalstudios/Meshify.git
cd Meshify
```

## 2. A 3D backend (pick one or both)

### TRELLIS (MIT - recommended)

```bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git ../TRELLIS
cd ../TRELLIS
. ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
conda activate trellis
cd ../Meshify
pip install -e ".[dev]"
# TRELLIS isn't a pip package; put the repo on the path:
export PYTHONPATH="$(realpath ../TRELLIS):$PYTHONPATH"
```

Add the `export PYTHONPATH=...` line to `~/.bashrc` so it sticks. If
`flash-attn` fails to build on your GPU, drop `--flash-attn` and set
`export ATTN_BACKEND=xformers`.

### Hunyuan3D 2 (read the license first)

The Tencent Hunyuan Community License excludes the EU, UK and South Korea
and limits large-scale commercial use. Check it fits before you ship assets.

```bash
git clone https://github.com/Tencent/Hunyuan3D-2.git ../Hunyuan3D-2
cd ../Hunyuan3D-2
pip install -r requirements.txt && pip install -e .
(cd hy3dgen/texgen/custom_rasterizer && python setup.py install)
(cd hy3dgen/texgen/differentiable_renderer && python setup.py install)
cd ../Meshify && pip install -e ".[dev]"
```

## RTX 50-series (Blackwell)

The upstream install scripts pin PyTorch versions that predate Blackwell, so
install a CUDA 12.8 PyTorch **first**, then build the extensions against it.
Do this inside WSL2 (see above).

Install the CUDA 12.8 toolkit in WSL2 (for compiling extensions):
https://developer.nvidia.com/cuda-downloads → Linux → x86_64 → WSL-Ubuntu.

```bash
conda create -n meshify python=3.11 -y && conda activate meshify
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
export TORCH_CUDA_ARCH_LIST="12.0"   # build CUDA extensions for Blackwell
export CUDA_HOME=/usr/local/cuda-12.8
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

That last line should print a 2.7+ version, `12.8` (or newer) and your GPU's name.

### Hunyuan3D on Blackwell (start here on a 12 GB card)

```bash
git clone https://github.com/Tencent/Hunyuan3D-2.git ../Hunyuan3D-2
cd ../Hunyuan3D-2
grep -v -E '^(torch|torchvision)' requirements.txt > /tmp/req.txt   # keep our cu128 torch
pip install -r /tmp/req.txt && pip install -e .
(cd hy3dgen/texgen/custom_rasterizer && python setup.py install)
(cd hy3dgen/texgen/differentiable_renderer && python setup.py install)
cd ../Meshify && pip install -e ".[dev,text]"
meshify backends
```

### TRELLIS on Blackwell (more experimental)

Run TRELLIS's setup **without** `--new-env` (that would install an old PyTorch):

```bash
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git ../TRELLIS
cd ../TRELLIS
pip install xformers --index-url https://download.pytorch.org/whl/cu128
. ./setup.sh --basic --diffoctreerast --spconv --mipgaussian --nvdiffrast
export ATTN_BACKEND=xformers
export PYTHONPATH="$(realpath .):$PYTHONPATH"
cd ../Meshify && pip install -e ".[dev,text]"
```

If a step fails, it's usually an extension (`spconv`, `kaolin`, `diffoctreerast`)
that doesn't have a Blackwell build yet. Copy the error and we'll sort it out.

## 3. Optional extras

```bash
pip install -e ".[text]"    # text prompts -> photoreal reference (FLUX.1-schnell, Apache-2.0)
pip install -e ".[rembg]"   # better background removal for real photos
```

## 4. Check and run

```bash
meshify backends            # shows what's installed and each license
meshify serve               # web UI on http://127.0.0.1:7860
```

Weights download from Hugging Face on first use (a few GB each), then they're cached.

## Useful environment variables

| Variable | Default | |
|---|---|---|
| `MESHIFY_BACKEND` | `trellis` | Default backend |
| `MESHIFY_OUTPUT_DIR` | `outputs` | Where jobs are written |
| `MESHIFY_LOW_VRAM` | auto (< 15 GB) | `1` / `0` to force low-VRAM mode on or off |
| `MESHIFY_T2I_MODEL` | FLUX.1-schnell, or RealVisXL in low-VRAM mode | Any diffusers text-to-image model |
| `MESHIFY_TRELLIS_MODEL` | `JeffreyXiang/TRELLIS-image-large` | |
| `MESHIFY_HUNYUAN_SHAPE` | `full`, or `mini` in low-VRAM mode | Hunyuan3D shape model size |

## Testing without a GPU

```bash
pip install -e ".[dev]" && pytest
meshify generate -i photo.png -b dummy
```

The `dummy` backend makes a textured blob so you can test the whole pipeline
(PBR maps, scaling, export, UI) on any machine.
