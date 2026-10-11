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
| `MESHIFY_T2I_MODEL` | `black-forest-labs/FLUX.1-schnell` | Any diffusers text-to-image model, e.g. an SDXL photoreal checkpoint |
| `MESHIFY_TRELLIS_MODEL` | `JeffreyXiang/TRELLIS-image-large` | |
| `MESHIFY_HUNYUAN_MODEL` | `tencent/Hunyuan3D-2` | |

## Testing without a GPU

```bash
pip install -e ".[dev]" && pytest
meshify generate -i photo.png -b dummy
```

The `dummy` backend makes a textured blob so you can test the whole pipeline
(PBR maps, scaling, export, UI) on any machine.
