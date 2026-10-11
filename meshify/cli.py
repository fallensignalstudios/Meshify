"""Command line entry point: `meshify generate ...`, `meshify serve`, `meshify backends`."""

from __future__ import annotations

import argparse
import logging
import sys


def _cmd_generate(args: argparse.Namespace) -> int:
    from .config import GenerationRequest
    from .pipeline.run import generate

    req = GenerationRequest(
        prompt=args.prompt,
        image_path=args.image,
        backend=args.backend,
        seed=args.seed,
        realism=not args.no_realism,
        delight_strength=args.delight,
        texture_size=args.texture_size,
        target_faces=None if args.faces == 0 else args.faces,
        real_world_height_m=args.height,
        asset_name=args.name,
    )
    result = generate(req)
    print(f"\nDone in {result.stats['seconds']}s -> {result.output_dir}")
    for label, fname in result.files.items():
        print(f"  {label:10s} {fname}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Meshify UI: http://{args.host}:{args.port}")
    uvicorn.run("meshify.server:app", host=args.host, port=args.port)
    return 0


def _cmd_backends(_: argparse.Namespace) -> int:
    from .backends import list_backends
    from .gpu import blackwell_torch_problem, gpu_info, low_vram

    info = gpu_info()
    if info:
        mode = "low-VRAM mode" if low_vram() else "full mode"
        print(f"GPU: {info['name']} ({info['vram_gb']} GB, compute {info['compute_capability']}, "
              f"torch {info['torch']} / CUDA {info['cuda']}) - {mode}")
        problem = blackwell_torch_problem()
        if problem:
            print(f"WARNING: {problem}")
    else:
        print("GPU: none detected (only the dummy backend will work)")
    print()
    for b in list_backends():
        status = "ready" if b["available"] else b["reason"]
        commercial = "commercial OK" if b["commercial_ok"] else "check license"
        print(f"{b['name']:10s} {status}\n           {b['license']} - {commercial}\n"
              f"           {b['notes']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from .config import DEFAULT_BACKEND

    parser = argparse.ArgumentParser(prog="meshify", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a 3D asset from a prompt and/or image")
    g.add_argument("prompt", nargs="?", help="Text prompt (used if no --image)")
    g.add_argument("--image", "-i", help="Reference image (photo or render)")
    g.add_argument("--backend", "-b", default=DEFAULT_BACKEND)
    g.add_argument("--name", "-n", default="asset", help="Asset name used in file names")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--no-realism", action="store_true", help="Skip photoreal shaping + PBR pass")
    g.add_argument("--delight", type=float, default=0.6, help="0..1, remove baked lighting")
    g.add_argument("--texture-size", type=int, default=2048, choices=[512, 1024, 2048, 4096])
    g.add_argument("--faces", type=int, default=50_000, help="Face budget (0 = full density)")
    g.add_argument("--height", type=float, help="Real-world height in metres, e.g. 1.8")
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("serve", help="Start the web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.set_defaults(func=_cmd_serve)

    b = sub.add_parser("backends", help="List 3D model backends and their licenses")
    b.set_defaults(func=_cmd_backends)

    args = parser.parse_args(argv)
    if args.cmd == "generate" and not args.prompt and not args.image:
        parser.error("generate needs a prompt or --image")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
