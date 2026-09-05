"""Phase 1: drive Hunyuan3D-2 shape generation from the CLI, no MCP.

Shape-only. Produces an untextured GLB and reports the numbers that decide
whether this box can actually run the engine.

Why the instrumentation matters: on WSL2 the failure mode is *thrashing, not
OOM*. WDDM satisfies CUDA allocations past VRAM out of system RAM, so an
oversized job completes rather than raising -- it just crawls at PCIe
bandwidth. "It finished" is therefore not evidence it fit. Peak resident
bytes and wall-clock are.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024 ** 3


def nvidia_smi_used():
    """Windows-side view of the GPU, for cross-checking against torch."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
        used, total = (int(x) for x in out.strip().split("\n")[0].split(","))
        return used, total
    except Exception:
        return None, None


def glb_attributes(path):
    """Which vertex attributes actually made it into the file.

    The documented Godot gotcha is a GLB carrying POSITION but no NORMAL:
    Godot does not synthesize normals and lights the whole mesh off one
    constant vector, which presents as a bad material rather than a missing
    attribute. Cheap to check here, expensive to diagnose later.
    """
    try:
        import pygltflib
        g = pygltflib.GLTF2().load(str(path))
        attrs = set()
        for m in g.meshes or []:
            for p in m.primitives:
                # vars() lists every field of the Attributes dataclass; only
                # the non-None ones are actually present in the file.
                attrs |= {k for k, v in vars(p.attributes).items() if v is not None}
        return sorted(attrs)
    except Exception as e:
        return [f"<unreadable: {e}>"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("-o", "--output", default="out.glb")
    ap.add_argument("--model", default="tencent/Hunyuan3D-2",
                    help="'tencent/Hunyuan3D-2' (3.3B) or 'tencent/Hunyuan3D-2mini'")
    ap.add_argument("--subfolder", default=None,
                    help="default: hunyuan3d-dit-v2-0, or -v2-mini for 2mini")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=5.0)
    ap.add_argument("--octree-resolution", type=int, default=384)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu-offload", action="store_true",
                    help="conditioner->model->vae sequential offload; the lever "
                         "if peak sits at the VRAM ceiling")
    ap.add_argument("--flashvdm", action="store_true",
                    help="faster VAE decode path; try only after a baseline run")
    ap.add_argument("--engine", default=os.environ.get(
        "HY3D_ENGINE_REPO", str(Path.home() / "git/repos/Hunyuan3D-2")))
    args = ap.parse_args()

    sys.path.insert(0, args.engine)
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable -- check /usr/lib/wsl/lib is on the loader path")

    free0, total = torch.cuda.mem_get_info()
    smi0, smi_total = nvidia_smi_used()
    print(f"[gpu] {torch.cuda.get_device_name(0)}  "
          f"free {free0/GIB:.2f} / {total/GIB:.2f} GiB"
          + (f"  (nvidia-smi used {smi0} MiB)" if smi0 is not None else ""))

    # Upstream's minimal_demo converts to RGBA and *then* tests mode == 'RGB',
    # so its background removal never runs. Test the source image instead, and
    # treat an all-opaque alpha channel as no alpha.
    src = Image.open(args.image)
    has_alpha = src.mode in ("RGBA", "LA") and src.getchannel("A").getextrema()[0] < 255
    image = src.convert("RGBA")
    if not has_alpha:
        print("[input] no usable alpha -- running background removal")
        from hy3dgen.rembg import BackgroundRemover
        image = BackgroundRemover()(image)
    else:
        print("[input] alpha present -- skipping background removal")

    subfolder = args.subfolder or (
        "hunyuan3d-dit-v2-mini" if "mini" in args.model else "hunyuan3d-dit-v2-0")
    print(f"[load] {args.model} :: {subfolder}")
    t = time.time()
    # A "missing keys" warning for the VAE encoder is expected and benign:
    # the bundled checkpoint ships a decoder-only VAE and loads strict=False.
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model, subfolder=subfolder, use_safetensors=True, variant="fp16")
    if args.flashvdm:
        pipe.enable_flashvdm()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    load_s = time.time() - t
    after_load = (free0 - torch.cuda.mem_get_info()[0]) / GIB
    print(f"[load] {load_s:.1f}s, {after_load:.2f} GiB resident")

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    mesh = pipe(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        generator=torch.manual_seed(args.seed),
    )[0]
    gen_s = time.time() - t

    peak = torch.cuda.max_memory_allocated() / GIB
    # Reserved, not resident, is the spill signal. Resident includes blocks the
    # caching allocator is holding after freeing them, so it drifts up to the
    # ceiling on a run that fit comfortably. Reserved is what torch actually
    # asked the driver for -- exceeding the ceiling is what pushes into host RAM.
    reserved = torch.cuda.max_memory_reserved() / GIB
    resident = (free0 - torch.cuda.mem_get_info()[0]) / GIB
    smi1, _ = nvidia_smi_used()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))

    stats = {
        "output": str(out),
        "bytes": out.stat().st_size,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "glb_attributes": glb_attributes(out),
        "load_s": round(load_s, 1),
        "generate_s": round(gen_s, 1),
        "peak_torch_alloc_gib": round(peak, 2),
        "peak_torch_reserved_gib": round(reserved, 2),
        "resident_after_gib": round(resident, 2),
        "free_at_baseline_gib": round(free0 / GIB, 2),
        "nvidia_smi_used_mib": smi1,
        "settings": {"model": args.model, "steps": args.steps,
                     "octree_resolution": args.octree_resolution,
                     "guidance_scale": args.guidance_scale, "seed": args.seed,
                     "cpu_offload": args.cpu_offload, "flashvdm": args.flashvdm},
    }
    print(json.dumps(stats, indent=2))

    # The thrash signature: torch *reserved* more than the card can hold, so
    # the remainder was served from host RAM over PCIe.
    if reserved >= (free0 / GIB) - 0.10:
        print("\n[warn] torch reserved %.2f GiB against a %.2f GiB ceiling -- this "
              "run likely spilled to host RAM. Re-run with --cpu-offload and "
              "compare generate_s." % (reserved, free0 / GIB), file=sys.stderr)


if __name__ == "__main__":
    main()
