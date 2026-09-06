"""Drive Hunyuan3D-2 shape generation under the engine interpreter.

Runs as a subprocess of the MCP server, never imported by it: the engine
venv carries torch/CUDA and the server venv does not, so the only thing
crossing between them is argv and stdout.

Shape-only, by design rather than by omission. The paint stage wants more
VRAM than an 8GB card has, so this emits an untextured GLB and leaves
material work to the caller's DCC or engine.

Two output contracts the server depends on:

  * progress lines of the form `[ NN%] message` on stdout, which the
    server's `_PROGRESS_LINE` parser relays to the MCP client. The budget
    is deliberately lopsided -- diffusion owns 8-40% and volume decoding
    40-95% -- because decoding is roughly twice the wall-clock of
    diffusion at octree 384. A bar that hit 100% a third of the way in
    would look hung for the remaining two thirds.
  * a single-line JSON object as the LAST line of stdout.

The instrumentation exists because on WSL2 the failure mode is *thrashing,
not OOM*: WDDM satisfies CUDA allocations past VRAM out of system RAM, so
an oversized job completes rather than raising -- it just crawls at PCIe
bandwidth. "It finished" is not evidence it fit.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GIB = 1024 ** 3

# Where each stage's slice of the progress bar starts and ends. Measured at
# octree 384 on a 3060 Ti: ~36s diffusion against ~84s decode.
P_LOADED, P_DIFFUSION_END, P_DECODE_END = 8.0, 40.0, 95.0


def emit(pct, message):
    """One progress line in the form the server's parser expects."""
    print("[%3d%%] %s" % (int(pct), message), flush=True)


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

    Shape-only output legitimately has no TEXCOORD_0 -- there are no UVs
    without a texture stage. Only a missing NORMAL is a defect.
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
        return ["<unreadable: %s>" % e]


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
    ap.add_argument("--max-faces", type=int, default=0,
                    help="decimate to this face count (0 = off). Raw output is "
                         "~700k faces, which is not a game-ready mesh. Needs "
                         "libopengl0 installed or pymeshlab's io plugins fail.")
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
    print("[gpu] %s  free %.2f / %.2f GiB%s"
          % (torch.cuda.get_device_name(0), free0 / GIB, total / GIB,
             "  (nvidia-smi used %d MiB)" % smi0 if smi0 is not None else ""))

    # Upstream's minimal_demo converts to RGBA and *then* tests mode == 'RGB',
    # so its background removal never runs. Test the source image instead, and
    # treat an all-opaque alpha channel as no alpha.
    src = Image.open(args.image)
    has_alpha = src.mode in ("RGBA", "LA") and src.getchannel("A").getextrema()[0] < 255
    image = src.convert("RGBA")
    if not has_alpha:
        emit(1, "removing background")
        from hy3dgen.rembg import BackgroundRemover
        image = BackgroundRemover()(image)
    else:
        print("[input] alpha present -- skipping background removal")

    subfolder = args.subfolder or (
        "hunyuan3d-dit-v2-mini" if "mini" in args.model else "hunyuan3d-dit-v2-0")
    emit(2, "loading %s" % args.model)
    t = time.time()
    # A "missing keys" warning for the VAE encoder is expected and benign:
    # the bundled checkpoint ships a decoder-only VAE and loads strict=False.
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model, subfolder=subfolder, use_safetensors=True, variant="fp16")
    if args.flashvdm:
        pipe.enable_flashvdm()
    if args.cpu_offload:
        # Upstream's enable_model_cpu_offload() and _execution_device were
        # lifted from diffusers' DiffusionPipeline without the base class that
        # provides `.components`, so both raise AttributeError as shipped.
        # Supply the mapping ourselves; the keys must match the names in
        # model_cpu_offload_seq ("conditioner->model->vae").
        if not hasattr(type(pipe), "components"):
            type(pipe).components = property(lambda self: {
                "conditioner": self.conditioner,
                "model": self.model,
                "vae": self.vae,
            })
        pipe.enable_model_cpu_offload()
        # Second half of the same incomplete lift: enable_model_cpu_offload()
        # moves the pipeline to CPU, and __call__ then reads `self.device` --
        # a plain attribute, now "cpu" -- to place latents and timesteps. The
        # hooked modules still execute on the GPU, so the sampler dies with
        # "found at least two devices, cuda:0 and cpu". `_execution_device`
        # exists for exactly this and is never used; restore the attribute.
        pipe.device = torch.device("cuda")
    load_s = time.time() - t
    after_load = (free0 - torch.cuda.mem_get_info()[0]) / GIB
    emit(P_LOADED, "loaded in %.0fs, %.2f GiB resident" % (load_s, after_load))

    # The denoising loop is the only part of pipe() that can report from the
    # inside. Volume decoding runs after it, still inside the same call, so
    # the last step hands the bar over to the heartbeat at P_DIFFUSION_END
    # and the decode's own tqdm goes to stderr where nothing parses it.
    span = P_DIFFUSION_END - P_LOADED
    done = {"n": 0}

    def on_step(step_idx, t_, outputs):
        # `outputs` holds scheduler tensors; touching it here would cost a
        # device sync per step for nothing. Count invocations instead --
        # step_idx is divided by the scheduler order and need not be dense.
        done["n"] += 1
        n = done["n"]
        emit(P_LOADED + span * min(n / max(args.steps, 1), 1.0),
             "diffusion step %d/%d" % (n, args.steps))
        if n >= args.steps:
            emit(P_DIFFUSION_END,
                 "decoding volume at octree %d" % args.octree_resolution)

    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    mesh = pipe(
        image=image,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        octree_resolution=args.octree_resolution,
        generator=torch.manual_seed(args.seed),
        callback=on_step,
        # Required, not merely advisory: the loop evaluates `i %
        # callback_steps` whenever a callback is set, and the default None
        # makes that a TypeError on the first step.
        callback_steps=1,
    )[0]
    gen_s = time.time() - t

    raw_faces = int(len(mesh.faces))
    reduce_s = None
    if args.max_faces and raw_faces > args.max_faces:
        from hy3dgen.shapegen import FaceReducer, FloaterRemover
        emit(P_DECODE_END, "decimating %d -> %d faces" % (raw_faces, args.max_faces))
        t = time.time()
        mesh = FaceReducer()(FloaterRemover()(mesh), max_facenum=args.max_faces)
        reduce_s = round(time.time() - t, 1)
    else:
        emit(P_DECODE_END, "%d faces" % raw_faces)

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
    # include_normals is not cosmetic. trimesh's GLB writer emits NORMAL only
    # if vertex_normals has been materialized, so a bare export() yields a
    # POSITION-only file; Godot does not synthesize normals and lights such a
    # mesh off one constant vector, which reads as a broken material rather
    # than a missing attribute.
    mesh.export(str(out), include_normals=True)
    emit(100, "wrote %s" % out.name)

    # The thrash signature: torch *reserved* more than the card can hold, so
    # the remainder was served from host RAM over PCIe.
    ceiling = free0 / GIB
    spilled = reserved >= ceiling - 0.10

    stats = {
        "output": str(out),
        "bytes": out.stat().st_size,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "raw_faces": raw_faces,
        "watertight": bool(mesh.is_watertight),
        "glb_attributes": glb_attributes(out),
        "load_s": round(load_s, 1),
        "generate_s": round(gen_s, 1),
        "reduce_s": reduce_s,
        "peak_torch_alloc_gib": round(peak, 2),
        "peak_torch_reserved_gib": round(reserved, 2),
        "resident_after_gib": round(resident, 2),
        "free_at_baseline_gib": round(ceiling, 2),
        "nvidia_smi_used_mib": smi1,
        "settings": {"model": args.model, "steps": args.steps,
                     "octree_resolution": args.octree_resolution,
                     "guidance_scale": args.guidance_scale, "seed": args.seed,
                     "cpu_offload": args.cpu_offload, "flashvdm": args.flashvdm},
    }
    if spilled:
        stats["warning"] = (
            "torch reserved %.2f GiB against a %.2f GiB ceiling -- this run "
            "likely spilled to host RAM and ran at PCIe speed. Retry with "
            "cpu_offload=True, or a lower octree, and compare generate_s."
            % (reserved, ceiling))
    # Last line of stdout, single line: the server reads it back as the
    # result contract.
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
