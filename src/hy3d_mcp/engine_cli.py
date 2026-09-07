"""Drive Hunyuan3D-2 shape generation under the engine interpreter.

Runs as a subprocess of the MCP server, never imported by it: the engine
venv carries torch/CUDA and the server venv does not, so the only thing
crossing between them is argv and stdout.

Shape by default, texture on request. `--paint` runs Hunyuan3D's paint
pipeline over the generated mesh and emits a GLB with UVs and a baked albedo;
`--paint-mesh` skips generation and paints a mesh that already exists.

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

# With a texture pass appended, shape gets squeezed into the front of the bar
# and paint owns the rest. Painting is ~65s against shape's ~115s, so the
# split is not even: P_SHAPE_END is where the shape stage's 95% lands.
P_SHAPE_END = 60.0

# Upstream defaults to 2048, which is 4x the buffer area across six camera
# views and does not fit on an 8GB card -- see docs/paint-spike-2026-09-06.md.
# 1024 measured 6.76 GiB peak against a 6.96 GiB ceiling at 40k faces.
DEFAULT_TEXTURE_SIZE = 1024
PAINT_SUB = "hunyuan3d-paint-v2-0-turbo"

# Which dit subfolder belongs to which repo. The pair has to move together:
# only one of the three names its subfolder after itself, and a mismatched
# pair is not a loud failure -- smart_load_model simply reports a path that
# does not exist, several frames away from the choice that caused it.
DITS = {
    "tencent/Hunyuan3D-2": "hunyuan3d-dit-v2-0",
    "tencent/Hunyuan3D-2mini": "hunyuan3d-dit-v2-mini",
    "tencent/Hunyuan3D-2mv": "hunyuan3d-dit-v2-mv",
}
DEFAULT_MODEL = "tencent/Hunyuan3D-2"
MV_MODEL = "tencent/Hunyuan3D-2mv"


# What fraction of the bar the current stage owns. The shape stage reports
# 0-100 whether or not it is the whole run, so appending a paint pass is a
# matter of rescaling here rather than threading a span through every emit.
SCALE = {"base": 0.0, "span": 100.0}


def emit(pct, message):
    """One progress line in the form the server's parser expects."""
    print("[%3d%%] %s" % (int(SCALE["base"] + pct * SCALE["span"] / 100.0),
                          message), flush=True)


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


def glb_material(path):
    """The PBR factors as written, read back from the file.

    Asserting a material fix in memory is not the same as shipping one: the
    paint pipeline returns a SimpleMaterial whose metallicFactor assignment
    goes nowhere, and the exporter then substitutes its own defaults. That
    failure is invisible from the mesh object and obvious from the file, so
    read the file.
    """
    try:
        import pygltflib
        g = pygltflib.GLTF2().load(str(path))
        if not g.materials:
            return None
        pbr = g.materials[0].pbrMetallicRoughness
        if pbr is None:
            return None
        # glTF's own defaults when a factor is absent, so the numbers here
        # are what a renderer will use rather than what the file spells out.
        return {"metallic": 1.0 if pbr.metallicFactor is None
                            else round(float(pbr.metallicFactor), 3),
                "roughness": 1.0 if pbr.roughnessFactor is None
                             else round(float(pbr.roughnessFactor), 3),
                "base_color_texture": pbr.baseColorTexture is not None}
    except Exception as e:
        return {"error": str(e)}


def watertight_probe(mesh):
    """A positionally merged copy, for topology questions only.

    See the note at the ``watertight`` stat. Never export this: the merge
    keeps one arbitrary UV per position and so breaks the texture.
    """
    probe = mesh.copy()
    probe.merge_vertices(merge_tex=True, merge_norm=True)
    return probe


def run_paint(mesh, image, texture_size):
    """Texture a mesh in place of its bare geometry. Returns (mesh, load_s, paint_s).

    The caller must have dropped every reference to the shape pipeline before
    calling this -- see the teardown in main().
    """
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()

    # diffusers >= 0.34 gates any custom_pipeline behind trust_remote_code, and
    # upstream predates the gate. The code being trusted is the file in the
    # engine checkout this process already imported from, not a hub download.
    from diffusers import DiffusionPipeline
    original = DiffusionPipeline.from_pretrained.__func__
    DiffusionPipeline.from_pretrained = classmethod(
        lambda cls, *a, **kw: original(cls, *a, **{"trust_remote_code": True, **kw}))

    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    emit(2, "loading the paint pipeline")
    t = time.time()
    pipe = Hunyuan3DPaintPipeline.from_pretrained(DEFAULT_MODEL, subfolder=PAINT_SUB)
    # render_size is read at call time, but MeshRender takes its resolutions at
    # construction -- so setting the config alone would leave the renderer at
    # 2048 and silently disagree with the images fed into it.
    pipe.config.render_size = pipe.config.texture_size = texture_size
    pipe.render.set_default_render_resolution(texture_size)
    pipe.render.set_default_texture_resolution(texture_size)
    # Not a lever: the delight and multiview pipelines are ~7.4 GiB of fp16
    # weights between them, more than the card holds before a single activation.
    pipe.enable_model_cpu_offload()
    load_s = time.time() - t

    emit(20, "painting at %d (six views)" % texture_size)
    t = time.time()
    painted = pipe(mesh, image)
    paint_s = time.time() - t
    emit(95, "painted in %.0fs" % paint_s)
    return painted, round(load_s, 1), round(paint_s, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="the front view, and the only required one")
    ap.add_argument("-o", "--output", default="out.glb")
    ap.add_argument("--model", default=None,
                    help="default: %s, or %s once a second view is given"
                         % (DEFAULT_MODEL, MV_MODEL))
    ap.add_argument("--subfolder", default=None,
                    help="default: whichever dit subfolder DITS pairs with the "
                         "chosen model")
    # front is the positional; these are the other three the mv conditioner
    # knows about, and any one of them switches the run to multiview.
    for tag in ("left", "back", "right"):
        ap.add_argument("--view-" + tag, metavar="IMAGE",
                        help="the %s view (multiview run)" % tag)
    ap.add_argument("--paint", action="store_true",
                    help="run the texture pass over the generated mesh; the "
                         "output then carries UVs and a baked albedo")
    ap.add_argument("--paint-mesh", metavar="GLB",
                    help="paint this existing mesh instead of generating one. "
                         "The positional image is still required -- it is the "
                         "concept the texture is drawn from.")
    ap.add_argument("--texture-size", type=int, default=DEFAULT_TEXTURE_SIZE,
                    help="render and texture resolution for the paint pass "
                         "(default %d; upstream's 2048 does not fit in 8GB)"
                         % DEFAULT_TEXTURE_SIZE)
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
                    help="faster VAE decode path; try only after a baseline "
                         "run. Swaps in a turbo VAE from a subfolder install.sh "
                         "does not fetch, so the first use downloads it.")
    ap.add_argument("--engine", default=os.environ.get(
        "HY3D_ENGINE_REPO", str(Path.home() / "git/repos/Hunyuan3D-2")))
    args = ap.parse_args()

    sys.path.insert(0, args.engine)
    import torch
    from PIL import Image

    painting = bool(args.paint or args.paint_mesh)
    if painting:
        SCALE["span"] = P_SHAPE_END

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable -- check /usr/lib/wsl/lib is on the loader path")

    free0, total = torch.cuda.mem_get_info()
    smi0, smi_total = nvidia_smi_used()
    print("[gpu] %s  free %.2f / %.2f GiB%s"
          % (torch.cuda.get_device_name(0), free0 / GIB, total / GIB,
             "  (nvidia-smi used %d MiB)" % smi0 if smi0 is not None else ""))

    remover = {"it": None}

    def load_view(path, tag):
        """One view, keyed if it needs it.

        Upstream's minimal_demo converts to RGBA and *then* tests mode ==
        'RGB', so its background removal never runs. Test the source image
        instead, and treat an all-opaque alpha channel as no alpha. The
        remover is built once and shared: it loads an onnx session, and four
        views would otherwise pay for four of them.
        """
        src = Image.open(path)
        has_alpha = (src.mode in ("RGBA", "LA")
                     and src.getchannel("A").getextrema()[0] < 255)
        img = src.convert("RGBA")
        if has_alpha:
            print("[input] %s: alpha present -- skipping background removal" % tag)
            return img
        emit(1, "removing background (%s)" % tag)
        if remover["it"] is None:
            from hy3dgen.rembg import BackgroundRemover
            remover["it"] = BackgroundRemover()
        return remover["it"](img)

    paths = {"front": args.image}
    for tag in ("left", "back", "right"):
        given = getattr(args, "view_" + tag)
        if given:
            paths[tag] = given
    multiview = len(paths) > 1
    images = {tag: load_view(path, tag) for tag, path in paths.items()}
    # MVImageProcessorV2 keys on exactly front/left/back/right and sorts by its
    # own view index, so a subset is fine but a renaming is not. A single-image
    # run hands over the bare image, because the plain v2 processor takes one.
    image = images if multiview else images["front"]

    if args.paint_mesh:
        # Painting a mesh someone else made: no shape pipeline, no diffusion,
        # and the stats below have nothing to report for either.
        import trimesh
        model = subfolder = shape_pipe = None
        load_s = gen_s = None
        reduce_s = None
        emit(50, "loading %s" % Path(args.paint_mesh).name)
        mesh = trimesh.load(args.paint_mesh, force="mesh")
        raw_faces = int(len(mesh.faces))
        emit(100, "%d faces" % raw_faces)
    else:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        model = args.model or (MV_MODEL if multiview else DEFAULT_MODEL)
        if multiview and model != MV_MODEL:
            # Switching under a caller who named a model is worth doing and worth
            # saying: the other two repos ship a single-image conditioner, so their
            # only way to honour four views is to ignore three of them.
            print("[model] %s has no multiview conditioner -- using %s instead"
                  % (model, MV_MODEL))
            model = MV_MODEL
        subfolder = args.subfolder or DITS.get(model, "hunyuan3d-dit-v2-0")
        print("[model] %s / %s%s"
              % (model, subfolder,
                 "  views: " + ",".join(images) if multiview else ""))
        emit(2, "loading %s" % model)
        t = time.time()
        # A "missing keys" warning for the VAE encoder is expected and benign:
        # the bundled checkpoint ships a decoder-only VAE and loads strict=False.
        pipe = shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model, subfolder=subfolder, use_safetensors=True, variant="fp16")
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

    paint_s = paint_load_s = None
    if painting:
        SCALE["base"], SCALE["span"] = P_SHAPE_END, 100.0 - P_SHAPE_END
        # Surrender the shape pipeline before the paint models load. Not
        # tidiness: the caching allocator is still holding ~6 GiB of shape
        # blocks, and paint wants 6.8 GiB of its own. Measured on a 3060 Ti,
        # painting without this teardown took 157s and reserved 11.57 GiB
        # against a 6.96 GiB ceiling -- a spill rather than an error, because
        # WDDM serves the overflow from host RAM. With it: 63s and 6.76 GiB,
        # matching a cold run. Both names have to go; dropping one inside the
        # callee leaves the other holding every block.
        pipe = shape_pipe = None
        mesh, paint_load_s, paint_s = run_paint(
            mesh, images["front"], args.texture_size)

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
    if painting:
        # trimesh's PBR default is metallicFactor 1.0, which renders a baked
        # albedo as near-black in Godot until an environment map saves it. The
        # texture the paint pass produces is diffuse colour, so say so.
        #
        # Setting the factors on what the paint pipeline returns is not
        # enough. It hands back a SimpleMaterial, which has no metallicFactor
        # field, so the assignment lands silently on an unused attribute and
        # the GLB exporter's own to_pbr() writes the 1.0 default anyway --
        # alongside a roughness of 0.9036, which is (2/(glossiness+2))**0.25
        # and the tell that this conversion ran. Convert here instead, and
        # assign the result back: to_pbr() returns a new object.
        visual = mesh.visual
        material = getattr(visual, "material", None)
        if material is not None and not hasattr(material, "metallicFactor"):
            material = material.to_pbr()
            visual.material = material
        if material is not None:
            material.metallicFactor = 0.0
            material.roughnessFactor = 1.0
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
        # Painting rewraps the UVs, and every atlas seam splits the vertices
        # along it: the same 40,000 faces arrive carrying 24,929 vertices
        # instead of 20,002. The surface did not change, but is_watertight
        # asks whether faces share vertices, so it reports False on a mesh
        # that is still closed -- merging by position restores watertight,
        # euler 2, one body, no broken faces. Ask a merged copy, so the
        # number describes the geometry rather than the atlas. The copy is
        # thrown away: merging in place would collapse exactly the duplicate
        # UVs the texture is painted against. Gated on the paint flag rather
        # than run unconditionally: shape-only output has no UVs to split, so
        # the probe there could only be a no-op or a new way for the default
        # path to fail after it has already written a good GLB.
        "watertight": bool(watertight_probe(mesh).is_watertight if painting
                           else mesh.is_watertight),
        "multiview": multiview,
        "views": list(images),
        "glb_attributes": glb_attributes(out),
        "glb_material": glb_material(out) if painting else None,
        "textured": painting,
        "load_s": round(load_s, 1) if load_s is not None else None,
        "generate_s": round(gen_s, 1) if gen_s is not None else None,
        "reduce_s": reduce_s,
        "paint_load_s": paint_load_s,
        "paint_s": paint_s,
        "peak_torch_alloc_gib": round(peak, 2),
        "peak_torch_reserved_gib": round(reserved, 2),
        "resident_after_gib": round(resident, 2),
        "free_at_baseline_gib": round(ceiling, 2),
        "nvidia_smi_used_mib": smi1,
        "settings": {"model": model, "steps": args.steps,
                     "octree_resolution": args.octree_resolution,
                     "guidance_scale": args.guidance_scale, "seed": args.seed,
                     "cpu_offload": args.cpu_offload, "flashvdm": args.flashvdm,
                     "texture_size": args.texture_size if painting else None},
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
