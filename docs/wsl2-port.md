# Running the engine on WSL2 + NVIDIA

Notes from porting this server off Apple Silicon / MLX onto Windows WSL2 with
an 8GB RTX 3060 Ti, using upstream Tencent Hunyuan3D-2 (PyTorch/CUDA).

Everything here is about the *engine and its environment*, not about one
machine. Machine-specific measurements live in the working notes.

## The pipeline is shape-only, by hardware

Hunyuan3D 2.1 wants ~10GB for shape and ~21GB for texture. Texture generation
is out of reach on an 8GB card, and system RAM does not rescue it: the
constraint is peak activation memory during multiview attention, not layer
residency, so offloading does not help that stage the way it helps shape.

Consequence for the build: **`custom_rasterizer` and `DifferentiableRenderer`
are never compiled.** Both are texture-stage components. Skipping them also
skips the compilation step that breaks most Windows and WSL installs.

## Do not install an NVIDIA driver inside WSL

The Windows driver is projected into the guest through `/usr/lib/wsl/lib`.
Installing a Linux driver inside the distro clobbers that projection and
breaks CUDA. Install CUDA-enabled PyTorch wheels only.

## VRAM does not fail the way you expect: it thrashes, not OOMs

This is the single most expensive thing to learn the hard way, and it is
specific to WSL2's WDDM model.

**WDDM satisfies CUDA allocations past physical VRAM out of system RAM.**
Allocation therefore does not raise when you exceed the card. It silently
migrates to host memory and keeps going.

Two consequences:

1. **Allocate-until-`OutOfMemoryError` does not measure VRAM.** On an 8GB card
   it will happily report a ceiling in the hundreds of gigabytes, because the
   loop never hard-fails until it has eaten the host's RAM. Measure instead by
   reading `torch.cuda.mem_get_info()` after each allocation and taking the
   plateau in *device-resident* bytes. Do not stop the walk on a low
   device-free reading either — that just reports your own threshold back at
   you. Stop when resident bytes stop climbing.

2. **A completed job is not evidence that it fit.** A stage that overshoots
   will not crash; it will crawl at PCIe bandwidth. Judge fit by memory
   counters and wall-clock, never by exit status.

Once a context has spilled, `torch.cuda.empty_cache()` itself raises
`RuntimeError: CUDA error: out of memory`, and the context stays wedged until
the process exits. Budget a process restart between jobs rather than trying to
reclaim in-process.

### Reading the three memory numbers

They are not interchangeable, and conflating them produces confident wrong
conclusions:

| | what it means |
|---|---|
| `max_memory_allocated` | bytes torch handed to tensors — the floor of true demand |
| `max_memory_reserved` | bytes torch asked the driver for — **this is the spill signal** |
| `mem_get_info` delta | device-resident bytes, including blocks the caching allocator is holding after freeing them |

Resident drifts up toward the whole free pool even on runs that fit
comfortably, because the caching allocator does not hand memory back. Reading
resident-at-ceiling as "we spilled" is a false positive. When the two
disagree, settle it empirically: re-run with `enable_model_cpu_offload()` and
compare wall-clock. If offload is *slower*, the baseline was not spilling.

## Dependency traps

- **`scikit-image` is required but commented out of upstream
  `requirements.txt`.** The default surface extractor is `MCSurfaceExtractor`
  (`shapegen/models/autoencoders/surface_extractors.py`), which calls
  `skimage.measure.marching_cubes`. Without it, every shape run fails.
- **Pin `numpy<2`.** pymeshlab wheels of this era are built against the
  numpy 1.x ABI and fail at *import*, not at use.
- **`diso` is optional.** It is needed only for `mc_algo='dmc'`. The default
  path does not touch it.
- **pymeshlab needs `libOpenGL.so.0` even headless.** Ubuntu ships
  `libGL.so.1` and `libGLX.so.0` but not `libOpenGL.so.0`; without it
  pymeshlab loads with plugin errors and `io_base` is among the casualties,
  which makes `FaceReducer` / `FloaterRemover` fail with a misleading
  `PyMeshLabException: Unknown format for load: ply`. Fix with
  `sudo apt install libopengl0`.

## `enable_model_cpu_offload()` is broken as shipped — in two places

`Hunyuan3DDiTPipeline` declares `model_cpu_offload_seq = "conditioner->model->vae"`
and carries `enable_model_cpu_offload()` and `_execution_device` lifted from
diffusers' `DiffusionPipeline`, but not the base class those depend on. Calling
it fails twice over:

1. Both methods read `self.components`, which is **never defined anywhere in
   the file**. `AttributeError` before anything happens. Supply a dict mapping
   `"conditioner"` / `"model"` / `"vae"` to the modules — the keys must match
   `model_cpu_offload_seq`.
2. Past that, `enable_model_cpu_offload()` moves the pipeline to CPU, and
   `__call__` then reads `self.device` — a plain attribute, now `"cpu"` — to
   place latents and timesteps, while the hooked modules still execute on the
   GPU. The sampler dies with *"Expected all tensors to be on the same device,
   but found at least two devices, cuda:0 and cpu"*. `_execution_device` exists
   for precisely this and is never called. Restore `pipe.device` to `cuda`
   after enabling.

Both shims are in `scripts/shape_cli.py` behind `--cpu-offload`.

Worth the trouble: on this hardware offload cost **+2% wall-clock** while
cutting GPU footprint by more than half. It is close to a free lever for
pushing octree resolution up, not a last resort.

Note that offload changes device placement enough to perturb the result —
same seed and input gave a slightly different mesh (351,324 vs 350,791
vertices, and watertight vs not). Treat it as a different configuration for
reproducibility, not a transparent optimization.

## Exporting a GLB Godot can light

trimesh's GLB writer emits a `NORMAL` accessor only if the mesh's
`vertex_normals` cache has been materialized. A bare `mesh.export(path)` on a
freshly generated mesh therefore writes **`POSITION` and nothing else**.

Godot does not synthesize normals. It lights such a mesh off a single constant
vector, which presents as a broken *material* rather than a missing attribute —
expensive to diagnose from the symptom.

Pass `include_normals=True` (or read `mesh.vertex_normals` before exporting).

Note the shape-only file also has no `TEXCOORD_0`. That is correct — there are
no UVs without a texture stage. Only a missing `NORMAL` is a defect.

## Input doctrine (unchanged from the MLX build, still true)

Naturally lit concept art; the model de-lights internally, so pre-flattened
albedo-style input bakes pale. No drop shadows — they reconstruct as literal
geometry. Single object, plain background, roughly centered, ¾ view.

Note that upstream's `minimal_demo.py` converts the input to RGBA and *then*
tests `image.mode == 'RGB'` to decide whether to run background removal, so its
background removal never runs. Test the source image's mode before converting,
and treat an all-opaque alpha channel as no alpha.
