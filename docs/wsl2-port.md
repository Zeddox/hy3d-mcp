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
cutting GPU footprint by more than half, which makes it a practical lever for
pushing octree resolution up rather than a last resort.

**It is not, however, output-neutral.** Same seed and input, offload produced
350,791 vertices and a **non-watertight** mesh where baseline produced 351,324
and watertight. Baseline is deterministic across runs, so this is offload's
device placement changing the result, not sampler noise. Watertightness gates
mesh cleanup, physics and boolean operations downstream, so treat offload as a
distinct configuration and check the result rather than assuming equivalence.

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

## What the shape stage does and does not give you

It reproduces **silhouette and major forms** well. It does **not** reproduce
surface relief — carved ornament, panel tracery, engraved detail come out
smooth, and lit or open panels come out solid.

Raising `octree_resolution` does not recover it. 512 costs roughly 2x the
wall-clock of 384 and yields no additional relief; it changes tessellation
density, not what the model represents. Reach for octree when thin struts fuse
together, not as a detail dial.

Plan for ornament to come from normal or displacement maps in the texture
step, not from geometry.

Decimation is not optional for a game target: raw output is 0.6-1.0M faces.
For 3D printing, prefer a higher budget or the undecimated mesh — 40k faces is
a game budget, and quadric decimation is tuned to preserve silhouette, not the
fine surface a print resolves.
`FaceReducer()(FloaterRemover()(mesh), max_facenum=40000)` takes ~10-16s and
stayed watertight on every mesh tested.

## Check enclosed volume, not just watertightness

`is_watertight` passes on meshes that are nothing alike. On the same concept
image, 2.0 and 2mini produced meshes with the same silhouette, the same
bounding box, both watertight, both single-bodied, both genus 0 — and volumes
that differed by 5.6x.

2mini had produced a **thin-walled hollow form** rather than a solid: the wall
wraps around at the bottom, so it is a single closed genus-0 surface enclosing
a cavity. Horizontal cross-sections tell them apart immediately, where the
silhouette cannot:

| at mid-body | 2.0 | 2mini |
|---|---|---|
| section bounding box | 0.575 x 0.566 | 0.583 x 0.569 |
| section area | 0.3243 | 0.0249 |
| **fill of that box** | **99.7%** | **7.5%** |
| section perimeter | 2.239 | 4.436 |

Renders will not show this — you never see inside a closed mesh. It matters for
generated collision shapes, for anything boolean, and decisively for 3D
printing, where it is the difference between a solid model and a shell with
walls near nozzle width.

Cheap screen: compare `mesh.volume` against `mesh.convex_hull.volume`, or take
a section and compare its area to its own bounding box. Surface area rising
while volume falls is the signature.

## Isolating a subject from real concept art

Concept art in the wild is usually a *scene*, not a single object on a plain
background. A plain-background colour key will correctly refuse such an image;
`rembg`/u2net handles it, but tends to keep a cast shadow fragment and any
object touching the subject.

Taking only the **largest connected component** of the alpha mask (dilated a
few pixels so the anti-aliased rim survives), then cropping to the alpha bbox
and padding square, cleaned this up reliably. Cast shadows are worth the effort
to remove: they reconstruct as literal geometry.

## Input doctrine (unchanged from the MLX build, still true)

Naturally lit concept art; the model de-lights internally, so pre-flattened
albedo-style input bakes pale. No drop shadows — they reconstruct as literal
geometry. Single object, plain background, roughly centered, ¾ view.

Note that upstream's `minimal_demo.py` converts the input to RGBA and *then*
tests `image.mode == 'RGB'` to decide whether to run background removal, so its
background removal never runs. Test the source image's mode before converting,
and treat an all-opaque alpha channel as no alpha.

## Phase 2: what the fork-and-swap actually touched

The MCP layer survived the platform change almost intact. What changed was
the bottom of it — the process the server spawns — and the honesty of what
sits on top.

**The swap itself.** `generate_model` used to build argv for a Swift
binary and hand it a Metal library through the environment. It now builds
argv for `engine_cli.py` under the engine venv's interpreter, invoked by
absolute path. The driver lives inside the package rather than in
`scripts/` for one concrete reason: a wheel carries no repo root, so
anything the server needs to locate at runtime has to ship beside it.

**The progress protocol did not change, and that was the point.** The
server parses `[ NN%] message` lines off the child's stdout and relays them
as MCP progress notifications, with a heartbeat underneath so a slow job
and a hung one do not look identical to the client. Upstream's pipeline
supports a `callback` — undocumented and unused by its own demos — so the
new driver emits exactly the lines the existing parser already understood.
Two details matter:

* `callback_steps=1` is mandatory, not advisory. The loop evaluates
  `i % callback_steps` whenever a callback is set, and the default `None`
  makes that a `TypeError` on the first step.
* The budget has to be lopsided. Diffusion owns 8–40% and volume decoding
  40–95%, because decoding is roughly twice diffusion's wall-clock at
  octree 384. Giving diffusion the whole bar would park it at 100% for the
  majority of the job — precisely the "slow job looks hung" case the
  heartbeat exists to prevent.

**Absolute paths are load-bearing.** The engine child runs with `cwd` set
to the Hunyuan3D checkout so its own imports resolve. Any relative path
handed in by a caller therefore resolves against the checkout, not the
user's directory. Every tool now resolves input and output paths before
they reach a subprocess.

**One venv, not two.** The Apple build separated a worker venv from the
engine so mesh and image work would not load MLX. `provision-engine.sh`
already builds an environment carrying numpy, PIL, trimesh, scipy, cv2,
pygltflib and pyrender, so `HY3D_PY` now defaults to the engine venv.
The split stays available through the env var; it just no longer names a
venv that nothing creates.

**Headless rendering needs `PYOPENGL_PLATFORM=egl` set before pyrender is
imported.** The platform is read at import time. Without it on headless
Linux, pyrender does not fail with anything resembling "no display" — it
fails with `Attempt to retrieve context when no valid context` from deep
inside the draw call, which reads as a corrupt mesh rather than a missing
context.

**Tools that cannot work now say so.** `paint_mesh` raises a sentence
explaining that texturing needs more VRAM than the card has, rather than
being removed (an unknown-tool error teaches the caller nothing) or left to
fail obscurely. `setup_engine` returns the one command to run instead of
half-honouring a contract — dry-run-first, seven resumable phases — that
`provision-engine.sh` does not yet implement. `finish_model` still works,
but only on GLBs textured elsewhere and round-tripped back through.

**The skill had to change with the server.** `SKILL.md` described the
pipeline as "shape + PBR paint" and told the agent never to decimate. With
texturing gone and `max_faces` defaulting to 40k, an unrevised skill would
have had the agent promising colour it would not get and refusing a step
the server now performs by default. A swap that leaves the caller's
instructions describing the old engine is not finished.

### Measured, end to end through the MCP tool

Concept `pagoda-clean.png`, defaults, seed 42:

| | |
|---|---|
| wall clock | 185s (of which ~35s model load, ~115s generate, ~11s decimate) |
| raw faces | 546,196 |
| after decimation | 40,000 faces / 20,002 verts, still watertight |
| peak reserved | 6.22 GiB against a 6.96 GiB ceiling |
| GLB attributes | `POSITION`, `NORMAL` |
| STL at 120mm | 59.5 × 59.2 × 120.0 mm, 101.5 cm³, 24.0% of bounding box |

The model reloads on every call — each generation is a fresh subprocess —
so ~35s of that 185s is fixed overhead per job, not per session.
