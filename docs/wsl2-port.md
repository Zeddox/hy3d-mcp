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

## Phase 3: what the worker port actually needed

The handoff called this phase "port the worker layer verbatim", and verbatim
turned out to be right for five of the six. Every worker was exercised under
the engine venv on this box before anything was changed:

| worker | verdict |
|---|---|
| `meshinfo.py` | works unchanged — 20,002 verts / 40,000 faces off accessor metadata |
| `normals.py` | works unchanged — injected `NORMAL` into a GLB stripped of it, and correctly no-ops on one that has it |
| `finish.py` | works unchanged on a synthesised textured GLB (49.9% accent coverage). Still unreachable from this build's own output, which has no albedo to tone |
| `preview.py` | works, five views in 1.3s at 512px — including a *textured* mesh |
| `tostl.py` | ported in Phase 2 |
| `cutout.py` | rewritten (below) |

Two findings worth recording.

**The pyopengl trap did not fire, and the check that guards it stays.** The
macOS installer fought pyrender's `pyopengl==3.1.0` pin, whose
`glGenTextures` wrapper cannot bind a texture against modern numpy — it
breaks textured renders only, so an untextured mesh hides it. This venv has
3.1.10 and renders textured meshes fine. That is not evidence the problem is
gone; it is evidence this venv was fixed by hand. `server_status`'s
`preview_ok >= 3.1.7` gate stays exactly as it is, because a fresh provision
is where pyrender would drag 3.1.0 back in.

**Rendering is software.** EGL comes up as `kms_swrast` under WSL2 (the
`libEGL warning: NEEDS EXTENSION` line on stderr is this, and is harmless).
Five 512px views in 1.3s is fast enough that hardware EGL is not worth
chasing.

### cutout.py: the fallback moved out of the engine

Phase 1 found that the corner-sampling key correctly refuses real concept art
— a garden scene with three lanterns, rocks, a pond and cast shadows keys at
corner std 43.0 against a 28.0 ceiling — and that rembg/u2net handles it, but
keeps a loose rock and part of a shadow. Dropping all but the largest
connected alpha component fixed both.

That two-step now lives in the worker. `cutout.py` tries the corner key, and
on refusal falls back to u2net + largest-component, and either way crops to
the subject and pads square. `--method corner` or `rembg` pins one key; the
refusal is then an error rather than a substitution, which is the point of
asking for it.

The framing is the part that matters, and it is why this belongs in the
worker rather than the engine. The engine's own rembg pass keys and stops:
no crop, no square pad. The generator frames its latent around the whole
image, so a subject occupying a third of the frame was being reconstructed
at a third of the available resolution. Measured end to end on the raw
garden scene, no manual prep:

    cutout (rembg+largest-component, 16.8% opaque, 1 stray island(s) dropped)
    shape
    decimate (542320 -> 40000)
    160.0s, 40,000 faces, watertight, 6.22 GiB peak reserved

The mesh that comes back is the lantern alone, correctly proportioned, with
no rock beside it.

Two consequences. `~/.u2net/u2net.onnx` (176MB) is now load-bearing rather
than incidental — it downloads silently on first use, so the installer
prefetches it and `server_status` checks for it. And the engine's own rembg
is now a safety net for one case only: a `HY3D_PY` pointed at an interpreter
that cannot import rembg. The stage line says so when it happens.

## Phase 4: the installer, and what it caught

The Phase 2 notes above say `setup_engine` returns a command instead of
running one, and that `provision-engine.sh` builds the environment. Both
are now superseded: `install.sh` is the CUDA installer, `provision-engine.sh`
is gone, and `setup_engine` runs the installer under the dry-run-first
contract it always promised.

**Writing the installer proved the environment was not reproducible.** The
discriminating check took one command — diff the packages the provisioner
names against the modules the workers actually import:

    pyrender       *** MISSING from installer ***
    PyOpenGL       *** MISSING from installer ***

Both were on this box because they were hand-installed while fixing EGL in
Phase 2, and `server_status` had been green ever since. A fresh provision
would have produced a box where `render_preview` could not rasterise, and
nothing in the setup path would have said so. Neither were the two apt
packages (`libopengl0`, `libegl1`) named anywhere. "It works here" is not
evidence that an installer is complete, and the only way to find that out
is to enumerate rather than to test.

That is also why the pyopengl-3.1.0 machinery inherited from the macOS
build stays. It looks stale — this venv has 3.1.10 and renders textured
meshes fine — but it looks stale *because someone fixed it by hand*. A
fresh install is exactly where pyrender drags its pin back in, so
`install.sh` still overrides it in a separate pass (resolving the two
together is reported unsatisfiable) and `server_status` still gates on
`>= 3.1.7`.

**Verify phase 6 proves, it does not import.** Three of this port's
failures were invisible to an import check: torch imports perfectly when it
is the CPU wheel and runs a hundred times slower; pymeshlab imports and
then reports `Unknown format for load: ply` when `libOpenGL.so.0` is
missing; pyrender imports and then dies inside the draw call with no EGL
platform. So phase 6 runs a real decimation and renders an actual offscreen
pixel.

**`weights_cached` was reporting a false green.** It globbed the HF hub
cache for `models--tencent--Hunyuan3D-2*`, which any metadata call leaves
behind as an empty directory. Meanwhile the real weights were never there:
`hy3dgen/shapegen/utils.py` consults `$HY3DGEN_MODELS` (default
`~/.cache/hy3dgen/<repo>/<subfolder>`) *before* it asks HuggingFace, so a
4.6GB checkpoint sits at a path the check never looked at. It now looks for
`model.fp16.safetensors` in both places, and reports which one it found.

**`server_status` gained `cutout_weights_cached`.** Phase 3 made u2net
load-bearing, and rembg downloads it silently on first use — 176MB
arriving in the middle of a generation, which reads as a stall rather than
a download. Phase 5 of the installer prefetches it.

**Phase numbers are a contract.** `setup_engine(only=N)` passes the number
straight through to `--only N`, so the six phases are named in the tool's
docstring and in the installer's header, and an unknown number is now
rejected. The original silently ran nothing and printed "setup complete",
which is indistinguishable from a clean install.

The installer does not use sudo. The two system libraries need root and
nothing else in the install does, so preflight reports the exact
`sudo apt install libopengl0 libegl1` line and phase 6 proves whether they
are actually working.
