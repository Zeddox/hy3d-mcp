# Texture painting on the 3060 Ti — the spike

**Status: it works.** 62.8s per mesh, 6.76 GiB peak against a 6.96 GiB
ceiling, on the same 8GB card that runs the shape pass. The output is a GLB
with real UVs and a baked albedo texture, which is what Godot wants and what
nobody wants to do by hand in Blender.

Measured 2026-09-06 on `front-20260906-171356.glb` (20,002 verts / 40,000
faces, the multiview viking) with its own front view as the prompt image.

## The numbers

| | |
|---|---|
| paint | 62.8 s |
| load + paint | 95.6 s |
| peak reserved | 6.76 GiB |
| free at baseline | 6.96 GiB |
| headroom left | 0.20 GiB |
| render / texture size | 1024 |

0.20 GiB is not comfortable. Under WSL2 an over-budget job does not raise
OOM, it spills into host RAM and crawls at PCIe bandwidth, so the failure
mode of pushing this further is a run that takes an hour rather than one that
stops. **1024 is the ceiling on this card at 40,000 faces**, not a
conservative starting point: upstream's default is 2048, which is 4x the
buffer area across six camera views and will not fit. The face count is part
of that claim rather than incidental -- `back_project` rasterises real
geometry per view, so a mesh generated with a higher `max_faces` moves the
number and has not been measured.

Both sub-pipelines need `enable_model_cpu_offload()`. That is not a tuning
knob here — the two of them are ~7.4 GiB of fp16 weights, more than the card
holds before a single activation.

## Four things upstream does not do on this box

The paint path needed more repair than the shape path did. All four are
mechanical, none needed a model change, and all are captured in
`scripts/build-rasterizer.sh` and `scripts/paint-weights.py`.

**1. `custom_rasterizer` has to be compiled, and needs a real nvcc.**
`mesh_render.py` has exactly one rasterizer branch (`raster_mode == 'cr'`),
so this CUDA extension is not an accelerator, it is the only implementation.
It builds against the CUDA that torch was built for — 12.4 — and this box has
no system CUDA and no passwordless sudo. `nvidia-cuda-nvcc-cu12` looks like
the answer and is not: at 12.4, 12.6 and 12.8 alike it ships `ptxas` and
`nvvm` and no nvcc frontend at all. What works is unpacking eight of NVIDIA's
own `.debs` from the ubuntu2204 repo into a user-owned prefix — 233MB, no
root, exact version. Two further snags, both in the script: the extension
includes `cusparse.h`, which on this box lives only inside the `nvidia-*`
wheels, so every one of their `include` dirs goes on the compiler flags; and
`cuda-cudart-dev` ships `libcudart.so` as a dangling symlink to the runtime
package we deliberately don't install, so it gets pointed at the copy torch
already carries.

**2. diffusers 0.40 gates the custom pipeline behind `trust_remote_code`.**
Upstream predates the gate and doesn't pass it. The code being gated is the
local file in the engine checkout, not something fetched from the hub.

**3. transformers 5 refuses to open the turbo text encoder.**
It ships only as `pytorch_model.bin`, and transformers now declines
`torch.load` outright under torch < 2.6 (CVE-2025-32434) — a blanket version
gate, not a claim about this file. Converting it and the vae to safetensors
once takes the pickle out of the loading path for good.

**4. The turbo unet's own loader wants a file the repo would rather not
ship.** Its `modules.py` reads `diffusion_pytorch_model.bin` by name and
never opens the 3.72GB safetensors sitting beside it. The `.bin` on the hub
is the 7.33GB fp32 copy of those same weights. `scripts/paint-weights.py`
builds the `.bin` locally from the safetensors instead: the safetensors holds
1571 tensors, the loader's model wants 1535 and loads strict, and the 36
extras are all IP-adapter — 32 `attn2.to_k_ip`/`to_v_ip` plus four
`image_proj_model` weights — which the loader never constructs (it hardcodes
`is_turbo = False`) and which nothing in the pipeline references. Dropping
them is not a guess: a strict load of anything wider would fail, so upstream's
own `.bin` must contain exactly these 1535. The rendered result confirms it —
wrong weights produce noise, not a correctly coloured viking.

Net: a 10.4GB download instead of ~18, and one fewer pickle in the loading
path than upstream has. On disk it settles at 16GB, because each conversion
leaves its source beside the result; `paint-weights.py` prints the 5.4GB that
nothing loads any more, and deletes none of it.

## What the texture actually looks like

Coherent and plausibly game-ready at 1024: skin, blue tunic, grey helm, cream
horns, brown boots, all landing on the right geometry with no smearing across
seams.

The known weakness is cross-view agreement. The fur pauldrons read brown from
the front and grey from the back, and the tunic is blue in front and
grey-white behind. Six camera views are baked with weights
`[1, 0.1, 0.5, 0.1, 0.05, 0.05]` — the front view dominates by 10x, so
anything the back view invents is kept wherever the front cannot see it.
Feeding the paint pass the same back view the multiview shape pass gets is
the obvious next experiment, and `__call__` already accepts a list of images.

## What painting does to the geometry: nothing, and the checks disagree

A painted GLB reports `watertight: false`, on the same faces the shape stage
called watertight. That is UV unwrapping, not damage.

| | vertices | faces | `is_watertight` | euler |
|---|---|---|---|---|
| shape only | 20,002 | 40,000 | true | 2 |
| after paint | 24,929 | 40,000 | **false** | 212 |
| after paint, merged by position | 20,002 | 40,000 | true | 2 |

`xatlas` cuts the surface into charts and duplicates the vertices along every
seam so each side can carry its own UV. `is_watertight` asks whether every
edge is shared by exactly two faces *by vertex index*, so the cuts read as
holes. Merging by position restores watertight, euler 2, a single body and
zero broken faces — the surface never opened.

Two places had to act on that:

* `workers/tostl.py` merges before it checks or writes. STL carries no UVs,
  so nothing is lost, and without it a painted model was both reported
  unprintable and handed to a slicer as an open shell. On the same mesh the
  painted STL now matches the untextured one exactly: 154.7 cm³, 33.9% bbox
  fill, `printable: true`.
* `engine_cli` measures the stat on a merged **copy**, so the number
  describes the geometry. The copy is thrown away — merging in place keeps
  one arbitrary UV per position and breaks the texture. Gated on the paint
  flag: shape-only output has no seams to merge, so running the probe there
  could only be a no-op or a new way for the default path to fail after it
  had already written a good GLB.

## The material fix that was not applied

Worth recording because the mistake is invisible from the mesh object.
Setting `metallicFactor = 0.0` on what the paint pipeline returns does
nothing: it hands back a `SimpleMaterial`, which has no such field, so the
assignment lands on an unused attribute and trimesh's exporter runs its own
`to_pbr()` with `metallicFactor` unset — which glTF reads as the 1.0 default.
Every painted GLB shipped with metallic 1.0 and roughness 0.9036 while the
code read as if it had been fixed. 0.9036 is `(2/(glossiness+2))**0.25` and
is the tell that the conversion ran.

The fix converts to `PBRMaterial` first and assigns the result back. The
guard against a repeat is `glb_material` in the stats: it reads both factors
back out of the written file, so the claim is measured rather than asserted.

## Chaining

`finish_model` now has something to chain onto — painted output is the first
textured GLB this server produces. Verified on the viking, through the tool
rather than the worker: attributes, base colour and the emissive accent map
all survive, and the pass sets metallic 0 / roughness 1 itself.
`accent_coverage_pct` came back 0.0, which is the extractor's range (it keys
on broad saturated red-dominant panels) rather than a fault.

The tool's normals post-step was the thing worth checking, since it rewrites
a GLB that `finish.py` has just given an emissive texture. On painted input
`finish.py` keeps NORMAL, so the step reports `normals_added: false` and
touches nothing; forcing the other branch by stripping NORMAL first, the
rewrite preserves both images and the emissive material. Either way the
accent map survives.

## Reproducing it

```sh
scripts/build-rasterizer.sh                  # toolchain + CUDA extension
python scripts/paint-weights.py ~/.cache/hy3dgen/tencent/Hunyuan3D-2/hunyuan3d-paint-v2-0-turbo
```

The weights themselves are the delight model (4.1GB) plus the turbo paint
subfolder, skipping `unet/diffusion_pytorch_model.bin` — see the note above.

**The fresh `--with-paint` install has not been run end to end.** Every
piece of phase 5 has been exercised by hand with the same argv, and
`bash -n` passes, but the weights were already present on this box, so
`install.sh --with-paint` has only ever taken the skip path. The download
block, the `paint-weights.py` call and the `build-rasterizer.sh` invocation
have not run in that order on a clean machine. First person to try it
should expect to debug phase 5, not the pipeline.

`scripts/build-rasterizer.sh` is idempotent and re-runnable. Note that
`install.sh --only 3` rebuilds the engine venv, which discards the compiled
extension **and** dangles the `libcudart.so.12` symlink that points into it;
re-run the build script after any venv rebuild.
