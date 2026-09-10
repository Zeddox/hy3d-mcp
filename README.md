# hy3d-mcp

An MCP server that turns a single concept image into a game-ready 3D mesh
(GLB), fully locally on an NVIDIA GPU under WSL2 or Linux, by driving
[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) (PyTorch). One tool
call: background cutout → shape → decimation → watertight GLB. A second
call exports print-ready STL.

**Texture is opt-in and fits.** `paint=True` adds UVs and a baked albedo in
about 85 seconds — 6.77 GiB peak against a 6.96 GiB ceiling, at a 1024 bake.
See [Texturing](#texturing). Diffuse colour only: no normal or roughness map.

**Relief is the thing neither stage gives you.** Carved ornament in your
concept art comes back as smooth surface, and the texture pass paints it on
rather than cutting it in. That is a property of the shape model, not a
tuning failure — `octree` is a tessellation-density dial, not a detail dial,
and raising it recovers no relief. Normal and displacement maps are the
answer, applied later.

This branch is a port of the original Apple Silicon / MLX server. See
[`docs/wsl2-port.md`](docs/wsl2-port.md) for what carried over, what did
not, and every measurement behind the numbers below.

Models land as **file paths**, never blobs — importing them into your
engine is the caller's job (for Godot: copy into the project and run
`godot --headless --import`).

## Requirements

- An NVIDIA GPU with 8GB or more, under WSL2 or Linux. Developed and
  measured on an RTX 3060 Ti (8GB), driver 610.62.
- [uv](https://docs.astral.sh/uv/), git, and python3
- ~13GB free disk: 4.6GB of weights, ~7GB of venv (the cu124 torch wheels
  are most of it)
- Two system libraries: `sudo apt install libopengl0 libegl1`
- A Hunyuan3D-2 checkout and an engine venv — **[`./install.sh`](#set-up-the-engine)
  builds both**

Never install an NVIDIA driver inside the WSL guest. The Windows driver is
projected in through `/usr/lib/wsl/lib`; installing one in the guest breaks
it.

The server itself carries no ML dependencies; it shells out to the engine
venv.

## Install as a Claude Code plugin (recommended)

The repo is also a Claude Code plugin that bundles the MCP server plus a
`create-3d-model` skill (prompt → concept image → GLB, with all the
input doctrine baked in):

```
/plugin marketplace add JimCline/hy3d-mcp
/plugin install hy3d-gen@hy3d-mcp
```

Once installed, ask for a 3D model in plain language or invoke
`/hy3d-gen:create-3d-model`. The server starts via
`uv run --project <plugin-root> hy3d-mcp` — uv resolves the venv on first
run.

## Install as a bare MCP server

```sh
git clone https://github.com/JimCline/hy3d-mcp ~/git/repos/hy3d-mcp
```

Register with your MCP client (e.g. in `.mcp.json` or Claude Code's
`claude mcp add`):

```json
"hy3d-gen": {
  "command": "uv",
  "args": ["run", "--project", "~/git/repos/hy3d-mcp", "hy3d-mcp"],
  "env": {
    "HY3D_ENGINE_REPO": "~/git/repos/Hunyuan3D-2",
    "HY3D_ENGINE_PY": "~/.hy3d/engine-venv/bin/python",
    "HY3D_OUT": "~/hy3d-output"
  }
}
```

All are optional; the values above are the defaults. `HY3D_PY` (the
interpreter the mesh/image workers run under) defaults to
`HY3D_ENGINE_PY`, since the engine venv already carries the whole worker
stack — set it only if you want the workers somewhere else.

## Set up the engine

The server is a thin wrapper — the actual pipeline is a separate checkout
that has to be cloned, given a python environment, and fed 4.6GB of
weights.

```sh
./install.sh --plan     # print exactly what it would do, change nothing
./install.sh            # do it, confirming the download
```

Six phases:

| | | |
|---|---|---|
| 1 | preflight | GPU, driver, uv, disk, system GL libraries |
| 2 | checkout | clone Hunyuan3D-2 at the pinned commit |
| 3 | venv + torch | python 3.10 and the cu124 wheels (~3GB) |
| 4 | deps | the shape-only package set (~2GB) |
| 5 | weights | shape checkpoint (~4.6GB) and u2net (~176MB) |
| 6 | verify | import the pipeline, prove CUDA, decimation and EGL |

**Every phase inspects before it acts**, so it is safe to re-run: finished
work is skipped and a failed run resumes where it stopped. The one
expensive phase stops and asks first. `--yes` runs unattended, `--only N`
runs one phase, `--repo` / `--venv` relocate the targets.

Phase 6 proves rather than assumes. It imports the pipeline, checks that
torch actually sees the card, runs a real decimation through pymeshlab,
and renders an actual offscreen pixel through EGL — because each of those
fails in a way an import check cannot see.

It will not install system packages: those need root and the rest does
not, so it prints the exact `sudo apt install` line instead.

From inside an MCP client, the `setup_engine` tool is the same script. It
defaults to a dry run and returns the plan; it only executes when called
again with `confirm=true`, so the agent has to show you the cost before
spending it. `setup_engine(confirm=true, only=5)` is the way to pre-fetch
weights on their own.

When it finishes it prints the values to put in your MCP config, and
`server_status` should come back all green.

## The setup gotchas

`install.sh` handles all of these; they are documented because they are
what a by-the-book install of the upstream repo gets wrong, and what
`server_status` is looking for when it fails.

1. **`scikit-image` is required and undeclared.** Upstream's
   `requirements.txt` has it commented out, but
   `shapegen/models/autoencoders/surface_extractors.py` needs it for the
   default `mc_algo='mc'` path. Omitting it is the first thing that breaks
   a shape run.
2. **torch and torchvision must come from the cu124 index in one
   command.** Installing torchvision from PyPI afterwards silently pulls a
   different torch and discards the cu124 build — which then imports
   perfectly and runs on the CPU at a hundredth of the speed.
3. **`numpy<2`.** pymeshlab wheels of this era are built against the numpy
   1.x ABI and fail at *import*, not at use.
4. **`libopengl0`.** pymeshlab dlopens `libOpenGL.so.0` even headless, and
   Ubuntu ships `libGL.so.1` and `libGLX.so.0` but not that one. Its plugin
   load fails, taking `io_base` with it, and decimation surfaces as the
   thoroughly misleading `PyMeshLabException: Unknown format for load: ply`.
5. **`PYOPENGL_PLATFORM=egl`, set before pyrender is imported.** The
   platform is read at import time. Without it, headless rendering fails
   inside the draw call with "Attempt to retrieve context when no valid
   context" rather than anything about a display.

## Tools

| Tool | What it does | Typical time |
| --- | --- | --- |
| `generate_model` | image → watertight GLB (auto cutout, decimation) | ~3 min |
| `generate_model` w/ views | front + left/back/right → GLB via Hunyuan3D-2mv | ~4 min |
| `export_stl` | GLB → print-ready STL, Z-up, scaled to a target height | seconds |
| `prepare_concept` | concept image → centered square RGBA cutout | seconds |
| `render_preview` | offscreen PNG renders from any angle | seconds |
| `server_status` | full setup diagnostic, queue depth, last job | instant |
| `setup_engine` | runs `install.sh`; dry run unless `confirm=true` | instant (plan) / up to an hour (apply) |
| `cancel_job` | kill the running engine and free the queue | instant |
| `generate_model` w/ `paint=True` | the same, plus a baked albedo | +~85s |
| `paint_mesh` | texture a GLB that already exists | ~85s |
| `finish_model` | game-look pass over an already-textured GLB | seconds |

Generation is serialized — one job at a time, machine-wide. The queue is an
`flock`, not just an in-process lock, so a job started from the workbench
or from a second Claude Code session waits its turn instead of thrashing
the card: on WSL2 two concurrent jobs do not fail, they both spill into
host RAM and crawl at PCIe bandwidth. `generate_model` streams MCP progress
notifications the whole way through (real diffusion steps, not a fake
clock), so a slow job stays distinguishable from a hung one, and
cancelling the call kills the engine process rather than leaving it
holding the queue.

## Multiview

A single image cannot tell the model what the back looks like, so the shape
stage infers one, and what it infers is a smooth mirror of the front. Passing
the real back stops it guessing. `generate_model` takes `left_image`,
`back_image` and `right_image` alongside `image_path`, and any one of them
switches the run to `tencent/Hunyuan3D-2mv` — a separate 4.6GB checkpoint
whose conditioner reads all four views at once. Fetch it with
`./hy3d install --only 5 --with-mv`; without it the single-image workflow is
unaffected and `server_status` reports `mv_weights_cached` as the one soft
failure.

The views have to agree with each other. They must be the same subject at the
same scale from those four angles — front, then 90° clockwise for left, 180°
for back, 270° for right. A turntable render or an orthographic sheet works;
four separately-prompted images usually do not, and views that disagree about
proportion give a worse mesh than the front alone. There is no slot for a
three-quarter or perspective view: the conditioner knows those four indices
and nothing else. A subset is fine — front plus back is a real improvement on
its own.

Measured on the 3060 Ti, upstream's `example_mv_images/1`, seed 42, 50 steps,
octree 384, both inputs pre-keyed so background removal was not a variable:

| | one image | front + left + back |
| --- | --- | --- |
| generate | 114.9s | 152.6s |
| peak VRAM | 6.22 GiB | 6.69 GiB of 6.96 |
| raw faces | 616,244 | 636,980 |

Multiview costs about a third more time and 0.47 GiB more VRAM, which on an
8GB card leaves under 0.3 GiB of headroom at octree 384. A fourth view or a
higher octree is where that runs out — drop to octree 320 or pass
`cpu_offload=True`, and check `peak_reserved_gib` in the result, because on
WSL2 the failure mode is not an OOM but a silent spill to host RAM.

What the extra views actually bought on that test: both overall straps
instead of one lumpy asymmetric one, bows with a knot rather than blocks, a
hairline where the single-image mesh had a smooth ball, and fingers instead
of mittens. The subject was near-symmetric, which understates it — a
backpack, cape or tail is where a single front view has nothing to go on.

## Texturing

Off by default, one flag to turn on:

```python
generate_model(image_path="viking.png", paint=True)     # shape, then texture
paint_mesh(mesh_path="viking.glb", image_path="viking.png")   # an existing GLB
```

You get a GLB carrying `TEXCOORD_0` and a baked albedo, which Godot imports
and lights without a trip through Blender. Diffuse colour only — no normal,
roughness or metallic map — and the exporter sets `metallicFactor` to 0,
because trimesh defaults it to 1.0 and an albedo under a metallic material
renders as near-black. The `glb_material` field reports both factors as
read back from the written file, not as set on the mesh, because the
difference is exactly where that bug hides.

**A painted GLB reports `watertight: false`, and the geometry is fine.**
UV unwrapping splits vertices along every atlas seam — the same 40,000
faces arrive carrying 24,929 vertices instead of 20,002 — and
`is_watertight` asks whether faces share vertices. Merge by position and
the mesh is watertight, euler 2, one body, no broken faces; nothing about
the surface changed. `export_stl` does that merge before it checks or
writes, so a painted model still prints: on the same mesh it reports the
identical volume, bbox fill and `printable: true` as the untextured
original. On a painted run the stat this server reports is measured on a
merged copy for the same reason; a third-party tool reading the GLB
directly will say non-watertight. Shape-only output is unaffected — it has
no UVs to split, so nothing there changed.

**`texture_size` 1024 is a ceiling, not a cautious default.** Upstream bakes
at 2048, which is four times the buffer area across six cameras and
does not fit in 8GB; on WSL2 it will not fail, it will spill into host RAM
and finish an hour later. At 1024 the pass peaks at 6.77 GiB against 6.96
GiB free — measured at 40,000 faces, and a much denser mesh moves that
number, since the bake rasterises real geometry per view.

**Cross-view agreement is the known weakness.** Six cameras are baked with
weights `[1, 0.1, 0.5, 0.1, 0.05, 0.05]`, so the front dominates by 10x and
anything only the back camera sees is resolved on its own. On a test viking
that showed up as fur pauldrons reading brown in front and grey behind.

Setup is separate from the shape install, because it is another 10.4GB
downloaded (16GB on disk, since each conversion leaves its source beside
the result) and the only part of this project that needs a compiler:

    ./hy3d install --with-paint

That fetches the weights, converts the two components that ship as pickles
(transformers 5 will not open them under torch 2.5), rebuilds the turbo
unet's `.bin` from the safetensors beside it rather than pulling a 7.33GB
fp32 duplicate, and compiles `custom_rasterizer` against a user-local CUDA
12.4 it unpacks from NVIDIA's own debs — no root, no system CUDA. Until it
has run, `server_status` reports `paint_ready: false` with the missing half
named, the workbench greys out its texture checkbox, and shape generation
carries on unaffected. Rebuilding the engine venv discards the compiled
extension; `scripts/build-rasterizer.sh` puts it back.

The measurements and the four upstream repairs behind all of this are in
[`docs/paint-spike-2026-09-06.md`](docs/paint-spike-2026-09-06.md).

## The workbench

A browser front end over the same tools, for when you would rather point at
an image than describe one. From a checkout, one command starts everything:

    ./hy3d
    # hy3d workbench  ->  http://localhost:8760

`hy3d-web` is a console-script entry point, which means it exists only inside
the server venv and only after the package has been installed there — a venv
built before 0.8.0 has `hy3d-mcp` and no `hy3d-web` at all. `./hy3d` is the
wrapper that makes that true before it runs one: it creates or repairs the
server venv, runs the same checks `server_status` reports (refusing to start
a page whose only working button would be the gallery), and hands over with
`exec` so Ctrl-C stops the server rather than the wrapper.

    ./hy3d                  start it on 127.0.0.1:8760
    ./hy3d web --lan        bind 0.0.0.0 for a browser off this host
    ./hy3d web --background  detach, logging to ~/.hy3d/web.log
    ./hy3d stop             stop a backgrounded one
    ./hy3d status           run the checks and exit
    ./hy3d install          hand off to install.sh (engine venv, weights)
    ./hy3d install --with-mv  the above plus the multiview checkpoint

Engine setup stays in `install.sh`, which `./hy3d` names but never runs
uninvited — phase 5 downloads 4.6GB and asks first. From a wheel rather than
a checkout, `pip install "hy3d-mcp[web]"` and `hy3d-web` are the equivalent.

Drop or paste a concept image, watch the engine's own progress, orbit the
result in three.js, and export an STL with the printability checks attached.
The **Multiview** toggle opens three more slots — left, back, right — and
switches the run to the checkpoint that reads them; it greys itself out with
the fix when that checkpoint is not downloaded.

The **target** picker under it is the one preset worth having, because the
setting it turns on is the least discoverable and most consequential one in
the page: *3D print* sets `max faces` to 0, which means "keep every triangle
marching cubes produced" and reads least like that in a box labelled max
faces, and clears the texture toggle, because an STL carries neither UVs nor
colour. *Game engine* puts back the 40,000-face budget. It reflects the form
rather than remembering the click, so hand-editing either field drops it back
to neither, and it deliberately leaves `octree` alone — 512 measured 1.78×
the triangles for 1.82× the wall clock with no gain in the face, and spilled
past the card's ceiling doing it. Min wall is remembered across sessions,
since the nozzle is a property of the printer rather than of the model; height
is not. **CPU offload** is inside Settings beside the octree box it exists
for: without it, raising octree is a trap, because a job that overruns VRAM
does not fail, it spills into host RAM and crawls at bus speed.
Anything already generated is in the Outputs list, and `?glb=/files/<name>.glb`
opens straight into a mesh — a reload keeps what you were looking at. The name
beside the download link is that file's, and copies on click: it is the one
handle worth quoting, because a job id dies with the server process while the
name is what is actually on disk.

One reading to expect from the print target: **watertight: no** on the generate
card. The decimation branch runs `FloaterRemover` on the way past, so setting
`max faces` to 0 skips it, and the raw mesh keeps the tail of detached specks
marching cubes leaves behind — measured at nine, from 400 faces down to 1,
against a closed 681,264-face figure. A handful of stray faces is enough to
make the whole file read open, so the stat is true and useless. It is not
flagged amber there, and the STL export is the authority: it drops them before
it checks, reports how many went, and that mesh comes back watertight.

It is a separate process from the MCP server and shares nothing with it but
the code and the GPU queue, so running both at once is safe. It binds
127.0.0.1: WSL2 normally forwards localhost, so `http://localhost:8760` in
the Windows browser reaches it. When the distro's networking mode does not,
re-run with `--host 0.0.0.0` and use the address the startup banner prints
— but note that binding wide exposes a service that runs generation and
serves files.

## What a run actually costs

Measured end to end, RTX 3060 Ti, defaults, from a raw garden-scene
concept with no manual prep:

| | |
|---|---|
| wall clock | ~160–185s (~35s model load, ~115s generate, ~11s decimate) |
| raw mesh | 542k faces |
| after decimation | 40,000 faces / 20,002 verts, still watertight |
| attributes | `NORMAL`, `POSITION` |
| peak VRAM reserved | 6.22 GiB against 6.96 GiB free |

With `paint=True`, on the same card at a 1024 bake:

| | one image | front + left + back |
|---|---|---|
| wall clock | ~160s total | ~254s total |
| of which the paint pass | ~48–70s | ~46s |
| attributes | `NORMAL`, `POSITION`, `TEXCOORD_0` + a baked albedo | same |
| peak VRAM reserved | 6.77 GiB of 6.96 | 6.76 GiB of 6.96 |

Multiview and paint stack without stacking their memory: the shape pipeline
is released before the paint models load, so the peak is the larger of the
two stages rather than their sum. That teardown is the difference between a
70s paint pass and a 157s one — see
[`docs/paint-spike-2026-09-06.md`](docs/paint-spike-2026-09-06.md).

The model reloads on every call — each generation is a fresh subprocess —
which is where the 35s floor comes from.

## Notes from production use

- **The WSL2 failure mode is not out-of-memory.** Under WDDM the driver
  serves oversized allocations out of host RAM instead of failing, so a
  job that does not fit still finishes, having crawled at PCIe bandwidth.
  A completed run is therefore not evidence that it fit. Watch
  `peak_reserved_gib` against `vram_ceiling_gib` in the result;
  `cpu_offload=true` is the lever when they meet.
- **Outputs carry vertex normals.** Godot does not synthesise `NORMAL` and
  lights the whole mesh off one constant vector when it is missing — which
  presents as a bad material, not a missing attribute, and is expensive to
  diagnose. The engine writer emits it directly.
- **`is_watertight` is not the check you want.** Two meshes that both pass
  it, and render identically, differed 5.6× in enclosed volume: one was a
  hollow shell. `export_stl` reports `bbox_fill_pct` (volume as a
  percentage of the bounding box), which is the number that catches it.
  A solid object sits well above 15%.
- **`octree` costs scale with concept detail, not just the number.**
  Raising it from 384 to 512 doubled the wall clock and recovered *no*
  surface relief — it changed proportions slightly. Reach for it when thin
  struts *fuse together*, not as a quality dial.
- **Face counts vary widely at identical settings** (542k–881k raw across
  subjects). More raw faces is often surface noise, not detail: the 2mini
  model emits *more* than the full 2.0 and holds edges less crisply.
  `max_faces` (default 40,000) decimates to a game-ready count and the
  result stays watertight.
- **"Separator is not found, and chunk exceed the limit" (fixed in 0.12.2).**
  This one is worth knowing about because of what it did *not* do. The
  server read the engine's output a line at a time, and asyncio caps a line
  at 64KiB; tqdm redraws its bars with a carriage return and no newline, so
  the whole volume-decode bar counted as one line and blew that cap. The
  job was reported as failed — but the exception skipped the path that
  kills the child, so the engine ran on, orphaned, and wrote a complete and
  perfectly good GLB. If you saw this error before 0.12.2, look in your
  output directory: the mesh is almost certainly there.

- **Long jobs and client timeouts.** The progress stream is what keeps a
  client's idle timer alive; if yours still gives up, raise its tool
  timeout (Claude Code: `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`, or a
  per-server `timeout` in MCP settings). If a job is ever abandoned
  mid-flight, `cancel_job` frees the queue without hunting for a pid.

## Input doctrine

- Feed **naturally lit** concept art — the model de-lights internally.
  Pre-flattened "albedo-style" input bakes pale and featureless.
- **No drop shadows** in the source image — they reconstruct as literal
  geometry under the model.
- Single object, roughly centered; ¾ view works best. A plain background
  keeps the cutout on its cheap corner key, but is not required: busy
  concept art falls back to u2net segmentation plus a largest-component
  filter, which is what makes real garden-scene art usable without manual
  prep.

## 3D printing

`export_stl` rotates glTF's Y-up into the Z-up every slicer expects,
scales to a target height in millimetres (STL is unitless and read as mm),
and drops the model onto the bed at the origin. It reports enclosed
volume, bounding-box fill, genus, and four solidity checks. It drops
detached specks first — undecimated output carries a tail of them, and they
are the only reason a closed figure reports not watertight, so this is what
makes `max_faces=0` printable — reporting the count as
`components_dropped`. It also reports
`detail_pitch_mm` — the surface sampling pitch at that scale, which on an
undecimated mesh is exactly `height_mm / octree`. It is warned against
`min_wall_mm` in both directions: below it the printer is the limit, well
above it the mesh is.

## Non-goals

- No cloud fallback.
- No PBR maps. The texture pass bakes diffuse colour and nothing else —
  no normal, roughness or metallic map.
- No batch tool — loop `generate_model`; the queue serializes.

## Investigations

- [`docs/wsl2-port.md`](docs/wsl2-port.md) — the port itself, phase by
  phase: what the 8GB budget rules out, how the progress protocol survived
  the engine swap, why the cutout fallback moved out of the engine, and
  every measurement quoted above.

The three below predate the port and describe the MLX build. The findings
about *inputs* still hold — those are properties of the shape model, which
is the same one — but the routes and costings are macOS-specific.

- [`docs/multiview-routes-2026-08-02.md`](docs/multiview-routes-2026-08-02.md)
  — multi-image → 3D. Three routes costed (native MLX port, ComfyUI hybrid,
  upstream PR), six open questions, and a Phase 0 A/B that settles whether
  multiview earns its keep before anything is built. **Tabled, decision open.**
- [`docs/multiview-findings-2026-08-02.md`](docs/multiview-findings-2026-08-02.md)
  — the investigation behind it. Read this for why contact sheets must never
  be fed back in, why generator sheets must never be used to judge geometry,
  and the measurement showing +31% geometry from input quality alone.
- [`docs/field-report-2026-08-01.md`](docs/field-report-2026-08-01.md)

## License

MIT — but that covers **this wrapper code only**. This repo distributes no
model weights and no Tencent code.

### Model weights license (read this)

The pipeline runs on Tencent's Hunyuan3D weights, which you download
yourself and which are governed by the **Tencent Hunyuan 3D 2.0 Community
License Agreement**
([2.0](https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE)).
This build fetches the 2.0 shape checkpoint, and — with `--with-mv` or
`--with-paint` — the 2.0 multiview and paint weights. The 2.1 weights the
macOS build used are not downloaded here. Highlights, not legal
advice; read the license:

- **Territory:** the license does not apply in the European Union, the
  United Kingdom, or South Korea. If you're there, you may not use the
  weights at all.
- **Scale:** products/services exceeding 1M monthly active users require
  written permission from Tencent.
- **Attribution:** distributing or productizing anything built on the
  weights requires the Tencent license notice; 2.1 asks for a "Powered by
  Tencent Hunyuan" mark.
- **Acceptable use:** no training competing models on it, no undisclosed
  synthetic-media deception, no military use, among others.
- **Your outputs are yours:** Tencent claims no rights to generated 3D
  models; you own them and are responsible for how you use them.

[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2), the upstream this
server shells out to, carries Tencent's own license — the code and the
weights are covered separately, so read both.
