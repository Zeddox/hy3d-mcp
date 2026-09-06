# hy3d-mcp

An MCP server that turns a single concept image into a game-ready 3D mesh
(GLB), fully locally on an NVIDIA GPU under WSL2 or Linux, by driving
[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) (PyTorch). One tool
call: background cutout → shape → decimation → watertight GLB. A second
call exports print-ready STL.

**Shape only — the output has no texture.** The texture stage needs far
more VRAM than an 8GB consumer card has, so `paint_mesh` refuses with an
explanation rather than half-running. Carved ornament in your concept art
comes back as smooth surface; that is what normal and displacement maps are
for, applied later. This is a property of the shape model, not a tuning
failure — `octree` is a tessellation-density dial, not a detail dial, and
raising it recovers no relief.

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
| `export_stl` | GLB → print-ready STL, Z-up, scaled to a target height | seconds |
| `prepare_concept` | concept image → centered square RGBA cutout | seconds |
| `render_preview` | offscreen PNG renders from any angle | seconds |
| `server_status` | full setup diagnostic, queue depth, last job | instant |
| `setup_engine` | runs `install.sh`; dry run unless `confirm=true` | instant (plan) / up to an hour (apply) |
| `cancel_job` | kill the running engine and free the queue | instant |
| `finish_model` | game-look texture pass — needs a GLB textured elsewhere | seconds |
| `paint_mesh` | unavailable on this build; refuses with an explanation | — |

Generation is serialized — one job at a time, machine-wide. The queue is an
`flock`, not just an in-process lock, so a job started from the workbench
or from a second Claude Code session waits its turn instead of thrashing
the card: on WSL2 two concurrent jobs do not fail, they both spill into
host RAM and crawl at PCIe bandwidth. `generate_model` streams MCP progress
notifications the whole way through (real diffusion steps, not a fake
clock), so a slow job stays distinguishable from a hung one, and
cancelling the call kills the engine process rather than leaving it
holding the queue.

## The workbench

A browser front end over the same tools, for when you would rather point at
an image than describe one:

    pip install "hy3d-mcp[web]"   # or: uv pip install ...
    hy3d-web
    # hy3d workbench  ->  http://localhost:8760

Drop or paste a concept image, watch the engine's own progress, orbit the
result in three.js, and export an STL with the printability checks attached.
Anything already generated is in the Outputs list, and `?glb=/files/<name>.glb`
opens straight into a mesh — a reload keeps what you were looking at.

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
volume, bounding-box fill, genus, and four solidity checks, and warns when
the finest detail in the mesh falls below your nozzle's minimum wall.

## Non-goals

- No cloud fallback.
- No texture stage on this build. Not a philosophical position — an 8GB
  card cannot run it.
- No batch tool — loop `generate_model`; the queue serializes.
- No multiview input **yet** — the pipeline is single-image at every entry
  point. Investigated and specced, not built; see below.

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
This build fetches the shape checkpoint only; the 2.1 paint weights the
macOS build also used are not downloaded here. Highlights, not legal
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
