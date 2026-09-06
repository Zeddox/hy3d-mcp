---
name: create-3d-model
description: Create an untextured 3D model (GLB, and STL for printing) from a text prompt or a concept image, fully locally, using the hy3d-gen MCP server. Use when the user wants a 3D model, asset, mesh, GLB or printable STL of something they describe or have a picture of.
---

# Create a 3D model from a prompt or image

You have the `hy3d-gen` MCP server (bundled with this plugin): local
image-to-3D via Hunyuan3D-2 on CUDA. The pipeline is concept image → RGBA
cutout → shape → GLB. Your job is to get the user from "I want a 3D model
of X" to a mesh file, with a preview.

**The output has no texture.** There is no paint stage on this build — the
texture pipeline needs more VRAM than the card has, so `paint_mesh` refuses
rather than half-running. What you deliver is clean geometry with normals
and no UVs or material. Tell the user that up front; do not let them expect
colour and find grey.

That is less of a loss than it sounds for the usual targets. The shape
stage resolves silhouette and large form well and does not resolve surface
relief at any setting, so ornament, panel lines and fabric folds were
always going to come from maps applied downstream rather than from the
mesh.

## First use in a session

Call `server_status` once before the first generation. Don't re-check on
later calls.

If any check is `ok: false`, relay that check's `fix` text to the user
verbatim — every failure carries its exact remedy. Setup is not automated
on this build: `setup_engine` returns the one command to run
(`bash scripts/provision-engine.sh`) rather than running it.

Read `cuda_ok` even when everything passes. It reports free VRAM, and the
WSL2 failure mode is not a clean out-of-memory error: the driver serves
oversized allocations out of system RAM, so a job that does not fit still
finishes, having crawled at PCIe bandwidth. If something else on the
machine is holding VRAM, a normal 3-minute job can take twenty.

## Step 1 — get a concept image

**If the user supplied an image**, use it directly and go to Step 2.

**If the user gave a text prompt**, generate a concept image with whatever
image-generation tool is available in the session (e.g. a Gemini or other
image-gen MCP). If none is available, ask the user for an image — do not
try to proceed without one.

The concept image makes or breaks the model, and it is the single biggest
lever on output quality — bigger than any generator knob. Spend effort here
before reaching for `octree`.

Compose the image prompt from the user's description plus ALL of these:

- **single object**, whole object in frame, roughly centered
- **three-quarter view** (shows front and side; best geometry recovery)
- **plain, uniform background** — nothing else in frame, and a flat single
  tone rather than a gradient or vignette. The local cutout keys on corner
  colour, so hard figure/ground separation matters. Light gray is the
  default. When the subject has no white or near-white parts, **pure white
  with product-cutout framing** ("isolated on seamless white, catalog
  product cutout") is worth trying — it came back clean once where two
  gray-background attempts kept a contact shadow through increasingly
  emphatic negative prompting. That is a single sample, and it changed the
  background colour and the framing language together, so which part did
  the work is unknown.
- **even, soft, neutral studio lighting.** Ask for "soft even studio
  lighting, neutral white". Avoid dramatic, moody, rim-lit, golden-hour
  or single-hard-key looks: baked-in directional shading and blown
  highlights are reconstructed as surface relief that isn't there.
  But never request "flat lighting", "unlit", or "albedo style" either —
  the generator de-lights internally, and pre-flattened input reconstructs
  featureless. Even and soft, not absent.
- **no drop shadow, no contact shadow** — shadows under the object are
  reconstructed as literal geometry. Say "floating, no shadow" in the
  prompt.
- no text, watermark, or frame

**Then look at what came back before generating.** Image generators
routinely ignore the shadow instruction, and a soft contact shadow is easy
to miss against a gray background. Check for it explicitly, along with a
background gradient or a hard key light. Regenerating a concept costs
seconds; a bad generation costs ~3 minutes and still has to be redone.

**Better than looking: measure the key.** Run `prepare_concept` and check
the alpha channel of the RGBA it writes. A clean key is almost entirely
alpha 0 or 255 with well under 1% in between. What the histogram catches is
**background non-uniformity**, which is easy to miss by eye: one observed
gradient-gray background keyed at 36% partial alpha and left patches of
background fully **opaque** — the key had smeared rather than separated. A
flat white background of the same subject keyed at 0.1% partial.

A `prepare_concept` refusal ("corner patches disagree") is a signal about
the image, not a blocker. `generate_model` falls back to a learned
background remover that handles painted concept art the corner-sampling key
cannot. Treat the refusal as a prompt to check whether the background is
really plain — then generate anyway if the user is working from art they
already have.

Show the concept to the user before spending generation time, unless they
asked you to just go ahead.

## Step 2 — generate

Call `generate_model` with the image path. Defaults are right for most
cases.

Expect **~3 minutes** end to end at the defaults. Roughly a third of that
is diffusion, most of the rest is volume decoding, and every call reloads
the model from scratch (~35s) because each generation is a fresh
subprocess. The call streams progress the whole way, so a working job is
visibly working. Between 40% and 95% the bar is running on elapsed time
rather than real completion — that stretch is the volume decode, which
reports nothing parseable. It is not stalled.

Generation is serialized on the server (one job at a time, it's
memory-bound) — a second call queuing behind a first is normal, not a hang.
Do not fire generations in parallel expecting speedup; on WSL2 they would
not fail, they would both spill into host RAM and crawl. If a call is ever
abandoned mid-flight and later calls appear to hang, `cancel_job` frees the
queue.

The knobs that matter:

- `max_faces` (40000) — the raw mesh is 500k–1M faces, which is not
  something to hand an engine. This decimates to a game-ready budget and
  keeps the mesh watertight. Set `0` to keep the raw mesh when the user is
  going to retopologise themselves or wants maximum density for a print.
- `auto_cutout` (true) — leave it on.
- `seed` — same image and seed reproduce the mesh exactly; change it to
  reroll geometry from the same picture.
- `cpu_offload` (false) — runs the stages sequentially through host RAM.
  Reach for it only if the returned `peak_reserved_gib` is sitting at
  `vram_ceiling_gib`, which means the run spilled. It costs runtime.
- `model` — `tencent/Hunyuan3D-2` by default. See the warning below before
  choosing `tencent/Hunyuan3D-2mini`.

Normals are written into every GLB automatically. Without them Godot lights
the whole mesh off one constant vector, which reads as a bad material
rather than a missing attribute.

### Fidelity knobs

These are weaker than they look, and none of them will fix a wrong shape —
that is always the concept image's job. Each is unset by default; move one
at a time and keep the seed fixed so you can attribute the difference.

- `octree` (384) — marching-cubes resolution. **This is a tessellation
  density dial, not a detail dial.** Measured on the same concept at the
  same seed, 512 produced nearly twice the triangles describing the same
  surface, recovered no additional relief, and took twice as long. Reach
  for it for one specific failure — thin parts (struts, masts, antennae)
  coming out **fused into the hull** — and not as a general quality lever.
- `steps` (50) — shape diffusion steps. Diminishing well before this.
- `guidance` (5.0) — how tightly the mesh follows the image. Higher is more
  faithful but over-sharpens.

**Surface relief is out of range at every setting.** Carved ornament, tile
courses, panel lines and fabric folds that are plainly visible in the
concept art come back smooth. Do not spend the user's time raising knobs to
chase them, and do not report their absence as a defect to fix — say
plainly that this detail belongs in a normal or displacement map applied
downstream.

### A caution about 2mini

`tencent/Hunyuan3D-2mini` is about twice as fast and its previews look
comparable to the full model. On one measured subject it produced a
**thin-walled hollow shell** where the full model produced a solid: matched
silhouettes, matched bounding boxes, both watertight and single-body, and
volumes differing by **5.6×**. No render shows this. At a 120mm print that
was roughly 0.7mm walls, at or under what a 0.4mm nozzle can lay down.

Prefer the default. If you do use 2mini, check `bbox_fill_pct` from
`export_stl` before telling anyone the mesh is printable.

## Step 3 — show the result

Call `render_preview` on the GLB (default iso view; add
`front,back,top,side` when the user wants a turnaround) and show the PNG(s)
to the user.

A preview tells you about silhouette and form and nothing about whether the
mesh is solid — see the 2mini caution. Do not infer thickness, wall depth
or interior structure from a render.

Report the `glb_path` plainly. If the user works in a game engine,
importing is their engine's job (Godot: copy into the project, then
`godot --headless --import`). The mesh arrives with no material, so they
will want to assign one.

## Step 4 — export for printing (when asked)

`export_stl` converts a GLB into an STL a slicer will accept. Two
conversions it does that are silent failures if skipped: STL carries no
units and every slicer reads it as millimetres, so the mesh is scaled to a
real `height_mm` (default 120) instead of arriving as a 2mm trinket; and
glTF is Y-up while slicers are Z-up, so it is rotated instead of landing on
its side. It also drops the model onto z=0 and centres it.

Read the result properly before calling anything printable:

- `checks` — watertight, consistent winding, enclosed volume, single body.
- `bbox_fill_pct` — **the one that matters and the one a render cannot
  show.** A hollow shell and a solid can both pass every check above with
  identical silhouettes. Under ~15% means walls thin enough that the slicer
  may drop them.
- `finest_detail_mm` — warns when the finest feature present falls under
  roughly two perimeters of a 0.4mm nozzle. It will often warn; that means
  the smallest details will soften, not that the print will fail.
- `genus` — tunnels through the surface. Slices fine, but worth checking it
  is intentional rather than a reconstruction artefact.

## Iterating

- Wrong shape → change the concept image, not the generator knobs. The mesh
  follows the picture.
- Missing surface relief → not fixable here. See above; it is the model's
  range, not a knob you have not found.
- Soft or mushy **form** → check the concept first. Dramatic lighting, a
  background gradient, or a surviving shadow cost more than any knob buys,
  and a regenerated concept is seconds against minutes.
- Thin parts fused into the hull → this is the one case for `octree` 512.
- Phantom pancake of geometry under the model → a shadow survived in the
  concept; regenerate with "floating, no shadow".
- Too heavy for the target engine → lower `max_faces`. Decimation happens
  inside the generation call, on the raw mesh, and preserves
  watertightness. Do not run a separate remesh or repair pass over a
  finished GLB.
- User wants colour → they need a texturing step outside this server.
