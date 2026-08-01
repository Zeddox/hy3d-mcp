---
name: create-3d-model
description: Create a textured 3D model (GLB) from a text prompt or a concept image, fully locally, using the hy3d-gen MCP server. Use when the user wants a 3D model, asset, mesh, or GLB of something they describe or have a picture of.
---

# Create a 3D model from a prompt or image

You have the `hy3d-gen` MCP server (bundled with this plugin): local
image-to-3D via Hunyuan3D-MLX. The pipeline is concept image → RGBA cutout
→ shape + PBR paint → GLB. Your job is to get the user from "I want a 3D
model of X" to a GLB file path, with a preview when possible.

## First use in a session

Call `server_status` once before the first generation. Don't re-check on
later calls.

If any check is `ok: false`, the engine behind this server isn't set up
yet. Call `setup_engine` — it defaults to a dry run, changes nothing, and
returns the plan. Show that plan to the user, costs included (a ~4 minute
`swift build`, a ~12GB weight download), and only call it again with
`confirm=true` once they agree. Phases are idempotent, so a re-run after
a failure resumes rather than restarting.

If they would rather do it by hand, relay the failing check's `fix` text
verbatim instead — every failure carries its exact remedy.

## Step 1 — get a concept image

**If the user supplied an image**, use it directly and go to Step 2.

**If the user gave a text prompt**, generate a concept image with whatever
image-generation tool is available in the session (e.g. a Gemini or other
image-gen MCP). If none is available, ask the user for an image — do not
try to proceed without one.

The concept image makes or breaks the model. Compose the image prompt
from the user's description plus ALL of these constraints:

- **single object**, whole object in frame, roughly centered
- **three-quarter view** (shows front and side; best geometry recovery)
- **plain, uniform light-gray background** — nothing else in frame
- **naturally lit** — normal soft studio lighting. Never request "flat
  lighting", "unlit", or "albedo style": the generator de-lights
  internally, and pre-flattened input bakes pale and featureless.
- **no drop shadow, no contact shadow** — shadows under the object are
  reconstructed as literal geometry. Say "floating, no shadow" in the
  prompt.
- no text, watermark, or frame

Show the concept to the user before spending generation time, unless they
asked you to just go ahead.

## Step 2 — generate

Call `generate_model` with the image path. Defaults are right for most
cases; the knobs that matter:

- `paint: false` for a ~20-second shape-only draft when the user is
  iterating on silhouette; full shape+paint takes ~3–4 minutes.
- `auto_cutout` (default true) keys the background out automatically. If
  it errors saying the corners disagree, the background isn't plain —
  regenerate the concept, don't fight the cutout.
- `finish: true` applies a stylized dark-hull game look (toned-down
  albedo, glowing accent/seam emissive). It was tuned for dark sci-fi
  game assets — leave it off for general-purpose models unless the user
  wants that look; `finish_model` can always be applied to the GLB later,
  and re-run with different knobs without regenerating.

Generation is serialized on the server (one job at a time, it's
memory-bound) — a second call queuing behind a first is normal, not a
hang. Do not fire generations in parallel expecting speedup.

### Fidelity knobs

These raise detail on a shape that is already correct. They will not fix
a wrong shape — that is always the concept image's job. Each is unset by
default, leaving the binary's own default (in parens) in force; the
binary validates none of them, so move one knob a single step at a time
and keep the seed fixed so you can attribute the difference.

- `octree` (256) — marching-cubes resolution, the geometry lever. Try 384
  when thin parts (struts, masts, antennae) come out fused into the hull.
  Vertex count grows roughly cubically, and meshes are already ~70-80k at
  the default.
- `paint_res` (512) — resolution the multiview texture diffusion runs at.
  The strongest texture-sharpness lever, and the one to reach for when
  fine surface markings smear.
- `paint_steps` (15) — texture diffusion steps, notably low next to
  shape's 30. Usually the cheapest texture win.
- `texture_size` (2048 here, but the binary's own pbr default is 4096) —
  baked texture resolution.
- `steps` (30) and `guidance` (5.0) — shape diffusion convergence and how
  tightly the mesh follows the image. Least interesting of the set;
  diminishing returns past ~50 steps, and high guidance over-sharpens.

Paint peaks ~25-33GB unified memory, and `octree`, `paint_res` and
`texture_size` each multiply that. Raising several at once is the usual
way to turn a 4-minute job into a swap-thrashing one.

## Step 3 — show the result

Call `render_preview` on the GLB (default iso view; add
`front,back,top,side` when the user wants a full turnaround) and show the
PNG(s) to the user. If it errors about pyrender, give the user the
install command from the error and move on — the GLB is still good.

Report the `glb_path` plainly. If the user works in a game engine,
importing is their engine's job (Godot: copy into the project, then
`godot --headless --import`).

## Iterating

- Wrong shape → change the concept image, not the generator knobs. The
  mesh follows the picture.
- Right shape but soft or fused detail → that is the one case the
  fidelity knobs above address; raise `octree` for geometry, `paint_res`
  for texture.
- Pale/washed-out texture → the concept was too flat-lit; regenerate it
  naturally lit.
- Phantom pancake of geometry under the model → a shadow survived in the
  concept; regenerate with "floating, no shadow".
- Same concept + same `seed` reproduces a model; change `seed` to reroll
  geometry from the same image.
- Do NOT attempt mesh post-processing (decimation, repair, remeshing) on
  the output — it destroys generated meshes. LODs belong to the engine
  importer.
