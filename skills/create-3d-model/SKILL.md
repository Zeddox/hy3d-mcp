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

The concept image makes or breaks the model, and it is the single biggest
lever on output quality — bigger than any generator knob. Measured on one
subject at identical seed and defaults, a clean evenly-lit input produced
**31% more geometry** (101k vs 77k verts) and visibly crisper panel and
edge detail than the painted concept art of the same object. Spend effort
here before reaching for `octree`.

Compose the image prompt from the user's description plus ALL of these:

- **single object**, whole object in frame, roughly centered
- **three-quarter view** (shows front and side; best geometry recovery)
- **plain, uniform background** — nothing else in frame, and a flat single
  tone rather than a gradient or vignette. The cutout keys on corner
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
  the generator de-lights internally, and pre-flattened input bakes pale
  and featureless. Even and soft, not absent.
- **no drop shadow, no contact shadow** — shadows under the object are
  reconstructed as literal geometry. Say "floating, no shadow" in the
  prompt.
- no text, watermark, or frame

**Then look at what came back before generating.** Image generators
routinely ignore the shadow instruction, and a soft contact shadow is easy
to miss against a gray background. Check for it explicitly, along with a
background gradient or a hard key light. Regenerating a concept costs
seconds; a bad concept costs a 4-minute paint pass and still has to be
redone. This is worth one deliberate look, not a glance.

**Better than looking: measure the key.** Run `prepare_concept` and check
the alpha channel of the RGBA it writes, because that — not the source
image — is what conditions the model. A clean key is almost entirely alpha
0 or 255 with well under 1% in between.

What the histogram catches is **background non-uniformity**, which is easy
to miss by eye. One observed gradient-gray background keyed at 36% partial
alpha and left patches of background gray fully **opaque** — the key had
smeared rather than separated. A flat white background of the same subject
keyed at 0.1% partial. Note this measures the background, not the shadow
specifically; the two were not varied independently.

Show the concept to the user before spending generation time, unless they
asked you to just go ahead.

## Step 2 — generate

Call `generate_model` with the image path. Defaults are right for most
cases; the knobs that matter:

- `paint: false` for a ~20-second shape-only draft when the user is
  iterating on silhouette; full shape+paint takes ~3–4 minutes. It runs a
  different engine subcommand, so `paint_res`, `paint_steps` and `finish`
  are rejected alongside it, and no preview sheet is written.
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
hang. Do not fire generations in parallel expecting speedup. The call
streams progress the whole way, so a long job is visibly working; a
detailed concept can legitimately paint for 13+ minutes. If a call is
ever abandoned mid-flight and later calls appear to hang, `cancel_job`
frees the queue.

Outputs get the glTF `NORMAL` attribute injected by default, which the
engine itself omits. Leave `normals` alone — without it Godot lights the
whole mesh off one constant vector, which looks like a bad material
rather than a missing attribute.

### Fidelity knobs

These raise detail on a shape that is already correct. They will not fix
a wrong shape — that is always the concept image's job. Each is unset by
default, leaving the binary's own default (in parens) in force; the
binary validates none of them, so move one knob a single step at a time
and keep the seed fixed so you can attribute the difference.

- `octree` (256) — marching-cubes resolution, the geometry lever. Try 384
  when thin parts (struts, masts, antennae) come out **fused into the
  hull** — that specific failure, not as a general quality dial. Its cost
  scales with how much fine detail the concept carries, not just with the
  number: a smooth hull at 384 took ~6 minutes, while a lattice/greeble-
  heavy subject took 16.5 and drove the machine into swap, having already
  resolved its truss braces fine at the default. Vertex count grows
  roughly cubically and meshes are already ~70-80k at the default, with
  ~4.5× spread across subjects at identical settings. Those are
  shape-stage figures; a painted model reports higher because the texture
  pass splits vertices along UV seams — one turret left the shape stage at
  63.5k and reached disk at 91k. Triangle count is unaffected, so size
  LODs off `faces`, not `verts`.
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
PNG(s) to the user.

Check `source` in the reply. `"rendered"` means you got the views you
asked for. `"generator_sheets"` means rasterising failed — most often a
worker venv still on the pyopengl 3.1.0 that pyrender pins, which cannot
render textured meshes; `server_status`'s `preview_ok` says so outright and
carries the one-line upgrade. Note it only breaks on *textured* meshes, so
the venv looks healthy until the first painted GLB. You are looking at the
multiview and render-check contact sheets the paint pass wrote beside the
GLB instead. Say so rather than describing them as the requested angles.
Only shape-only output has no sheets, and there the call errors outright —
the GLB is still fine.

**Do not judge geometry from the sheets.** They are small, heavily
foreshortened, and good only for confirming the paint pass ran. Reading
them as if they were renders has produced confident, entirely false defect
reports — a base ring that looked like a flat slab and a rear assembly
that looked fused both turned out sharp and correct at full
`render_preview` resolution. If `source` is `"generator_sheets"`, you do
not yet know whether the mesh is good; say that plainly instead of
assessing shape from them.

Report the `glb_path` plainly. If the user works in a game engine,
importing is their engine's job (Godot: copy into the project, then
`godot --headless --import`).

## Texturing a mesh you already have

`paint_mesh` runs the paint half on its own — an existing mesh plus a
concept image in, a textured GLB out, geometry untouched. Two uses:
re-texturing at different knobs without paying for the shape pass again,
and texturing geometry this engine did not make (a blockout, a sculpt, a
reconstruction from somewhere else). `.glb`, `.gltf` and `.obj` load
directly; anything else falls back to ModelIO and is flagged as a warning
rather than refused.

The conditioning image follows the same rules as Step 1, and `auto_cutout`
matters more here than it looks: the paint pipeline composites existing
alpha over white but does no background keying of its own, so an unkeyed
backdrop ends up conditioning the texture instead of being ignored.

There is no `seed`. The paint pipeline re-seeds to 0 internally, so the
same mesh and image always produce the same texture — rerolling is not
available the way it is on shape. Change the image or the knobs instead.

The image cannot move a vertex. Where it disagrees with the mesh you get
smeared projection, not corrected geometry.

Paint dominates the pipeline's runtime, so what this saves is the shape
pass — tens of seconds — not most of the wait. Reach for it because you
have geometry that needs texturing, not to go faster.

Progress on this call is heartbeat only. The paint subcommand does not emit
the percentage lines `generate` does, so the bar creeps toward its ceiling
on elapsed time rather than tracking real completion — a run sitting at
~43% after three minutes is normal, not stalled. The elapsed time in the
message is the honest part.

## Iterating

- Wrong shape → change the concept image, not the generator knobs. The
  mesh follows the picture.
- Soft or mushy detail → check the concept **first**, before touching
  `octree`. Dramatic lighting, a background gradient, or a surviving
  shadow cost more detail than a resolution step buys, and a regenerated
  concept is seconds against minutes. Only once the concept is clean and
  evenly lit is this the case the fidelity knobs address — `octree` for
  geometry, `paint_res` for texture.
- Detail that looks fused in a contact sheet → re-check with
  `render_preview` before believing it. See Step 3.
- Pale/washed-out texture → the concept was too flat-lit; regenerate it
  naturally lit.
- Phantom pancake of geometry under the model → a shadow survived in the
  concept; regenerate with "floating, no shadow".
- Same concept + same `seed` reproduces a model; change `seed` to reroll
  geometry from the same image.
- Do NOT attempt mesh post-processing (decimation, repair, remeshing) on
  the output — it destroys generated meshes. LODs belong to the engine
  importer.
