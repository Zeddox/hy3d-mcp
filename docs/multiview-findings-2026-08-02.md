# Multiview image-to-3D: findings

Date: 2026-08-02
Context: investigating whether feeding multiple views of an object produces
better geometry than the current single-image pipeline.

## 1. The current pipeline is single-image at every entry point

Both engine subcommands take exactly one image:

- `Sources/hy3d/Generate.swift:16`
- `Sources/hy3d/Shape.swift:21`
- `Sources/Hy3DMLX/ShapeGenerator.swift:95-106` — one `CGImage` in, DINOv2
  embeds it, the DiT conditions on that single embedding.

There is no code path that accepts a second view.

## 2. Feeding a contact sheet does not work

Feeding the 6-view sheet `enterprise-d-octree384.glb.views.png` back through
shape-only produced **a dozen separate blobby mini-Enterprises scattered
across a plane** (41,960 verts, 15.7s). The model reads a contact sheet as a
scene containing many objects, not as one object seen six ways.

Do not try this again.

## 3. The baseline is better than the generator's own contact sheets suggest

The `.rendercheck.png` and `.views.png` sheets the paint pass writes are small
and heavily foreshortened. Judging a mesh from them produces false defects.

Concretely, `cannon-hy.glb` looked like it had a flat slab base and a mushy
rear ammo drum in the rendercheck sheet. At full `render_preview` resolution
both are fine: the cog-toothed base ring is present with individual teeth, and
the ammo drum is resolved down to the individual feed rounds.

**Judge geometry with `render_preview`, never with the generator sheets.**
The sheets are for confirming the paint pass ran, not for assessing shape.

## 4. The real blocker is authoring consistent views, not the 3D model

This is the finding that matters.

Gemini image editing (nano-banana-2) **cannot produce consistent multiview
concept art of a hard-surface object.** Three attempts from the same source
image (`intermediate/cannon-rgba.png`):

| Attempt | Requested | What came back |
|---|---|---|
| 1 | left profile | barrels pointed the wrong way; **invented a second cyan accent strip**; still a 3/4 angle |
| 2 | left profile | usable — one cyan strip, correct features, near-orthographic |
| 3 | rear elevation | **two ammo drums instead of one**; invented a rotary turbine on the rear face; proportions squatter; different scale |

Every attempt also kept a soft contact shadow despite explicit instructions
against one.

The generator has no persistent 3D representation, so each view is an
independent hallucination that happens to be conditioned on the source. Detail
count drifts, and drift makes any multiview test unfalsifiable — you cannot
tell a 3D model failing to fuse views from views that genuinely disagree.

This is exactly the problem Google's CAT3D (NeurIPS 2024) was built to solve,
and there is no released implementation of it.

### Consequence for the pipeline

Multiview only helps when you can *author* consistent views:

- **Hand-drawn orthographic turnarounds** — works, but that is concept-artist
  labour, not a generative shortcut.
- **Rendering an existing mesh** — works, but if the mesh already exists you
  do not need to generate it.
- **Generating views with an image model** — does not work today.

So even a perfect multiview 3D model does not currently unlock a faster path
from "idea" to "asset". It unlocks a path from "blockout or turnaround" to
"asset", which is a different and much narrower workflow.

## 4a. Trial result, and a confound

Run through `huggingface.co/spaces/tencent/Hunyuan3D-2mv` with three views of
the cannon turret (front + both flanks, ammo drum edited out). **The mesh could
not be downloaded from the Space**, so the result is a visual judgement made in
the browser: the multiview output carried *more accurate detail* than the
single-view generation.

That observation is real, but it does not isolate multiview, because the views
fed to 2mv were **renders of a finished mesh** — evenly lit, no baked
photographic shading, no shadow, clean background — while the single-image
baseline was built from painted concept art. 2mv got a materially easier input.

### Control run

Same subject, same seed (42), same defaults, **one** view — the baseline's own
iso render, background keyed out:

| Run | Input | Verts | Faces |
|---|---|---|---|
| baseline `cannon-hy.glb` | painted concept art, 3/4 | 77,210 | 154,412 |
| control `control-single-clean.glb` | single clean render, 3/4 | **101,088** | **202,184** |

**+31% geometry from input quality alone, with no multiview involved.** The
control's hull facets, muzzle brakes and drum teeth are all visibly crisper
than the baseline's.

So an unknown but substantial share of the improvement seen in the 2mv trial is
attributable to clean input rather than to multiview conditioning. The two
cannot be separated without the 2mv mesh in hand.

Caveat: n=1, one subject, one seed. Treat the 31% as directional, not a
measured effect size.

### The transferable lesson

The concept image that produced the baseline (`intermediate/cannon-rgba.png`)
carries a soft contact shadow and photographic studio shading. The render that
produced the control carries neither. For a **new** asset — where no mesh exists
to render — the reachable version of this win is a cleaner concept image:
flatter even lighting, hard background separation, genuinely no shadow.

## 5. Other models

- **Hunyuan3D-2mv** — genuine multiview-conditioned shape generation. Free to
  try at `huggingface.co/spaces/tencent/Hunyuan3D-2mv` (ZeroGPU). Locally it
  runs via ComfyUI's `Hunyuan3Dv2ConditioningMultiView` node on Apple Silicon,
  but **geometry only** — Hunyuan's texture stage needs a CUDA rasterizer that
  does not exist on this hardware.
- **TRELLIS.2** — no multi-image API (`trellis2_image_to_3d.py:489` takes a
  single `image`). v1 had `run_multi_image`; v2 dropped it. Previously
  evaluated here and rejected for noisy output.
- **Google** — no product. CAT3D and DreamFusion are papers with no released
  weights or API. Nothing in Gemini or Vertex emits a mesh.

## 6. Texture is conditioned on one image regardless

2mv's multiview conditioning is on the **shape** DiT. The paint pass still
takes a single image, so single-image texture artifacts (e.g. the mirrored
registry lettering on the Enterprise-D) are **not** addressed by any of this.

## Unexposed capability worth noting

The engine has a standalone paint subcommand the MCP server does not expose
(`Sources/hy3d/Paint.swift:8`, registered at `main.swift:13`):

```
hy3d paint <mesh> <image> -o out.glb --paint-weights weights/paint-large --paint-model pbr
```

This textures an *existing* mesh. It is the piece that would let a mesh
generated elsewhere (ComfyUI multiview, a blockout, a downloaded model) be
textured by the working MLX paint stage, sidestepping the missing CUDA
rasterizer. Worth exposing as a `paint_mesh` tool if the multiview path is
ever pursued.
