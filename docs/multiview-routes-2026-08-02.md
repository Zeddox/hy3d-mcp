# Multiview image-to-3D: implementation routes

Date: 2026-08-02
Status: **superseded 2026-09-06 — shipped in 0.9.0, by none of these routes.**
See "Outcome" at the foot of this document; every route below assumed an
Apple Silicon/MLX engine, and the port to CUDA made the whole question a
different one.
Prior art: [`multiview-findings-2026-08-02.md`](multiview-findings-2026-08-02.md),
which investigated whether multiview is worth pursuing. This document assumes
we pursue it and specs *how*.

## Why this is worth reopening

Two facts changed after the findings doc was written, both on the same day.

**1. `paint_mesh` shipped (v0.4.0).** The findings doc closed by naming the
standalone paint subcommand as "the piece that would let a mesh generated
elsewhere be textured by the working MLX paint stage" and marked it unexposed.
It is now an MCP tool. Any route that produces geometry elsewhere can now be
textured locally without a second texture stack.

**2. The 2mv checkpoint is architecturally identical to the one we run.**
The findings doc assumed a native MLX port was out of reach. Diffing
`tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/config.yaml` against the local
`weights/shape-small/config.yaml` says otherwise:

| | `shape-small` (dit-v2-mini) | `dit-v2-mv` | classification |
|---|---|---|---|
| `in_channels` | 64 | 64 | same |
| `context_in_dim` | 1536 | 1536 | same |
| `hidden_size` | 1024 | 1024 | same |
| `num_heads` | 16 | 16 | same |
| `mlp_ratio` | 4.0 | 4.0 | same |
| `axes_dim` / `theta` / `qkv_bias` | [64] / 10000 / true | [64] / 10000 / true | same |
| VAE `embed_dim` / `width` | 64 / 1024 | 64 / 1024 | same |
| VAE `heads` / `num_decoder_layers` | 16 / 16 | 16 / 16 | same |
| VAE `num_freqs` / `include_pi` | 8 / false | 8 / false | same |
| `depth` | 8 | **16** | scalar |
| `depth_single_blocks` | 16 | **32** | scalar |
| VAE `num_latents` | 512 | **3072** | scalar |
| VAE `scale_factor` | 1.0188137142395404 | **0.9990943042622529** | scalar |
| `guidance_embed` | false | **absent** (defaults false) | scalar |
| image encoder | `DinoImageEncoder` | **`DinoImageEncoderMV`** | **new code** |
| image processor | `ImageProcessorV2` | **`MVImageProcessorV2`** | **new code** |
| DINOv2 sub-config | dino giant, 40 layers, 1536 | identical | same |

Every *structural* hyperparameter matches. The DiT and VAE differences are
loop counts and floats that the MLX loader already reads from `config.yaml`
(the `weights/shape-small/README.md` states the loader parses model geometry —
e.g. the VAE scale factor — out of it). So the transformer itself plausibly
needs **no new Swift**. See open question 2 — this is inferred from the config,
not yet demonstrated.

Supporting: the repo already ships weight converters
(`python/shape/hy3dmlx/convert.py`, `python/shape/scripts/convert_v21_ckpt.py`),
and the MLX weight bundles are a **layout repack of unmodified tensors**, not a
numerical conversion.

## What has NOT changed

These constraints from the findings doc still hold and are not addressed by any
route below.

- **Authoring consistent views is still the blocker for new assets.** Image
  models cannot hold a hard-surface design across viewpoints (three documented
  failures: invented accent strips, duplicated ammo drum, invented turbine).
  Multiview serves *blockout / turnaround / reference-photo* workflows. It does
  not shorten text-prompt-to-asset.
- **Texture stays single-image.** 2mv's multiview conditioning is on the shape
  DiT only. The paint pass takes one image on every route, so single-image
  texture artifacts (mirrored lettering and similar) are untouched.
- **Judge geometry with `render_preview`, never the generator sheets.**

## Phase 0 — settle the premise first (cheap, no commitment)

The case for multiview is currently **unmeasured**. The one positive
observation in the findings doc was confounded: 2mv received clean renders of a
finished mesh while the baseline received painted concept art, and the control
run showed **+31% geometry from input quality alone** with no multiview
involved.

Route A/B below would introduce a *second* confound: 2mv is the full-size model
(depth 16/32, 3072 latents) against our mini (8/16, 512). A win could be model
size rather than view count.

The only design that isolates multiview holds the model fixed:

> **2mv with 1 view vs 2mv with N views.** Identical render source, same seed,
> same subject.

Mechanism: drive the public Space `tencent/Hunyuan3D-2mv` with `gradio_client`
rather than the browser. The findings doc records that the mesh could not be
downloaded through the browser UI; `gradio_client` typically returns output
file paths where the UI does not. Verify this before assuming Phase 0 is cheap
(open question 4).

Subject: reuse the cannon turret, so results are comparable to the existing
`cannon-hy.glb` (77,210 verts) and `control-single-clean.glb` (101,088 verts)
figures. Compare at full `render_preview` resolution.

Cost: roughly an hour. No fork, no 5 GB download, no engine changes.

## Route A — native MLX multiview (fork the Swift engine)

Multiview lands in the same binary as everything else. Best end state; requires
owning a fork.

Work items:

1. Fork `ZimengXiong/Hunyuan3D-MLX`. The local clone at
   `~/git/repos/hunyuan3d-mlx` is currently clean and tracking `origin/main`.
2. Convert `tencent/Hunyuan3D-2mv` → MLX layout with the existing converter
   (`python/shape/hy3dmlx/convert.py`). Must confirm it emits the
   view-embedding tensors (open question 1).
3. Swift `DinoImageEncoderMV`: load the learned per-view embedding table, add
   it to each view's DINOv2 tokens, concatenate on the **token** axis.
   The current single-view path is `Sources/Hy3DMLX/ShapeGenerator.swift:100-106`:
   ```swift
   guard let pix = Preprocess.dinoPixels(cgImage: image) else { return nil }
   let embed = dino(pix)
   let cond = dit.guidanceEmbed ? embed : concatenated([embed, dino.unconditional(1)], axis: 0)
   ```
   Note the existing `axis: 0` is the classifier-free-guidance batch concat, not
   a view concat. View concatenation is a separate axis and must not be
   conflated with it.
4. Swift `MVImageProcessorV2`: per-view preprocessing (size 512, border_ratio
   0.15, same as the single-view processor — confirm whether it differs beyond
   being applied N times).
5. CLI: accept N images, e.g.
   `hy3d shape front.png left.png back.png -o out.glb --weights weights/shape-mv`.
6. Verify the DiT and VAE construct unchanged at depth 16/32 and
   `num_latents: 3072` purely from config (open question 2).
7. MCP surface: either `generate_model(images=[...])` or a separate
   `generate_multiview` tool. A separate tool avoids overloading a parameter
   that is single-image on every other path; decide when the engine side works.
8. Repoint the weights download in `install.sh` (the `download_weights()`
   phase, using `huggingface_hub.snapshot_download()`) and update its disk-space
   preflight — currently `~15GB (12GB weights, 1.3GB build)`, and the mv
   checkpoint is another 4.93 GB.

Cost: 4.93 GB of additional weights. Fork maintenance against an upstream we do
not control.

Risks: the two unverified assumptions (open questions 1 and 2), plus unknown
memory and wall time (open question 3).

## Route B — ComfyUI hybrid (no fork)

Geometry from ComfyUI, texture from the MLX paint stage already here.

- Install ComfyUI — **none on this machine** (verified: no install, no running
  process; only a `~/.hermes/skills/creative/comfyui` skill directory).
- Add the Hunyuan3D wrapper custom nodes and the 2mv weights.
- Use `Hunyuan3Dv2ConditioningMultiView` for geometry. Geometry only on Apple
  Silicon: Hunyuan's own texture stage needs a CUDA rasterizer that does not
  exist on this hardware.
- Texture the resulting mesh with `paint_mesh`. Formats: `.glb`, `.gltf` and
  `.obj` load directly; anything else falls back to ModelIO.

Cost: no Swift fork. A second Python stack, a cross-process handoff, and MCP
wiring to drive the ComfyUI API.

Risks: MPS support for these nodes is reportedly uneven. Two dependency trees
to keep working instead of one.

## Route C — upstream the MV support

Route A's work, submitted as a PR to `ZimengXiong/Hunyuan3D-MLX` instead of
carried as a permanent fork. Same engineering; the fork becomes temporary if it
lands. Slower, and merge timing is not in our control.

## Open questions to resolve before starting

1. **Does `python/shape/hy3dmlx/convert.py` handle the mv checkpoint's
   view-embedding tensors?** If not, the converter needs extending. Unverified.
2. **Is the MLX loader genuinely config-driven for `depth`,
   `depth_single_blocks` and `num_latents`?** The config diff implies zero DiT
   changes; a hardcoded assumption anywhere would break that. This is the
   cheapest high-value de-risking check: convert the mv weights, point
   `--weights` at them, and see whether the DiT and VAE construct at all —
   before writing any MV encoder code.
3. **Peak memory and wall time for depth 16/32 at 3072 latents on this
   machine.** The latent sequence is 6× longer, and attention cost grows
   superlinearly in sequence length; combined with 2× depth, shape could go from
   ~20s to minutes. Paint already peaks 25–33 GB, so headroom matters.
4. **Does `gradio_client` return a downloadable mesh from the 2mv Space?**
   Gates whether Phase 0 is actually cheap. The browser UI would not.
5. **Does 2mv accept a variable view count, or fixed slots with masking?** The
   model card demonstrates front/left/back but does not state a hard
   requirement. Determines whether the CLI takes N images or named view slots.
6. **Confirm the ComfyUI wrapper's `Hunyuan3Dv2ConditioningMultiView` node
   actually runs on MPS, and at what view count.** Route B rests entirely on
   this. It is carried over from the findings doc as an assertion whose
   provenance is unclear — it was not verified here. If the node is
   CUDA-bound in practice, Route B collapses and the choice is A or C.

## Reference

- Weights: `tencent/Hunyuan3D-2mv`, subfolder `hunyuan3d-dit-v2-mv`,
  `model.fp16.safetensors` 4.93 GB. License: `tencent-hunyuan-community`.
- Local weights today: `shape-small` 3.6 GB (`hunyuan3d-dit-v2-mini`),
  `paint-large` 8.1 GB.
- Engine repo: `ZimengXiong/Hunyuan3D-MLX` at `~/git/repos/hunyuan3d-mlx`.


## Outcome (2026-09-06)

None of routes A, B or C happened, because the premise underneath all three
went away: the engine moved from MLX to CUDA/PyTorch. Multiview on the ported
engine is not a port, a fork or a hybrid — it is upstream's own code path,
already in the repo we already run.

**What it took.** Four arguments on `engine_cli.py`, three parameters on
`generate_model`, a second workflow in the workbench, and one download. The
model and its dit subfolder had to move together (a `DITS` table): the old
default subfolder was `hunyuan3d-dit-v2-0`, which does not exist inside the
mv repo, and asking for it fails as a missing path several frames from the
choice that caused it.

**The open questions, answered.**

1. *Does the converter emit view-embedding tensors?* Moot — no conversion.
   `DinoImageEncoderMV` builds its view embedding from `sincos` at
   construction time (`conditioner.py:147-152`), so it is not a checkpoint
   tensor at all.
2. *Do the DiT and VAE construct unchanged?* Yes. The mv `config.yaml` is the
   base one with two lines different: `DinoImageEncoderMV` in place of
   `DinoImageEncoder`, and `MVImageProcessorV2` in place of
   `ImageProcessorV2`. Same depth 16/32, same `num_latents: 3072`, and the
   checkpoint is the same size to the byte (4,928,151,562).
3. *Memory and wall time?* Measured, not guessed: 6.69 GiB peak against a
   6.96 GiB ceiling and 152.6s, versus 6.22 GiB and 114.9s single-image at
   identical settings. +33% time, +0.47 GiB. Nothing spilled, but under
   0.3 GiB of headroom is not much to spend on a fourth view.
4. *`gradio_client` from the Space?* Moot — Phase 0 was a way to avoid running
   the model locally, and the model runs locally.
5. *Variable view count, or fixed slots?* **Fixed named slots, and a subset is
   accepted.** `MVImageProcessorV2.view2idx` is exactly
   `{front: 0, left: 1, back: 2, right: 3}`; `__call__` sorts the given views
   by index and concatenates. Passing three works; passing a fifth view, or a
   three-quarter one, has nowhere to go.
6. *ComfyUI wrapper on MPS?* Moot — no ComfyUI, no MPS.

**The API question (item 7 above), decided.** The spec leaned toward a
separate `generate_multiview` tool "to avoid overloading a parameter that is
single-image on every other path". It went the other way, as three optional
flat parameters — `left_image`, `back_image`, `right_image` — because the
concern does not apply to them: `image_path` keeps its meaning exactly, as
the front view, and nothing about a single-image call changes. A second tool
would have duplicated ten shared tuning parameters and the whole
queue/cutout/decimate body to vary two lines.

**What was not free.** The obvious `allow_patterns=["<subfolder>/*"]` costs
9.5GB, not 4.6GB: the repo ships the same fp16 weights twice, once as
`.safetensors` and once as `.ckpt`. `install.sh` now names the two files it
wants. The same bug was in the original shape download and is fixed with it.
