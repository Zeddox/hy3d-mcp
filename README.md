# hy3d-mcp

An MCP server that turns a single concept image into a game-ready textured
3D model (GLB), fully locally on Apple Silicon, by wrapping the
[Hunyuan3D-MLX](https://github.com/ZimengXiong/hunyuan3d-mlx) pipeline
(Swift + MLX). One tool call: background cutout → shape → PBR paint →
optional game-look finishing pass. Proven in production on a real game
fleet.

Models land as **file paths**, never blobs — importing them into your
engine is the caller's job (for Godot: copy into the project and run
`godot --headless --import`).

## Requirements

- Apple Silicon Mac with ~48GB unified memory (texture paint peaks
  ~25–33GB)
- A built [Hunyuan3D-MLX](https://github.com/ZimengXiong/hunyuan3d-mlx)
  checkout with weights downloaded (~12GB)
- A "worker" Python environment with `opencv-python numpy trimesh pillow
  scipy` (and `pyrender` if you want `render_preview`)
- [uv](https://docs.astral.sh/uv/) to run the server

The server itself carries no ML dependencies; it shells out to the Swift
binary and the worker venv.

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
    "HY3D_REPO": "~/git/repos/hunyuan3d-mlx",
    "HY3D_PY": "~/git/repos/trellis-mac/.venv/bin/python",
    "HY3D_OUT": "~/hy3d-output"
  }
}
```

All three env vars are optional; the values above are the defaults.
`HY3D_PY` is any python interpreter with the worker packages installed.

**First run: call the `server_status` tool.** It validates every setup
requirement and each failing check carries the exact fix — including the
three gotchas below, which every fresh Hunyuan3D-MLX build hits.

## The three setup gotchas

1. **Metallib** — `swift build` never emits the MLX metallib (mlx-swift
   SwiftPM limitation). Harvest `mlx.metallib` from the pip `mlx` wheel of
   the *same version* as Package.resolved, and copy it as both
   `mlx.metallib` and `default.metallib` into `metallib/` **and** into
   `.build/arm64-apple-macosx/release/` (the real dir — `.build/release`
   is a symlink).
2. **Weight layout** — the paint-large HF repo ships flat, the binary
   expects nested: symlink `hunyuan3d-paint-v2-0/{vae,unet}` and
   `hunyuan3d-paintpbr-v2-1/{vae,unet}` → `../vae`, `../unet`, and
   `dinov2-giant` → `dinov2`, inside `weights/paint-large`.
3. **Paint model flag** — the server always passes `--paint-model pbr`;
   the rgb default targets a weight set that isn't installed.

## Tools

| Tool | What it does | Typical time |
| --- | --- | --- |
| `generate_model` | image → textured GLB (auto cutout, optional finish) | ~3–4 min (shape only: ~20s) |
| `prepare_concept` | plain-background image → centered square RGBA | seconds |
| `finish_model` | game-look texture pass: toned albedo + accent/seam emissive | seconds |
| `render_preview` | offscreen PNG renders (iso/front/back/top/side) | seconds |
| `server_status` | full setup diagnostic, queue depth, last job | instant |

Generation is serialized — one job at a time; concurrent calls queue
rather than OOM the machine.

## Input doctrine

- Feed **naturally lit** concept art — the model de-lights internally.
  Pre-flattened "albedo-style" input bakes pale and featureless.
- **No drop shadows** in the source image — they reconstruct as literal
  geometry under the model.
- Single object, plain background, roughly centered; ¾ view works best.
  `prepare_concept` / `auto_cutout` handle the background keying.

## Non-goals

- No cloud fallback.
- No mesh post-processing (decimation/repair proved destructive on
  generated meshes; LODs belong to your engine's importer).
- No batch tool — loop `generate_model`; the queue serializes.

## License

MIT — but that covers **this wrapper code only**. This repo distributes no
model weights and no Tencent code.

### Model weights license (read this)

The pipeline runs on Tencent's Hunyuan3D weights, which you download
yourself and which are governed by the **Tencent Hunyuan 3D 2.0 / 2.1
Community License Agreements** ([2.0](https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE),
[2.1](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE)) —
the paint stage uses both generations, so both apply. Highlights, not
legal advice; read the licenses:

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

The [Hunyuan3D-MLX](https://github.com/ZimengXiong/Hunyuan3D-MLX) Swift
port this server shells out to is itself MIT-licensed.
