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
- Xcode or the Command Line Tools (for `swift`), and
  [uv](https://docs.astral.sh/uv/)
- ~15GB free disk: 12GB of weights, 1.3GB of build output
- A built [Hunyuan3D-MLX](https://github.com/ZimengXiong/hunyuan3d-mlx)
  checkout and a worker Python environment — **[`./install.sh`](#set-up-the-engine)
  builds both for you**; see that section before doing any of it by hand.

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

## Set up the engine

The server is a thin wrapper — the actual pipeline is a separate Swift
checkout that has to be cloned, built, and fed 12GB of weights. Either
let the installer do it or follow the manual sequence below; both end at
the same place.

### The installer

```sh
./install.sh --plan     # print exactly what it would do, change nothing
./install.sh            # do it, confirming the build and the download
```

Seven phases — preflight, clone, `swift build`, metallib, weights,
paint-large relayout, worker venv. **Every phase inspects before it
acts**, so it is safe to re-run: finished work is skipped and a failed
run resumes where it stopped. The cheap and idempotent phases run
unattended; the two expensive ones (a ~4 minute build, a ~12GB download)
stop and ask first. `--yes` runs unattended, `--only N` runs one phase,
and `--repo` / `--worker-venv` relocate the targets.

From inside an MCP client, the `setup_engine` tool is the same script.
It defaults to a dry run and returns the plan; it only executes when
called again with `confirm=true`, so the agent has to show you the cost
before spending it.

When it finishes it prints the `HY3D_REPO` and `HY3D_PY` values to put in
your MCP config, and `server_status` should then come back all green.

### Or by hand

The engine's own
[README](https://github.com/ZimengXiong/Hunyuan3D-MLX) covers steps 2–4;
steps 5–7 are the parts it does not mention.

```sh
# 1. clone
git clone https://github.com/ZimengXiong/Hunyuan3D-MLX.git ~/git/repos/hunyuan3d-mlx
cd ~/git/repos/hunyuan3d-mlx

# 2. build (~4 min)
swift build -c release

# 3-4. weights (~12GB)
uvx --from huggingface_hub hf download \
  zimengxiong/hunyuan3d-mlx-shape-small --local-dir weights/shape-small
uvx --from huggingface_hub hf download \
  zimengxiong/hunyuan3d-mlx-paint-large --local-dir weights/paint-large

# 5. metallib — swift build never emits it; harvest it from the pip mlx wheel.
#    NOTE: mlx-swift and pip mlx are separate version series. Package.resolved
#    pins mlx-swift 0.31.4, but no such pip release exists — take the newest
#    pip mlx in the matching 0.31.x series (0.31.2 at time of writing).
uv venv /tmp/mlxharvest
uv pip install --python /tmp/mlxharvest/bin/python mlx==0.31.2
SRC=$(find /tmp/mlxharvest -name mlx.metallib | head -1)
for d in metallib .build/arm64-apple-macosx/release; do
  mkdir -p "$d" && cp "$SRC" "$d/mlx.metallib" && cp "$SRC" "$d/default.metallib"
done

# 6. paint-large ships flat, the binary wants it nested
cd weights/paint-large
mkdir -p hunyuan3d-paint-v2-0 hunyuan3d-paintpbr-v2-1
ln -s ../vae ../unet hunyuan3d-paint-v2-0/
ln -s ../vae ../unet hunyuan3d-paintpbr-v2-1/
ln -s dinov2 dinov2-giant
cd ../..

# 7. worker venv — uv, not pip: uv-created venvs have no pip in them
uv venv ~/.hy3d/worker-venv
uv pip install --python ~/.hy3d/worker-venv/bin/python \
  opencv-python numpy trimesh pillow scipy pyrender
```

Then set `HY3D_REPO=~/git/repos/hunyuan3d-mlx` and
`HY3D_PY=~/.hy3d/worker-venv/bin/python`.

**Either way, verify with the `server_status` tool.** It re-checks every
requirement and each failing check carries its own fix.

## The three setup gotchas

`install.sh` handles all three; they are documented here because they are
what a by-the-book install of the upstream repo gets wrong, and what
`server_status` is looking for when it fails.

1. **Metallib** — `swift build` never emits the MLX metallib (mlx-swift
   SwiftPM limitation). Harvest `mlx.metallib` from the pip `mlx` wheel —
   *not* the version string in Package.resolved, which is mlx-**swift**'s
   own series and has no pip counterpart (there is no pip `mlx` 0.31.4).
   Take the newest pip `mlx` sharing its major.minor, and copy it as both
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
| `setup_engine` | runs `install.sh`; dry run unless `confirm=true` | instant (plan) / up to an hour (apply) |

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
