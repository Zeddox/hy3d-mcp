# Field report — 2026-08-01

Findings from a real asset run: nine planned turret models for a Godot 4.7
tower-defense game (AEGIS), of which two were baked end-to-end before the run
was cut short on art grounds rather than tool grounds. Four defects surfaced.
Two of them (§1, §2) make documented features unreachable; one (§3) leaves
orphaned compute running after the client has given up; one (§4) is a gap that
makes every output unusable in Godot without a post-step.

Environment: macOS Darwin 25.5.0, Apple M5 Pro. `hy3d-mcp` at `4f2d86d`
(Release 0.2.0). `hunyuan3d-mlx` at `HY3D_REPO=/Users/jimcline/git/repos/hunyuan3d-mlx`.
Consumer was Claude Code's MCP client.

Severity is from the caller's seat: how much it cost, and whether a workaround
existed.

---

## 1. `paint=false` is unreachable — the wrapper calls the wrong subcommand

**Severity: high.** The advertised cheap path does not exist in practice.

### Symptom

```
Error calling tool 'generate_model': hy3d generate failed (exit 1):
error: generate: missing --paint-weights <dir>
```

Raised immediately on any call with `paint=False`. The tool description
promises "`paint=False` skips texturing (shape only, ~20s vs ~3-4 min)".

### Root cause

`server.py:189` guards the paint flags behind the boolean:

```python
if paint:
    cmd += ["--paint-weights", "weights/paint-large",
            "--paint-model", "pbr", "--tex", str(texture_size)]
```

but the binary's `generate` subcommand requires that option unconditionally —
`Sources/hy3d/Generate.swift:12`:

```swift
guard let paintW = args.str("paint-weights") else {
    throw CLIError("generate: missing --paint-weights <dir>")
}
```

There is no `--no-paint` or `--shape-only` flag on `generate`. Shape-only is a
**separate subcommand**, `Sources/hy3d/Shape.swift:17`, which takes `--weights`
(not `--shape-weights`):

```swift
guard let weightsDir = args.str("weights") else {
    throw CLIError("shape: missing --weights <dir>")
}
```

So `paint=False` can never succeed while the wrapper always invokes `generate`.

### Suggested fix

Route on the boolean at the subcommand level, not the flag level: `paint=True`
→ `hy3d generate ... --shape-weights … --paint-weights …`; `paint=False` →
`hy3d shape … --weights <shape weights dir>`. The two subcommands take
different option names, so this is a branch in argv assembly rather than a
conditional append. Worth checking which of `octree`/`steps`/`guidance` `shape`
accepts, since the fidelity knobs should keep working on the cheap path.

### Why it mattered

This is the difference between a 20-second silhouette check and a 3-to-13-minute
commitment. With nine models planned and concept iteration expected, the cheap
path was the whole reason to iterate in the tool rather than in the image
generator. Losing it meant every silhouette question cost a full paint.

### Repro

```
generate_model(image_path="<any plain-background png>", paint=False)
```

---

## 2. `render_preview` cannot open a display and always crashes

**Severity: high.** The only preview path in the server is unusable from a
headless or non-foreground session, which is where an MCP server lives.

### Symptom

```
Error calling tool 'render_preview': preview.py failed: Traceback (most recent call last):
  File ".../workers/preview.py", line 105, in <module>
    main()
  File ".../workers/preview.py", line 64, in main
    renderer = pyrender.OffscreenRenderer(args.size, args.size)
  ...
  File ".../pyglet/display/cocoa.py", line 32, in get_default_screen
    return screens[0]
IndexError: list index out of range
```

### Root cause

`preview.py:64` constructs `pyrender.OffscreenRenderer`, which on this platform
selects pyglet's Cocoa backend and asks the window server for a screen list.
From a daemonised MCP worker there is no window server session attached, so the
list is empty and indexing it raises. Despite the name, this `OffscreenRenderer`
path still needs a display connection.

Notably the tool description claims "Offscreen renders of a GLB, **no engine
needed**", and the error-handling hint in the docstring only anticipates
pyrender being *absent*, not present-and-unable-to-open-a-context.

### Suggested fix

Two options, and the second is nearly free:

1. Force a genuinely headless GL backend before importing pyrender — set
   `PYOPENGL_PLATFORM=egl` (or `osmesa`) in the worker environment, and make
   the dependency explicit. This is the standard pyrender headless recipe.
2. **Fall back to the sheets the generator already wrote.** `generate_model`
   emits `<out>.glb.views.png` and `<out>.glb.rendercheck.png` next to the GLB.
   Those were the only previews I could actually look at all day, and they were
   good. If `render_preview` cannot open a context, returning the existing
   `.views.png` beats returning a traceback.

Doing (2) unconditionally as a fallback would have removed this from my path
entirely.

### Repro

```
render_preview(glb_path="<any generated glb>", views=["iso","front","top"])
```
from a server started outside a foreground GUI session.

---

## 3. Long jobs orphan: no progress heartbeat, and the worker survives the abort

**Severity: medium-high.** Costs real wall-clock and silently blocks the queue.

### Symptom

```
Task failed: MCP server "plugin:hy3d-gen:hy3d-gen" tool "generate_model"
sent no response or progress for 1803s; aborting.
```

After this abort, `server_status` still reported `queue_depth: 1`, and
`ps` showed the `hy3d` worker alive at 296% CPU, 31 minutes elapsed. The client
had given up; the compute had not. Because generation is serialised, that
orphan blocked every subsequent call until I found the pid and killed it by
hand.

Confirmed no progress reporting exists — `grep -rn "progress\|report_progress\|
send_progress\|notification" --include=*.py src` returns nothing.

### Suggested fix

Two independent halves, both worth doing:

- **Emit MCP progress notifications** during generation. Even a coarse
  stage-level heartbeat (`cutout` → `shape` → `paint` → `superres`) resets the
  client's idle timer and makes a slow job distinguishable from a hung one.
  Right now they are identical from outside.
- **Bind the worker's lifetime to the request.** If the tool call is cancelled
  or the connection drops, terminate the child process rather than leaving it
  holding the single-job queue. As it stands the only recovery is out-of-band
  `kill`, which a caller shouldn't have to reason about.

The README could also name the tunable the error message mentions
(`CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`, or a per-server `timeout` in MCP
settings), since a 13-minute paint is normal for a detailed concept and will
exceed common defaults.

---

## 4. Output GLBs carry no vertex normals, and `finish_model` does not add them

**Severity: medium.** Every output needs a post-step before it can be lit.

### Symptom

Both the raw generation and the finished output carry only `POSITION` and
`TEXCOORD_0`:

```
deck-hy.glb        attrs: ['POSITION', 'TEXCOORD_0']
cannon-hy.glb      attrs: ['POSITION', 'TEXCOORD_0']
cannon-finished.glb attrs: ['POSITION', 'TEXCOORD_0']
```

Godot does not synthesise the missing attribute. It hands the shader a single
constant normal for the entire mesh, so lighting, fresnel and every
normal-derived effect collapse into a flat wash. It does not look like a
missing attribute; it looks like a bad material, which is what makes it
expensive — this same hole cost hours on an earlier batch of ship hulls before
it was diagnosed.

`finish_model` is the natural place to close it: it already rewrites the
texture set, and its contract ("Geometry untouched") is arguably still honoured
by adding a derived attribute that changes no vertex position.

### Suggested fix

Compute and write `NORMAL` in the finish pass. **Inject rather than
round-trip** — re-exporting through trimesh rebuilds the material block and
destroys the emissive texture `finish_model` just wrote. The working approach:
load with `trimesh.load(..., process=False)` (vertex order must match the
POSITION accessor index-for-index; the default merges coincident vertices and
breaks that), read `mesh.vertex_normals`, then append a bufferView + accessor
with pygltflib and set `primitive.attributes.NORMAL`.

A complete working implementation is in the consuming repo at
`tower-defense/turrets/normals.py` — lift it if useful.

### Repro

```
python -c "
from pygltflib import GLTF2
g = GLTF2().load('<any output>.glb')
for m in g.meshes:
    for p in m.primitives:
        print([k for k,v in p.attributes.__dict__.items() if v is not None])
"
```

---

## Observations, not defects

**`octree` cost scales with concept detail, not just the number.** Two
octree-384 jobs on the same machine on the same day: a smooth-hulled subject
finished in ~6 minutes; a deck covered in fine truss lattice ran 16.5 minutes
and drove the machine to 4.9 GB of 6 GB swap before I killed it. At defaults
that same deck took 790s and still resolved the truss braces fine — the knob
bought nothing. The docstring's warning about raising several knobs at once is
good; a sentence noting that *lattice/greeble-heavy concepts are the expensive
case* would have steered me away from reaching for it at all.

**Vertex counts vary ~4.5× across subjects at identical settings.** Same
defaults: 82,478 verts for a gun housing, 366,576 for the trussed deck. Worth a
note, since the second is heavy enough to matter for a game asset and there is
no knob that trades it back.

**`accent_coverage_pct` came back 0.2% and 0.0%** for two concepts that both
carried deliberate bright-cyan emissive strips, so the emissive channel ended up
carrying nothing. Not a blocker for me — the consuming game keeps its own
procedural accent lights — but the extractor may be tuned for larger accent
areas than a hard-surface subject with thin indicator strips provides. If
that's expected, saying so in the `finish_model` docstring would save a caller
wondering whether they mis-specified the concept.

---

## What worked

Worth stating plainly, since the above is all complaints: `generate_model` at
defaults produced genuinely good hard-surface geometry from a single concept
image, twice, with no retries — including the geared slew-ring and X-braced
truss detail I assumed I'd need to raise `octree` for. `auto_cutout` handled
both plain-background concepts without comment. The `.views.png` and
`.rendercheck.png` sheets written beside each GLB were accurate and were what I
ended up judging every result from. The shared deck from that run shipped.
