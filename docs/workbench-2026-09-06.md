# Phase 5: the workbench — plan

Date: 2026-09-06
Status: plan, nothing built yet.

Phases 1–4 came from the WSL2 handoff document. This one does not; it is the
tooling the user asked for after Phase 2 — *"a way to select an image and then
view the generated mesh"* — and it is a different animal from an MCP server.
The MCP path is an agent driving the tools through a conversation. This is a
person driving them through a browser: pick an image, watch it run, orbit the
result, export the STL.

## What already exists, and what has to be built

The viewer half is largely solved. The Phase 1 Pagoda Shape Bench was a
three.js `GLTFLoader` scene with orbit controls, a JPEG fallback per stage and
a palette that works in both themes — `tmp/viewer_template.html`, 437 lines. It
embedded its GLBs as base64 because it was an Artifact and had no server. Here
there is a server, so the meshes come down as files and the template is a
starting point rather than a thing to rebuild.

Everything else is new: the picker, an HTTP surface, and job plumbing.

## Architecture

**A second process, not a second transport.** The MCP server is stdio and its
lifecycle belongs to Claude Code — it is spawned per client and dies with it.
The workbench has to outlive that and be reachable from a browser, so it is its
own process: `python -m hy3d_mcp.web`, exposed as a `hy3d-web` console script.

**It imports `hy3d_mcp.server` and calls the tool functions directly.**
`@mcp.tool` returns the plain function, so `generate_model`, `export_stl`,
`render_preview` and `server_status` are all callable as ordinary Python. That
buys the entire tested pipeline — cutout with its two keys, the engine driver,
the progress regex, the heartbeat, the timeout, the kill-on-cancel — with no
second implementation to keep in step. Importing the module is safe: everything
at module scope is env reads, `expanduser`, one `is_file()` stat and the
`FastMCP` constructor. No I/O, no mkdir, no server started.

**Progress arrives through a duck-typed `ctx`.** `_run_engine` only ever calls
`await ctx.report_progress(value, total, message)` on whatever it was handed.
The web process passes an object with that one coroutine method, writing into
an in-memory job record instead of onto an MCP wire. Zero changes to
`server.py` for this part.

    class WebCtx:
        def __init__(self, job): self.job = job
        async def report_progress(self, value, total, message):
            self.job["pct"], self.job["message"] = value, message

It must be `async def`. `_run_engine` awaits the result inside a bare
`except Exception: pass` written to keep a dead client from killing a live
bake, so a sync method raises, gets swallowed, and progress silently never
appears — with no traceback to find it by.

`generate_model` will return `progress: "unavailable"`, because `streamed` is
derived from an MCP `progressToken` this ctx has none of. The web UI does not
surface that field; faking a request context to make one word come out right
would be worse than ignoring it.

**Starlette on uvicorn, one event loop.** Both are already in the server venv
(starlette 1.6.0, uvicorn 0.52.4, python-multipart present) as fastmcp
dependencies; they get declared explicitly in `pyproject.toml` under an
optional `web` extra rather than leaned on transitively.

The loop is not a style preference. `_job_lock` is a module-level
`asyncio.Lock` bound to the loop that first awaits it, so a threaded stdlib
server calling `asyncio.run()` per job would hand each job a fresh loop and get
`RuntimeError: ... bound to a different event loop` — the single-job guarantee
would die on the second request. One long-lived loop is required, and starlette
gives one without a hand-rolled loop thread.

**Progress reaches the browser by polling, not SSE.** One `GET /api/job/<id>`
per second against a 150-second job is 150 requests of a few hundred bytes;
that is not a cost worth engineering around. Polling survives a tab reload and
a laptop sleep for free, where a dangling event stream needs reconnect logic.

## The cross-process hazard, and the one change to `server.py`

`_job_lock` is in-process. With a workbench running, two processes can drive
the GPU: the MCP server inside Claude Code and the web server in a terminal.
On WSL2 that is not a clean failure — WDDM serves CUDA allocations past VRAM
out of host RAM, so both jobs finish, each having crawled at PCIe bandwidth.
It is the exact failure the single-job queue exists to prevent, and the queue
does not currently see it.

Fix: an `fcntl.flock` on `~/.hy3d/job.lock`, taken inside the region
`_job_lock` already guards, so there is one critical section rather than two
nested ones with different extents. `flock` rather than a pidfile because the
OS releases it when the holder dies, which is the case that matters — a
crashed engine must not wedge the queue. The syscall blocks, so it goes through
`asyncio.to_thread` or `LOCK_EX|LOCK_NB` in a sleep loop; a bare call would
stall the heartbeat and every other request on the loop.

This changes MCP behaviour: a second Claude Code session now waits (or errors)
instead of thrashing alongside the first. That is the point, and
`generate_model`'s docstring line — *"Blocks while an earlier generation is
running (single-job queue)"* — gets updated to say the queue is machine-wide.

## HTTP surface

    GET  /                     the app
    POST /api/generate         multipart upload, or {"path": ...} for a
                               file already on disk → {"job": id}
    GET  /api/job/<id>         {state, pct, message, stages, result, error}
    POST /api/job/<id>/cancel  → server.cancel_job()
    POST /api/stl              {"glb": ..., "height_mm": ...} → tostl worker
    GET  /api/gallery          recent GLBs in HY3D_OUT, newest first
    GET  /files/<relpath>      GLB, PNG and STL out of HY3D_OUT only

`/files` resolves and checks `is_relative_to(HY3D_OUT)` before opening
anything. The upload handler caps the body and derives the saved name itself
rather than trusting the multipart filename, which is where `../` arrives.

**Uploads get a unique stem.** `_out_path(src.stem, ".glb", None)` writes
`HY3D_OUT/<stem>.glb`, so two uploads both called `image.png`, or one image
re-run at a different octree, silently overwrite the earlier GLB while the
gallery goes on listing it. Under MCP the agent names its outputs and this
never bites; behind an upload button it is a bug on day one. Uploads are saved
as `<stem>-<yyyymmdd-hhmmss>` and that stem is passed through as
`output_path`, so the input, the RGBA cutout and the GLB share one name.

STL export goes through the existing `tostl.py` worker rather than three.js's
`STLExporter`: the worker does the millimetre scaling, the min-wall check and
`bbox_fill_pct`, which is the number that separates a solid from a hollow
shell with the same silhouette. A client-side export would look identical and
tell the user nothing about whether it can be printed.

## Front end

three.js 0.169.0 from jsDelivr through an import map — `build/three.module.js`,
`examples/jsm/loaders/GLTFLoader.js`, `examples/jsm/controls/OrbitControls.js`,
all three verified reachable. None of the Artifact CSP rules apply to a locally
served page, so the CDN choice is free and a plain `<a download>` works for the
STL. If offline robustness is wanted later, the precedent is u2net: prefetch in
install.sh phase 5 and check it in `server_status`. Not now.

Layout, one page, three regions: a drop target that doubles as the input
preview; the viewer, which shows the mesh as soon as the job finishes; and a
result panel carrying what the run actually cost — faces against raw faces,
watertight, peak reserved against the card's ceiling, wall clock — plus the
export controls. Progress rides a bar above the viewer with the engine's own
message under it, because "decoding volume at octree 384" at 40% is the
difference between a slow job and a hung one.

## Bind address

127.0.0.1 by default, `--host` to override. WSL2 forwards localhost from
Windows, so `http://localhost:8760` in the Windows browser normally reaches a
loopback bind in the guest. When it does not, the cause is the distro's
networking mode and the fallback is `--host 0.0.0.0` plus the eth0 address —
which the startup banner prints, because a page that will not load is the most
likely way this looks broken on first run. Binding wide is opt-in: this
service runs arbitrary generation and serves files.

## Build order

1. `web.py` with `/`, `/api/generate`, `/api/job/<id>` and a placeholder page.
   Smoke-test that `WebCtx.report_progress` actually fires before any front end
   exists — that is the failure mode with no traceback.
2. The flock in `server.py`, proved by holding it from a second interpreter.
3. `/files`, `/api/gallery`, `/api/stl`, cancel.
4. The front end, from `tmp/viewer_template.html`.
5. A real end-to-end run from the Windows browser, and the README section.
