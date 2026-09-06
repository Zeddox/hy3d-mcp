"""hy3d-gen MCP server: local image-to-3D via Hunyuan3D-2 on CUDA.

Stdio FastMCP process; no ML code lives here. Generation shells out to
engine_cli.py under the engine venv (torch + CUDA), image and mesh work
shells out to the worker venv (HY3D_PY) via the scripts in workers/. Tools
return file paths, never blobs.

Shape-only. The paint stage wants more VRAM than a consumer 8GB card has,
so this server produces untextured geometry and says so rather than
half-running a texture pass; materials are the caller's job downstream.

Config (env, with defaults):
  HY3D_ENGINE_REPO  Hunyuan3D-2 checkout   (~/git/repos/Hunyuan3D-2)
  HY3D_ENGINE_PY    engine venv python     (~/.hy3d/engine-venv/bin/python)
  HY3D_PY    worker python w/ cv2 etc   (defaults to the engine venv)
  HY3D_OUT   default output directory      (~/hy3d-output)
"""
import asyncio
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from fastmcp import Context, FastMCP
from PIL import Image

ENGINE_REPO = Path(os.environ.get(
    "HY3D_ENGINE_REPO", "~/git/repos/Hunyuan3D-2")).expanduser()
ENGINE_PY = Path(os.environ.get(
    "HY3D_ENGINE_PY", "~/.hy3d/engine-venv/bin/python")).expanduser()
# Defaults to the engine venv rather than a second one. provision-engine.sh
# builds a single environment that already carries the worker stack
# (numpy/PIL/trimesh/scipy/cv2/pygltflib/pyrender), so splitting it here
# would only invent a venv that is never created. Still overridable: the
# separation exists so mesh and image work can run without loading torch.
HY3D_PY = Path(os.environ.get(
    "HY3D_PY", os.environ.get("HY3D_ENGINE_PY", "~/.hy3d/engine-venv/bin/python"))
).expanduser()
HY3D_OUT = Path(os.environ.get("HY3D_OUT", "~/hy3d-output")).expanduser()

WORKERS = Path(__file__).parent / "workers"


# Ships inside the package rather than in scripts/, so a wheel install can
# still find it: a wheel carries no repo root. Invoked by absolute path
# under ENGINE_PY, never imported -- the engine venv has torch, this one
# does not, and the only thing crossing between them is argv and stdout.
ENGINE_CLI = Path(__file__).parent / "engine_cli.py"

# A runaway backstop, not a normal bound: shape at octree 384 runs ~2
# minutes on a 3060 Ti, but a run that has spilled into host RAM crawls at
# PCIe bandwidth and can take an order of magnitude longer.
GENERATE_TIMEOUT = 5400.0
WORKER_TIMEOUT = 300.0
# Idle gap the client tolerates is what this has to stay under. Volume
# decoding is the majority of a job's wall-clock and reports nothing the
# server can parse, so the heartbeat carries that whole stretch.
HEARTBEAT_SECONDS = 15.0

mcp = FastMCP(
    "hy3d-gen",
    instructions=(
        "Local image-to-3D generation (Hunyuan3D-2 on CUDA). Produces "
        "UNTEXTURED geometry — there is no texture stage on this build, so "
        "do not promise the user a painted model. Feed NATURALLY LIT "
        "concept art of a single object on a plain background, no drop "
        "shadows. Models land as file paths; importing them into an engine "
        "is the caller's job, and export_stl converts one for printing. "
        "Generation is serialized — one job at a time — and takes ~2 min at "
        "the default octree."
    ),
)

# One generation at a time. On WSL2 the second job would not fail cleanly:
# WDDM serves CUDA allocations past VRAM out of system RAM, so both jobs
# would finish having crawled at PCIe bandwidth instead of one erroring.
_job_lock = asyncio.Lock()
_queue_depth = 0
_queue_guard = threading.Lock()
_last_job: dict | None = None
# The engine child currently running, so cancel_job can reach it without a pid
# hunt. Only one exists at a time — _job_lock sees to that.
_current_proc: asyncio.subprocess.Process | None = None


def _record_job(tool: str, target: str, ok: bool, seconds: float) -> None:
    global _last_job
    _last_job = {"tool": tool, "target": target, "ok": ok,
                 "seconds": round(seconds, 1)}


def _run_worker(script: str, argv: list[str], timeout: float = WORKER_TIMEOUT) -> dict:
    """Run a workers/ script under HY3D_PY; its last stdout line is JSON."""
    cmd = [str(HY3D_PY), str(WORKERS / script)] + argv
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    payload: dict | None = None
    if lines:
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload = None
    if proc.returncode != 0 or payload is None:
        detail = (payload or {}).get("error") or proc.stderr.strip()[-2000:] \
            or proc.stdout.strip()[-2000:] or "no output"
        raise RuntimeError("%s failed: %s" % (script, detail))
    return payload


def _has_real_alpha(path: Path) -> bool:
    with Image.open(path) as im:
        if "A" not in im.getbands():
            return False
        lo, _hi = im.getchannel("A").getextrema()
        return lo < 128


_PROGRESS_LINE = re.compile(r"^\s*\[\s*(\d+)\s*%\]\s*(.*)$")


def _mmss(seconds: float) -> str:
    return "%dm%02ds" % (int(seconds) // 60, int(seconds) % 60)


async def _run_engine(cmd: list[str], env: dict[str, str], ctx: Context | None,
                      stage: str = "shape") -> tuple[int, str, str, bool]:
    """Run the engine driver, relaying its progress to the MCP client.

    Returns (returncode, stdout, stderr, streamed). Two things a plain
    blocking run cannot do: relay a heartbeat — from outside, a slow job and
    a hung one look identical, and clients abort on idle — and kill the
    child when the call is cancelled, since an abandoned engine holds both
    the single-job queue and the GPU until someone hunts down its pid.

    `streamed` reports whether progress actually went anywhere: a client
    that sends no progressToken gets no notifications, and that is worth
    surfacing rather than leaving to be inferred from a later timeout.
    """
    global _current_proc
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(ENGINE_REPO), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _current_proc = proc
    state = {"base": 0.0, "width": 100.0, "last": 0.0, "stage": stage}
    out: list[str] = []
    err: list[str] = []
    started = time.monotonic()

    async def report(value: float, message: str) -> None:
        # MCP requires the value to rise, so clamp instead of stepping back.
        state["last"] = max(state["last"], value)
        if ctx is None:
            return
        try:
            await ctx.report_progress(state["last"], 100.0, message)
        except Exception:
            # A heartbeat is never worth a bake. A client that has stopped
            # listening should not take down the job it stopped listening to.
            pass

    async def drain(stream, buf: list[str], watch: bool) -> None:
        # Both pipes get drained concurrently: a full stderr buffer blocks the
        # child, which presents as exactly the hang this is here to prevent.
        # The text is accumulated as well as streamed — the vert/face regex and
        # the failure message downstream both need the whole thing.
        async for raw in stream:
            line = raw.decode("utf-8", "replace")
            buf.append(line)
            if not watch:
                continue
            m = _PROGRESS_LINE.match(line)
            if m:
                await report(
                    state["base"] + float(m.group(1)) / 100.0 * state["width"],
                    "%s: %s" % (state["stage"], m.group(2).strip()))

    async def heartbeat() -> None:
        # Volume decoding is the majority of the run and emits no line the
        # server parses, so real progress alone would not keep the client's
        # idle timer alive across it.
        # The bar creeps toward the stage ceiling because the value must rise;
        # the message carries the honest part, which is elapsed time.
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            ceiling = state["base"] + state["width"]
            await report(state["last"] + (ceiling - state["last"]) * 0.05,
                         "%s — %s elapsed"
                         % (state["stage"], _mmss(time.monotonic() - started)))

    meta = getattr(getattr(ctx, "request_context", None), "meta", None)
    streamed = getattr(meta, "progressToken", None) is not None

    beat = asyncio.create_task(heartbeat())
    try:
        async with asyncio.timeout(GENERATE_TIMEOUT):
            await asyncio.gather(drain(proc.stdout, out, True),
                                 drain(proc.stderr, err, False))
            rc = await proc.wait()
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("engine ran past %.0fs and was killed"
                           % GENERATE_TIMEOUT)
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    finally:
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)
        _current_proc = None
    return rc, "".join(out), "".join(err), streamed


def _add_normals(glb: Path) -> dict:
    """Inject glTF NORMAL into a GLB in place.

    Never raises. This runs after a model is already on disk, and losing a
    finished bake to a post-step would be worse than shipping it flat-shaded.
    """
    try:
        return _run_worker("normals.py", [str(glb)])
    except Exception as e:
        return {"normals_added": False, "warning": "normals pass failed: %s" % e}


def _out_path(stem: str, suffix: str, explicit: str | None) -> Path:
    # Absolute, always. The engine child runs with cwd set to the Hunyuan3D
    # checkout so its own imports resolve, which means a relative path handed
    # in by the caller would be written somewhere neither of us meant.
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        p = HY3D_OUT / (stem + suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _engine_json(stdout: str) -> dict:
    """Pull engine_cli's result object off the last JSON line of stdout.

    Progress lines share the stream, so this scans upward for the first line
    that parses rather than assuming the last line is the payload.
    """
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


@mcp.tool
async def generate_model(
    image_path: str,
    output_path: str | None = None,
    seed: int = 42,
    auto_cutout: bool = True,
    max_faces: int = 40000,
    octree: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    model: str | None = None,
    cpu_offload: bool = False,
    paint: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Turn a concept image into an UNTEXTURED 3D model (GLB).

    There is no texture stage on this build — the paint pipeline wants more
    VRAM than the card has — so the result is clean geometry with normals
    and no UVs or material. Say that to the user rather than letting them
    expect colour; passing paint=True is an error, not a slow path.

    What the shape stage does and does not give you: it resolves silhouette
    and large forms faithfully, and it does not resolve surface relief.
    Carved ornament, panel lines and fabric folds present in the concept art
    come back smooth at every octree setting. That is the model's range, not
    a knob you have not found — such detail belongs in a normal or
    displacement map applied later.

    auto_cutout keys out a plain background first unless the input already
    carries real transparency. Leave it on.
    max_faces decimates the raw output (typically 600k–1M faces, which is
    not something to hand an engine) down to a game-ready budget; 0 keeps
    the raw mesh. Decimation preserves watertightness.
    Blocks while an earlier generation is running (single-job queue).

    Quality knobs. Each defaults to None, which lets the driver's own
    default (in parens) apply.

      octree (384)      marching-cubes resolution. This is a TESSELLATION
                        density dial, not a detail dial: raising it to 512
                        buys more triangles describing the same surface and
                        recovers no relief, at roughly double the runtime.
                        Leave it alone unless thin struts are fusing.
      steps (50)        shape diffusion steps; diminishing well before this.
      guidance (5.0)    how tightly shape follows the image; higher is more
                        faithful but can over-sharpen.

    model: 'tencent/Hunyuan3D-2' (default, 3.3B) or 'tencent/Hunyuan3D-2mini'.
    2mini is roughly twice as fast and its renders look comparable, but it
    tends to produce a thin hollow shell where the full model produces a
    solid — a difference invisible in a preview and fatal to a print. Prefer
    the default; check export_stl's bbox_fill_pct if you use 2mini.
    cpu_offload=True runs the stages sequentially through host RAM. It is
    the lever when the returned peak_reserved_gib sits at the card's
    ceiling, and it costs runtime, so it is off by default.
    """
    global _queue_depth
    src = Path(image_path).expanduser().resolve()
    if not src.is_file():
        raise ValueError("input image not found: %s" % src)
    try:
        with Image.open(src) as im:
            im.verify()
    except Exception as e:
        raise ValueError("input is not a readable image: %s (%s)" % (src, e))
    if paint:
        raise ValueError(
            "this build has no texture stage: Hunyuan3D's paint pipeline "
            "needs more VRAM than this card has, so the server generates "
            "shape only. Call with paint=False (the default) and texture "
            "the GLB downstream.")
    for name, val in (("octree", octree), ("steps", steps)):
        if val is not None and val < 1:
            raise ValueError("%s must be a positive integer" % name)
    if guidance is not None and guidance < 0:
        raise ValueError("guidance must be >= 0")
    if max_faces < 0:
        raise ValueError("max_faces must be >= 0 (0 disables decimation)")

    dst = _out_path(src.stem, ".glb", output_path)
    stages: list[str] = []
    # Bound before the lock: the result dict reads both outside the try, and an
    # engine that writes the GLB but garbles its final line must not take the
    # call down with a NameError on its way to reporting success.
    stats: dict = {}
    warning: str | None = None
    started = time.monotonic()
    with _queue_guard:
        _queue_depth += 1
    try:
        async with _job_lock:
            gen_input = src
            if auto_cutout and not _has_real_alpha(src):
                rgba = HY3D_OUT / "intermediate" / (src.stem + "-rgba.png")
                try:
                    cut = await asyncio.to_thread(
                        _run_worker, "cutout.py", [str(src), str(rgba)])
                    gen_input = Path(cut["png_path"])
                    stages.append("cutout (%.1f%% opaque)" % cut["opaque_pct"])
                except RuntimeError as e:
                    # The local key refuses busy backgrounds rather than
                    # shredding them. That is not a dead end here: the engine
                    # venv carries rembg, which handles painted concept art
                    # the corner-sample key cannot. Let the engine do it --
                    # but rembg only keys. cutout.py also crops to the alpha
                    # bbox and pads square, so the subject fills the latent
                    # instead of sharing it with empty background; the fallback
                    # input is framed differently, not merely keyed differently.
                    stages.append(
                        "cutout skipped (%s); keyed by rembg in-engine, "
                        "without the square recrop"
                        % str(e).split(": ", 1)[-1][:200])

            cmd = [str(ENGINE_PY), str(ENGINE_CLI), str(gen_input),
                   "-o", str(dst), "--seed", str(seed),
                   "--max-faces", str(max_faces),
                   "--engine", str(ENGINE_REPO)]
            if model:
                cmd += ["--model", model]
            if steps is not None:
                cmd += ["--steps", str(steps)]
            if guidance is not None:
                cmd += ["--guidance-scale", str(guidance)]
            if octree is not None:
                cmd += ["--octree-resolution", str(octree)]
            if cpu_offload:
                cmd.append("--cpu-offload")
            rc, so, se, streamed = await _run_engine(cmd, os.environ.copy(), ctx)
            if rc != 0 or not dst.is_file():
                _record_job("generate_model", src.name, False,
                            time.monotonic() - started)
                raise RuntimeError("shape generation failed (exit %d):\n%s"
                                   % (rc, (se or so)[-2000:]))
            stats = _engine_json(so)
            if not stats:
                # The GLB is on disk and the engine exited clean, so this is a
                # real result -- but every count below comes back null, and a
                # null that looks like a measurement is worse than one that
                # explains itself.
                warning = ("the engine wrote the model but its result line "
                           "was unreadable, so counts and memory figures are "
                           "unavailable. The file itself is fine -- run "
                           "render_preview or export_stl to inspect it.")
            stages.append("shape")
            if stats.get("raw_faces") and stats.get("faces") \
                    and stats["faces"] < stats["raw_faces"]:
                stages.append("decimate (%d -> %d)"
                              % (stats["raw_faces"], stats["faces"]))
    finally:
        with _queue_guard:
            _queue_depth -= 1

    seconds = time.monotonic() - started
    _record_job("generate_model", src.name, True, seconds)
    out = {"glb_path": str(dst), "verts": stats.get("vertices"),
           "faces": stats.get("faces"), "raw_faces": stats.get("raw_faces"),
           "watertight": stats.get("watertight"),
           "attributes": stats.get("glb_attributes"),
           "peak_reserved_gib": stats.get("peak_torch_reserved_gib"),
           "vram_ceiling_gib": stats.get("free_at_baseline_gib"),
           "seconds": round(seconds, 1), "stages": stages,
           "textured": False,
           "progress": "streamed" if streamed else "unavailable"}
    if stats.get("warning") or warning:
        out["warning"] = stats.get("warning") or warning
    return out


@mcp.tool
def paint_mesh(mesh_path: str, image_path: str,
               output_path: str | None = None) -> dict:
    """Unavailable on this build — texturing needs more VRAM than the card has.

    Kept as a tool so the failure is a sentence rather than an unknown-tool
    error: Hunyuan3D's paint pipeline runs multiview diffusion at a peak
    well past a consumer 8GB card, and the WSL2 failure mode is not a clean
    OOM but a silent spill into host RAM that runs at PCIe bandwidth. Half
    of it would appear to work and take an hour.

    Texture the GLB downstream instead — Blender, Substance, or Godot's own
    material tools — or run the paint stage on a larger GPU.
    """
    raise RuntimeError(
        "paint_mesh is unavailable: this server generates shape only. The "
        "GLB has normals but no UVs or material; texture it downstream, or "
        "run Hunyuan3D's paint pipeline on a GPU with more VRAM.")


@mcp.tool
def export_stl(
    glb_path: str,
    output_path: str | None = None,
    height_mm: float = 120.0,
    min_wall_mm: float = 0.8,
) -> dict:
    """Convert a generated GLB into an STL a slicer will accept.

    Two conversions that are not optional, both silent failures if skipped.
    STL carries no units and every slicer reads it as millimetres, so the
    engine's roughly-unit-box mesh would arrive as a 2mm trinket — hence
    height_mm, measured along the print Z axis. And glTF is Y-up while
    slicers are Z-up, so an unrotated export lands on its side.

    Also drops the model onto z=0 so it sits on the plate, centres it, and
    reports the manifold checks that decide whether it slices at all.

    Read bbox_fill_pct in the result, not just the checks. A thin hollow
    shell and a solid can both be watertight, single-body and genus 0 with
    identical silhouettes; the enclosed volume is what separates them, and
    no preview render will show you the difference. Under ~15% means walls
    thin enough that the slicer may drop them.
    min_wall_mm warns when the finest detail present falls under roughly two
    perimeters of a 0.4mm nozzle.
    """
    src = Path(glb_path).expanduser().resolve()
    if not src.is_file():
        raise ValueError("GLB not found: %s" % src)
    if height_mm <= 0:
        raise ValueError("height_mm must be positive")
    dst = _out_path(src.stem, ".stl", output_path)
    started = time.monotonic()
    out = _run_worker("tostl.py", [str(src), str(dst),
                                   "--height", str(height_mm),
                                   "--min-wall", str(min_wall_mm)])
    _record_job("export_stl", src.name, True, time.monotonic() - started)
    return out


@mcp.tool
def prepare_concept(image_path: str, output_path: str | None = None) -> dict:
    """Key a plain-background concept image out to a centered square RGBA PNG.

    Standalone version of generate_model's auto_cutout, for callers that
    want the intermediate. Refuses inputs whose corners disagree (busy
    background) rather than shredding them. Warns when the opaque fraction
    looks like a bad key.

    A refusal here does not block generation: generate_model falls back to
    the engine venv's rembg, which handles painted concept art this
    corner-sampling key cannot. Reach for this tool when you want to see and
    check the cutout, not as a required first step.
    """
    src = Path(image_path).expanduser().resolve()
    if not src.is_file():
        raise ValueError("input image not found: %s" % src)
    dst = _out_path(src.stem + "-rgba", ".png", output_path)
    started = time.monotonic()
    out = _run_worker("cutout.py", [str(src), str(dst)])
    _record_job("prepare_concept", src.name, True, time.monotonic() - started)
    pct = out["opaque_pct"]
    if pct < 8.0:
        out["warning"] = ("only %.1f%% opaque — the key may have eaten the "
                          "subject; check the PNG" % pct)
    elif pct > 60.0:
        out["warning"] = ("%.1f%% opaque — background may not be plain; "
                          "check the PNG for kept background" % pct)
    return out


@mcp.tool
def finish_model(
    glb_path: str,
    output_path: str | None = None,
    tone_gamma: float = 1.35,
    contrast: float = 1.30,
    saturation: float = 1.35,
    accent_emissive: bool = True,
    seam_pinstripes: float = 0.58,
    seam_halo: float = 0.10,
    normals: bool = True,
) -> dict:
    """Apply the game-look texture pass to a GLB that already has a texture.

    Not reachable from this server's own output: shape-only generation
    produces no albedo map for this to tone, so calling it on a fresh
    generate_model result fails. It stays available for GLBs textured
    elsewhere and round-tripped back through here.

    Tones the albedo (gamma/contrast/saturation), extracts saturated accents
    and blackhat panel seams into a dedicated glTF emissive texture.
    Separated from generation so it can be re-run with new knobs without
    regenerating. seam_pinstripes 0-1 (0 disables); defaults are the values
    proven on the AEGIS fleet.

    The accent extractor keys on saturated red-dominant regions and is tuned
    for broad accent panels — a hard-surface subject whose only accents are
    thin indicator strips will report accent_coverage_pct near 0. That is
    the extractor's range, not a mis-specified concept.
    normals=True re-injects the glTF NORMAL attribute after the texture
    rewrite, which would otherwise drop it. Geometry is still untouched:
    normals are derived from the vertex positions already in the file.
    """
    src = Path(glb_path).expanduser().resolve()
    if not src.is_file():
        raise ValueError("GLB not found: %s" % src)
    dst = _out_path(src.stem + "-finished", ".glb", output_path)
    argv = [str(src), str(dst),
            "--tone-gamma", str(tone_gamma),
            "--contrast", str(contrast),
            "--saturation", str(saturation),
            "--seam-pinstripes", str(seam_pinstripes),
            "--seam-halo", str(seam_halo)]
    if not accent_emissive:
        argv.append("--no-accent-emissive")
    started = time.monotonic()
    out = _run_worker("finish.py", argv)
    if normals:
        # After finish.py, not before: its trimesh round-trip drops NORMAL.
        nrm = _add_normals(dst)
        out["normals_added"] = bool(nrm.get("normals_added"))
        if nrm.get("warning"):
            out["warning"] = nrm["warning"]
    _record_job("finish_model", src.name, True, time.monotonic() - started)
    return out


@mcp.tool
def render_preview(glb_path: str, views: list[str] | None = None,
                   size: int = 1024) -> dict:
    """Offscreen renders of a GLB, no engine needed.

    views: subset of iso/front/back/top/side (default [iso]).

    Rasterising needs pyrender, and headless it needs an EGL context —
    PYOPENGL_PLATFORM=egl, falling back to software rendering when the WSL
    device node is not reachable. The pyopengl pin (3.1.0, whose
    glGenTextures wrapper breaks against modern numpy) only bites on
    TEXTURED meshes, so it does not affect this server's own untextured
    output; `server_status`'s preview_ok flags it anyway for GLBs from
    elsewhere.

    A preview tells you about silhouette and form, and nothing about whether
    the mesh is solid. Two models that render identically can differ 5x in
    enclosed volume — export_stl's bbox_fill_pct is where that shows up.
    """
    src = Path(glb_path).expanduser().resolve()
    if not src.is_file():
        raise ValueError("GLB not found: %s" % src)
    outdir = HY3D_OUT / "previews"
    started = time.monotonic()
    try:
        out = _run_worker("preview.py", [
            str(src), str(outdir),
            "--views", ",".join(views or ["iso"]),
            "--size", str(size)])
        out["source"] = "rendered"
    except RuntimeError as e:
        # Sibling sheets keep the full GLB filename: foo.glb.views.png.
        sheets = [p for p in (Path(str(src) + ".views.png"),
                              Path(str(src) + ".rendercheck.png")) if p.is_file()]
        reason = str(e).strip().splitlines()[-1]
        if not sheets:
            raise RuntimeError(
                "%s — and no generator sheets beside %s to fall back on "
                "(only the paint pass writes those, so shape-only output has "
                "none)" % (e, src.name))
        out = {"png_paths": [str(p) for p in sheets],
               "source": "generator_sheets",
               "note": "could not rasterise (%s); these are the multiview and "
                       "render-check sheets written beside the GLB during "
                       "paint, not the views requested" % reason}
    _record_job("render_preview", src.name, True, time.monotonic() - started)
    return out


def _check(ok: bool, fix: str) -> dict:
    return {"ok": ok} if ok else {"ok": False, "fix": fix}


def _probe(python: Path, code: str, timeout: float = 120.0):
    """Run a probe snippet under another interpreter; None if it cannot run."""
    if not python.is_file():
        return None
    try:
        return subprocess.run([str(python), "-c", code], capture_output=True,
                              text=True, timeout=timeout)
    except Exception:
        return None


def _server_status() -> dict:
    repo = _check(
        (ENGINE_REPO / "hy3dgen" / "shapegen").is_dir(),
        "Hunyuan3D-2 checkout not found at %s — run "
        "`bash scripts/provision-engine.sh`, or set HY3D_ENGINE_REPO to an "
        "existing checkout" % ENGINE_REPO)

    engine_ok = False
    engine_fix = ("engine venv not found at %s — run "
                  "`bash scripts/provision-engine.sh`" % ENGINE_PY)
    cuda: dict = {"ok": False, "fix": "engine venv missing, so CUDA is unverified"}
    # One probe covers both: the imports the driver needs, and whether torch
    # can actually see the card. A CPU-only wheel imports perfectly and then
    # runs the job at a hundredth of the speed, so torch importing is not the
    # check that matters.
    probe = _probe(ENGINE_PY,
                   "import torch, trimesh, pygltflib, skimage, pymeshlab, rembg\n"
                   "print('CORE_OK')\n"
                   "print('CUDA', torch.cuda.is_available(), torch.version.cuda)\n"
                   "if torch.cuda.is_available():\n"
                   "    f, t = torch.cuda.mem_get_info()\n"
                   "    print('VRAM', torch.cuda.get_device_name(0), f, t)\n")
    if probe is not None:
        engine_ok = "CORE_OK" in probe.stdout
        if not engine_ok:
            engine_fix = (
                "engine venv at %s is missing packages: %s — re-run "
                "`bash scripts/provision-engine.sh`, which installs them in "
                "the order that matters (torch from the cu124 index, numpy<2, "
                "and scikit-image, which upstream needs for marching cubes "
                "but does not declare)"
                % (ENGINE_PY, probe.stderr.strip()[-300:]))
        line = next((l for l in probe.stdout.splitlines()
                     if l.startswith("CUDA ")), "")
        if line.startswith("CUDA True"):
            vram = next((l for l in probe.stdout.splitlines()
                         if l.startswith("VRAM ")), "")
            cuda = {"ok": True}
            if vram:
                *name, free, total = vram[5:].rsplit(" ", 2)
                cuda["device"] = " ".join(name)
                cuda["free_gib"] = round(int(free) / 1024 ** 3, 2)
                cuda["total_gib"] = round(int(total) / 1024 ** 3, 2)
                if cuda["free_gib"] < 5.0:
                    cuda["warning"] = (
                        "only %.2f GiB free — something else is holding VRAM. "
                        "On WSL2 an oversized job does not fail, it spills "
                        "into host RAM and crawls, so free this up or expect "
                        "a very slow run" % cuda["free_gib"])
        elif engine_ok:
            cuda = {"ok": False, "fix":
                    "torch imports but reports no CUDA device. Never install "
                    "an NVIDIA driver inside WSL — the Windows driver is "
                    "projected in. Check that /usr/lib/wsl/lib is on the "
                    "loader path, and that the venv has the cu124 torch build "
                    "rather than the CPU wheel (`%s -c \"import torch; "
                    "print(torch.__version__)\"` should end in +cu124)"
                    % ENGINE_PY}

    driver = _check(ENGINE_CLI.is_file(),
                    "engine driver missing at %s — reinstall the server "
                    "package" % ENGINE_CLI)

    # Weights live in the HF cache, not the checkout: the driver names a repo
    # id and lets huggingface_hub resolve it. Absent means a ~5GB download on
    # first generate, which is worth knowing before a call appears to hang.
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cached = list((hf / "hub").glob("models--tencent--Hunyuan3D-2*")) \
        if (hf / "hub").is_dir() else []
    weights = _check(
        bool(cached),
        "no Hunyuan3D weights in the HF cache at %s — the first generate_model "
        "call will download ~5GB before it starts, which looks like a hang. "
        "Pre-fetch it, or just budget for the first call being long" % (hf / "hub"))

    venv_ok = False
    venv_fix = ("worker venv missing: run `bash scripts/provision-engine.sh`, "
                "or set HY3D_PY to a python with cv2/numpy/trimesh/PIL/scipy/"
                "pygltflib")
    preview_ok = False
    preview_fix = "worker venv missing, so render_preview cannot rasterise"
    wprobe = _probe(HY3D_PY,
                    "import cv2, numpy, trimesh, PIL, scipy, pygltflib\n"
                    "print('CORE_OK')\n"
                    "try:\n"
                    "    import pyrender, OpenGL\n"
                    "    print('GL', OpenGL.__version__)\n"
                    "except Exception as e:\n"
                    "    print('GL_ERR', type(e).__name__, e)\n", timeout=60.0)
    if wprobe is not None:
        venv_ok = "CORE_OK" in wprobe.stdout
        if not venv_ok:
            # uv, not `python -m pip`: uv-created venvs ship without pip, so
            # the pip form fails with "No module named pip" on a stock setup.
            venv_fix = ("worker venv at %s is missing packages: %s — install "
                        "them with `uv pip install --python %s opencv-python "
                        "numpy trimesh pillow scipy pygltflib`"
                        % (HY3D_PY, wprobe.stderr.strip()[-300:], HY3D_PY))

        gl = next((ln.split(None, 1)[1].strip()
                   for ln in wprobe.stdout.splitlines()
                   if ln.startswith("GL ")), None)
        # 3.1.0 is what pyrender pins, and its glGenTextures wrapper cannot
        # bind a texture against modern numpy. This build produces untextured
        # meshes, so it does not block previews here — but it would bite the
        # moment a textured GLB from elsewhere is rendered.
        if gl is not None:
            parts = tuple(int("".join(c for c in p if c.isdigit()) or 0)
                          for p in gl.split(".")[:3])
            preview_ok = parts >= (3, 1, 7)
        if gl is None:
            preview_fix = ("pyrender/pyopengl not importable under %s — "
                           "install with `uv pip install --python %s pyrender` "
                           "then apply the pin override below. Headless "
                           "rendering also needs PYOPENGL_PLATFORM=egl"
                           % (HY3D_PY, HY3D_PY))
        elif not preview_ok:
            preview_fix = ("pyopengl is %s; below 3.1.7 it cannot render "
                           "textured meshes. This server's own output is "
                           "untextured so previews still work, but a textured "
                           "GLB from elsewhere would fail. Fix with `uv pip "
                           "install --python %s --upgrade 'PyOpenGL>=3.1.7'` — "
                           "resolving it alongside pyrender fails, so it must "
                           "be a separate upgrade" % (gl, HY3D_PY))

    return {
        "engine_repo_ok": repo, "engine_venv_ok": _check(engine_ok, engine_fix),
        "cuda_ok": cuda, "driver_ok": driver, "weights_cached": weights,
        "venv_ok": _check(venv_ok, venv_fix),
        "preview_ok": _check(preview_ok, preview_fix),
        "textured_output": False,
        "queue_depth": _queue_depth, "last_job": _last_job,
        "config": {"HY3D_ENGINE_REPO": str(ENGINE_REPO),
                   "HY3D_ENGINE_PY": str(ENGINE_PY),
                   "HY3D_PY": str(HY3D_PY), "HY3D_OUT": str(HY3D_OUT)},
    }


@mcp.tool
async def cancel_job() -> dict:
    """Kill the generation currently running and free the queue.

    For when a client has stopped waiting but the engine has not stopped
    working — a job abandoned that way holds the single-job queue and the
    GPU, and every later call blocks behind it. Safe to call when nothing
    is running; it just reports that.
    """
    proc = _current_proc
    if proc is None or proc.returncode is not None:
        return {"killed": False, "note": "no engine process is running",
                "queue_depth": _queue_depth}
    pid = proc.pid
    proc.kill()
    return {"killed": True, "pid": pid,
            "note": "engine killed; the call that started it will return an "
                    "error, and the queue frees once it unwinds"}


@mcp.tool
def server_status() -> dict:
    """Health check and first-run diagnostic.

    Validates every setup requirement — the Hunyuan3D-2 checkout, the engine
    venv, that torch actually sees the GPU, the driver script, whether the
    weights are already cached, and the worker venv — and each failing check
    carries the exact fix. Also reports queue depth and the last job.

    Worth reading even when everything passes: cuda_ok reports free VRAM,
    and on WSL2 a job that does not fit does not fail, it spills into host
    RAM and runs at PCIe speed.
    """
    return _server_status()


@mcp.tool
def setup_engine(confirm: bool = False, only: int | None = None) -> dict:
    """Not automated yet on this platform — returns the command to run.

    The CUDA provisioner exists as `scripts/provision-engine.sh` and is
    idempotent, but it does not yet speak the phase-by-phase, dry-run-first
    protocol this tool's contract promises, and a half-honoured contract is
    worse than an honest pointer. Wiring it up is the next phase of the
    port.
    """
    raise RuntimeError(
        "automated setup is not wired up on the CUDA build yet. Run this in a "
        "shell from the repo root, then call server_status:\n\n"
        "    bash scripts/provision-engine.sh\n\n"
        "It clones Hunyuan3D-2, builds the engine venv with the cu124 torch "
        "wheels, and is safe to re-run — it inspects before acting. Note it "
        "needs the system package `libopengl0` for pymeshlab's mesh IO.")


def main() -> None:
    HY3D_OUT.mkdir(parents=True, exist_ok=True)
    mcp.run()


if __name__ == "__main__":
    main()
