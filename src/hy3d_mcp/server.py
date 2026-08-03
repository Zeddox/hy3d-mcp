"""hy3d-gen MCP server: local image-to-3D via Hunyuan3D-MLX.

Stdio FastMCP process; no ML code lives here. Generation shells out to the
hy3d Swift binary, image work shells out to the worker venv (HY3D_PY) via
the scripts in workers/. Tools return file paths, never blobs.

Config (env, with defaults):
  HY3D_REPO  Hunyuan3D-MLX checkout        (~/git/repos/hunyuan3d-mlx)
  HY3D_PY    worker venv python w/ cv2 etc (~/.hy3d/worker-venv/bin/python)
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

HY3D_REPO = Path(os.environ.get("HY3D_REPO", "~/git/repos/hunyuan3d-mlx")).expanduser()
HY3D_PY = Path(os.environ.get("HY3D_PY", "~/.hy3d/worker-venv/bin/python")).expanduser()
HY3D_OUT = Path(os.environ.get("HY3D_OUT", "~/hy3d-output")).expanduser()

WORKERS = Path(__file__).parent / "workers"


def _find_installer() -> Path:
    """install.sh sits at the repo root in a checkout or plugin cache, and
    beside the package in a wheel (force-included there, since a wheel
    carries no repo root). Prefer the checkout so a source tree under
    development wins over a stale installed copy."""
    here = Path(__file__).resolve()
    root, packaged = here.parent.parent.parent / "install.sh", here.parent / "install.sh"
    return packaged if packaged.is_file() and not root.is_file() else root


INSTALLER = _find_installer()
BINARY = HY3D_REPO / ".build/release/hy3d"
METALLIB_DIR = HY3D_REPO / "metallib"
# .build/release is a symlink; this is the real dir the binary also loads from
BUILD_REAL = HY3D_REPO / ".build/arm64-apple-macosx/release"

# A runaway backstop, not a normal bound: a lattice-heavy concept at octree
# 384 legitimately runs 15+ minutes, and progress notifications now keep the
# client from timing out underneath a job that is making progress.
GENERATE_TIMEOUT = 5400.0
WORKER_TIMEOUT = 300.0
# A cold setup is a ~4 min build plus a 12GB download on whatever link is going.
SETUP_TIMEOUT = 7200.0
# Idle gap the client tolerates is what this has to stay under; paint's pbr
# path can run many minutes without printing a line.
HEARTBEAT_SECONDS = 15.0

mcp = FastMCP(
    "hy3d-gen",
    instructions=(
        "Local image-to-3D generation (Hunyuan3D-MLX). Feed NATURALLY LIT "
        "concept art of a single object on a plain background, no drop "
        "shadows. Models land as file paths; importing them into an engine "
        "is the caller's job. Generation is serialized — one job at a time "
        "(paint peaks ~30GB unified memory); shape-only ~20s, "
        "shape+paint ~3-4 min."
    ),
)

# One generation at a time: paint peaks ~25-33GB unified memory, so parallel
# jobs would OOM the machine, not just slow it. Callers queue on the lock.
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
_STAGE_MARKER = re.compile(r"^generate\[(\d)/2\]\s*(\w+)")


def _mmss(seconds: float) -> str:
    return "%dm%02ds" % (int(seconds) // 60, int(seconds) % 60)


async def _run_engine(cmd: list[str], env: dict[str, str], ctx: Context | None,
                      two_stage: bool,
                      stage: str = "shape") -> tuple[int, str, str, bool]:
    """Run the hy3d binary, relaying its progress to the MCP client.

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
        *cmd, cwd=str(HY3D_REPO), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _current_proc = proc
    # In a two-stage generate, shape owns the first half of the bar and paint
    # the second; the shape subcommand alone owns all of it.
    state = {"base": 0.0, "width": 50.0 if two_stage else 100.0,
             "last": 0.0, "stage": stage}
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
            m = _STAGE_MARKER.match(line)
            if m:
                state["stage"] = m.group(2)
                state["base"] = 0.0 if m.group(1) == "1" else 50.0
                continue
            m = _PROGRESS_LINE.match(line)
            if m:
                await report(
                    state["base"] + float(m.group(1)) / 100.0 * state["width"],
                    "%s: %s" % (state["stage"], m.group(2).strip()))

    async def heartbeat() -> None:
        # The pbr paint path prints nothing for minutes at a stretch, so real
        # progress lines alone would not keep the client's idle timer alive.
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


def _mesh_counts(glb: Path) -> dict:
    """Read vertex and triangle counts off a finished GLB.

    Never raises, for the same reason as _add_normals: a reporting detail is
    not worth failing a completed bake over.
    """
    try:
        return _run_worker("meshinfo.py", [str(glb)])
    except Exception:
        return {}


def _out_path(stem: str, suffix: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
    else:
        p = HY3D_OUT / (stem + suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@mcp.tool
async def generate_model(
    image_path: str,
    output_path: str | None = None,
    paint: bool = True,
    texture_size: int = 2048,
    seed: int = 42,
    auto_cutout: bool = True,
    finish: bool = False,
    octree: int | None = None,
    paint_res: int | None = None,
    paint_steps: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    superres: bool = True,
    normals: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Turn a concept image into a textured 3D model (GLB).

    paint=False skips texturing (shape only, ~20s vs ~3-4 min) — it runs a
    different engine subcommand, so paint_res/paint_steps/finish are
    rejected rather than ignored, and no .views.png sheet is written.
    auto_cutout keys out a plain background first unless the input already
    carries real transparency. finish=True applies the game-look texture
    pass (see finish_model) after generation.
    normals=True (default) injects the glTF NORMAL attribute the engine
    omits; without it Godot and friends light the whole mesh off one
    constant vector, which reads as a bad material rather than a missing
    attribute. Geometry is untouched either way.
    Blocks while an earlier generation is running (single-job queue).

    Quality knobs, in rough order of effect. Each defaults to None, which
    omits the flag so the binary's own default (in parens) applies. Paint
    peaks ~25-33GB unified memory and octree/paint_res/texture_size each
    multiply that, so raise them one at a time.

      octree (256)      marching-cubes resolution — the geometry lever.
                        384/512 resolve thin struts the default fuses;
                        vertex count grows roughly cubically.
      paint_res (512)   resolution the multiview texture diffusion runs
                        at — the texture-sharpness lever.
      paint_steps (15)  texture diffusion steps; low next to shape's 30.
      steps (30)        shape diffusion steps; diminishing past ~50.
      guidance (5.0)    how tightly shape follows the image; higher is
                        more faithful but can over-sharpen.

    texture_size: 512|1024|2048|4096, baked texture resolution. The binary's
    own pbr default is 4096; 2048 here keeps peak memory modest.
    superres=False skips the texture super-resolution pass.
    The binary itself validates none of these — out-of-range values fail
    slowly or bake garbage, so prefer moving one knob a single step.
    """
    global _queue_depth
    src = Path(image_path).expanduser()
    if not src.is_file():
        raise ValueError("input image not found: %s" % src)
    try:
        with Image.open(src) as im:
            im.verify()
    except Exception as e:
        raise ValueError("input is not a readable image: %s (%s)" % (src, e))
    if texture_size not in (512, 1024, 2048, 4096):
        raise ValueError("texture_size must be 512, 1024, 2048 or 4096")
    for name, val in (("octree", octree), ("paint_res", paint_res),
                      ("paint_steps", paint_steps), ("steps", steps)):
        if val is not None and val < 1:
            raise ValueError("%s must be a positive integer" % name)
    if guidance is not None and guidance < 0:
        raise ValueError("guidance must be >= 0")
    if not paint:
        # Caught here rather than after a successful shape run, which would
        # spend the geometry pass before failing.
        dead = [n for n, v in (("paint_res", paint_res),
                               ("paint_steps", paint_steps)) if v is not None]
        if dead:
            raise ValueError("paint=False runs the shape-only subcommand, "
                             "which has no paint pass — remove %s or set "
                             "paint=True" % " and ".join(dead))
        if finish:
            raise ValueError("finish=True needs a painted model — it retextures "
                             "an albedo map that shape-only output lacks")

    dst = _out_path(src.stem, ".glb", output_path)
    stages: list[str] = []
    warning: str | None = None
    started = time.monotonic()
    with _queue_guard:
        _queue_depth += 1
    try:
        async with _job_lock:
            gen_input = src
            if auto_cutout and not _has_real_alpha(src):
                rgba = HY3D_OUT / "intermediate" / (src.stem + "-rgba.png")
                cut = await asyncio.to_thread(
                    _run_worker, "cutout.py", [str(src), str(rgba)])
                gen_input = Path(cut["png_path"])
                stages.append("cutout (%.1f%% opaque)" % cut["opaque_pct"])

            # Shape-only is a separate subcommand, not a dropped flag: the
            # binary's `generate` requires --paint-weights unconditionally,
            # and `shape` names the shape checkpoint --weights instead of
            # --shape-weights. The shared knobs below apply to both.
            if paint:
                cmd = [str(BINARY), "generate", str(gen_input), "-o", str(dst),
                       "--shape-weights", "weights/shape-small",
                       "--paint-weights", "weights/paint-large",
                       "--paint-model", "pbr", "--tex", str(texture_size)]
                if paint_steps is not None:
                    cmd += ["--paint-steps", str(paint_steps)]
                if paint_res is not None:
                    cmd += ["--res", str(paint_res)]
                if not superres:
                    cmd.append("--no-superres")
            else:
                cmd = [str(BINARY), "shape", str(gen_input), "-o", str(dst),
                       "--weights", "weights/shape-small"]
            cmd += ["--seed", str(seed)]
            if steps is not None:
                cmd += ["--steps", str(steps)]
            if guidance is not None:
                cmd += ["--guidance", str(guidance)]
            if octree is not None:
                cmd += ["--octree", str(octree)]
            env = os.environ | {"METAL_PATH": str(METALLIB_DIR),
                                "MLX_METAL_PATH": str(METALLIB_DIR)}
            rc, so, se, streamed = await _run_engine(cmd, env, ctx, paint)
            if rc != 0 and "--seed" in " ".join(se[-500:].lower().split()):
                # Older builds without a seed flag: drop it and retry once.
                cmd = [c for i, c in enumerate(cmd)
                       if c != "--seed" and cmd[i - 1] != "--seed"]
                rc, so, se, streamed = await _run_engine(cmd, env, ctx, paint)
            if rc != 0 or not dst.is_file():
                _record_job("generate_model", src.name, False,
                            time.monotonic() - started)
                raise RuntimeError("hy3d %s failed (exit %d):\n%s"
                                   % (cmd[1], rc, (se or so)[-2000:]))
            stages.append("shape+paint" if paint else "shape")

            verts = faces = None
            m = re.search(r"([\d,]+)\s*vert", so, re.I)
            if m:
                verts = int(m.group(1).replace(",", ""))
            m = re.search(r"([\d,]+)\s*(?:faces|tris|triangles)", so, re.I)
            if m:
                faces = int(m.group(1).replace(",", ""))

            if finish:
                fin = await asyncio.to_thread(
                    _run_worker, "finish.py", [str(dst), str(dst)])
                stages.append("finish (%.1f%% accents)"
                              % fin["accent_coverage_pct"])

            if normals:
                # Last: finish.py round-trips through trimesh, which would
                # drop the attribute again if it were injected before.
                nrm = await asyncio.to_thread(_add_normals, dst)
                if nrm.get("normals_added"):
                    stages.append("normals (%d)" % nrm["count"])
                elif nrm.get("warning"):
                    warning = nrm["warning"]

            # The counts above came from the engine's own line, which describes
            # the shape stage. Paint re-parameterises the mesh and splits
            # vertices along UV seams, so that line understates what actually
            # reached disk. Read the file instead, and keep the parsed values
            # as a fallback so this can never return null.
            info = await asyncio.to_thread(_mesh_counts, dst)
            if info.get("verts") is not None:
                verts, faces = info["verts"], info["faces"]
    finally:
        with _queue_guard:
            _queue_depth -= 1

    seconds = time.monotonic() - started
    _record_job("generate_model", src.name, True, seconds)
    out = {"glb_path": str(dst), "verts": verts, "faces": faces,
           "seconds": round(seconds, 1), "stages": stages,
           "progress": "streamed" if streamed else "unavailable"}
    if warning:
        out["warning"] = warning
    return out


@mcp.tool
async def paint_mesh(
    mesh_path: str,
    image_path: str,
    output_path: str | None = None,
    texture_size: int = 2048,
    paint_res: int | None = None,
    paint_steps: int | None = None,
    superres: bool = True,
    auto_cutout: bool = True,
    finish: bool = False,
    normals: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Texture an existing mesh from a concept image. Geometry untouched.

    The paint half of generate_model, reachable on its own: it takes a mesh
    you already have and bakes a PBR texture onto it. Two things that buys —
    re-texturing at different knobs without paying for the shape pass again,
    and texturing geometry this engine did not produce (a blockout, a
    multiview reconstruction, anything the modeller hands you).

    mesh_path: .glb, .gltf and .obj are loaded directly; anything else goes
    through ModelIO and may or may not work, so it is reported as a warning
    rather than refused.
    image_path: the conditioning image, under the same rules as
    generate_model — single object, three-quarter view, evenly lit, plain
    background. It drives the texture only; it cannot move a vertex, so a
    mismatch against the mesh shows up as smeared projection.
    auto_cutout keys out a plain background first unless the image already
    carries real transparency. Leave it on: the paint pipeline composites
    alpha over white but does no keying of its own, so an unkeyed gray
    backdrop is conditioning the texture rather than being ignored.
    finish=True applies the game-look pass afterwards (see finish_model).
    Blocks while an earlier job is running (single-job queue).

    No seed: the paint pipeline re-seeds to 0 internally, so the same mesh
    and image reproduce the same texture. Re-running is free of variance,
    and rerolling is not an option the way it is on shape.

    Knobs, each defaulting to None so the binary's own default (in parens)
    applies:

      paint_res (512)   resolution the multiview texture diffusion runs
                        at — the sharpness lever.
      paint_steps (15)  texture diffusion steps.

    texture_size: 512|1024|2048|4096. The engine's own pbr default is 4096;
    2048 here matches generate_model and keeps peak memory modest.
    superres=False skips the texture super-resolution pass.
    """
    global _queue_depth
    mesh = Path(mesh_path).expanduser()
    src = Path(image_path).expanduser()
    if not mesh.is_file():
        raise ValueError("mesh not found: %s" % mesh)
    if not src.is_file():
        raise ValueError("input image not found: %s" % src)
    try:
        with Image.open(src) as im:
            im.verify()
    except Exception as e:
        raise ValueError("input is not a readable image: %s (%s)" % (src, e))
    if texture_size not in (512, 1024, 2048, 4096):
        raise ValueError("texture_size must be 512, 1024, 2048 or 4096")
    for name, val in (("paint_res", paint_res), ("paint_steps", paint_steps)):
        if val is not None and val < 1:
            raise ValueError("%s must be a positive integer" % name)

    dst = _out_path(mesh.stem + "-painted", ".glb", output_path)
    if dst.resolve() == mesh.resolve():
        raise ValueError("output would overwrite the input mesh (%s) — the "
                         "engine reads it while writing, so give a different "
                         "output_path" % dst)

    stages: list[str] = []
    warning: str | None = None
    if mesh.suffix.lower() not in (".glb", ".gltf", ".obj"):
        warning = ("%s is outside the formats the engine loads directly "
                   "(.glb/.gltf/.obj); it falls back to ModelIO, which may "
                   "not handle it" % (mesh.suffix or "a missing extension"))
    started = time.monotonic()
    with _queue_guard:
        _queue_depth += 1
    try:
        async with _job_lock:
            gen_input = src
            if auto_cutout and not _has_real_alpha(src):
                rgba = HY3D_OUT / "intermediate" / (src.stem + "-rgba.png")
                cut = await asyncio.to_thread(
                    _run_worker, "cutout.py", [str(src), str(rgba)])
                gen_input = Path(cut["png_path"])
                stages.append("cutout (%.1f%% opaque)" % cut["opaque_pct"])

            # The paint subcommand names its flags differently from generate:
            # --weights (the paint root, not the checkpoint), --model, and
            # --steps, which here means paint steps — on generate that same
            # flag is the shape steps.
            cmd = [str(BINARY), "paint", str(mesh), str(gen_input),
                   "-o", str(dst),
                   "--weights", "weights/paint-large",
                   "--model", "pbr", "--tex", str(texture_size)]
            if paint_steps is not None:
                cmd += ["--steps", str(paint_steps)]
            if paint_res is not None:
                cmd += ["--res", str(paint_res)]
            if not superres:
                cmd.append("--no-superres")
            env = os.environ | {"METAL_PATH": str(METALLIB_DIR),
                                "MLX_METAL_PATH": str(METALLIB_DIR)}
            rc, so, se, streamed = await _run_engine(
                cmd, env, ctx, False, stage="paint")
            if rc != 0 or not dst.is_file():
                _record_job("paint_mesh", mesh.name, False,
                            time.monotonic() - started)
                raise RuntimeError("hy3d paint failed (exit %d):\n%s"
                                   % (rc, (se or so)[-2000:]))
            stages.append("paint")

            if finish:
                fin = await asyncio.to_thread(
                    _run_worker, "finish.py", [str(dst), str(dst)])
                stages.append("finish (%.1f%% accents)"
                              % fin["accent_coverage_pct"])

            if normals:
                # Last: finish.py round-trips through trimesh, which would
                # drop the attribute again if it were injected before.
                nrm = await asyncio.to_thread(_add_normals, dst)
                if nrm.get("normals_added"):
                    stages.append("normals (%d)" % nrm["count"])
                elif nrm.get("warning"):
                    warning = nrm["warning"]

            # Paint prints no vert/face line at all, so unlike generate_model
            # there is nothing to parse and the file is the only source.
            info = await asyncio.to_thread(_mesh_counts, dst)
    finally:
        with _queue_guard:
            _queue_depth -= 1

    seconds = time.monotonic() - started
    _record_job("paint_mesh", mesh.name, True, seconds)
    out = {"glb_path": str(dst), "verts": info.get("verts"),
           "faces": info.get("faces"), "seconds": round(seconds, 1),
           "stages": stages,
           "progress": "streamed" if streamed else "unavailable"}
    if warning:
        out["warning"] = warning
    return out


@mcp.tool
def prepare_concept(image_path: str, output_path: str | None = None) -> dict:
    """Key a plain-background concept image out to a centered square RGBA PNG.

    Standalone version of generate_model's auto_cutout, for callers that
    want the intermediate. Refuses inputs whose corners disagree (busy
    background). Warns when the opaque fraction looks like a bad key.
    """
    src = Path(image_path).expanduser()
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
    """Apply the game-look texture pass to a generated GLB. Geometry untouched.

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
    src = Path(glb_path).expanduser()
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

    Rasterising needs pyrender, which despite the "offscreen" name still
    wants a window-server connection — from a daemonised MCP server there
    often isn't one. When it fails for any reason this falls back to the
    contact sheets the paint pass wrote beside the GLB and says so in
    `source`, since those are fixed views, not the ones requested. Only the
    paint pass writes them, so shape-only output has no fallback.
    """
    src = Path(glb_path).expanduser()
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


def _server_status() -> dict:
    binary = _check(
        BINARY.is_file() and os.access(BINARY, os.X_OK),
        "run the setup_engine tool, or build it directly: "
        "cd %s && swift build -c release" % HY3D_REPO)

    metallib = _check(
        (METALLIB_DIR / "default.metallib").is_file()
        and (BUILD_REAL / "default.metallib").is_file(),
        "swift build never emits the MLX metallib (mlx-swift SwiftPM "
        "limitation). Run the setup_engine tool, or by hand: pip mlx and "
        "mlx-swift are separate version series, so install the newest pip "
        "mlx sharing Package.resolved's major.minor — mlx-swift 0.31.4 "
        "means `uv pip install mlx==0.31.2`, as there is no pip 0.31.4 — "
        "then copy "
        "site-packages/mlx/lib/mlx.metallib as BOTH mlx.metallib and "
        "default.metallib into %s AND into %s (the real build dir — "
        ".build/release is a symlink)" % (METALLIB_DIR, BUILD_REAL))

    weights_root = HY3D_REPO / "weights"
    weights = _check(
        (weights_root / "shape-small").is_dir()
        and (weights_root / "paint-large").is_dir(),
        "run the setup_engine tool, or download the model weights (~12GB) "
        "into %s/{shape-small,paint-large} yourself per the Hunyuan3D-MLX "
        "README" % weights_root)

    paint = weights_root / "paint-large"
    needed = [paint / "hunyuan3d-paint-v2-0" / "vae",
              paint / "hunyuan3d-paint-v2-0" / "unet",
              paint / "hunyuan3d-paintpbr-v2-1" / "vae",
              paint / "hunyuan3d-paintpbr-v2-1" / "unet",
              paint / "dinov2-giant"]
    layout = _check(
        all(p.exists() for p in needed),
        "the paint-large HF repo ships flat but the binary expects nested "
        "paths. Inside %s run: mkdir -p hunyuan3d-paint-v2-0 "
        "hunyuan3d-paintpbr-v2-1 && ln -s ../vae ../unet "
        "hunyuan3d-paint-v2-0/ && ln -s ../vae ../unet "
        "hunyuan3d-paintpbr-v2-1/ && ln -s dinov2 dinov2-giant" % paint)

    venv_ok = False
    venv_fix = ("worker venv missing: run the setup_engine tool, or set "
                "HY3D_PY to a python with cv2/numpy/trimesh/PIL/scipy/"
                "pygltflib")
    if HY3D_PY.is_file():
        probe = subprocess.run(
            [str(HY3D_PY), "-c",
             "import cv2, numpy, trimesh, PIL, scipy, pygltflib"],
            capture_output=True, text=True, timeout=60.0)
        venv_ok = probe.returncode == 0
        if not venv_ok:
            # uv, not `python -m pip`: uv-created venvs ship without pip, so
            # the pip form fails with "No module named pip" on a stock setup.
            venv_fix = ("worker venv at %s is missing packages: %s — install "
                        "them with `uv pip install --python %s opencv-python "
                        "numpy trimesh pillow scipy pygltflib`, or run the "
                        "setup_engine tool"
                        % (HY3D_PY, probe.stderr.strip()[-300:], HY3D_PY))

    return {
        "binary_ok": binary, "metallib_ok": metallib, "weights_ok": weights,
        "layout_ok": layout, "venv_ok": _check(venv_ok, venv_fix),
        "queue_depth": _queue_depth, "last_job": _last_job,
        "config": {"HY3D_REPO": str(HY3D_REPO), "HY3D_PY": str(HY3D_PY),
                   "HY3D_OUT": str(HY3D_OUT)},
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

    Validates every setup requirement (binary, metallib, weights, weight
    layout, worker venv); each failing check carries the exact fix. Also
    reports queue depth and the last job.
    """
    return _server_status()


@mcp.tool
async def setup_engine(
    confirm: bool = False,
    only: int | None = None,
    repo: str | None = None,
    worker_venv: str | None = None,
) -> dict:
    """Install or repair the Hunyuan3D-MLX engine this server shells out to.

    Runs the bundled install.sh: checkout, swift build, metallib harvest,
    ~12GB weight download, paint-large relayout, worker venv. Every phase
    inspects before acting, so re-running after a failure resumes.

    Defaults to a DRY RUN — it reports the plan and changes nothing. Show
    that plan to the user, and only re-call with confirm=True once they
    have agreed: applying costs a ~4 minute build and a ~12GB download.
    only=N runs a single phase (1 preflight, 2 clone, 3 build, 4 metallib,
    5 weights, 6 layout, 7 worker venv). Blocks generation while it runs.
    """
    global _queue_depth
    if not INSTALLER.is_file():
        raise RuntimeError("installer not found at %s" % INSTALLER)
    if only is not None and not 1 <= only <= 7:
        raise ValueError("only must be a phase number from 1 to 7")

    argv = [str(INSTALLER), "--yes" if confirm else "--plan"]
    if only is not None:
        argv += ["--only", str(only)]
    if repo:
        argv += ["--repo", str(Path(repo).expanduser())]
    if worker_venv:
        argv += ["--worker-venv", str(Path(worker_venv).expanduser())]

    started = time.monotonic()
    with _queue_guard:
        _queue_depth += 1
    try:
        async with _job_lock:
            proc = await asyncio.to_thread(
                subprocess.run, ["/bin/bash"] + argv, capture_output=True,
                text=True, timeout=SETUP_TIMEOUT)
    finally:
        with _queue_guard:
            _queue_depth -= 1

    seconds = time.monotonic() - started
    ok = proc.returncode == 0
    _record_job("setup_engine", "plan" if not confirm else "apply", ok, seconds)
    out = (proc.stdout or "") + (proc.stderr or "")
    result = {"mode": "plan" if not confirm else "apply", "ok": ok,
              "seconds": round(seconds, 1), "output": out.strip()[-6000:]}
    if not confirm:
        result["next"] = ("nothing was executed. Relay this plan to the user; "
                          "re-call with confirm=True only once they agree to "
                          "the build and the ~12GB download.")
    else:
        result["next"] = ("call server_status to verify" if ok else
                          "setup did not finish; the output says which phase "
                          "failed. Re-calling resumes from there.")
    return result


def main() -> None:
    HY3D_OUT.mkdir(parents=True, exist_ok=True)
    mcp.run()


if __name__ == "__main__":
    main()
