"""The workbench: a local web front end over the same tools the MCP server has.

Its own process, not another MCP transport. The MCP server is stdio and its
lifecycle belongs to whatever spawned it; this has to outlive that and be
reachable from a browser, so it runs standalone:

    hy3d-web                 # or: python -m hy3d_mcp.web
    http://localhost:8760

No ML code and no second copy of the pipeline. It imports server.py and calls
the tool functions directly -- @mcp.tool returns the plain function -- so the
cutout, the engine driver, the progress parser, the heartbeat, the timeout and
the kill-on-cancel are all the ones the MCP path uses and the ones that have
been tested. What this file adds is HTTP, a job registry, and file naming that
survives an upload button.

Generation is serialized machine-wide by the flock in server.py, so running
this alongside a Claude Code session is safe: the second job waits instead of
spilling into host RAM beside the first.
"""
import argparse
import asyncio
import json
import mimetypes
import os
import re
import socket
import time
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Route

from . import server as S

APP_DIR = Path(__file__).parent / "webapp"
UPLOADS = S.HY3D_OUT / "uploads"
# Concept art, not video. 64MB is already generous for a PNG.
MAX_UPLOAD = 64 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# What /files will serve. Everything else under HY3D_OUT stays private even
# though the path check would allow it.
SERVE_SUFFIXES = {".glb", ".png", ".jpg", ".jpeg", ".webp", ".stl"}

# One event loop owns this, so a plain dict needs no lock. That is also why
# this is uvicorn and not a threaded stdlib server: server._job_lock is an
# asyncio.Lock bound to the first loop that awaits it, and a fresh loop per
# job would lose the single-job guarantee on the second request.
JOBS: dict[str, dict] = {}


class WebCtx:
    """The progress sink _run_engine expects, writing into a job record.

    report_progress must be a coroutine. _run_engine awaits it inside a bare
    `except Exception: pass` -- there so a client that stopped listening
    cannot kill a running bake -- which means a plain def raises, gets
    swallowed, and progress never appears with nothing in any log to say why.
    """

    def __init__(self, job: dict) -> None:
        self.job = job

    async def report_progress(self, value: float, total: float,
                              message: str) -> None:
        self.job["pct"] = round(float(value), 1)
        self.job["message"] = message


def _safe_stem(name: str) -> str:
    """A filename derived from the client's, never the client's.

    The multipart filename is attacker-controlled in principle and careless
    in practice: `../`, a Windows path, a colon, an emoji.
    """
    stem = Path(name.replace("\\", "/")).name
    stem = Path(stem).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.") or "concept"
    return stem[:60]


def _unique_stem(stem: str) -> str:
    """Stamped, because _out_path names the GLB after the input.

    Two uploads both called `image.png`, or one image re-run at a different
    octree, would otherwise overwrite the earlier GLB in place while the
    gallery went on listing it. Under MCP the agent names its outputs; behind
    an upload button this is a bug on day one.
    """
    return "%s-%s" % (stem, time.strftime("%Y%m%d-%H%M%S"))


def _rel(path: Path) -> str | None:
    """A /files URL for something under HY3D_OUT, or None if it is outside."""
    try:
        return "/files/" + str(Path(path).resolve().relative_to(S.HY3D_OUT))
    except ValueError:
        return None


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "task"}


async def index(request):
    page = APP_DIR / "index.html"
    if not page.is_file():
        return HTMLResponse("<h1>hy3d workbench</h1><p>webapp/index.html is "
                            "missing from the install.</p>", status_code=500)
    return HTMLResponse(page.read_text())


async def api_generate(request):
    """Start a job. Either a multipart upload, or a path already on disk."""
    ctype = request.headers.get("content-type", "")
    settings: dict = {}
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None or not getattr(upload, "filename", ""):
            return JSONResponse({"error": "no image in the upload"}, 400)
        blob = await upload.read()
        if len(blob) > MAX_UPLOAD:
            return JSONResponse(
                {"error": "image is %.1fMB; the cap is %dMB"
                          % (len(blob) / 1e6, MAX_UPLOAD // (1024 * 1024))}, 413)
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            return JSONResponse({"error": "unsupported image type %r" % suffix}, 415)
        stem = _unique_stem(_safe_stem(upload.filename))
        UPLOADS.mkdir(parents=True, exist_ok=True)
        src = UPLOADS / (stem + suffix)
        src.write_bytes(blob)
        for key in ("octree", "steps", "max_faces", "seed", "guidance", "model"):
            if form.get(key):
                settings[key] = form[key]
    else:
        body = await request.json()
        given = body.get("path", "")
        src = Path(given).expanduser().resolve()
        if not src.is_file():
            return JSONResponse({"error": "no such file: %s" % given}, 400)
        stem = _unique_stem(_safe_stem(src.name))
        settings = {k: v for k, v in body.items() if k != "path"}

    kwargs: dict = {}
    try:
        for key, cast in (("octree", int), ("steps", int), ("max_faces", int),
                          ("seed", int), ("guidance", float)):
            if settings.get(key) not in (None, ""):
                kwargs[key] = cast(settings[key])
        if settings.get("model"):
            kwargs["model"] = str(settings["model"])
    except (TypeError, ValueError) as e:
        return JSONResponse({"error": "bad setting: %s" % e}, 400)

    job_id = uuid.uuid4().hex[:12]
    # "queued", not "starting": the job may sit on _job_lock behind another
    # tab, or on the flock behind another process, before the engine is ever
    # launched. The first progress line overwrites this.
    job = {"id": job_id, "state": "running", "pct": 0.0,
           "message": "queued", "name": stem,
           "input": _rel(src) or str(src), "started": time.time(),
           "settings": kwargs, "result": None, "error": None}
    JOBS[job_id] = job
    # The registry is a page's worth of history, not a database. Trimming the
    # oldest keeps a long-lived server from growing a record per job forever.
    for stale in sorted(JOBS.values(), key=lambda j: j["started"])[:-50]:
        JOBS.pop(stale["id"], None)
    job["task"] = asyncio.create_task(_run(job, src, stem, kwargs))
    return JSONResponse({"job": job_id})


async def _run(job: dict, src: Path, stem: str, kwargs: dict) -> None:
    try:
        out = await S.generate_model(
            image_path=str(src),
            output_path=str(S.HY3D_OUT / (stem + ".glb")),
            ctx=WebCtx(job), **kwargs)
    except asyncio.CancelledError:
        job.update(state="cancelled", message="cancelled")
        raise
    except Exception as e:
        job.update(state="error", error=str(e)[-2000:], message="failed")
        return
    glb = Path(out["glb_path"])
    # progress is 'unavailable' here by construction -- it reports whether an
    # MCP progressToken was present, and this ctx is not MCP. Dropping it
    # beats faking a request context to make one word come out right.
    out.pop("progress", None)
    out["glb_url"] = _rel(glb)
    job.update(state="done", pct=100.0, message="done", result=out)


async def api_job(request):
    job = JOBS.get(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "unknown job"}, 404)
    return JSONResponse(_public(job))


async def api_cancel(request):
    job = JOBS.get(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "unknown job"}, 404)
    # cancel_job kills the engine child; cancelling the task alone would leave
    # it holding the GPU and the machine-wide lock.
    killed = await S.cancel_job()
    task = job.get("task")
    if task is not None:
        task.cancel()
    job.update(state="cancelled", message="cancelled")
    return JSONResponse({"cancelled": True, "engine": killed})


async def api_stl(request):
    body = await request.json()
    # The front end holds /files URLs, and "/files/x.glb" is an absolute path
    # as far as pathlib is concerned -- strip the prefix before asking.
    given = str(body.get("glb", ""))
    if given.startswith("/files/"):
        glb = (S.HY3D_OUT / given[len("/files/"):]).resolve()
        if not glb.is_relative_to(S.HY3D_OUT.resolve()):
            return JSONResponse({"error": "outside the output directory"}, 403)
    else:
        glb = Path(given).expanduser().resolve()
    if not glb.is_file():
        return JSONResponse({"error": "no such GLB: %s" % glb}, 400)
    try:
        out = await asyncio.to_thread(
            S.export_stl, str(glb),
            height_mm=float(body.get("height_mm", 120.0)),
            min_wall_mm=float(body.get("min_wall_mm", 0.8)))
    except Exception as e:
        return JSONResponse({"error": str(e)[-2000:]}, 500)
    out["stl_url"] = _rel(Path(out["stl_path"]))
    return JSONResponse(out)


async def api_gallery(request):
    items = []
    for glb in S.HY3D_OUT.glob("*.glb"):
        st = glb.stat()
        items.append({"name": glb.stem, "url": _rel(glb),
                      "mb": round(st.st_size / 1e6, 1), "mtime": st.st_mtime})
    items.sort(key=lambda i: i["mtime"], reverse=True)
    return JSONResponse({"items": items[:24]})


async def api_status(request):
    return JSONResponse(await asyncio.to_thread(S.server_status))


async def files(request):
    """Serve a GLB, image or STL out of HY3D_OUT and nothing else."""
    want = (S.HY3D_OUT / request.path_params["path"]).resolve()
    if not want.is_relative_to(S.HY3D_OUT.resolve()):
        return JSONResponse({"error": "outside the output directory"}, 403)
    if want.suffix.lower() not in SERVE_SUFFIXES or not want.is_file():
        return JSONResponse({"error": "not found"}, 404)
    mime = mimetypes.guess_type(want.name)[0] or "application/octet-stream"
    if want.suffix.lower() == ".glb":
        mime = "model/gltf-binary"
    return FileResponse(want, media_type=mime)


app = Starlette(routes=[
    Route("/", index),
    Route("/api/generate", api_generate, methods=["POST"]),
    Route("/api/job/{job_id}", api_job),
    Route("/api/job/{job_id}/cancel", api_cancel, methods=["POST"]),
    Route("/api/stl", api_stl, methods=["POST"]),
    Route("/api/gallery", api_gallery),
    Route("/api/status", api_status),
    Route("/files/{path:path}", files),
])


def _lan_ip() -> str:
    """The guest's own address, for when localhost forwarding is not on.

    No packet is sent -- connect() on a UDP socket only picks the route.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default: this serves files and runs "
                         "generation, so binding wide is opt-in")
    ap.add_argument("--port", type=int, default=8760)
    args = ap.parse_args()

    import uvicorn
    S.HY3D_OUT.mkdir(parents=True, exist_ok=True)
    print("hy3d workbench  ->  http://localhost:%d" % args.port, flush=True)
    print("  outputs: %s" % S.HY3D_OUT, flush=True)
    if args.host == "127.0.0.1":
        # The likeliest way this looks broken on first run: WSL2 normally
        # forwards localhost from Windows, and when the distro's networking
        # mode does not, a loopback bind is invisible from the browser.
        print("  if the Windows browser cannot reach that, re-run with "
              "--host 0.0.0.0 and use http://%s:%d" % (_lan_ip(), args.port),
              flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
