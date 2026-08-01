"""hy3d-gen MCP server: local image-to-3D via Hunyuan3D-MLX.

Stdio FastMCP process; no ML code lives here. Generation shells out to the
hy3d Swift binary, image work shells out to the worker venv (HY3D_PY) via
the scripts in workers/. Tools return file paths, never blobs.

Config (env, with defaults):
  HY3D_REPO  Hunyuan3D-MLX checkout        (~/git/repos/hunyuan3d-mlx)
  HY3D_PY    worker venv python w/ cv2 etc (~/git/repos/trellis-mac/.venv/bin/python)
  HY3D_OUT   default output directory      (~/hy3d-output)
"""
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from fastmcp import FastMCP
from PIL import Image

HY3D_REPO = Path(os.environ.get("HY3D_REPO", "~/git/repos/hunyuan3d-mlx")).expanduser()
HY3D_PY = Path(os.environ.get("HY3D_PY", "~/git/repos/trellis-mac/.venv/bin/python")).expanduser()
HY3D_OUT = Path(os.environ.get("HY3D_OUT", "~/hy3d-output")).expanduser()

WORKERS = Path(__file__).parent / "workers"
BINARY = HY3D_REPO / ".build/release/hy3d"
METALLIB_DIR = HY3D_REPO / "metallib"
# .build/release is a symlink; this is the real dir the binary also loads from
BUILD_REAL = HY3D_REPO / ".build/arm64-apple-macosx/release"

GENERATE_TIMEOUT = 1800.0
WORKER_TIMEOUT = 300.0

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
_job_lock = threading.Lock()
_queue_depth = 0
_queue_guard = threading.Lock()
_last_job: dict | None = None


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


def _out_path(stem: str, suffix: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
    else:
        p = HY3D_OUT / (stem + suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@mcp.tool
def generate_model(
    image_path: str,
    output_path: str | None = None,
    paint: bool = True,
    texture_size: int = 2048,
    seed: int = 42,
    auto_cutout: bool = True,
    finish: bool = False,
) -> dict:
    """Turn a concept image into a textured 3D model (GLB).

    paint=False skips texturing (shape only, ~20s vs ~3-4 min).
    auto_cutout keys out a plain background first unless the input already
    carries real transparency. finish=True applies the game-look texture
    pass (see finish_model) after generation. texture_size: 512|1024|2048.
    Blocks while an earlier generation is running (single-job queue).
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
    if texture_size not in (512, 1024, 2048):
        raise ValueError("texture_size must be 512, 1024 or 2048")

    dst = _out_path(src.stem, ".glb", output_path)
    stages: list[str] = []
    started = time.monotonic()
    with _queue_guard:
        _queue_depth += 1
    try:
        with _job_lock:
            gen_input = src
            if auto_cutout and not _has_real_alpha(src):
                rgba = HY3D_OUT / "intermediate" / (src.stem + "-rgba.png")
                cut = _run_worker("cutout.py", [str(src), str(rgba)])
                gen_input = Path(cut["png_path"])
                stages.append("cutout (%.1f%% opaque)" % cut["opaque_pct"])

            cmd = [str(BINARY), "generate", str(gen_input), "-o", str(dst),
                   "--shape-weights", "weights/shape-small",
                   "--seed", str(seed)]
            if paint:
                cmd += ["--paint-weights", "weights/paint-large",
                        "--paint-model", "pbr", "--tex", str(texture_size)]
            env = os.environ | {"METAL_PATH": str(METALLIB_DIR),
                                "MLX_METAL_PATH": str(METALLIB_DIR)}
            proc = subprocess.run(cmd, cwd=HY3D_REPO, env=env,
                                  capture_output=True, text=True,
                                  timeout=GENERATE_TIMEOUT)
            if proc.returncode != 0 and "--seed" in " ".join(
                    (proc.stderr or "")[-500:].lower().split()):
                # Older builds without a seed flag: drop it and retry once.
                cmd = [c for i, c in enumerate(cmd)
                       if c != "--seed" and cmd[i - 1] != "--seed"]
                proc = subprocess.run(cmd, cwd=HY3D_REPO, env=env,
                                      capture_output=True, text=True,
                                      timeout=GENERATE_TIMEOUT)
            if proc.returncode != 0 or not dst.is_file():
                _record_job("generate_model", src.name, False,
                            time.monotonic() - started)
                raise RuntimeError(
                    "hy3d generate failed (exit %d):\n%s"
                    % (proc.returncode, (proc.stderr or proc.stdout)[-2000:]))
            stages.append("shape+paint" if paint else "shape")

            verts = faces = None
            m = re.search(r"([\d,]+)\s*vert", proc.stdout, re.I)
            if m:
                verts = int(m.group(1).replace(",", ""))
            m = re.search(r"([\d,]+)\s*(?:faces|tris|triangles)", proc.stdout, re.I)
            if m:
                faces = int(m.group(1).replace(",", ""))

            if finish:
                fin = _run_worker("finish.py", [str(dst), str(dst)])
                stages.append("finish (%.1f%% accents)"
                              % fin["accent_coverage_pct"])
    finally:
        with _queue_guard:
            _queue_depth -= 1

    seconds = time.monotonic() - started
    _record_job("generate_model", src.name, True, seconds)
    return {"glb_path": str(dst), "verts": verts, "faces": faces,
            "seconds": round(seconds, 1), "stages": stages}


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
) -> dict:
    """Apply the game-look texture pass to a generated GLB. Geometry untouched.

    Tones the albedo (gamma/contrast/saturation), extracts saturated accents
    and blackhat panel seams into a dedicated glTF emissive texture.
    Separated from generation so it can be re-run with new knobs without
    regenerating. seam_pinstripes 0-1 (0 disables); defaults are the values
    proven on the AEGIS fleet.
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
    _record_job("finish_model", src.name, True, time.monotonic() - started)
    return out


@mcp.tool
def render_preview(glb_path: str, views: list[str] | None = None,
                   size: int = 1024) -> dict:
    """Offscreen renders of a GLB, no engine needed.

    views: subset of iso/front/back/top/side (default [iso]). Needs pyrender
    in the worker venv; the error message says how to add it if missing.
    """
    src = Path(glb_path).expanduser()
    if not src.is_file():
        raise ValueError("GLB not found: %s" % src)
    outdir = HY3D_OUT / "previews"
    started = time.monotonic()
    out = _run_worker("preview.py", [
        str(src), str(outdir),
        "--views", ",".join(views or ["iso"]),
        "--size", str(size)])
    _record_job("render_preview", src.name, True, time.monotonic() - started)
    return out


def _check(ok: bool, fix: str) -> dict:
    return {"ok": ok} if ok else {"ok": False, "fix": fix}


def _server_status() -> dict:
    binary = _check(
        BINARY.is_file() and os.access(BINARY, os.X_OK),
        "build the generator: cd %s && swift build -c release" % HY3D_REPO)

    metallib = _check(
        (METALLIB_DIR / "default.metallib").is_file()
        and (BUILD_REAL / "default.metallib").is_file(),
        "swift build never emits the MLX metallib (mlx-swift SwiftPM "
        "limitation). Install pip mlx matching Package.resolved (e.g. "
        "`uv pip install mlx==0.31.4`), then copy "
        "site-packages/mlx/lib/mlx.metallib as BOTH mlx.metallib and "
        "default.metallib into %s AND into %s (the real build dir — "
        ".build/release is a symlink)" % (METALLIB_DIR, BUILD_REAL))

    weights_root = HY3D_REPO / "weights"
    weights = _check(
        (weights_root / "shape-small").is_dir()
        and (weights_root / "paint-large").is_dir(),
        "download the model weights (~12GB) into %s/{shape-small,paint-large} "
        "per the Hunyuan3D-MLX README" % weights_root)

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
    venv_fix = ("worker venv missing: set HY3D_PY to a python with "
                "cv2/numpy/trimesh/PIL/scipy installed")
    if HY3D_PY.is_file():
        probe = subprocess.run(
            [str(HY3D_PY), "-c", "import cv2, numpy, trimesh, PIL, scipy"],
            capture_output=True, text=True, timeout=60.0)
        venv_ok = probe.returncode == 0
        if not venv_ok:
            venv_fix = ("worker venv at %s is missing packages: %s — install "
                        "them with `%s -m pip install opencv-python numpy "
                        "trimesh pillow scipy`"
                        % (HY3D_PY, probe.stderr.strip()[-300:], HY3D_PY))

    return {
        "binary_ok": binary, "metallib_ok": metallib, "weights_ok": weights,
        "layout_ok": layout, "venv_ok": _check(venv_ok, venv_fix),
        "queue_depth": _queue_depth, "last_job": _last_job,
        "config": {"HY3D_REPO": str(HY3D_REPO), "HY3D_PY": str(HY3D_PY),
                   "HY3D_OUT": str(HY3D_OUT)},
    }


@mcp.tool
def server_status() -> dict:
    """Health check and first-run diagnostic.

    Validates every setup requirement (binary, metallib, weights, weight
    layout, worker venv); each failing check carries the exact fix. Also
    reports queue depth and the last job.
    """
    return _server_status()


def main() -> None:
    HY3D_OUT.mkdir(parents=True, exist_ok=True)
    mcp.run()


if __name__ == "__main__":
    main()
