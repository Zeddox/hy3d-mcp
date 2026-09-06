#!/usr/bin/env bash
#
# Guided setup for the Hunyuan3D-2 engine that hy3d-mcp shells out to,
# on WSL2 or Linux with an NVIDIA card.
#
# Shape-only by design: an 8GB card cannot run the texture stage, so this
# skips custom_rasterizer and DifferentiableRenderer entirely — the two
# components whose compilation breaks most Windows/WSL installs.
#
# Every phase is idempotent: it inspects the target and skips work that is
# already done, so re-running after a failure resumes rather than restarts.
# The expensive phase (a ~4.6GB weight download) asks before it spends,
# unless --yes is passed.
#
#   ./install.sh              inspect, then run, confirming expensive steps
#   ./install.sh --plan       print what would happen and exit
#   ./install.sh --yes        run unattended (for CI or the setup_engine tool)
#   ./install.sh --only 3     run a single phase
#
# Phase numbers are a public contract — setup_engine(only=N) passes them
# straight through — so they do not get renumbered.
#
#   1 preflight   2 checkout   3 venv+torch   4 deps   5 weights   6 verify
#
set -uo pipefail

ENGINE_REPO="${HY3D_ENGINE_REPO:-$HOME/git/repos/Hunyuan3D-2}"
ENGINE_REPO="${ENGINE_REPO/#\~/$HOME}"
ENGINE_VENV="${HY3D_ENGINE_VENV:-$HOME/.hy3d/engine-venv}"
ENGINE_VENV="${ENGINE_VENV/#\~/$HOME}"
ENGINE_GIT="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
# Pinned: upstream moves, and the callback signature the progress protocol
# depends on is not part of any published API.
ENGINE_REF="${HY3D_ENGINE_REF:-f8db63096c8282cb27354314d896feba5ba6ff8a}"

# Shape weights only. The same repo carries the paint stage, which is
# multiples of this and useless on 8GB, so the download is filtered to the
# one subfolder rather than snapshotting the repo.
SHAPE_HF="tencent/Hunyuan3D-2"
SHAPE_SUB="hunyuan3d-dit-v2-0"
# Upstream resolves weights from $HY3DGEN_MODELS (default ~/.cache/hy3dgen)
# before it ever asks HuggingFace — see hy3dgen/shapegen/utils.py — so
# fetching into that tree is what makes the first run offline-clean.
MODELS_DIR="${HY3DGEN_MODELS:-$HOME/.cache/hy3dgen}"
MODELS_DIR="${MODELS_DIR/#\~/$HOME}"

# numpy<2: pymeshlab wheels of this era are built against the numpy 1.x ABI
# and fail at *import*, not at use.
# scikit-image: required by shapegen/models/autoencoders/surface_extractors.py
# for the default mc_algo='mc' path, but COMMENTED OUT in upstream's
# requirements.txt. Omitting it is the first thing that breaks a shape run.
# rembg+onnxruntime: the cutout worker's fallback key for real concept art.
# Deliberately absent: xatlas + diso (texture UV / mc_algo='dmc'),
# gradio/fastapi/uvicorn (demo app), custom_rasterizer, DifferentiableRenderer.
ENGINE_PKGS=(
    "numpy<2"
    transformers diffusers accelerate safetensors
    einops omegaconf pyyaml tqdm
    opencv-python pillow scipy
    trimesh pygltflib scikit-image
    pymeshlab
    rembg onnxruntime
    huggingface_hub
    pyrender
)

# Every pyrender release pins pyopengl==3.1.0, which uv treats as hard, so this
# cannot join ENGINE_PKGS — resolving the two together is reported
# unsatisfiable. 3.1.0 predates numpy 2 and its glGenTextures wrapper raises
# "No array-type handler for type _ctypes.type", which only bites on TEXTURED
# meshes; an untextured render succeeds and hides it. pyrender itself works
# fine against newer pyopengl, so the pin gets overridden after the fact.
GL_MIN="PyOpenGL>=3.1.7"

CORE_IMPORT="import torch, numpy, cv2, PIL, scipy, trimesh, pygltflib, skimage, pymeshlab, rembg"

# System libraries. pymeshlab dlopens libOpenGL.so.0 even headless, and
# failing to find it takes its io_base plugin down with it — decimation then
# surfaces as the thoroughly misleading "Unknown format for load: ply".
# libegl1 is what gives pyrender a context with no display.
APT_LIBS=(libopengl0 libegl1)

MODE=run          # run | plan
ASSUME_YES=0
ONLY=""

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; RST=$'\033[0m'
[ -t 1 ] || { RED=""; GRN=""; YEL=""; DIM=""; RST=""; }

FAILED=0

say()  { printf '%s\n' "$*"; }
ok()   { printf '  %sok%s      %s\n' "$GRN" "$RST" "$*"; }
skip() { printf '  %sskip%s    %s\n' "$DIM" "$RST" "$*"; }
work() { printf '  %srun%s     %s\n' "$YEL" "$RST" "$*"; }
# Degraded but usable — unlike bad(), leaves FAILED alone so the phase passes.
warn() { printf '  %swarn%s    %s\n' "$YEL" "$RST" "$*"; }
bad()  { printf '  %sFAILED%s  %s\n' "$RED" "$RST" "$*"; FAILED=1; }
phase(){ printf '\n%s[%s/6] %s%s\n' "$DIM" "$1" "$2" "$RST"; }

usage() {
    sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --plan)   MODE=plan ;;
        --yes|-y) ASSUME_YES=1 ;;
        --only)   ONLY="${2:-}"; shift ;;
        --repo)   ENGINE_REPO="${2:-}"; shift ;;
        --venv)   ENGINE_VENV="${2:-}"; shift ;;
        -h|--help) usage ;;
        *) say "unknown argument: $1 (try --help)"; exit 2 ;;
    esac
    shift
done

PY="$ENGINE_VENV/bin/python"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

# A phase number nobody runs must not report success: --only is a public
# contract through setup_engine(only=N), and silently doing nothing there
# looks exactly like a clean install.
case "$ONLY" in
    ""|1|2|3|4|5|6) ;;
    *) say "unknown phase: $ONLY (valid: 1 preflight, 2 checkout, 3 venv+torch, 4 deps, 5 weights, 6 verify)"
       exit 2 ;;
esac

wanted() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

# Ask before anything slow or large. In --yes mode every answer is yes; with
# no tty and no --yes we refuse rather than hang waiting on stdin.
confirm() {
    [ "$ASSUME_YES" = 1 ] && return 0
    if [ ! -t 0 ]; then
        bad "$1 needs confirmation but there is no terminal; re-run with --yes"
        return 1
    fi
    printf '  %s? %s [y/N] ' "$YEL$RST" "$1"
    read -r reply
    case "$reply" in [yY]*) return 0 ;; *) skip "declined: $1"; return 1 ;; esac
}

# ---------------------------------------------------------------- phase 1
preflight() {
    phase 1 "preflight"

    if grep -qi microsoft /proc/version 2>/dev/null; then
        ok "WSL2"
    else
        ok "$(uname -s) $(uname -m)"
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local line name total
        line="$(nvidia-smi --query-gpu=name,memory.total,driver_version \
                --format=csv,noheader 2>/dev/null | head -1)"
        if [ -z "$line" ]; then
            bad "nvidia-smi is present but reports no GPU. Inside WSL the
        Windows driver is projected in — never install an NVIDIA driver in
        the WSL guest. Check that /usr/lib/wsl/lib is on the loader path."
        else
            name="${line%%,*}"
            total="$(printf '%s' "$line" | cut -d, -f2 | tr -dc '0-9')"
            ok "$line"
            if [ -n "$total" ] && [ "$total" -lt 8000 ]; then
                warn "${total}MiB of VRAM — this pipeline is tuned for 8GB and \
peaks near 6.2GiB at defaults. Expect to need cpu_offload, or a lower octree."
            fi
        fi
    else
        bad "nvidia-smi not found — no GPU is visible. On WSL this means the
        Windows-side NVIDIA driver is missing or /usr/lib/wsl/lib is not on
        the loader path. Do NOT install a driver inside the WSL guest."
    fi

    local tool
    for tool in git python3; do
        if command -v "$tool" >/dev/null 2>&1; then ok "$tool"; else bad "$tool not found"; fi
    done
    if [ -x "$UV" ]; then
        ok "uv ($UV)"
    else
        bad "uv not found — https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)"
    fi

    # weights 4.6G + venv ~7G (the cu124 torch wheels are most of it) + repo,
    # with headroom.
    local free
    free="$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [ -n "$free" ] && [ "$free" -lt 15 ]; then
        bad "only ${free}GB free on $HOME — this needs ~13GB (4.6GB weights, ~7GB venv)"
    elif [ -n "$free" ]; then
        ok "${free}GB free"
    fi

    # Reported, not installed: these need root, and the rest of the install
    # does not. Phase 6 proves whether they are actually working.
    local missing=()
    for lib in "${APT_LIBS[@]}"; do
        dpkg -s "$lib" 2>/dev/null | grep -q '^Status: install ok' || missing+=("$lib")
    done
    if [ ${#missing[@]} -eq 0 ]; then
        ok "system GL libraries (${APT_LIBS[*]})"
    else
        warn "missing system libraries: ${missing[*]} — install them with:
             sudo apt install ${missing[*]}
          libopengl0 gates pymeshlab, and therefore decimation; libegl1 gates
          headless preview rendering. Generation itself works without both."
    fi
}

# ---------------------------------------------------------------- phase 2
clone_engine() {
    phase 2 "engine checkout — $ENGINE_REPO"
    if [ -d "$ENGINE_REPO/.git" ]; then
        skip "already cloned ($(git -C "$ENGINE_REPO" rev-parse --short HEAD 2>/dev/null))"
        return 0
    fi
    if [ -e "$ENGINE_REPO" ]; then
        bad "$ENGINE_REPO exists but is not a git checkout — move it aside first"
        return 1
    fi
    [ "$MODE" = plan ] && { work "git clone $ENGINE_GIT at ${ENGINE_REF:0:8} (~200MB)"; return 0; }
    work "cloning $ENGINE_GIT"
    mkdir -p "$(dirname "$ENGINE_REPO")"
    git clone "$ENGINE_GIT" "$ENGINE_REPO" || { bad "clone failed"; return 1; }
    git -C "$ENGINE_REPO" checkout --quiet "$ENGINE_REF" \
        || { bad "checkout of $ENGINE_REF failed"; return 1; }
    ok "cloned at ${ENGINE_REF:0:8}"
}

# ---------------------------------------------------------------- phase 3
build_venv() {
    phase 3 "engine venv — $ENGINE_VENV (python 3.10, torch cu124)"
    if [ -x "$PY" ]; then
        skip "venv present, $("$PY" -V 2>&1)"
    else
        [ "$MODE" = plan ] && { work "uv venv --python 3.10 $ENGINE_VENV"; return 0; }
        work "creating venv"
        mkdir -p "$(dirname "$ENGINE_VENV")"
        "$UV" venv --python 3.10 "$ENGINE_VENV" || { bad "uv venv failed"; return 1; }
    fi

    if "$PY" -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith("2.5.1+cu124") else 1)' 2>/dev/null; then
        skip "torch $("$PY" -c 'import torch;print(torch.__version__)')"
        return 0
    fi
    [ "$MODE" = plan ] && { work "install torch 2.5.1+cu124 and torchvision 0.20.1+cu124 (~3GB)"; return 0; }
    work "installing torch 2.5.1+cu124 / torchvision 0.20.1+cu124"
    # Both from the cu124 index in ONE command. Installing torchvision from
    # PyPI afterwards silently pulls a different torch and discards the cu124
    # build — which then imports perfectly and runs on the CPU.
    "$UV" pip install --python "$PY" \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
        || { bad "torch install failed"; return 1; }
    ok "torch $("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null)"
}

# ---------------------------------------------------------------- phase 4
engine_deps() {
    phase 4 "shape-only python deps"
    [ -x "$PY" ] || { bad "no venv at $ENGINE_VENV — run phase 3 first"; return 1; }

    if "$PY" -c "$CORE_IMPORT" 2>/dev/null && "$PY" -c "import pyrender" 2>/dev/null; then
        skip "every engine package imports"
    else
        [ "$MODE" = plan ] && { work "install ${ENGINE_PKGS[*]} (~2GB)"; return 0; }
        work "installing ${#ENGINE_PKGS[@]} packages"
        # uv, not `python -m pip`: uv-created venvs ship without pip at all.
        "$UV" pip install --python "$PY" "${ENGINE_PKGS[@]}" \
            || { bad "package install failed — retry with: $UV pip install --python $PY ${ENGINE_PKGS[*]}"; return 1; }
    fi

    # Separate pass on purpose: resolving this alongside pyrender fails, so it
    # has to land after pyrender has pulled in the 3.1.0 it asks for.
    if "$PY" -c "import OpenGL, sys
v = tuple(int(''.join(c for c in p if c.isdigit()) or 0) for p in OpenGL.__version__.split('.')[:3])
sys.exit(0 if v >= (3, 1, 7) else 1)" 2>/dev/null; then
        skip "pyopengl $("$PY" -c 'import OpenGL;print(OpenGL.__version__)') clears the render floor"
        return 0
    fi
    [ "$MODE" = plan ] && { work "override pyrender's stale pyopengl pin ($GL_MIN)"; return 0; }
    work "overriding pyrender's stale pyopengl pin ($GL_MIN)"
    # Not fatal: this build's own output is untextured, and an untextured
    # render works on 3.1.0. It bites the moment a textured GLB from
    # elsewhere is previewed.
    "$UV" pip install --python "$PY" --upgrade "$GL_MIN" >/dev/null 2>&1 \
        || warn "could not upgrade pyopengl — this build's own previews still \
render, but a textured GLB from elsewhere will not. Retry with: $UV pip \
install --python $PY --upgrade '$GL_MIN'"
}

# ---------------------------------------------------------------- phase 5
download_weights() {
    phase 5 "model weights — $MODELS_DIR/$SHAPE_HF/$SHAPE_SUB"
    local dest="$MODELS_DIR/$SHAPE_HF"
    local ckpt="$dest/$SHAPE_SUB/model.fp16.safetensors"
    local u2net="$HOME/.u2net/u2net.onnx"

    local need_shape=1 need_u2net=1
    [ -f "$ckpt" ] && need_shape=0
    [ -f "$u2net" ] && need_u2net=0

    [ "$need_shape" = 0 ] && skip "shape weights present ($(du -sh "$dest/$SHAPE_SUB" 2>/dev/null | cut -f1))"
    [ "$need_u2net" = 0 ] && skip "u2net present ($(du -h "$u2net" 2>/dev/null | cut -f1))"
    [ "$need_shape" = 0 ] && [ "$need_u2net" = 0 ] && return 0

    if [ "$MODE" = plan ]; then
        [ "$need_shape" = 1 ] && work "download $SHAPE_HF/$SHAPE_SUB (~4.6GB)"
        [ "$need_u2net" = 1 ] && work "download u2net for rembg (~176MB)"
        return 0
    fi
    [ -x "$PY" ] || { bad "no venv at $ENGINE_VENV — run phase 3 first"; return 1; }

    if [ "$need_shape" = 1 ]; then
        confirm "download ~4.6GB of shape weights from HuggingFace" || return 0
        work "downloading $SHAPE_HF/$SHAPE_SUB (this is the slow one)"
        # allow_patterns, not a bare snapshot: the same repo carries the paint
        # stage. And local_dir into ~/.cache/hy3dgen, because that is the tree
        # upstream consults before it ever contacts HuggingFace, so a run here
        # never re-resolves the repo over the network.
        "$PY" - "$SHAPE_HF" "$SHAPE_SUB" "$dest" <<'PY' || { bad "weight download failed"; return 1; }
import sys
from huggingface_hub import snapshot_download
repo, sub, dest = sys.argv[1:4]
snapshot_download(repo_id=repo, allow_patterns=["%s/*" % sub], local_dir=dest)
PY
        [ -f "$ckpt" ] && ok "shape weights in place" || { bad "download finished but $ckpt is missing"; return 1; }
    fi

    if [ "$need_u2net" = 1 ]; then
        # Load-bearing since the cutout worker took over the fallback key:
        # without it, concept art with a busy background has no path.
        work "fetching u2net for the cutout fallback (~176MB)"
        "$PY" -c "from rembg import new_session; new_session('u2net')" >/dev/null 2>&1 \
            || warn "could not prefetch u2net — it will download on first use instead"
        [ -f "$u2net" ] && ok "u2net cached at $u2net"
    fi
}

# ---------------------------------------------------------------- phase 6
verify() {
    phase 6 "verify"
    [ "$MODE" = plan ] && { work "import the pipeline, check CUDA, render one offscreen pixel"; return 0; }
    [ -x "$PY" ] || { bad "no venv at $ENGINE_VENV — run phase 3 first"; return 1; }

    local out
    out="$("$PY" - "$ENGINE_REPO" <<'PY' 2>&1
import sys
sys.path.insert(0, sys.argv[1])
import torch
print("TORCH", torch.__version__)
print("CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    print("DEVICE", torch.cuda.get_device_name(0))
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
print("PIPELINE ok")
PY
)"
    printf '%s\n' "$out" | grep -q "PIPELINE ok" \
        && ok "engine imports ($(printf '%s' "$out" | grep '^TORCH' | cut -d' ' -f2))" \
        || bad "engine import failed:
$(printf '%s' "$out" | tail -5)"
    if printf '%s\n' "$out" | grep -q "^CUDA True"; then
        ok "$(printf '%s' "$out" | grep '^DEVICE' | cut -d' ' -f2-)"
    else
        bad "torch does not see the GPU. The venv may hold the CPU wheel —
        \`$PY -c 'import torch; print(torch.__version__)'\` must end in +cu124."
    fi

    # pymeshlab's failure mode is a lie ("Unknown format for load: ply"), so
    # prove decimation rather than the import.
    "$PY" - <<'PY' >/dev/null 2>&1
import pymeshlab, numpy as np
m = pymeshlab.MeshSet()
v = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
m.add_mesh(pymeshlab.Mesh(v, f))
m.meshing_decimation_quadric_edge_collapse(targetfacenum=2)
PY
    [ $? -eq 0 ] && ok "pymeshlab decimates" \
        || bad "pymeshlab cannot decimate — this is almost always the missing
        system library, not the wheel: sudo apt install libopengl0"

    # An import proves nothing about EGL: the failure is inside the draw call
    # ("Attempt to retrieve context when no valid context"), so make a context
    # and render an actual pixel.
    "$PY" - <<'PY' >/dev/null 2>&1
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender
s = pyrender.Scene()
s.add(pyrender.Mesh.from_trimesh(trimesh.creation.box()))
s.add(pyrender.PerspectiveCamera(yfov=1.0), pose=np.array(
    [[1.,0,0,0],[0,1,0,0],[0,0,1,4],[0,0,0,1]]))
r = pyrender.OffscreenRenderer(8, 8)
r.render(s)
r.delete()
PY
    [ $? -eq 0 ] && ok "offscreen rendering works (EGL)" \
        || warn "offscreen rendering failed — generation is unaffected, but
        render_preview will not rasterise. Check: sudo apt install libegl1"
}

# ----------------------------------------------------------------- driver
say "${DIM}hy3d engine setup — shape-only, CUDA${RST}"
say "  engine repo : $ENGINE_REPO"
say "  engine venv : $ENGINE_VENV"
say "  weights     : $MODELS_DIR/$SHAPE_HF"
[ "$MODE" = plan ] && say "  ${YEL}plan only — nothing will be executed${RST}"

wanted 1 && preflight
if [ "$FAILED" = 1 ] && [ -z "$ONLY" ]; then
    say "\n${RED}preflight failed${RST} — fix the items above and re-run."
    exit 1
fi
wanted 2 && clone_engine
wanted 3 && build_venv
wanted 4 && engine_deps
wanted 5 && download_weights
wanted 6 && verify

say ""
if [ "$MODE" = plan ]; then
    say "${DIM}plan complete — re-run without --plan to execute.${RST}"
    exit 0
fi
if [ "$FAILED" = 1 ]; then
    say "${RED}setup incomplete${RST} — re-run to resume; finished phases will skip."
    exit 1
fi

say "${GRN}setup complete.${RST}"
say ""
say "Point the MCP server at this environment:"
say "  HY3D_ENGINE_REPO=$ENGINE_REPO"
say "  HY3D_ENGINE_PY=$ENGINE_VENV/bin/python"
say ""
say "Then call the ${DIM}server_status${RST} tool — every check should report ok."
