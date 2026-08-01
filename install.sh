#!/usr/bin/env bash
#
# Guided setup for the Hunyuan3D-MLX engine that hy3d-mcp shells out to.
#
# Every phase is idempotent: it inspects the target and skips work that is
# already done, so re-running after a failure resumes rather than restarts.
# The two expensive phases (swift build, 12GB weight download) ask before
# they spend, unless --yes is passed.
#
#   ./install.sh              inspect, then run, confirming expensive steps
#   ./install.sh --plan       print what would happen and exit
#   ./install.sh --yes        run unattended (for CI or the setup_engine tool)
#   ./install.sh --only 3     run a single phase
#
set -uo pipefail

HY3D_REPO="${HY3D_REPO:-$HOME/git/repos/hunyuan3d-mlx}"
HY3D_REPO="${HY3D_REPO/#\~/$HOME}"
WORKER_VENV="${HY3D_WORKER_VENV:-$HOME/.hy3d/worker-venv}"
ENGINE_GIT="https://github.com/ZimengXiong/Hunyuan3D-MLX.git"

# .build/release is a symlink into this; the metallib has to land in the real
# directory too or the binary starts and then fails to find its Metal kernels.
BUILD_REAL_REL=".build/arm64-apple-macosx/release"

WORKER_PKGS=(opencv-python numpy trimesh pillow scipy pyrender)
SHAPE_HF="zimengxiong/hunyuan3d-mlx-shape-small"
PAINT_HF="zimengxiong/hunyuan3d-mlx-paint-large"

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
bad()  { printf '  %sFAILED%s  %s\n' "$RED" "$RST" "$*"; FAILED=1; }
phase(){ printf '\n%s[%s/7] %s%s\n' "$DIM" "$1" "$2" "$RST"; }

usage() {
    sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --plan)   MODE=plan ;;
        --yes|-y) ASSUME_YES=1 ;;
        --only)   ONLY="${2:-}"; shift ;;
        --repo)   HY3D_REPO="${2:-}"; shift ;;
        --worker-venv) WORKER_VENV="${2:-}"; shift ;;
        -h|--help) usage ;;
        *) say "unknown argument: $1 (try --help)"; exit 2 ;;
    esac
    shift
done

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
    local arch os
    arch="$(uname -m)"
    if [ "$arch" != "arm64" ]; then
        bad "this pipeline is Apple Silicon only (found $arch)"
    else
        ok "Apple Silicon"
    fi

    os="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
    ok "macOS $os"

    if command -v swift >/dev/null 2>&1; then
        ok "swift $(swift --version 2>/dev/null | head -1 | sed 's/.*version \([0-9.]*\).*/\1/')"
    else
        bad "swift not found — install Xcode or the Command Line Tools: xcode-select --install"
    fi

    for tool in git uv python3; do
        if command -v "$tool" >/dev/null 2>&1; then
            ok "$tool"
        else
            case "$tool" in
                uv) bad "uv not found — https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)" ;;
                *)  bad "$tool not found" ;;
            esac
        fi
    done

    # weights 12G + build 1.3G + metallib, with headroom for the HF cache.
    local free
    free="$(df -g "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')"
    if [ -n "$free" ] && [ "$free" -lt 20 ]; then
        bad "only ${free}GB free on $HOME — the engine needs ~15GB (12GB weights, 1.3GB build)"
    elif [ -n "$free" ]; then
        ok "${free}GB free"
    fi

    local mem
    mem="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
    if [ "$mem" -gt 0 ] && [ "$mem" -lt 32 ]; then
        say "  ${YEL}warn${RST}    ${mem}GB unified memory — texture paint peaks 25-33GB and will swap"
    elif [ "$mem" -gt 0 ]; then
        ok "${mem}GB unified memory"
    fi
}

# ---------------------------------------------------------------- phase 2
clone_engine() {
    phase 2 "engine checkout — $HY3D_REPO"
    if [ -d "$HY3D_REPO/.git" ]; then
        skip "already cloned ($(git -C "$HY3D_REPO" rev-parse --short HEAD 2>/dev/null))"
        return 0
    fi
    if [ -e "$HY3D_REPO" ]; then
        bad "$HY3D_REPO exists but is not a git checkout — move it aside first"
        return 1
    fi
    [ "$MODE" = plan ] && { work "git clone $ENGINE_GIT (~200MB)"; return 0; }
    work "cloning $ENGINE_GIT"
    mkdir -p "$(dirname "$HY3D_REPO")"
    git clone --depth 1 "$ENGINE_GIT" "$HY3D_REPO" || { bad "clone failed"; return 1; }
    ok "cloned"
}

# ---------------------------------------------------------------- phase 3
build_binary() {
    phase 3 "build the hy3d binary"
    local bin="$HY3D_REPO/.build/release/hy3d"
    if [ -x "$bin" ]; then
        skip "already built ($(cd "$HY3D_REPO" && du -h .build 2>/dev/null | tail -1 | cut -f1))"
        return 0
    fi
    [ -d "$HY3D_REPO" ] || { bad "no checkout at $HY3D_REPO — run phase 2 first"; return 1; }
    [ "$MODE" = plan ] && { work "swift build -c release (~4 min, ~1.3GB)"; return 0; }
    confirm "swift build -c release — takes about 4 minutes and writes ~1.3GB" || return 0
    work "building (this is the slow one)"
    ( cd "$HY3D_REPO" && swift build -c release ) || { bad "swift build failed"; return 1; }
    [ -x "$bin" ] && ok "built $bin" || bad "build reported success but $bin is missing"
}

# ---------------------------------------------------------------- phase 4
install_metallib() {
    phase 4 "metallib"
    local dest_a="$HY3D_REPO/metallib"
    local dest_b="$HY3D_REPO/$BUILD_REAL_REL"
    if [ -f "$dest_a/default.metallib" ] && [ -f "$dest_b/default.metallib" ]; then
        skip "present in both metallib/ and $BUILD_REAL_REL"
        return 0
    fi

    local ver
    ver="$(python3 - "$HY3D_REPO/Package.resolved" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
pins = d.get("pins") or d.get("object", {}).get("pins", [])
for p in pins:
    name = p.get("identity") or p.get("package", "")
    if "mlx-swift" in name.lower():
        st = p.get("state", {})
        print(st.get("version") or st.get("revision", ""))
        break
PY
)"
    if [ -z "$ver" ]; then
        bad "could not read the mlx-swift version from $HY3D_REPO/Package.resolved"
        return 1
    fi
    ok "mlx-swift pinned at $ver"

    # mlx-swift and the pip mlx package are separate version series — there is
    # no pip mlx 0.31.4 to match mlx-swift 0.31.4. Matching on major.minor and
    # taking the newest patch is what reproduces a working metallib (verified:
    # mlx-swift 0.31.4 -> pip mlx 0.31.2, byte-identical metallib).
    local pip_ver
    pip_ver="$(python3 - "$ver" <<'PY' 2>/dev/null
import json, sys, urllib.request
series = ".".join(sys.argv[1].split(".")[:2])
try:
    with urllib.request.urlopen("https://pypi.org/pypi/mlx/json", timeout=30) as r:
        releases = json.load(r)["releases"]
except Exception:
    sys.exit(1)
cand = [v for v in releases if v.startswith(series + ".") and releases[v]]
if not cand:
    sys.exit(2)
print(max(cand, key=lambda v: tuple(int(x) for x in v.split(".") if x.isdigit())))
PY
)"
    if [ -z "$pip_ver" ]; then
        bad "no pip mlx release in the ${ver%.*}.x series (checked PyPI) — the
        metallib must come from an mlx wheel matching mlx-swift $ver"
        return 1
    fi
    ok "matching pip mlx release: $pip_ver"

    [ "$MODE" = plan ] && { work "harvest mlx.metallib from pip mlx==$pip_ver (~158MB, copied to 2 locations)"; return 0; }

    # swift build never emits the MLX metallib (mlx-swift SwiftPM limitation),
    # so it has to come out of the pip wheel of the matching version.
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    work "installing pip mlx==$pip_ver into a scratch venv to harvest its metallib"
    uv venv "$tmp/venv" >/dev/null 2>&1 || { bad "uv venv failed"; return 1; }
    uv pip install --python "$tmp/venv/bin/python" "mlx==$pip_ver" >/dev/null 2>&1 \
        || { bad "uv pip install mlx==$pip_ver failed"; return 1; }

    local src
    src="$(find "$tmp/venv" -path "*/mlx/lib/mlx.metallib" -print -quit 2>/dev/null)"
    [ -n "$src" ] || src="$(find "$tmp/venv" -name "mlx.metallib" -print -quit 2>/dev/null)"
    if [ -z "$src" ]; then
        bad "mlx.metallib not found inside the mlx==$pip_ver wheel"
        return 1
    fi

    mkdir -p "$dest_a" "$dest_b"
    for dest in "$dest_a" "$dest_b"; do
        cp "$src" "$dest/mlx.metallib"     || { bad "copy to $dest failed"; return 1; }
        cp "$src" "$dest/default.metallib" || { bad "copy to $dest failed"; return 1; }
    done
    ok "metallib installed in metallib/ and $BUILD_REAL_REL"
}

# ---------------------------------------------------------------- phase 5
download_weights() {
    phase 5 "model weights"
    local w="$HY3D_REPO/weights"
    local need=()
    [ -d "$w/shape-small" ] || need+=("$SHAPE_HF -> weights/shape-small")
    [ -d "$w/paint-large" ] || need+=("$PAINT_HF -> weights/paint-large")
    if [ ${#need[@]} -eq 0 ]; then
        skip "both weight sets present ($(du -sh "$w" 2>/dev/null | awk '{print $1}'))"
        return 0
    fi
    [ "$MODE" = plan ] && { for n in "${need[@]}"; do work "download $n"; done
                            work "total ~12GB from HuggingFace"; return 0; }

    confirm "download ~12GB of model weights from HuggingFace" || return 0

    local py="$WORKER_VENV/bin/python"
    [ -x "$py" ] || py="$(command -v python3)"
    "$py" -c "import huggingface_hub" 2>/dev/null || {
        work "installing huggingface_hub into the worker venv"
        ensure_worker_venv_quiet
        py="$WORKER_VENV/bin/python"
        uv pip install --python "$py" huggingface_hub >/dev/null 2>&1 \
            || { bad "could not install huggingface_hub"; return 1; }
    }

    mkdir -p "$w"
    local pair
    for pair in "$SHAPE_HF:shape-small" "$PAINT_HF:paint-large"; do
        local repo="${pair%%:*}" dir="${pair##*:}"
        [ -d "$w/$dir" ] && { skip "weights/$dir already there"; continue; }
        work "downloading $repo (this takes a while)"
        "$py" - "$repo" "$w/$dir" <<'PY' || { bad "download of $repo failed"; return 1; }
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
        ok "weights/$dir"
    done
}

# ---------------------------------------------------------------- phase 6
relayout_paint() {
    phase 6 "paint-large layout"
    local p="$HY3D_REPO/weights/paint-large"
    if [ ! -d "$p" ]; then
        skip "paint-large not downloaded yet"
        return 0
    fi
    if [ -e "$p/hunyuan3d-paint-v2-0/vae" ] && [ -e "$p/hunyuan3d-paintpbr-v2-1/vae" ] \
       && [ -e "$p/dinov2-giant" ]; then
        skip "already nested correctly"
        return 0
    fi
    [ "$MODE" = plan ] && { work "symlink the flat HF layout into the nested one the binary expects"; return 0; }

    # The HF repo ships flat (vae/, unet/, dinov2/) but the binary looks for
    # them nested under per-model directories, and under dinov2-giant.
    work "relinking"
    ( cd "$p" || exit 1
      mkdir -p hunyuan3d-paint-v2-0 hunyuan3d-paintpbr-v2-1
      for model in hunyuan3d-paint-v2-0 hunyuan3d-paintpbr-v2-1; do
          for part in vae unet; do
              [ -e "$model/$part" ] || ln -s "../$part" "$model/$part"
          done
      done
      [ -e dinov2-giant ] || ln -s dinov2 dinov2-giant
    ) || { bad "relayout failed"; return 1; }
    ok "nested layout in place"
}

# ---------------------------------------------------------------- phase 7
ensure_worker_venv_quiet() {
    [ -x "$WORKER_VENV/bin/python" ] && return 0
    mkdir -p "$(dirname "$WORKER_VENV")"
    uv venv "$WORKER_VENV" >/dev/null 2>&1
}

worker_venv() {
    phase 7 "worker python environment — $WORKER_VENV"
    local py="$WORKER_VENV/bin/python"

    # An existing HY3D_PY that already satisfies the imports is the environment
    # the server is actually configured to use; don't build a second one.
    if [ -n "${HY3D_PY:-}" ]; then
        local configured="${HY3D_PY/#\~/$HOME}"
        if [ -x "$configured" ] && "$configured" -c \
           "import cv2, numpy, trimesh, PIL, scipy, pyrender" 2>/dev/null; then
            skip "HY3D_PY already satisfies every worker package ($configured)"
            WORKER_VENV="$(dirname "$(dirname "$configured")")"
            return 0
        fi
    fi

    if [ -x "$py" ] && "$py" -c "import cv2, numpy, trimesh, PIL, scipy, pyrender" 2>/dev/null; then
        skip "already has every worker package"
        return 0
    fi
    [ "$MODE" = plan ] && { work "create venv and install ${WORKER_PKGS[*]} (~400MB)"; return 0; }

    if [ ! -x "$py" ]; then
        work "creating venv"
        ensure_worker_venv_quiet || { bad "uv venv failed"; return 1; }
    fi
    work "installing ${WORKER_PKGS[*]}"
    # uv, not pip: uv-created venvs have no pip in them at all.
    uv pip install --python "$py" "${WORKER_PKGS[@]}" >/dev/null 2>&1 \
        || { bad "package install failed — retry with: uv pip install --python $py ${WORKER_PKGS[*]}"; return 1; }
    "$py" -c "import cv2, numpy, trimesh, PIL, scipy, pyrender" 2>/dev/null \
        && ok "all worker packages import" \
        || bad "packages installed but the import check still fails"
}

# ----------------------------------------------------------------- driver
say "${DIM}hy3d engine setup${RST}"
say "  engine repo : $HY3D_REPO"
say "  worker venv : $WORKER_VENV"
[ "$MODE" = plan ] && say "  ${YEL}plan only — nothing will be executed${RST}"

wanted 1 && preflight
if [ "$FAILED" = 1 ] && [ -z "$ONLY" ]; then
    say "\n${RED}preflight failed${RST} — fix the items above and re-run."
    exit 1
fi
wanted 2 && clone_engine
wanted 3 && build_binary
wanted 4 && install_metallib
wanted 5 && download_weights
wanted 6 && relayout_paint
wanted 7 && worker_venv

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
say "  HY3D_REPO=$HY3D_REPO"
say "  HY3D_PY=$WORKER_VENV/bin/python"
say ""
say "Then call the ${DIM}server_status${RST} tool — every check should report ok."
