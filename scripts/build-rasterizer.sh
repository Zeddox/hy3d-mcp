#!/usr/bin/env bash
# Build custom_rasterizer, the CUDA extension the texture pipeline needs.
#
# mesh_render.py has exactly one rasterizer branch (raster_mode == 'cr'), so
# painting a mesh without this extension is not slow, it is impossible. It is
# not on PyPI: it ships as source inside the engine checkout and has to be
# compiled against the same CUDA the installed torch was built for.
#
# The awkward part is nvcc. torch 2.5.1+cu124 needs 12.4 exactly, apt needs a
# password this script does not have, and nvidia-cuda-nvcc-cu12 is a red
# herring -- at 12.4, 12.6 and 12.8 alike that wheel ships ptxas and nvvm and
# no nvcc frontend at all. So we unpack NVIDIA's own .debs into a user-owned
# prefix: no root, exact version, 233MB.
#
# Idempotent: re-running with the extension already importable does nothing.
#
# Usage: scripts/build-rasterizer.sh [--force]
set -uo pipefail

ENGINE_REPO="${HY3D_ENGINE_REPO:-$HOME/git/repos/Hunyuan3D-2}"
ENGINE_REPO="${ENGINE_REPO/#\~/$HOME}"
ENGINE_VENV="${HY3D_ENGINE_VENV:-$HOME/.hy3d/engine-venv}"
ENGINE_VENV="${ENGINE_VENV/#\~/$HOME}"
PY="$ENGINE_VENV/bin/python"
CUDA_PREFIX="${HY3D_CUDA_PREFIX:-$HOME/.hy3d/cuda-12.4}"

# 8.6 is the RTX 3060 Ti (Ampere). Building for every architecture torch knows
# about turns a 72s compile into a very long one for no gain on one machine.
ARCH="${TORCH_CUDA_ARCH_LIST:-8.6}"

# Ubuntu 22.04's repo, not 24.04's: the 2404 repo starts at 12.5, and 12.4 is
# what this torch build wants. These are every package nvcc actually opens --
# the compiler, its runtime headers, and the headers the extension includes.
CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
DEBS=(
    cuda-nvcc-12-4_12.4.131-1_amd64.deb
    cuda-crt-12-4_12.4.131-1_amd64.deb
    cuda-nvvm-12-4_12.4.131-1_amd64.deb
    cuda-cudart-dev-12-4_12.4.127-1_amd64.deb
    cuda-cccl-12-4_12.4.127-1_amd64.deb
    cuda-driver-dev-12-4_12.4.127-1_amd64.deb
    cuda-nvtx-12-4_12.4.127-1_amd64.deb
    cuda-profiler-api-12-4_12.4.127-1_amd64.deb
)

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; RST=$'\033[0m'
[ -t 1 ] || { RED=""; GRN=""; YEL=""; DIM=""; RST=""; }
ok()   { printf '  %sok%s      %s\n' "$GRN" "$RST" "$*"; }
skip() { printf '  %sskip%s    %s\n' "$DIM" "$RST" "$*"; }
work() { printf '  %srun%s     %s\n' "$YEL" "$RST" "$*"; }
bad()  { printf '  %sFAILED%s  %s\n' "$RED" "$RST" "$*"; exit 1; }

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

printf '\n%scustom_rasterizer — %s%s\n' "$DIM" "$ENGINE_VENV" "$RST"

[ -x "$PY" ] || bad "no engine venv at $ENGINE_VENV — run ./install.sh first"
[ -d "$ENGINE_REPO/hy3dgen/texgen/custom_rasterizer" ] \
    || bad "no custom_rasterizer source under $ENGINE_REPO — run ./install.sh --only 2"

# torch first: the extension links against libc10, so importing the kernel
# on its own fails with a missing-library error that looks like a bad build.
if [ "$FORCE" = 0 ] && "$PY" -c 'import torch, custom_rasterizer' 2>/dev/null; then
    skip "already built (pass --force to rebuild)"
    exit 0
fi

# ------------------------------------------------------------------ toolchain
if [ -x "$CUDA_PREFIX/bin/nvcc" ]; then
    skip "nvcc $("$CUDA_PREFIX/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p') at $CUDA_PREFIX"
else
    work "unpacking CUDA 12.4 into $CUDA_PREFIX"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    for deb in "${DEBS[@]}"; do
        curl -fsSL -o "$tmp/$deb" "$CUDA_REPO/$deb" || bad "could not fetch $deb"
        # dpkg-deb is in dpkg, which is present on any Debian-family box and
        # needs no privileges to extract into a directory we own.
        dpkg-deb -x "$tmp/$deb" "$tmp/root" || bad "could not unpack $deb"
    done
    mkdir -p "$CUDA_PREFIX"
    cp -a "$tmp/root/usr/local/cuda-12.4/." "$CUDA_PREFIX/" || bad "could not stage $CUDA_PREFIX"
    ok "nvcc $("$CUDA_PREFIX/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
fi

# cuda-cudart-dev ships libcudart.so as a symlink to libcudart.so.12, which
# lives in the *runtime* deb we deliberately do not install: torch already
# carries that exact library, and having two on the system is how you get a
# process with two CUDA runtimes in it. Point the link at torch's copy.
CUDART="$ENGINE_VENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
[ -f "$CUDART" ] || bad "no libcudart.so.12 under the venv's nvidia wheels — is torch installed?"
ln -sf "$CUDART" "$CUDA_PREFIX/lib64/libcudart.so.12"

# ---------------------------------------------------------------------- build
# The extension includes cusparse.h and friends, which on this box exist only
# inside the nvidia-* wheels rather than under the CUDA prefix. Sweep every
# one of their include dirs onto the host and device compiler both.
NV="$ENGINE_VENV/lib/python3.10/site-packages/nvidia"
INC=""
for d in "$NV"/*/include; do [ -d "$d" ] && INC="$INC -I$d"; done

# Build from a copy: setup.py drops build/ and egg-info into its own
# directory, and the engine checkout is pinned to a commit that should stay
# clean so install.sh can verify it.
src="$(mktemp -d)"
trap 'rm -rf "${tmp:-}" "$src"' EXIT
cp -a "$ENGINE_REPO/hy3dgen/texgen/custom_rasterizer/." "$src/"

work "compiling for sm_${ARCH/./} (about a minute)"
CUDA_HOME="$CUDA_PREFIX" PATH="$CUDA_PREFIX/bin:$PATH" \
TORCH_CUDA_ARCH_LIST="$ARCH" \
CFLAGS="$INC" CXXFLAGS="$INC" NVCC_PREPEND_FLAGS="$INC" \
    "${UV:-uv}" pip install --python "$PY" --no-build-isolation "$src" \
    || bad "build failed — the compiler output above says why"

"$PY" - <<'PY' || bad "the extension imported but could not rasterize"
import torch, custom_rasterizer as cr
pos = torch.tensor([[[-0.5, -0.5, 0.5, 1.0], [0.5, -0.5, 0.5, 1.0],
                     [0.0, 0.5, 0.5, 1.0]]], dtype=torch.float32, device="cuda")
tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device="cuda")
idx, _ = cr.rasterize(pos, tri, (64, 64))
hits = int((idx > 0).sum())
assert hits > 0, "rasterized a triangle and hit nothing"
print("  rasterized %d pixels on %s" % (hits, torch.cuda.get_device_name(0)))
PY
ok "custom_rasterizer built and verified on the GPU"
