"""Report a GLB's vertex and triangle counts.

The engine prints the counts its shape stage produced, but the paint pass
re-parameterises the mesh afterwards and splits vertices along UV seams — a
twin-cannon turret left the shape stage with 63,552 vertices and reached disk
with 90,964. Quoting the engine's line therefore understates a painted model
by a wide margin, which matters to anyone sizing LODs off it.

Counts come from accessor metadata, so no buffer is decoded and the cost is a
file open regardless of mesh size.

Runs under the worker interpreter (HY3D_PY); needs pygltflib. Prints one JSON
line on stdout: {"glb_path", "verts", "faces"}. Anything it cannot handle
exits 0 with a "warning" instead of failing, so a caller running this as a
post-step can never lose a finished model to it.
"""
import argparse
import json
import sys
from pathlib import Path


def emit(payload: dict) -> None:
    print(json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    args = ap.parse_args()
    src = Path(args.input)

    try:
        from pygltflib import GLTF2
    except ImportError as e:
        emit({"glb_path": str(src),
              "warning": "cannot count mesh: %r missing from the worker venv "
                         "— install it with: uv pip install --python %s %s"
                         % (e.name, sys.executable, e.name)})
        return

    try:
        gltf = GLTF2().load(str(src))
    except Exception as e:
        emit({"glb_path": str(src), "warning": "cannot read %s: %s"
                                               % (src.name, e)})
        return

    verts = faces = 0
    for mesh in gltf.meshes or []:
        for prim in mesh.primitives:
            pos = getattr(prim.attributes, "POSITION", None)
            if pos is None:
                continue
            n = gltf.accessors[pos].count
            verts += n
            if prim.indices is not None:
                faces += gltf.accessors[prim.indices].count // 3
            else:
                # Non-indexed geometry draws every three vertices as a triangle.
                faces += n // 3

    emit({"glb_path": str(src), "verts": verts, "faces": faces})


if __name__ == "__main__":
    main()
