"""Give a generated GLB the vertex normals it shipped without.

The pipeline writes POSITION and TEXCOORD_0 and nothing else. Godot does not
synthesise the gap — it hands the shader one constant vector for the whole
mesh, so lighting, fresnel and every normal-derived effect collapse into a
flat wash. It does not present as a missing attribute; it presents as a bad
material, which is what makes it expensive to diagnose.

The attribute is injected rather than round-tripped through a mesh library:
re-exporting rebuilds the material block and destroys the emissive texture
the finish pass writes.

Runs under the worker interpreter (HY3D_PY); needs trimesh, numpy and
pygltflib. Prints one JSON line on stdout:
{"glb_path", "normals_added", "count"}. Anything it cannot handle exits 0
with a "warning" instead of failing, so a caller running this as a
post-step can never lose a finished model to it.
"""
import argparse
import json
import os
import sys
from pathlib import Path

FLOAT = 5126
ARRAY_BUFFER = 34962


def emit(payload: dict) -> None:
    print(json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output", nargs="?")
    args = ap.parse_args()
    src = Path(args.input)
    dst = Path(args.output) if args.output else src

    try:
        import numpy as np
        import trimesh
        from pygltflib import GLTF2, Accessor, BufferView
    except ImportError as e:
        emit({"glb_path": str(src), "normals_added": False,
              "warning": "cannot add normals: %r missing from the worker venv "
                         "— install it with: uv pip install --python %s %s"
                         % (e.name, sys.executable, e.name)})
        return

    gltf = GLTF2().load(str(src))
    prims = [p for m in gltf.meshes for p in m.primitives]
    if len(prims) != 1:
        emit({"glb_path": str(src), "normals_added": False,
              "warning": "expected one primitive, found %d — normals not "
                         "injected" % len(prims)})
        return
    prim = prims[0]
    if prim.attributes.NORMAL is not None:
        emit({"glb_path": str(src), "normals_added": False,
              "count": gltf.accessors[prim.attributes.NORMAL].count,
              "note": "already had normals"})
        return

    # process=False keeps the vertex order the file was written in — trimesh
    # merges coincident vertices by default, and a merged mesh's normals no
    # longer line up index-for-index with the POSITION accessor they sit beside.
    mesh = trimesh.load(str(src), process=False, force="mesh")
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    # A vertex touching only degenerate faces averages to zero length, which
    # glTF forbids (NORMAL must be unit). A handful per mesh is normal;
    # point them up rather than ship an invalid accessor.
    lengths = np.linalg.norm(normals, axis=1)
    degenerate = lengths < 1e-6
    normals[degenerate] = (0.0, 1.0, 0.0)
    count = gltf.accessors[prim.attributes.POSITION].count
    if len(normals) != count:
        emit({"glb_path": str(src), "normals_added": False,
              "warning": "normal count %d != POSITION count %d — normals not "
                         "injected" % (len(normals), count)})
        return

    blob = gltf.binary_blob()
    pad = (-len(blob)) % 4
    offset = len(blob) + pad
    data = normals.tobytes()
    gltf.bufferViews.append(BufferView(
        buffer=0, byteOffset=offset, byteLength=len(data), target=ARRAY_BUFFER))
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews) - 1, componentType=FLOAT,
        count=count, type="VEC3",
        min=normals.min(axis=0).tolist(), max=normals.max(axis=0).tolist()))
    prim.attributes.NORMAL = len(gltf.accessors) - 1
    blob = blob + b"\x00" * pad + data
    gltf.set_binary_blob(blob)
    gltf.buffers[0].byteLength = len(blob)

    # Via a temp so an interrupted write cannot leave a half-written GLB
    # where a good one was — the usual call site rewrites the file in place.
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".normals.tmp")
    # save_binary, not save: pygltflib picks the container from the file
    # extension, and the temp name does not end in .glb.
    gltf.save_binary(str(tmp))
    os.replace(tmp, dst)
    emit({"glb_path": str(dst), "normals_added": True, "count": count})


if __name__ == "__main__":
    main()
