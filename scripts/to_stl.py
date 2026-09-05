"""Convert a generated GLB into an STL a slicer will accept.

Two conversions matter and neither is optional:

* **Units.** The engine emits a mesh normalized to roughly a unit box. STL
  carries no units and every slicer reads it as millimetres, so an
  unscaled export arrives as a 2mm trinket. Scale to a real height.
* **Up axis.** glTF is Y-up; slicers are Z-up. Skipping this lands the
  model on its side on the build plate.

Also drops the mesh onto z=0 so it sits on the plate rather than floating,
and reports the manifold checks that decide whether the file slices at all.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh


def report(mesh, label):
    checks = {
        "watertight": mesh.is_watertight,
        "consistent winding": mesh.is_winding_consistent,
        "enclosed volume": mesh.is_volume,
        "single body": mesh.body_count == 1,
    }
    print(f"[{label}]")
    for k, v in checks.items():
        print(f"   {'ok  ' if v else 'FAIL'}  {k}")
    genus = (2 - mesh.euler_number) // 2
    if genus:
        print(f"   note  genus {genus} — the surface has {genus} tunnel(s) through it. "
              "Slices fine, but check it is intentional.")
    return all(checks.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--height", type=float, default=120.0,
                    help="target height in mm along the print Z axis (default 120)")
    ap.add_argument("--min-wall", type=float, default=0.8,
                    help="warn when scaled features fall under this thickness in mm "
                         "(default 0.8, roughly two perimeters of a 0.4mm nozzle)")
    args = ap.parse_args()

    mesh = trimesh.load(args.glb, force="mesh")
    ok = report(mesh, "as generated")

    # glTF Y-up -> slicer Z-up.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    mesh.apply_scale(args.height / mesh.extents[2])
    # Sit on the build plate, centred in X/Y.
    mesh.apply_translation([-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(),
                            -mesh.bounds[0][2]])
    # Re-centre X/Y properly now that Z is pinned.
    c = mesh.bounds.mean(axis=0)
    mesh.apply_translation([-c[0], -c[1], 0])

    x, y, z = mesh.extents
    print(f"\n   size    {x:.1f} x {y:.1f} x {z:.1f} mm")
    print(f"   volume  {mesh.volume / 1000:.1f} cm3 "
          f"({100 * mesh.volume / np.prod(mesh.extents):.1f}% of its bounding box)")

    # Thin-feature screen: the shortest edge is a cheap proxy for the finest
    # detail present. It does not measure wall thickness, but a model whose
    # detail is already sub-nozzle at this scale will lose it.
    edges = mesh.edges_unique_length
    p1 = float(np.percentile(edges, 1))
    if p1 < args.min_wall:
        print(f"   warn    finest detail ~{p1:.2f}mm is under the {args.min_wall}mm "
              f"min-wall guide; scale up or expect it to be dropped")

    out = Path(args.output or Path(args.glb).with_suffix(".stl").name)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"\n   wrote   {out}  ({out.stat().st_size / 1048576:.2f} MB)")
    if not ok:
        print("   NOTE: failed checks above — repair before slicing.")


if __name__ == "__main__":
    main()
