"""
convert_field_map.py: an sPHENIX field map (ROOT ntuple "fieldmap", as PHField3DCartesian / PHField2D read
it) to the plain-text map ACTS's core Python bindings load (acts.MagneticFieldMapXyz / MagneticFieldMapRz).
Runs in the training environment (needs uproot and numpy, not ACTS or ROOT).

    python scripts/acts/detectors/sphenix/convert_field_map.py sphenix3dtrackingmapxyz.root field_xyz.txt
    python scripts/acts/detectors/sphenix/convert_field_map.py map2d.root field_rz.txt --kind rz
    python scripts/acts/detectors/sphenix/convert_field_map.py --uniform 1.4 uniform_xyz.txt     (a test map: constant Bz)

Input: cartesian ntuple branches x, y, z (cm), bx, by, bz (T); or r-z ntuple branches r, z (cm), br, bz
(T; an old "map" ntuple is in Gauss: --field_unit 1e-4). --rescale multiplies the field, as sPHENIX's
magfield_rescale does.
Output: whitespace columns x y z Bx By Bz (rz: r z Br Bz) in mm and T (ACTS's default units), every grid
point of a full regular grid, ordered as ACTS indexes it (xyz: x slowest, z fastest; rz: z slowest, r
fastest). ACTS has no field outside the map: propagation through a point outside it fails, so the map must
cover the beam line out to the TPC.
"""
import argparse
import sys

import numpy as np


def regular_axis(values, name):
    axis = np.unique(values)
    steps = np.diff(axis)
    if len(axis) > 1 and not np.allclose(steps, steps[0], rtol=1e-4, atol=1e-6):
        raise ValueError(f"{name} is not on a regular grid (ACTS needs equal spacing): steps {steps.min()} to {steps.max()}")
    return axis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", help="sPHENIX ROOT field map")
    parser.add_argument("output", help="text map for ACTS")
    parser.add_argument("--kind", default="xyz", choices=("xyz", "rz"))
    parser.add_argument("--tree", default="fieldmap", help="ntuple name")
    parser.add_argument("--length_unit", type=float, default=10.0, help="mm per file length unit (cm: 10)")
    parser.add_argument("--field_unit", type=float, default=1.0, help="T per file field unit (Gauss: 1e-4)")
    parser.add_argument("--rescale", type=float, default=1.0, help="field scale factor (sPHENIX magfield_rescale)")
    parser.add_argument("--uniform", type=float, default=None, help="instead of an input: constant Bz (T) test map")
    parser.add_argument("--uniform_extent", type=float, nargs=2, default=(1000.0, 1500.0),
                        help="test map half-sizes in mm: transverse, z")
    parser.add_argument("--uniform_bins", type=int, default=21, help="test map points per axis")
    args = parser.parse_args()

    if args.uniform is not None:
        t, zl = args.uniform_extent
        if args.kind == "xyz":
            x, y, z = np.meshgrid(*(np.linspace(-h, h, args.uniform_bins) for h in (t, t, zl)), indexing="ij")
            columns = {"x": x.ravel(), "y": y.ravel(), "z": z.ravel(), "bx": 0 * x.ravel(), "by": 0 * x.ravel(),
                       "bz": np.full(x.size, args.uniform)}
        else:
            r, z = np.meshgrid(np.linspace(0, t, args.uniform_bins), np.linspace(-zl, zl, args.uniform_bins), indexing="ij")
            columns = {"r": r.ravel(), "z": z.ravel(), "br": 0 * r.ravel(), "bz": np.full(r.size, args.uniform)}
        length, field = 1.0, args.rescale
    else:
        import uproot
        names = ("x", "y", "z", "bx", "by", "bz") if args.kind == "xyz" else ("r", "z", "br", "bz")
        with uproot.open(args.input) as f:
            columns = {k: np.asarray(v, dtype=np.float64) for k, v in f[args.tree].arrays(list(names), library="np").items()}
        length, field = args.length_unit, args.field_unit * args.rescale

    coords = ("x", "y", "z") if args.kind == "xyz" else ("z", "r")       # ACTS order: slowest first
    for c in coords:
        columns[c] = columns[c] * length
    axes = {c: regular_axis(columns[c], c) for c in coords}
    n_grid = int(np.prod([len(a) for a in axes.values()]))
    if n_grid != len(columns[coords[0]]):
        raise ValueError(f"not a full grid: {len(columns[coords[0]])} points, {n_grid} grid points "
                         f"({' x '.join(str(len(a)) for a in axes.values())}); ACTS needs every grid point")
    order = np.lexsort([columns[c] for c in reversed(coords)])           # lexsort: last key slowest
    if args.kind == "xyz":
        table = np.stack([columns[k][order] * (field if k.startswith("b") else 1.0)
                          for k in ("x", "y", "z", "bx", "by", "bz")], -1)
        header = "# x y z [mm]  Bx By Bz [T]"
    else:
        table = np.stack([columns[k][order] * (field if k.startswith("b") else 1.0) for k in ("r", "z", "br", "bz")], -1)
        header = "# r z [mm]  Br Bz [T]"
    np.savetxt(args.output, table, fmt="%.9g", header=header[2:], comments="# ")
    ranges = ", ".join(f"{c} {a[0]:.1f}..{a[-1]:.1f} mm ({len(a)})" for c, a in axes.items())
    print(f"wrote {args.output}: {len(table)} points, {ranges}; |B| max {np.abs(table[:, len(coords):]).max():.4f} T")
    return 0


if __name__ == "__main__":
    sys.exit(main())
