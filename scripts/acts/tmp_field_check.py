#!/usr/bin/env python3
"""
TEMPORARY (delete after use): the converted sPHENIX field map against VertexHead's uniform field. Training environment:

    python scripts/acts/tmp_field_check.py scripts/acts/detectors/sphenix/field_xyz.txt [--bz 1.4]

Prints the map's grid and coverage (against the TPC and its hits' eta range), Bz on a few (r, z) points, and the
mean Bz over the TPC and over the inner path from the TPC to the beam line, relative to the uniform field.
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map")
    parser.add_argument("--bz", type=float, default=1.4, help="VertexHead's uniform field, T")
    args = parser.parse_args()

    with open(args.map) as f:
        table = np.loadtxt((line for line in f if not line.startswith("#")))
    axes = [np.unique(table[:, k]) for k in range(3)]
    field = table[:, 3:6].reshape(*(len(a) for a in axes), 3)                 # x slowest, z fastest
    for name, a in zip("xyz", axes):
        print(f"{name}: {a[0]:.0f} .. {a[-1]:.0f} mm, step {a[1] - a[0]:.1f} mm ({len(a)} points)")
    r_max, z_max = min(-axes[0][0], axes[0][-1], -axes[1][0], axes[1][-1]), min(-axes[2][0], axes[2][-1])
    print(f"covers r <= {r_max:.0f} mm, |z| <= {z_max:.0f} mm; TPC hits: r 314..754 mm, |z| <= about 1055 mm "
          f"(half-length); propagation to the beam stays within r < 314 mm")

    def bz(r, z):
        """Nearest-grid Bz at x = r, y = 0, z (mm)."""
        i, j, k = (np.abs(a - v).argmin() for a, v in zip(axes, (r, 0.0, z)))
        return field[i, j, k, 2]

    zs = [0, 250, 500, 750, 1000, -500, -1000]
    print(f"\nBz [T] at y = 0      " + " ".join(f"z={z:>5d}" for z in zs))
    for r in (0, 100, 200, 300, 400, 500, 600, 700, 780):
        print(f"  r = {r:3d} mm         " + " ".join(f"{bz(r, z):7.4f}" for z in zs))

    x, y, z = np.meshgrid(*axes, indexing="ij")
    r = np.hypot(x, y)
    for name, inside in (("TPC (r 314..754, |z| <= 1000 mm)", (r >= 314) & (r <= 754) & (np.abs(z) <= 1000)),
                         ("inner path (r < 314, |z| <= 1000 mm)", (r < 314) & (np.abs(z) <= 1000))):
        b = field[..., 2][inside]
        print(f"\nmean Bz over the {name}: {b.mean():.4f} T ({(b.mean() / args.bz - 1) * 100:+.2f}% vs {args.bz} T), "
              f"range {b.min():.4f} .. {b.max():.4f}; mean |Br| {np.hypot(field[..., 0], field[..., 1])[inside].mean():.4f} T")


if __name__ == "__main__":
    main()
