#!/usr/bin/env python3
"""
check_geometry.py: check a detector's ACTS geometry and material map, inside the ACTS image:

    singularity exec acts-base-48.2.0.sif acts-python scripts/acts/check_geometry.py \
        --geometry <tgeo.root> --geometry_config <config.json> [--material_map <material.json>] [--layers]

Prints surfaces per ACTS volume, how many carry material, material-map entries that found no surface and, with
--layers, each sensitive layer's radial extent as ACTS computes it (ProtoLayer) and its overlap with the next layer;
ACTS logs overlapping layers ("Layers are overlapping") while building.
sPHENIX (ana559): 432 MVTX, 224 INTT, 13824 TPC, 16 micromegas sensitive surfaces; all 56 map entries land. Its
layers overlap by design (sPHENIX assigns its layers by radius instead).
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

import acts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acts_worker import build_tracking_geometry  # noqa: E402


def geometry_id(surface):
    """(volume, layer, approach, sensitive) of a surface's geometry identifier."""
    gid = surface.geometryId
    return gid.volume, gid.layer, gid.approach, gid.sensitive


def layer_table(surfaces):
    """Per sensitive layer (volume, layer): surface count, radius of the surface centres, the layer's r and z extent
    from ACTS's ProtoLayer (no envelope), the sensors' thickness and the radial overlap with the next layer (mm)."""
    gctx = acts.GeometryContext.dangerouslyDefaultConstruct()
    axis_r, axis_z = acts.AxisDirection.AxisR, acts.AxisDirection.AxisZ
    layers = collections.defaultdict(list)
    for surface in surfaces:
        vol, lay, _, sen = geometry_id(surface)
        if sen:
            layers[(vol, lay)].append(surface)
    rows = []
    for (vol, lay), members in sorted(layers.items()):
        proto = acts.ProtoLayer(gctx, members)
        centre_r = [float(np.hypot(*np.asarray(m.center(gctx))[:2])) for m in members]
        rows.append((vol, lay, len(members), min(centre_r), max(centre_r), proto.min(axis_r, False), proto.max(axis_r, False),
                     proto.min(axis_z, False), proto.max(axis_z, False), float(np.mean([m.thickness for m in members]))))
    print(f"  {'volume':>6} {'layer':>5} {'sensors':>7} {'centre r':>17} {'extent r':>17} {'extent z':>19} "
          f"{'thick':>6} {'overlap':>8}   (mm; overlap = this r max - next r min, same volume, overlapping z)")
    for i, (vol, lay, n, cr0, cr1, r0, r1, z0, z1, thick) in enumerate(rows):
        nxt = next((r for r in rows[i + 1:] if r[0] == vol and r[7] < z1 and r[8] > z0 and r[5] > r0), None)
        overlap = f"{r1 - nxt[5]:8.3f}" if nxt is not None else f"{'':8s}"
        print(f"  {vol:6d} {lay:5d} {n:7d} {cr0:8.2f}-{cr1:8.2f} {r0:8.2f}-{r1:8.2f} {z0:9.1f}-{z1:9.1f} {thick:6.2f} {overlap}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", required=True, help="TGeo ROOT file")
    parser.add_argument("--geometry_config", required=True, help="ACTS TGeo JSON configuration")
    parser.add_argument("--material_map", default=None, help="ACTS JSON material map; none = geometry only")
    parser.add_argument("--layers", action="store_true", help="print each sensitive layer's extent and overlap")
    args = parser.parse_args()

    t0 = time.perf_counter()
    _, geometry = build_tracking_geometry(args.geometry, args.geometry_config, args.material_map)
    print(f"geometry built in {time.perf_counter() - t0:.1f} s from {args.geometry}")

    surfaces = []
    geometry.visitSurfaces(surfaces.append, False)
    by_volume, with_material, ids_with_material = collections.Counter(), collections.Counter(), set()
    for surface in surfaces:
        vol, lay, app, sen = geometry_id(surface)
        kind = "sensitive" if sen else ("approach" if app else "layer/other")
        by_volume[(vol, kind)] += 1
        if surface.surfaceMaterial is not None:
            with_material[(vol, kind)] += 1
            ids_with_material.add((vol, lay, app, sen))
    print(f"{len(surfaces)} surfaces visited (sensitive and passive)")
    print(f"  {'volume':>6} {'kind':12} {'surfaces':>9} {'with material':>14}")
    for key in sorted(by_volume):
        print(f"  {key[0]:6d} {key[1]:12} {by_volume[key]:9d} {with_material[key]:14d}")

    if args.layers:
        layer_table(surfaces)

    if args.material_map:
        with open(args.material_map) as f:
            entries = json.load(f)["Surfaces"]["entries"]
        keys = ("volume", "layer", "approach", "sensitive")
        unmatched = [e for e in entries if tuple(e.get(k, 0) for k in keys) not in ids_with_material]
        print(f"material map: {len(entries)} surface entries, {len(entries) - len(unmatched)} landed on a surface, "
              f"{len(unmatched)} found none")
        for e in unmatched[:20]:
            print("  no surface for", {k: e.get(k) for k in keys})
    return 0


if __name__ == "__main__":
    sys.exit(main())
