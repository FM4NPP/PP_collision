#!/usr/bin/env python3
"""
check_geometry.py: check a detector's ACTS geometry and material map, inside the ACTS image:

    singularity exec acts-base-48.2.0.sif acts-python scripts/acts/check_geometry.py \
        --geometry <tgeo.root> --geometry_config <config.json> [--material_map <material.json>]

Prints surfaces per ACTS volume, how many carry material and material-map entries that found no surface, with
ACTS's full build log.
"""
import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acts_worker import build_tracking_geometry  # noqa: E402


def geometry_id(surface):
    """(volume, layer, approach, sensitive) of a surface's geometry identifier."""
    gid = surface.geometryId
    return gid.volume, gid.layer, gid.approach, gid.sensitive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", required=True, help="TGeo ROOT file")
    parser.add_argument("--geometry_config", required=True, help="ACTS TGeo JSON configuration")
    parser.add_argument("--material_map", default=None, help="ACTS JSON material map; none = geometry only")
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
