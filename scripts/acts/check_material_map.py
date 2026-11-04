#!/usr/bin/env python3
"""TEMPORARY TEST SCRIPT (2026-11-03): delete once the sPHENIX material map is known to apply.

Builds sPHENIX's ACTS tracking geometry the way sPHENIX does (TGeoDetector from the exported TGeo geometry and
the TGeo JSON configuration) with its JSON material map, and reports how the map lands: surfaces built per
ACTS volume, surfaces carrying material, and which material-map entries found no surface. Runs INSIDE the
material container (acts-material.def):

    singularity exec acts-material-48.2.0.sif acts-python scripts/acts/check_material_map.py \\
        --geometry sphenix_ana559_tgeo.root --config <tgeo-sphenix-mms json> --material <material map json>

ACTS Python names used here were read from the ACTS 48.2 source but not run before: where one is missing,
the script prints what the module offers instead.
"""
import argparse
import collections
import json
import sys
import time

import acts


def offered(module, words):
    return sorted(n for n in dir(module) if any(w.lower() in n.lower() for w in words))


def import_plugins():
    """acts.examples.tgeo and acts.json, whatever their exact import path in this build."""
    import importlib
    tgeo = json_plugin = None
    for name in ("acts.examples.tgeo", "acts.tgeo"):
        try:
            tgeo = importlib.import_module(name)
            break
        except ImportError:
            pass
    for name in ("acts.json", "acts.examples.json"):
        try:
            json_plugin = importlib.import_module(name)
            break
        except ImportError:
            pass
    if tgeo is None or json_plugin is None:
        import acts.examples
        print("MISSING:", "tgeo" if tgeo is None else "", "json" if json_plugin is None else "")
        print("acts offers:", offered(acts, ("json", "tgeo", "material", "examples")))
        print("acts.examples offers:", offered(acts.examples, ("tgeo", "json", "detector", "material")))
        sys.exit(2)
    return tgeo, json_plugin


def geometry_id(surface):
    """(volume, layer, approach, sensitive) of a surface's geometry identifier."""
    gid = surface.geometryId
    gid = gid() if callable(gid) else gid
    return tuple((lambda v: v() if callable(v) else v)(getattr(gid, k)) for k in ("volume", "layer", "approach", "sensitive"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", required=True, help="exported TGeo geometry (export_sphenix_geometry.C)")
    parser.add_argument("--config", required=True, help="TGeo JSON configuration (ACTSGEOMETRYCONFIG)")
    parser.add_argument("--material", default=None, help="JSON material map (ACTSMATERIALMAP); none = geometry only")
    args = parser.parse_args()

    tgeo, json_plugin = import_plugins()
    config = tgeo.TGeoDetector.Config()
    config.readJson(args.config)
    config.fileName = args.geometry
    decorator = None
    if args.material:
        t0 = time.perf_counter()
        decorator = json_plugin.JsonMaterialDecorator(
            rConfig=json_plugin.MaterialMapJsonConverter.Config(), jFileName=args.material,
            level=acts.logging.WARNING)
        config.materialDecorator = decorator
        print(f"material map read in {time.perf_counter() - t0:.1f} s: {args.material}")
    t0 = time.perf_counter()
    detector = tgeo.TGeoDetector(config)
    geometry = detector.trackingGeometry()
    print(f"geometry built in {time.perf_counter() - t0:.1f} s from {args.geometry}")

    surfaces = []
    try:
        geometry.visitSurfaces(lambda s: surfaces.append(s), False)
    except TypeError:
        geometry.visitSurfaces(lambda s: surfaces.append(s))
    by_volume = collections.Counter()
    with_material = collections.Counter()
    ids_with_material = set()
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

    if args.material:
        with open(args.material) as f:
            entries = json.load(f)["Surfaces"]["entries"]
        unmatched = [e for e in entries if (e.get("volume", 0), e.get("layer", 0), e.get("approach", 0),
                                            e.get("sensitive", 0)) not in ids_with_material]
        print(f"material map: {len(entries)} surface entries, {len(entries) - len(unmatched)} landed on a surface, "
              f"{len(unmatched)} found none")
        for e in unmatched[:20]:
            print("  no surface for", {k: e.get(k) for k in ("volume", "layer", "approach", "sensitive")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
