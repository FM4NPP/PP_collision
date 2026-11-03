#!/usr/bin/env python3
"""TEMPORARY TEST SCRIPT (2026-11-03): delete once the ACTS material/field setup is designed.

Prints the structure of the sPHENIX inputs for ACTS, so the material and field-map modes can be built
for them: a field map (ROOT), a material map (JSON or CBOR) and the TGeo geometry configuration (JSON).
Runs in the training environment (uses uproot for ROOT files). Any argument can be left out.

    python scripts/acts/inspect_acts_inputs.py --field FIELD.root --material MATERIAL.json --geometry GEOMETRY.json
    ... --tgeo GEOMETRY.root      (the TGeo detector geometry, if you have one)
"""
import argparse
import json
import os
import sys

import numpy as np


def show_json(path, depth=3, items=4):
    size = os.path.getsize(path)
    print(f"  size {size / 1e6:.2f} MB")
    if path.endswith(".cbor"):
        print("  CBOR (binary JSON): not decoded here")
        return
    with open(path) as f:
        data = json.load(f)

    def walk(node, level, prefix):
        if level > depth:
            return
        if isinstance(node, dict):
            print(f"{prefix}{{dict, {len(node)} keys}}: {list(node)[:12]}")
            for key in list(node)[:items]:
                print(f"{prefix}  [{key!r}]")
                walk(node[key], level + 1, prefix + "    ")
        elif isinstance(node, list):
            print(f"{prefix}[list, {len(node)} items]")
            for value in node[:min(items, 2)]:
                walk(value, level + 1, prefix + "    ")
        else:
            print(f"{prefix}{type(node).__name__}: {str(node)[:120]}")

    walk(data, 0, "  ")


def show_root(path):
    import uproot
    with uproot.open(path) as f:
        print(f"  keys: {f.keys()[:20]}")
        for key, cls in list(f.classnames().items())[:20]:
            print(f"  {key}: {cls}")
            if cls in ("TNtuple", "TTree", "TNtupleD", "ROOT::RNTuple"):
                tree = f[key]
                print(f"    entries {tree.num_entries}, branches {tree.keys()}")
                arrays = tree.arrays(library="np", entry_stop=5_000_000)
                for name, values in arrays.items():
                    values = np.asarray(values, dtype=np.float64)
                    print(f"    {name:4s}: min {values.min():.6g}, max {values.max():.6g}, unique {len(np.unique(values))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field", help="sPHENIX field map (ROOT)")
    parser.add_argument("--material", help="ACTS material map (JSON / CBOR)")
    parser.add_argument("--geometry", help="TGeo geometry configuration (JSON)")
    parser.add_argument("--tgeo", help="TGeo detector geometry (ROOT), if available")
    args = parser.parse_args()
    for label, path, show in (("field map", args.field, show_root), ("material map", args.material, show_json),
                              ("geometry config", args.geometry, show_json), ("TGeo geometry", args.tgeo, show_root)):
        if path:
            print(f"== {label}: {path}")
            try:
                show(path)
            except Exception as error:  # noqa: BLE001 - report and go on to the next file
                print(f"  could not read: {type(error).__name__}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
