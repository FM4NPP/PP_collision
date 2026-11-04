#!/usr/bin/env python3
"""TEMPORARY TEST SCRIPT (2026-11-04): delete once the vertexing variants are decided.

Every VertexHead implementation on the same real events (eval_vertex_reconstruction.py --save_csv per-point
CSV), PV only and SV default, against the true vertex:

    own        VertexHead's own perigee (hit fit, constant Bz)
    ACTS       ACTS propagates the hit fit to the beam spot (constant field, or --field_map)
    ACTS geo   ... through sPHENIX's tracking geometry, no material (must equal ACTS)
    ACTS mat   ... with the material map: energy loss, and multiple scattering in the covariance
    ACTS mat-E ... material map, energy loss off (multiple scattering only)

each with vertex_weighting "hits" (weight = hit count x P(real)) and "covariance" (inverse variance of each
track's d0 / z0: the hit fit's, transported and widened by ACTS). Variants needing an option that is not
given are skipped. Reports truth-error quantiles, events within 1 cm, the median PV chi2/ndf (covariance
weighting: about 1 if the errors are right), tracks ACTS dropped and time per event (after the first, which
starts the worker).

    python scripts/acts/compare_vertexing.py --csv vertex_per_point_data.csv \\
        --image scripts/acts/acts-material-worker-48.2.0.sif --geometry sphenix_ana559_tgeo.root \\
        --geometry_config tgeo-sphenix-mms-actsv45.0.0.json --material_map sphenix-mm-acts-material-actsv45.0.0.json
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "train", "downstream"))
import vertexhead as vh  # noqa: E402
from check_vertexhead_acts import load_events  # noqa: E402


def variants(args):
    acts = dict(use_acts=True, acts_image=args.image, acts_command=args.command, acts_field_map=args.field_map,
                acts_field_map_kind=args.field_map_kind)
    geo = dict(acts, acts_geometry=args.geometry, acts_geometry_config=args.geometry_config)
    mat = dict(geo, acts_material_map=args.material_map)
    have_acts, have_geo, have_mat = bool(args.image or args.command), bool(args.geometry and args.geometry_config), bool(args.material_map)
    for name, kw, ok in (("own", {}, True), ("ACTS", acts, have_acts), ("ACTS geo", geo, have_acts and have_geo),
                         ("ACTS mat", mat, have_acts and have_geo and have_mat),
                         ("ACTS mat-E", dict(mat, acts_energy_loss=False), have_acts and have_geo and have_mat)):
        if ok:
            for weighting in ("hits", "covariance"):
                if name == "ACTS mat-E" and weighting == "hits":
                    continue        # no covariance: identical to ACTS geo
                yield f"{name} / {weighting}", dict(kw, vertex_weighting=weighting)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image", default=None, help="ACTS image (the material worker image runs every ACTS variant)")
    parser.add_argument("--command", nargs="+", default=None, help="instead of --image: command running Python with ACTS")
    parser.add_argument("--field_map", default=None)
    parser.add_argument("--field_map_kind", default="xyz", choices=("xyz", "rz"))
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--geometry_config", default=None)
    parser.add_argument("--material_map", default=None)
    parser.add_argument("--max_events", type=int, default=500)
    args = parser.parse_args()

    events = load_events(args.csv, args.max_events)
    n_truth = sum(bool(np.isfinite(t).all()) for _, t in events)
    print(f"{len(events)} events ({n_truth} with a true vertex) from {args.csv}")
    rows = []
    for mode, mode_kw in (("PV only", {"find_secondary": False}), ("SV default", {})):
        for name, kw in variants(args):
            head = vh.VertexHead(**mode_kw, **kw).eval()
            outs, times = [], []
            with torch.no_grad():
                for inputs, _ in events:
                    t0 = time.perf_counter()
                    outs.append(head(*inputs))
                    times.append(time.perf_counter() - t0)
            err = np.array([np.linalg.norm(o["vertex_estimate"][0].double().numpy() - t) for o, (_, t) in zip(outs, events)])
            err = err[np.isfinite(err)]
            ndf = [(2 * int((o["track_weight"] > 0).sum()) - 3) for o in outs]
            chi2 = np.array([o["chi_square"].item() / d for o, d in zip(outs, ndf) if d > 0 and o["fit_is_valid"].all()])
            dropped = sum(int(o["acts_failed"].sum()) for o in outs if "acts_failed" in o)
            rows.append((mode, name, len(err), *(np.percentile(err, q) if len(err) else np.nan for q in (50, 68, 90)),
                         (err < 1.0).mean() if len(err) else np.nan,
                         np.median(chi2) if len(chi2) and kw["vertex_weighting"] == "covariance" else np.nan, dropped,
                         np.mean(times[1:] if len(times) > 1 else times) * 1e3))
            print(f"  done: {mode} / {name}", flush=True)

    print(f"\n{'mode':10s} {'variant':24s} {'events':>6s} {'median':>8s} {'68%':>8s} {'90%':>8s} {'<1cm':>6s} "
          f"{'chi2/ndf':>8s} {'dropped':>7s} {'ms/evt':>7s}   (truth error, cm)")
    for mode, name, n, m, p68, p90, within, chi2, dropped, ms in rows:
        print(f"{mode:10s} {name:24s} {n:6d} {m:8.4f} {p68:8.4f} {p90:8.4f} {within:6.1%} "
              f"{'' if np.isnan(chi2) else f'{chi2:8.2f}':>8s} {dropped:7d} {ms:7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
