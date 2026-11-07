#!/usr/bin/env python3
"""
TEMPORARY (delete after use): why ACTS's AMVF has the better PV tail on the same events. Run like compare_vertexing:

    python scripts/acts/tmp_tail_check.py --csv <per-point csv> --detector <detector yaml> [--max_events 500]

1. How large the d0-z0 correlation of the ACTS track covariance is (the part of the "full covariance" VertexHead's
   metric leaves out).
2. VertexHead seeded like ACTS: first annealing weights from compatibility with ACTS's vertex (the seeding difference).
PV only, finder tracks, on the events where ACTS finds a vertex.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_vertexing as cv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--max_events", type=int, default=500)
    args = parser.parse_args()

    head = cv.vh.VertexHead.from_detector(args.detector, find_secondary=False).eval()
    events = cv.load_events(args.csv, args.max_events, "finder")
    with torch.no_grad():
        outs = [head(*inputs) for inputs, *_ in events]
    acts_pv, _ = cv.acts_vertices(head, outs)

    # 1. d0-z0 correlation of the tracks VertexHead fits
    rho = []
    for o in outs:
        c, keep = o["track_covariance"][0].double(), (o["track_weight"][0] > 0)
        rho.append((c[keep, 0, 1] / (c[keep, 0, 0] * c[keep, 1, 1]).sqrt()).numpy())
    rho = np.abs(np.concatenate(rho))
    rho = rho[np.isfinite(rho)]
    print(f"\n|corr(d0, z0)| of {len(rho)} fitted tracks: median {np.median(rho):.3f}, 90% {np.percentile(rho, 90):.3f}, "
          f"max {rho.max():.3f}")

    # 2. VertexHead seeded like ACTS: the first annealing weights from each track's compatibility with ACTS's vertex
    #    (cm) instead of with the all-track fit; on the events where ACTS has one
    fit = head._fit
    seeded = np.full_like(acts_pv, np.nan)
    with torch.no_grad():
        for e, (inputs, *_) in enumerate(events):
            if np.isfinite(acts_pv[e]).all():
                calls = []

                def seeded_fit(track_position, track_direction, track_weight, params, *a, **k):
                    result = fit(track_position, track_direction, track_weight, params, *a, **k)
                    if not calls:            # the adaptive fit's first fit: replaced by the seed
                        seed = torch.tensor(acts_pv[e][None], dtype=track_position.dtype)
                        result = {**result, "vertex_estimate": seed, "track_chi2": cv.vh._track_compatibility(
                            seed, track_position, track_direction, params["track_sigma"])}
                    calls.append(1)
                    return result
                head._fit = seeded_fit
                seeded[e] = head(*inputs)["vertex_estimate"][0].double().numpy()
    head._fit = fit

    on_acts = np.isfinite(acts_pv).all(axis=1)
    vertices = np.array([o["vertex_estimate"][0].double().numpy() for o in outs])
    print(f"\n{'PV, finder tracks, ACTS events':32s} {'events':>6s} {'median':>8s} {'68%':>8s} {'90%':>8s} {'<1cm':>6s}")
    for name, v in (("VertexHead", np.where(on_acts[:, None], vertices, np.nan)), ("ACTS AMVF", acts_pv),
                    ("VertexHead seeded at ACTS", seeded)):
        n, m, p68, p90, within = cv.pv_stats(v, events)[0]
        print(f"{name:32s} {n:6d} {m:8.4f} {p68:8.4f} {p90:8.4f} {within:6.1%}")
    moved = np.linalg.norm(seeded - vertices, axis=1)[on_acts]
    print(f"VertexHead's PV moved by the ACTS start: median {np.median(moved):.4f} cm, 90% {np.percentile(moved, 90):.4f}, "
          f"> 0.5 cm in {(moved > 0.5).sum()} events")


if __name__ == "__main__":
    main()
