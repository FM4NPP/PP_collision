#!/usr/bin/env python3
"""
TEMPORARY (delete after use): why ACTS's AMVF has the better transverse PV resolution on the same tracks. Run like
compare_vertexing (finder tracks):

    python scripts/acts/tmp_transverse_check.py --csv <per-point csv> --detector <detector yaml> [--max_events 3000]

1. Correlations in the ACTS track covariance VertexHead's metric leaves out: d0-phi, d0-z0, z0-theta.
2. On the events where ACTS finds a vertex, per axis: VertexHead, ACTS AMVF, and VertexHead seeded at ACTS's
   vertex (its first annealing weights from compatibility with it). Seeded VertexHead matching AMVF transversely
   -> the gap is association (which tracks), not the metric.
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
    parser.add_argument("--max_events", type=int, default=3000)
    args = parser.parse_args()

    head = cv.vh.VertexHead.from_detector(args.detector, find_secondary=True).eval()
    events = cv.load_events(args.csv, args.max_events, "finder")
    with torch.no_grad():
        outs = [head(*e[0]()) for e in events]
    acts_pv, _ = cv.acts_vertices(head, outs)

    # 1. correlations of the tracks VertexHead fits (ACTS perigee covariance: d0, z0, phi, theta, q/p)
    rho = {name: [] for name in ("d0-phi", "d0-z0", "z0-theta")}
    for o in outs:
        c, keep = o["track_covariance"][0].double(), (o["track_weight"][0] > 0)
        for name, (i, j) in (("d0-phi", (0, 2)), ("d0-z0", (0, 1)), ("z0-theta", (1, 3))):
            rho[name].append((c[keep, i, j] / (c[keep, i, i] * c[keep, j, j]).sqrt()).numpy())
    print()
    for name, values in rho.items():
        v = np.abs(np.concatenate(values))
        v = v[np.isfinite(v)]
        print(f"|corr({name})| of {len(v)} fitted tracks: median {np.median(v):.3f}, 90% {np.percentile(v, 90):.3f}")

    # 2. VertexHead seeded at ACTS's vertex (the PV fit's seed replaced; the SV refit keeps its own start)
    fit_adaptive = head._fit_adaptive
    seeded = np.full_like(acts_pv, np.nan)
    with torch.no_grad():
        for e, event in enumerate(events):
            if np.isfinite(acts_pv[e]).all():
                seed = torch.tensor(acts_pv[e][None])
                head._fit_adaptive = lambda *a, **k: fit_adaptive(*a, **(
                    {**k, "seed": seed.to(a[0].dtype)} if k.get("seed") is not None else k))
                seeded[e] = head(*event[0]())["vertex_estimate"][0].double().numpy()
    head._fit_adaptive = fit_adaptive

    on_acts = np.isfinite(acts_pv).all(axis=1)
    vertices = np.array([o["vertex_estimate"][0].double().numpy() for o in outs])
    print(f"\n{'ACTS events, finder tracks':28s} {'events':>6s} {'median':>7s} {'90%':>7s} {'<1cm':>6s} {'xy med':>7s} "
          f"{'|z| med':>7s} {'sig x':>7s} {'sig y':>7s} {'sig z':>7s}   (cm)")
    for name, v in (("VertexHead", np.where(on_acts[:, None], vertices, np.nan)), ("ACTS AMVF", acts_pv),
                    ("VertexHead seeded at ACTS", seeded)):
        (n, m, _, p90, within), axes = cv.pv_stats(v, events)
        print(f"{name:28s} {n:6d} {m:7.4f} {p90:7.4f} {within:6.1%} {axes[0]:7.4f} {axes[2]:7.4f} {axes[7]:7.4f} "
              f"{axes[8]:7.4f} {axes[9]:7.4f}")


if __name__ == "__main__":
    main()
