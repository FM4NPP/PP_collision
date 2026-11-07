#!/usr/bin/env python3
"""
TEMPORARY (delete after use): VertexHead's soft z window (seed_z_window, seed_z_window_softness) against no window and
ACTS AMVF, finder tracks; run like compare_vertexing:

    python scripts/acts/tmp_zwindow_sweep.py --csv <per-point csv> --detector <detector yaml> [--max_events 3000]

Per setting, on all events and on the events where ACTS finds a vertex: median, 90%, within 1 cm, xy and |z| medians
and robust sigma x, y, z (cm).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_vertexing as cv  # noqa: E402

SETTINGS = [("no window", None, None), ("1 mm, soft 0.2 mm", 0.1, 0.02), ("1 mm, soft 0.5 mm", 0.1, 0.05),
            ("2 mm, soft 0.5 mm", 0.2, 0.05), ("3 mm, soft 1 mm", 0.3, 0.1), ("5 mm, soft 1 mm", 0.5, 0.1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--max_events", type=int, default=3000)
    args = parser.parse_args()

    events = cv.load_events(args.csv, args.max_events, "finder")
    rows, acts_pv = [], None
    for name, window, softness in SETTINGS:
        overrides = {} if window is None else {"seed_z_window": window, "seed_z_window_softness": softness}
        head = cv.vh.VertexHead.from_detector(args.detector, find_secondary=True, **overrides).eval()
        with torch.no_grad():
            outs = [head(*e[0]()) for e in events]
        if acts_pv is None:           # ACTS once, on the tracks without a window (the window only reweights)
            acts_pv, _ = cv.acts_vertices(head, outs)
            rows.append(("ACTS AMVF", acts_pv))
        rows.append((name, np.array([o["vertex_estimate"][0].double().numpy() for o in outs])))
        print(f"  done: {name}", flush=True)

    on_acts = np.isfinite(acts_pv).all(axis=1)
    for scope, select in (("all events", None), ("ACTS events", on_acts)):
        print(f"\n{scope + ', finder tracks':24s} {'events':>6s} {'median':>7s} {'90%':>7s} {'<1cm':>6s} {'xy med':>7s} "
              f"{'|z| med':>7s} {'sig x':>7s} {'sig y':>7s} {'sig z':>7s}   (cm)")
        for name, v in rows:
            if select is None and name == "ACTS AMVF":
                continue
            v = v if select is None else np.where(select[:, None], v, np.nan)
            (n, m, _, p90, within), axes = cv.pv_stats(v, events)
            print(f"{name:24s} {n:6d} {m:7.4f} {p90:7.4f} {within:6.1%} {axes[0]:7.4f} {axes[2]:7.4f} {axes[7]:7.4f} "
                  f"{axes[8]:7.4f} {axes[9]:7.4f}")


if __name__ == "__main__":
    main()
