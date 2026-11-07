#!/usr/bin/env python3
"""
TEMPORARY diagnostic (delete after use): why a quarter of ACTS's perigee propagations fail (mostly "maximum number of
steps"). Runs VertexHead on the same eval CSV events with one ACTS setting changed at a time and prints, per
variant, the tracks ACTS lost, the PV accuracy, the propagation time and the worker's failure summary:

    python scripts/acts/diagnose_propagation.py --csv <per_point_data.csv> --detector scripts/configs/sphenix_<site>.yaml
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_vertexing as cv  # noqa: E402

vh = cv.vh
NO_GEOMETRY = {"acts_geometry": None, "acts_geometry_config": None, "acts_material_map": None}
VARIANTS = [("hit fit only (no ACTS)", {"use_acts": False}),
            ("current", {}),
            ("max_steps 10000", {"acts_max_steps": 10000}),
            ("no geometry (field map only)", NO_GEOMETRY),
            ("geometry, uniform field", {"field_map": None}),
            ("no geometry, uniform field", {**NO_GEOMETRY, "field_map": None})]

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--detector", required=True)
parser.add_argument("--max_events", type=int, default=300)
args = parser.parse_args()

events = cv.load_events(args.csv, args.max_events, "finder")
print(f"{len(events)} events from {args.csv}", flush=True)
hit_fit_usable = None
for name, overrides in VARIANTS:
    head = vh.VertexHead.from_detector(args.detector, find_secondary=False, **overrides).eval()
    for key in vh.ACTS_TIMING:
        vh.ACTS_TIMING[key] = type(vh.ACTS_TIMING[key])(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        outs = [o for k in range(0, len(events), 32) for o in cv.run_chunk(head, [e[0]() for e in events[k:k + 32]])]
    elapsed = time.perf_counter() - t0
    usable = sum(int(o["track_quality_mask"].sum()) for o in outs)
    lost = sum(int(o["acts_failed"].sum()) for o in outs if "acts_failed" in o)
    quality = [o["track_quality_mask"][0] for o in outs]
    if hit_fit_usable is None:
        hit_fit_usable = quality
    # tracks the hit fit rejects but ACTS's perigee passes (lost if only hit-fit-usable tracks went to ACTS)
    rescued = sum(int((q & ~h).sum()) for q, h in zip(quality, hit_fit_usable))
    (n, p50, p68, p90, within), _ = cv.pv_stats(np.array([o["vertex_estimate"][0].double().numpy() for o in outs]), events)
    print(f"\n== {name}: {usable} usable, ACTS lost {lost} hit-fit-usable and rescued {rescued} hit-fit-rejected; PV median {p50:.3f} / 68% {p68:.3f} / "
          f"90% {p90:.3f} cm, < 1 cm {within:.1%} ({n} events); propagation {vh.ACTS_TIMING['propagate_s']:.1f} s, "
          f"worker start {vh.ACTS_TIMING['start_s']:.1f} s, total {elapsed:.1f} s", flush=True)
    for worker in list(vh._ACTS_WORKERS.values()):      # its failure summary follows on stderr
        worker.close()
    vh._ACTS_WORKERS.clear()
    sys.stderr.flush()
