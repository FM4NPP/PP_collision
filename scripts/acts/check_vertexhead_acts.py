#!/usr/bin/env python3
"""TEMPORARY TEST SCRIPT (2026-11-03): delete once VertexHead(use_acts=True) is verified on the server.

VertexHead with its own perigee against VertexHead(use_acts=True), which has ACTS (in its container)
build the perigee parameters through the persistent worker. Real events from eval_vertex_reconstruction.py
--save_csv's per-point CSV, PV only and SV default. Reports vertex and perigee differences, tracks ACTS
dropped, error against the true vertex for both, and timing (the first ACTS call includes starting the
container).

    python scripts/acts/check_vertexhead_acts.py --csv vertex_per_point_data.csv --image acts-ref-48.2.0.sif
    ... --field_map uniform_xyz.txt          (a converted field map; a --uniform 1.4 map must match constant)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "train", "downstream"))
import vertexhead as vh  # noqa: E402


def load_events(path, max_events):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    ix = {h: i for i, h in enumerate(header)}
    by_event = {}
    for row in rows:
        by_event.setdefault(int(row[ix["batch_idx"]]), []).append(row)
    events = []
    for e in sorted(by_event)[:max_events]:
        R = by_event[e]
        truth = np.array([float(R[0][ix[f"test_truth_vtx_{a}"]] or "nan") for a in "xyz"]) \
            if "test_truth_vtx_x" in ix else np.full(3, np.nan)
        label = np.array([int(r[ix["pred_assignment"]]) for r in R])
        keep = label != -1
        if not keep.any():
            continue
        _, slot = np.unique(label[keep], return_inverse=True)
        mask = torch.zeros(1, len(R), slot.max() + 1)
        mask[0, torch.tensor(np.flatnonzero(keep)), torch.tensor(slot)] = 0.9
        class_probs = torch.zeros(1, slot.max() + 1, 2)
        class_probs[..., 1] = 0.8
        points = torch.tensor([[float(r[ix[k]]) for k in ("E_norm", "eta_norm", "phi_norm", "r_norm")] for r in R])[None]
        events.append(((class_probs, mask, points, torch.ones(1, len(R), dtype=torch.bool)), truth))
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="per-point CSV from eval_vertex_reconstruction.py --save_csv")
    parser.add_argument("--image", default=None, help="ACTS container image (acts-ref-48.2.0.sif)")
    parser.add_argument("--command", nargs="+", default=None, help="instead of --image: command running Python with ACTS")
    parser.add_argument("--field_map", default=None, help="text field map (convert_field_map.py)")
    parser.add_argument("--field_map_kind", default="xyz", choices=("xyz", "rz"))
    parser.add_argument("--max_events", type=int, default=500)
    args = parser.parse_args()

    events = load_events(args.csv, args.max_events)
    print(f"{len(events)} events from {args.csv}; ACTS via {args.command or args.image}"
          + (f", field map {args.field_map}" if args.field_map else ", constant field"))
    acts_kw = dict(use_acts=True, acts_image=args.image, acts_command=args.command, acts_field_map=args.field_map,
                   acts_field_map_kind=args.field_map_kind)
    for label, kw in (("PV only", {"find_secondary": False}), ("SV default", {})):
        results = {}
        for name, head in (("own", vh.VertexHead(**kw).eval()), ("ACTS", vh.VertexHead(**kw, **acts_kw).eval())):
            outs, times = [], []
            with torch.no_grad():
                for inputs, _ in events:
                    t0 = time.perf_counter()
                    outs.append(head(*inputs))
                    times.append(time.perf_counter() - t0)
            results[name] = (outs, times)
        own, acts = results["own"][0], results["ACTS"][0]
        dv = np.array([(a["vertex_estimate"] - b["vertex_estimate"]).norm().item() for a, b in zip(own, acts)
                       if a["fit_is_valid"].all() and b["fit_is_valid"].all()])
        usable = [a["track_quality_mask"] & b["track_quality_mask"] for a, b in zip(own, acts)]
        dp = np.array([(a["track_parameters"] - b["track_parameters"])[u][:, :5].abs().max(0).values.tolist()
                       for a, b, u in zip(own, acts, usable) if u.any()])
        fitted = sum(int(torch.isfinite(a["track_parameters"][..., 0]).sum()) for a in own)
        dropped = sum(int(b["acts_failed"].sum()) for b in acts)
        print(f"== {label}")
        if len(dv):
            print(f"  vertex |own - ACTS| cm: median {np.median(dv):.2e}, 99% {np.percentile(dv, 99):.2e}, max {dv.max():.2e}")
        if len(dp):
            print("  perigee |own - ACTS| max over usable tracks: " + ", ".join(
                f"{n} {v:.2e}" for n, v in zip(("d0 cm", "z0 cm", "phi", "theta", "q/p"), dp.max(0))))
        print(f"  ACTS dropped {dropped} of {fitted} fitted tracks; fit_is_valid differs in "
              f"{sum(int((a['fit_is_valid'] != b['fit_is_valid']).sum()) for a, b in zip(own, acts))} events")
        for name, (outs, times) in results.items():
            err = np.array([np.linalg.norm(o["vertex_estimate"][0].double().numpy() - t) for o, (_, t) in zip(outs, events)])
            err = err[np.isfinite(err)]
            timing = (f"first call {times[0]:.2f} s, then {np.mean(times[1:]) * 1e3:.1f} ms/event"
                      if name == "ACTS" and len(times) > 1 else f"{np.mean(times) * 1e3:.1f} ms/event")
            truth = (f"truth error median {np.median(err):.4f} cm, 68% {np.percentile(err, 68):.4f}, "
                     f"90% {np.percentile(err, 90):.4f} ({len(err)} events)") if len(err) else "no truth vertex"
            print(f"  {name:4s}: {truth}; {timing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
