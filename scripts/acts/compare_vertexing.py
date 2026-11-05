#!/usr/bin/env python3
"""
compare_vertexing.py: VertexHead variants for one detector on the same real events (the per-point CSV of
eval_vertex_reconstruction.py --save_csv), PV only and with SV finding, against the true vertex. The detector
file (scripts/acts/detectors/<name>/<name>.yaml) gives the configured setup; the variants switch one thing
each, so the table shows what every ingredient is worth:

    own track fit             the detector's settings without ACTS
    detector settings         the detector file as it is
    no material map           ... ACTS without the material map (if one is set)
    energy loss on / off      ... with the detector's energy-loss setting flipped (if a material map is set)
    error scale x<s>          ... with track_error_scale = s (per --error_scales)

Reports truth-error quantiles (cm), events within 1 cm, the median PV chi2/ndf (about 1
if the track errors are right), tracks ACTS dropped and time per event (after the first, which starts the ACTS
worker).

    python scripts/acts/compare_vertexing.py --csv vertex_per_point_data.csv \\
        --detector scripts/acts/detectors/sphenix/sphenix.yaml [--error_scales 1 2.5] [--max_events 500]
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
    """Per event: (class_probs, mask_probs, points, padding_mask) as VertexHead takes them, and the true vertex."""
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


def variants(detector, error_scales):
    """(name, VertexHead overrides) for this detector."""
    base = vh.VertexHead.from_detector(detector)
    if base.use_acts:
        yield "own track fit", dict(use_acts=False)
    yield "detector settings", {}
    if base.use_acts and base.acts_material_map:
        yield "no material map", dict(acts_material_map=None)
        yield f"energy loss {'off' if base.acts_energy_loss else 'on'}", dict(acts_energy_loss=not base.acts_energy_loss)
    for scale in error_scales:
        if scale != base.track_error_scale:
            yield f"error scale x{scale:g}", dict(track_error_scale=scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--detector", required=True, help="detector file (scripts/acts/detectors/<name>/<name>.yaml)")
    parser.add_argument("--error_scales", type=float, nargs="*", default=[1.0, 2.5])
    parser.add_argument("--max_events", type=int, default=500)
    args = parser.parse_args()

    events = load_events(args.csv, args.max_events)
    n_truth = sum(bool(np.isfinite(t).all()) for _, t in events)
    print(f"{len(events)} events ({n_truth} with a true vertex) from {args.csv}; detector {args.detector}")
    rows = []
    for mode, mode_kw in (("PV only", {"find_secondary": False}), ("with SVs", {"find_secondary": True})):
        for name, kw in variants(args.detector, args.error_scales):
            head = vh.VertexHead.from_detector(args.detector, **mode_kw, **kw).eval()
            outs, times = [], []
            with torch.no_grad():
                for inputs, _ in events:
                    t0 = time.perf_counter()
                    outs.append(head(*inputs))
                    times.append(time.perf_counter() - t0)
            err = np.array([np.linalg.norm(o["vertex_estimate"][0].double().numpy() - t) for o, (_, t) in zip(outs, events)])
            err = err[np.isfinite(err)]
            # ndf = 2 sum(w) - 3 (ACTS: tracks count by their annealing weight)
            ndf = [2 * float((o["track_weight"] > 0).double().mul(o.get("track_vertex_weight", torch.ones(1)).double()).sum()) - 3
                   for o in outs]
            chi2 = np.array([o["chi_square"].item() / d for o, d in zip(outs, ndf) if d > 0 and o["fit_is_valid"].all()])
            rows.append((mode, name, len(err), *(np.percentile(err, q) if len(err) else np.nan for q in (50, 68, 90)),
                         (err < 1.0).mean() if len(err) else np.nan,
                         np.median(chi2) if len(chi2) else np.nan,
                         sum(int(o["acts_failed"].sum()) for o in outs if "acts_failed" in o),
                         np.mean(times[1:] if len(times) > 1 else times) * 1e3))
            print(f"  done: {mode} / {name}", flush=True)

    print(f"\n{'mode':9s} {'variant':22s} {'events':>6s} {'median':>8s} {'68%':>8s} {'90%':>8s} {'<1cm':>6s} "
          f"{'chi2/ndf':>8s} {'dropped':>7s} {'ms/evt':>7s}   (truth error, cm)")
    for mode, name, n, m, p68, p90, within, chi2, dropped, ms in rows:
        print(f"{mode:9s} {name:22s} {n:6d} {m:8.4f} {p68:8.4f} {p90:8.4f} {within:6.1%} "
              f"{'' if np.isnan(chi2) else f'{chi2:8.2f}':>8s} {dropped:7d} {ms:7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
