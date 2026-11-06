#!/usr/bin/env python3
"""
compare_vertexing.py: VertexHead variants for one detector against truth, on the per-point CSV of
eval_vertex_reconstruction.py, PV only and with SVs:

    python scripts/acts/compare_vertexing.py --csv <per-point csv> --detector <detector yaml> \
        [--tracks finder|truth|both] [--loose_sv] [--max_events 500]

Variants (one change each): own track fit, detector settings, no material map, energy loss flipped,
other error scales, loose SV cuts. Tracks: the track finder's (pred_assignment) or truth tracks (seg_target).
Tables: PV error |reco - true| quantiles; PV residual per axis (xy, z, bias, robust sigma); SVs against truth
(secondary = truth ID < 0; a true SV: its tracks share a displaced truth origin vtx_*).
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

LOOSE_SV = dict(sv_min_track_pv_chi2=1.0, sv_max_track_chi2=50.0, sv_min_displacement_significance=0.0,
                max_sv_chi2_per_dof=100.0, max_pair_doca=3.0)


def load_events(path, max_events, tracks="finder"):
    """
    Per event: (class_probs, mask_probs, points, padding_mask) as VertexHead takes them, the true vertex and, per
    track slot, whether it is a Geant secondary and its particle's truth production vertex.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    col = dict(zip(rows[0], map(np.array, zip(*rows[1:]))))
    num = lambda name, empty="nan": np.array([float(v or empty) for v in col[name]])  # noqa: E731
    event = num("batch_idx").astype(int)
    label = num("pred_assignment" if tracks == "finder" else "seg_target", "-999999").astype(int)
    truth_id = num("seg_target", "0").astype(int)
    points = np.stack([num(k) for k in ("E_norm", "eta_norm", "phi_norm", "r_norm")], axis=1)
    true_vertex = np.stack([num(f"test_truth_vtx_{a}") for a in "xyz"], axis=1)
    origin_all = np.stack([num(f"vtx_{a}") for a in "xyz"], axis=1)
    events = []
    for e in np.unique(event)[:max_events]:
        hits = np.flatnonzero(event == e)
        keep = (label[hits] != -999999) & ((label[hits] != -1) if tracks == "finder" else True)
        if not keep.any():
            continue
        _, slot = np.unique(label[hits][keep], return_inverse=True)
        n_slots = slot.max() + 1
        mask = torch.zeros(1, len(hits), n_slots)
        mask[0, torch.tensor(np.flatnonzero(keep)), torch.tensor(slot)] = 0.9
        class_probs = torch.zeros(1, n_slots, 2)
        class_probs[..., 1] = 0.8
        # per track slot: secondary (majority truth ID < 0) and that particle's production vertex (first such hit)
        ids_kept, origin_kept = truth_id[hits][keep], origin_all[hits][keep]
        secondary, origin = np.zeros(n_slots, dtype=bool), np.full((n_slots, 3), np.nan)
        for t in range(n_slots):
            ids, counts = np.unique(ids_kept[slot == t], return_counts=True)
            secondary[t] = ids[counts.argmax()] < 0
            origin[t] = origin_kept[np.flatnonzero((slot == t) & (ids_kept == ids[counts.argmax()]))[0]]
        inputs = (class_probs, mask, torch.tensor(points[hits], dtype=torch.float32)[None],
                  torch.ones(1, len(hits), dtype=torch.bool))
        events.append((inputs, true_vertex[hits[0]], secondary, origin))
    return events


def variants(detector, error_scales, loose_sv):
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
    if loose_sv:
        yield "loose SV cuts", dict(LOOSE_SV)


def pv_stats(outs, events, times):
    """The PV table's and the per-axis table's numbers for one run."""
    residual = np.array([o["vertex_estimate"][0].double().numpy() - e[1] for o, e in zip(outs, events)])
    residual = residual[np.isfinite(residual).all(axis=1)]          # (events, 3): reconstructed - true, cm
    if not len(residual):
        return (0, *[np.nan] * 5, 0, np.nan), [np.nan] * 10
    err, transverse, bias = np.linalg.norm(residual, axis=1), np.linalg.norm(residual[:, :2], axis=1), np.median(residual, axis=0)
    # ndf = 2 sum(w) - 3 (ACTS: tracks count by their annealing weight)
    ndf = [2 * float((o["track_weight"] > 0).double().mul(o["track_vertex_weight"].double()).sum()) - 3 for o in outs]
    chi2 = [o["chi_square"].item() / d for o, d in zip(outs, ndf) if d > 0 and o["fit_is_valid"].all()]
    pv = (len(err), *(np.percentile(err, q) for q in (50, 68, 90)), (err < 1.0).mean(), np.median(chi2) if chi2 else np.nan,
          sum(int(o["acts_failed"].sum()) for o in outs if "acts_failed" in o), np.mean(times[1:] or times) * 1e3)
    axes = (np.median(transverse), np.percentile(transverse, 68), np.median(np.abs(residual[:, 2])),
            np.percentile(np.abs(residual[:, 2]), 68), *bias, *(1.4826 * np.median(np.abs(residual - bias), axis=0)))
    return pv, axes


def sv_stats(outs, events, origin_tolerance):
    """The SV table's numbers for one run with SV finding."""
    n_sec = n_cand = n_cand_sec = n_sv = sv_tracks = sv_tracks_sec = sv_all_sec = sv_true = 0
    residual = []
    for o, (_, truth, sec, origin) in zip(outs, events):
        usable, cand = o["track_quality_mask"][0].numpy(), o["displaced_mask"][0].numpy()
        n_sec += int((sec & usable).sum()); n_cand += int(cand.sum()); n_cand_sec += int((cand & sec).sum())
        for k in np.flatnonzero(o["sv_is_valid"][0].numpy()):
            m = o["sv_track_mask"][0, k].numpy()
            n_sv += 1; sv_tracks += int(m.sum()); sv_tracks_sec += int((m & sec).sum()); sv_all_sec += int(sec[m].all())
            if np.isfinite(origin[m]).all():
                true_sv = np.median(origin[m], axis=0)
                residual.append(np.linalg.norm(o["sv_estimate"][0, k].double().numpy() - true_sv))
                shared = np.linalg.norm(origin[m] - true_sv, axis=1).max() <= origin_tolerance
                sv_true += int(shared and np.linalg.norm(true_sv - truth) > origin_tolerance)
    return (f"{n_cand:6d} {f'{n_cand_sec}/{n_sec}':>11s} {n_cand_sec / max(n_sec, 1):5.0%} {n_cand_sec / max(n_cand, 1):5.0%} "
            f"{n_sv:5d} {f'{sv_tracks_sec}/{sv_tracks}':>12s} {sv_all_sec:7d} "
            f"{str(sv_true) if residual else '':>8s} {f'{np.median(residual):.3f}' if residual else '':>8s}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--detector", required=True, help="detector file (scripts/acts/detectors/<name>/<name>.yaml)")
    parser.add_argument("--tracks", choices=("finder", "truth", "both"), default="both")
    parser.add_argument("--error_scales", type=float, nargs="*", default=[1.0, 2.5])
    parser.add_argument("--loose_sv", action="store_true", help="also a variant with loose SV cuts")
    parser.add_argument("--origin_tolerance", type=float, default=0.1, help="cm, for true (shared, displaced) origins")
    parser.add_argument("--max_events", type=int, default=500)
    args = parser.parse_args()

    pv_rows, axis_rows, sv_rows = [], [], []
    for tracks in (("finder", "truth") if args.tracks == "both" else (args.tracks,)):
        events = load_events(args.csv, args.max_events, tracks)
        n_truth = sum(bool(np.isfinite(e[1]).all()) for e in events)
        print(f"{tracks} tracks: {len(events)} events ({n_truth} with a true vertex) from {args.csv}; detector "
              f"{args.detector}", flush=True)
        for mode, find_secondary in (("PV only", False), ("with SVs", True)):
            for name, kw in variants(args.detector, args.error_scales, args.loose_sv):
                if name == "loose SV cuts" and not find_secondary:
                    continue
                head = vh.VertexHead.from_detector(args.detector, find_secondary=find_secondary, **kw).eval()
                outs, times = [], []
                with torch.no_grad():
                    for inputs, *_ in events:
                        t0 = time.perf_counter()
                        outs.append(head(*inputs))
                        times.append(time.perf_counter() - t0)
                label = f"{tracks:7s} {mode:9s} {name:22s}"
                pv, axes = pv_stats(outs, events, times)
                n, m, p68, p90, within, chi2, dropped, ms = pv
                pv_rows.append(f"{label} {n:6d} {m:8.4f} {p68:8.4f} {p90:8.4f} {within:6.1%} "
                               f"{'' if np.isnan(chi2) else f'{chi2:8.2f}':>8s} {dropped:7d} {ms:7.1f}")
                axis_rows.append(f"{label} " + " ".join(f"{x:7.4f}" for x in axes))
                if find_secondary:
                    sv_rows.append(f"{tracks:7s} {name:22s} {sv_stats(outs, events, args.origin_tolerance)}")
                print(f"  done: {tracks} tracks / {mode} / {name}", flush=True)

    head = f"{'tracks':7s} {'mode':9s} {'variant':22s}"
    print(f"\n{head} {'events':>6s} {'median':>8s} {'68%':>8s} {'90%':>8s} {'<1cm':>6s} {'chi2/ndf':>8s} {'dropped':>7s} "
          f"{'ms/evt':>7s}   (PV truth error, cm)", *pv_rows, sep="\n")
    print(f"\n{head} {'xy med':>7s} {'xy 68%':>7s} {'|z| med':>7s} {'|z| 68%':>7s} {'bias x':>7s} {'bias y':>7s} "
          f"{'bias z':>7s} {'sig x':>7s} {'sig y':>7s} {'sig z':>7s}   (PV residual reco - true, cm; bias = median, "
          "sig = 1.4826 MAD)", *axis_rows, sep="\n")
    print(f"\n{'tracks':7s} {'variant':22s} {'cands':>6s} {'sec. found':>11s} {'eff':>5s} {'pure':>5s} {'SVs':>5s} "
          f"{'sec. tracks':>12s} {'all-sec':>7s} {'true SVs':>8s} {'residual':>8s}   (SVs against truth; residual cm)",
          *sv_rows, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
