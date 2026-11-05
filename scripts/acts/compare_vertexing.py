#!/usr/bin/env python3
"""
compare_vertexing.py: the regular check of VertexHead for one detector, against truth, on real events (the
per-point CSV of eval_vertex_reconstruction.py --save_csv), PV only and with SV finding. The detector file
(scripts/acts/detectors/<name>/<name>.yaml) gives the configured setup; the variants switch one thing each, so
the tables show what every ingredient is worth:

    own track fit             the detector's settings without ACTS
    detector settings         the detector file as it is
    no material map           ... ACTS without the material map (if one is set)
    energy loss on / off      ... with the detector's energy-loss setting flipped (if a material map is set)
    error scale x<s>          ... with track_error_scale = s (per --error_scales)
    loose SV cuts             ... with loose SV-finding cuts (--loose_sv; many seeds per event)

Tracks (--tracks): the track finder's (pred_assignment), and/or truth tracks (hits grouped by their truth
particle, seg_target), which separates vertexing errors from track-finding errors.

PV table: truth-error quantiles (cm), events within 1 cm, the median PV chi2/ndf (about 1 if the track errors are
right), tracks ACTS dropped, time per event (after the first, which starts the ACTS worker).

SV table (runs with SV finding), against the truth IDs (a track is a Geant secondary if the majority truth ID of
its hits is < 0): SV candidates, the share of the usable secondaries among them and their purity; SVs found, the
share of their tracks that are secondaries, SVs made only of secondaries. With a CSV that has each particle's
truth production vertex (vtx_x/y/z, written since 2026-11-05 with the reconstructed PV as reco_vtx_*), also SVs
whose tracks share one true origin displaced from the true PV (all within --origin_tolerance), and the SV residual:
the median distance of each SV to the median true origin of its tracks.

    python scripts/acts/compare_vertexing.py --csv vertex_per_point_data.csv \\
        --detector scripts/acts/detectors/sphenix/sphenix.yaml [--tracks both] [--loose_sv] [--max_events 500]
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
    Per event: (class_probs, mask_probs, points, padding_mask) as VertexHead takes them, the true vertex, and per
    track slot whether it is a Geant secondary and its particle's truth production vertex (NaN without one).
    tracks: "finder" (pred_assignment) or "truth" (seg_target).
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    ix = {h: i for i, h in enumerate(header)}
    has_origin = "reco_vtx_x" in ix          # vtx_* are truth origins only in the newer CSV format
    column = "pred_assignment" if tracks == "finder" else "seg_target"
    by_event = {}
    for row in rows:
        by_event.setdefault(int(row[ix["batch_idx"]]), []).append(row)
    events = []
    for e in sorted(by_event)[:max_events]:
        R = by_event[e]
        truth = np.array([float(R[0][ix[f"test_truth_vtx_{a}"]] or "nan") for a in "xyz"]) \
            if "test_truth_vtx_x" in ix else np.full(3, np.nan)
        label = np.array([int(float(r[ix[column]])) if r[ix[column]] != "" else -999999 for r in R])
        keep = (label != -999999) & ((label != -1) if tracks == "finder" else True)
        if not keep.any():
            continue
        _, slot = np.unique(label[keep], return_inverse=True)
        n_slots = slot.max() + 1
        mask = torch.zeros(1, len(R), n_slots)
        mask[0, torch.tensor(np.flatnonzero(keep)), torch.tensor(slot)] = 0.9
        class_probs = torch.zeros(1, n_slots, 2)
        class_probs[..., 1] = 0.8
        points = torch.tensor([[float(r[ix[k]]) for k in ("E_norm", "eta_norm", "phi_norm", "r_norm")] for r in R])[None]
        # truth per track slot: secondary (majority truth ID < 0) and that particle's production vertex
        truth_id = np.array([int(float(r[ix["seg_target"]])) if r[ix["seg_target"]] != "" else 0 for r in R])[keep]
        kept = [r for r, k in zip(R, keep) if k]
        secondary, origin = np.zeros(n_slots, dtype=bool), np.full((n_slots, 3), np.nan)
        for t in range(n_slots):
            ids, counts = np.unique(truth_id[slot == t], return_counts=True)
            secondary[t] = ids[counts.argmax()] < 0
            if has_origin:
                row = kept[int(np.flatnonzero((slot == t) & (truth_id == ids[counts.argmax()]))[0])]
                origin[t] = [float(row[ix[f"vtx_{a}"]] or "nan") for a in "xyz"]
        events.append(((class_probs, mask, points, torch.ones(1, len(R), dtype=torch.bool)), truth, secondary, origin))
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


def sv_truth(outs, events, origin_tolerance):
    """The SV table's numbers for one run with SV finding."""
    n_sec = n_cand = n_cand_sec = n_sv = sv_tracks = sv_tracks_sec = sv_all_sec = sv_true = 0
    residual = []
    for o, (_, truth, sec, origin) in zip(outs, events):
        usable, cand = o["track_quality_mask"][0].numpy(), o["displaced_mask"][0].numpy()
        n_sec += int((sec & usable).sum()); n_cand += int(cand.sum()); n_cand_sec += int((cand & sec).sum())
        for k in np.flatnonzero(o["sv_is_valid"][0].numpy()):
            m = o["sv_track_mask"][0, k].numpy()
            n_sv += 1; sv_tracks += int(m.sum()); sv_tracks_sec += int((m & sec).sum()); sv_all_sec += int(sec[m].all())
            o_k = origin[m]
            if np.isfinite(o_k).all():
                true_sv = np.median(o_k, axis=0)
                residual.append(np.linalg.norm(o["sv_estimate"][0, k].double().numpy() - true_sv))
                shared = np.linalg.norm(o_k - true_sv, axis=1).max() <= origin_tolerance
                sv_true += int(shared and np.linalg.norm(true_sv - truth) > origin_tolerance)
    return {"candidates": n_cand, "secondaries found": f"{n_cand_sec}/{n_sec}",
            "eff": n_cand_sec / max(n_sec, 1), "purity": n_cand_sec / max(n_cand, 1), "SVs": n_sv,
            "secondary tracks": f"{sv_tracks_sec}/{sv_tracks}", "all-secondary": sv_all_sec,
            "true SVs": sv_true if residual else None, "residual": np.median(residual) if residual else None}


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

    pv_rows, sv_rows = [], []
    for tracks in (("finder", "truth") if args.tracks == "both" else (args.tracks,)):
        events = load_events(args.csv, args.max_events, tracks)
        n_truth = sum(bool(np.isfinite(e[1]).all()) for e in events)
        print(f"{tracks} tracks: {len(events)} events ({n_truth} with a true vertex) from {args.csv}; detector "
              f"{args.detector}", flush=True)
        for mode, mode_kw in (("PV only", {"find_secondary": False}), ("with SVs", {"find_secondary": True})):
            for name, kw in variants(args.detector, args.error_scales, args.loose_sv):
                if name == "loose SV cuts" and mode == "PV only":
                    continue
                head = vh.VertexHead.from_detector(args.detector, **mode_kw, **kw).eval()
                outs, times = [], []
                with torch.no_grad():
                    for inputs, *_ in events:
                        t0 = time.perf_counter()
                        outs.append(head(*inputs))
                        times.append(time.perf_counter() - t0)
                err = np.array([np.linalg.norm(o["vertex_estimate"][0].double().numpy() - e[1]) for o, e in zip(outs, events)])
                err = err[np.isfinite(err)]
                # ndf = 2 sum(w) - 3 (ACTS: tracks count by their annealing weight)
                ndf = [2 * float((o["track_weight"] > 0).double().mul(o["track_vertex_weight"].double()).sum()) - 3
                       for o in outs]
                chi2 = np.array([o["chi_square"].item() / d for o, d in zip(outs, ndf) if d > 0 and o["fit_is_valid"].all()])
                pv_rows.append((tracks, mode, name, len(err),
                                *(np.percentile(err, q) if len(err) else np.nan for q in (50, 68, 90)),
                                (err < 1.0).mean() if len(err) else np.nan, np.median(chi2) if len(chi2) else np.nan,
                                sum(int(o["acts_failed"].sum()) for o in outs if "acts_failed" in o),
                                np.mean(times[1:] if len(times) > 1 else times) * 1e3))
                if mode == "with SVs":
                    sv_rows.append((tracks, name, sv_truth(outs, events, args.origin_tolerance)))
                print(f"  done: {tracks} tracks / {mode} / {name}", flush=True)

    print(f"\n{'tracks':7s} {'mode':9s} {'variant':22s} {'events':>6s} {'median':>8s} {'68%':>8s} {'90%':>8s} {'<1cm':>6s} "
          f"{'chi2/ndf':>8s} {'dropped':>7s} {'ms/evt':>7s}   (PV truth error, cm)")
    for tracks, mode, name, n, m, p68, p90, within, chi2, dropped, ms in pv_rows:
        print(f"{tracks:7s} {mode:9s} {name:22s} {n:6d} {m:8.4f} {p68:8.4f} {p90:8.4f} {within:6.1%} "
              f"{'' if np.isnan(chi2) else f'{chi2:8.2f}':>8s} {dropped:7d} {ms:7.1f}")
    print(f"\n{'tracks':7s} {'variant':22s} {'cands':>6s} {'sec. found':>11s} {'eff':>5s} {'pure':>5s} {'SVs':>5s} "
          f"{'sec. tracks':>12s} {'all-sec':>7s} {'true SVs':>8s} {'residual':>8s}   (SVs against truth; residual cm)")
    for tracks, name, s in sv_rows:
        true_svs = "" if s["true SVs"] is None else str(s["true SVs"])
        residual = "" if s["residual"] is None else f"{s['residual']:.3f}"
        print(f"{tracks:7s} {name:22s} {s['candidates']:6d} {s['secondaries found']:>11s} {s['eff']:5.0%} "
              f"{s['purity']:5.0%} {s['SVs']:5d} {s['secondary tracks']:>12s} {s['all-secondary']:7d} {true_svs:>8s} "
              f"{residual:>8s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
