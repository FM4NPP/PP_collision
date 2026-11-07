#!/usr/bin/env python3
"""
TEMPORARY (delete after use): where VertexHead's SV positions go wrong. Truth tracks (seg_target), so track
finding is out of the picture; run like compare_vertexing:

    python scripts/acts/tmp_sv_check.py --csv <per-point csv> --detector <detector yaml> [--max_events 500]

For each true SV (at least 2 usable tracks sharing a truth origin displaced from the PV, within --tolerance):
its displacement; the error of the straight-line pair seed (best pair's closest-approach midpoint, as VertexHead
seeds); of VertexHead's helix fit given exactly its tracks; and of the nearest SV VertexHead found.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_vertexing as cv  # noqa: E402


def quantiles(name, values, unit="cm"):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    text = f"median {np.median(v):.3f}, 68% {np.percentile(v, 68):.3f}, 90% {np.percentile(v, 90):.3f} {unit}" if len(v) else "-"
    print(f"  {name:44s} {len(v):4d}  {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--max_events", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=0.1, help="cm: tracks share an origin within this")
    args = parser.parse_args()

    head = cv.vh.VertexHead.from_detector(args.detector, find_secondary=True).eval()
    captured = {}
    find = head._find_secondary_vertices

    def capture(pv_fit, track_position, track_direction, track_weight, params):
        captured.update(position=track_position, direction=track_direction, params=params)
        return find(pv_fit, track_position, track_direction, track_weight, params)
    head._find_secondary_vertices = capture

    displacement, r_xy, seed_error, fit_error, found_error, n_tracks, primary, event_index = [], [], [], [], [], [], [], []
    n_events, beam_line = 0, []
    with torch.no_grad():
        for inputs, truth, secondary, origin, _ in cv.load_events(args.csv, args.max_events, "truth"):
            n_events += 1
            out = head(*inputs)
            usable = out["track_quality_mask"][0].numpy() & np.isfinite(origin).all(axis=1)
            displaced = usable & (np.linalg.norm(origin - truth, axis=1) > args.tolerance)
            left = list(np.flatnonzero(displaced))
            while left:                                     # groups of tracks sharing an origin
                i = left.pop(0)
                group = [i] + [j for j in left if np.linalg.norm(origin[j] - origin[i]) <= args.tolerance]
                left = [j for j in left if j not in group]
                if len(group) < 2:
                    continue
                true_sv = np.median(origin[group], axis=0)
                if np.hypot(*true_sv[:2]) < 0.1:             # a beam-line group: against the truth PV's own tracks
                    at_truth = int((usable & (np.linalg.norm(origin - truth, axis=1) <= args.tolerance)).sum())
                    pv = out["vertex_estimate"][0].double().numpy()
                    beam_line.append((at_truth, len(group), np.linalg.norm(pv - truth), np.linalg.norm(pv - true_sv)))
                primary.append(not secondary[group].any())
                event_index.append(n_events - 1)
                displacement.append(np.linalg.norm(true_sv - truth))
                r_xy.append(np.hypot(*true_sv[:2]))
                n_tracks.append(len(group))
                # straight-line seed: the closest pair's closest-approach midpoint (perigee lines)
                P, d = captured["position"][0, group].double(), captured["direction"][0, group].double()
                best = (np.inf, None)
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        dp, c = P[b] - P[a], (d[a] * d[b]).sum()
                        denom = max(1 - float(c) ** 2, 1e-12)
                        t_a = float((dp * d[a]).sum() - c * (dp * d[b]).sum()) / denom
                        t_b = float(c * (dp * d[a]).sum() - (dp * d[b]).sum()) / denom
                        p_a, p_b = P[a] + t_a * d[a], P[b] + t_b * d[b]
                        if float((p_a - p_b).norm()) < best[0]:
                            best = (float((p_a - p_b).norm()), ((p_a + p_b) / 2).numpy())
                seed_error.append(np.linalg.norm(best[1] - true_sv))
                # VertexHead's helix fit with exactly these tracks
                weight = torch.zeros_like(out["track_weight"])
                weight[0, group] = 1.0
                fit = head._fit(captured["position"], captured["direction"], weight.to(captured["position"].dtype),
                                captured["params"])
                fit_error.append(np.linalg.norm(fit["vertex_estimate"][0].double().numpy() - true_sv))
                # the nearest SV VertexHead found
                svs = out["sv_estimate"][0][out["sv_is_valid"][0]].double().numpy()
                found_error.append(np.linalg.norm(svs - true_sv, axis=1).min() if len(svs) else np.nan)

    displacement, found_error = np.array(displacement), np.array(found_error)
    print(f"\n{len(displacement)} true SVs (truth tracks, >= 2 usable tracks sharing a displaced origin); tracks per SV: "
          f"{np.bincount(n_tracks)[2:].tolist()} (2, 3, ...)")
    quantiles("displacement from the PV", displacement)
    quantiles("transverse radius", r_xy)
    quantiles("straight-line pair seed error", seed_error)
    quantiles("helix fit error, exactly its tracks", fit_error)
    quantiles("nearest found SV error (all)", found_error)
    print(f"  found within 1 cm: {(found_error < 1).sum()} of {len(found_error)}")
    for lo, hi in ((0, 1), (1, 3), (3, 10), (10, np.inf)):
        sel = (displacement >= lo) & (displacement < hi)
        if sel.any():
            print(f"  displacement {lo:>2}-{hi:<4} cm: {sel.sum():3d} SVs; seed error median "
                  f"{np.median(np.array(seed_error)[sel]):.3f}, fit error median {np.nanmedian(np.array(fit_error)[sel]):.3f}, "
                  f"found within 1 cm {(found_error[sel] < 1).sum()}")

    # beam-line vertices (other collisions?) against vertices displaced from the beam line (decays?)
    r_xy, primary, fit_error = np.array(r_xy), np.array(primary), np.array(fit_error)
    print(f"\n{n_events} events")
    for name, sel in (("on the beam line (r_xy < 0.1 cm)", r_xy < 0.1), ("off the beam line (r_xy >= 0.1 cm)", r_xy >= 0.1)):
        if not sel.any():
            continue
        per_event = np.bincount(np.array(event_index)[sel], minlength=n_events)
        print(f"  {name}: {sel.sum()} groups, all-primary tracks in {primary[sel].sum()}; per event "
              f"{np.bincount(per_event).tolist()} (0, 1, 2, ...)")
        quantiles("    displacement from the PV", displacement[sel])
        quantiles("    transverse radius", r_xy[sel])
        quantiles("    helix fit error, exactly its tracks", fit_error[sel])
        print(f"    found within 1 cm: {(found_error[sel] < 1).sum()}")

    if beam_line:
        at_truth, in_group, to_truth, to_group = np.array(beam_line, dtype=float).T
        print(f"\nevents with a beam-line group ({len(at_truth)}): usable tracks at the truth PV median {np.median(at_truth):.0f}, "
              f"in the group median {np.median(in_group):.0f}; group larger in {(in_group > at_truth).sum()}, "
              f"equal in {(in_group == at_truth).sum()}")
        print(f"  VertexHead's PV - truth PV median {np.nanmedian(to_truth):.3f} cm, - group median {np.nanmedian(to_group):.3f} cm; "
              f"closer to the group in {(to_group < to_truth).sum()} events")


if __name__ == "__main__":
    main()
