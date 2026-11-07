#!/usr/bin/env python3
"""
compare_vertexing.py: VertexHead against ACTS's own vertex finder and truth, on the per-point CSV of
eval_vertex_reconstruction.py:

    python scripts/acts/compare_vertexing.py --csv <per-point csv> --detector <detector yaml> \
        [--tracks finder|truth|both] [--max_events 500]

VertexHead runs with the detector file's settings (PV and SVs). ACTS's finder as shipped (acts_vertexing.py: AMVF,
Gaussian seeder; its first vertex as the PV) runs on the same tracks with the same errors (x track_error_scale);
it needs the detector file's ACTS image. "VH on ACTS": VertexHead on the events where ACTS found a vertex; "VH 2+
tracks": its vertices with >= 2 tracks of annealing weight >= 0.5 (ACTS's good-vertex criterion). Tracks: the track
finder's (pred_assignment) or truth tracks (seg_target).
Tables: PV error |reco - true| quantiles; PV residual per axis (xy, z, bias, robust sigma); VertexHead's SVs against
truth (secondary = truth ID < 0; a true SV: its tracks share a displaced truth origin vtx_*).
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "train", "downstream"))
import vertexhead as vh  # noqa: E402

PARAMS = ("d0", "z0", "phi", "theta", "qop")
TRACK_COLUMNS = ["trackId", *PARAMS, *(f"var_{a}" for a in PARAMS),
                 *(f"cov_{a}{b}" for a in PARAMS for b in PARAMS if a != b)]   # CsvTrackParameterReader's columns


COLUMNS = ("batch_idx", "E_norm", "eta_norm", "phi_norm", "r_norm", "pred_assignment", "seg_target", "vtx_x", "vtx_y",
           "vtx_z", "test_truth_vtx_x", "test_truth_vtx_y", "test_truth_vtx_z", "test_truth_pv_index")
_COLUMN_CACHE = {}


def read_columns(path, max_events):
    """The CSV columns compare_vertexing uses (empty fields NaN) for the first max_events events, read once, in
    ~64 MB blocks with numpy's parser."""
    if (path, max_events) in _COLUMN_CACHE:
        return _COLUMN_CACHE[(path, max_events)]
    with open(path) as f:
        header = f.readline().strip().split(",")
        names = [c for c in COLUMNS if c in header]
        use = [header.index(c) for c in names]
        blocks, events = [], set()
        while True:
            lines = f.readlines(64 << 20)
            if not lines:
                break
            # numpy's parser has no empty fields: NaN for them (a field can be empty only between or after commas)
            text = "".join(lines).replace("\r\n", "\n").replace(",,", ",nan,").replace(",,", ",nan,").replace(",\n", ",nan\n")
            block = np.loadtxt(text.splitlines(), delimiter=",", usecols=use, ndmin=2)
            blocks.append(block)
            events.update(np.unique(block[:, 0]).tolist())
            if max_events and len(events) > max_events:
                break
    data = np.concatenate(blocks)
    first = np.unique(data[:, 0])[:max_events] if max_events else np.unique(data[:, 0])
    data = data[np.isin(data[:, 0], first)]
    _COLUMN_CACHE[(path, max_events)] = {c: data[:, k] for k, c in enumerate(names)}
    return _COLUMN_CACHE[(path, max_events)]


def load_events(path, max_events, tracks="finder"):
    """
    Per event: a function giving (class_probs, mask_probs, points, padding_mask) as VertexHead takes them (built on
    demand), the true vertex, per track slot whether it is a Geant secondary and its particle's truth production
    vertex, the number of primary vertices (test_truth_pv_index; NaN in older CSVs) and their truth positions
    (V, 3), ranked as get_vertex_label ranks them (None in older CSVs).
    """
    col = read_columns(path, max_events)
    nan_to = lambda name, value: np.where(np.isnan(col[name]), value, col[name]).astype(np.int64)  # noqa: E731
    label = nan_to("pred_assignment" if tracks == "finder" else "seg_target", -999999)
    truth_id = nan_to("seg_target", 0)
    points = np.stack([col[k] for k in ("E_norm", "eta_norm", "phi_norm", "r_norm")], axis=1)
    true_vertex = np.stack([col[f"test_truth_vtx_{a}"] for a in "xyz"], axis=1)
    origin_all = np.stack([col[f"vtx_{a}"] for a in "xyz"], axis=1)
    pv_index = col.get("test_truth_pv_index")
    # each event's rows: one stable sort, then slices
    order = np.argsort(col["batch_idx"], kind="stable")
    _, starts = np.unique(col["batch_idx"][order], return_index=True)
    events = []
    for hits in np.split(order, starts[1:]):
        keep = (label[hits] != -999999) & ((label[hits] != -1) if tracks == "finder" else True)
        if not keep.any():
            continue
        _, slot = np.unique(label[hits][keep], return_inverse=True)
        n_slots = slot.max() + 1
        # per track slot: secondary (majority truth ID < 0) and that particle's production vertex (first such hit)
        ids_kept, origin_kept = truth_id[hits][keep], origin_all[hits][keep]
        secondary, origin = np.zeros(n_slots, dtype=bool), np.full((n_slots, 3), np.nan)
        for t in range(n_slots):
            ids, counts = np.unique(ids_kept[slot == t], return_counts=True)
            secondary[t] = ids[counts.argmax()] < 0
            origin[t] = origin_kept[np.flatnonzero((slot == t) & (ids_kept == ids[counts.argmax()]))[0]]

        def inputs(hits=hits, keep=keep, slot=slot, n_slots=n_slots):
            mask = torch.zeros(1, len(hits), n_slots)
            mask[0, torch.tensor(np.flatnonzero(keep)), torch.tensor(slot)] = 0.9
            class_probs = torch.zeros(1, n_slots, 2)
            class_probs[..., 1] = 0.8
            return (class_probs, mask, torch.tensor(points[hits], dtype=torch.float32)[None],
                    torch.ones(1, len(hits), dtype=torch.bool))
        n_pv = np.nanmax(pv_index[hits], initial=-1) + 1 if pv_index is not None else np.nan
        truth_pvs = None if pv_index is None else np.array(
            [origin_all[hits][pv_index[hits] == k].mean(axis=0) for k in range(int(n_pv))]).reshape(-1, 3)
        events.append((inputs, true_vertex[hits[0]], secondary, origin, n_pv, truth_pvs))
    return events


TRACK_KEYS = ("track_parameters", "track_covariance", "track_weight", "track_quality_mask", "track_sigma",
              "track_position", "track_direction", "track_vertex_weight", "track_chi2", "track_dca",
              "track_closest_approach_point", "track_primary_vertex", "displaced_mask", "displaced_weight", "acts_failed")


def run_chunk(head, inputs):
    """VertexHead once over several events (padded to common hit and track counts, as the eval runs it), split
    back into one output per event with its own track count."""
    pad = torch.nn.functional.pad
    n_hits, n_tracks = max(i[1].shape[1] for i in inputs), max(i[0].shape[1] for i in inputs)
    out = head(torch.cat([pad(i[0], (0, 0, 0, n_tracks - i[0].shape[1])) for i in inputs]),
               torch.cat([pad(i[1], (0, n_tracks - i[1].shape[2], 0, n_hits - i[1].shape[1])) for i in inputs]),
               torch.cat([pad(i[2], (0, 0, 0, n_hits - i[2].shape[1])) for i in inputs]),
               torch.cat([pad(i[3], (0, n_hits - i[3].shape[1]), value=False) for i in inputs]))
    split = []
    for b, i in enumerate(inputs):
        tracks = i[0].shape[1]
        one = {}
        for key, value in out.items():
            if not torch.is_tensor(value):
                one[key] = {k: v[b:b + 1] for k, v in value.items()}      # pv_original: per-event PV values only
                continue
            value = value[b:b + 1]
            if key in TRACK_KEYS:
                value = value[:, :tracks]
            elif key in ("sv_track_mask", "sv_track_dca"):
                value = value[..., :tracks]
            one[key] = value
        split.append(one)
    return split


def acts_vertices(head, outs):
    """ACTS's own vertex finder (acts_vertexing.py) on each event's VertexHead input tracks (weight > 0), errors
    scaled by track_error_scale: the first vertex found per event (E, 3) cm (NaN without one), vertices per event."""
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmp:      # the home is visible in the image
        for e, o in enumerate(outs):
            params = o["track_parameters"][0].double().numpy().copy()
            params[:, :2] /= head.track_parameter_length_scale                   # d0, z0: cm -> mm
            cov = o["track_covariance"][0].double().numpy() * head.track_error_scale ** 2
            keep = (o["track_weight"][0] > 0).numpy() & np.isfinite(params[:, :5]).all(1) & np.isfinite(cov).all((1, 2))
            with open(os.path.join(tmp, f"event{e:09d}-tracks.csv"), "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")   # LF only: ACTS keeps a CR in the last column
                writer.writerow(TRACK_COLUMNS)
                for i, (p, c) in enumerate(zip(params[keep], cov[keep])):
                    writer.writerow([i, *p[:5], *np.diag(c)[:5], *(c[a, b] for a in range(5) for b in range(5) if a != b)])
        command = head.acts_worker_command()
        if command[:2] == ["singularity", "exec"]:
            command = [*command[:2], "--bind", tmp, *command[2:]]
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acts_vertexing.py")
        field = ["--field_map", head.field_map, "--field_map_kind", head.field_map_kind] if head.field_map else []
        subprocess.run([*command, script, tmp, "--bz_tesla", str(head.Bz), *field], check=True)
        found = []
        for e in range(len(outs)):
            with open(os.path.join(tmp, f"event{e:09d}-vertices.csv"), newline="") as f:
                found.append([[float(r[k]) for k in "xyz"] for r in csv.DictReader(f)])
    first = np.array([v[0] if v else [np.nan] * 3 for v in found]) * head.track_parameter_length_scale
    return first, [len(v) for v in found]


def pv_stats(vertices, events):
    """The PV table's and the per-axis table's numbers for PVs (E, 3), cm."""
    residual = vertices - np.array([e[1] for e in events])
    residual = residual[np.isfinite(residual).all(axis=1)]          # (events, 3): reconstructed - true, cm
    if not len(residual):
        return (0, *[np.nan] * 4), [np.nan] * 10
    err, transverse, bias = np.linalg.norm(residual, axis=1), np.linalg.norm(residual[:, :2], axis=1), np.median(residual, axis=0)
    pv = (len(err), *(np.percentile(err, q) for q in (50, 68, 90)), (err < 1.0).mean())
    axes = (np.median(transverse), np.percentile(transverse, 68), np.median(np.abs(residual[:, 2])),
            np.percentile(np.abs(residual[:, 2]), 68), *bias, *(1.4826 * np.median(np.abs(residual - bias), axis=0)))
    return pv, axes


def sv_stats(outs, events, origin_tolerance):
    """The SV table's numbers for VertexHead's SVs."""
    n_sec = n_cand = n_cand_sec = n_sv = sv_tracks = sv_tracks_sec = sv_all_sec = sv_true = 0
    residual = []
    for o, (_, truth, sec, origin, *_) in zip(outs, events):
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
    parser.add_argument("--origin_tolerance", type=float, default=0.1, help="cm, for true (shared, displaced) origins")
    parser.add_argument("--max_events", type=int, default=500)
    parser.add_argument("--chunk_events", type=int, default=32, help="events per VertexHead call (as the eval)")
    args = parser.parse_args()

    head = vh.VertexHead.from_detector(args.detector, find_secondary=True).eval()
    head.start_acts()       # boots while the CSV loads
    pv_rows, axis_rows, sv_rows, notes = [], [], [], []
    timing = dict.fromkeys(("reading the CSV", "VertexHead", "ACTS AMVF"), 0.0)
    for tracks in (("finder", "truth") if args.tracks == "both" else (args.tracks,)):
        t0 = time.perf_counter()
        events = load_events(args.csv, args.max_events, tracks)
        timing["reading the CSV"] += time.perf_counter() - t0
        n_truth = sum(bool(np.isfinite(e[1]).all()) for e in events)
        print(f"{tracks} tracks: {len(events)} events ({n_truth} with a true vertex) from {args.csv}; detector "
              f"{args.detector}", flush=True)
        outs, times = [], []    # times: per chunk, per event
        with torch.no_grad():
            for k in range(0, len(events), args.chunk_events):
                chunk = [e[0]() for e in events[k:k + args.chunk_events]]
                t0 = time.perf_counter()
                outs += run_chunk(head, chunk)
                times.append((time.perf_counter() - t0) / len(chunk))
                timing["VertexHead"] += times[-1] * len(chunk)
        chi2 = [o["chi_square"].item() / float(o["vertex_ndf"]) for o in outs
                if float(o["vertex_ndf"]) > 0 and o["fit_is_valid"].all()]
        vertices = np.array([o["vertex_estimate"][0].double().numpy() for o in outs])
        runs = [("VertexHead", vertices, f"{np.median(chi2):8.2f}" if chi2 else "",
                 sum(int(o["acts_failed"].sum()) for o in outs if "acts_failed" in o), np.mean(times[1:] or times) * 1e3)]
        # VertexHead's vertices that are good by ACTS's criterion (>= 2 tracks with annealing weight >= 0.5)
        good = np.array([int(o["vertex_n_tracks"]) >= 2 for o in outs])
        runs.append(("VH 2+ tracks", np.where(good[:, None], vertices, np.nan), "", runs[0][3], None))
        if head.use_acts:
            t0 = time.perf_counter()
            acts_pv, n_found = acts_vertices(head, outs)
            timing["ACTS AMVF"] += time.perf_counter() - t0
            runs.append(("ACTS AMVF", acts_pv, "", runs[0][3], None))
            # VertexHead on the events where ACTS found a vertex: the like-for-like comparison
            runs.append(("VH on ACTS", np.where(np.isfinite(acts_pv), vertices, np.nan), "", runs[0][3], None))
            distance = np.linalg.norm(acts_pv - vertices, axis=1)
            notes.append(f"{tracks} tracks: ACTS AMVF found {np.mean(n_found):.2f} vertices per event, none in "
                         f"{n_found.count(0)}; its first vertex - VertexHead's PV: median {np.nanmedian(distance):.4f} cm, "
                         f"90% {np.nanpercentile(distance, 90):.4f} cm")
        n_pv = np.array([e[4] for e in events])
        if np.isfinite(n_pv).all():
            error = np.linalg.norm(vertices - np.array([e[1] for e in events]), axis=1)
            text = lambda sel: (f"median {np.nanmedian(error[sel]):.4f}, 90% {np.nanpercentile(error[sel], 90):.4f} cm"  # noqa: E731
                                if np.isfinite(error[sel]).any() else "-")
            notes.append(f"{tracks} tracks: {(n_pv >= 2).sum()} of {len(n_pv)} events have >= 2 primary vertices "
                         f"(pileup; most {int(n_pv.max())}); VertexHead's PV error with pileup {text(n_pv >= 2)}, "
                         f"without {text(n_pv < 2)}")
        if "primary_vertices" in outs[0] and np.isfinite(n_pv).all():
            # pileup vertices (truth rank >= 1) VertexHead found (a non-PV vertex within 2 mm in z), and its extra
            # vertices matching no truth vertex
            found = total = fakes = 0
            for o, e in zip(outs, events):
                z = o["primary_vertices"][0, :, 2].double().numpy()
                others = z[o["primary_vertex_valid"][0].numpy() & (np.arange(len(z)) != int(o["pv_index"]))]
                truth_z = e[5][:, 2]
                found += sum(bool((np.abs(others - tz) < 0.2).any()) for tz in truth_z[1:])
                total += len(truth_z[1:])
                fakes += sum(not (np.abs(truth_z - oz) < 0.2).any() for oz in others)
            notes.append(f"{tracks} tracks: pileup vertices found {found} of {total} (within 2 mm in z); extra "
                         f"vertices matching no truth vertex {fakes} in {len(events)} events")
        for name, run_vertices, run_chi2, dropped, ms in runs:
            label = f"{tracks:7s} {name:12s}"
            (n, m, p68, p90, within), axes = pv_stats(run_vertices, events)
            pv_rows.append(f"{label} {n:6d} {m:8.4f} {p68:8.4f} {p90:8.4f} {within:6.1%} {run_chi2:>8s} {dropped:7d} "
                           f"{'' if ms is None else f'{ms:7.1f}':>7s}")
            axis_rows.append(f"{label} " + " ".join(f"{x:7.4f}" for x in axes))
        sv_rows.append(f"{tracks:7s} {sv_stats(outs, events, args.origin_tolerance)}")

    head_row = f"{'tracks':7s} {'vertexing':12s}"
    print(f"\n{head_row} {'events':>6s} {'median':>8s} {'68%':>8s} {'90%':>8s} {'<1cm':>6s} {'chi2/ndf':>8s} {'dropped':>7s} "
          f"{'ms/evt':>7s}   (PV truth error, cm)", *pv_rows, sep="\n")
    print(f"\n{head_row} {'xy med':>7s} {'xy 68%':>7s} {'|z| med':>7s} {'|z| 68%':>7s} {'bias x':>7s} {'bias y':>7s} "
          f"{'bias z':>7s} {'sig x':>7s} {'sig y':>7s} {'sig z':>7s}   (PV residual reco - true, cm; bias = median, "
          "sig = 1.4826 MAD)", *axis_rows, sep="\n")
    print(f"\n{'tracks':7s} {'cands':>6s} {'sec. found':>11s} {'eff':>5s} {'pure':>5s} {'SVs':>5s} {'sec. tracks':>12s} "
          f"{'all-sec':>7s} {'true SVs':>8s} {'residual':>8s}   (VertexHead's SVs against truth; residual cm)",
          *sv_rows, sep="\n")
    if notes:
        print("", *notes, sep="\n")
    print("\ntime: " + ", ".join(f"{name} {seconds:.1f} s" for name, seconds in timing.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
