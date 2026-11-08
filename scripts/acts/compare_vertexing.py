#!/usr/bin/env python3
"""
compare_vertexing.py: the primary-vertex reconstructions on the per-point CSV of eval_vertex_reconstruction.py, side by
side against the truth PV:

    python scripts/acts/compare_vertexing.py --csv <per-point csv> --detector <detector yaml> \
        [--tracks finder|truth|both] [--max_events 0] [--out <dir>] [--range 1.0] [--sv_range 2.0]

- VertexHead: VertexHead with the detector file's settings on the track finder's tracks (pred_assignment) or truth
  tracks (seg_target), every track with class probability 1.
- ACTS AMVF: ACTS's own vertex finder as shipped (acts_vertexing.py: AMVF, Gaussian seeder; its first vertex) on the
  same tracks with the same errors (x track_error_scale); needs the detector file's ACTS image.
- VertexHead, AMVF events: VertexHead on the events where AMVF found a vertex.

Writes to --out (default: compare/ beside the CSV's folder), named after the CSV (<name>_per_point_data.csv),
<name>_reco_vertices.csv (one row per event and track set: the truth PV, the eval's own PV and each reconstruction's
PV, cm), <name>_residuals.png (PV residual reco - true in x, y, z and 3D, one row per track set; values beyond --range
go in the end bins) and <name>_secondary.png (SVs per event: VertexHead's, the true ones and the true ones it could
find, those with >= 2 candidate tracks; and the SV residual against the truth, within --sv_range), and prints the PV
error table and an SV summary.

SV definitions (truth from the CSV; the same for both track sets):
- a track's truth origin: the production vertex (vtx_x/y/z) of its majority truth particle (seg_target);
- a true SV: a truth origin more than --origin_tolerance from the true PV shared by >= 2 tracks of the set (so it
  counts only tracks the set has: with finder tracks, reconstructed ones); its radius: from the beam line;
- candidate track: VertexHead's displaced_mask (its SV candidate cuts);
- findable: a true SV with >= 2 candidate tracks (the like-for-like truth for VertexHead's SVs);
- found: a VertexHead SV whose tracks (weight >= 0.5) include >= 2 of the true SV's tracks; a VertexHead SV is
  matched when its tracks include >= 2 tracks of one true SV; its residual: against its tracks' median truth origin.
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
import vertexing as vh  # noqa: E402

PARAMS = ("d0", "z0", "phi", "theta", "qop")
TRACK_COLUMNS = ["trackId", *PARAMS, *(f"var_{a}" for a in PARAMS),
                 *(f"cov_{a}{b}" for a in PARAMS for b in PARAMS if a != b)]   # CsvTrackParameterReader's columns
COLUMNS = ("batch_idx", "E_norm", "eta_norm", "phi_norm", "r_norm", "pred_assignment", "seg_target", "reco_vtx_x",
           "reco_vtx_y", "reco_vtx_z", "test_truth_vtx_x", "test_truth_vtx_y", "test_truth_vtx_z", "vtx_x", "vtx_y", "vtx_z")
TRACK_KEYS = ("track_parameters", "track_covariance", "track_weight", "displaced_mask")
SV_TRACK_KEYS = ("sv_track_mask", "sv_track_weight")


def read_columns(path, max_events):
    """The CSV columns compare_vertexing uses (empty fields NaN) for the first max_events events (0: all), in ~64 MB
    blocks with numpy's parser."""
    with open(path) as f:
        header = f.readline().strip().split(",")
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            sys.exit(f"{path}: no column(s) {missing}; rerun the eval (eval_vertex_reconstruction.py)")
        use = [header.index(c) for c in COLUMNS]
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
    return {c: data[:, k] for k, c in enumerate(COLUMNS)}


def load_events(col, tracks, origin_tolerance):
    """Per event with at least one track: its batch_idx, a function giving (class_probs, mask_probs, points,
    padding_mask) as VertexHead takes them (built on demand), the true PV, the eval's PV (cm), per track slot its
    truth origin (S, 3) cm and its true SV (S,) (index, -1: none)."""
    label = np.where(np.isnan(col[tracks]), -999999, col[tracks]).astype(np.int64)
    truth_id = np.where(np.isnan(col["seg_target"]), 0, col["seg_target"]).astype(np.int64)
    origin_all = np.stack([col[f"vtx_{a}"] for a in "xyz"], axis=1)
    points = np.stack([col[k] for k in ("E_norm", "eta_norm", "phi_norm", "r_norm")], axis=1)
    true_vertex = np.stack([col[f"test_truth_vtx_{a}"] for a in "xyz"], axis=1)
    eval_vertex = np.stack([col[f"reco_vtx_{a}"] for a in "xyz"], axis=1)
    order = np.argsort(col["batch_idx"], kind="stable")
    _, starts = np.unique(col["batch_idx"][order], return_index=True)
    events = []
    for hits in np.split(order, starts[1:]):
        keep = (label[hits] != -999999) & ((label[hits] != -1) if tracks == "pred_assignment" else True)
        if not keep.any():
            continue
        _, slot = np.unique(label[hits][keep], return_inverse=True)
        # per track slot: its majority truth particle's production vertex (first such hit)
        ids_kept, origin_kept = truth_id[hits][keep], origin_all[hits][keep]
        origin = np.full((slot.max() + 1, 3), np.nan)
        for t in range(slot.max() + 1):
            ids, counts = np.unique(ids_kept[slot == t], return_counts=True)
            origin[t] = origin_kept[np.flatnonzero((slot == t) & (ids_kept == ids[counts.argmax()]))[0]]
        # true SVs: displaced origins (equal to 1e-4 cm) shared by >= 2 tracks
        displaced = np.isfinite(origin).all(axis=1) & (np.linalg.norm(origin - true_vertex[hits[0]], axis=1)
                                                       > origin_tolerance)
        true_sv = np.full(len(origin), -1)
        if displaced.any():
            _, inverse, counts = np.unique(np.round(origin[displaced] / 1e-4), axis=0, return_inverse=True,
                                           return_counts=True)
            inverse = inverse.reshape(-1)
            true_sv[displaced] = np.where(counts[inverse] >= 2, (np.cumsum(counts >= 2) - 1)[inverse], -1)

        def inputs(hits=hits, keep=keep, slot=slot):
            mask = torch.zeros(1, len(hits), slot.max() + 1)
            mask[0, torch.tensor(np.flatnonzero(keep)), torch.tensor(slot)] = 0.9
            class_probs = torch.zeros(1, slot.max() + 1, 2)
            class_probs[..., 1] = 1.0       # every track counts fully, as for ACTS
            return (class_probs, mask, torch.tensor(points[hits], dtype=torch.float32)[None],
                    torch.ones(1, len(hits), dtype=torch.bool))
        events.append((int(col["batch_idx"][hits[0]]), inputs, true_vertex[hits[0]], eval_vertex[hits[0]], origin,
                       true_sv))
    return events


def run_chunk(head, inputs):
    """VertexHead once over several events (padded to common hit and track counts, as the eval runs it), split
    back into one output per event with its own track count."""
    pad = torch.nn.functional.pad
    n_hits, n_tracks = max(i[1].shape[1] for i in inputs), max(i[0].shape[1] for i in inputs)
    out = head(torch.cat([pad(i[0], (0, 0, 0, n_tracks - i[0].shape[1])) for i in inputs]),
               torch.cat([pad(i[1], (0, n_tracks - i[1].shape[2], 0, n_hits - i[1].shape[1])) for i in inputs]),
               torch.cat([pad(i[2], (0, 0, 0, n_hits - i[2].shape[1])) for i in inputs]),
               torch.cat([pad(i[3], (0, n_hits - i[3].shape[1]), value=False) for i in inputs]))
    return [{k: (v[b:b + 1, :i[0].shape[1]] if k in TRACK_KEYS else v[b:b + 1, ..., :i[0].shape[1]]
                 if k in SV_TRACK_KEYS else v[b:b + 1]) for k, v in out.items()} for b, i in enumerate(inputs)]


def secondary_vertices(outs, events):
    """VertexHead's SVs against the true ones: per event the number found, true and findable (>= 2 candidate tracks)
    (E,); each SV's residual against its tracks' median truth origin (N, 3) cm; the true SVs found, the findable ones
    found and the matched SVs."""
    n_found, n_true, n_findable, residual, found, found_findable, matched = [], [], [], [], 0, 0, 0
    for o, e in zip(outs, events):
        valid = o["sv_is_valid"][0].numpy()
        masks = o["sv_track_mask"][0].numpy().astype(bool)
        candidate = o["displaced_mask"][0].numpy().astype(bool)
        origin, true_sv = e[4], e[5]
        n_found.append(int(valid.sum()))
        n_true.append(int(true_sv.max(initial=-1)) + 1)
        findable = [(candidate & (true_sv == t)).sum() >= 2 for t in range(n_true[-1])]
        n_findable.append(sum(findable))
        for k in np.flatnonzero(valid):
            tracks = origin[masks[k]]
            tracks = tracks[np.isfinite(tracks).all(axis=1)]
            if len(tracks):
                residual.append(o["sv_estimate"][0, k].double().numpy() - np.median(tracks, axis=0))
            ids = true_sv[masks[k]]
            matched += int(bool((ids >= 0).any()) and np.bincount(ids[ids >= 0]).max() >= 2)
        for t in range(n_true[-1]):
            hit = any((masks[k] & (true_sv == t)).sum() >= 2 for k in np.flatnonzero(valid))
            found += hit
            found_findable += hit and findable[t]
    return (np.array(n_found), np.array(n_true), np.array(n_findable), np.array(residual).reshape(-1, 3), found,
            found_findable, matched)


def acts_vertices(head, outs):
    """ACTS's own vertex finder (acts_vertexing.py) on each event's VertexHead input tracks (weight > 0), errors
    scaled by track_error_scale: the first vertex found per event (E, 3) cm (NaN without one) and the vertices per
    event."""
    with tempfile.TemporaryDirectory() as tmp:
        for e, o in enumerate(outs):
            params = o["track_parameters"][0].double().numpy().copy()
            params[:, :2] *= vh.MM_PER_CM                                         # d0, z0: cm -> mm
            cov = o["track_covariance"][0].double().numpy() * head.tracks.track_error_scale ** 2
            keep = (o["track_weight"][0] > 0).numpy() & np.isfinite(params[:, :5]).all(1) & np.isfinite(cov).all((1, 2))
            with open(os.path.join(tmp, f"event{e:09d}-tracks.csv"), "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")   # LF only: ACTS keeps a CR in the last column
                writer.writerow(TRACK_COLUMNS)
                for i, (p, c) in enumerate(zip(params[keep], cov[keep])):
                    writer.writerow([i, *p[:5], *np.diag(c)[:5], *(c[a, b] for a in range(5) for b in range(5) if a != b)])
        command = head.tracks.acts_worker_command()
        if command[:2] == ["singularity", "exec"]:
            command = [*command[:2], "--bind", tmp, *command[2:]]
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acts_vertexing.py")
        field = ["--field_map", head.tracks.field_map, "--field_map_kind", head.tracks.field_map_kind] if head.tracks.field_map else []
        beamspot = [str(v) for v in head.tracks.acts_worker_config()["reference"]]       # the perigee reference, mm
        subprocess.run([*command, script, tmp, "--bz_tesla", str(head.tracks.Bz), *field, "--beamspot", *beamspot], check=True)
        found = []
        for e in range(len(outs)):
            with open(os.path.join(tmp, f"event{e:09d}-vertices.csv"), newline="") as f:
                found.append([[float(r[k]) for k in "xyz"] for r in csv.DictReader(f)])
    first = np.array([v[0] if v else [np.nan] * 3 for v in found]).reshape(-1, 3) / vh.MM_PER_CM
    return first, np.array([len(v) for v in found])


def table_row(label, residual):
    """The PV table's row for residuals (E, 3) reco - true, cm (NaN: no vertex)."""
    residual = residual[np.isfinite(residual).all(axis=1)]
    if not len(residual):
        return f"{label} {0:6d}"
    err, bias = np.linalg.norm(residual, axis=1), np.median(residual, axis=0)
    sigma = 1.4826 * np.median(np.abs(residual - bias), axis=0)
    return (f"{label} {len(err):6d} " + " ".join(f"{np.percentile(err, q):7.4f}" for q in (50, 68, 90))
            + f" {(err < 1.0).mean():6.1%} " + " ".join(f"{v:7.4f}" for v in (*bias, *sigma)))


STYLE = {"ACTS AMVF": dict(histtype="stepfilled", alpha=0.35, color="tab:gray"),
         "truth": dict(histtype="stepfilled", alpha=0.35, color="tab:gray"),
         "VertexHead": dict(histtype="step", lw=1.6, color="tab:blue"),
         "VertexHead, AMVF events": dict(histtype="step", lw=1.2, ls="--", color="tab:orange")}


def residual_panels(row, tracks, runs, limit, what, unit):
    """Histograms of residuals (N, 3) reco - true in x, y, z and 3D on four axes, one histogram per run (values
    beyond limit, 3D 1.5 x limit, in the end bins)."""
    for k, (ax, name) in enumerate(zip(row, ("x", "y", "z", "3D"))):
        edges = np.linspace(0 if k == 3 else -limit, 1.5 * limit if k == 3 else limit, 61)
        for label, residual in runs.items():
            residual = residual[np.isfinite(residual).all(axis=1)]
            value = np.linalg.norm(residual, axis=1) if k == 3 else residual[:, k]
            if not len(value):
                continue
            sigma = 1.4826 * np.median(np.abs(value - np.median(value))) if k < 3 else np.median(value)
            ax.hist(np.clip(value, edges[0], edges[-1]), bins=edges, label=f"{label} ({len(value)}, "
                    f"{'sigma' if k < 3 else 'median'} {sigma:.3f})", **STYLE[label])
        ax.set_xlabel(f"{'|reco - true|' if k == 3 else f'reco - true, {name}'} (cm)")
        ax.set_ylabel(unit)
        ax.set_title(f"{tracks} tracks: {what} residual {name}")
        ax.legend(fontsize=7, loc="upper right")


def sv_counts(ax, tracks, n_found, n_true, n_findable):
    """Histogram of SVs per event: VertexHead's, the true ones and the findable true ones (>= 2 candidate tracks)."""
    edges = np.arange(max(n_found.max(initial=0), n_true.max(initial=0)) + 2) - 0.5
    ax.hist(n_true, bins=edges, label=f"truth, all ({n_true.sum()})", **STYLE["truth"])
    ax.hist(n_findable, bins=edges, label=f"truth, >= 2 candidate tracks ({n_findable.sum()})",
            **STYLE["VertexHead, AMVF events"])
    ax.hist(n_found, bins=edges, label=f"VertexHead ({n_found.sum()})", **STYLE["VertexHead"])
    ax.set_xlabel("SVs per event")
    ax.set_ylabel("events")
    ax.set_yscale("log")
    ax.set_title(f"{tracks} tracks: SVs per event")
    ax.legend(fontsize=7, loc="upper right")


def figure(path, rows, columns, draw):
    """A PNG of len(rows) rows of columns panels; draw(axes_row, row) fills each row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), columns, figsize=(4.5 * columns, 4.2 * len(rows)), squeeze=False)
    for axes_row, row in zip(axes, rows):
        draw(axes_row, row)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--detector", required=True, help="detector file (scripts/acts/detectors/<name>/<name>.yaml)")
    parser.add_argument("--tracks", choices=("finder", "truth", "both"), default="both")
    parser.add_argument("--max_events", type=int, default=0, help="the CSV's first events (0: all)")
    parser.add_argument("--out", help="output folder (default: compare/ beside the CSV's folder)")
    parser.add_argument("--range", type=float, default=1.0, help="plotted PV residual range, cm (3D: 1.5x)")
    parser.add_argument("--sv_range", type=float, default=2.0, help="plotted SV residual range, cm (3D: 1.5x)")
    parser.add_argument("--origin_tolerance", type=float, default=0.1, help="cm: a displaced truth origin (true SV)")
    parser.add_argument("--chunk_events", type=int, default=32, help="events per VertexHead call (as the eval)")
    args = parser.parse_args()
    out_dir = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.csv))), "compare")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(args.csv).replace("_per_point_data.csv", "").replace(".csv", "")
    reco_path, plot_path, sv_path = (os.path.join(out_dir, f"{name}_{k}")
                                     for k in ("reco_vertices.csv", "residuals.png", "secondary.png"))

    head = vh.VertexHead.from_detector(args.detector).eval()
    head.tracks.start_acts()       # boots while the CSV loads
    timing = dict.fromkeys(("reading the CSV", "VertexHead", "ACTS AMVF"), 0.0)
    t0 = time.perf_counter()
    col = read_columns(args.csv, args.max_events)
    timing["reading the CSV"] += time.perf_counter() - t0
    rows, table, panels, sv = [], [], {}, {}
    for tracks in (("finder", "truth") if args.tracks == "both" else (args.tracks,)):
        t0 = time.perf_counter()
        events = load_events(col, "pred_assignment" if tracks == "finder" else "seg_target", args.origin_tolerance)
        timing["reading the CSV"] += time.perf_counter() - t0
        truth = np.array([e[2] for e in events]).reshape(-1, 3)
        print(f"{tracks} tracks: {len(events)} events ({int(np.isfinite(truth).all(axis=1).sum())} with a true PV) "
              f"from {args.csv}; detector {args.detector}", flush=True)
        outs = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for k in range(0, len(events), args.chunk_events):
                outs += run_chunk(head, [e[1]() for e in events[k:k + args.chunk_events]])
        timing["VertexHead"] += time.perf_counter() - t0
        ours = np.array([o["vertex_estimate"][0].double().numpy() for o in outs]).reshape(-1, 3)
        runs = {"VertexHead": ours}
        acts, n_found = np.full_like(ours, np.nan), np.zeros(len(outs), dtype=int)
        if head.tracks.use_acts:
            t0 = time.perf_counter()
            acts, n_found = acts_vertices(head, outs)
            timing["ACTS AMVF"] += time.perf_counter() - t0
            runs.update({"ACTS AMVF": acts, "VertexHead, AMVF events": np.where(np.isfinite(acts), ours, np.nan)})
        sv[tracks] = secondary_vertices(outs, events)
        panels[tracks] = {k: v - truth for k, v in runs.items()}
        table += [table_row(f"{tracks:7s} {name:24s}", residual) for name, residual in panels[tracks].items()]
        for e, o, a, n in zip(events, outs, acts, n_found):
            rows.append([e[0], tracks, int((o["track_weight"][0] > 0).sum()), *e[2], *e[3],
                         *o["vertex_estimate"][0].double().tolist(), int(o["vertex_fallback_step"]),
                         int(o["vertex_n_tracks"]), int(o["sv_is_valid"].sum()), int(e[5].max(initial=-1)) + 1,
                         *a, int(n)])

    with open(reco_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["batch_idx", "tracks", "n_tracks", *(f"true_{a}" for a in "xyz"), *(f"eval_{a}" for a in "xyz"),
                         *(f"vh_{a}" for a in "xyz"), "vh_fallback_step", "vh_n_tracks", "vh_n_sv", "true_n_sv",
                         *(f"acts_{a}" for a in "xyz"),
                         "acts_n_vertices"])
        writer.writerows(rows)
    figure(plot_path, list(panels), 4, lambda axes, tracks: residual_panels(axes, tracks, panels[tracks], args.range,
                                                                            "PV", "events"))
    figure(sv_path, list(sv), 5, lambda axes, tracks: (
        sv_counts(axes[0], tracks, *sv[tracks][:3]),
        residual_panels(axes[1:], tracks, {"VertexHead": sv[tracks][3]}, args.sv_range, "SV", "SVs")))

    print(f"\n{'tracks':7s} {'vertexing':24s} {'events':>6s} {'median':>7s} {'68%':>7s} {'90%':>7s} {'<1cm':>6s} "
          f"{'bias x':>7s} {'bias y':>7s} {'bias z':>7s} {'sig x':>7s} {'sig y':>7s} {'sig z':>7s}   (PV residual "
          "reco - true, cm: 3D error quantiles; bias = median, sig = 1.4826 MAD)", *table, sep="\n")
    print()
    for tracks, (n, t, f, r, found, found_findable, matched) in sv.items():
        error = np.linalg.norm(r, axis=1)
        print(f"{tracks:7s} SVs: VertexHead {n.sum()} ({matched} matched to a true SV); true {t.sum()}, with >= 2 "
              f"candidate tracks {f.sum()}; found {found} of all ({found / max(t.sum(), 1):.1%}), {found_findable} of "
              f"those with >= 2 candidate tracks ({found_findable / max(f.sum(), 1):.1%}); residual to their tracks' "
              f"truth origin: median {np.median(error) if len(error) else float('nan'):.3f} cm")
    print(f"\nwrote {reco_path}, {plot_path} and {sv_path}")
    print("time: " + ", ".join(f"{name} {seconds:.1f} s" for name, seconds in timing.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
