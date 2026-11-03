#!/usr/bin/env python3
"""TEMPORARY TEST SCRIPT (2026-11-02): delete once the ACTS hooks are verified on the server.

Compare VertexHead's own perigee parameters with the ones ACTS builds (use_acts=True).

WHY THIS EXISTS

VertexHead (train/downstream/vertexhead.py) represents tracks as ACTS bound perigee parameters
(d0, z0, phi, theta, q/p, t) w.r.t. the beam spot. They come either from its own conversion of the
fitted helix (_perigee_from_circle) or, with use_acts=True, from ACTS: the fitted helix's state nearest
the innermost hit is propagated back to the beam spot (acts_perigee_parameters). The ACTS hooks were
written against the binding source of ACTS main and tested only against a stand-in module, because the
ACTS Python bindings were not available where they were written. This script checks them against the
real bindings. In a uniform field both routes describe the same helix, so they must agree.

WHAT IT DOES

1. API probe: every ACTS binding the hooks use, and the calls whose behaviour was not verified
   (Vector4, the keyword overloads of createCurvilinear, zero-length propagation onto the perigee).
2. Single helices: random perigees (exact maths) -> a state 30 cm downstream on the helix -> ACTS back
   to the perigee; and from the perigee state forwards (zero length). Failures by pT.
3. Events (--n_events, default 300, 25 tracks each): simulated helices from one vertex per event
   (uniform Bz, r-phi and z smearing, TPC-like layers), as normalized hits like the model's input, through
   VertexHead with use_acts False and True, PV only and SV default: vertex and perigee differences,
   quality masks, truth-vertex error of both, timing.
4. Optionally (--csv) the same on real events from eval_vertex_reconstruction.py --save_csv's per-point
   CSV (columns batch_idx, E_norm, eta_norm, phi_norm, r_norm, pred_assignment, test_truth_vtx_x/y/z).

Expected differences that are not failures: t (ACTS reports c*t from the start state back to the
perigee, < 0; VertexHead's own route reports 0), and z0 of unusable tracks far from the beam (ACTS can
stop at another turn of a looping helix). Only usable tracks are compared.

USAGE

    python scripts/check_acts_perigee.py
    python scripts/check_acts_perigee.py --n_events 500 --csv /path/to/vertex_per_point_data.csv

Paste the whole output back. Exit status 0 on agreement, 1 on divergence, 2 when ACTS is not importable.
"""
import argparse
import math
import os
import platform
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train", "downstream"))
import vertexhead as vh  # noqa: E402

PERIGEE_NAMES = ("d0", "z0", "phi", "theta", "q/p")


def perigee_difference(a, b):
    """Max |a - b| per perigee parameter (phi wrapped), over rows where both are finite; lengths as given."""
    d = (a[..., :5] - b[..., :5]).abs()
    d[..., 2] = torch.remainder(a[..., 2] - b[..., 2] + math.pi, 2 * math.pi).sub(math.pi).abs()
    both = torch.isfinite(a[..., :5]).all(-1) & torch.isfinite(b[..., :5]).all(-1)
    return {n: (d[both][:, k].max().item() if both.any() else float("nan")) for k, n in enumerate(PERIGEE_NAMES)}


def probe_api(acts, Bz):
    """Section 1: the bindings and the unverified calls; returns the number of failures."""
    print("== 1. ACTS API probe")
    u = acts.UnitConstants
    failures = 0

    def probe(name, fn):
        nonlocal failures
        try:
            result = fn()
            print(f"  OK    {name}" + (f": {result}" if result is not None else ""))
            return result
        except Exception as error:  # noqa: BLE001 - report anything the bindings raise
            failures += 1
            print(f"  FAIL  {name}: {type(error).__name__}: {error}".rstrip())
            return None

    for path in ("UnitConstants.mm", "UnitConstants.GeV", "UnitConstants.e", "UnitConstants.T", "Vector3", "Vector4",
                 "ConstantBField", "EigenStepper", "VoidNavigator", "EigenVoidPropagator", "PropagatorPlainOptions",
                 "GeometryContext.dangerouslyDefaultConstruct", "MagneticFieldContext", "Surface.createPerigee",
                 "BoundTrackParameters.createCurvilinear", "ParticleHypothesis.pion"):
        probe(f"acts.{path}", lambda path=path: (lambda o: repr(o) if isinstance(o, float) else None)(
            __import__("functools").reduce(getattr, path.split("."), acts)))
    field = probe("ConstantBField(Vector3(0, 0, Bz T))", lambda: acts.ConstantBField(acts.Vector3(0.0, 0.0, Bz * u.T)) and None) \
        or acts.ConstantBField(acts.Vector3(0.0, 0.0, Bz * u.T))
    propagator = acts.EigenVoidPropagator(acts.EigenStepper(field), acts.VoidNavigator())
    options = acts.PropagatorPlainOptions(acts.GeometryContext.dangerouslyDefaultConstruct(), acts.MagneticFieldContext())
    surface = acts.Surface.createPerigee(acts.Vector3(0.0, 0.0, 0.0))
    hypothesis = acts.ParticleHypothesis.pion
    phi, theta, qop = 0.3, 1.2, 1.0 / 1.5
    direction = (math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), math.cos(theta))
    probe("createCurvilinear(pos4=, dir=, qOverP=, cov=None, particleHypothesis=)", lambda: acts.BoundTrackParameters.createCurvilinear(
        pos4=acts.Vector4(10.0, 20.0, 5.0, 0.0), dir=acts.Vector3(*direction), qOverP=qop, cov=None,
        particleHypothesis=hypothesis) and None)
    probe("createCurvilinear(pos4=, phi=, theta=, qOverP=, cov=None, particleHypothesis=)", lambda: acts.BoundTrackParameters.createCurvilinear(
        pos4=acts.Vector4(10.0, 20.0, 5.0, 0.0), phi=phi, theta=theta, qOverP=qop, cov=None,
        particleHypothesis=hypothesis) and None)
    # on the surface: d0 = 1 mm, z0 = 5 mm -> should come back unchanged after a zero-length propagation
    start = (-1.0 * math.sin(phi), 1.0 * math.cos(phi), 5.0)
    probe("zero-length propagateToSurface onto the perigee (expect d0 1, z0 5, phi 0.3, theta 1.2)", lambda: [
        round(v, 6) for v in list(propagator.propagateToSurface(
            start=acts.BoundTrackParameters.createCurvilinear(pos4=acts.Vector4(*start, 0.0), phi=phi, theta=theta,
                                                              qOverP=qop, cov=None, particleHypothesis=hypothesis),
            target=surface, options=options).parameters)])
    probe("BoundTrackParameters.referenceSurface / .parameters on a result", lambda: (lambda r: f"{type(r.referenceSurface).__name__}, "
          f"{len(list(r.parameters))} parameters")(propagator.propagateToSurface(
              start=acts.BoundTrackParameters.createCurvilinear(pos4=acts.Vector4(*start, 0.0), phi=phi, theta=theta, qOverP=qop,
                                                                cov=None, particleHypothesis=hypothesis),
              target=surface, options=options)))
    return failures


def random_perigees(n, Bz, rng):
    """(n, 6) float64 perigee parameters in cm, pT log-uniform 0.1-10 GeV."""
    pT = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(math.log(0.1), math.log(10.0), generator=rng))
    theta = torch.empty(n, dtype=torch.float64).uniform_(0.4, 2.7, generator=rng)
    q = torch.where(torch.rand(n, generator=rng, dtype=torch.float64) < 0.5, -1.0, 1.0)
    return torch.stack([torch.empty(n, dtype=torch.float64).uniform_(-0.5, 0.5, generator=rng),
                        torch.empty(n, dtype=torch.float64).uniform_(-10, 10, generator=rng),
                        torch.empty(n, dtype=torch.float64).uniform_(-math.pi, math.pi, generator=rng),
                        theta, q * torch.sin(theta) / pT, torch.zeros(n, dtype=torch.float64)], dim=-1)


def single_helices(n, Bz, rng, beam):
    """Section 2: exact helices through acts_perigee_parameters, backward and forward."""
    print(f"== 2. Single helices ({n}, beam {beam} cm, Bz {Bz} T)")
    per = random_perigees(n, Bz, rng)[None]
    d0, z0, phi, theta, qop = per[..., :5].unbind(-1)
    p_pos = torch.stack([beam[0] - d0 * torch.sin(phi), beam[1] + d0 * torch.cos(phi), z0], -1)
    p_dir = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)], -1)
    # exact helix (dT/ds = (q/p) T x B): a point 30 cm of transverse path downstream; omega per cm of path
    w = qop * Bz * 0.000299792458 * 10.0
    s = 30.0 / torch.sin(theta)
    phis = phi - w * s
    st_pos = torch.stack([p_pos[..., 0] - torch.sin(theta) * (torch.sin(phis) - torch.sin(phi)) / w,
                          p_pos[..., 1] + torch.sin(theta) * (torch.cos(phis) - torch.cos(phi)) / w,
                          z0 + torch.cos(theta) * s], -1)
    st_dir = torch.stack([torch.sin(theta) * torch.cos(phis), torch.sin(theta) * torch.sin(phis), torch.cos(theta)], -1)
    reference = per.clone()
    reference[..., [0, 1]] *= 10.0                                          # ACTS mm
    pT = (torch.sin(theta) / qop.abs())[0]
    # a track that curls up before 30 cm of transverse path (2R < 30 cm) has no such state: skip those
    reaches = (2 * pT / (0.299792458 * Bz) * 100.0 > 30.0)
    worst = {}
    for name, (pos, direction, backward) in {"backward from 30 cm downstream": (st_pos, st_dir, True),
                                             "forward from the perigee": (p_pos, p_dir, False)}.items():
        t0 = time.perf_counter()
        out, _ = vh.acts_perigee_parameters(pos, direction, qop, Bz=Bz, beam_position=beam, backward=backward)
        elapsed = time.perf_counter() - t0
        keep = reaches if backward else torch.ones_like(reaches)
        failed = torch.isnan(out[0, :, 0]) & keep
        diff = perigee_difference(out[0][keep], reference[0][keep])
        worst[name] = diff
        bins = [(0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 10.1)]
        fails = ", ".join(f"pT {lo}-{hi}: {int((failed & (pT >= lo) & (pT < hi)).sum())}/{int((keep & (pT >= lo) & (pT < hi)).sum())}"
                          for lo, hi in bins)
        print(f"  {name:32s} max |ACTS - exact| " + ", ".join(f"{k} {v:.2e}" for k, v in diff.items())
              + f"  (d0, z0 in mm)  {elapsed / max(int(keep.sum()), 1) * 1e3:.2f} ms/track")
        print(f"  {'':32s} failed propagations {fails};  t range [{out[0, keep, 5].nan_to_num(0).min():.1f}, "
              f"{out[0, keep, 5].nan_to_num(0).max():.1f}] mm")
    return worst


def simulate_events(n_events, n_tracks, Bz, rng, sigma_rphi=0.034, sigma_z=0.10, n_layers=48):
    """Simulated events as VertexHead hit input: (class_probs, mask_probs, points, padding_mask, truth) per event."""
    norm = vh.TPC_NORMALIZATION
    layers = np.linspace(norm["r_lim"][0] + 0.5, norm["r_lim"][1] - 0.5, n_layers)
    gen = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=rng)))
    events = []
    for _ in range(n_events):
        v = np.array([gen.normal(0, 0.005), gen.normal(0, 0.005), gen.normal(0, 6.8)])
        hx, hy, hz, trk = [], [], [], []
        for t in range(n_tracks):
            pt = math.exp(gen.uniform(math.log(0.2), math.log(3.0)))
            eta, phi0, q = gen.uniform(-1.0, 1.0), gen.uniform(-math.pi, math.pi), gen.choice([-1, 1])
            R = pt / (0.299792458 * Bz) * 100.0
            h = -q * (1 if Bz >= 0 else -1)                         # +1 = anticlockwise
            cx, cy = v[0] - h * R * math.sin(phi0), v[1] + h * R * math.cos(phi0)
            a0 = math.atan2(v[1] - cy, v[0] - cx)
            turn = np.linspace(0, math.pi, 3000)
            X, Y = cx + R * np.cos(a0 + h * turn), cy + R * np.sin(a0 + h * turn)
            rr = np.hypot(X, Y)[: np.hypot(X, Y).argmax() + 1]
            ok = layers < rr[-1]
            ti = np.interp(layers[ok], rr, turn[: len(rr)])           # outgoing crossing of each layer
            x, y = cx + R * np.cos(a0 + h * ti), cy + R * np.sin(a0 + h * ti)
            z = v[2] + R * ti * math.sinh(eta) + gen.normal(0, sigma_z, ok.sum())
            d = gen.normal(0, sigma_rphi, ok.sum()) / layers[ok]
            x, y = x * np.cos(d) - y * np.sin(d), x * np.sin(d) + y * np.cos(d)
            inside = np.abs(z) < 105
            hx += list(x[inside]); hy += list(y[inside]); hz += list(z[inside]); trk += [t] * int(inside.sum())
        hx, hy, hz = map(np.array, (hx, hy, hz))
        r = np.hypot(hx, hy)
        points = np.stack([np.zeros_like(r),
                           (np.arcsinh(hz / r) - norm["eta_lim"][0]) / (norm["eta_lim"][1] - norm["eta_lim"][0]),
                           (np.arctan2(hy, hx) - norm["phi_lim"][0]) / (norm["phi_lim"][1] - norm["phi_lim"][0]),
                           (r - norm["r_lim"][0]) / (norm["r_lim"][1] - norm["r_lim"][0])], -1)
        mask = torch.zeros(1, len(r), n_tracks)
        mask[0, torch.arange(len(r)), torch.tensor(trk)] = 0.9
        class_probs = torch.zeros(1, n_tracks, 2)
        class_probs[..., 1] = 0.8
        events.append((class_probs, mask, torch.tensor(points, dtype=torch.float32)[None],
                       torch.ones(1, len(r), dtype=torch.bool), v))
    return events


def csv_events(path, max_events):
    """Real events from eval_vertex_reconstruction.py's per-point CSV (head clusters)."""
    import csv
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
        points = torch.tensor([[float(r[ix[k]]) for k in ("E_norm", "eta_norm", "phi_norm", "r_norm")] for r in R])[None]
        label = np.array([int(r[ix["pred_assignment"]]) for r in R])
        keep = label != -1
        if not keep.any():
            continue
        _, slot = np.unique(label[keep], return_inverse=True)
        mask = torch.zeros(1, len(R), slot.max() + 1)
        mask[0, torch.tensor(np.flatnonzero(keep)), torch.tensor(slot)] = 0.9
        class_probs = torch.zeros(1, slot.max() + 1, 2)
        class_probs[..., 1] = 0.8
        events.append((class_probs, mask, points, torch.ones(1, len(R), dtype=torch.bool), truth))
    return events


def compare_events(events, label, Bz, tolerances):
    """Sections 3 / 4: VertexHead use_acts False vs True; returns True on agreement."""
    print(f"== {label}: {len(events)} events")
    agree = True
    for mode, kw in (("PV only", {"find_secondary": False}), ("SV default", {})):
        heads = {"own": vh.VertexHead(Bz=Bz, **kw).eval(), "acts": vh.VertexHead(Bz=Bz, use_acts=True, **kw).eval()}
        dv, err, times, perigee, n_tracks = [], {"own": [], "acts": []}, {"own": 0.0, "acts": 0.0}, [], 0
        valid_mismatch = usable_mismatch = acts_failed = crashes = 0
        first_crash = None
        for class_probs, mask, points, padding, truth in events:
            out = {}
            for name, head in heads.items():
                t0 = time.perf_counter()
                try:
                    with torch.no_grad():
                        out[name] = head(class_probs, mask, points, padding)
                except Exception:  # noqa: BLE001 - count and show the first crash, keep going
                    crashes += 1
                    first_crash = first_crash or traceback.format_exc(limit=3)
                times[name] += time.perf_counter() - t0
            if len(out) < 2:
                continue
            a, b = out["own"], out["acts"]
            valid_mismatch += int((a["fit_is_valid"] != b["fit_is_valid"]).sum())
            usable_mismatch += int((a["track_quality_mask"] != b["track_quality_mask"]).sum())
            acts_failed += int((torch.isfinite(a["track_parameters"][..., 0]) & ~torch.isfinite(b["track_parameters"][..., 0])).sum())
            both = a["fit_is_valid"] & b["fit_is_valid"]
            if both.any():
                dv.append((a["vertex_estimate"] - b["vertex_estimate"])[both].norm(dim=-1).max().item())
            usable = a["track_quality_mask"] & b["track_quality_mask"]
            n_tracks += int(usable.sum())
            if usable.any():
                perigee.append(perigee_difference(a["track_parameters"][usable], b["track_parameters"][usable]))
            if np.isfinite(truth).all():
                for name in out:
                    err[name].append(np.linalg.norm(out[name]["vertex_estimate"][0].double().numpy() - truth))
        dv = np.array(dv)
        worst = {k: max((p[k] for p in perigee if p[k] == p[k]), default=float("nan")) for k in PERIGEE_NAMES}
        print(f"  {mode}: time own {times['own']:.1f} s, ACTS {times['acts']:.1f} s; crashes {crashes}")
        if first_crash:
            print("  first crash:\n" + "\n".join("    " + line for line in first_crash.rstrip().splitlines()))
        if len(dv):
            print(f"    vertex |own - ACTS| cm: median {np.median(dv):.2e}, 99% {np.percentile(dv, 99):.2e}, max {dv.max():.2e}")
        print(f"    usable tracks compared {n_tracks}; max |own - ACTS| " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items())
              + " (d0, z0 in cm)")
        print(f"    fit_is_valid differs in {valid_mismatch} events, track_quality_mask in {usable_mismatch} tracks, "
              f"ACTS propagation failed for {acts_failed} fitted tracks")
        for name in err:
            e = np.array(err[name])
            e = e[np.isfinite(e)]
            if len(e):
                print(f"    truth error {name:4s}: median {np.median(e):.4f} cm, 68% {np.percentile(e, 68):.4f}, "
                      f"90% {np.percentile(e, 90):.4f}, valid {len(e)}/{len(err[name])}")
        ok = (crashes == 0 and valid_mismatch == 0 and usable_mismatch == 0
              and (not len(dv) or dv.max() < tolerances["vertex"])
              and all(not (worst[k] >= tolerances[k]) for k in PERIGEE_NAMES))
        print(f"    -> {'AGREE' if ok else 'DIVERGE'}")
        agree &= ok
    return agree


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_events", type=int, default=300, help="simulated events (100-500 is plenty)")
    ap.add_argument("--n_tracks", type=int, default=25, help="tracks per simulated event")
    ap.add_argument("--n_helices", type=int, default=2000, help="single helices in section 2")
    ap.add_argument("--csv", default=None, help="per-point CSV from eval_vertex_reconstruction.py --save_csv")
    ap.add_argument("--max_csv_events", type=int, default=500)
    ap.add_argument("--Bz", type=float, default=1.4, help="Tesla")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vertex_tol", type=float, default=1e-3, help="cm, max vertex difference to agree")
    ap.add_argument("--length_tol", type=float, default=1e-3, help="cm, max d0 / z0 difference of usable tracks")
    ap.add_argument("--angle_tol", type=float, default=1e-4, help="rad, max phi / theta difference")
    ap.add_argument("--qop_tol", type=float, default=1e-4, help="e/GeV, max q/p difference")
    args = ap.parse_args()
    tolerances = {"vertex": args.vertex_tol, "d0": args.length_tol, "z0": args.length_tol, "phi": args.angle_tol,
                  "theta": args.angle_tol, "q/p": args.qop_tol}

    print(f"python {platform.python_version()} ({platform.platform()}), torch {torch.__version__}, numpy {np.__version__}")
    try:
        acts = vh._import_acts()
    except ImportError as error:
        print(f"SKIP: {error}")
        return 2
    version = next((str(getattr(acts, k)) for k in ("__version__", "version") if hasattr(acts, k)), "unknown")
    print(f"acts {version} from {getattr(acts, '__file__', '?')}")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    rng = torch.Generator().manual_seed(args.seed)

    api_failures = probe_api(acts, args.Bz)
    if api_failures:
        print(f"{api_failures} API probe(s) failed; sections 2-4 may crash where they use them")
    helix_worst = {}
    for beam in ((0.0, 0.0), (0.3, -0.2)):
        try:
            helix_worst.update({f"{beam} {k}": v for k, v in single_helices(args.n_helices, args.Bz, rng, beam).items()})
        except Exception:  # noqa: BLE001
            print("  CRASH\n" + traceback.format_exc(limit=4))
            helix_worst[f"{beam} crash"] = {k: float("inf") for k in PERIGEE_NAMES}
    helix_ok = all(not (v[k] >= (tolerances[k] * 10 if k in ("d0", "z0") else tolerances[k]))   # mm vs cm
                   for v in helix_worst.values() for k in PERIGEE_NAMES)

    agree = compare_events(simulate_events(args.n_events, args.n_tracks, args.Bz, rng),
                           f"3. Simulated events ({args.n_tracks} tracks, Bz {args.Bz} T)", args.Bz, tolerances)
    if args.csv:
        agree &= compare_events(csv_events(args.csv, args.max_csv_events), f"4. CSV events ({args.csv})", args.Bz, tolerances)
    ok = api_failures == 0 and helix_ok and agree
    print(f"== RESULT: {'AGREE' if ok else 'DIVERGE'} (API probes failed {api_failures}, single helices "
          f"{'ok' if helix_ok else 'off'}, events {'ok' if agree else 'off'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
