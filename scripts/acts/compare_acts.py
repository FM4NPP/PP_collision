"""
compare_acts.py: check the package's own track maths against ACTS, and mark each method's performance.
Runs in the training environment (torch); ACTS itself runs in its container (acts_reference.py).

    python scripts/acts/compare_acts.py make states.npz
    singularity exec acts-ref-48.2.0.sif acts-python scripts/acts/acts_reference.py states.npz acts_out.npz
    python scripts/acts/compare_acts.py compare states.npz acts_out.npz

make: exact helices from random perigee parameters (pT 0.1-20 GeV, uniform field), as three sets of
    track states for ACTS:
    exact, behind: on the helix 300 mm of transverse path downstream (less for tracks that curl up
        sooner), so the perigee is behind it, as for a state at the innermost hit;
    exact, on: on the perigee itself (a zero-length propagation);
    MomentumHead: TPC-like hits where each helix crosses the layers (smeared in r-phi and z), fitted by
        MomentumHead (train/downstream/momentumhead.py); its fitted state at the innermost hit.
compare: per set, ACTS and the torch maths (acts_port.propagate_to_perigee) against the exact perigee and
    against each other; MomentumHead's own fit quality; and the time per track of each method. Exit
    status 0 when ACTS and the torch maths agree within the tolerances on the exact sets.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "train", "downstream"))
import acts_port as ap  # noqa: E402

NAMES = ("d0 [mm]", "z0 [mm]", "phi", "theta", "q/p [e/GeV]", "t [mm]")
SETS = ("exact, behind", "exact, on", "MomentumHead")
CM = ap.UnitConstants.cm


def helix_points(start, phi0, theta, qop, bz, arc):
    """Exact helix (d(direction)/ds = (q/p) direction x B): positions and directions a transverse arc
    `arc` (mm) along from start (..., 3) with heading phi0."""
    path = arc / torch.sin(theta)
    omega = qop * bz
    phi = phi0 - omega * path
    position = torch.stack([start[..., 0] - torch.sin(theta) * (torch.sin(phi) - torch.sin(phi0)) / omega,
                            start[..., 1] + torch.sin(theta) * (torch.cos(phi) - torch.cos(phi0)) / omega,
                            start[..., 2] + torch.cos(theta) * path], dim=-1)
    direction = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi),
                             torch.cos(theta).expand_as(phi)], dim=-1)
    return position, direction


def momentumhead_states(truth, reference, args, gen):
    """Hits from each exact helix, MomentumHead's fit of them, and its state (mm) at the innermost hit."""
    import vertexhead as vh
    from momentumhead import MomentumHead

    bz = args.bz * ap.UnitConstants.T
    norm = vh.TPC_NORMALIZATION
    layers = np.linspace(norm["r_lim"][0] * CM + 5.0, norm["r_lim"][1] * CM - 5.0, args.n_layers)     # mm
    rng = np.random.default_rng(args.seed)
    position, _, qop, _ = ap.perigee_to_free(truth, reference)
    hits = []
    for i in range(len(truth)):
        # dense samples along the outgoing half turn; each layer's first crossing
        radius = float(torch.sin(truth[i, 3]) / (qop[i].abs() * bz))
        arc = torch.linspace(0.0, min(0.999 * math.pi * radius, 3000.0), 4000, dtype=torch.float64)
        xyz, _ = helix_points(position[i], truth[i, 2], truth[i, 3], qop[i], bz, arc)
        r = torch.hypot(xyz[:, 0], xyz[:, 1]).numpy()
        r, xyz = r[: r.argmax() + 1], xyz[: r.argmax() + 1].numpy()
        crossed = layers[(layers > r[0]) & (layers < r[-1])]
        x, y, z = (np.interp(crossed, r, xyz[:, k]) for k in range(3))
        turn = rng.normal(0.0, args.sigma_rphi, len(crossed)) / crossed
        x, y = x * np.cos(turn) - y * np.sin(turn), x * np.sin(turn) + y * np.cos(turn)
        z = z + rng.normal(0.0, args.sigma_z, len(crossed))
        hits.append(np.stack([x, y, z], -1)[np.abs(z) < args.z_max])

    head = MomentumHead(Bz=args.bz, noise_pt_threshold=0.0, valid_radius=1e9,
                        beam_position=(reference[0].item() / CM, reference[1].item() / CM)).to(args.device)
    state = {k: np.full((len(truth), n), np.nan) for k, n in (("position", 3), ("direction", 3))}
    state["qop"], state["pT"], state["charge"] = (np.full(len(truth), np.nan) for _ in range(3))
    seconds = 0.0
    for first in range(0, len(truth), args.tracks_per_event):
        tracks = range(first, min(first + args.tracks_per_event, len(truth)))
        xyz = np.concatenate([hits[t] for t in tracks]) / CM                                       # cm
        slot = torch.from_numpy(np.concatenate([np.full(len(hits[t]), j) for j, t in enumerate(tracks)]))
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        points = torch.tensor(np.stack([np.zeros_like(r),
                                        (np.arcsinh(xyz[:, 2] / r) - norm["eta_lim"][0]) / (norm["eta_lim"][1] - norm["eta_lim"][0]),
                                        (np.arctan2(xyz[:, 1], xyz[:, 0]) - norm["phi_lim"][0]) / (norm["phi_lim"][1] - norm["phi_lim"][0]),
                                        (r - norm["r_lim"][0]) / (norm["r_lim"][1] - norm["r_lim"][0])], -1),
                              dtype=torch.float32)[None].to(args.device)
        mask = torch.zeros(1, len(r), len(tracks), device=args.device)
        mask[0, torch.arange(len(r)), slot] = 0.9
        if first == 0:              # warm-up, untimed: CUDA start-up and first-call costs are not the fit's
            with torch.no_grad():
                head(mask, points, torch.ones(1, len(r), dtype=torch.bool, device=args.device))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = head(mask, points, torch.ones(1, len(r), dtype=torch.bool, device=args.device))
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        out = {k: v[0].double().cpu() for k, v in out.items() if torch.is_tensor(v)}

        # the fitted helix's point nearest the innermost hit, its tangent there, and q/p (as VertexHead did)
        centre, R_s = out["circle_centre"], out["R_s"]
        n = out["track_position"][:, :2].double() - centre
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        xy = centre + R_s.abs().unsqueeze(-1) * n
        z = vh._arc_length(xy[:, 0], xy[:, 1], centre[:, 0], centre[:, 1], R_s,
                           head.beam_position) * out["zslope"] + out["z0"]
        theta = torch.atan2(torch.ones_like(out["zslope"]), out["zslope"])
        t_xy = torch.sign(R_s).unsqueeze(-1) * torch.stack([n[:, 1], -n[:, 0]], dim=-1)
        q = torch.sign(R_s) * (1.0 if args.bz >= 0 else -1.0)
        rows = list(tracks)
        state["position"][rows] = (torch.cat([xy, z.unsqueeze(-1)], -1) * CM).numpy()
        state["direction"][rows] = torch.cat([torch.sin(theta).unsqueeze(-1) * t_xy, torch.cos(theta).unsqueeze(-1)], -1).numpy()
        state["qop"][rows] = (q * torch.sin(theta) / out["pT"]).numpy()
        state["pT"][rows], state["charge"][rows] = out["pT"].numpy(), q.numpy()
    ok = out_ok = np.isfinite(state["qop"])
    for k in ("position", "direction", "qop"):
        state[k][~ok] = np.nan
    print(f"MomentumHead: fitted {int(out_ok.sum())}/{len(truth)} tracks in {seconds:.2f} s on {args.device} "
          f"(mean hits per track {np.mean([len(h) for h in hits]):.1f})")
    return state, seconds


def make(args):
    gen = torch.Generator().manual_seed(args.seed)
    n = args.n
    uniform = lambda lo, hi: torch.empty(n, dtype=torch.float64).uniform_(lo, hi, generator=gen)  # noqa: E731
    pT = torch.exp(uniform(math.log(0.1), math.log(20.0)))
    theta = uniform(0.3, 2.8)
    q = torch.where(uniform(0, 1) < 0.5, -1.0, 1.0)
    truth = torch.stack([uniform(-5, 5), uniform(-100, 100), uniform(-math.pi, math.pi), theta,
                         q * torch.sin(theta) / pT, torch.zeros(n, dtype=torch.float64)], dim=-1)
    reference = torch.tensor(args.reference, dtype=torch.float64)
    bz = args.bz * ap.UnitConstants.T
    position, direction, qop, _ = ap.perigee_to_free(truth, reference)
    arc = torch.minimum(torch.full_like(pT, 300.0), 0.9 * math.pi * pT / bz)
    down, down_dir = helix_points(position, truth[:, 2], theta, qop, bz, arc)
    fit, fit_seconds = momentumhead_states(truth, reference, args, gen)

    # sets stacked: exact behind, exact on, MomentumHead; truth t = time at the perigee relative to the
    # state (t = 0 there): -path / beta behind it, 0 on it, unknown for the fit (not compared)
    behind_t = -arc / torch.sin(theta) * torch.sqrt(1.0 + (ap.PION_MASS * qop) ** 2)
    truths = [truth.clone(), truth.clone(), truth.clone()]
    truths[0][:, 5], truths[2][:, 5] = behind_t, float("nan")
    np.savez(args.states,
             position=np.concatenate([down.numpy(), position.numpy(), fit["position"]]),
             direction=np.concatenate([down_dir.numpy(), direction.numpy(), fit["direction"]]),
             qop=np.concatenate([qop.numpy(), qop.numpy(), fit["qop"]]),
             backward=np.concatenate([np.ones(n, bool), np.zeros(n, bool), np.ones(n, bool)]),
             kind=np.repeat(np.arange(3), n), truth=torch.cat(truths).numpy(), pT=np.tile(pT.numpy(), 3),
             fit_pT=fit["pT"], fit_charge=fit["charge"], fit_seconds=np.float64(fit_seconds),
             bz_tesla=np.float64(args.bz), reference=reference.numpy())
    print(f"wrote {3 * n} states ({n} per set: {', '.join(SETS)}) to {args.states}")
    return 0


def difference(a, b):
    d = np.abs(a - b)
    d[:, 2] = np.abs(np.remainder(a[:, 2] - b[:, 2] + math.pi, 2 * math.pi) - math.pi)
    return d


def compare(args):
    states, out = np.load(args.states), np.load(args.acts_out)
    truth, pT, backward, kind = states["truth"], states["pT"], states["backward"], states["kind"]
    acts_params, acts_ok = out["params"], out["ok"]

    # the torch maths, timed per navigation group
    field = ap.ConstantBField([0.0, 0.0, float(states["bz_tesla"]) * ap.UnitConstants.T])
    has_input = np.isfinite(states["qop"])
    port = np.full_like(truth, np.nan)
    port_seconds = 0.0
    for nav, rows in (("backward", backward & has_input), ("forward", ~backward & has_input)):
        args_ = [torch.from_numpy(states[k][rows]).to(args.device) for k in ("position", "direction", "qop")]
        # warm-up, untimed: CUDA start-up and first-call costs are not the maths'
        ap.propagate_to_perigee(*args_, reference=torch.from_numpy(states["reference"]).to(args.device),
                                field=field, navigation=nav)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = ap.propagate_to_perigee(*args_, reference=torch.from_numpy(states["reference"]).to(args.device),
                                         field=field, navigation=nav)[0]
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        port_seconds += time.perf_counter() - t0
        port[rows] = result.cpu().numpy()

    tolerances = np.array([args.length_tol, args.length_tol, args.angle_tol, args.angle_tol, args.qop_tol, args.length_tol])
    bins = [(0.1, 0.3), (0.3, 1.0), (1.0, 3.0), (3.0, 20.1)]
    agree = True
    for k_set, label in enumerate(SETS):
        rows = kind == k_set
        use = rows & acts_ok & np.isfinite(port).all(1)
        print(f"== {label}: {int(rows.sum())} tracks, input for {int((rows & has_input).sum())}, ACTS propagated "
              f"{int((rows & acts_ok).sum())}")
        print("  ACTS failures by true pT [GeV]: " + ", ".join(
            f"{lo}-{hi}: {int((rows & has_input & ~acts_ok & (pT >= lo) & (pT < hi)).sum())}/"
            f"{int((rows & has_input & (pT >= lo) & (pT < hi)).sum())}" for lo, hi in bins))
        exact = k_set < 2
        stat, stat_name = ((lambda d: d.max(axis=0, initial=0)), "max") if exact else \
            ((lambda d: np.percentile(d, 68, axis=0) if len(d) else np.zeros(6)), "68%")
        acts_vs_port = difference(acts_params[use], port[use])
        print(f"  {'':12s} {'|ACTS - truth| ' + stat_name:>20s} {'|torch - truth| ' + stat_name:>21s} "
              f"{'|ACTS - torch| 99%':>20s} {'|ACTS - torch| max':>20s}")
        acts_truth, port_truth = stat(difference(acts_params[use], truth[use])), stat(difference(port[use], truth[use]))
        for k, name in enumerate(NAMES):
            if not exact and k == 5:
                continue                          # no true time relative to a fitted state
            print(f"  {name:12s} {acts_truth[k]:20.2e} {port_truth[k]:21.2e} "
                  f"{np.percentile(acts_vs_port[:, k], 99) if use.any() else 0:20.2e} {acts_vs_port[:, k].max(initial=0):20.2e}")
        if exact:
            agree &= bool((acts_vs_port.max(axis=0, initial=0) <= tolerances).all())

    # MomentumHead's own fit against the truth
    fit = kind == 2
    fitted = fit & np.isfinite(states["qop"])
    true_pT, true_q = pT[fit][fitted[fit]], np.sign(truth[fit, 4])[fitted[fit]]
    rel = states["fit_pT"][fitted[fit]] / true_pT - 1.0
    print(f"== MomentumHead fit quality: {int(fitted.sum())}/{int(fit.sum())} fitted; pT rel bias {np.median(rel):+.4f}, "
          f"68% |rel| {np.percentile(np.abs(rel), 68):.4f}; charge correct {np.mean(states['fit_charge'][fitted[fit]] == true_q):.4f}")
    for lo, hi in bins:
        b = (true_pT >= lo) & (true_pT < hi)
        if b.any():
            print(f"  true pT {lo}-{hi} GeV: {int(b.sum())} tracks, 68% |pT rel| {np.percentile(np.abs(rel[b]), 68):.4f}, "
                  f"charge correct {np.mean(states['fit_charge'][fitted[fit]][b] == true_q[b]):.4f}")

    n_fit, n_port, n_acts = int(fit.sum()), int(has_input.sum()), int(out["n_attempted"])
    print("== Time per track")
    print(f"  MomentumHead hit fit ({states['fit_seconds']:.2f} s / {n_fit}): {states['fit_seconds'] / max(n_fit, 1) * 1e3:.4f} ms "
          f"(device of make)")
    print(f"  torch perigee ({port_seconds:.3f} s / {n_port}): {port_seconds / max(n_port, 1) * 1e3:.4f} ms on {args.device}")
    print(f"  ACTS perigee ({float(out['seconds']):.2f} s / {n_acts}): {float(out['seconds']) / max(n_acts, 1) * 1e3:.4f} ms (CPU, container)")
    print(f"== RESULT: {'AGREE' if agree else 'DIVERGE'} on the exact sets (tolerances: lengths {args.length_tol} mm, "
          f"angles {args.angle_tol}, q/p {args.qop_tol})")
    return 0 if agree else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("make")
    m.add_argument("states")
    m.add_argument("--n", type=int, default=2000, help="tracks per set")
    m.add_argument("--bz", type=float, default=1.4, help="Tesla")
    m.add_argument("--reference", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="perigee surface centre, mm")
    m.add_argument("--n_layers", type=int, default=48, help="TPC-like layers between the dataset's r limits")
    m.add_argument("--sigma_rphi", type=float, default=0.34, help="mm, hit smearing along the layer")
    m.add_argument("--sigma_z", type=float, default=1.0, help="mm, hit smearing in z")
    m.add_argument("--z_max", type=float, default=1050.0, help="mm, hits beyond |z| are dropped")
    m.add_argument("--tracks_per_event", type=int, default=50, help="tracks per MomentumHead call")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--device", default="cpu", help="for MomentumHead, e.g. cuda")
    c = sub.add_parser("compare")
    c.add_argument("states")
    c.add_argument("acts_out")
    c.add_argument("--length_tol", type=float, default=1e-3, help="mm")
    c.add_argument("--angle_tol", type=float, default=1e-5, help="rad")
    c.add_argument("--qop_tol", type=float, default=1e-8, help="e/GeV")
    c.add_argument("--device", default="cpu", help="for the torch perigee maths, e.g. cuda")
    args = parser.parse_args()
    return make(args) if args.mode == "make" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
