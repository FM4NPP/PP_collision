"""
acts_reference.py: ACTS's bound perigee parameters for given track states. Runs INSIDE the ACTS container
(Python 3.12, NumPy, ACTS bindings; no torch): ACTS is run, not copied.

    singularity exec acts-ref-48.2.0.sif acts-python scripts/acts/acts_reference.py states.npz acts_out.npz

Input (.npz, ACTS units: mm, GeV, e, T for the field), from compare_acts.py make:
    position (N, 3), direction (N, 3) unit, qop (N,) e/GeV, backward (N,) bool: the perigee lies behind
    the state; bz_tesla (), reference (3,) the perigee surface centre.
Output (.npz): params (N, 6) (d0, z0, phi, theta, q/p, t), NaN where ACTS failed or the state is not
    finite; ok (N,) bool; seconds, n_attempted: ACTS's time over the finite states.

Each state becomes curvilinear parameters, propagated (EigenStepper in a constant field, no geometry)
onto the perigee surface. A backward state is propagated as the reversed, charge-conjugated track (same
helix) and the result flipped back: d0, phi + pi, pi - theta, q/p and t change sign, z0 does not.
"""
import math
import sys
import time
import traceback

import numpy as np

import acts


def main(in_path, out_path):
    print("acts from", acts.__file__, "| version", next((str(getattr(acts, k)) for k in ("__version__", "version")
                                                        if hasattr(acts, k)), "unknown"))
    names = ("UnitConstants", "Vector3", "Vector4", "ConstantBField", "EigenStepper", "VoidNavigator",
             "EigenVoidPropagator", "PropagatorPlainOptions", "GeometryContext", "MagneticFieldContext",
             "Surface", "BoundTrackParameters", "ParticleHypothesis")
    missing = [n for n in names if not hasattr(acts, n)]
    if missing:
        # what this ACTS version offers instead, to adapt the calls
        print("MISSING in acts:", missing)
        print("related names:", sorted(n for n in dir(acts) if any(k in n for k in (
            "Propagat", "Stepper", "Navigator", "Curvilinear", "Perigee", "Bound", "Context", "Vector", "Field"))))
        return 2

    u = acts.UnitConstants
    data = np.load(in_path)
    position, direction, qop, backward = data["position"], data["direction"], data["qop"], data["backward"]
    reference = data["reference"].astype(float).tolist()
    field = acts.ConstantBField(acts.Vector3(0.0, 0.0, float(data["bz_tesla"]) * u.T))
    propagator = acts.EigenVoidPropagator(acts.EigenStepper(field), acts.VoidNavigator())
    options = acts.PropagatorPlainOptions(acts.GeometryContext.dangerouslyDefaultConstruct(), acts.MagneticFieldContext())
    surface = acts.Surface.createPerigee(acts.Vector3(*reference))
    hypothesis = acts.ParticleHypothesis.pion

    n = len(qop)
    params = np.full((n, 6), np.nan)
    errors = {}
    t0 = time.perf_counter()
    finite = np.isfinite(position).all(axis=1) & np.isfinite(direction).all(axis=1) & np.isfinite(qop)
    for i in np.flatnonzero(finite):           # non-finite = no state (e.g. a failed hit fit): left NaN
        flip = -1.0 if backward[i] else 1.0
        try:
            start = acts.BoundTrackParameters.createCurvilinear(
                pos4=acts.Vector4(*position[i].tolist(), 0.0), dir=acts.Vector3(*(flip * direction[i]).tolist()),
                qOverP=float(flip * qop[i]), cov=None, particleHypothesis=hypothesis)
            d0, z0, phi, theta, q_over_p, t = list(propagator.propagateToSurface(
                start=start, target=surface, options=options).parameters)
            params[i] = ([-d0, z0, math.remainder(phi + math.pi, 2 * math.pi), math.pi - theta, -q_over_p, -t]
                         if backward[i] else [d0, z0, phi, theta, q_over_p, t])
        except Exception:  # noqa: BLE001 - count each distinct failure, keep going
            key = traceback.format_exc(limit=2).strip().splitlines()[-1]
            errors.setdefault(key, [0, traceback.format_exc(limit=3)])[0] += 1
    elapsed = time.perf_counter() - t0
    ok = np.isfinite(params).all(axis=1)
    np.savez(out_path, params=params, ok=ok, seconds=np.float64(elapsed), n_attempted=np.int64(finite.sum()))
    print(f"{n} states ({int(finite.sum())} with input), {int(ok.sum())} propagated, {elapsed:.1f} s "
          f"({elapsed / max(int(finite.sum()), 1) * 1e3:.2f} ms/state)")
    for key, (count, trace) in errors.items():
        print(f"FAILED {count}x: {key}\n" + "\n".join("    " + line for line in trace.rstrip().splitlines()))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
