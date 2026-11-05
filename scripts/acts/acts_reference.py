"""
acts_reference.py: ACTS's bound perigee parameters for a file of track states, for compare_acts.py. Runs
INSIDE the ACTS container (Python 3.12, NumPy, ACTS bindings; no torch): ACTS is run, not copied. Uses
the same ACTS calls as the VertexHead worker (acts_worker.py).

    singularity exec acts-worker-48.2.0.sif acts-python scripts/acts/acts_reference.py states.npz acts_out.npz
    ... --field_map field.txt [--field_map_kind rz]      (a converted field map instead of the constant field)

Input (.npz, ACTS units: mm, GeV, e, T for the field), from compare_acts.py make:
    position (N, 3), direction (N, 3) unit, qop (N,) e/GeV, backward (N,) bool: the perigee lies behind
    the state; bz_tesla (), reference (3,) the perigee surface centre.
Output (.npz): params (N, 6) (d0, z0, phi, theta, q/p, t), NaN where ACTS failed or the state is not
    finite; ok (N,) bool; seconds, n_attempted: ACTS's time over the finite states.
"""
import argparse
import sys
import time

import numpy as np

import acts
from acts_worker import build_propagator, propagate_to_perigee


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("states")
    parser.add_argument("out")
    parser.add_argument("--field_map", default=None, help="text field map (detectors/<name>/convert_field_map.py), mm and T")
    parser.add_argument("--field_map_kind", default="xyz", choices=("xyz", "rz"))
    parser.add_argument("--max_steps", type=int, default=None, help="ACTS propagation step limit (ACTS default if unset)")
    args = parser.parse_args()

    print("acts from", acts.__file__, "| version", next((str(getattr(acts, k)) for k in ("__version__", "version")
                                                        if hasattr(acts, k)), "unknown"))
    data = np.load(args.states)
    setup = build_propagator({"bz_tesla": float(data["bz_tesla"]), "reference": data["reference"].tolist(),
                              "field_map": args.field_map, "field_map_kind": args.field_map_kind,
                              "max_steps": args.max_steps})
    print("field:", f"map {args.field_map} ({args.field_map_kind})" if args.field_map else f"constant {float(data['bz_tesla'])} T")
    finite = np.isfinite(data["qop"])
    errors = {}
    t0 = time.perf_counter()
    params, ok = propagate_to_perigee(setup, data["position"], data["direction"], data["qop"], data["backward"], errors)
    elapsed = time.perf_counter() - t0
    np.savez(args.out, params=params, ok=ok, seconds=np.float64(elapsed), n_attempted=np.int64(finite.sum()))
    print(f"{len(ok)} states ({int(finite.sum())} with input), {int(ok.sum())} propagated, {elapsed:.1f} s "
          f"({elapsed / max(int(finite.sum()), 1) * 1e3:.2f} ms/state)")
    for key, (count, trace) in errors.items():
        print(f"FAILED {count}x: {key}\n" + "\n".join("    " + line for line in trace.rstrip().splitlines()))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
