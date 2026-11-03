"""
acts_worker.py: ACTS (unmodified, through its Python bindings) computing bound perigee parameters for
track states. Runs INSIDE the ACTS container (Python 3.12, NumPy, ACTS; no torch). Two uses:

- as a module: build_propagator() and propagate_to_perigee(), shared with acts_reference.py;
- as a persistent worker for VertexHead(use_acts=True), which starts it once and streams batches:

    singularity exec acts-ref-48.2.0.sif acts-python scripts/acts/acts_worker.py '<json config>'

Config (JSON): reference [x, y, z] mm (perigee surface centre); bz_tesla (constant field) or field_map
(a text map in the container's view, see convert_field_map.py) with field_map_kind "xyz" or "rz",
field_map_length_unit / field_map_field_unit (ACTS units per file unit; default mm and T); max_steps
(ACTS default if absent).

Protocol on stdin/stdout, after a line "ACTS_WORKER_READY": per batch the parent writes n (uint64), then
position (n, 3), direction (n, 3), qop (n,) as float64 and backward (n,) as uint8; the worker answers
params (n, 6) float64 (d0, z0, phi, theta, q/p, t in mm, GeV; NaN where ACTS failed or the input is not
finite) and ok (n,) uint8. EOF on stdin ends it. Everything else the worker or ACTS prints goes to stderr.

A backward state (perigee behind it, e.g. at the innermost hit) is propagated as the reversed,
charge-conjugated track (the same helix) and the result flipped back: d0, phi + pi, pi - theta, q/p and
t change sign, z0 does not. No material effects: ACTS's core Python propagation has none.
"""
import json
import math
import os
import struct
import sys
import traceback

import numpy as np

import acts


def build_propagator(config):
    """ACTS propagator, options, perigee surface and particle hypothesis from a config dict."""
    u = acts.UnitConstants
    if config.get("field_map"):
        loader = {"xyz": acts.MagneticFieldMapXyz, "rz": acts.MagneticFieldMapRz}[config.get("field_map_kind", "xyz")]
        field = loader(config["field_map"], lengthUnit=config.get("field_map_length_unit", u.mm),
                       bFieldUnit=config.get("field_map_field_unit", u.T))
    else:
        field = acts.ConstantBField(acts.Vector3(0.0, 0.0, float(config["bz_tesla"]) * u.T))
    propagator = acts.EigenVoidPropagator(acts.EigenStepper(field), acts.VoidNavigator())
    options = acts.PropagatorPlainOptions(acts.GeometryContext.dangerouslyDefaultConstruct(), acts.MagneticFieldContext())
    if config.get("max_steps") is not None:
        options.maxSteps = int(config["max_steps"])
    surface = acts.Surface.createPerigee(acts.Vector3(*[float(v) for v in config.get("reference", (0.0, 0.0, 0.0))]))
    return propagator, options, surface, acts.ParticleHypothesis.pion


def propagate_to_perigee(setup, position, direction, qop, backward, errors=None):
    """(n, 6) perigee parameters (NaN where ACTS failed or the input is not finite) and ok (n,)."""
    propagator, options, surface, hypothesis = setup
    n = len(qop)
    params = np.full((n, 6), np.nan)
    finite = np.isfinite(position).all(axis=1) & np.isfinite(direction).all(axis=1) & np.isfinite(qop)
    for i in np.flatnonzero(finite):
        flip = -1.0 if backward[i] else 1.0
        try:
            start = acts.BoundTrackParameters.createCurvilinear(
                pos4=acts.Vector4(*position[i].tolist(), 0.0), dir=acts.Vector3(*(flip * direction[i]).tolist()),
                qOverP=float(flip * qop[i]), cov=None, particleHypothesis=hypothesis)
            d0, z0, phi, theta, q_over_p, t = list(propagator.propagateToSurface(
                start=start, target=surface, options=options).parameters)
            params[i] = ([-d0, z0, math.remainder(phi + math.pi, 2 * math.pi), math.pi - theta, -q_over_p, -t]
                         if backward[i] else [d0, z0, phi, theta, q_over_p, t])
        except Exception:  # noqa: BLE001 - one track failing must not stop the batch
            if errors is not None:
                key = traceback.format_exc(limit=2).strip().splitlines()[-1]
                errors.setdefault(key, [0, traceback.format_exc(limit=3)])[0] += 1
    return params, np.isfinite(params).all(axis=1)


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data += chunk
    return bytes(data)


def serve(config):
    setup = build_propagator(config)
    # the protocol gets the real stdout; anything printed from here on (Python or ACTS's C++) goes to stderr
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    stdin = sys.stdin.buffer
    out.write(b"ACTS_WORKER_READY\n")
    out.flush()
    while True:
        header = read_exact(stdin, 8)
        if header is None:
            return 0
        (n,) = struct.unpack("<Q", header)
        body = read_exact(stdin, n * 8 * 7 + n)
        if body is None:
            return 1
        values = np.frombuffer(body[: n * 56], dtype="<f8")
        position, direction, qop = values[: 3 * n].reshape(n, 3), values[3 * n: 6 * n].reshape(n, 3), values[6 * n:]
        backward = np.frombuffer(body[n * 56:], dtype=np.uint8).astype(bool)
        params, ok = propagate_to_perigee(setup, position, direction, qop, backward)
        out.write(params.astype("<f8").tobytes() + ok.astype(np.uint8).tobytes())
        out.flush()


if __name__ == "__main__":
    sys.exit(serve(json.loads(sys.argv[1])))
