"""
acts_worker.py: ACTS computing perigee parameters and covariances for track states, inside the ACTS image with
the module from build_images.sh (perigee_propagator/build/install). VertexHead(use_acts=True) starts it once and
streams batches:

    singularity exec acts-base-48.2.0.sif acts-python scripts/acts/acts_worker.py '<json config>'

Config (VertexHead.acts_worker_config, mm and T): reference, bz_tesla or field_map (+ field_map_kind),
geometry + geometry_config (+ material_map, energy_loss), max_steps.

Protocol, after the line ACTS_WORKER_READY: n (uint64), then float64 position (n, 3), direction (n, 3), qop (n,),
covariance (n, 6, 6); reply float64 params (n, 6), covariance (n, 6, 6), then ok (n,) uint8. States are
propagated backward to the perigee. EOF ends it and prints a failure summary; everything else goes to stderr.
"""
import collections
import json
import os
import struct
import sys

import numpy as np

import acts

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "perigee_propagator", "build", "install"))
import acts_perigee_propagator  # noqa: E402


def build_tracking_geometry(geometry, geometry_config, material_map=None, quiet=False):
    """ACTS tracking geometry from TGeo (TGeoDetector with its JSON configuration), with a JSON material map if
    given. quiet: ACTS logs only fatal errors while building, as sPHENIX's MakeActsGeometry does (sPHENIX's layers
    overlap by design; check_geometry.py shows the full log). Returns (detector, geometry): the detector owns the
    detector elements and must be kept alive."""
    import acts.examples.tgeo
    import acts.json
    config = acts.examples.tgeo.TGeoDetector.Config()
    config.readJson(geometry_config)
    config.fileName = geometry
    level = acts.logging.FATAL if quiet else acts.logging.WARNING
    if quiet:
        config.surfaceLogLevel = config.layerLogLevel = config.volumeLogLevel = level
    if material_map:
        config.materialDecorator = acts.json.JsonMaterialDecorator(
            rConfig=acts.json.MaterialMapJsonConverter.Config(), jFileName=material_map, level=level)
    detector = acts.examples.tgeo.TGeoDetector(config)
    return detector, detector.trackingGeometry()


def build_propagator(config):
    """(PerigeePropagator, objects to keep alive) from a config dict (see the module docstring)."""
    # ACTS reads a missing file as empty input ("parse error ... empty input"): name it instead
    missing = [f"{k}={config[k]}" for k in ("field_map", "geometry", "geometry_config", "material_map")
               if config.get(k) and not os.path.isfile(config[k])]
    if missing:
        raise FileNotFoundError(f"not visible inside the container: {', '.join(missing)} (a wrong path, or its "
                                "directory not bound: --bind / SINGULARITY_BIND)")
    u = acts.UnitConstants
    if config.get("field_map"):
        loader = {"xyz": acts.MagneticFieldMapXyz, "rz": acts.MagneticFieldMapRz}[config.get("field_map_kind", "xyz")]
        field = loader(config["field_map"], lengthUnit=u.mm, bFieldUnit=u.T)
    else:
        field = acts.ConstantBField(acts.Vector3(0.0, 0.0, float(config["bz_tesla"]) * u.T))
    detector = geometry = None
    if config.get("geometry"):
        detector, geometry = build_tracking_geometry(config["geometry"], config["geometry_config"],
                                                     config.get("material_map"), quiet=True)
    propagator = acts_perigee_propagator.PerigeePropagator(
        field=field, reference=[float(v) for v in config.get("reference", (0.0, 0.0, 0.0))], geometry=geometry,
        energy_loss=bool(config.get("energy_loss", True)), max_steps=int(config.get("max_steps") or 0))
    return propagator, (field, detector, geometry)


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data += chunk
    return bytes(data)


def serve(config):
    # the protocol gets the real stdout; anything printed from here on (Python or ACTS's C++) goes to stderr
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    propagator, _keep = build_propagator(config)
    stdin = sys.stdin.buffer
    out.write(b"ACTS_WORKER_READY\n")
    out.flush()
    errors, attempted, propagated = collections.Counter(), 0, 0
    while True:
        header = read_exact(stdin, 8)
        if header is None:
            # at the end (VertexHead closes the worker at exit): why ACTS failed, for the dropped tracks
            print(f"ACTS worker: {propagated} of {attempted} states propagated" + ("; failures:" if errors else ""))
            for key, count in errors.most_common():
                print(f"  {count:7d}  {key}")
            sys.stdout.flush()
            return 0
        (n,) = struct.unpack("<Q", header)
        body = read_exact(stdin, n * 8 * 43)
        if body is None:
            return 1
        values = np.frombuffer(body, dtype="<f8")
        position, direction = values[: 3 * n].reshape(n, 3), values[3 * n: 6 * n].reshape(n, 3)
        qop, covariance = values[6 * n: 7 * n], values[7 * n:].reshape(n, 6, 6)
        params, cov, ok, failures = propagator.propagate(position, direction, qop, covariance)
        errors.update(failures)
        attempted += int((np.isfinite(position).all(axis=1) & np.isfinite(direction).all(axis=1) & np.isfinite(qop)).sum())
        propagated += int(ok.sum())
        out.write(params.astype("<f8").tobytes() + cov.astype("<f8").tobytes() + ok.astype(np.uint8).tobytes())
        out.flush()


if __name__ == "__main__":
    sys.exit(serve(json.loads(sys.argv[1])))
