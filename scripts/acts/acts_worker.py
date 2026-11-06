"""
acts_worker.py: ACTS computing perigee parameters (and covariances) for track states, inside the ACTS image with
the module from build_images.sh (perigee_propagator/build/install). VertexHead(use_acts=True) starts it once and
streams batches:

    singularity exec acts-base-48.2.0.sif acts-python scripts/acts/acts_worker.py '<json config>'

Config (VertexHead.acts_worker_config, mm and T): reference, bz_tesla or field_map (+ field_map_kind),
geometry + geometry_config (+ material_map, layer_envelope_mm, energy_loss, multiple_scattering), max_steps,
covariance.

Protocol, after the line ACTS_WORKER_READY: n (uint64), position (n, 3), direction (n, 3), qop (n,) float64,
backward (n,) uint8 [, covariance (n, 6, 6)]; reply params (n, 6) [, covariance (n, 6, 6)], ok (n,) uint8.
EOF ends it and prints a failure summary; everything else goes to stderr.
"""
import importlib
import json
import os
import struct
import sys

import numpy as np

import acts

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "perigee_propagator", "build", "install"))
import acts_perigee_propagator  # noqa: E402


def first_module(*names):
    for name in names:
        try:
            return importlib.import_module(name)
        except ImportError:
            pass
    raise ImportError(f"none of {names} in this ACTS build")


def build_tracking_geometry(geometry, geometry_config, material_map=None, layer_envelope_mm=None):
    """ACTS tracking geometry from TGeo (TGeoDetector with its JSON configuration), with a JSON material map if
    given; layer_envelope_mm: radial envelope added to each layer (None: ACTS's 1 mm). Returns (detector, geometry): the detector owns the detector elements and must
    be kept alive."""
    tgeo = first_module("acts.examples.tgeo", "acts.tgeo")
    config = tgeo.TGeoDetector.Config()
    config.readJson(geometry_config)
    config.fileName = geometry
    if layer_envelope_mm is not None:
        config.layerEnvelopeR = float(layer_envelope_mm) * acts.UnitConstants.mm
    if material_map:
        json_plugin = first_module("acts.json", "acts.examples.json")
        config.materialDecorator = json_plugin.JsonMaterialDecorator(
            rConfig=json_plugin.MaterialMapJsonConverter.Config(), jFileName=material_map, level=acts.logging.WARNING)
    detector = tgeo.TGeoDetector(config)
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
        field = loader(config["field_map"], lengthUnit=config.get("field_map_length_unit", u.mm),
                       bFieldUnit=config.get("field_map_field_unit", u.T))
    else:
        field = acts.ConstantBField(acts.Vector3(0.0, 0.0, float(config["bz_tesla"]) * u.T))
    detector = geometry = None
    if config.get("geometry"):
        detector, geometry = build_tracking_geometry(config["geometry"], config["geometry_config"], config.get("material_map"),
                                                     layer_envelope_mm=config.get("layer_envelope_mm"))
    propagator = acts_perigee_propagator.PerigeePropagator(
        field=field, reference=[float(v) for v in config.get("reference", (0.0, 0.0, 0.0))], geometry=geometry,
        energy_loss=bool(config.get("energy_loss", True)), multiple_scattering=bool(config.get("multiple_scattering", True)),
        max_steps=int(config.get("max_steps") or 0))
    return propagator, (field, detector, geometry)


def propagate_to_perigee(setup, position, direction, qop, backward, errors=None, covariance=None):
    """(n, 6) perigee parameters (NaN where ACTS failed or the input is not finite) and ok (n,); with a start
    covariance (n, 6, 6) (ACTS curvilinear, mm) also the perigee covariance (n, 6, 6). errors: a dict that
    collects {failure: [count, detail]}."""
    params, ok, failures, cov = setup[0].propagate(
        np.asarray(position, dtype=np.float64), np.asarray(direction, dtype=np.float64),
        np.asarray(qop, dtype=np.float64), np.asarray(backward, dtype=bool),
        None if covariance is None else np.asarray(covariance, dtype=np.float64))
    if errors is not None:
        for key, count in failures.items():
            errors.setdefault(key, [0, key])[0] += count
    return (params, ok.astype(bool)) if covariance is None else (params, ok.astype(bool), cov)


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data += chunk
    return bytes(data)


def serve(config):
    # the protocol gets the real stdout; anything printed from here on (Python or ACTS's C++, e.g. the geometry
    # build's log) goes to stderr
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    setup = build_propagator(config)
    stdin = sys.stdin.buffer
    out.write(b"ACTS_WORKER_READY\n")
    out.flush()
    errors, counts = {}, [0, 0]  # failure reason -> [count, detail]; [states attempted, propagated]
    with_covariance = bool(config.get("covariance"))
    while True:
        header = read_exact(stdin, 8)
        if header is None:
            # at the end (VertexHead closes the worker at exit): why ACTS failed, for the dropped tracks
            print(f"ACTS worker: {counts[1]} of {counts[0]} states propagated" + ("; failures:" if errors else ""))
            for key, (count, _) in sorted(errors.items(), key=lambda item: -item[1][0]):
                print(f"  {count:7d}  {key}")
            sys.stdout.flush()
            return 0
        (n,) = struct.unpack("<Q", header)
        body = read_exact(stdin, n * 8 * 7 + n + (n * 288 if with_covariance else 0))
        if body is None:
            return 1
        values = np.frombuffer(body[: n * 56], dtype="<f8")
        position, direction, qop = values[: 3 * n].reshape(n, 3), values[3 * n: 6 * n].reshape(n, 3), values[6 * n:]
        backward = np.frombuffer(body[n * 56: n * 57], dtype=np.uint8).astype(bool)
        covariance = np.frombuffer(body[n * 57:], dtype="<f8").reshape(n, 6, 6) if with_covariance else None
        result = propagate_to_perigee(setup, position, direction, qop, backward, errors, covariance)
        counts[0] += int((np.isfinite(position).all(axis=1) & np.isfinite(direction).all(axis=1) & np.isfinite(qop)).sum())
        counts[1] += int(result[1].sum())
        out.write(result[0].astype("<f8").tobytes() + (result[2].astype("<f8").tobytes() if with_covariance else b"")
                  + result[1].astype(np.uint8).tobytes())
        out.flush()


if __name__ == "__main__":
    sys.exit(serve(json.loads(sys.argv[1])))
