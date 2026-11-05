# ACTS for VertexHead

[ACTS](https://github.com/acts-project/acts) (MPL-2.0), unmodified, in a container, so it never touches
`fm4npp-env`. It is run as a separate program, not copied, so its licence places no conditions on this
repository's code. `VertexHead(use_acts=True)` hands each fitted track to ACTS, which propagates it from the
innermost hit to the beam line (through the field, or through the detector's tracking geometry with its material)
and returns ACTS bound perigee parameters and their covariance. The package's own pure-torch alternative is kept,
experimental, in `train/downstream/momentumhead.py`.

What ACTS adds, measured on 100 sPHENIX events (truth PV error, median / 68% / 90%, cm):

| setup | median | 68% | 90% |
|---|---|---|---|
| hit-count weighting (VertexHead's default) | 0.400 | 0.699 | 5.82 |
| covariance weighting + ACTS adaptive fit, own track fit | 0.328 | 0.417 | 6.20 |
| ... with ACTS and the material map (multiple scattering in the errors) | 0.292 | 0.412 | 5.69 |

## Pieces

| file | what |
|---|---|
| `build_images.sh`, `acts-base.def`, `acts-worker.def` | the container: base (ROOT, HepMC3, ACTS; slow, once) + worker (the module below; minutes) |
| `perigee_propagator/` | one C++ file: ACTS propagation to the perigee with direction, covariance and material, which ACTS's Python bindings lack |
| `acts_worker.py` | runs in the container; VertexHead starts it once and streams tracks to it |
| `detectors/<name>/` | everything detector-specific: the detector file `<name>.yaml` and the tools that make its inputs |
| `check_geometry.py` | checks a detector's geometry and material map in ACTS |
| `compare_vertexing.py` | VertexHead variants for one detector against the true vertex |
| `compare_acts.py`, `acts_reference.py` | the experimental torch perigee maths against ACTS |

## Build (once)

Singularity (or apptainer) with `--fakeroot`. From anywhere:

```bash
JOBS=8 scripts/acts/build_images.sh
```

The base image (ROOT and ACTS compile: an hour or more) is built only if missing (`--rebuild-base` to force); the
worker image takes minutes. Each ends with an import check (`acts OK`, `perigee propagator OK`). Put
`scripts/acts/acts-worker-48.2.0.sif` where users can reach it.

## Use

```python
head = VertexHead.from_detector("scripts/acts/detectors/sphenix/sphenix.yaml")
```

or `eval_vertex_reconstruction.py --detector scripts/acts/detectors/sphenix/sphenix.yaml` (YAML key
`vertex_detector`). Keyword arguments override the file. ACTS is used when the file's `acts.image` is set; its
paths must be visible inside the container: for directories outside your home, `export SINGULARITY_BIND=/zfs0`
(for example) first. Tracks ACTS fails on (mostly very soft curlers that never reach the beam line) are dropped
and flagged in the output `acts_failed`; the worker prints a summary of failures when it exits.

## Adding a detector

1. **Detector file.** Copy `detectors/sphenix/` to `detectors/<name>/` and fill in `<name>.yaml`: field (uniform
   Bz, or an ACTS text field map), beam position, the dataset's hit normalization, per-hit errors, the ACTS inputs
   below and any VertexHead settings.
2. **Geometry (optional, for material).** ACTS builds the tracking geometry from a TGeo ROOT file and an ACTS TGeo
   JSON configuration (TGeoDetector), the way the experiment's own ACTS tracking does. Export the TGeo file from
   the experiment's geometry (sPHENIX: `export_sphenix_geometry.C`, which also adds the TPC measurement boxes
   sPHENIX's tracking inserts) and take the configuration and the ACTS JSON material map from the experiment.
3. **Field map (optional).** Convert to ACTS's text format (mm, T; a full regular grid covering beam line to
   tracker): sPHENIX `convert_field_map.py`.
4. **Check** the geometry and material map in ACTS:

   ```bash
   singularity exec scripts/acts/acts-worker-48.2.0.sif acts-python scripts/acts/check_geometry.py \
       --geometry <tgeo.root> --geometry_config <config.json> --material_map <material.json>
   ```

   Every sensitive volume should have its module count and every material-map entry should land.
5. **Compare** the variants on real events (`eval_vertex_reconstruction.py --save_csv` output) and set the best in
   the file's `vertex_head` block; a PV chi2/ndf far from 1 means `track_error_scale` needs adjusting:

   ```bash
   python scripts/acts/compare_vertexing.py --csv vertex_per_point_data.csv --detector scripts/acts/detectors/<name>/<name>.yaml
   ```

## sPHENIX inputs

- Geometry: CDB key `Tracking_Geometry` (a PHOOL file), exported once on a machine with the sPHENIX software:
  `root -l -b -q 'scripts/acts/detectors/sphenix/export_sphenix_geometry.C("<file>.root", "sphenix_ana559")'`
  writes `sphenix_ana559_tgeo.root` (plus the TPC layer table and volume tree, for checking).
- TGeo configuration: CDB key `ACTSGEOMETRYCONFIG`; material map: CDB key `ACTSMATERIALMAP` (made for ACTS v45; it
  lands completely on the ACTS v48 geometry). TGeoDetector's "Layers are overlapping" errors for the TPC come
  from the configuration's 5 mm layer tolerance and are harmless.
- Field map: the CDB field map (ROOT ntuple `fieldmap`, cm and T):
  `python scripts/acts/detectors/sphenix/convert_field_map.py <map>.root field_xyz.txt` (`--uniform 1.4` writes a
  test map that must reproduce the constant field).

## Reference check: the experimental torch maths against ACTS

```bash
python scripts/acts/compare_acts.py make states.npz
```

```bash
singularity exec scripts/acts/acts-worker-48.2.0.sif acts-python scripts/acts/acts_reference.py states.npz acts_out.npz
```

```bash
python scripts/acts/compare_acts.py compare states.npz acts_out.npz
```

The container has Python 3.12 and NumPy only (no torch): data goes between the environments through files, or
through the worker's pipe.
