# ACTS for VertexHead

[ACTS](https://github.com/acts-project/acts) v48.2 (MPL-2.0), unmodified, in a Singularity/apptainer image, so it
never touches `fm4npp-env`; it runs as a separate program, so its licence places no conditions on this repository.
`VertexHead(use_acts=True)` hands each fitted track to ACTS, which propagates it from the innermost hit back to the
beam line, through the field or the detector's tracking geometry with its material, and returns perigee
parameters with their covariance.

Truth PV error on 1000 sPHENIX events (median / 68% / 90%, cm), VertexHead's adaptive fit with each track model:

| track model | median | 68% | 90% |
|---|---|---|---|
| VertexHead's own hit fit | 0.247 | 0.380 | 1.49 |
| ACTS, geometry without material | 0.246 | 0.377 | 1.41 |
| ACTS with the material map | 0.234 | 0.351 | 1.37 |

## Quick start

```bash
scripts/run/vertexing.sh setup      # image (once, an hour or more) + module (a minute); site detector file
scripts/run/vertexing.sh eval       # backbone, track finder, VertexHead -> per-point CSV
scripts/run/vertexing.sh compare    # VertexHead variants against truth on that CSV
```

The run is `scripts/configs/vertex_m5_gpuserver0.yaml` (`RUN_CONFIG=<file>` for another). Its `detector:` file,
`scripts/configs/sphenix_<site>.yaml` (gitignored), is written by `setup` from `detectors/sphenix/sphenix.yaml`;
fill in its ACTS inputs (see sPHENIX inputs). Options after the command go to the Python script, e.g.
`eval --max_eval_events 5000` or `compare --loose_sv`. Rerun `setup` after changing the C++.

## Pieces

| file | what |
|---|---|
| `build_images.sh`, `acts-base.def` | the image (ROOT, HepMC3, ACTS) and the module compiled inside it into `perigee_propagator/build/install` |
| `perigee_propagator/` | one C++ file: backward propagation to the perigee with covariance and material (not in ACTS's Python bindings) |
| `acts_worker.py` | runs in the image; VertexHead starts it once and streams track states to it |
| `detectors/<name>/` | the detector file template `<name>.yaml` and the tools that make its inputs |
| `check_geometry.py` | a detector's geometry and material map in ACTS (`--layers`: each layer's extent) |
| `compare_vertexing.py` | VertexHead variants against truth: PV, PV per axis, SVs |

Tracks ACTS fails on (soft curlers the hit fit already rejects) are dropped and flagged in `acts_failed`; the
worker prints a failure summary when it exits.

## Adding a detector

1. **Detector file.** Copy `detectors/sphenix/` to `detectors/<name>/` and fill in `<name>.yaml`: field (uniform
   Bz or an ACTS text field map), beam position, the dataset's hit normalization, per-hit errors, ACTS inputs and
   any VertexHead settings. Relative paths are relative to the file.
2. **Geometry (optional, for material).** A TGeo ROOT file and the experiment's ACTS TGeo JSON configuration and
   material map, as its own ACTS tracking uses them.
3. **Field map (optional).** ACTS text format (mm, T; a regular grid from the beam line to the tracker).
4. **Check** in ACTS:

   ```bash
   singularity exec scripts/acts/acts-base-48.2.0.sif acts-python scripts/acts/check_geometry.py \
       --geometry <tgeo.root> --geometry_config <config.json> --material_map <material.json>
   ```

   Every sensitive volume should have its module count and every material-map entry should land.
5. **Compare** on real events and set the best variant in the file's `vertex_head` block; a PV chi2/ndf far from
   1 means `track_error_scale` needs adjusting.

## sPHENIX inputs

- **Geometry:** CDB key `Tracking_Geometry` (a PHOOL file), exported once on a machine with the sPHENIX software:
  `root -l -b -q 'scripts/acts/detectors/sphenix/export_sphenix_geometry.C("<file>.root", "sphenix_ana559")'`
  writes `sphenix_ana559_tgeo.root` with the TPC measurement boxes sPHENIX's tracking adds.
- **TGeo configuration:** CDB key `ACTSGEOMETRYCONFIG`; **material map:** CDB key `ACTSMATERIALMAP` (made for ACTS
  v45, whose layer building is the same as v48.2's; all 56 entries land).
- **Field map:** the CDB field map (ROOT ntuple `fieldmap`, cm and T):
  `python scripts/acts/detectors/sphenix/convert_field_map.py <map>.root field_xyz.txt`.

ACTS builds sPHENIX's MVTX as one layer, merges INTT sub-layers and lets TPC layers overlap ("Layers are
overlapping"): that is the configuration's layering, the same in sPHENIX's own tracking, which assigns its layers
by radius and logs ACTS at FATAL only. The worker does the same; `check_geometry.py` shows the full log.

The image has Python 3.12 and NumPy only (no torch): data goes between the environments through the worker's pipe.
