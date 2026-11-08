# ACTS for VertexHead

[ACTS](https://github.com/acts-project/acts) v48.2 (MPL-2.0), unmodified, in a Singularity/apptainer image, so it
never touches `fm4npp-env`; it runs as a separate program, so its licence places no conditions on this repository.
With `acts.image` set in the detector file, VertexHead's track builder (`TrackBuilder`, `use_acts`) hands each
fitted track to ACTS. ACTS's Kalman fitter refits its hits (`kalman_refit`), then ACTS propagates the track from the
innermost hit back to the beam line, through the field or the detector's tracking geometry with its material, and
returns perigee parameters with their covariance.

## Quick start

```bash
scripts/run/vertexing.sh setup      # image (once, an hour or more) + module (a minute); site detector file
scripts/run/vertexing.sh eval       # backbone, track finder, VertexHead -> per-point CSV
scripts/run/vertexing.sh compare    # VertexHead and ACTS's own vertex finder against truth on that CSV:
                                    # compare/reco_vertices.csv, compare/residuals.png beside the CSV's folder
```

The run is `scripts/configs/vertex_m5_gpuserver0.yaml` (`RUN_CONFIG=<file>` for another). Its `detector:` file,
`scripts/configs/sphenix_<site>.yaml` (gitignored), is rewritten by every `setup` from `detectors/sphenix/sphenix.yaml`
(so it gets the template's new keys and comments), keeping the site's own values (any setting differing from the
template) and then applying its options:

```bash
scripts/run/vertexing.sh setup --geometry <tgeo.root> --geometry_config <config.json> --material_map <material.json> \
    [--field_map <map.root or field.txt> [--field_kind rz] [--field_rescale <factor>]] \
    [--set acts.energy_loss=true] [--set vertex_head.track_error_scale=2.5]
```

The input files must exist (see sPHENIX inputs); `--set [section.]key=value` sets any other key, keeping its comment.
Options after `eval` or `compare` go to the Python script, e.g. `eval --max_eval_events 5000` or
`compare --tracks finder --max_events 500 --range 0.5`. Rerun `setup` after changing the C++.

## Pieces

| file | what |
|---|---|
| `build_images.sh`, `acts-base.def` | the image (ROOT, HepMC3, ACTS) and the module compiled inside it into `perigee_propagator/build/install` |
| `perigee_propagator/` | one C++ file: ACTS Kalman refit of the hits, then backward propagation to the perigee with covariance and material (not in ACTS's Python bindings) |
| `acts_worker.py` | runs in the image; VertexHead starts it once and streams track states to it |
| `detectors/<name>/` | the detector file template `<name>.yaml` and the tools that make its inputs |
| `check_geometry.py` | a detector's geometry and material map in ACTS |
| `compare_vertexing.py` | the PV reconstructions (the eval's, VertexHead's, ACTS's own vertex finder's) against truth: per-event CSV, residual plots in x, y, z and 3D, error table |
| `acts_vertexing.py` | ACTS's vertex finder as shipped (AMVF, Gaussian seeder) on given tracks, for the comparison |

Tracks ACTS fails to propagate are dropped; those the hit fit had kept are flagged in `acts_failed`. Where the
Kalman refit fails, the hit fit's state is propagated instead, with its covariance x `kalman_fallback_covariance_scale`,
flagged in `acts_refit_failed`. The worker prints a failure summary when it exits.

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
5. **Compare** VertexHead with ACTS's own finder on real events; a PV chi2/ndf far from 1 means
   `track_error_scale` (the file's `vertex_head` block) needs adjusting.

## sPHENIX inputs

- **Geometry:** CDB key `Tracking_Geometry` (a PHOOL file), exported once on a machine with the sPHENIX software:
  `root -l -b -q 'scripts/acts/detectors/sphenix/export_sphenix_geometry.C("<file>.root", "sphenix_ana559")'`
  writes `sphenix_ana559_tgeo.root` with the TPC measurement boxes sPHENIX's tracking adds.
- **TGeo configuration:** CDB key `ACTSGEOMETRYCONFIG`; **material map:** CDB key `ACTSMATERIALMAP`.
- **Field map:** part of the detector: the CDB field map (ROOT ntuple `fieldmap`, cm and T), given once to
  `setup --field_map <map>.root`, is converted (`convert_field_map.py`, needs uproot) to
  `detectors/sphenix/field_xyz.txt` (gitignored); every site file without a map of its own then uses it
  (`field.map`). Without a map, ACTS and the hit fit use the uniform `bz_tesla`.

ACTS builds sPHENIX's MVTX as one layer, merges INTT sub-layers and lets TPC layers overlap ("Layers are
overlapping"): that is the configuration's layering, as in sPHENIX's own tracking, which assigns its layers by
radius and logs ACTS at FATAL only. The worker does the same; `check_geometry.py` shows the full log.

The image has Python 3.12 and NumPy only (no torch): data goes between the environments through the worker's pipe.
