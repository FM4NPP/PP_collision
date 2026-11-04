# ACTS reference container

Unmodified [ACTS](https://github.com/acts-project/acts) (MPL-2.0) with its Python bindings, in a container,
so it never touches `fm4npp-env`. ACTS is how this package computes momentum and track parameters. It is
run as a separate program and exchanges data through files; it is run, not copied, so its licence places
no conditions on this repository's code. The package's own pure-torch alternative is kept, experimental,
in `train/downstream/momentumhead.py`, and checked against ACTS below.

## Build (once)

The ACTS compile takes a while; set `JOBS` to the cores you can use. Either route gives the same image.

### Without docker rights: Singularity

From `scripts/acts` (`--fakeroot` needs the admin to have enabled it for your user; without it the build
stops with a fakeroot or subuid error):

```bash
JOBS=8 singularity build --fakeroot acts-ref-48.2.0.sif acts.def
```

The build ends by importing `acts`; if that line prints `acts OK`, the image works. Put the `.sif` in a
shared location.

### With docker rights (the `docker` group)

From the repository root:

```bash
docker build --build-arg JOBS=8 -t acts-ref:48.2.0 scripts/acts
```

The build ends by importing `acts`; if that line prints `acts OK`, the image works.

Convert it to a Singularity image file that anyone can use without docker rights, and put it in a
shared location:

```bash
singularity build acts-ref-48.2.0.sif docker-daemon://acts-ref:48.2.0
```

## Use

Run a Python script with ACTS (the current directory is visible inside the container):

```bash
singularity exec acts-ref-48.2.0.sif acts-python my_script.py
```

or with docker:

```bash
docker run --rm -v "$PWD":/work -w /work acts-ref:48.2.0 acts-python my_script.py
```

If the build printed `WARNING: acts import check failed`, the image is still usable without the launcher:

```bash
singularity exec acts-ref-48.2.0.sif bash -c "source /opt/acts-build/python/setup.sh && python3 my_script.py"
```

## VertexHead with ACTS

`VertexHead(use_acts=True, acts_image="acts-ref-48.2.0.sif")` has ACTS build each track's perigee
parameters before its own fit: the fitted helix's state nearest the innermost hit, propagated back to the
beam spot by ACTS. It starts one persistent worker process (`acts_worker.py` in the container) on first
use and streams every batch to it, so the container starts once per run (about a second), not per batch.
Tracks ACTS fails on are dropped (zero weight) and flagged in the output `acts_failed`. VertexHead itself
never imports ACTS. Options: `acts_command` (instead of the image), `acts_field_map`,
`acts_field_map_kind`, `acts_max_steps`.

Check it on real events (`--save_csv` output of `eval_vertex_reconstruction.py`) against VertexHead's own
perigee (temporary script):

```bash
python scripts/acts/check_vertexhead_acts.py --csv vertex_per_point_data.csv --image acts-ref-48.2.0.sif
```

## Field maps

ACTS's core Python bindings read plain-text field maps. Convert an sPHENIX map (ROOT ntuple `fieldmap`:
`x y z bx by bz` in cm and T, as PHField3DCartesian reads it; `--kind rz` for `r z br bz`) in the
training environment (uses uproot):

```bash
python scripts/acts/convert_field_map.py sphenix_field_map.root field_xyz.txt --rescale 1.0
```

then pass `acts_field_map="field_xyz.txt"` to VertexHead, or `--field_map field_xyz.txt` to the scripts.
The map must be a full regular grid covering the beam line out to the TPC: ACTS has no field outside it,
and a track whose propagation leaves the map fails (and is dropped). A uniform test map, which must
reproduce the constant-field results:

```bash
python scripts/acts/convert_field_map.py --uniform 1.4 uniform_xyz.txt
```

## Material

The sPHENIX material map is keyed to the ACTS surfaces built from sPHENIX's detector geometry
(conditions database key `Tracking_Geometry`, a PHOOL file only sPHENIX's software can read). Export it
once, on a machine with the sPHENIX software set up, to plain files the container can read:

```bash
root -l -b -q 'scripts/acts/export_sphenix_geometry.C("<Tracking_Geometry file>.root", "sphenix_ana559")'
```

This writes `sphenix_ana559_tgeo.root` (the geometry as a plain ROOT TGeoManager, with the TPC
measurement boxes sPHENIX's tracking adds before building its ACTS geometry: 48 layers x 12 modules x 12
phi segments), `sphenix_ana559_tpc_layers.txt` (TPC layer radii, thicknesses and the maximum drift length)
and `sphenix_ana559_volumes.txt` (the volume tree). Material propagation itself is not available with this image yet: ACTS's core Python
propagation has no material effects (no MaterialInteractor), and reading the material map (CDB key
ACTSMATERIALMAP) onto the geometry built with the TGeo configuration (CDB key ACTSGEOMETRYCONFIG) needs
ACTS's JSON and ROOT/TGeo plugins, which this image does not build.

### Material container (in progress)

A second image adds what material needs: ROOT (built from source for C++20), HepMC3 with ROOT I/O, and
ACTS's examples with its ROOT/TGeo and JSON plugins, all in C++20. The bindings-only image above stays as
it is. It is built in two steps, so a failure in the ACTS step does not recompile ROOT. From `scripts/acts`
(both compiles take a while):

```bash
JOBS=8 singularity build --fakeroot acts-material-base.sif acts-material-base.def
```

```bash
JOBS=8 singularity build --fakeroot acts-material-48.2.0.sif acts-material.def
```

Then check that sPHENIX's material map lands on the geometry built from the export (temporary script):

```bash
singularity exec acts-material-48.2.0.sif acts-python scripts/acts/check_material_map.py \
    --geometry sphenix_ana559_tgeo.root --config tgeo-sphenix-mms-actsv45.0.0.json \
    --material sphenix-mm-acts-material-actsv45.0.0.json
```

It reports the surfaces built per ACTS volume, how many carry material, and which material-map entries
found no surface (ana559 with the ACTS v45 map: all 56 land, on approach surfaces; sensitive surfaces
432 MVTX, 224 INTT, 13824 TPC, 16 micromegas). TGeoDetector's "Layers are overlapping" errors for the TPC
come from the configuration's 5 mm layer tolerance against TPC layers 5.6 to 11 mm apart; they do not stop
the map from landing.

### Propagation with material

ACTS's core Python propagation has no material effects (no MaterialInteractor), so a small module,
`material_propagator/` (one C++ file, linked against the unmodified ACTS libraries), adds that one call. It
is built in a third image layered on the material image, so changing it recompiles only the module:

```bash
JOBS=8 singularity build --fakeroot acts-material-worker-48.2.0.sif acts-material-worker.def
```

The build ends with `material propagator OK` if the module imports. Then VertexHead runs ACTS through
sPHENIX's geometry with `acts_image="acts-material-worker-48.2.0.sif"`, `acts_geometry` (exported TGeo file),
`acts_geometry_config` (TGeo JSON) and `acts_material_map` (material JSON); `acts_energy_loss=False` keeps
the geometry and material but adds no energy back. Tracks are propagated from the innermost hit in their real
(backward) direction, so energy lost on the way in is added back to q/p. Multiple scattering only inflates a
covariance, and none is propagated yet, so it does not change the parameters. Check against VertexHead's
own perigee (temporary script; paths as the container sees them: for files outside your home directory,
e.g. on /zfs0, `export SINGULARITY_BIND=/zfs0` first, which every singularity call, the worker's included,
picks up):

```bash
python scripts/acts/check_vertexhead_acts.py --csv vertex_per_point_data.csv \
    --image acts-material-worker-48.2.0.sif --geometry sphenix_ana559_tgeo.root \
    --geometry_config tgeo-sphenix-mms-actsv45.0.0.json --material_map sphenix-mm-acts-material-actsv45.0.0.json
```

Without `--material_map` (geometry only), the result must match the constant-field run without geometry.

### Covariance-weighted vertexing

`VertexHead(vertex_weighting="covariance")` weights each track by the inverse variance of its d0 and z0
instead of its hit count. The covariance is the hit fit's, from per-hit errors `hit_sigma_rphi` and
`hit_sigma_z` (no material); with `use_acts` it is set at the innermost hit and ACTS transports it to the
beam spot, adding multiple scattering where the material map has material. The worker then also exchanges
covariances (`"covariance": true` in its config). `chi_square` becomes a chi-square, so the PV chi2/ndf
shows whether the errors are right (about 1). With `use_acts` this needs the geometry (`acts_geometry`,
`acts_geometry_config`, worker image): ACTS 48.2's Python `createCurvilinear` rejects a start covariance, so
covariances go through the material module; without a material map that is the constant-field result.

`SINGULARITY_BIND` also applies to `singularity build`, where the image has no such directory to mount:
build with it cleared (`SINGULARITY_BIND= JOBS=8 singularity build ...`).

`adaptive_fit=True` (with covariance weighting) adds ACTS's adaptive vertex fit: tracks are down-weighted by
their compatibility chi2 with ACTS's annealing (AnnealingUtility: cut-off 9, temperatures 64 to 1) and the
AdaptiveMultiVertexFitter's stopping rules, all exposed as VertexHead arguments. `track_error_scale`
inflates the track errors where they are known to be too small (PV chi2/ndf above 1).

All variants on the same events (own, ACTS, ACTS through the geometry, with material, with material but no
energy loss; each with hit-count and covariance weighting, and the adaptive fit per `--adaptive_error_scales`
value), PV only and SV default (temporary script):

```bash
python scripts/acts/compare_vertexing.py --csv vertex_per_point_data.csv \
    --image scripts/acts/acts-material-worker-48.2.0.sif --geometry sphenix_ana559_tgeo.root \
    --geometry_config tgeo-sphenix-mms-actsv45.0.0.json --material_map sphenix-mm-acts-material-actsv45.0.0.json
```

## Reference check: the experimental torch maths against ACTS

Three steps, from the repository root: generate test tracks and MomentumHead fits in the training
environment, let ACTS compute their perigee parameters in the container, compare in the training
environment (ACTS and the torch maths against the exact answer and each other, MomentumHead's fit
quality, and the time per track of each method):

```bash
python scripts/acts/compare_acts.py make states.npz
```

```bash
singularity exec acts-ref-48.2.0.sif acts-python scripts/acts/acts_reference.py states.npz acts_out.npz
```

```bash
python scripts/acts/compare_acts.py compare states.npz acts_out.npz
```

Add `--device cuda` to `make` (MomentumHead) and `compare` (the torch maths) to time them on the GPU.

The container has Python 3.12 and NumPy only (no torch): exchange data with the training environment
through files (e.g. `.npz`).
