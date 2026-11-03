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
