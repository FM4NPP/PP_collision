# ACTS reference container

Unmodified [ACTS](https://github.com/acts-project/acts) (MPL-2.0) with its Python bindings, in a container,
so it never touches `fm4npp-env`. It is used as a separate program for reference checks of the
package's own (pure-torch, MIT) track maths, not for training. ACTS is run, not copied, so its licence
places no conditions on this repository's code.

## Build (once, by someone in the `docker` group)

From the repository root (the ACTS compile takes a while; set `JOBS` to the cores you can use):

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

The container has Python 3.12 and NumPy only (no torch): exchange data with the training environment
through files (e.g. `.npz`).
