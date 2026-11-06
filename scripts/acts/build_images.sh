#!/usr/bin/env bash
# Builds the ACTS setup in this directory: the image acts-base-48.2.0.sif (ROOT, HepMC3, ACTS; slow, only if missing
# or with --rebuild-base) and, inside it, the perigee propagator module into perigee_propagator/build/install
# (about a minute; acts_worker.py loads it from there). Needs singularity (or apptainer); the image build needs
# --fakeroot. Usage, from anywhere:
#   JOBS=8 scripts/acts/build_images.sh [--rebuild-base]
set -euo pipefail
cd "$(dirname "$0")"
SINGULARITY=$(command -v singularity || command -v apptainer)
# a bind path (e.g. exported for VertexHead's worker) would be mounted into the image being built and fail
unset SINGULARITY_BIND APPTAINER_BIND
export JOBS=${JOBS:-8}

if [[ ! -f acts-base-48.2.0.sif || "${1:-}" == "--rebuild-base" ]]; then
    "$SINGULARITY" build --force --fakeroot acts-base-48.2.0.sif acts-base.def
fi

# the module, against the image's ACTS and the pybind11 ACTS's bindings were built with (one shared type registry)
"$SINGULARITY" exec --bind "$(pwd)" acts-base-48.2.0.sif bash -euo pipefail -c '
    export LD_LIBRARY_PATH=/opt/root/lib:/opt/hepmc3/lib:/opt/hepmc3/lib64:${LD_LIBRARY_PATH:-}
    PYBIND11_SOURCE_DIR=$(dirname "$(find /opt/acts-build /opt/acts-src -path "*pybind11*" -name pybind11Common.cmake -print -quit)")
    PYBIND11_SOURCE_DIR=$(cd "$PYBIND11_SOURCE_DIR/.." && pwd)
    cmake -S perigee_propagator -B perigee_propagator/build -DCMAKE_PREFIX_PATH="/opt/acts;/opt/root" \
        -DPYBIND11_SOURCE_DIR="$PYBIND11_SOURCE_DIR" -DACTS_LIBRARY_DIR=/opt/acts-build/lib \
        -DCMAKE_INSTALL_PREFIX="$(pwd)/perigee_propagator/build/install"
    cmake --build perigee_propagator/build -j "$JOBS" || cmake --build perigee_propagator/build -j 1
    cmake --install perigee_propagator/build > /dev/null
    PYTHONPATH=perigee_propagator/build/install acts-python -c \
        "import acts, acts_perigee_propagator; print(\"perigee propagator OK:\", acts_perigee_propagator.__file__)"
'
