#!/usr/bin/env bash
# Builds the ACTS images in this directory: acts-base-48.2.0.sif (ROOT, HepMC3, ACTS; slow, only if missing or
# with --rebuild-base) and acts-worker-48.2.0.sif (the perigee propagator module; minutes). Needs singularity
# (or apptainer) with --fakeroot. Usage, from anywhere:
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
"$SINGULARITY" build --force --fakeroot acts-worker-48.2.0.sif acts-worker.def
echo "built $(pwd)/acts-worker-48.2.0.sif"
