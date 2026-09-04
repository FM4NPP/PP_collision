#!/bin/bash -l
# Module 01 -- environment setup.
#
#   INPUT   a Perlmutter account (or any machine with a CUDA GPU)
#   OUTPUT  $FM4NPP_WORK/.venv, the PP_collision clone, and uv on PATH
#
# Run this on a LOGIN NODE. It needs network access (Zenodo, HuggingFace, GitHub,
# and one git-only pip dependency), which compute nodes do not have.
#
# Nothing here compiles CUDA code. See common/fm4npp_naming.md section 6 for why
# mamba-ssm is not needed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/paths.sh"

echo "==> work dir: $FM4NPP_WORK"
mkdir -p "$FM4NPP_WORK"

# ---------------------------------------------------------------- modules
# NERSC module names drift. If `module load` fails, run `module avail python`
# and `module avail cudatoolkit` and substitute -- then please open an issue
# so we can fix this script.
if command -v module &>/dev/null; then
    echo "==> loading modules"
    module load python || echo "!! 'module load python' failed; using system python3"
    # cudatoolkit only affects nvcc visibility. We install prebuilt wheels that
    # bundle their own CUDA runtime, so this is not fatal if it is missing.
    module load cudatoolkit || echo "!! 'module load cudatoolkit' failed; continuing"
else
    echo "==> no 'module' command; assuming a non-NERSC machine"
fi

# ---------------------------------------------------------------- uv
if ! command -v uv &>/dev/null; then
    echo "==> installing uv to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "==> uv $(uv --version)"

# ---------------------------------------------------------------- the code
if [ ! -d "$FM4NPP_ROOT/.git" ]; then
    echo "==> cloning PP_collision (branch downstream-reproducibility)"
    # This branch matters. The downstream code on main does not run -- it references
    # an undefined Embedder, an undefined downstream_dropout, and builds the wrong
    # head architecture. Do not substitute main.
    git clone --branch downstream-reproducibility \
        https://github.com/FM4NPP/PP_collision.git "$FM4NPP_ROOT"
else
    echo "==> PP_collision already cloned at $FM4NPP_ROOT"
fi

BRANCH=$(git -C "$FM4NPP_ROOT" rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "downstream-reproducibility" ] || {
    echo "!! WARNING: on branch '$BRANCH', expected downstream-reproducibility"; }

# ---------------------------------------------------------------- venv
echo "==> creating venv at $FM4NPP_WORK/.venv"
cd "$FM4NPP_WORK"
uv venv .venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> installing dependencies (a few minutes; torch is ~2.5 GB)"
uv pip install -r "$HERE/requirements-tutorial.txt"

# ---------------------------------------------------------------- done
cat <<EOF

==> environment ready.

Activate it in every new shell with:

    source $HERE/../common/paths.sh
    source \$FM4NPP_WORK/.venv/bin/activate

Next:
    bash $HERE/fetch_data.sh        # ~1 GB of Zenodo + 2.2 GB of checkpoints
    python $HERE/verify_install.py  # confirms all of the above
EOF
