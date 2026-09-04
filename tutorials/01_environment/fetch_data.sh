#!/bin/bash -l
# Module 01 -- data and checkpoints.
#
#   INPUT   network access
#   OUTPUT  $FM4NPP_PRETRAIN_ROOT   RaggedMmap, both splits, for module 03
#           $FM4NPP_EVAL_ROOT       RaggedMmap, test split, for module 04
#           $FM4NPP_CKPT/*.ckpt     paper m3 (59 MB) and m6 (2.1 GB)
#
# Run on a LOGIN NODE (needs network).
#
# The Zenodo archive is 118.5 GB. We need under 1 GB of it: the labeled splits.
# Zenodo honours HTTP range requests, so fetch_labeled_data.py reads the zip's
# central directory and pulls only the members it wants. ~12 minutes instead of ~26 hours.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../common/paths.sh"

command -v python >/dev/null || { echo "activate the venv first"; exit 1; }
python -c "import remotezip" 2>/dev/null || { echo "run setup_perlmutter.sh first"; exit 1; }

mkdir -p "$FM4NPP_ZENODO" "$FM4NPP_DATA" "$FM4NPP_CKPT"

# ------------------------------------------------------ 1. labeled .npz from Zenodo
if [ ! -d "$FM4NPP_ZENODO/labeled/train" ]; then
    echo "==> fetching labeled splits from Zenodo (~1 GB of a 118.5 GB archive)"
    python "$FM4NPP_ROOT/scripts/fetch_labeled_data.py" \
        --out "$FM4NPP_ZENODO" --splits train,validation,test
else
    echo "==> Zenodo data already present"
fi

# ------------------------------------------------------ 2. .npz -> RaggedMmap
#
# Two roots, because the two use cases want different things:
#
#   pretrain_root/  needs BOTH splits in ONE directory, because get_data_loader()
#                   hardcodes split='pretrain' for train and split='test' for val
#                   against a single data_root. The split is a SUFFIX on each array
#                   directory, not a subdirectory -- see common/fm4npp_naming.md section 3.
#
#   eval_root/      holds the held-out test events for module 04.
#
if [ ! -d "$FM4NPP_PRETRAIN_ROOT/features_pretrain" ]; then
    echo "==> building pretrain root (train -> 'pretrain', validation -> 'test')"
    python "$FM4NPP_ROOT/scripts/prepare_data.py" \
        --in_dir "$FM4NPP_ZENODO/labeled/train" \
        --out "$FM4NPP_PRETRAIN_ROOT" --split pretrain --end 20000
    python "$FM4NPP_ROOT/scripts/prepare_data.py" \
        --in_dir "$FM4NPP_ZENODO/labeled/validation" \
        --out "$FM4NPP_PRETRAIN_ROOT" --split test --end 2000
else
    echo "==> pretrain root already built"
fi

if [ ! -d "$FM4NPP_EVAL_ROOT/features_test" ]; then
    echo "==> building eval root (test -> 'test')"
    python "$FM4NPP_ROOT/scripts/prepare_data.py" \
        --in_dir "$FM4NPP_ZENODO/labeled/test" \
        --out "$FM4NPP_EVAL_ROOT" --split test --end 2000
else
    echo "==> eval root already built"
fi

# ------------------------------------------------------ 3. checkpoints
echo "==> downloading checkpoints from HuggingFace"
python - <<'PY'
import os
from huggingface_hub import hf_hub_download

dest = os.environ['FM4NPP_CKPT']
# Filenames follow the REPO convention: pp_nerf_m1 is the paper's m3,
# pp_nerf_m5 is the paper's m6. See common/fm4npp_naming.md section 1.
for fname, paper in (('pp_nerf_m1_k30.ckpt', 'm3  5.3M'),
                     ('pp_nerf_m5_k30.ckpt', 'm6  175M')):
    if os.path.exists(os.path.join(dest, fname)):
        print(f'  have {fname:24s} (paper {paper})')
        continue
    print(f'  downloading {fname:20s} (paper {paper})')
    p = hf_hub_download(repo_id='FM4NPP/PP_collision', filename=fname,
                        local_dir=dest)
    print(f'    -> {p}')
PY

echo
echo "==> data ready. Now run:  python $HERE/verify_install.py"
