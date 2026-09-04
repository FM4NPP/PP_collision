#!/bin/bash
# Fetch the smallest released backbone: pp_nerf_m1_k30.ckpt, 59 MB.
#
# NAMING TRAP: this file is the repo's m1 but the PAPER's m3 (width 256, 5.3M). The two
# naming schemes both run m1..m6 and do not agree. Download "m1" expecting the paper's m1
# and you get a model 16x larger than you wanted.
set -euo pipefail
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
mkdir -p "$HERE/checkpoints"
python - "$HERE/checkpoints" <<'PY'
import sys, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id='FM4NPP/PP_collision', filename='pp_nerf_m1_k30.ckpt')
shutil.copy(p, sys.argv[1] + '/pp_nerf_m1_k30.ckpt')
print('  ->', sys.argv[1] + '/pp_nerf_m1_k30.ckpt')
PY
