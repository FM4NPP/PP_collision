# Single source of truth for every path these tutorials use.
#   source common/paths.sh
# Override any of these by exporting it beforehand.

# Where everything lives. On Perlmutter $SCRATCH is set for you; elsewhere it is not.
: "${SCRATCH:=$HOME}"
: "${FM4NPP_WORK:=$SCRATCH/fm4npp_tutorial}"

# The official code, cloned by 01_environment/setup_perlmutter.sh
: "${FM4NPP_ROOT:=$FM4NPP_WORK/PP_collision}"

# Raw Zenodo .npz, and the RaggedMmap roots built from them
: "${FM4NPP_ZENODO:=$FM4NPP_WORK/zenodo}"
: "${FM4NPP_DATA:=$FM4NPP_WORK/data}"

# Pretraining wants ONE root holding both splits (see common/fm4npp_naming.md §3)
: "${FM4NPP_PRETRAIN_ROOT:=$FM4NPP_DATA/pretrain_root}"
# Downstream/feature work reads the test split from its own root
: "${FM4NPP_EVAL_ROOT:=$FM4NPP_DATA/eval_root}"

# HuggingFace checkpoints
: "${FM4NPP_CKPT:=$FM4NPP_WORK/checkpoints}"
: "${FM4NPP_CKPT_M3:=$FM4NPP_CKPT/pp_nerf_m1_k30.ckpt}"   # paper m3, 5.3M
: "${FM4NPP_CKPT_M6:=$FM4NPP_CKPT/pp_nerf_m5_k30.ckpt}"   # paper m6, 175M

# Bin edges + loss stats. MUST be the repo's own stats/ -- see naming.md §5.
: "${FM4NPP_STATS:=$FM4NPP_ROOT/stats}"

# Run outputs
: "${FM4NPP_RUNS:=$FM4NPP_WORK/runs}"

export SCRATCH FM4NPP_WORK FM4NPP_ROOT FM4NPP_ZENODO FM4NPP_DATA \
       FM4NPP_PRETRAIN_ROOT FM4NPP_EVAL_ROOT FM4NPP_CKPT FM4NPP_CKPT_M3 \
       FM4NPP_CKPT_M6 FM4NPP_STATS FM4NPP_RUNS

# The repo is not pip-installable (no setup.py / pyproject.toml), so it has to be on the path.
export PYTHONPATH="$FM4NPP_ROOT${PYTHONPATH:+:$PYTHONPATH}"

fm4npp_show_paths() {
    printf '%-24s %s\n' \
        FM4NPP_WORK "$FM4NPP_WORK" \
        FM4NPP_ROOT "$FM4NPP_ROOT" \
        FM4NPP_ZENODO "$FM4NPP_ZENODO" \
        FM4NPP_PRETRAIN_ROOT "$FM4NPP_PRETRAIN_ROOT" \
        FM4NPP_EVAL_ROOT "$FM4NPP_EVAL_ROOT" \
        FM4NPP_CKPT_M3 "$FM4NPP_CKPT_M3" \
        FM4NPP_CKPT_M6 "$FM4NPP_CKPT_M6" \
        FM4NPP_STATS "$FM4NPP_STATS" \
        FM4NPP_RUNS "$FM4NPP_RUNS"
}
