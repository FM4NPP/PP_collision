#!/bin/bash
#SBATCH -A                                  # <-- your project account; find it with: iris
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:25:00
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -J fm4npp_verify
#SBATCH -o verify_%j.out
#
# Does this branch reproduce the paper on real hardware with real kernels?
#
# Everything in this repository was developed on a machine where mamba-ssm cannot be
# built (aarch64), so every number here came from the pure-PyTorch fallback. That
# fallback was wrong for a long time and is now believed correct, but "believed correct"
# on one machine is not evidence. Perlmutter is x86 + CUDA, the kernels build, and this
# job is the check that closes the loop.
#
# PREREQUISITE -- run once on a LOGIN node, not here. These compile:
#
#     source common/paths.sh && source $FM4NPP_WORK/.venv/bin/activate
#     pip install mamba-ssm causal-conv1d
#
# Submit:  sbatch tutorials/05_downstream_training/perlmutter_verify.sbatch
#
# The job is a gate, not a survey. Any FAIL below means results from this branch should
# not be quoted until it is understood.

set -uo pipefail

module load pytorch/2.6.0 2>/dev/null || echo "!! module load failed; relying on the venv"
source "${FM4NPP_WORK:?set FM4NPP_WORK, or source tutorials/common/paths.sh}/.venv/bin/activate"
cd "${FM4NPP_ROOT:?set FM4NPP_ROOT}"

fail=0
step () { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# ---------------------------------------------------------------- 1. kernels present
step "1/4  are the real kernels installed?"
python - <<'PY'
import sys
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # noqa
    print("  mamba-ssm importable -- the fused path will be used")
except Exception as e:
    print(f"  mamba-ssm NOT importable ({type(e).__name__})")
    print("  This job cannot do what it exists to do. Install on a login node:")
    print("      pip install mamba-ssm causal-conv1d")
    sys.exit(1)
PY
[ $? -ne 0 ] && { echo "STOP: install the kernels first."; exit 1; }

# ------------------------------------------------- 2. fallback vs kernels, numerically
step "2/4  does the pure-PyTorch fallback match the kernels?"
echo "  This is the check whose absence cost ~0.09 ARI. Both widths, since the"
echo "  fallback's old error grew with d_state (16 at m3, 96 at m6)."
for args in "--d_model 256 --d_state 16" "--d_model 1536 --d_state 96"; do
    python scripts/check_kernel_equivalence.py $args || fail=1
done

# ------------------------------------------------------- 3. structural regression suite
step "3/4  regression checks"
python scripts/verify_repro.py . | tail -3 || fail=1

# ---------------------------------------------- 4. the acceptance criterion: transfer
step "4/4  does a head trained by the authors score correctly here?"
echo "  This is the decisive test. The head below was trained on their machine with the"
echo "  real kernels. On a correct stack it scores ~0.95; with the old broken fallback it"
echo "  scored 0.054. There is no ambiguous middle, which is what makes it a good gate."
: "${FM4NPP_HEADS:=$FM4NPP_WORK/heads}"
CKPT="$FM4NPP_HEADS/d9_m5_k30_p20_nerf_tracking_head_d70000_1_seed8_checkpoint.pth"
if [ ! -f "$CKPT" ]; then
    echo "  SKIP: released head not found at $CKPT"
    echo "  Copy it from the authors' downstream_log to \$FM4NPP_HEADS and rerun."
else
    python train/downstream/eval_track_finding.py \
        --yaml_config scripts/configs/mamba_tracking.yaml --config d9_m5_k30_p20 \
        --run_num 1 --seed 8 --eventnumber 300 --eval_batch_size 1 \
        --max_eval_events 300 --checkpoint "$CKPT" \
        --root_dir "$FM4NPP_WORK/verify/" --no-csv
    python - "$FM4NPP_WORK/verify" <<'PY' || fail=1
import glob, os, sys
hits = glob.glob(os.path.join(sys.argv[1], '*eval_tracking_head*seed8.log'))
if not hits:
    print("  FAIL: evaluation produced no metric file"); sys.exit(1)
ari = float(open(sorted(hits)[-1]).read().strip().split('\n')[-1].split('\t')[1])
print(f"\n  ARI = {ari:.4f}   (their log for this checkpoint: 0.9479; paper: 0.9448)")
if ari > 0.90:
    print("  PASS: this branch reproduces the paper with real kernels."); sys.exit(0)
if ari < 0.30:
    print("  FAIL: this is the broken-kernel signature. Do not quote any number from");
    print("        this environment until step 2 passes.")
else:
    print("  FAIL: neither correct (~0.95) nor the known failure (~0.05). Something new.")
sys.exit(1)
PY
fi

step "result"
if [ $fail -eq 0 ]; then
    echo "  ALL CHECKS PASSED"
else
    echo "  SOMETHING FAILED -- see above. Results from this branch are not trustworthy"
    echo "  until it is resolved."
fi
exit $fail
