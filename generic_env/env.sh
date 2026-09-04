# Source this before using generic_env.
#
#     source env.sh
#
# FM4NPP_ALLOW_FALLBACK is set deliberately. Without mamba-ssm, fm4npp/models/mamba2.py
# refuses to build a model rather than quietly running its pure-PyTorch path -- because that
# path used to compute a different model and cost ~0.09 ARI with no visible symptom. It is
# corrected now, but it has never been compared against the real kernels on any machine, so
# opting in has to be an explicit act. That is what this line is.
#
# To upgrade to the real kernels (x86 + CUDA only):
#     bash ../tutorials/01_environment/install_kernels.sh
#     python ../scripts/check_kernel_equivalence.py
export FM4NPP_ALLOW_FALLBACK=1

_here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export FM4NPP_ROOT="$( dirname "$_here" )"
export PYTHONPATH="$FM4NPP_ROOT:$FM4NPP_ROOT/train/downstream${PYTHONPATH:+:$PYTHONPATH}"
export FM4NPP_CKPT="${FM4NPP_CKPT:-$_here/checkpoints/pp_nerf_m1_k30.ckpt}"
unset _here
echo "generic_env ready.  backbone: \$FM4NPP_CKPT -> $FM4NPP_CKPT"
