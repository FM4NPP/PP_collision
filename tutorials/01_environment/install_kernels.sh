#!/bin/bash
# Install the Mamba2 CUDA/Triton kernels, without a 40-minute compile.
#
#     source common/paths.sh && source $FM4NPP_WORK/.venv/bin/activate
#     bash tutorials/01_environment/install_kernels.sh
#
# WHY NOT JUST `pip install mamba-ssm causal-conv1d`
#
# Because PyPI carries source distributions only -- zero wheels -- so that command always
# compiles CUDA extensions from source. Three ways it goes wrong:
#
#   1. Build isolation. setup.py imports torch at build time, and pip's isolated build
#      environment does not have it. Without --no-build-isolation the build fails before
#      it starts.
#   2. Time and memory. A source build is 20-60 minutes and enough parallel nvcc to get
#      killed on a shared login node.
#   3. Dependencies. mamba-ssm 2.3.2 declares tilelang, quack-kernels, apache-tvm-ffi and
#      triton>=3.5, which will fight whatever torch the venv already has.
#
# The upstream projects publish prebuilt wheels on their GitHub releases, keyed by
# (CUDA major, torch minor, C++11 ABI, Python). This script reads those four values off
# the installed torch and fetches the matching wheel. mamba-ssm is pinned to 2.2.4, which
# declares no dependencies at all and covers torch 2.1-2.6.
#
# If nothing matches, it says so and tells you what to do rather than starting a build
# you did not ask for.
set -uo pipefail

MAMBA_VER=2.2.4
CONV_VER=1.5.0.post8

python - <<'PY'
import sys, torch
cu = torch.version.cuda
if cu is None:
    sys.exit("torch reports no CUDA build; these kernels are CUDA-only.")
print(f"  python      {sys.version_info.major}.{sys.version_info.minor}")
print(f"  torch       {torch.__version__}")
print(f"  cuda        {cu}")
print(f"  cxx11 abi   {torch._C._GLIBCXX_USE_CXX11_ABI}")
PY
[ $? -ne 0 ] && exit 1

read -r PYTAG TORCHMM CUMAJ ABI <<<"$(python - <<'PY'
import sys, torch
print(f"cp{sys.version_info.major}{sys.version_info.minor}",
      ".".join(torch.__version__.split("+")[0].split(".")[:2]),
      "cu12" if int(torch.version.cuda.split(".")[0]) >= 12 else "cu11",
      "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")
PY
)"

MB="https://github.com/state-spaces/mamba/releases/download/v${MAMBA_VER}/mamba_ssm-${MAMBA_VER}+${CUMAJ}torch${TORCHMM}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
CC="https://github.com/Dao-AILab/causal-conv1d/releases/download/v${CONV_VER}/causal_conv1d-${CONV_VER}+${CUMAJ}torch${TORCHMM}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"

echo
echo "  wheel: $(basename "$MB")"

ok=1
for url in "$CC" "$MB"; do
    if ! curl -sfIL "$url" >/dev/null 2>&1; then
        echo "  !! no prebuilt wheel published for this combination:"
        echo "     $(basename "$url")"
        ok=0
    fi
done

if [ $ok -eq 1 ]; then
    # --no-deps: 2.2.4 declares none, and this keeps a newer resolver from pulling
    # tilelang/quack-kernels in on top of a torch that is already working.
    pip install --no-deps "$CC" "$MB" || exit 1
else
    cat <<EOF

  No matching wheel. Two options, in order of preference:

  1. Move torch to a version that has one. mamba-ssm ${MAMBA_VER} publishes wheels for
     torch 2.1 through 2.6 on cp39-cp313, cu11 and cu12. Check the list at
       https://github.com/state-spaces/mamba/releases/tag/v${MAMBA_VER}

  2. Build from source, deliberately and with the brakes on:
       module load cudatoolkit          # nvcc must be on PATH
       pip install ninja packaging setuptools wheel
       MAX_JOBS=4 pip install --no-build-isolation causal-conv1d==${CONV_VER}
       MAX_JOBS=4 pip install --no-build-isolation mamba-ssm==${MAMBA_VER}
     Expect 20-60 minutes. Do it in an salloc session, not on a login node.

EOF
    exit 1
fi

echo
echo "  verifying..."
python - <<'PY'
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # noqa
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm  # noqa
    print("  the two ops fm4npp needs import cleanly")
except Exception as e:
    raise SystemExit(f"  installed, but the import fm4npp needs still fails: {e!r}")
PY
[ $? -ne 0 ] && exit 1

echo
echo "  now confirm the fallback agrees with them:"
echo "      python scripts/check_kernel_equivalence.py"
echo "      python scripts/check_kernel_equivalence.py --d_model 1536 --d_state 96"
