#!/usr/bin/env python3
"""Check that the pure-PyTorch Mamba2 fallback matches the real mamba-ssm kernels.

WHY THIS EXISTS

The fallback in `fm4npp/models/mamba2.py` engages silently whenever `mamba-ssm` is not
importable. For a long time it computed a *different function* than the kernels the
released checkpoints were trained with:

  1. RMSNorm was applied to the SSM *input*, ungated, with sigmoid(z) where the kernel
     uses silu(z), and without grouping. The kernel computes
     `rmsnorm(y * silu(z)) * weight` on the SSM *output*.
  2. The SSD scan was replaced by an EMA that discarded B and C entirely, collapsing the
     state from rank d_state to a scalar per channel.

Neither error touches a parameter name or shape, so every structural check passed:
`load_state_dict(strict=True)` succeeded, `verify_repro.py` scored 22/22, and two
independently built trees agreed bit-for-bit. The only symptom was that results came out
about 0.09 ARI below the paper -- which reads as a research finding, not a bug.

What finally exposed it was loading a downstream head trained against the real kernels:
it scored 0.054 here against 0.948 in its own training log. This script makes that check
cheap and routine instead of accidental.

WHAT IT DOES

With mamba-ssm installed, builds one Mamba2 block, runs the same input through the fused
path and the fallback path, and compares. Without mamba-ssm installed it cannot compare,
and says so -- that is a skip, not a pass.

USAGE

    python scripts/check_kernel_equivalence.py
    python scripts/check_kernel_equivalence.py --d_model 1536 --d_state 96 --seqlen 512

Exit status is 0 on agreement, 1 on divergence, 2 when the comparison is impossible.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The fallback refuses to construct without this; we are deliberately exercising it.
os.environ.setdefault("FM4NPP_ALLOW_FALLBACK", "1")

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--rtol", type=float, default=2e-2,
                    help="bf16-scale tolerance; the kernels run in reduced precision")
    ap.add_argument("--atol", type=float, default=2e-2)
    args = ap.parse_args()

    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined  # noqa
        have_kernels = True
    except Exception:
        have_kernels = False

    from fm4npp.models.mamba2 import Mamba2

    if not have_kernels:
        print("SKIP: mamba-ssm is not installed, so there is nothing to compare against.")
        print()
        print("  This is NOT a pass. It means this machine can only run the fallback, and")
        print("  the fallback is unvalidated here. Install the kernels to compare:")
        print("      pip install mamba-ssm causal-conv1d")
        print()
        print("  On a machine where they cannot be built (e.g. aarch64), run this script")
        print("  once on a machine where they can, at the same --d_model/--d_state.")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    block = Mamba2(d_model=args.d_model, d_state=args.d_state).to(device).eval()
    u = torch.randn(args.batch, args.seqlen, args.d_model, device=device)

    with torch.no_grad():
        block.use_mem_eff_path = True
        fused = block(u.clone())
        block.use_mem_eff_path = False
        manual = block(u.clone())

    fused32, manual32 = fused.float(), manual.float()
    diff = (fused32 - manual32).abs()
    denom = fused32.abs().max().clamp(min=1e-12)
    print(f"d_model={args.d_model} d_state={args.d_state} "
          f"seqlen={args.seqlen} batch={args.batch} device={device}")
    print(f"  max |fused - fallback|   {diff.max().item():.3e}")
    print(f"  mean |fused - fallback|  {diff.mean().item():.3e}")
    print(f"  relative to signal       {(diff.max() / denom).item():.3e}")

    ok = torch.allclose(fused32, manual32, rtol=args.rtol, atol=args.atol)
    print()
    if ok:
        print(f"PASS: the two paths agree within rtol={args.rtol} atol={args.atol}.")
        return 0
    print(f"FAIL: the two paths DISAGREE beyond rtol={args.rtol} atol={args.atol}.")
    print("  Do not trust any number produced by the fallback on this machine.")
    print("  Install mamba-ssm and rerun, or fix fm4npp/models/mamba2.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
