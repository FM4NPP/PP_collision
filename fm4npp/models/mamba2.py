# model.py
# Minimal, dependency-light Mamba2 that runs in plain PyTorch.
# - Falls back when fused Triton kernels / distributed layers are unavailable
# - Implements a pure-PyTorch "manual path" (streaming-friendly EMA SSM) so it works anywhere
# - Keeps the same class name, args, and forward signature for easy swap-in

import math
import warnings
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# ---------------------------
# Optional deps: guard imports
# ---------------------------
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except Exception:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except Exception:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except Exception:
    selective_state_update = None

try:
    # Fused RMSNorm with gating
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
except Exception:
    RMSNormGated = None

# Fused kernels (we'll just detect presence; we won't require them)
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
except Exception:
    mamba_chunk_scan_combined = None
    mamba_split_conv1d_scan_combined = None

# TP utils (fallback to local ops)
try:
    from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
except Exception:
    ColumnParallelLinear = None
    RowParallelLinear = None

try:
    from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
except Exception:
    all_reduce = None
    reduce_scatter = None

# HF hub mixin (optional)
try:
    from huggingface_hub import PyTorchModelHubMixin
except Exception:
    class PyTorchModelHubMixin:
        pass


# ---------------------------
# Lightweight fallbacks
# ---------------------------

class _IdentityReduce:
    def __call__(self, x, *_, **__):
        return x

_ID_REDUCE = _IdentityReduce()

def _maybe_reduce(x, process_group=None, sequence_parallel=False):
    # If TP ops exist, the original code will call them.
    # In this simplified environment, do nothing.
    return x

FM4NPP_ALLOW_FALLBACK = "FM4NPP_ALLOW_FALLBACK"
_fallback_warned = False


def _require_kernels_or_optin():
    """Refuse to run the pure-PyTorch path unless the caller has opted in.

    History: this fallback silently replaced two kernels with functions that are not
    equivalent -- an ungated RMSNorm applied to the SSM input, and an EMA in place of the
    SSD scan that discarded B and C. Released checkpoints loaded into it with
    strict=True, verify_repro scored 22/22, and two independently built trees agreed
    bit-for-bit, because none of those observe forward semantics. The only symptom was a
    number ~0.09 ARI lower than the paper, which looked like a research result rather
    than a bug. A head trained elsewhere scored 0.054 here against 0.948 in its own log.

    Both errors are fixed, and the fallback now tracks the kernels. It is still slower by
    orders of magnitude (a Python loop over sequence positions) and still unverified
    against the kernels on any given machine, so it must be asked for explicitly.
    """
    global _fallback_warned
    if os.environ.get(FM4NPP_ALLOW_FALLBACK, "") not in ("1", "true", "TRUE", "yes"):
        raise ImportError(
            "mamba-ssm is not installed, so fm4npp would run its pure-PyTorch fallback.\n"
            "That path has silently produced wrong features before -- see the docstring of\n"
            "_require_kernels_or_optin in fm4npp/models/mamba2.py.\n\n"
            "  pip install mamba-ssm causal-conv1d\n\n"
            "If the kernels cannot be built on this machine (e.g. aarch64), opt in with\n"
            "  export FM4NPP_ALLOW_FALLBACK=1\n"
            "and validate first:  python scripts/check_kernel_equivalence.py"
        )
    if not _fallback_warned:
        _fallback_warned = True
        warnings.warn(
            "fm4npp: running the pure-PyTorch Mamba2 fallback (mamba-ssm not installed). "
            "It is corrected but unverified on this machine and far slower. "
            "Validate with scripts/check_kernel_equivalence.py before quoting any number.",
            RuntimeWarning, stacklevel=2)


class _PlainRMSNorm(nn.Module):
    """Gated RMSNorm, matching mamba_ssm.ops.triton.layernorm_gated.RMSNorm.

    The previous fallback dropped the gate and normalised over the full channel
    axis, which is a DIFFERENT FUNCTION from the kernel the pretrained weights
    were trained with -- so a checkpoint would load cleanly and then produce
    features the trained head has never seen.

    The real op, with norm_before_gate=False (the setting this repo uses):

        x = x * silu(z);  out = rmsnorm(x, over group_size) * weight

    and with norm_before_gate=True the gate is applied after the norm instead.
    group_size restricts the reduction to d_ssm // ngroups channels, so a single
    full-width mean is also wrong whenever ngroups > 1.
    """
    def __init__(self, dim, eps=1e-5, norm_before_gate=False, group_size=None, **_):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size
        self.weight = nn.Parameter(torch.ones(dim))

    def _rms(self, x):
        if self.group_size is None or self.group_size == x.shape[-1]:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps)
        shp = x.shape
        xg = x.reshape(*shp[:-1], shp[-1] // self.group_size, self.group_size)
        var = xg.pow(2).mean(dim=-1, keepdim=True)
        return (xg * torch.rsqrt(var + self.eps)).reshape(shp)

    def forward(self, x, z=None):
        dt = x.dtype
        x = x.float()
        if z is not None:
            z = z.float()
            if self.norm_before_gate:
                return (self._rms(x) * self.weight.float() * F.silu(z)).to(dt)
            x = x * F.silu(z)
        return (self._rms(x) * self.weight.float()).to(dt)

def _get_rmsnorm(d_ssm, eps=1e-5, norm_before_gate=False, group_size=None, device=None, dtype=None):
    if RMSNormGated is not None:
        # Use the fused/gated variant if available
        return RMSNormGated(d_ssm, eps=eps, norm_before_gate=norm_before_gate, group_size=group_size,
                            device=device, dtype=dtype)
    # Otherwise, the gated equivalent implemented above.
    return _PlainRMSNorm(d_ssm, eps=eps, norm_before_gate=norm_before_gate,
                         group_size=group_size).to(device=device, dtype=dtype)

class _MaybeColumnParallelLinear(nn.Linear):
    """Fallback to plain Linear if TP not available or process_group is None."""
    def __init__(self, in_features, out_features, bias=True, process_group=None, sequence_parallel=False, **factory_kwargs):
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)

class _MaybeRowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, process_group=None, sequence_parallel=False, **factory_kwargs):
        super().__init__(in_features, out_features, bias=bias, **factory_kwargs)


# ---------------------------
# Mamba2 Block
# ---------------------------

class Mamba2(nn.Module, PyTorchModelHubMixin):
    """
    Dependency-light Mamba2:
    - Splits out B, C from `in_proj`, then reassembles for a manual (PyTorch) path.
    - If fused kernels exist, you *can* turn them on with use_mem_eff_path=True.
    - Otherwise, we run a plain-PyTorch EMA-based SSM that preserves shapes/APIs.
    """

    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, only apply SSM on that many dims; the rest = MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel

        # If no TP group is provided OR TP layers not available, act as single shard
        self.world_size = 1 if process_group is None else getattr(process_group, "size", lambda: 1)()
        self.local_rank = 0 if process_group is None else getattr(process_group, "rank", lambda: 0)()

        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model

        self.headdim = headdim
        # actual SSM dimension
        if d_ssm is None:
            self.d_ssm = self.d_inner
        else:
            self.d_ssm = d_ssm // self.world_size

        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim

        self.D_has_hdim = D_has_hdim
        self.rmsnorm_enabled = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        # Only enable fused path if kernel is present
        self.use_mem_eff_path = bool(use_mem_eff_path and (mamba_split_conv1d_scan_combined is not None))
        if not self.use_mem_eff_path:
            _require_kernels_or_optin()
        self.layer_idx = layer_idx

        # (z, x, dt) from in_proj => shape (2*d_inner + nheads)
        d_in_proj = (2 * self.d_inner) + self.nheads
        if (self.process_group is None) or (ColumnParallelLinear is None):
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(
                self.d_model, d_in_proj * self.world_size,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs
            )

        # Separate B, C => (d_model -> ngroups*d_state)
        self.lin_B = nn.Linear(self.d_model, self.ngroups * self.d_state, bias=False, **factory_kwargs)
        self.lin_C = nn.Linear(self.d_model, self.ngroups * self.d_state, bias=False, **factory_kwargs)

        # depthwise conv over [x_ssm, B, C]
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # dt bias (per head)
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # A (per head), later used as negative rate
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D skip (per head or per (head, headdim))
        D_shape = (self.d_ssm if D_has_hdim else self.nheads,)
        self.D = nn.Parameter(torch.ones(*D_shape, device=device, dtype=dtype if dtype else torch.float32))
        self.D._no_weight_decay = True

        # Normalization (SSM channels)
        if self.rmsnorm_enabled:
            self.norm = _get_rmsnorm(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // self.ngroups,
                device=device,
                dtype=dtype,
            )
        else:
            self.norm = None

        # Output projection
        if (self.process_group is None) or (RowParallelLinear is None):
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(
                self.d_inner * self.world_size, self.d_model,
                bias=bias,
                process_group=self.process_group,
                sequence_parallel=self.sequence_parallel,
                **factory_kwargs
            )

    # ---------------------------
    # Pure-PyTorch manual SSM path
    # ---------------------------

    def _manual_ssm(self, x_ssm, dt_slice, B_ssm, C_ssm):
        """Mamba2 SSD recurrence -- the reference math of mamba_chunk_scan_combined.

            s[t] = exp(A*dt[t]) * s[t-1] + dt[t] * (x[t] outer B[t])
            y[t] = C[t] . s[t] + D * x[t]

        The previous stand-in was, by its own description, "a simplified EMA SSM":
        it dropped B and C entirely and collapsed the state from rank d_state to a
        scalar per channel. That is a different model, not an approximation -- and
        because it touches no parameter shapes, a pretrained checkpoint loads into
        it without complaint and then produces features it was never trained to
        produce.

        x_ssm    (B, L, d_ssm)          dt_slice (B, L, nheads)
        B_ssm    (B, L, ngroups*d_state)  C_ssm  (B, L, ngroups*d_state)
        returns  (B, L, d_ssm)
        """
        Bsz, L, Dssm = x_ssm.shape
        H, P, G, N = self.nheads, self.headdim, self.ngroups, self.d_state
        assert Dssm == H * P
        assert H % G == 0, f'nheads {H} not divisible by ngroups {G}'

        A = -torch.exp(self.A_log.float())                       # (H,)
        dt = F.softplus(dt_slice.float() + self.dt_bias.float())  # (B,L,H)
        lo, hi = self.dt_limit
        if (lo != 0.0) or (hi != float('inf')):
            dt = torch.clamp(dt, min=lo, max=hi)

        x = rearrange(x_ssm.float(), 'b l (h p) -> b l h p', h=H, p=P)
        Bm = rearrange(B_ssm.float(), 'b l (g n) -> b l g n', g=G, n=N)
        Cm = rearrange(C_ssm.float(), 'b l (g n) -> b l g n', g=G, n=N)
        # each head reads the group its index falls in
        rep = H // G
        Bm = Bm.repeat_interleave(rep, dim=2)                    # (B,L,H,N)
        Cm = Cm.repeat_interleave(rep, dim=2)

        dA = torch.exp(dt * A.view(1, 1, H))                     # (B,L,H)
        if self.D_has_hdim:
            Dp = rearrange(self.D.float(), '(h p) -> 1 h p', h=H, p=P)
        else:
            Dp = self.D.float().view(1, H, 1)

        s = x.new_zeros(Bsz, H, P, N)
        outs = []
        for t in range(L):
            xt, bt, ct = x[:, t], Bm[:, t], Cm[:, t]             # (B,H,P) (B,H,N) (B,H,N)
            dt_t = dt[:, t].unsqueeze(-1).unsqueeze(-1)          # (B,H,1,1)
            s = dA[:, t].unsqueeze(-1).unsqueeze(-1) * s + dt_t * (xt.unsqueeze(-1) * bt.unsqueeze(-2))
            outs.append(torch.einsum('bhpn,bhn->bhp', s, ct) + Dp * xt)
        y = torch.stack(outs, dim=1)                             # (B,L,H,P)
        return rearrange(y, 'b l h p -> b l (h p)').to(x_ssm.dtype)

    # ---------------------------

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (B, L, d_model) or flattened (B*L, d_model) with seqlen provided
        returns: (B*L, d_model) (matches your original)
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, _ = u.shape
            is_packed = False
        else:
            batch_seqlen, _ = u.shape
            batch = batch_seqlen // seqlen
            is_packed = True

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if getattr(inference_params, "seqlen_offset", 0) > 0:
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        # Project to (z, x, dt) and to (B, C)
        zxdtemp = self.in_proj(u)             # (B,L, 2*d_inner + H) or (B*L, ...)
        B_slice = self.lin_B(u)               # (B,L, g*d_state)
        C_slice = self.lin_C(u)

        if is_packed:
            zxdtemp = rearrange(zxdtemp, "(b l) d -> b l d", l=seqlen)
            B_slice  = rearrange(B_slice,  "(b l) d -> b l d", l=seqlen)
            C_slice  = rearrange(C_slice,  "(b l) d -> b l d", l=seqlen)

        # Split z, x, dt
        d_mlp = (zxdtemp.shape[-1] - self.nheads) // 2
        z0, x0, dt_slice = torch.split(zxdtemp, [d_mlp, d_mlp, self.nheads], dim=-1)

        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path and (inference_params is None):
            # Fused Triton path: pack [z, x, B, C, dt] into one tensor and call the kernel.
            zxbcdt_fused = torch.cat([z0, x0, B_slice, C_slice, dt_slice], dim=-1)

            try:
                from fm4npp.models.lora import LoRALinear
                out_proj_has_lora = isinstance(self.out_proj, LoRALinear)
            except ImportError:
                out_proj_has_lora = False

            out = mamba_split_conv1d_scan_combined(
                zxbcdt_fused,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                return_final_states=False,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm_enabled else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm_enabled else 1e-6,
                outproj_weight=None if out_proj_has_lora else self.out_proj.weight,
                outproj_bias=None if out_proj_has_lora else self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs
            )
            if out_proj_has_lora:
                out = self.out_proj(out)
            if is_packed:
                out = rearrange(out, "b l d -> (b l) d")
        else:
            # Pure-PyTorch fallback: used when Triton is unavailable.
            x_ssm = x0[..., :self.d_ssm]  # (B, L, d_ssm)

            # Depthwise conv over [x_ssm, B, C]
            conv_in = torch.cat([x_ssm, B_slice, C_slice], dim=-1)
            conv_in_chfirst = rearrange(conv_in, "b l c -> b c l")
            conv_out = self.conv1d(conv_in_chfirst)
            conv_out = conv_out[..., :seqlen]
            conv_out = rearrange(conv_out, "b c l -> b l c")

            ssm_in = conv_out[..., :self.d_ssm]
            gn = self.ngroups * self.d_state
            B_conv = conv_out[..., self.d_ssm:self.d_ssm + gn]
            C_conv = conv_out[..., self.d_ssm + gn:self.d_ssm + 2 * gn]
            # the fused kernel applies the activation to x, B and C alike
            ssm_in, B_conv, C_conv = self.act(ssm_in), self.act(B_conv), self.act(C_conv)

            # The norm belongs on the SSM OUTPUT and is gated by z, not on the
            # input and not ungated. mamba_split_conv1d_scan_combined applies
            # rmsnorm(y * silu(z)) after the scan; doing it here on ssm_in with a
            # plain norm, then gating with sigmoid(z) instead of silu(z), is three
            # separate departures from the trained computation.
            y_ssm = self._manual_ssm(ssm_in, dt_slice, B_conv, C_conv)  # (B, L, d_ssm)
            if self.rmsnorm_enabled and (self.norm is not None):
                y = self.norm(y_ssm, z0)
            else:
                y = y_ssm * F.silu(z0)

            if is_packed:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)

        out = _maybe_reduce(out, self.process_group, self.sequence_parallel)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """
        Single-token decode path (minimal, stateless fallback).
        For portability, we run the same manual path on a length-1 sequence and
        leave states untouched. This preserves the API without requiring kernels.
        """
        if hidden_states.dim() == 3:
            assert hidden_states.shape[1] == 1, "step() only supports (B,1,D)"
            u = hidden_states
        else:
            # (B, D) -> (B,1,D)
            u = hidden_states.unsqueeze(1)

        # Forward with seqlen=1
        out = self.forward(u)
        # Re-add token dim
        if out.dim() == 2:
            out = out.unsqueeze(1)
        return out, conv_state, ssm_state

    # ---------------------------
    # Caching (kept for API parity)
    # ---------------------------

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0],
            device=device, dtype=conv_dtype
        ).transpose(1, 2)  # (B, C, K)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state,
            device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None, "layer_idx must be set to use cache"
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state, ssm_state = self.allocate_inference_cache(batch_size, 0)
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
