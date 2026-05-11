# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import torch
import torch.nn.functional as F

from fla.utils import input_guard

from .fused_recurrent_gdn2 import fused_recurrent_gdn2_fwd


def _expand_q(q: torch.Tensor, num_householder: int) -> torch.Tensor:
    """Place queries at the LAST position of each Householder group.

    Returns a tensor of shape ``[B, T * num_householder, H, K]`` with zero
    fills at all positions except the last of each group, where ``q`` lives.
    Non-query positions produce zero outputs (since ``o = S^T q`` and
    ``q = 0`` there) that are discarded on extraction.

    Short-circuits to the identity when ``num_householder == 1``.
    """
    if num_householder == 1:
        return q
    B, T, H, K = q.shape
    q_exp = q.new_zeros(B, T * num_householder, H, K)
    q_exp[:, num_householder - 1::num_householder] = q
    return q_exp


def _interleave_gate(g: torch.Tensor, num_householder: int) -> torch.Tensor:
    """Place the activated log-decay at the FIRST position of each group.

    The remaining ``num_householder - 1`` positions get 0 in log-space,
    which is ``exp(0) = 1`` = identity decay. This encodes "decay only
    between tokens, not between Householder steps of the same token".

    Preserves the dtype of ``g`` (unlike the chunk-kernel wrapper, we do
    NOT force fp32 here -- the recurrent kernel casts to fp32 internally
    on load, and most callers will have already produced g in fp32).

    Short-circuits to the identity when ``num_householder == 1``.
    """
    if num_householder == 1:
        return g
    B, T, HV, K = g.shape
    g_int = g.new_zeros(B, T * num_householder, HV, K)
    g_int[:, 0::num_householder] = g
    return g_int


def _extract_output(o_exp: torch.Tensor, num_householder: int) -> torch.Tensor:
    """Gather per-token outputs at every ``num_householder``-th position.

    Queries were placed at the last of each group in ``_expand_q``, so the
    output at positions ``n_h - 1, 2*n_h - 1, ...`` is ``S_{i,n_h}^T q_i``,
    the correct per-token output. Other positions carry zero outputs from
    the zero queries there; we drop them.

    Short-circuits to the identity when ``num_householder == 1``.
    """
    if num_householder == 1:
        return o_exp
    return o_exp[:, num_householder - 1::num_householder].contiguous()


def _apply_gate_activation(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    """Mirror of the USE_GATE_IN_KERNEL=True branch of the recurrent kernel.

    Applied externally in torch ops so that we can run the activation
    BEFORE interleaving. If we ran it after interleaving, the zero-fills
    at identity positions would be transformed into nonzero log-decays
    (``softplus(0) = ln 2 != 0``), corrupting the identity-decay semantic.

    Reproduces the kernel's arithmetic exactly:
        * unbounded  : ``g_out = -exp(A_log) * softplus(g + dt_bias)``
        * bounded    : ``g_out = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``

    GVA handling: ``A_log`` is per-head (shape ``[H]``) and ``dt_bias`` is
    per-head-per-channel (shape ``[H*K]``). When the gate has ``HV`` heads
    with ``HV > H``, we broadcast via ``repeat_interleave(HV // H)`` --
    the torch-level equivalent of the kernel's ``i_h = i_hv // (HV // H)``
    indexing trick.

    Runs in fp32 regardless of input dtype, matching the kernel's internal
    ``tl.float32`` accumulation.

    Args:
        g       : raw pre-activation gate, shape ``[B, T, HV, K]``.
        A_log   : per-head log-magnitude, shape ``[H]``.
        dt_bias : per-channel bias, shape ``[H*K]``, optional.
        lower_bound : if set, use the bounded sigmoid variant.

    Returns:
        Activated log-space gate, shape ``[B, T, HV, K]``, dtype fp32.
    """
    B, T, HV, K = g.shape
    H = A_log.shape[0]

    # GVA-aware broadcast of A_log from [H] to [HV].
    if HV != H:
        assert HV % H == 0, (
            f"HV={HV} must be divisible by H={H} to broadcast A_log "
            f"(GVA pattern)."
        )
        A_log_bcast = A_log.repeat_interleave(HV // H)
    else:
        A_log_bcast = A_log

    # exp(A_log), shape [1, 1, HV, 1] for broadcast over [B, T, HV, K].
    A_exp = A_log_bcast.float().exp().view(1, 1, HV, 1)

    x = g.float()
    if dt_bias is not None:
        # dt_bias has layout [H, K] flattened to [H*K]. Reshape, GVA-broadcast
        # to [HV, K], and add with broadcasting over (B, T).
        bias = dt_bias.float().view(H, K)
        if HV != H:
            bias = bias.repeat_interleave(HV // H, dim=0)
        x = x + bias.view(1, 1, HV, K)

    if lower_bound is not None:
        # Bounded variant: matches `lower_bound * tl.sigmoid(exp(b_A) * b_g)`
        # in the kernel. Output range: (lower_bound, 0).
        return lower_bound * torch.sigmoid(A_exp * x)
    else:
        # Standard variant: matches `-exp(b_A) * softplus(b_g)` in the kernel.
        # Output range: (-inf, 0).
        return -A_exp * F.softplus(x)


# ===========================================================================
# Public API
# ===========================================================================

@input_guard
def fused_recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    num_householder: int = 1,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Token-by-token forward for GDN-2-Plus-Channel. Inference-only.

    Same recurrence as ``chunk_gdn2_plus_channel``, driven through the plain
    GDN-2 recurrent kernel on an ``n_h``-fold expanded sequence. Use this
    for fast autoregressive decoding with ``q_len <= 64`` or so; use the
    chunk kernel for training and long-sequence inference.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``. **Not** pre-expanded -- we
            internally place ``q_i`` at the last position of each
            Householder group.
        k (torch.Tensor):
            Keys of shape ``[B, T * num_householder, H, K]``, pre-interleaved
            by the caller as
            ``[k_{1,1}, ..., k_{1,nh}, k_{2,1}, ..., k_{2,nh}, ...]``.
        v (torch.Tensor):
            Values of shape ``[B, T * num_householder, H, V]``, pre-interleaved.
        g (torch.Tensor):
            Decay gate of shape ``[B, T, H, K]``. **Not** pre-expanded;
            interleaving is handled internally (with log-zero fills that
            encode identity decay between Householder steps of the same
            token). If ``use_gate_in_kernel=True``, this is the raw
            pre-activation; otherwise it is the activated log-space decay.
        b (torch.Tensor):
            Channel-wise erase gate, shape ``[B, T * num_householder, H, K]``,
            pre-interleaved.
        w (torch.Tensor):
            Channel-wise write gate, shape ``[B, T * num_householder, H, V]``,
            pre-interleaved.
        num_householder (int):
            Number of Householder steps per token. Default: ``1`` (reduces
            to plain GDN-2, pass-through to ``fused_recurrent_gdn2_fwd``).
        A_log (Optional[torch.Tensor]):
            Gate magnitude parameter, shape ``[H]``. Used when
            ``use_gate_in_kernel=True``.
        dt_bias (Optional[torch.Tensor]):
            Per-channel bias, shape ``[H * K]``. Used when
            ``use_gate_in_kernel=True``.
        scale (Optional[float]):
            Attention scale. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial recurrent state, ``[N, H, K, V]`` fp32 (or
            ``[N, H, V, K]`` when ``transpose_state_layout=True``).
        output_final_state (bool):
            Whether to return the final recurrent state. Default: ``False``.
        use_qk_l2norm_in_kernel (bool):
            L2-normalize q and k inside the kernel. Default: ``False``.
        use_gate_in_kernel (bool):
            If True and ``num_householder > 1``, the activation is applied
            in torch ops here (before interleaving). If True and
            ``num_householder == 1``, it is delegated to the kernel. If
            False, ``g`` is taken as already activated. Default: ``False``.
        lower_bound (Optional[float]):
            Bounded-sigmoid variant of the gate activation (only consulted
            when ``use_gate_in_kernel=True``). See
            ``_apply_gate_activation`` for the formula.
        cu_seqlens (Optional[torch.LongTensor]):
            Packed-sequence boundaries in ORIGINAL (pre-expansion) token
            units. We multiply by ``num_householder`` internally before
            calling the kernel so the kernel resets its state at the right
            positions.
        transpose_state_layout (bool):
            Use the transposed ``[V, K]`` state layout. Default: ``False``.

    Returns:
        o (torch.Tensor):
            Per-token outputs, shape ``[B, T, H, V]``.
        final_state (Optional[torch.Tensor]):
            Final recurrent state, shape ``[N, H, K, V]`` (or
            ``[N, H, V, K]`` with ``transpose_state_layout=True``), if
            ``output_final_state=True``; else ``None``.

    Note:
        Advanced features of the base kernel (``ssm_state_indices`` for
        continuous batching, ``num_accepted_tokens`` for speculative
        decoding) are not exposed here. They would need careful semantic
        handling under ``n_h``-fold expansion (each original "accepted"
        token maps to ``n_h`` kernel positions) and are outside this
        wrapper's scope.
    """
    nh = num_householder
    B, T, H, K = q.shape
    V = v.shape[-1]

    # --------------------- shape validation --------------------------------
    assert k.shape == (B, T * nh, H, K), (
        f"k shape: expected {(B, T * nh, H, K)}, got {tuple(k.shape)}. "
        f"Keys must be pre-interleaved by the caller."
    )
    assert v.shape == (B, T * nh, H, V), (
        f"v shape: expected {(B, T * nh, H, V)}, got {tuple(v.shape)}. "
        f"Values must be pre-interleaved by the caller."
    )
    assert b.shape == (B, T * nh, H, K), (
        f"b shape: expected {(B, T * nh, H, K)}, got {tuple(b.shape)}. "
        f"b is the channel-wise erase gate."
    )
    assert w.shape == (B, T * nh, H, V), (
        f"w shape: expected {(B, T * nh, H, V)}, got {tuple(w.shape)}. "
        f"w is the channel-wise write gate."
    )
    assert g.shape == (B, T, H, K), (
        f"g shape: expected {(B, T, H, K)}, got {tuple(g.shape)}. "
        f"Decay gate must NOT be pre-expanded by num_householder."
    )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                f"when using `cu_seqlens`. Please flatten variable-length "
                f"inputs before processing.",
            )
        if (
            initial_state is not None
            and initial_state.shape[0] != len(cu_seqlens) - 1
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to "
                f"the number of input sequences, i.e., "
                f"{len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, (
            "initial_state must be float32."
        )

    if use_gate_in_kernel and nh > 1:
        assert A_log is not None, (
            "A_log must be provided when use_gate_in_kernel=True."
        )
        g = _apply_gate_activation(g, A_log, dt_bias, lower_bound)
        # Gate is now activated; don't let the kernel do it again.
        use_gate_in_kernel = False

    # --------------------- sequence expansion ------------------------------
    q_exp = _expand_q(q, nh)                                        # [B, T*nh, H, K]
    g_int = _interleave_gate(g, nh)                                 # [B, T*nh, H, K]
    cu_seqlens_exp = cu_seqlens * nh if cu_seqlens is not None else None

    if scale is None:
        scale = K ** -0.5

    o_exp, final_state = fused_recurrent_gdn2_fwd(
        q=q_exp,
        k=k,
        v=v,
        g=g_int,
        b=b,
        w=w,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=False,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens_exp,
        transpose_state_layout=transpose_state_layout,
    )

    # --------------------- extract per-token outputs -----------------------
    o = _extract_output(o_exp, nh)

    return o, final_state


__all__ = [
    "fused_recurrent_gdn2",
]