# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)

from .chunk_kda import fused_kda_gate

from .chunk_gdn2 import chunk_gdn2_fwd, chunk_gdn2_bwd


def _expand_q(q: torch.Tensor, num_householder: int) -> torch.Tensor:
    """
    Place queries at the LAST position of each Householder group.

    The GDN-2+-Channel output o_i = S_{i,n_h}^T q_i is read from the state
    at the final Householder step, so placing q there ensures the output
    reflects all n_h gradient-descent refinements at token i. Non-query
    positions are filled with zero, which contribute nothing to the output
    (since o = S^T q and q = 0 there).

    Args:
        q: Queries, shape [B, T, H, K].
        num_householder: Number of Householder steps per token.

    Returns:
        Expanded queries, shape [B, T * num_householder, H, K]. When
        num_householder == 1, returns the input unchanged.
    """
    if num_householder == 1:
        return q
    B, T, H, K = q.shape
    q_exp = q.new_zeros(B, T * num_householder, H, K)
    q_exp[:, num_householder - 1::num_householder] = q
    return q_exp


def _interleave_gate(g: torch.Tensor, num_householder: int) -> torch.Tensor:
    """
    Interleave the channel-wise decay gate for Householder expansion.

    The activated log-gate g_i is placed at the FIRST position of each
    Householder group, and 0 in log-space -- which is exp(0) = 1, i.e.,
    identity decay -- is placed at the remaining num_householder - 1
    positions. This encodes "decay only between tokens, not between
    Householder steps of the same token."

    The output is fp32 to match the dtype convention of GDN-2's cumsum path:
    the decay cumsum is a path-length-proportional error source and bf16's
    7-bit mantissa is not enough to keep it stable over long sequences.

    Args:
        g: Activated gate in log-space, shape [B, T, H, K].
        num_householder: Number of Householder steps per token.

    Returns:
        Interleaved log-gate, shape [B, T * num_householder, H, K], fp32.
        When num_householder == 1, returns the input unchanged.
    """
    if num_householder == 1:
        return g
    B, T, H, K = g.shape
    g_int = g.new_zeros(B, T, num_householder, H, K, dtype=torch.float32)
    g_int[:, :, 0] = g.float()
    return rearrange(g_int, 'b t n h k -> b (t n) h k').contiguous()


def _extract_output(o_exp: torch.Tensor, num_householder: int) -> torch.Tensor:
    """
    Gather the GDN-2 kernel outputs at the per-token query positions.

    Since queries were placed at the last position of each Householder group
    in forward, the output at every num_householder-th position (specifically
    at positions num_householder - 1, 2*num_householder - 1, ...) is
    S_{i,n_h}^T q_i -- the correct per-token output.

    Args:
        o_exp: Kernel output, shape [B, T * num_householder, H, V].
        num_householder: Number of Householder steps per token.

    Returns:
        Per-token output, shape [B, T, H, V]. When num_householder == 1,
        returns the input unchanged.
    """
    if num_householder == 1:
        return o_exp
    return o_exp[:, num_householder - 1::num_householder].contiguous()


def _extract_dg(
    dg_int: torch.Tensor,
    num_householder: int,
    T: int,
) -> torch.Tensor:
    """
    Gather log-gate gradients at the real-gate position (0) of each group.

    GDN-2's backward applies `chunk_local_cumsum(..., reverse=True)` to the
    interleaved gate gradient. That reverse cumsum naturally accumulates the
    gradients from the num_householder - 1 identity positions back into
    position 0, where the real gate lives. So reading position 0 of each
    group yields the full per-token gate gradient without any extra
    summation.

    Args:
        dg_int: Interleaved gate gradient,
                shape [B, T * num_householder, H, K].
        num_householder: Number of Householder steps per token.
        T: Original (non-expanded) sequence length.

    Returns:
        Per-token gate gradient, shape [B, T, H, K]. When num_householder
        == 1, returns the input unchanged.
    """
    if num_householder == 1:
        return dg_int
    B, _, H, K = dg_int.shape
    return dg_int.reshape(B, T, num_householder, H, K)[:, :, 0].contiguous()


class ChunkGDN2PlusChannelFunction(torch.autograd.Function):
    """
    Autograd wrapper around GDN-2+ (channel-wise b, w) with expansion.

    Forward flow:
        1. Optionally L2-normalize q and k.
        2. Expand q and interleave g for the n_h-long expanded sequence.
        3. Scale cu_seqlens by n_h.
        4. Run chunk_gdn2_fwd on the expanded sequence with
           use_gate_in_kernel=False (gate activation, if any, happens in the
           public API before calling this Function).
        5. Extract outputs at every n_h-th position.

    Backward flow:
        1. Expand do identically to q.
        2. Run chunk_gdn2_bwd on the expanded sequence.
        3. Extract dq at query positions and dg at gate positions
           (dk, dv, db, dw flow back in expanded form directly).
        4. Apply L2-norm VJPs to dq, dk if needed.

    Inputs `k`, `v`, `b`, `w` are already in expanded form
    [B, T*n_h, H, D]; gradients for them flow back in the same expanded
    form and are consumed by the nn.Module's projections directly.
    """

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,               # [B, T,      H, K]
        k: torch.Tensor,               # [B, T*nh,   H, K]   pre-expanded
        v: torch.Tensor,               # [B, T*nh,   H, V]   pre-expanded
        g: torch.Tensor,               # [B, T,      H, K]   activated log-gate
        b: torch.Tensor,               # [B, T*nh,   H, K]   pre-expanded
        w: torch.Tensor,               # [B, T*nh,   H, V]   pre-expanded
        scale: float,
        num_householder: int,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
        safe_gate: bool,
        disable_recompute: bool,
        return_intermediate_states: bool,
        transpose_state_layout: bool,
    ):
        chunk_size = 64
        nh = num_householder
        B, T, H, K = q.shape

        # -------- L2 normalization (save reciprocal norms for backward) -----
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)          # q is non-expanded, [B, T, H, K]
            k, k_rstd = l2norm_fwd(k)          # k is expanded,     [B, T*nh, H, K]

        # -------- Sequence expansion -----------------------------------------
        q_exp = _expand_q(q, nh)               # [B, T*nh, H, K]
        g_int = _interleave_gate(g, nh)        # [B, T*nh, H, K]  (fp32)

        # -------- Scale varlen cu_seqlens by nh ------------------------------
        cu_seqlens_exp = cu_seqlens * nh if cu_seqlens is not None else None
        cu_seqlens_cpu_exp = (
            cu_seqlens_cpu * nh if cu_seqlens_cpu is not None else None
        )
        chunk_indices_exp = (
            prepare_chunk_indices(cu_seqlens_exp, chunk_size)
            if cu_seqlens_exp is not None else None
        )

        # -------- Run GDN-2 on the expanded sequence -------------------------
        # Gate is already activated and interleaved -> use_gate_in_kernel=False
        # so chunk_gdn2_fwd only does the local cumsum (not the activation).
        # safe_gate=False because the safe-gate kernel has assumptions that
        # may not hold for the interleaved (zero-filled) gate.
        (o_exp, final_state, g_cumsum, Aqk, Akk,
         w_wy, u_wy, qg, kg, v_new, h,
         initial_state_out) = chunk_gdn2_fwd(
            q=q_exp,
            k=k,
            v=v,
            g=g_int,
            b=b,
            wg=w,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens_exp,
            cu_seqlens_cpu=cu_seqlens_cpu_exp,
            chunk_indices=chunk_indices_exp,
            chunk_size=chunk_size,
            safe_gate=False,
            lower_bound=None,
            use_gate_in_kernel=False,
            A_log=None,
            dt_bias=None,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_intermediate_states,
            transpose_state_layout=transpose_state_layout,
        )

        # -------- Extract per-token outputs ---------------------------------
        o = _extract_output(o_exp, nh)         # [B, T, H, V]

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled(), (
                "return_intermediate_states is only allowed in inference mode"
            )
            return o.type_as(q), final_state, h

        # -------- Save for backward ------------------------------------------
        # We save:
        #   * q, k: expanded-or-not as produced (post l2norm if applicable).
        #     q is [B, T, H, K] (non-expanded, post-l2norm),
        #     k is [B, T*nh, H, K] (expanded, post-l2norm).
        #     Both used for dtype casting of outputs; k also used to rerun
        #     the recompute paths inside chunk_gdn2_bwd.
        #   * q_rstd, k_rstd: reciprocal norms for L2 backward (None if
        #     L2-norm was not applied).
        #   * v: expanded.
        #   * g: non-expanded activated gate (for dg's dtype cast at return).
        #   * g_cumsum: expanded post-cumsum gate (input to chunk_gdn2_bwd).
        #   * b, w: expanded channel-wise gates.
        #   * Aqk, Akk: WY intermediates.
        #   * w_wy, u_wy, qg, kg, v_new, h: intermediates that are tensors
        #     when disable_recompute=True and None otherwise; chunk_gdn2_bwd
        #     handles either case.
        #   * initial_state_out: may be None.
        #   * cu_seqlens: required for re-expansion in backward.
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_cumsum, b, w,
            Aqk, Akk,
            w_wy, u_wy, qg, kg, v_new, h,
            initial_state_out, cu_seqlens,
        )
        ctx.chunk_size = chunk_size
        ctx.num_householder = nh
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.disable_recompute = disable_recompute
        ctx.transpose_state_layout = transpose_state_layout
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        (q, q_rstd, k, k_rstd, v, g, g_cumsum, b, w,
         Aqk, Akk,
         w_wy, u_wy, qg, kg, v_new, h,
         initial_state, cu_seqlens) = ctx.saved_tensors

        nh = ctx.num_householder
        B, T, H, K = q.shape

        # -------- Re-expand do (symmetric to q) and reconstruct q_exp -------
        # do must be placed at the same positions where q was placed, because
        # the forward output at the other positions is zero (q = 0 there).
        # _expand_q does exactly that: last position of each Householder
        # group gets the real value, others are zero.
        do_exp = _expand_q(do, nh)             # [B, T*nh, H, V]
        q_exp = _expand_q(q, nh)               # [B, T*nh, H, K]

        cu_seqlens_exp = cu_seqlens * nh if cu_seqlens is not None else None
        chunk_indices_exp = (
            prepare_chunk_indices(cu_seqlens_exp, ctx.chunk_size)
            if cu_seqlens_exp is not None else None
        )

        # -------- Run GDN-2 backward on the expanded sequence ---------------
        # use_gate_in_kernel=False (gate activation is done outside this
        # Function via fused_kda_gate in the public API; PyTorch autograd
        # composes its VJP after our backward returns). We pass g_cumsum
        # -- the post-local-cumsum expanded gate -- as `g`. dA_log and
        # dt_bias_grad from chunk_gdn2_bwd are both None in this path.
        (dq_exp, dk, dv, db, dw, dg_int, dh0,
         _dA_log_unused, _dt_bias_unused) = chunk_gdn2_bwd(
            q=q_exp,
            k=k,
            v=v,
            b=b,
            wg=w,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do_exp,
            dht=dht,
            g=g_cumsum,
            g_org=None,
            cu_seqlens=cu_seqlens_exp,
            chunk_indices=chunk_indices_exp,
            chunk_size=ctx.chunk_size,
            safe_gate=False,
            lower_bound=None,
            use_gate_in_kernel=False,
            A_log=None,
            dt_bias=None,
            transpose_state_layout=ctx.transpose_state_layout,
            w_wy=w_wy, u_wy=u_wy, qg=qg, kg=kg, v_new=v_new, h=h,
            disable_recompute=ctx.disable_recompute,
        )

        # -------- Extract per-token dq and dg -------------------------------
        # dk, dv, db, dw need no extraction: they came back in expanded form
        # and flow to the nn.Module's projection weights in expanded form too.
        dq = _extract_output(dq_exp, nh)       # [B, T, H, K]
        dg = _extract_dg(dg_int, nh, T)        # [B, T, H, K]

        # -------- L2-norm VJPs on q (non-expanded) and k (expanded) ---------
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # -------- Map gradients back to the forward() argument list ---------
        # Forward args, in order:
        #    q, k, v, g, b, w,
        #    scale, num_householder, initial_state,
        #    output_final_state, use_qk_l2norm_in_kernel,
        #    cu_seqlens, cu_seqlens_cpu,
        #    safe_gate, disable_recompute,
        #    return_intermediate_states, transpose_state_layout
        # Non-tensor args return None.
        return (
            dq.to(q.dtype),      # q
            dk.to(k.dtype),      # k
            dv.to(v.dtype),      # v
            dg.to(g.dtype),      # g
            db.to(b.dtype),      # b
            dw.to(w.dtype),      # w
            None,                # scale
            None,                # num_householder
            dh0,                 # initial_state
            None,                # output_final_state
            None,                # use_qk_l2norm_in_kernel
            None,                # cu_seqlens
            None,                # cu_seqlens_cpu
            None,                # safe_gate
            None,                # disable_recompute
            None,                # return_intermediate_states
            None,                # transpose_state_layout
        )


# =============================================================================
# Public API
# =============================================================================

@torch.compiler.disable
def chunk_gdn2_plus_channel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    num_householder: int = 1,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    transpose_state_layout: bool = False,
    **kwargs,
):
    r"""Chunkwise forward for GDN-2+-Channel.

    GDN-2 with channel-wise erase/write gates composed with DeltaProduct's
    multiple Householder products. For each token i taking n_h Householder
    steps:

    .. math::
        S_{i,1} &= \bigl(I - k_{i,1} (b_{i,1} \odot k_{i,1})^\top\bigr)
                    \operatorname{Diag}(\alpha_i)\, S_{i-1}
                    + k_{i,1}\, (w_{i,1} \odot v_{i,1})^\top \\
        S_{i,j} &= \bigl(I - k_{i,j} (b_{i,j} \odot k_{i,j})^\top\bigr)\,
                    S_{i,j-1}
                    + k_{i,j}\, (w_{i,j} \odot v_{i,j})^\top
                    \quad \text{for } j = 2, \ldots, n_h \\
        o_i     &= S_{i,n_h}^\top q_i

    The full state-transition matrix for token i is

    .. math::
        A(x_i) = \prod_{j=1}^{n_h} \bigl(I - k_{i,j} (b_{i,j} \odot
                  k_{i,j})^\top\bigr) \cdot \operatorname{Diag}(\alpha_i),

    which has rank at most n_h for the non-decay part. When
    ``num_householder=1`` the model reduces to plain GDN-2.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``. Not pre-expanded.
        k (torch.Tensor):
            Keys of shape ``[B, T * num_householder, H, K]``. Pre-interleaved
            by the caller: ``[k_{1,1}, ..., k_{1,nh}, k_{2,1}, ...]``.
        v (torch.Tensor):
            Values of shape ``[B, T * num_householder, H, V]``. Pre-interleaved.
        g (torch.Tensor):
            Channel-wise decay gate of shape ``[B, T, H, K]``. **Not**
            pre-expanded -- interleaving is handled internally. If
            ``use_gate_in_kernel=True``, this is the raw pre-activation;
            otherwise it is the activated log-decay.
        b (torch.Tensor):
            Channel-wise erase gate, shape ``[B, T * num_householder, H, K]``.
            Pre-interleaved. Typical range ``[0, 2]`` (``[0, 1]`` without
            ``allow_neg_eigval``).
        w (torch.Tensor):
            Channel-wise write gate, shape ``[B, T * num_householder, H, V]``.
            Pre-interleaved. Typical range ``[0, 1]``.
        num_householder (int):
            Number of Householder steps per token. Default: ``1``.
        scale (Optional[float]):
            Attention scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial recurrent state ``[N, H, K, V]``, fp32. Default: ``None``.
        output_final_state (bool):
            Whether to return the final recurrent state. Default: ``False``.
        use_qk_l2norm_in_kernel (bool):
            L2-normalize q and k inside the autograd function so that the
            Householder factors are true orthogonal projectors in the scalar
            limit. Default: ``False``.
        use_gate_in_kernel (bool):
            If ``True``, the nonlinear gate activation
            ``g = -exp(A_log) * softplus(g_raw + dt_bias)`` is computed
            inside this API (via ``fused_kda_gate``) before interleaving.
            Requires ``A_log`` in kwargs. Default: ``False``.
        cu_seqlens (torch.LongTensor, optional):
            Packed-sequence cumulative lengths ``[N+1]`` in original
            (pre-expansion) token units. When provided the batch dim must
            be 1; this module multiplies by ``num_householder`` internally
            before calling the GDN-2 kernel.
        cu_seqlens_cpu (torch.LongTensor, optional):
            CPU mirror of ``cu_seqlens`` (forwarded to the kernel).
        safe_gate (bool):
            Exposed for API parity with ``chunk_gdn2``, but always forwarded
            as ``False`` to the underlying kernel: the safe-gate path assumes
            all log-decays lie in ``[-5, 0)``, which does not hold for the
            interleaved (zero-filled) gate.
        lower_bound (Optional[float]):
            Lower bound for the safe-gate clamp. Only honored by the gate
            activation itself (``fused_kda_gate``); ignored in the kernel
            because ``safe_gate`` is forced to ``False``.
        disable_recompute (bool):
            Retain forward intermediates for a faster backward at the cost
            of memory. Default: ``False``.
        return_intermediate_states (bool):
            Return intermediate per-chunk states. Must be used in
            ``torch.inference_mode()``. Default: ``False``.
        transpose_state_layout (bool):
            Use the transposed state layout. Default: ``False``.
        **kwargs:
            Accepts ``A_log`` (required when ``use_gate_in_kernel=True``)
            and ``dt_bias`` (optional).

    Returns:
        Normal:  ``(o, final_state)``.
        Intermediate (when ``return_intermediate_states=True``):
                 ``(o, final_state, h)``.

    Example::

        >>> import torch
        >>> B, T, H, K, V, nh = 2, 512, 4, 128, 128, 2
        >>> device = 'cuda'
        >>> q = torch.randn(B, T,    H, K, dtype=torch.bfloat16, device=device)
        >>> k = torch.randn(B, T*nh, H, K, dtype=torch.bfloat16, device=device)
        >>> v = torch.randn(B, T*nh, H, V, dtype=torch.bfloat16, device=device)
        >>> b = torch.rand (B, T*nh, H, K, dtype=torch.bfloat16, device=device) * 2
        >>> w = torch.rand (B, T*nh, H, V, dtype=torch.bfloat16, device=device)
        >>> g = torch.rand (B, T,    H, K, dtype=torch.bfloat16, device=device)
        >>> A_log = torch.randn(H, K, dtype=torch.float32, device=device)
        >>> o, _ = chunk_gdn2_plus_channel(
        ...     q, k, v, g, b, w,
        ...     num_householder=nh,
        ...     use_qk_l2norm_in_kernel=True,
        ...     use_gate_in_kernel=True,
        ...     A_log=A_log,
        ... )
        >>> assert o.shape == (B, T, H, V)
    """
    assert q.dtype != torch.float32, (
        "chunk_gdn2_plus_channel does not support float32 inputs. Please use "
        "bfloat16 (or float16)."
    )

    B, T, H, K = q.shape
    V = v.shape[-1]
    nh = num_householder

    # ------------------ shape validation ---------------------------------
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
        f"b is the channel-wise erase gate (per Householder step)."
    )
    assert w.shape == (B, T * nh, H, V), (
        f"w shape: expected {(B, T * nh, H, V)}, got {tuple(w.shape)}. "
        f"w is the channel-wise write gate (per Householder step)."
    )
    assert g.shape == (B, T, H, K), (
        f"g shape: expected {(B, T, H, K)}, got {tuple(g.shape)}. "
        f"The decay gate should NOT be pre-expanded -- interleaving happens "
        f"internally (after gate activation if use_gate_in_kernel=True)."
    )
    assert K <= 256, (
        f"Currently we only support key headdim <= 256 for GDN-2 "
        f"(got K={K}) :-("
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
            "initial_state must be in float32."
        )

    if use_gate_in_kernel:
        assert "A_log" in kwargs, (
            "A_log must be provided when use_gate_in_kernel=True."
        )
        A_log = kwargs["A_log"]
        dt_bias = kwargs.get("dt_bias")
        if safe_gate and lower_bound is None:
            raise ValueError(
                "`lower_bound` must be specified when `safe_gate=True` "
                "and `use_gate_in_kernel=True`."
            )
        g = fused_kda_gate(
            g=g, A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
        )

    if scale is None:
        scale = K ** -0.5

    return ChunkGDN2PlusChannelFunction.apply(
        q, k, v, g, b, w,
        scale,
        num_householder,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
        False,                          
        disable_recompute,
        return_intermediate_states,
        transpose_state_layout,
    )


__all__ = [
    "chunk_gdn2_plus_channel",
    "ChunkGDN2PlusChannelFunction",
]