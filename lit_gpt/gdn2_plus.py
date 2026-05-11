# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import (
    get_layer_cache,
    get_unpad_data,
    index_first_axis,
    pad_input,
    update_layer_cache,
)
from fla.modules import FusedRMSNormSwishGate, ShortConvolution

from .gdn2_ops.chunk_gdn2_plus import chunk_gdn2_plus_channel
from .gdn2_ops.fused_recurrent_gdn2_plus import fused_recurrent_gdn2

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GatedDeltaNet2Plus(nn.Module):
    """
    GDN-2-Plus-Channel layer.

    Combines GDN-2's channel-wise erase/write gates with DeltaProduct's
    multiple Householder products. State-transition matrix per token:

        A(x_i) = prod_{j=1..n_h} (I - k_{i,j} (b_{i,j} ⊙ k_{i,j})^T)
                  · Diag(exp(g_i))

    where `⊙` denotes the Hadamard product. The non-decay part of the update
    has rank up to n_h; the decay `Diag(exp(g_i))` is applied once per token
    between the previous token's final state and this token's first
    Householder step.

    Setting `num_householder=1` reduces the layer to plain GDN-2. Setting in
    addition `b = beta · 1` and `w = beta · 1` (scalar broadcast) recovers
    KDA exactly.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection. Equal to `num_heads`
            if ``None``. GVA (Grouped Value Attention) is applied if
            `num_v_heads > num_heads`. Default: ``None``.
        num_householder (int, Optional):
            Number of Householder transformations per token. Controls the rank
            of the non-decay update: rank-1 (=GDN-2) to rank-n_h (DeltaProduct).
            Default: 1.
        mode (str, Optional):
            Which kernel to use. Available: ``chunk`` (training + long
            inference) and ``fused_recurrent`` (short-sequence inference).
            The layer automatically falls back to ``fused_recurrent`` for
            sequences with ``q_len <= 64`` at inference. Default: ``chunk``.
        use_short_conv (bool, Optional):
            Whether to use short convolutions on q, k, v. Default: ``True``.
        allow_neg_eigval (bool, Optional):
            If ``True``, multiplies the erase gate `b` by 2 to extend the
            per-channel eigenvalue range from [0, 1] into [-1, 1], enabling
            reflections (and, for n_h >= 2, rotations). Default: ``False``.
            Note: only the erase side `b` is doubled; the write gate `w`
            stays in [0, 1] (as in base GDN-2).
        conv_size (int, Optional):
            Kernel size of the short convolutions. Default: 4.
        conv_bias (bool, Optional):
            Bias in the short convolutions. Default: ``False``.
        layer_idx (int, Optional):
            Layer index (used by the cache). Default: ``None``.
        norm_eps (float, Optional):
            Epsilon for the output RMSNorm. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        num_householder: int = 2,
        mode: Literal["chunk", "fused_recurrent"] = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> "GatedDeltaNet2":
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.num_householder = num_householder

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # ---------------- consistency checks -------------------------------
        if not math.isclose(
            self.num_v_heads * self.head_dim * expand_v,
            self.value_dim,
            rel_tol=1e-5,
        ):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when "
                f"multiplied by key_dim={self.key_dim}. Resulting value_dim "
                f"would be {self.num_v_heads * self.head_dim * expand_v}, "
                f"which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by "
                f"num_heads={self.num_heads}.",
            )
        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when "
                f"multiplied by head_dim={head_dim}. Resulting head_v_dim "
                f"would be {head_dim * expand_v}, which is invalid for "
                f"FusedRMSNormSwishGate.",
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."
        assert num_householder >= 1, (
            f"num_householder must be >= 1, got {num_householder}."
        )

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_size, self.key_dim * num_householder, bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size, self.value_dim * num_householder, bias=False,
        )

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim * num_householder,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim * num_householder,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )

        self.b_proj = nn.Linear(
            hidden_size, self.key_dim * num_householder, bias=False,
        )

        self.w_proj = nn.Linear(
            hidden_size, self.value_dim * num_householder, bias=False,
        )

        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
            )
        )
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def _reshape_for_deltaproduct(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        w: torch.Tensor,
    ):
        """
        Reshape projected k, v, b, w from per-token layout to interleaved
        layout for DeltaProduct (n_h Householder steps per token).

        Input shapes  (after projection + optional short-conv):
            k:  [B, T, n_h * num_heads   * head_k_dim]
            v:  [B, T, n_h * num_v_heads * head_v_dim]
            b:  [B, T, n_h * num_heads   * head_k_dim]   (channel-wise on K)
            w:  [B, T, n_h * num_v_heads * head_v_dim]   (channel-wise on V)

        Output shapes (after interleaving along the time axis):
            k:  [B, T*n_h, num_heads,   head_k_dim]
            v:  [B, T*n_h, num_v_heads, head_v_dim]
            b:  [B, T*n_h, num_heads,   head_k_dim]
            w:  [B, T*n_h, num_v_heads, head_v_dim]

        Interleaving order:
            [t1_step1, t1_step2, ..., t1_step_nh,
             t2_step1, t2_step2, ..., t2_step_nh, ...]

        At n_h == 1 this is just a view-reshape (no interleaving).
        """
        nh = self.num_householder

        if nh == 1:
            k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
            v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
            b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
            w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)
            return k, v, b, w

        k = rearrange(k, "... (n h d) -> ... n h d", n=nh, d=self.head_k_dim)
        k = rearrange(k, "b t n h d -> b (t n) h d")

        v = rearrange(v, "... (n h d) -> ... n h d", n=nh, d=self.head_v_dim)
        v = rearrange(v, "b t n h d -> b (t n) h d")

        b = rearrange(b, "... (n h d) -> ... n h d", n=nh, d=self.head_k_dim)
        b = rearrange(b, "b t n h d -> b (t n) h d")

        w = rearrange(w, "... (n h d) -> ... n h d", n=nh, d=self.head_v_dim)
        w = rearrange(w, "b t n h d -> b (t n) h d")

        return k, v, b, w

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: "Cache | None" = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: "Unpack[dict]",
    ) -> tuple[torch.Tensor, torch.Tensor | None, "Cache | None"]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape "
                "[batch_size, seq_len] for padding purposes (0 indicates "
                "padding). Arbitrary attention masks of shape "
                "[batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        nh = self.num_householder

        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."),
                indices,
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),        # [B, T, key_dim * nh]
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),        # [B, T, value_dim * nh]
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        b = self.b_proj(hidden_states).sigmoid()     # [B, T, key_dim * nh]
        w = self.w_proj(hidden_states).sigmoid()     # [B, T, value_dim * nh]

        g = (
            -self.A_log.float().exp().repeat_interleave(self.head_k_dim)
            * F.softplus(self.f_proj(hidden_states).float() + self.dt_bias)
        )

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        k, v, b, w = self._reshape_for_deltaproduct(k, v, b, w)

        if self.num_v_heads > self.num_heads:
            gva_ratio = self.num_v_heads // self.num_heads
            q = repeat(q, "... h d -> ... (h g) d", g=gva_ratio)
            k = repeat(k, "... h d -> ... (h g) d", g=gva_ratio)
            g = repeat(g, "... h d -> ... (h g) d", g=gva_ratio)
            b = repeat(b, "... h d -> ... (h g) d", g=gva_ratio)

        if self.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )

        if mode == "chunk":
            o, recurrent_state = chunk_gdn2_plus_channel(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                num_householder=nh,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )

        elif mode == "fused_recurrent":
            if nh > 1:
                B_actual, T_actual, H_actual, K_actual = q.shape
                V_actual = v.shape[-1]
                Hv_actual = v.shape[-2]
                q_exp = q.new_zeros(B_actual, T_actual * nh, H_actual, K_actual)
                q_exp[:, nh - 1::nh] = q

                g_exp = g.new_zeros(B_actual, T_actual * nh, H_actual, K_actual)
                g_exp[:, 0::nh] = g

                cu_seqlens_exp = (
                    cu_seqlens * nh if cu_seqlens is not None else None
                )

                o_exp, recurrent_state = fused_recurrent_gdn2(
                    q=q_exp,
                    k=k,
                    v=v,
                    g=g_exp,
                    b=b,
                    w=w,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    cu_seqlens=cu_seqlens_exp,
                )
                o = o_exp[:, nh - 1::nh]
            else:
                # n_h == 1: identical to plain GDN-2's recurrent path.
                o, recurrent_state = fused_recurrent_gdn2(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    b=b,
                    w=w,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    cu_seqlens=cu_seqlens,
                )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(
                (conv_state_q, conv_state_k, conv_state_v)
                if self.use_short_conv else None
            ),
            offset=q_len,
        )

        o = self.o_norm(
            o,
            rearrange(
                self.g_proj(hidden_states),
                "... (h d) -> ... h d",
                d=self.head_v_dim,
            ),
        )
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values