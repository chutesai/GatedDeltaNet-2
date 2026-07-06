// SPDX-License-Identifier: MIT
// GDN-2 inter-chunk recurrence, hand-written SM120 CUDA (campaign kernel 3).
//
// Covers the GDN-2 configuration of chunk_gated_delta_rule_fwd_h:
// K == V == 128, BT == 64, non-transposed state, per-channel decay via the
// cumulative gate (USE_GK, USE_G == false), exp2 gates, pre-gated operands
// (k == kg, q raw + gk for the o-arm), dense batches.
//
// The chunk recurrence is affine in the state:
//     v_new = u - w @ S_old
//     S_new = Diag(2^gk_last) S_old + kg^T @ v_new
//
// v2 design (occupancy-first — the v1 8-warp/88KB variant sat at 1 CTA/SM
// and lost to Triton's small-grain autotune config by 2x):
//   - 4 warps / 128 threads, __launch_bounds__(128, 2).
//   - state [K=128, Vt=64] fp32 in registers (64/thread), partitioned as the
//     update-MMA C-fragments (2 m16 tiles per warp) so the accumulate is
//     in-place: acc = S*decay, then mma accumulates kg^T @ v_new on top.
//   - shared memory 48.5KB in the store-h arm -> 2 CTAs/SM. Buffer overlays:
//       * kg is cp.async'd into the w buffer once w@S has consumed w,
//       * kg^T is transposed into the Sb buffer once the read-GEMMs are done.
//     The o-arm needs two extra tiles (qg, Aqk) -> 72.5KB -> 1 CTA/SM; the
//     hotter recompute instance (store-h, in every backward) gets the 2-CTA
//     shape. Dynamic smem size is chosen per arm at launch.
//   - next-chunk u and w are prefetched during the update MMA.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#define DEVINL __device__ __forceinline__

namespace {

constexpr int kBT = 64;
constexpr int kD = 128;
constexpr int kBV = 64;
constexpr int kW = 4;                 // warps per CTA

using bf16 = __nv_bfloat16;

DEVINL int sw128(int row, int col_e) {
  const int unit = col_e >> 3;
  return row * kD + ((unit ^ (row & 7)) << 3) + (col_e & 7);
}
DEVINL int sw64(int row, int col_e) {
  const int unit = col_e >> 3;
  return row * kBT + ((unit ^ (row & 7)) << 3) + (col_e & 7);
}
DEVINL unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
DEVINL void mma_16x8x16(const unsigned a[4], const unsigned b[2], float c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEVINL void ldm_x4(unsigned f[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(a));
}
DEVINL void ldm_x2t(unsigned f[2], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(f[0]), "=r"(f[1]) : "r"(a));
}
DEVINL void ldm_x4t(unsigned f[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(a));
}
DEVINL unsigned pack_f2(float lo, float hi) {
  __nv_bfloat162 t = __halves2bfloat162(__float2bfloat16(lo),
                                        __float2bfloat16(hi));
  unsigned u;
  memcpy(&u, &t, 4);
  return u;
}
DEVINL void cp16(unsigned dst, const void* src, bool valid) {
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src), "r"(bytes));
}
DEVINL void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> DEVINL void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Shared layout. The o-arm members are LAST so the store-h arm can launch
// with a truncated dynamic-smem allocation (offsetof(qsm)).
struct FwdHShared {
  bf16 sb[kD * kBV];      // 16KB: bf16 state (read GEMMs) -> kg^T (update)
  bf16 wsm[kBT * kD];     // 16KB: w (w@S) -> raw kg (after w@S)
  bf16 vb[kBT * kBV];     // 8KB  v_new
  bf16 usm[kBT * kBV];    // 8KB  u
  float glast[kD];        // 0.5KB per-K decay 2^gk_last
  // ---- o-arm only ----
  bf16 qsm[kBT * kD];     // 16KB qg = q * 2^gk
  bf16 asm_[kBT * kBT];   // 8KB  tril(Aqk)
};

// state fragment coordinates: warp owns K-rows [warp*32, warp*32+32) as two
// m16 tiles; index i = mt*32 + n*4 + rr*2 + e.
DEVINL int st_row(int warp, int mt, int lane, int rr) {
  return warp * 32 + mt * 16 + (lane >> 2) + rr * 8;
}
DEVINL int st_col(int lane, int n, int e) {
  return n * 8 + (lane & 3) * 2 + e;
}

__global__ void __launch_bounds__(kW * 32, 2)
gdn2_fwd_h_kernel(
    const bf16* __restrict__ kg,
    const bf16* __restrict__ w,
    const bf16* __restrict__ u,
    const float* __restrict__ gk,       // cumulative gate [B,T,H,K] fp32
    const float* __restrict__ h0,       // nullable initial state [B,H,K,V]
    const bf16* __restrict__ q_in,      // nullable raw q [B,T,H,K] (o-arm)
    const bf16* __restrict__ Aqk,       // nullable [B,T,H,BT] (o-arm)
    bf16* __restrict__ h_o,             // nullable per-chunk state [B,NT,H,K,V]
    bf16* __restrict__ vnew_o,          // nullable [B,T,H,V]
    bf16* __restrict__ o_o,             // nullable [B,T,H,V] (o-arm)
    float* __restrict__ ht_o,           // nullable final/per-seg state
    float scale, int T, int H, int NT, int nt_seg) {
  // Segmented mode (gridDim.z > 1): CTA z owns chunks [z*nt_seg, +nt_seg);
  // h0/ht_o are then per-segment [S,B,H,K,V] (entry states / c_seg), and only
  // the LAST segment's ht is the true final state. nt_seg == NT and z == 0
  // reproduce the plain sequential kernel bit-for-bit.
  extern __shared__ char smem_raw[];
  FwdHShared& sm = *reinterpret_cast<FwdHShared*>(smem_raw);

  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_seg = blockIdx.z;
  const int i_b = i_nh / H, i_h = i_nh % H;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long bos = (long)i_b * T;
  const int v0 = i_v * kBV;
  const bool o_arm = (o_o != nullptr);
  const int c_lo = i_seg * nt_seg;                    // first chunk of segment
  const int nt_here = min(nt_seg, NT - c_lo);
  if (nt_here <= 0) return;
  const long seg_state = ((long)i_seg * gridDim.y + i_nh) * kD * kD;

  auto rowD = [&](const bf16* base, int t, int d) -> const bf16* {
    return base + ((bos + t) * H + i_h) * (long)kD + d;
  };
  auto gRow = [&](int t, int d) -> const float* {
    return gk + ((bos + t) * H + i_h) * (long)kD + d;
  };

  // ---- init state (registers) from h0 or zero ----
  float S[64];
  #pragma unroll
  for (int mt = 0; mt < 2; ++mt)
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = mt * 32 + n * 4 + rr * 2 + e;
          S[i] = 0.f;
          if (h0 != nullptr) {
            const int r = st_row(warp, mt, lane, rr), c = st_col(lane, n, e);
            S[i] = h0[seg_state + (long)r * kD + v0 + c];
          }
        }

  // prime the first chunk's w/u (+qg/Aqk) loads
  auto stage_wu = [&](int i_t) {
    const int t_lo = (c_lo + i_t) * kBT;
    const int rows = min(kBT, T - t_lo);
    for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kW * 32) {
      const int r = idx >> 4, c8 = (idx & 15) << 3;
      const bool ok = r < rows;
      cp16(smem_u32(&sm.wsm[sw128(r, c8)]),
           w + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kD + c8, ok);
    }
    for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kW * 32) {
      const int r = idx >> 3, c8 = (idx & 7) << 3;
      const bool ok = r < rows;
      cp16(smem_u32(&sm.usm[sw64(r, c8)]),
           u + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kD + v0 + c8, ok);
    }
  };
  stage_wu(0);
  cp_commit();
  // prime chunk 0's staged state (subsequent chunks are staged by the fused
  // update epilogue); h stores flow through sb as coalesced 16B copies.
  auto stage_state = [&](float* Sreg) {
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
      #pragma unroll
      for (int n = 0; n < 8; ++n)
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int i = mt * 32 + n * 4 + rr * 2;
          const int r = st_row(warp, mt, lane, rr);
          const int c = st_col(lane, n, 0);   // c, c+1 adjacent
          *reinterpret_cast<unsigned*>(&sm.sb[sw64(r, c)]) =
              pack_f2(Sreg[i], Sreg[i + 1]);
        }
  };
  auto copy_h = [&](int i_chunk) {   // coalesced sb -> h_o[c_lo + i_chunk]
    for (int idx = threadIdx.x; idx < kD * (kBV / 8); idx += kW * 32) {
      const int r = idx >> 3, c8 = (idx & 7) << 3;
      *reinterpret_cast<uint4*>(
          &h_o[((((long)i_b * NT + c_lo + i_chunk) * H + i_h) * kD + r) * kD
               + v0 + c8]) =
          *reinterpret_cast<const uint4*>(&sm.sb[sw64(r, c8)]);
    }
  };
  stage_state(S);
  if (h_o != nullptr) {
    __syncthreads();
    copy_h(0);
  }

  for (int i_t = 0; i_t < nt_here; ++i_t) {
    const int t_lo = (c_lo + i_t) * kBT;
    const int rows = min(kBT, T - t_lo);
    const int last = t_lo + rows - 1;

    // o-arm staging: qg = q*2^gk (fused transform) and tril(Aqk)
    if (o_arm) {
      for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kW * 32) {
        const int r = idx >> 4, c8 = (idx & 15) << 3;
        bf16 out[8];
        if (r < rows) {
          uint4 qr = *reinterpret_cast<const uint4*>(rowD(q_in, t_lo + r, c8));
          const bf16* qp = reinterpret_cast<const bf16*>(&qr);
          #pragma unroll
          for (int e = 0; e < 8; ++e)
            out[e] = __float2bfloat16(__bfloat162float(qp[e])
                                      * exp2f(*gRow(t_lo + r, c8 + e)));
        } else {
          #pragma unroll
          for (int e = 0; e < 8; ++e) out[e] = __float2bfloat16(0.f);
        }
        *reinterpret_cast<uint4*>(&sm.qsm[sw128(r, c8)]) =
            *reinterpret_cast<uint4*>(out);
      }
      for (int idx = threadIdx.x; idx < kBT * kBT; idx += kW * 32) {
        const int r = idx / kBT, c = idx % kBT;
        const bool keep = (c <= r) && (r < rows) && (c < rows);
        sm.asm_[sw64(r, c)] = keep
            ? *reinterpret_cast<const bf16*>(
                  Aqk + ((bos + t_lo + r) * H + i_h) * (long)kBT + c)
            : __float2bfloat16(0.f);
      }
    }
    // decay factors
    for (int k = threadIdx.x; k < kD; k += kW * 32)
      sm.glast[k] = exp2f(*gRow(last, k));
    cp_wait<0>();
    __syncthreads();

    // ---- wh = w @ Sb; v_new = u - wh (invalid tail rows -> 0) ----
    // warp owns t-rows [warp*16, +16), all 64 Vt cols.
    {
      const int mr = warp * 16;
      float acc[32];
      #pragma unroll
      for (int i = 0; i < 32; ++i) acc[i] = 0.f;
      #pragma unroll
      for (int kc = 0; kc < kD / 16; ++kc) {
        unsigned a[4];
        ldm_x4(a, smem_u32(&sm.wsm[sw128(mr + (lane & 15),
                                         kc * 16 + ((lane & 16) ? 8 : 0))]));
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          unsigned b[2];
          ldm_x2t(b, smem_u32(&sm.sb[sw64(kc * 16 + (lane & 7)
                                          + ((lane & 8) ? 8 : 0), n * 8)]));
          mma_16x8x16(a, b, &acc[n * 4]);
        }
      }
      __syncthreads();   // all reads of wsm done before kg overwrites it
      #pragma unroll
      for (int n = 0; n < 8; ++n)
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int i = n * 4 + rr * 2;
          const int row = mr + (lane >> 2) + rr * 8;
          const int col = n * 8 + (lane & 3) * 2;
          const unsigned upair =
              *reinterpret_cast<const unsigned*>(&sm.usm[sw64(row, col)]);
          __nv_bfloat162 u2;
          memcpy(&u2, &upair, 4);
          const float v0f = (row < rows)
              ? (__bfloat162float(u2.x) - acc[i]) : 0.f;
          const float v1f = (row < rows)
              ? (__bfloat162float(u2.y) - acc[i + 1]) : 0.f;
          *reinterpret_cast<unsigned*>(&sm.vb[sw64(row, col)]) =
              pack_f2(v0f, v1f);
        }
    }
    // kg -> the freed w buffer (overlaps the o-arm GEMMs below)
    for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kW * 32) {
      const int r = idx >> 4, c8 = (idx & 15) << 3;
      const bool ok = r < rows;
      cp16(smem_u32(&sm.wsm[sw128(r, c8)]),
           kg + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kD + c8, ok);
    }
    cp_commit();
    __syncthreads();   // vb visible to all warps
    // coalesced v_new store from the staged tile
    if (vnew_o != nullptr) {
      for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kW * 32) {
        const int r = idx >> 3, c8 = (idx & 7) << 3;
        if (r < rows)
          *reinterpret_cast<uint4*>(
              &vnew_o[((bos + t_lo + r) * H + i_h) * (long)kD + v0 + c8]) =
              *reinterpret_cast<const uint4*>(&sm.vb[sw64(r, c8)]);
      }
    }

    // ---- o-arm: o = scale*(qg @ Sb) + tril(Aqk) @ v_new ----
    if (o_arm) {
      const int mr = warp * 16;
      float acc[32];
      #pragma unroll
      for (int i = 0; i < 32; ++i) acc[i] = 0.f;
      #pragma unroll
      for (int kc = 0; kc < kD / 16; ++kc) {
        unsigned a[4];
        ldm_x4(a, smem_u32(&sm.qsm[sw128(mr + (lane & 15),
                                         kc * 16 + ((lane & 16) ? 8 : 0))]));
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          unsigned b[2];
          ldm_x2t(b, smem_u32(&sm.sb[sw64(kc * 16 + (lane & 7)
                                          + ((lane & 8) ? 8 : 0), n * 8)]));
          mma_16x8x16(a, b, &acc[n * 4]);
        }
      }
      #pragma unroll
      for (int i = 0; i < 32; ++i) acc[i] *= scale;
      #pragma unroll
      for (int kc = 0; kc < kBT / 16; ++kc) {
        unsigned a[4];
        ldm_x4(a, smem_u32(&sm.asm_[sw64(mr + (lane & 15),
                                         kc * 16 + ((lane & 16) ? 8 : 0))]));
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          unsigned b[2];
          ldm_x2t(b, smem_u32(&sm.vb[sw64(kc * 16 + (lane & 7)
                                          + ((lane & 8) ? 8 : 0), n * 8)]));
          mma_16x8x16(a, b, &acc[n * 4]);
        }
      }
      // stage o fragments into usm (dead after v_new), then coalesce out.
      #pragma unroll
      for (int n = 0; n < 8; ++n)
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int i = n * 4 + rr * 2;
          const int row = mr + (lane >> 2) + rr * 8;
          const int col = n * 8 + (lane & 3) * 2;
          *reinterpret_cast<unsigned*>(&sm.usm[sw64(row, col)]) =
              pack_f2(acc[i], acc[i + 1]);
        }
      __syncthreads();
      for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kW * 32) {
        const int r = idx >> 3, c8 = (idx & 7) << 3;
        if (r < rows)
          *reinterpret_cast<uint4*>(
              &o_o[((bos + t_lo + r) * H + i_h) * (long)kD + v0 + c8]) =
              *reinterpret_cast<const uint4*>(&sm.usm[sw64(r, c8)]);
      }
    }

    // kg resident (raw [t,k]) in wsm — its A^T fragments load directly via
    // ldmatrix.x4.trans; no smem transpose pass. NOTE: invalid tail rows of
    // the kg tile may hold garbage (0-byte cp.async), but their matching
    // v_new rows are exact zeros — 0 * garbage is only unsafe for inf/NaN
    // bit patterns, so scrub the tail rows once.
    cp_wait<0>();
    __syncthreads();
    if (rows < kBT) {
      for (int idx = threadIdx.x; idx < (kBT - rows) * (kD / 8);
           idx += kW * 32) {
        const int r = rows + (idx >> 4), c8 = (idx & 15) << 3;
        if (r < kBT) {
          bf16 z[8];
          #pragma unroll
          for (int e = 0; e < 8; ++e) z[e] = __float2bfloat16(0.f);
          *reinterpret_cast<uint4*>(&sm.wsm[sw128(r, c8)]) =
              *reinterpret_cast<uint4*>(z);
        }
      }
      __syncthreads();
    }
    // prefetch the NEXT chunk's u into the freed usm during the update
    // (w must wait for wsm — it holds kg until the update finishes).
    const bool have_next = (i_t + 1 < nt_here);
    if (have_next) {
      const int nt_lo = (c_lo + i_t + 1) * kBT;
      const int nrows = min(kBT, T - nt_lo);
      for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kW * 32) {
        const int r = idx >> 3, c8 = (idx & 7) << 3;
        const bool ok = r < nrows;
        cp16(smem_u32(&sm.usm[sw64(r, c8)]),
             u + ((bos + nt_lo + (ok ? r : 0)) * H + i_h) * (long)kD + v0 + c8,
             ok);
      }
      cp_commit();
    }

    // ---- update: S = Diag(2^gk_last) S + kg^T @ v_new, then the fused
    // epilogue stages bf16(S_new) into sb (+ h_o[i_t+1]) for the next iter.
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int kr = warp * 32 + mt * 16;
      #pragma unroll
      for (int n = 0; n < 8; ++n)
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr)
          #pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int i = mt * 32 + n * 4 + rr * 2 + e;
            S[i] *= sm.glast[st_row(warp, mt, lane, rr)];
          }
      #pragma unroll
      for (int kc = 0; kc < kBT / 16; ++kc) {
        // A = kg^T fragment straight from the raw [t,k] tile (x4 trans):
        // lanes 0-7/8-15/16-23/24-31 address t-rows (kc*16 + tq) at column
        // groups m0 / m0+8 with the second t-half in the upper lane quads.
        unsigned a[4];
        const int trow = kc * 16 + (lane & 7) + ((lane & 16) ? 8 : 0);
        const int mcol = kr + ((lane & 8) ? 8 : 0);
        ldm_x4t(a, smem_u32(&sm.wsm[sw128(trow, mcol)]));
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          unsigned b[2];
          ldm_x2t(b, smem_u32(&sm.vb[sw64(kc * 16 + (lane & 7)
                                          + ((lane & 8) ? 8 : 0), n * 8)]));
          mma_16x8x16(a, b, &S[mt * 32 + n * 4]);
        }
      }
    }
    __syncthreads();   // wsm/vb reads done; next iter may overwrite them
    // fused stage: bf16(S_new) -> sb for the next chunk's read-GEMMs, plus
    // the next chunk's pre-state store.
    if (have_next) {
      stage_state(S);
      if (h_o != nullptr) {
        __syncthreads();
        copy_h(i_t + 1);
      }
      // now safe to prefetch the next w into wsm
      const int nt_lo = (c_lo + i_t + 1) * kBT;
      const int nrows = min(kBT, T - nt_lo);
      for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kW * 32) {
        const int r = idx >> 4, c8 = (idx & 15) << 3;
        const bool ok = r < nrows;
        cp16(smem_u32(&sm.wsm[sw128(r, c8)]),
             w + ((bos + nt_lo + (ok ? r : 0)) * H + i_h) * (long)kD + c8, ok);
      }
      cp_commit();
    }
  }

  if (ht_o != nullptr) {
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
      #pragma unroll
      for (int n = 0; n < 8; ++n)
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr)
          #pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int i = mt * 32 + n * 4 + rr * 2 + e;
            const int r = st_row(warp, mt, lane, rr), c = st_col(lane, n, e);
            ht_o[seg_state + (long)r * kD + v0 + c] = S[i];
          }
  }
}


// ---------------------------------------------------------------------------
// Phase A1 of the segmented scan: per-segment transition products
//     M_seg = prod_{chunks oldest->newest} (Diag(2^gk_last) - kg^T w)
// folded as  M <- Diag(2^gk_last) M - kg^T (w M),  starting from M = I.
// One CTA per (segment, b*h); 8 warps; M [K,K] fp32 lives in registers
// (64/thread), warp owns M-rows [warp*16, +16) x 128 cols (16 n-frags).
// ---------------------------------------------------------------------------
struct MprodShared {
  bf16 mb[kD * kD];       // 32KB bf16 M (B-operand of w @ M)
  bf16 wsm[kBT * kD];     // 16KB w -> kg (overlay once w @ M is done)
  bf16 xneg[kBT * kD];    // 16KB -(w @ M) staged [t, col]
  float glast[kD];
};

__global__ void __launch_bounds__(256, 1)
gdn2_fwd_h_mprod_kernel(
    const bf16* __restrict__ kg,
    const bf16* __restrict__ w,
    const float* __restrict__ gk,
    bf16* __restrict__ m_o,             // [S, B*H, K, K]
    int T, int H, int NT, int nt_seg) {
  extern __shared__ char smem_raw[];
  MprodShared& sm = *reinterpret_cast<MprodShared*>(smem_raw);

  const int i_seg = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int i_b = i_nh / H, i_h = i_nh % H;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long bos = (long)i_b * T;
  const int c_lo = i_seg * nt_seg;
  const int nt_here = min(nt_seg, NT - c_lo);
  if (nt_here <= 0) return;

  auto gRow = [&](int t, int d) -> const float* {
    return gk + ((bos + t) * H + i_h) * (long)kD + d;
  };

  // M = I at fragment coords: warp rows [warp*16,+16), cols n*8+(lane&3)*2+e
  float M[64];
  #pragma unroll
  for (int n = 0; n < 16; ++n)
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr)
      #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int i = n * 4 + rr * 2 + e;
        const int r = warp * 16 + (lane >> 2) + rr * 8;
        const int c = n * 8 + (lane & 3) * 2 + e;
        M[i] = (r == c) ? 1.f : 0.f;
      }

  for (int i_t = 0; i_t < nt_here; ++i_t) {
    const int t_lo = (c_lo + i_t) * kBT;
    const int rows = min(kBT, T - t_lo);
    const int last = t_lo + rows - 1;

    // stage bf16(M) -> mb  (packed 4B stores) and start the w load
    #pragma unroll
    for (int n = 0; n < 16; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        const int i = n * 4 + rr * 2;
        const int r = warp * 16 + (lane >> 2) + rr * 8;
        const int c = n * 8 + (lane & 3) * 2;
        *reinterpret_cast<unsigned*>(&sm.mb[sw128(r, c)]) =
            pack_f2(M[i], M[i + 1]);
      }
    for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += 256) {
      const int r = idx >> 4, c8 = (idx & 15) << 3;
      const bool ok = r < rows;
      cp16(smem_u32(&sm.wsm[sw128(r, c8)]),
           w + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kD + c8, ok);
    }
    cp_commit();
    for (int k = threadIdx.x; k < kD; k += 256)
      sm.glast[k] = exp2f(*gRow(last, k));
    cp_wait<0>();
    __syncthreads();

    // X = w @ M  (contract K=128); warp = (t-strip, col-half): [16, 64]
    {
      const int mr = (warp & 3) * 16;
      const int nc = (warp >> 2) * 64;
      float acc[32];
      #pragma unroll
      for (int i = 0; i < 32; ++i) acc[i] = 0.f;
      #pragma unroll
      for (int kc = 0; kc < kD / 16; ++kc) {
        unsigned a[4];
        ldm_x4(a, smem_u32(&sm.wsm[sw128(mr + (lane & 15),
                                         kc * 16 + ((lane & 16) ? 8 : 0))]));
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
          unsigned b[2];
          ldm_x2t(b, smem_u32(&sm.mb[sw128(kc * 16 + (lane & 7)
                                           + ((lane & 8) ? 8 : 0),
                                           nc + n * 8)]));
          mma_16x8x16(a, b, &acc[n * 4]);
        }
      }
      __syncthreads();   // wsm reads done before kg overwrites it
      // stage -X (zeroing invalid tail rows)
      #pragma unroll
      for (int n = 0; n < 8; ++n)
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int i = n * 4 + rr * 2;
          const int row = mr + (lane >> 2) + rr * 8;
          const int col = nc + n * 8 + (lane & 3) * 2;
          const bool ok = row < rows;
          *reinterpret_cast<unsigned*>(&sm.xneg[sw128(row, col)]) =
              pack_f2(ok ? -acc[i] : 0.f, ok ? -acc[i + 1] : 0.f);
        }
    }
    // kg into the freed w buffer
    for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += 256) {
      const int r = idx >> 4, c8 = (idx & 15) << 3;
      const bool ok = r < rows;
      cp16(smem_u32(&sm.wsm[sw128(r, c8)]),
           kg + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kD + c8, ok);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    if (rows < kBT) {   // scrub kg tail rows (0 * NaN-garbage protection)
      for (int idx = threadIdx.x; idx < (kBT - rows) * (kD / 8); idx += 256) {
        const int r = rows + (idx >> 4), c8 = (idx & 15) << 3;
        if (r < kBT) {
          bf16 z[8];
          #pragma unroll
          for (int e = 0; e < 8; ++e) z[e] = __float2bfloat16(0.f);
          *reinterpret_cast<uint4*>(&sm.wsm[sw128(r, c8)]) =
              *reinterpret_cast<uint4*>(z);
        }
      }
      __syncthreads();
    }

    // M <- Diag(2^gk_last) M + kg^T @ (-X)
    #pragma unroll
    for (int n = 0; n < 16; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          M[i] *= sm.glast[warp * 16 + (lane >> 2) + rr * 8];
        }
    #pragma unroll
    for (int kc = 0; kc < kBT / 16; ++kc) {
      unsigned a[4];
      const int trow = kc * 16 + (lane & 7) + ((lane & 16) ? 8 : 0);
      const int mcol = warp * 16 + ((lane & 8) ? 8 : 0);
      ldm_x4t(a, smem_u32(&sm.wsm[sw128(trow, mcol)]));
      #pragma unroll
      for (int n = 0; n < 16; ++n) {
        unsigned b[2];
        ldm_x2t(b, smem_u32(&sm.xneg[sw128(kc * 16 + (lane & 7)
                                           + ((lane & 8) ? 8 : 0), n * 8)]));
        mma_16x8x16(a, b, &M[n * 4]);
      }
    }
    __syncthreads();
  }

  // store M_seg (coalesced via mb)
  #pragma unroll
  for (int n = 0; n < 16; ++n)
    #pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      const int i = n * 4 + rr * 2;
      const int r = warp * 16 + (lane >> 2) + rr * 8;
      const int c = n * 8 + (lane & 3) * 2;
      *reinterpret_cast<unsigned*>(&sm.mb[sw128(r, c)]) =
          pack_f2(M[i], M[i + 1]);
    }
  __syncthreads();
  {
    const long base = ((long)i_seg * gridDim.y + i_nh) * kD * kD;
    for (int idx = threadIdx.x; idx < kD * (kD / 8); idx += 256) {
      const int r = idx >> 4, c8 = (idx & 15) << 3;
      *reinterpret_cast<uint4*>(&m_o[base + (long)r * kD + c8]) =
          *reinterpret_cast<const uint4*>(&sm.mb[sw128(r, c8)]);
    }
  }
}

}  // namespace

void gdn2_fwd_h(
    torch::Tensor kg, torch::Tensor w, torch::Tensor u, torch::Tensor gk,
    c10::optional<torch::Tensor> h0, c10::optional<torch::Tensor> qg,
    c10::optional<torch::Tensor> Aqk, c10::optional<torch::Tensor> h_o,
    c10::optional<torch::Tensor> vnew_o, c10::optional<torch::Tensor> o_o,
    c10::optional<torch::Tensor> ht_o, double scale, int64_t nseg) {
  const at::cuda::CUDAGuard guard{kg.device()};
  const int B = kg.size(0), T = kg.size(1), H = kg.size(2);
  const int NT = (T + kBT - 1) / kBT;
  const int S = (int)nseg;
  const int nt_seg = (NT + S - 1) / S;
  dim3 grid(kD / kBV, B * H, S);
  dim3 block(kW * 32);
  const bool o_arm = o_o.has_value();
  size_t smem = o_arm ? sizeof(FwdHShared) : offsetof(FwdHShared, qsm);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        gdn2_fwd_h_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(FwdHShared));
    TORCH_CHECK(e == cudaSuccess, "smem opt-in failed: ", cudaGetErrorString(e));
  }
  auto bp = [](c10::optional<torch::Tensor>& t) -> bf16* {
    return t.has_value() ? reinterpret_cast<bf16*>(t->data_ptr()) : nullptr;
  };
  gdn2_fwd_h_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const bf16*>(kg.data_ptr()),
      reinterpret_cast<const bf16*>(w.data_ptr()),
      reinterpret_cast<const bf16*>(u.data_ptr()),
      gk.data_ptr<float>(),
      h0.has_value() ? h0->data_ptr<float>() : nullptr,
      qg.has_value() ? reinterpret_cast<const bf16*>(qg->data_ptr()) : nullptr,
      Aqk.has_value() ? reinterpret_cast<const bf16*>(Aqk->data_ptr()) : nullptr,
      bp(h_o), bp(vnew_o), bp(o_o),
      ht_o.has_value() ? ht_o->data_ptr<float>() : nullptr,
      (float)scale, T, H, NT, nt_seg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn2_fwd_h_mprod(
    torch::Tensor kg, torch::Tensor w, torch::Tensor gk, torch::Tensor m_o,
    int64_t nseg) {
  const at::cuda::CUDAGuard guard{kg.device()};
  const int B = kg.size(0), T = kg.size(1), H = kg.size(2);
  const int NT = (T + kBT - 1) / kBT;
  const int S = (int)nseg;
  const int nt_seg = (NT + S - 1) / S;
  dim3 grid(S, B * H);
  dim3 block(256);
  size_t smem = sizeof(MprodShared);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaError_t e = cudaFuncSetAttribute(
      gdn2_fwd_h_mprod_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem);
  TORCH_CHECK(e == cudaSuccess, "smem opt-in failed: ", cudaGetErrorString(e));
  gdn2_fwd_h_mprod_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const bf16*>(kg.data_ptr()),
      reinterpret_cast<const bf16*>(w.data_ptr()),
      gk.data_ptr<float>(),
      reinterpret_cast<bf16*>(m_o.data_ptr()),
      T, H, NT, nt_seg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn2_fwd_h", &gdn2_fwd_h, "GDN-2 forward inter-chunk recurrence (SM120)");
  m.def("gdn2_fwd_h_mprod", &gdn2_fwd_h_mprod,
        "GDN-2 segmented-scan phase A: per-segment transition products");
}
