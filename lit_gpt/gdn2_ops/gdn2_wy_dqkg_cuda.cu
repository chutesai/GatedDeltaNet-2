// SPDX-License-Identifier: MIT
// GDN-2 wy_dqkg backward, hand-written SM120 CUDA (campaign kernel 1).
//
// One CTA per (chunk, head), 8 warps as (strip s = warp&3: 16 rows of the
// 64-row chunk) x (half hf = warp>>2: 64-column half of K or V). Unlike the
// Triton kernel (one 255-register program, 1 CTA/SM), accumulators are
// warp-split: each warp owns [16 x 64] fp32 tiles of dq/dk/dw_flow and a
// [16 x 64] dA partial. Phases:
//   P1 (V loop): dq += do@h^T, dk += v_new@dh^T, dwf += dv@h^T, dgk += h.dh,
//       dA += dv@(v*wg)^T, dvb = A@dv -> dv2/dw stores.
//   P2: gate algebra, dA += -dwf@(kg*b)^T, dkgb = A@(-dwf), dq/dk/dg/db.
//   P3: dA = -tril(A @ (tril(dA) @ A)).
// Math replicates chunk_gdn2_bwd_kernel_wy_dqkg_fused exactly (bf16 dots,
// fp32 accumulation, same masking).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#define DEVINL __device__ __forceinline__

namespace {

constexpr int kBT = 64;    // chunk rows
constexpr int kD = 128;    // K == V == 128
constexpr int kBV = 64;    // V tile per P1 iteration
constexpr int kWarps = 8;

using bf16 = __nv_bfloat16;

DEVINL int sw_off(int row, int col_e) {  // bf16 tiles, 128-elem rows
  const int unit = col_e >> 3;
  return row * kD + ((unit ^ (row & 7)) << 3) + (col_e & 7);
}

DEVINL int sw_off64(int row, int col_e) {  // bf16 tiles, 64-elem rows
  const int unit = col_e >> 3;
  return row * kBT + ((unit ^ (row & 7)) << 3) + (col_e & 7);
}

DEVINL unsigned mask_all() { return 0xffffffffu; }

DEVINL void mma_16x8x16(const unsigned a[4], const unsigned b[2], float c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

DEVINL unsigned smem_u32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

DEVINL void ldmatrix_x2(unsigned frag[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(frag[0]), "=r"(frag[1]) : "r"(addr));
}

DEVINL void ldmatrix_x2_trans(unsigned frag[2], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(frag[0]), "=r"(frag[1]) : "r"(addr));
}

DEVINL void ldmatrix_x4(unsigned frag[4], unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(addr));
}

DEVINL void cp_async_16(unsigned dst, const void* src, bool valid) {
  const int bytes = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src), "r"(bytes));
}

DEVINL void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
DEVINL void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

DEVINL unsigned pack_bf16x2(bf16 lo, bf16 hi) {
  unsigned u;
  __nv_bfloat162 t = __halves2bfloat162(lo, hi);
  memcpy(&u, &t, 4);
  return u;
}

DEVINL unsigned pack_f2(float lo, float hi) {
  return pack_bf16x2(__float2bfloat16(lo), __float2bfloat16(hi));
}

struct SharedStorage {
  bf16 A[kBT * kBT];        // 8KB, resident (row t, col j) as in memory
  bf16 At[kBT * kBT];       // 8KB, A^T — the operand the backward consumes
  bf16 t2[kBT * kBV];       // dv (P1) | -dwf (P2) | masked dA (P3)
  bf16 t3[kBT * kBV];       // v*wg (P1) | kg*b (P2)
  bf16 h_t[kD * kBV];       // h (P1) | k tile (P2)
  bf16 dh_t[kD * kBV];      // dh (P1) | b tile (P2) | st1 scratch (P3)
  float red[5 * kD];        // [0,kD): h.dh colsum; [kD,5kD): per-strip kdk partials
  // Phase-exclusive region (32KB): P1 v_new/do | P2 fp32 g | P3 fp32 dA.
  union {
    struct { bf16 t0[kBT * kBV]; bf16 t1[kBT * kBV]; } p1;
    float gtile[kBT * kD];
    float dAf[kBT * kBT];
  } u;
};

// P4 stages q across A||At and b across t2||t3 as single contiguous
// 64x128 bf16 buffers (sw_off uses 128-element rows); the pairs must stay
// adjacent with no padding or the staged operands silently scramble.
static_assert(offsetof(SharedStorage, At) ==
              offsetof(SharedStorage, A) + sizeof(bf16) * kBT * kBT,
              "A||At must be contiguous for P4 q staging");
static_assert(offsetof(SharedStorage, t3) ==
              offsetof(SharedStorage, t2) + sizeof(bf16) * kBT * kBV,
              "t2||t3 must be contiguous for P4 b staging");

__global__ void __launch_bounds__(kWarps * 32, 1)
gdn2_wy_dqkg_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ v_new,
    const float* __restrict__ g,
    const bf16* __restrict__ b,
    const bf16* __restrict__ wg,
    const bf16* __restrict__ A,
    const bf16* __restrict__ h,
    const bf16* __restrict__ dov,
    const bf16* __restrict__ dh,
    const bf16* __restrict__ dvin,
    float* __restrict__ dq_o,
    float* __restrict__ dk_o,
    bf16* __restrict__ dv2_o,
    float* __restrict__ dg_o,
    float* __restrict__ db_o,
    bf16* __restrict__ dw_o,
    float* __restrict__ dA_o,
    int T, int H, int NT, float scale) {
  extern __shared__ char smem_raw[];
  SharedStorage& sm = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int i_t = blockIdx.x;
  const int i_bh = blockIdx.y;
  const int i_b = i_bh / H, i_h = i_bh % H;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int s = warp & 3;          // row strip
  const int hf = warp >> 2;        // column half
  const int r0 = s * 16 + (lane >> 2);
  const int r1 = r0 + 8;
  const long bos = (long)i_b * T;
  const int t_lo = i_t * kBT;
  const int rows_here = min(kBT, T - t_lo);
  if (rows_here <= 0) return;
  const int last_row = rows_here - 1;

  // Row-major token pointers: X[(bos + t)*H + i_h]*D + d
  auto rowp = [&](const bf16* base, int t, int d) -> const bf16* {
    return base + ((bos + t_lo + t) * H + i_h) * (long)kD + d;
  };

  // ---- stage A once ----
  for (int idx = threadIdx.x; idx < kBT * (kBT / 8); idx += kWarps * 32) {
    const int r = idx >> 3, c8 = (idx & 7) << 3;
    const bool ok = r < rows_here;
    const bf16* src = A + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kBT + c8;
    cp_async_16(smem_u32(&sm.A[sw_off64(r, c8)]), src, ok);
  }
  cp_async_commit();

  float dq_acc[32], dk_acc[32], dwf_acc[32], dA_acc[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) { dq_acc[i] = dk_acc[i] = dwf_acc[i] = dA_acc[i] = 0.f; }

  const long hbase = (((long)i_b * NT + i_t) * H + i_h) * kD * kD;

  cp_async_wait<0>();
  __syncthreads();
  // The WY backward multiplies by A^T (gradient of A @ x): build the
  // transposed tile once (Triton reads it via a transposed block pointer).
  for (int idx = threadIdx.x; idx < kBT * kBT; idx += kWarps * 32) {
    const int r = idx / kBT, c = idx % kBT;
    sm.At[sw_off64(r, c)] = sm.A[sw_off64(c, r)];
  }
  __syncthreads();

  // =============================== P1: V loop ===============================
  for (int iv = 0; iv < kD / kBV; ++iv) {
    const int v0 = iv * kBV;
    // stage v_new, do, dv, v*wg (t0..t3) and h/dh [K rows, BV cols]
    for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kWarps * 32) {
      const int r = idx >> 3, c8 = (idx & 7) << 3;
      const bool ok = r < rows_here;
      cp_async_16(smem_u32(&sm.u.p1.t0[sw_off64(r, c8)]), rowp(v_new, r, v0 + c8), ok);
      cp_async_16(smem_u32(&sm.u.p1.t1[sw_off64(r, c8)]), rowp(dov, r, v0 + c8), ok);
      cp_async_16(smem_u32(&sm.t2[sw_off64(r, c8)]), rowp(dvin, r, v0 + c8), ok);
    }
    for (int idx = threadIdx.x; idx < kD * (kBV / 8); idx += kWarps * 32) {
      const int r = idx >> 3, c8 = (idx & 7) << 3;
      cp_async_16(smem_u32(&sm.h_t[sw_off64(r, c8)]), h + hbase + (long)r * kD + v0 + c8, true);
      cp_async_16(smem_u32(&sm.dh_t[sw_off64(r, c8)]), dh + hbase + (long)r * kD + v0 + c8, true);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    // v*wg into t3: vectorized 8-elem gmem reads, swizzled smem writes.
    for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kWarps * 32) {
      const int r = idx >> 3, c8 = (idx & 7) << 3;
      bf16 res[8];
      if (r < rows_here) {
        uint4 xv = *reinterpret_cast<const uint4*>(rowp(v, r, v0 + c8));
        uint4 xw = *reinterpret_cast<const uint4*>(rowp(wg, r, v0 + c8));
        const bf16* pv = reinterpret_cast<const bf16*>(&xv);
        const bf16* pw = reinterpret_cast<const bf16*>(&xw);
        #pragma unroll
        for (int e = 0; e < 8; ++e)
          res[e] = __float2bfloat16(__bfloat162float(pv[e]) * __bfloat162float(pw[e]));
      } else {
        #pragma unroll
        for (int e = 0; e < 8; ++e) res[e] = __float2bfloat16(0.f);
      }
      *reinterpret_cast<uint4*>(&sm.t3[sw_off64(r, c8)]) = *reinterpret_cast<uint4*>(res);
    }
    __syncthreads();

    // A-fragments of this strip's rows for do/v_new/dv (k over BV).
    // dq += do @ h^T ; dk += v_new @ dh^T ; dwf += dv @ h^T
    #pragma unroll
    for (int kc = 0; kc < kBV / 16; ++kc) {
      unsigned a_do[4], a_vn[4], a_dv[4];
      {
        const int rr = s * 16 + (lane & 15);
        const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
        ldmatrix_x4(a_do, smem_u32(&sm.u.p1.t1[sw_off64(rr, cc)]));
        ldmatrix_x4(a_vn, smem_u32(&sm.u.p1.t0[sw_off64(rr, cc)]));
        ldmatrix_x4(a_dv, smem_u32(&sm.t2[sw_off64(rr, cc)]));
      }
      #pragma unroll
      for (int n = 0; n < 8; ++n) {  // 8 n-tiles cover this K-half (64 cols)
        const int kcol = hf * 64 + n * 8;
        // B (kdim = v rows of h_t, n = k cols): consecutive v at fixed k row
        // -> h_t rows are k; TRANS gives consecutive source-rows... h_t is
        // [K rows, BV cols]: we need B[kdim=v, n=kcol] = h_t[kcol, v] ->
        // consecutive v within h_t row kcol -> non-trans over h_t rows kcol.
        const int lrow = kcol + (lane & 7);
        const int lcol = kc * 16 + ((lane & 8) ? 8 : 0);
        unsigned bh[2], bdh[2];
        ldmatrix_x2(bh, smem_u32(&sm.h_t[sw_off64(lrow, lcol)]));
        ldmatrix_x2(bdh, smem_u32(&sm.dh_t[sw_off64(lrow, lcol)]));
        float* dqf = &dq_acc[n * 4];
        float* dkf = &dk_acc[n * 4];
        float* dwff = &dwf_acc[n * 4];
        mma_16x8x16(a_do, bh, dqf);
        mma_16x8x16(a_vn, bdh, dkf);
        mma_16x8x16(a_dv, bh, dwff);
      }
      // dA += dv @ (v*wg)^T over this V-half's k-range (split by hf):
      // B (kdim = v, n = t') from t3 rows t' (consecutive v in-row).
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int tcol = n * 8;  // t' columns 0..63
        const int lrow = tcol + (lane & 7);
        const int lcol = kc * 16 + ((lane & 8) ? 8 : 0);
        unsigned bvw[2];
        ldmatrix_x2(bvw, smem_u32(&sm.t3[sw_off64(lrow, lcol)]));
        // accumulate only on hf==iv? No: split k(v) ranges via kc over BV of
        // THIS iv; both halves accumulate distinct V ranges when iv==hf.
        if (iv == hf) {
          float* dAf_ = &dA_acc[n * 4];
          mma_16x8x16(a_dv, bvw, dAf_);
        }
      }
    }

    // dgk += colsum(h * dh): 8 warps x 16 K-rows, lanes over V columns.
    {
      const int krow0 = (warp) * 16;  // 8 warps x 16 = 128 K rows
      float accv = 0.f;
      for (int rr = 0; rr < 16; ++rr) {
        const int krow = krow0 + rr;
        // lane sums 2 V columns
        const int c0 = lane * 2;
        float hv0 = __bfloat162float(sm.h_t[sw_off64(krow, c0)]);
        float dv0 = __bfloat162float(sm.dh_t[sw_off64(krow, c0)]);
        float hv1 = __bfloat162float(sm.h_t[sw_off64(krow, c0 + 1)]);
        float dv1 = __bfloat162float(sm.dh_t[sw_off64(krow, c0 + 1)]);
        accv = hv0 * dv0 + hv1 * dv1;
        // reduce across the warp: every lane holds a partial for THIS krow
        #pragma unroll
        for (int w = 16; w >= 1; w >>= 1)
          accv += __shfl_xor_sync(mask_all(), accv, w);
        if (lane == 0) sm.red[krow] = (iv == 0 ? 0.f : sm.red[krow]) + accv;
      }
    }
    __syncthreads();

    // dvb = A @ dv -> dv2 = dvb*wg, dw = dvb*v (this iv's 64 V cols; warps
    // split rows by strip and V-cols by half: [16 x 32] each).
    {
      float dvb[16];  // [16 rows x 32 cols]/32 lanes = 16 f32
      #pragma unroll
      for (int i = 0; i < 16; ++i) dvb[i] = 0.f;
      #pragma unroll
      for (int kc = 0; kc < kBT / 16; ++kc) {
        unsigned a_A[4];
        const int rr = s * 16 + (lane & 15);
        const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
        ldmatrix_x4(a_A, smem_u32(&sm.At[sw_off64(rr, cc)]));
        #pragma unroll
        for (int n = 0; n < 4; ++n) {  // 4 n-tiles = 32 cols (this half)
          const int vcol = hf * 32 + n * 8;
          // B (kdim = t' rows of dv, n = vcol): consecutive t' at fixed vcol
          // -> TRANS over t2 rows.
          const int lrow = kc * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
          unsigned bdv[2];
          ldmatrix_x2_trans(bdv, smem_u32(&sm.t2[sw_off64(lrow, vcol)]));
          mma_16x8x16(a_A, bdv, &dvb[n * 4]);
        }
      }
      // Stage dvb (fp32, exact) into the dead t0+t1 space, then one
      // coalesced vectorized pass computes dv2 = dvb*wg and dw = dvb*v —
      // the fragment-order scalar stores were the kernel's biggest
      // memory-inefficiency per NCU.
      float* dvb_s = reinterpret_cast<float*>(&sm.u.p1);  // [kBT][kBV] fp32
      #pragma unroll
      for (int n = 0; n < 4; ++n) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          #pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int trow = s * 16 + (lane >> 2) + rr * 8;
            const int vcol = hf * 32 + n * 8 + (lane & 3) * 2 + e;
            dvb_s[trow * kBV + vcol] = dvb[n * 4 + rr * 2 + e];
          }
        }
      }
      __syncthreads();
      for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kWarps * 32) {
        const int r = idx >> 3, c8 = (idx & 7) << 3;
        if (r >= rows_here) continue;
        uint4 xv = *reinterpret_cast<const uint4*>(rowp(v, r, v0 + c8));
        uint4 xw = *reinterpret_cast<const uint4*>(rowp(wg, r, v0 + c8));
        const bf16* pv = reinterpret_cast<const bf16*>(&xv);
        const bf16* pw = reinterpret_cast<const bf16*>(&xw);
        bf16 o2[8], ow[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float x = dvb_s[r * kBV + c8 + e];
          o2[e] = __float2bfloat16(x * __bfloat162float(pw[e]));
          ow[e] = __float2bfloat16(x * __bfloat162float(pv[e]));
        }
        *reinterpret_cast<uint4*>(
            &dv2_o[((bos + t_lo + r) * H + i_h) * (long)kD + v0 + c8]) =
            *reinterpret_cast<uint4*>(o2);
        *reinterpret_cast<uint4*>(
            &dw_o[((bos + t_lo + r) * H + i_h) * (long)kD + v0 + c8]) =
            *reinterpret_cast<uint4*>(ow);
      }
    }
    __syncthreads();
  }

  // =============================== P2 ===============================
  // stage k -> h_t, b -> dh_t (repurposed slots), fp32 g tile.
  // k/b/q are [64,128]: reuse t0/t1/t2 as [64,64] PER HALF? Instead stage the
  // full 128 cols into h_t/dh_t (128x64 slots repurposed as 64x128):
  for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kWarps * 32) {
    const int r = idx >> 4, c8 = (idx & 15) << 3;
    const bool ok = r < rows_here;
    cp_async_16(smem_u32(&sm.h_t[sw_off(r, c8)]), rowp(k, r, c8), ok);   // k
    cp_async_16(smem_u32(&sm.dh_t[sw_off(r, c8)]), rowp(b, r, c8), ok);  // b
  }
  for (int idx = threadIdx.x; idx < kBT * (kD / 4); idx += kWarps * 32) {
    const int r = idx >> 5, c4 = (idx & 31) << 2;
    const bool ok = r < rows_here;
    cp_async_16(smem_u32(&sm.u.gtile[r * kD + c4]),
                g + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kD + c4, ok);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  // per-lane columns of the K-half: c = hf*64 + (lane&3)*2 + e + n*8-ish;
  // reuse the accumulator layout: entry [n*4 + rr*2 + e] is
  // (row = s*16 + (lane>>2) + rr*8, col = hf*64 + n*8 + (lane&3)*2 + e).
  const float gn0 = sm.u.gtile[last_row * kD + hf * 64 + 0];  // per-col below
  (void)gn0;

  float dkgb[32];
  {
    // dwf = -dwf; store dwf (bf16) to smem t3-as-[64x64 half] for the two
    // A-consuming dots; also compute dA += dwf @ (kg*b)^T via C->A reuse.
    // Each half-warp holds its OWN K-half of dwf in registers, so its dA
    // partial needs kgb columns from THAT K-half: stage the lower K-half
    // into t2 and the upper into t3 (same two-buffer rule the dwf staging
    // below already documents; t2/dv is dead here, both tiles re-clobbered
    // only after the dA dot). The previous single-tile version indexed the
    // source by the WRITER thread's half (gc = hf*64 + c) with a destination
    // keyed only on (r, c) — a period-4 K-half scramble of kgb that biased
    // dA (and through it db/dg/dk) by ~10% in a structured, non-zero-mean
    // way: the erase/decay-gate gradient bias behind the slow state-norm
    // runaway (o_norm absmax march -> NaN) in long training runs.
    __syncthreads();
    for (int idx = threadIdx.x; idx < kBT * kBV; idx += kWarps * 32) {
      const int r = idx / kBV, c = idx % kBV;      // c within half
      const float gg0 = (r < rows_here) ? exp2f(sm.u.gtile[r * kD + c]) : 0.f;
      const float gg1 = (r < rows_here) ? exp2f(sm.u.gtile[r * kD + 64 + c]) : 0.f;
      const float kk0 = __bfloat162float(sm.h_t[sw_off(r, c)]);
      const float bb0 = __bfloat162float(sm.dh_t[sw_off(r, c)]);
      const float kk1 = __bfloat162float(sm.h_t[sw_off(r, 64 + c)]);
      const float bb1 = __bfloat162float(sm.dh_t[sw_off(r, 64 + c)]);
      sm.t2[sw_off64(r, c)] = __float2bfloat16(kk0 * gg0 * bb0);  // kg*b, K-half 0
      sm.t3[sw_off64(r, c)] = __float2bfloat16(kk1 * gg1 * bb1);  // kg*b, K-half 1
    }
    __syncthreads();

    // dA += (-dwf) @ (kgb)^T : A-frags = dwf C-fragments (negated), B from
    // this half-warp's OWN K-half tile.
    bf16* kgb_tile = hf ? sm.t3 : sm.t2;
    #pragma unroll
    for (int kc = 0; kc < kBV / 16; ++kc) {
      unsigned a_dwf[4];
      a_dwf[0] = pack_f2(-dwf_acc[(2 * kc) * 4 + 0], -dwf_acc[(2 * kc) * 4 + 1]);
      a_dwf[1] = pack_f2(-dwf_acc[(2 * kc) * 4 + 2], -dwf_acc[(2 * kc) * 4 + 3]);
      a_dwf[2] = pack_f2(-dwf_acc[(2 * kc + 1) * 4 + 0], -dwf_acc[(2 * kc + 1) * 4 + 1]);
      a_dwf[3] = pack_f2(-dwf_acc[(2 * kc + 1) * 4 + 2], -dwf_acc[(2 * kc + 1) * 4 + 3]);
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int tcol = n * 8;
        const int lrow = tcol + (lane & 7);
        const int lcol = kc * 16 + ((lane & 8) ? 8 : 0);
        unsigned bkgb[2];
        ldmatrix_x2(bkgb, smem_u32(&kgb_tile[sw_off64(lrow, lcol)]));
        mma_16x8x16(a_dwf, bkgb, &dA_acc[n * 4]);
      }
    }

    // store -dwf (bf16): each column-half needs its own 64x64 buffer or the
    // two halves clobber each other -> hf==0 uses t2, hf==1 uses t3 (kgb is
    // dead after the dA dot above).
    __syncthreads();
    bf16* dwf_buf = hf ? sm.t3 : sm.t2;
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int c = n * 8 + (lane & 3) * 2 + e;
          dwf_buf[sw_off64(trow, c)] = __float2bfloat16(-dwf_acc[n * 4 + rr * 2 + e]);
        }
      }
    }
    __syncthreads();

    // dkgb = A @ (-dwf half): [64,64]@[64,64-half]
    #pragma unroll
    for (int i = 0; i < 32; ++i) dkgb[i] = 0.f;
    #pragma unroll
    for (int kc = 0; kc < kBT / 16; ++kc) {
      unsigned a_A[4];
      const int rr = s * 16 + (lane & 15);
      const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
      ldmatrix_x4(a_A, smem_u32(&sm.At[sw_off64(rr, cc)]));
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int kcol = n * 8;
        const int lrow = kc * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
        unsigned bw[2];
        ldmatrix_x2_trans(bw, smem_u32(&dwf_buf[sw_off64(lrow, kcol)]));
        mma_16x8x16(a_A, bw, &dkgb[n * 4]);
      }
    }
  }

  // gate algebra + stores for this warp's [16 rows x 64 half-cols].
  {
    // finish dgk: dgk_full[k] = red[k]*exp2(gn[k]) + colsum(k*dk_t)[k]
    // colsum needs cross-strip reduction: accumulate per-warp partials into
    // red[kWarps*..] then combine. Layout: red2[c] per K col.
    __syncthreads();
    // compute per-element pieces and per-col partial sums of kdk
    float col_kdk[2] = {0.f, 0.f};
    float dq_out[32], dk_t[32];
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int gc = hf * 64 + n * 8 + (lane & 3) * 2 + e;
          const bool rv = trow < rows_here;
          const float gg = rv ? sm.u.gtile[trow * kD + gc] : 0.f;
          const float gnn = sm.u.gtile[last_row * kD + gc];
          const float ge = exp2f(gg);
          dq_out[i] = dq_acc[i] * ge * scale;
          const float dkm = rv ? dk_acc[i] * exp2f(gnn - gg) : 0.f;
          dk_t[i] = dkm;
        }
      }
    }
    (void)col_kdk;
    // write dq
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int gc = hf * 64 + n * 8 + (lane & 3) * 2 + e;
          if (trow < rows_here)
            dq_o[((bos + t_lo + trow) * H + i_h) * (long)kD + gc] = dq_out[i];
        }

    // kdk colsums (atomic version, reverted while bisecting)
    float* red2 = sm.red + kD;
    for (int idx = threadIdx.x; idx < kD; idx += kWarps * 32) red2[idx] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int c = n * 8 + (lane & 3) * 2 + e;
          const float kk = (trow < rows_here)
              ? __bfloat162float(sm.h_t[sw_off(trow, hf * 64 + c)]) : 0.f;
          atomicAdd(&red2[(hf * 64 + c)], kk * dk_t[i]);
        }
    __syncthreads();

    // final per-element outputs
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int cin = n * 8 + (lane & 3) * 2 + e;
          const int gc = hf * 64 + cin;
          if (trow >= rows_here) continue;
          const float gg = sm.u.gtile[trow * kD + gc];
          const float gnn = sm.u.gtile[last_row * kD + gc];
          const float ge = exp2f(gg);
          const float kk = __bfloat162float(sm.h_t[sw_off(trow, gc)]);
          const float bb = __bfloat162float(sm.dh_t[sw_off(trow, gc)]);
          const float qq = __bfloat162float(*rowp(q, trow, gc));
          const float kg = kk * ge;
          const float kdk = kk * dk_t[i];
          const float dgk_full = sm.red[gc] * exp2f(gnn) + red2[gc];
          const float dkgb_v = dkgb[i];
          const float m_last = (trow == last_row) ? 1.f : 0.f;
          const float dgv = qq * dq_out[i] - kdk + m_last * dgk_full
                            + kg * dkgb_v * bb;
          const float dkv = dk_t[i] + dkgb_v * ge * bb;
          const float dbv = dkgb_v * kg;
          const long off = ((bos + t_lo + trow) * H + i_h) * (long)kD + gc;
          dk_o[off] = dkv;
          dg_o[off] = dgv;
          db_o[off] = dbv;
        }
  }

  // =============================== P3: dA ===============================
  // assemble full fp32 dA from per-warp partials, then dA = -tril(A@(tril(dA)@A)).
  __syncthreads();
  // hf==0 writes its partial, hf==1 adds after a barrier: same [16,64]
  // strip regions, disjoint by construction — no atomics needed.
  if (hf == 0) {
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int tcol = n * 8 + (lane & 3) * 2 + e;
          sm.u.dAf[trow * kBT + tcol] = dA_acc[i];
        }
  }
  __syncthreads();
  if (hf == 1) {
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int tcol = n * 8 + (lane & 3) * 2 + e;
          sm.u.dAf[trow * kBT + tcol] += dA_acc[i];
        }
  }
  __syncthreads();
  // mask strict lower + valid, to bf16 in t2
  for (int idx = threadIdx.x; idx < kBT * kBT; idx += kWarps * 32) {
    const int r = idx / kBT, c = idx % kBT;
    const bool m = (r > c) && (r < rows_here) && (c < rows_here);
    sm.t2[sw_off64(r, c)] = __float2bfloat16(m ? sm.u.dAf[idx] : 0.f);
  }
  __syncthreads();
  // step1 = dA_m @ A  (each warp: strip rows x 64 cols? use pairs: strip x half)
  float st1[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) st1[i] = 0.f;
  #pragma unroll
  for (int kc = 0; kc < kBT / 16; ++kc) {
    unsigned a_dA[4];
    const int rr = s * 16 + (lane & 15);
    const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
    ldmatrix_x4(a_dA, smem_u32(&sm.t2[sw_off64(rr, cc)]));
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      // st1[t,c] = sum_j dA[t,j] * A[c][j]: B fragment reads consecutive j
      // within A's row c -> non-trans over the ORIGINAL A rows.
      const int lrow = n * 8 + (lane & 7);
      const int lcol = kc * 16 + ((lane & 8) ? 8 : 0);
      unsigned bA[2];
      ldmatrix_x2(bA, smem_u32(&sm.A[sw_off64(lrow, lcol)]));
      mma_16x8x16(a_dA, bA, &st1[n * 4]);
    }
  }
  // both halves computed the same st1; halve later or just let hf==0 write
  __syncthreads();
  if (hf == 0) {
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int tcol = n * 8 + (lane & 3) * 2 + e;
          sm.dh_t[sw_off64(trow, tcol)] = __float2bfloat16(st1[n * 4 + rr * 2 + e]);
        }
  }
  __syncthreads();
  // step2 = A @ step1
  float st2[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) st2[i] = 0.f;
  #pragma unroll
  for (int kc = 0; kc < kBT / 16; ++kc) {
    unsigned a_A[4];
    const int rr = s * 16 + (lane & 15);
    const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
    ldmatrix_x4(a_A, smem_u32(&sm.At[sw_off64(rr, cc)]));
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int col = n * 8;
      const int lrow = kc * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
      unsigned bS[2];
      ldmatrix_x2_trans(bS, smem_u32(&sm.dh_t[sw_off64(lrow, col)]));
      mma_16x8x16(a_A, bS, &st2[n * 4]);
    }
  }
  if (hf == 0) {
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int tcol = n * 8 + (lane & 3) * 2 + e;
          const bool m = (trow > tcol) && (trow < rows_here) && (tcol < rows_here);
          if (trow < rows_here)
            dA_o[((bos + t_lo + trow) * H + i_h) * (long)kBT + tcol] =
                m ? -st2[n * 4 + rr * 2 + e] : 0.f;
        }
  }
}

// ============================================================================
// Merged kernel: wy_dqkg backward + intra-chunk backward in one pass.
// P1-P3 mirror gdn2_wy_dqkg_kernel (wy dq/dk go to fp32 workspaces, the
// masked fp32 dAkk is finalized in smem instead of gmem). P4 adds the
// intra-chunk backward with upstream's exact math: two-step first/last-row
// anchored off-diagonal blocks (all exp2 arguments <= 0 by within-chunk
// monotonicity of g, so no overflow is possible) and EXACT per-(t, j,
// channel) scalar diagonals — no anchored diagonal algebra (that
// approximation was the step-~1000 NaN engine). Final dq/dk are the
// wy+intra sums rounded once to bf16; db/dg are read-modify-write on the
// wy values. Structure matches the Triton pair term-for-term.
// ============================================================================
__global__ void __launch_bounds__(kWarps * 32, 1)
gdn2_wy_intra_bwd_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ v_new,
    const float* __restrict__ g,
    const bf16* __restrict__ b,
    const bf16* __restrict__ wg,
    const bf16* __restrict__ A,
    const bf16* __restrict__ h,
    const bf16* __restrict__ dov,
    const bf16* __restrict__ dh,
    const bf16* __restrict__ dvin,
    const float* __restrict__ dAqk_g,
    float* __restrict__ dq_ws,
    float* __restrict__ dk_ws,
    bf16* __restrict__ dq_o,
    bf16* __restrict__ dk_o,
    bf16* __restrict__ dv2_o,
    float* __restrict__ dg_o,
    float* __restrict__ db_o,
    bf16* __restrict__ dw_o,
    int T, int H, int NT, float scale) {
  extern __shared__ char smem_raw[];
  SharedStorage& sm = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int i_t = blockIdx.x;
  const int i_bh = blockIdx.y;
  const int i_b = i_bh / H, i_h = i_bh % H;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int s = warp & 3;          // row strip == sub-chunk (BC = 16)
  const int hf = warp >> 2;        // column half
  const long bos = (long)i_b * T;
  const int t_lo = i_t * kBT;
  const int rows_here = min(kBT, T - t_lo);
  if (rows_here <= 0) return;
  const int last_row = rows_here - 1;

  auto rowp = [&](const bf16* base, int t, int d) -> const bf16* {
    return base + ((bos + t_lo + t) * H + i_h) * (long)kD + d;
  };

  // ---- stage A once ----
  for (int idx = threadIdx.x; idx < kBT * (kBT / 8); idx += kWarps * 32) {
    const int r = idx >> 3, c8 = (idx & 7) << 3;
    const bool ok = r < rows_here;
    const bf16* src = A + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kBT + c8;
    cp_async_16(smem_u32(&sm.A[sw_off64(r, c8)]), src, ok);
  }
  cp_async_commit();

  float dq_acc[32], dk_acc[32], dwf_acc[32], dA_acc[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) { dq_acc[i] = dk_acc[i] = dwf_acc[i] = dA_acc[i] = 0.f; }

  const long hbase = (((long)i_b * NT + i_t) * H + i_h) * kD * kD;

  cp_async_wait<0>();
  __syncthreads();
  for (int idx = threadIdx.x; idx < kBT * kBT; idx += kWarps * 32) {
    const int r = idx / kBT, c = idx % kBT;
    sm.At[sw_off64(r, c)] = sm.A[sw_off64(c, r)];
  }
  __syncthreads();

  // =============================== P1: V loop ===============================
  for (int iv = 0; iv < kD / kBV; ++iv) {
    const int v0 = iv * kBV;
    for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kWarps * 32) {
      const int r = idx >> 3, c8 = (idx & 7) << 3;
      const bool ok = r < rows_here;
      cp_async_16(smem_u32(&sm.u.p1.t0[sw_off64(r, c8)]), rowp(v_new, r, v0 + c8), ok);
      cp_async_16(smem_u32(&sm.u.p1.t1[sw_off64(r, c8)]), rowp(dov, r, v0 + c8), ok);
      cp_async_16(smem_u32(&sm.t2[sw_off64(r, c8)]), rowp(dvin, r, v0 + c8), ok);
    }
    for (int idx = threadIdx.x; idx < kD * (kBV / 8); idx += kWarps * 32) {
      const int r = idx >> 3, c8 = (idx & 7) << 3;
      cp_async_16(smem_u32(&sm.h_t[sw_off64(r, c8)]), h + hbase + (long)r * kD + v0 + c8, true);
      cp_async_16(smem_u32(&sm.dh_t[sw_off64(r, c8)]), dh + hbase + (long)r * kD + v0 + c8, true);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kWarps * 32) {
      const int r = idx >> 3, c8 = (idx & 7) << 3;
      bf16 res[8];
      if (r < rows_here) {
        uint4 xv = *reinterpret_cast<const uint4*>(rowp(v, r, v0 + c8));
        uint4 xw = *reinterpret_cast<const uint4*>(rowp(wg, r, v0 + c8));
        const bf16* pv = reinterpret_cast<const bf16*>(&xv);
        const bf16* pw = reinterpret_cast<const bf16*>(&xw);
        #pragma unroll
        for (int e = 0; e < 8; ++e)
          res[e] = __float2bfloat16(__bfloat162float(pv[e]) * __bfloat162float(pw[e]));
      } else {
        #pragma unroll
        for (int e = 0; e < 8; ++e) res[e] = __float2bfloat16(0.f);
      }
      *reinterpret_cast<uint4*>(&sm.t3[sw_off64(r, c8)]) = *reinterpret_cast<uint4*>(res);
    }
    __syncthreads();

    #pragma unroll
    for (int kc = 0; kc < kBV / 16; ++kc) {
      unsigned a_do[4], a_vn[4], a_dv[4];
      {
        const int rr = s * 16 + (lane & 15);
        const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
        ldmatrix_x4(a_do, smem_u32(&sm.u.p1.t1[sw_off64(rr, cc)]));
        ldmatrix_x4(a_vn, smem_u32(&sm.u.p1.t0[sw_off64(rr, cc)]));
        ldmatrix_x4(a_dv, smem_u32(&sm.t2[sw_off64(rr, cc)]));
      }
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int kcol = hf * 64 + n * 8;
        const int lrow = kcol + (lane & 7);
        const int lcol = kc * 16 + ((lane & 8) ? 8 : 0);
        unsigned bh[2], bdh[2];
        ldmatrix_x2(bh, smem_u32(&sm.h_t[sw_off64(lrow, lcol)]));
        ldmatrix_x2(bdh, smem_u32(&sm.dh_t[sw_off64(lrow, lcol)]));
        mma_16x8x16(a_do, bh, &dq_acc[n * 4]);
        mma_16x8x16(a_vn, bdh, &dk_acc[n * 4]);
        mma_16x8x16(a_dv, bh, &dwf_acc[n * 4]);
      }
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int tcol = n * 8;
        const int lrow = tcol + (lane & 7);
        const int lcol = kc * 16 + ((lane & 8) ? 8 : 0);
        unsigned bvw[2];
        ldmatrix_x2(bvw, smem_u32(&sm.t3[sw_off64(lrow, lcol)]));
        if (iv == hf) {
          mma_16x8x16(a_dv, bvw, &dA_acc[n * 4]);
        }
      }
    }

    {
      const int krow0 = (warp) * 16;
      float accv = 0.f;
      for (int rr = 0; rr < 16; ++rr) {
        const int krow = krow0 + rr;
        const int c0 = lane * 2;
        float hv0 = __bfloat162float(sm.h_t[sw_off64(krow, c0)]);
        float dv0 = __bfloat162float(sm.dh_t[sw_off64(krow, c0)]);
        float hv1 = __bfloat162float(sm.h_t[sw_off64(krow, c0 + 1)]);
        float dv1 = __bfloat162float(sm.dh_t[sw_off64(krow, c0 + 1)]);
        accv = hv0 * dv0 + hv1 * dv1;
        #pragma unroll
        for (int w = 16; w >= 1; w >>= 1)
          accv += __shfl_xor_sync(mask_all(), accv, w);
        if (lane == 0) sm.red[krow] = (iv == 0 ? 0.f : sm.red[krow]) + accv;
      }
    }
    __syncthreads();

    {
      float dvb[16];
      #pragma unroll
      for (int i = 0; i < 16; ++i) dvb[i] = 0.f;
      #pragma unroll
      for (int kc = 0; kc < kBT / 16; ++kc) {
        unsigned a_A[4];
        const int rr = s * 16 + (lane & 15);
        const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
        ldmatrix_x4(a_A, smem_u32(&sm.At[sw_off64(rr, cc)]));
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
          const int vcol = hf * 32 + n * 8;
          const int lrow = kc * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
          unsigned bdv[2];
          ldmatrix_x2_trans(bdv, smem_u32(&sm.t2[sw_off64(lrow, vcol)]));
          mma_16x8x16(a_A, bdv, &dvb[n * 4]);
        }
      }
      float* dvb_s = reinterpret_cast<float*>(&sm.u.p1);
      #pragma unroll
      for (int n = 0; n < 4; ++n) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          #pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int trow = s * 16 + (lane >> 2) + rr * 8;
            const int vcol = hf * 32 + n * 8 + (lane & 3) * 2 + e;
            dvb_s[trow * kBV + vcol] = dvb[n * 4 + rr * 2 + e];
          }
        }
      }
      __syncthreads();
      for (int idx = threadIdx.x; idx < kBT * (kBV / 8); idx += kWarps * 32) {
        const int r = idx >> 3, c8 = (idx & 7) << 3;
        if (r >= rows_here) continue;
        uint4 xv = *reinterpret_cast<const uint4*>(rowp(v, r, v0 + c8));
        uint4 xw = *reinterpret_cast<const uint4*>(rowp(wg, r, v0 + c8));
        const bf16* pv = reinterpret_cast<const bf16*>(&xv);
        const bf16* pw = reinterpret_cast<const bf16*>(&xw);
        bf16 o2[8], ow[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float x = dvb_s[r * kBV + c8 + e];
          o2[e] = __float2bfloat16(x * __bfloat162float(pw[e]));
          ow[e] = __float2bfloat16(x * __bfloat162float(pv[e]));
        }
        *reinterpret_cast<uint4*>(
            &dv2_o[((bos + t_lo + r) * H + i_h) * (long)kD + v0 + c8]) =
            *reinterpret_cast<uint4*>(o2);
        *reinterpret_cast<uint4*>(
            &dw_o[((bos + t_lo + r) * H + i_h) * (long)kD + v0 + c8]) =
            *reinterpret_cast<uint4*>(ow);
      }
    }
    __syncthreads();
  }

  // =============================== P2 ===============================
  for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kWarps * 32) {
    const int r = idx >> 4, c8 = (idx & 15) << 3;
    const bool ok = r < rows_here;
    cp_async_16(smem_u32(&sm.h_t[sw_off(r, c8)]), rowp(k, r, c8), ok);   // k
    cp_async_16(smem_u32(&sm.dh_t[sw_off(r, c8)]), rowp(b, r, c8), ok);  // b
  }
  for (int idx = threadIdx.x; idx < kBT * (kD / 4); idx += kWarps * 32) {
    const int r = idx >> 5, c4 = (idx & 31) << 2;
    const bool ok = r < rows_here;
    cp_async_16(smem_u32(&sm.u.gtile[r * kD + c4]),
                g + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kD + c4, ok);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  float dkgb[32];
  {
    // kgb staging: two K-half tiles (t2 = lower, t3 = upper) so each
    // half-warp's dA partial contracts its OWN dwf K-half against the same
    // K-half of kgb. See the standalone kernel above for the full account
    // of the writer-parity scramble this replaces (biased db/dg ->
    // state-norm runaway in long runs).
    __syncthreads();
    for (int idx = threadIdx.x; idx < kBT * kBV; idx += kWarps * 32) {
      const int r = idx / kBV, c = idx % kBV;
      const float gg0 = (r < rows_here) ? exp2f(sm.u.gtile[r * kD + c]) : 0.f;
      const float gg1 = (r < rows_here) ? exp2f(sm.u.gtile[r * kD + 64 + c]) : 0.f;
      const float kk0 = __bfloat162float(sm.h_t[sw_off(r, c)]);
      const float bb0 = __bfloat162float(sm.dh_t[sw_off(r, c)]);
      const float kk1 = __bfloat162float(sm.h_t[sw_off(r, 64 + c)]);
      const float bb1 = __bfloat162float(sm.dh_t[sw_off(r, 64 + c)]);
      sm.t2[sw_off64(r, c)] = __float2bfloat16(kk0 * gg0 * bb0);  // kg*b half 0
      sm.t3[sw_off64(r, c)] = __float2bfloat16(kk1 * gg1 * bb1);  // kg*b half 1
    }
    __syncthreads();

    bf16* kgb_tile = hf ? sm.t3 : sm.t2;
    #pragma unroll
    for (int kc = 0; kc < kBV / 16; ++kc) {
      unsigned a_dwf[4];
      a_dwf[0] = pack_f2(-dwf_acc[(2 * kc) * 4 + 0], -dwf_acc[(2 * kc) * 4 + 1]);
      a_dwf[1] = pack_f2(-dwf_acc[(2 * kc) * 4 + 2], -dwf_acc[(2 * kc) * 4 + 3]);
      a_dwf[2] = pack_f2(-dwf_acc[(2 * kc + 1) * 4 + 0], -dwf_acc[(2 * kc + 1) * 4 + 1]);
      a_dwf[3] = pack_f2(-dwf_acc[(2 * kc + 1) * 4 + 2], -dwf_acc[(2 * kc + 1) * 4 + 3]);
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int tcol = n * 8;
        const int lrow = tcol + (lane & 7);
        const int lcol = kc * 16 + ((lane & 8) ? 8 : 0);
        unsigned bkgb[2];
        ldmatrix_x2(bkgb, smem_u32(&kgb_tile[sw_off64(lrow, lcol)]));
        mma_16x8x16(a_dwf, bkgb, &dA_acc[n * 4]);
      }
    }

    __syncthreads();
    bf16* dwf_buf = hf ? sm.t3 : sm.t2;
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int c = n * 8 + (lane & 3) * 2 + e;
          dwf_buf[sw_off64(trow, c)] = __float2bfloat16(-dwf_acc[n * 4 + rr * 2 + e]);
        }
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 32; ++i) dkgb[i] = 0.f;
    #pragma unroll
    for (int kc = 0; kc < kBT / 16; ++kc) {
      unsigned a_A[4];
      const int rr = s * 16 + (lane & 15);
      const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
      ldmatrix_x4(a_A, smem_u32(&sm.At[sw_off64(rr, cc)]));
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int kcol = n * 8;
        const int lrow = kc * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
        unsigned bw[2];
        ldmatrix_x2_trans(bw, smem_u32(&dwf_buf[sw_off64(lrow, kcol)]));
        mma_16x8x16(a_A, bw, &dkgb[n * 4]);
      }
    }
  }

  {
    __syncthreads();
    float dq_out[32], dk_t[32];
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int gc = hf * 64 + n * 8 + (lane & 3) * 2 + e;
          const bool rv = trow < rows_here;
          const float gg = rv ? sm.u.gtile[trow * kD + gc] : 0.f;
          const float gnn = sm.u.gtile[last_row * kD + gc];
          const float ge = exp2f(gg);
          dq_out[i] = dq_acc[i] * ge * scale;
          const float dkm = rv ? dk_acc[i] * exp2f(gnn - gg) : 0.f;
          dk_t[i] = dkm;
        }
      }
    }
    // wy dq goes to the fp32 workspace; P4 adds the intra part and emits
    // the final bf16.
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int gc = hf * 64 + n * 8 + (lane & 3) * 2 + e;
          if (trow < rows_here)
            dq_ws[((bos + t_lo + trow) * H + i_h) * (long)kD + gc] = dq_out[i];
        }

    float* red2 = sm.red + kD;
    for (int idx = threadIdx.x; idx < kD; idx += kWarps * 32) red2[idx] = 0.f;
    __syncthreads();
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int c = n * 8 + (lane & 3) * 2 + e;
          const float kk = (trow < rows_here)
              ? __bfloat162float(sm.h_t[sw_off(trow, hf * 64 + c)]) : 0.f;
          atomicAdd(&red2[(hf * 64 + c)], kk * dk_t[i]);
        }
    __syncthreads();

    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int cin = n * 8 + (lane & 3) * 2 + e;
          const int gc = hf * 64 + cin;
          if (trow >= rows_here) continue;
          const float gg = sm.u.gtile[trow * kD + gc];
          const float gnn = sm.u.gtile[last_row * kD + gc];
          const float ge = exp2f(gg);
          const float kk = __bfloat162float(sm.h_t[sw_off(trow, gc)]);
          const float bb = __bfloat162float(sm.dh_t[sw_off(trow, gc)]);
          const float qq = __bfloat162float(*rowp(q, trow, gc));
          const float kg = kk * ge;
          const float kdk = kk * dk_t[i];
          const float dgk_full = sm.red[gc] * exp2f(gnn) + red2[gc];
          const float dkgb_v = dkgb[i];
          const float m_last = (trow == last_row) ? 1.f : 0.f;
          const float dgv = qq * dq_out[i] - kdk + m_last * dgk_full
                            + kg * dkgb_v * bb;
          const float dkv = dk_t[i] + dkgb_v * ge * bb;
          const float dbv = dkgb_v * kg;
          const long off = ((bos + t_lo + trow) * H + i_h) * (long)kD + gc;
          dk_ws[off] = dkv;
          dg_o[off] = dgv;
          db_o[off] = dbv;
        }
  }

  // =============================== P3: dA ===============================
  __syncthreads();
  // dAqk prefetch into the second 16KB of the union (dead gtile rows 32-63);
  // overlaps the P3 mma work below.
  {
    float* dAqk_s_w = reinterpret_cast<float*>(&sm.u) + kBT * kBT;
    for (int idx = threadIdx.x; idx < kBT * (kBT / 4); idx += kWarps * 32) {
      const int r = idx >> 4, c4 = (idx & 15) << 2;
      const bool ok = r < rows_here;
      cp_async_16(smem_u32(&dAqk_s_w[r * kBT + c4]),
                  dAqk_g + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kBT + c4,
                  ok);
    }
    cp_async_commit();
  }
  if (hf == 0) {
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int tcol = n * 8 + (lane & 3) * 2 + e;
          sm.u.dAf[trow * kBT + tcol] = dA_acc[i];
        }
  }
  __syncthreads();
  if (hf == 1) {
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int tcol = n * 8 + (lane & 3) * 2 + e;
          sm.u.dAf[trow * kBT + tcol] += dA_acc[i];
        }
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < kBT * kBT; idx += kWarps * 32) {
    const int r = idx / kBT, c = idx % kBT;
    const bool m = (r > c) && (r < rows_here) && (c < rows_here);
    sm.t2[sw_off64(r, c)] = __float2bfloat16(m ? sm.u.dAf[idx] : 0.f);
  }
  __syncthreads();
  float st1[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) st1[i] = 0.f;
  #pragma unroll
  for (int kc = 0; kc < kBT / 16; ++kc) {
    unsigned a_dA[4];
    const int rr = s * 16 + (lane & 15);
    const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
    ldmatrix_x4(a_dA, smem_u32(&sm.t2[sw_off64(rr, cc)]));
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int lrow = n * 8 + (lane & 7);
      const int lcol = kc * 16 + ((lane & 8) ? 8 : 0);
      unsigned bA[2];
      ldmatrix_x2(bA, smem_u32(&sm.A[sw_off64(lrow, lcol)]));
      mma_16x8x16(a_dA, bA, &st1[n * 4]);
    }
  }
  __syncthreads();
  if (hf == 0) {
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int tcol = n * 8 + (lane & 3) * 2 + e;
          sm.dh_t[sw_off64(trow, tcol)] = __float2bfloat16(st1[n * 4 + rr * 2 + e]);
        }
  }
  __syncthreads();
  float st2[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) st2[i] = 0.f;
  #pragma unroll
  for (int kc = 0; kc < kBT / 16; ++kc) {
    unsigned a_A[4];
    const int rr = s * 16 + (lane & 15);
    const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
    ldmatrix_x4(a_A, smem_u32(&sm.At[sw_off64(rr, cc)]));
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int col = n * 8;
      const int lrow = kc * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
      unsigned bS[2];
      ldmatrix_x2_trans(bS, smem_u32(&sm.dh_t[sw_off64(lrow, col)]));
      mma_16x8x16(a_A, bS, &st2[n * 4]);
    }
  }
  __syncthreads();
  // Final masked fp32 dAkk goes to smem (u.dAf), replacing the raw partials
  // — it never touches gmem in the merged kernel.
  if (hf == 0) {
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int trow = s * 16 + (lane >> 2) + rr * 8;
          const int tcol = n * 8 + (lane & 3) * 2 + e;
          const bool m = (trow > tcol) && (trow < rows_here) && (tcol < rows_here);
          sm.u.dAf[trow * kBT + tcol] = m ? -st2[n * 4 + rr * 2 + e] : 0.f;
        }
  }
  // stage q -> A||At and b -> t2||t3 (all four dead after the st2 mma).
  __syncthreads();
  {
    bf16* q_s_w = sm.A;
    bf16* b_s_w = sm.t2;
    for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kWarps * 32) {
      const int r = idx >> 4, c8 = (idx & 15) << 3;
      const bool ok = r < rows_here;
      cp_async_16(smem_u32(&q_s_w[sw_off(r, c8)]), rowp(q, r, c8), ok);
      cp_async_16(smem_u32(&b_s_w[sw_off(r, c8)]), rowp(b, r, c8), ok);
    }
    cp_async_commit();
    cp_async_wait<0>();
  }
  __syncthreads();

  // =============================== P4: intra ===============================
  // Per warp (s, hf): rows = sub-chunk s (16), cols = K-half hf (64).
  // Lane layout: lane owns column pair c = hf*64 + lane*2 + {0,1} across all
  // 16 rows, so off-diagonal exp2 factors are computed once per (j, c) and
  // dA reads broadcast across the warp. The whole phase is barrier-free:
  // strips past the tail simply skip it.
  if (s * 16 < rows_here) {
    const bf16* q_s = sm.A;
    const bf16* b_s = sm.t2;
    const bf16* k_s = sm.h_t;
    const float* dAkk_s = sm.u.dAf;
    const float* dAqk_s = reinterpret_cast<const float*>(&sm.u) + kBT * kBT;

    const int c_in = lane * 2;              // column-in-half of e=0
    const int gc0 = hf * 64 + c_in;         // absolute channel of e=0
    const long hk = (long)H * kD;
    const long grow0 = ((bos + t_lo) * H + i_h) * (long)kD + gc0;
    auto g_at = [&](int t, int e) -> float {
      return __ldg(g + grow0 + (long)t * hk + e);
    };

    const int jmax = min(16, rows_here - s * 16);  // valid rows in this strip

    float g_own[32];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
      const bool ok = t < jmax;
      g_own[t * 2 + 0] = ok ? g_at(s * 16 + t, 0) : 0.f;
      g_own[t * 2 + 1] = ok ? g_at(s * 16 + t, 1) : 0.f;
    }

    float dq2[32], dk2[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) { dq2[i] = dk2[i] = 0.f; }

    // ---- row side, off-diagonal: earlier sub-chunks, first-row anchor ----
    if (s > 0) {
      const float gn0 = g_own[0];
      const float gn1 = g_own[1];
      // 4-deep load batching: issue the g/k loads for four j's before any
      // exp2/FMA consumes them (s*16 is always a multiple of 4). Same math,
      // same accumulation order — pure memory-level parallelism.
      for (int j = 0; j < s * 16; j += 4) {
        float gj0[4], gj1[4], kj0[4], kj1[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
          gj0[u] = g_at(j + u, 0);
          gj1[u] = g_at(j + u, 1);
          kj0[u] = __bfloat162float(k_s[sw_off(j + u, gc0 + 0)]);
          kj1[u] = __bfloat162float(k_s[sw_off(j + u, gc0 + 1)]);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
          const float kE0 = kj0[u] * exp2f(gn0 - gj0[u]);
          const float kE1 = kj1[u] * exp2f(gn1 - gj1[u]);
          #pragma unroll
          for (int t = 0; t < 16; ++t) {
            const float a_qk = dAqk_s[(s * 16 + t) * kBT + (j + u)];
            const float a_kk = dAkk_s[(s * 16 + t) * kBT + (j + u)];
            dq2[t * 2 + 0] += a_qk * kE0;
            dq2[t * 2 + 1] += a_qk * kE1;
            dk2[t * 2 + 0] += a_kk * kE0;
            dk2[t * 2 + 1] += a_kk * kE1;
          }
        }
      }
      #pragma unroll
      for (int t = 0; t < 16; ++t) {
        // Dead tail lanes (t >= jmax) carry g_own = 0, so exp2f(-gn0) can be
        // +inf under strong decay and 0*inf would park NaN in dq2/dk2. The
        // stores are jmax-guarded today, but keep the registers finite.
        const bool live = t < jmax;
        const float s0 = live ? exp2f(g_own[t * 2 + 0] - gn0) : 0.f;
        const float s1 = live ? exp2f(g_own[t * 2 + 1] - gn1) : 0.f;
        dq2[t * 2 + 0] *= s0;
        dq2[t * 2 + 1] *= s1;
        dk2[t * 2 + 0] *= s0;
        dk2[t * 2 + 1] *= s1;
      }
    }

    // ---- row side, diagonal: EXACT per-(t, j, channel) scalar form ----
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (j < jmax) {
        const float kj0 = __bfloat162float(k_s[sw_off(s * 16 + j, gc0 + 0)]);
        const float kj1 = __bfloat162float(k_s[sw_off(s * 16 + j, gc0 + 1)]);
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
          if (t >= j && t < jmax) {
            const float a_qk = dAqk_s[(s * 16 + t) * kBT + (s * 16 + j)];
            const float a_kk = dAkk_s[(s * 16 + t) * kBT + (s * 16 + j)];
            const float E0 = exp2f(g_own[t * 2 + 0] - g_own[j * 2 + 0]);
            const float E1 = exp2f(g_own[t * 2 + 1] - g_own[j * 2 + 1]);
            dq2[t * 2 + 0] += a_qk * kj0 * E0;
            dq2[t * 2 + 1] += a_qk * kj1 * E1;
            dk2[t * 2 + 0] += a_kk * kj0 * E0;
            dk2[t * 2 + 1] += a_kk * kj1 * E1;
          }
        }
      }
    }

    // ---- dg2 (q side), final dq, db RMW, fold b into dk2 ----
    float dg2[32];
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
      #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int i = t * 2 + e;
        if (t < jmax) {
          const int row = s * 16 + t;
          const long off = grow0 + (long)row * hk + e;
          const float qq = __bfloat162float(q_s[sw_off(row, gc0 + e)]);
          dg2[i] = qq * dq2[i];
          dq_o[off] = __float2bfloat16(dq2[i] + dq_ws[off]);
          const float kk = __bfloat162float(k_s[sw_off(row, gc0 + e)]);
          db_o[off] += dk2[i] * kk;
          dk2[i] *= __bfloat162float(b_s[sw_off(row, gc0 + e)]);
        } else {
          dg2[i] = 0.f;
        }
      }
    }

    // ---- col side, off-diagonal: later sub-chunks, last-row anchor ----
    float dkt[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) dkt[i] = 0.f;
    if (s * 16 + 16 < rows_here) {
      const float gl0 = g_own[(jmax - 1) * 2 + 0];
      const float gl1 = g_own[(jmax - 1) * 2 + 1];
      // 4-deep load batching with a scalar tail (rows_here need not be a
      // multiple of 4). Same math and accumulation order as the plain loop.
      const int jend4 = s * 16 + 16 + (((rows_here - (s * 16 + 16)) >> 2) << 2);
      auto col_body = [&](int j, float E0, float E1, float q0, float q1,
                          float kb0, float kb1) {
        const float qE0 = q0 * E0, qE1 = q1 * E1;
        const float kbE0 = kb0 * E0, kbE1 = kb1 * E1;
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
          const float a_qk = dAqk_s[j * kBT + (s * 16 + t)];
          const float a_kk = dAkk_s[j * kBT + (s * 16 + t)];
          dkt[t * 2 + 0] += a_qk * qE0 + a_kk * kbE0;
          dkt[t * 2 + 1] += a_qk * qE1 + a_kk * kbE1;
        }
      };
      for (int j = s * 16 + 16; j < jend4; j += 4) {
        float gj0[4], gj1[4], q0[4], q1[4], kb0[4], kb1[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
          gj0[u] = g_at(j + u, 0);
          gj1[u] = g_at(j + u, 1);
          q0[u] = __bfloat162float(q_s[sw_off(j + u, gc0 + 0)]);
          q1[u] = __bfloat162float(q_s[sw_off(j + u, gc0 + 1)]);
          kb0[u] = __bfloat162float(k_s[sw_off(j + u, gc0 + 0)])
                   * __bfloat162float(b_s[sw_off(j + u, gc0 + 0)]);
          kb1[u] = __bfloat162float(k_s[sw_off(j + u, gc0 + 1)])
                   * __bfloat162float(b_s[sw_off(j + u, gc0 + 1)]);
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u)
          col_body(j + u, exp2f(gj0[u] - gl0), exp2f(gj1[u] - gl1),
                   q0[u], q1[u], kb0[u], kb1[u]);
      }
      for (int j = jend4; j < rows_here; ++j) {
        const float E0 = exp2f(g_at(j, 0) - gl0);
        const float E1 = exp2f(g_at(j, 1) - gl1);
        col_body(j, E0, E1,
                 __bfloat162float(q_s[sw_off(j, gc0 + 0)]),
                 __bfloat162float(q_s[sw_off(j, gc0 + 1)]),
                 __bfloat162float(k_s[sw_off(j, gc0 + 0)])
                     * __bfloat162float(b_s[sw_off(j, gc0 + 0)]),
                 __bfloat162float(k_s[sw_off(j, gc0 + 1)])
                     * __bfloat162float(b_s[sw_off(j, gc0 + 1)]));
      }
      #pragma unroll
      for (int t = 0; t < 16; ++t) {
        dkt[t * 2 + 0] *= exp2f(gl0 - g_own[t * 2 + 0]);
        dkt[t * 2 + 1] *= exp2f(gl1 - g_own[t * 2 + 1]);
      }
    }

    // ---- col side, diagonal: EXACT scalar transpose form ----
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (j < jmax) {
        const float qj0 = __bfloat162float(q_s[sw_off(s * 16 + j, gc0 + 0)]);
        const float qj1 = __bfloat162float(q_s[sw_off(s * 16 + j, gc0 + 1)]);
        const float kbj0 = __bfloat162float(k_s[sw_off(s * 16 + j, gc0 + 0)])
                           * __bfloat162float(b_s[sw_off(s * 16 + j, gc0 + 0)]);
        const float kbj1 = __bfloat162float(k_s[sw_off(s * 16 + j, gc0 + 1)])
                           * __bfloat162float(b_s[sw_off(s * 16 + j, gc0 + 1)]);
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
          if (t <= j) {
            const float a_qk = dAqk_s[(s * 16 + j) * kBT + (s * 16 + t)];
            const float a_kk = dAkk_s[(s * 16 + j) * kBT + (s * 16 + t)];
            const float E0 = exp2f(g_own[j * 2 + 0] - g_own[t * 2 + 0]);
            const float E1 = exp2f(g_own[j * 2 + 1] - g_own[t * 2 + 1]);
            dkt[t * 2 + 0] += (a_qk * qj0 + a_kk * kbj0) * E0;
            dkt[t * 2 + 1] += (a_qk * qj1 + a_kk * kbj1) * E1;
          }
        }
      }
    }

    // ---- finals: dg RMW, dk bf16 (wy-add order matches the Triton pair) --
    #pragma unroll
    for (int t = 0; t < 16; ++t) {
      if (t < jmax) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = t * 2 + e;
          const int row = s * 16 + t;
          const long off = grow0 + (long)row * hk + e;
          const float kk = __bfloat162float(k_s[sw_off(row, gc0 + e)]);
          dg_o[off] = dg2[i] + ((dk2[i] - dkt[i]) * kk + dg_o[off]);
          dk_o[off] = __float2bfloat16((dk2[i] + dk_ws[off]) + dkt[i]);
        }
      }
    }
  }
}

// ============================================================================
// Fused forward intra kernel: sub-chunk scores + triangular solve + WY
// auxiliaries in one pass (replaces chunk_gdn2_fwd_kernel_intra_sub_chunk,
// chunk_gdn2_fwd_kernel_inter_solve_fused and recompute_w_u_fwd_gdn2_kernel,
// plus the Akkd round-trip and the Akk zeros-init).
//
//   F1  scores: 6 warps own the six lower off-diagonal (i,j) block pairs
//       (first-row-of-block anchor; all factors <= 1, no clamp — matches
//       Triton), 2 warps own the four diagonal blocks (mid-anchor with the
//       2^110 overflow clamp) plus the 16x16 unit-lower-triangular inverses
//       by forward substitution (16 lanes per block, inverse columns held
//       in registers, rows broadcast by shuffle).
//   F2  block assembly of (I+T)^{-1}: level 1 Ai10/Ai21/Ai32, level 2
//       Ai20/Ai31, level 3 Ai30 — fp32 FMA (upstream uses tf32 here; fp32
//       is strictly closer to the fp64 reference, same accepted class as
//       the merged backward kernel).
//   F3  w = A @ (b*exp2(g)*k), u = A @ (wg*v), kg = k*exp2(g_last - g),
//       optional qg = q*exp2(g), and the bf16 store of A (upper zeros).
//
// g stays in gmem (__ldg; the Triton pipeline re-reads it per kernel too).
// ============================================================================

// 40KB total -> 2 CTAs/SM (16 warps): F1 reads q/k/b/g straight from gmem
// (L1-resident for the CTA's lifetime — the Triton chain re-reads them from
// DRAM three times), and F3 fuses its operand transforms into the load path
// like the w_u kernel. The fp32 workspaces are overlaid by the F3 bf16
// operand tiles after their last use.
struct FwdShared {
  float Traw[kBT * kBT];   // 16KB raw off-diag T (F1/F2) -> kb*2^g bf16 (F3)
  float Afull[kBT * kBT];  // 16KB (I+T)^{-1} (F1/F2)     -> v*wg  bf16 (F3)
  bf16 Abf[kBT * kBT];     //  8KB bf16 A (F3 mma; also the Akk store source)
  bf16 rslab[8 * 256];     //  4KB per-warp B slabs (F1 mma; 6 off-diag + 2 diag)
  float tstage[4 * 256];   //  4KB diag warps: masked T staging for the solve
};

__global__ void __launch_bounds__(kWarps * 32, 2)
gdn2_fwd_intra_fused_kernel(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const float* __restrict__ g,
    const bf16* __restrict__ b,
    const bf16* __restrict__ wg,
    bf16* __restrict__ Aqk_o,
    bf16* __restrict__ Akk_o,
    bf16* __restrict__ w_o,
    bf16* __restrict__ u_o,
    bf16* __restrict__ kg_o,
    bf16* __restrict__ qg_o,   // nullable
    int T, int H, float scale) {
  extern __shared__ char smem_raw[];
  FwdShared& sm = *reinterpret_cast<FwdShared*>(smem_raw);

  const int i_t = blockIdx.x;
  const int i_bh = blockIdx.y;
  const int i_b = i_bh / H, i_h = i_bh % H;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long bos = (long)i_b * T;
  const int t_lo = i_t * kBT;
  const int rows_here = min(kBT, T - t_lo);
  if (rows_here <= 0) return;

  auto rowp = [&](const bf16* base, int t, int d) -> const bf16* {
    return base + ((bos + t_lo + t) * H + i_h) * (long)kD + d;
  };
  const long hk = (long)H * kD;
  const long g0 = ((bos + t_lo) * H + i_h) * (long)kD;
  auto g_at = [&](int t, int c) -> float {
    return __ldg(g + g0 + (long)t * hk + c);
  };

  // ---- zero the raw-T workspace (upper/invalid blocks must read as 0) ----
  for (int idx = threadIdx.x; idx < kBT * kBT; idx += kWarps * 32)
    sm.Traw[idx] = 0.f;
  __syncthreads();

  // ================================ F1 ================================
  // Scores on tensor cores. Off-diagonal warps build the decayed left
  // operands (q*e, b*k*e) directly in mma A-fragments (pack_f2, same trick
  // as the merged kernel's -dwf fragments) and stage the right operand
  // k_j * 2^(gn_i - g_j) through a 512B per-warp smem slab. Diagonal warps
  // do the same with the mid-anchor CLAMPED factors, apply the tril masks
  // post-mma, and hand the masked T to the existing register-column
  // substitution through a small fp32 stage.
  if (warp < 6) {
    // off-diagonal pair: (i, j), i > j
    const int OD_I[6] = {1, 2, 2, 3, 3, 3};
    const int OD_J[6] = {0, 0, 1, 0, 1, 2};
    const int bi = OD_I[warp], bj = OD_J[warp];
    if (bi * 16 < rows_here) {
      const int ri = min(16, rows_here - bi * 16);
      const int fr0 = lane >> 2;         // fragment row base (0..7)
      const int fc0 = (lane & 3) * 2;    // fragment col base within a slab
      bf16* rsl = &sm.rslab[warp * 256];
      const long ganc = g0 + (long)(bi * 16) * hk;
      float c_qk[8], c_kk[8];
      #pragma unroll
      for (int i = 0; i < 8; ++i) { c_qk[i] = c_kk[i] = 0.f; }
      for (int kc = 0; kc < 8; ++kc) {
        const int cbase = kc * 16;
        unsigned a_q[4], a_bk[4];
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int rloc = fr0 + half * 8;
          const bool rv = rloc < ri;
          const int rc = bi * 16 + (rv ? rloc : 0);
          #pragma unroll
          for (int ch = 0; ch < 2; ++ch) {
            const int c = cbase + fc0 + ch * 8;
            const unsigned qp = __ldg(reinterpret_cast<const unsigned*>(rowp(q, rc, c)));
            const unsigned bp = __ldg(reinterpret_cast<const unsigned*>(rowp(b, rc, c)));
            const unsigned kp = __ldg(reinterpret_cast<const unsigned*>(rowp(k, rc, c)));
            const float2 gp = __ldg(reinterpret_cast<const float2*>(g + g0 + (long)rc * hk + c));
            const float2 gn = __ldg(reinterpret_cast<const float2*>(g + ganc + c));
            const bf16* q2 = reinterpret_cast<const bf16*>(&qp);
            const bf16* b2 = reinterpret_cast<const bf16*>(&bp);
            const bf16* k2 = reinterpret_cast<const bf16*>(&kp);
            const float e0 = rv ? exp2f(gp.x - gn.x) : 0.f;   // <= 1
            const float e1 = rv ? exp2f(gp.y - gn.y) : 0.f;
            a_q[half + ch * 2] = pack_f2(__bfloat162float(q2[0]) * e0,
                                         __bfloat162float(q2[1]) * e1);
            a_bk[half + ch * 2] =
                pack_f2(__bfloat162float(b2[0]) * __bfloat162float(k2[0]) * e0,
                        __bfloat162float(b2[1]) * __bfloat162float(k2[1]) * e1);
          }
        }
        // right-operand slab: R[j, c] = k_j * 2^(gn_i - g_j), both <= 1.
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
          const int idx = lane * 8 + u;
          const int jj = idx >> 4, cc = idx & 15;
          const int jrow = bj * 16 + jj;
          const int c = cbase + cc;
          const float gj = __ldg(g + g0 + (long)jrow * hk + c);
          const float gnv = __ldg(g + ganc + c);
          const float kv = __bfloat162float(__ldg(rowp(k, jrow, c)));
          rsl[jj * 16 + cc] = __float2bfloat16(kv * exp2f(gnv - gj));
        }
        __syncwarp();
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
          unsigned bfr[2];
          ldmatrix_x2(bfr, smem_u32(&rsl[(n * 8 + (lane & 7)) * 16
                                         + ((lane & 8) ? 8 : 0)]));
          mma_16x8x16(a_q, bfr, &c_qk[n * 4]);
          mma_16x8x16(a_bk, bfr, &c_kk[n * 4]);
        }
        __syncwarp();
      }
      #pragma unroll
      for (int n = 0; n < 2; ++n)
        #pragma unroll
        for (int half = 0; half < 2; ++half)
          #pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int i = n * 4 + half * 2 + e;
            const int rloc = fr0 + half * 8;
            if (rloc < ri) {
              const int trow = bi * 16 + rloc;
              const int col = bj * 16 + n * 8 + fc0 + e;
              Aqk_o[((bos + t_lo + trow) * H + i_h) * (long)kBT + col] =
                  __float2bfloat16(c_qk[i] * scale);
              sm.Traw[trow * kBT + col] = c_kk[i];
            }
          }
    }
  } else {
    // diagonal blocks: warp 6 -> blocks 0,1 ; warp 7 -> blocks 2,3.
    // Full-warp mma per block (sequential), then the existing half-warp
    // column-register substitution, fed through a masked fp32 stage.
    const int dw = warp - 6;
    float* tst = &sm.tstage[dw * 512];             // 2 blocks x 256 floats
    bf16* rsl = &sm.rslab[(6 + dw) * 256];         // own B-slab slot
    const int fr0 = lane >> 2;
    const int fc0 = (lane & 3) * 2;
    #pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
      const int d = dw * 2 + blk;
      const int rd = min(16, rows_here - d * 16);
      float c_qk[8], c_kk[8];
      #pragma unroll
      for (int i = 0; i < 8; ++i) { c_qk[i] = c_kk[i] = 0.f; }
      if (rd > 0) {
        const int anchor = d * 16 + min(8, rows_here - d * 16 - 1);
        const long ganc = g0 + (long)anchor * hk;
        for (int kc = 0; kc < 8; ++kc) {
          const int cbase = kc * 16;
          unsigned a_q[4], a_bk[4];
          #pragma unroll
          for (int half = 0; half < 2; ++half) {
            const int rloc = fr0 + half * 8;
            const bool rv = rloc < rd;
            const int rc = d * 16 + (rv ? rloc : 0);
            #pragma unroll
            for (int ch = 0; ch < 2; ++ch) {
              const int c = cbase + fc0 + ch * 8;
              const unsigned qp = __ldg(reinterpret_cast<const unsigned*>(rowp(q, rc, c)));
              const unsigned bp = __ldg(reinterpret_cast<const unsigned*>(rowp(b, rc, c)));
              const unsigned kp = __ldg(reinterpret_cast<const unsigned*>(rowp(k, rc, c)));
              const float2 gp = __ldg(reinterpret_cast<const float2*>(g + g0 + (long)rc * hk + c));
              const float2 gn = __ldg(reinterpret_cast<const float2*>(g + ganc + c));
              const bf16* q2 = reinterpret_cast<const bf16*>(&qp);
              const bf16* b2 = reinterpret_cast<const bf16*>(&bp);
              const bf16* k2 = reinterpret_cast<const bf16*>(&kp);
              const float e0 = rv ? exp2f(fminf(gp.x - gn.x, 110.f)) : 0.f;
              const float e1 = rv ? exp2f(fminf(gp.y - gn.y, 110.f)) : 0.f;
              a_q[half + ch * 2] = pack_f2(__bfloat162float(q2[0]) * e0,
                                           __bfloat162float(q2[1]) * e1);
              a_bk[half + ch * 2] =
                  pack_f2(__bfloat162float(b2[0]) * __bfloat162float(k2[0]) * e0,
                          __bfloat162float(b2[1]) * __bfloat162float(k2[1]) * e1);
            }
          }
          #pragma unroll
          for (int u = 0; u < 8; ++u) {
            const int idx = lane * 8 + u;
            const int jj = idx >> 4, cc = idx & 15;
            const bool jv = jj < rd;
            const int jrow = d * 16 + (jv ? jj : 0);
            const int c = cbase + cc;
            const float gj = __ldg(g + g0 + (long)jrow * hk + c);
            const float gnv = __ldg(g + ganc + c);
            const float kv = __bfloat162float(__ldg(rowp(k, jrow, c)));
            rsl[jj * 16 + cc] = __float2bfloat16(
                jv ? kv * exp2f(fminf(gnv - gj, 110.f)) : 0.f);
          }
          __syncwarp();
          #pragma unroll
          for (int n = 0; n < 2; ++n) {
            unsigned bfr[2];
            ldmatrix_x2(bfr, smem_u32(&rsl[(n * 8 + (lane & 7)) * 16
                                           + ((lane & 8) ? 8 : 0)]));
            mma_16x8x16(a_q, bfr, &c_qk[n * 4]);
            mma_16x8x16(a_bk, bfr, &c_kk[n * 4]);
          }
          __syncwarp();
        }
        // Aqk diagonal block: full rows < rd, zeros above the inclusive
        // diagonal (downstream dots rely on the zeros).
        #pragma unroll
        for (int n = 0; n < 2; ++n)
          #pragma unroll
          for (int half = 0; half < 2; ++half)
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
              const int i = n * 4 + half * 2 + e;
              const int rloc = fr0 + half * 8;
              const int cloc = n * 8 + fc0 + e;
              if (rloc < rd) {
                Aqk_o[((bos + t_lo + d * 16 + rloc) * H + i_h) * (long)kBT
                      + d * 16 + cloc] =
                    __float2bfloat16((rloc >= cloc) ? c_qk[i] * scale : 0.f);
              }
            }
      }
      // masked strict-lower T into the fp32 stage (zeros elsewhere).
      __syncwarp();
      #pragma unroll
      for (int n = 0; n < 2; ++n)
        #pragma unroll
        for (int half = 0; half < 2; ++half)
          #pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int i = n * 4 + half * 2 + e;
            const int rloc = fr0 + half * 8;
            const int cloc = n * 8 + fc0 + e;
            const bool keep = (rloc > cloc) && (rloc < rd);
            tst[blk * 256 + rloc * 16 + cloc] = keep ? c_kk[i] : 0.f;
          }
      __syncwarp();
    }
    // existing substitution, per half-warp block, columns from the stage.
    const int n = lane & 15;
    const int blk = lane >> 4;
    const int d = dw * 2 + blk;
    const int rd = min(16, rows_here - d * 16);
    const unsigned hmask = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
    float Tcol[16];
    #pragma unroll
    for (int m = 0; m < 16; ++m) Tcol[m] = tst[blk * 256 + m * 16 + n];
    float Aicol[16];
    #pragma unroll
    for (int m = 0; m < 16; ++m) Aicol[m] = -Tcol[m];
    const int bound = max(rd, 0);
    #pragma unroll
    for (int i = 2; i < 16; ++i) {
      float a_n = (n < i) ? -Tcol[i] : 0.f;
      float upd = 0.f;
      #pragma unroll
      for (int m = 0; m < 16; ++m) {
        const float am = __shfl_sync(hmask, a_n, m, 16);
        if (m < i) upd += am * Aicol[m];
      }
      if (i < bound) Aicol[i] = a_n + upd;
    }
    #pragma unroll
    for (int m = 0; m < 16; ++m) {
      float val = Aicol[m] + ((m == n) ? 1.f : 0.f);
      sm.Afull[(d * 16 + m) * kBT + d * 16 + n] = val;
    }
  }
  __syncthreads();

  // ================================ F2 ================================
  // Afull[i,j] = -Ai_ii @ Traw[i,j] @ Ai_jj (level 1), then the two-term and
  // three-term rows. 16 lanes per product, lane = output column.
  {
    const int n = lane & 15;
    float x[16], y[16];
    // ---- level 1: Ai10, Ai21, Ai32 (warps 0,1,2; lanes 0-15) ----
    if (lane < 16 && warp < 3) {
      const int i = warp + 1, j = warp;   // (1,0), (2,1), (3,2)
      // X = Ai_ii @ T[i,j]   (x[m] = column n of X)
      #pragma unroll
      for (int m = 0; m < 16; ++m) {
        float s = 0.f;
        #pragma unroll
        for (int p = 0; p < 16; ++p)
          s += sm.Afull[(i * 16 + m) * kBT + i * 16 + p]
               * sm.Traw[(i * 16 + p) * kBT + j * 16 + n];
        x[m] = s;
      }
      // Y = -X @ Ai_jj (row m of X is spread across lanes -> shuffle)
      #pragma unroll
      for (int m = 0; m < 16; ++m) {
        float s = 0.f;
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
          const float xmp = __shfl_sync(0x0000ffffu, x[m], p, 16);
          s += xmp * sm.Afull[(j * 16 + p) * kBT + j * 16 + n];
        }
        y[m] = -s;
      }
      #pragma unroll
      for (int m = 0; m < 16; ++m)
        sm.Afull[(i * 16 + m) * kBT + j * 16 + n] = y[m];
    }
  }
  __syncthreads();
  {
    const int n = lane & 15;
    // ---- level 2: Ai20 = -Ai22 @ (T20 @ Ai00 + T21 @ Ai10)
    //               Ai31 = -Ai33 @ (T31 @ Ai11 + T32 @ Ai21)  (warps 0,1) ----
    if (lane < 16 && warp < 2) {
      const int i = warp + 2;             // 2 or 3
      const int j = warp;                 // 0 or 1
      float s_col[16];
      #pragma unroll
      for (int m = 0; m < 16; ++m) {
        float s = 0.f;
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
          // (T[i,j] @ Ai[j,j])[m,n] + (T[i,j+1] @ Ai[j+1,j])[m,n]
          s += sm.Traw[(i * 16 + m) * kBT + j * 16 + p]
               * sm.Afull[(j * 16 + p) * kBT + j * 16 + n];
          s += sm.Traw[(i * 16 + m) * kBT + (j + 1) * 16 + p]
               * sm.Afull[((j + 1) * 16 + p) * kBT + j * 16 + n];
        }
        s_col[m] = s;
      }
      float y[16];
      // Y = -Ai_ii @ S. Unlike level 1 (X row spread across lanes), S here is
      // the RIGHT operand and S[p][n] is this lane's own s_col[p] — a shuffle
      // of s_col[m] from lane p would fetch S[m][p] and contract row-with-row.
      #pragma unroll
      for (int m = 0; m < 16; ++m) {
        float s = 0.f;
        #pragma unroll
        for (int p = 0; p < 16; ++p)
          s += sm.Afull[(i * 16 + m) * kBT + i * 16 + p] * s_col[p];
        y[m] = -s;
      }
      #pragma unroll
      for (int m = 0; m < 16; ++m)
        sm.Afull[(i * 16 + m) * kBT + j * 16 + n] = y[m];
    }
  }
  __syncthreads();
  {
    const int n = lane & 15;
    // ---- level 3: Ai30 = -Ai33 @ (T30 @ Ai00 + T31 @ Ai10 + T32 @ Ai20) ----
    if (lane < 16 && warp == 0) {
      float s_col[16];
      #pragma unroll
      for (int m = 0; m < 16; ++m) {
        float s = 0.f;
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
          s += sm.Traw[(48 + m) * kBT + p] * sm.Afull[p * kBT + n];
          s += sm.Traw[(48 + m) * kBT + 16 + p] * sm.Afull[(16 + p) * kBT + n];
          s += sm.Traw[(48 + m) * kBT + 32 + p] * sm.Afull[(32 + p) * kBT + n];
        }
        s_col[m] = s;
      }
      float y[16];
      // Same as level 2: S is the right operand, S[p][n] = lane-local
      // s_col[p]; no shuffle.
      #pragma unroll
      for (int m = 0; m < 16; ++m) {
        float s = 0.f;
        #pragma unroll
        for (int p = 0; p < 16; ++p)
          s += sm.Afull[(48 + m) * kBT + 48 + p] * s_col[p];
        y[m] = -s;
      }
      #pragma unroll
      for (int m = 0; m < 16; ++m)
        sm.Afull[(48 + m) * kBT + n] = y[m];
    }
  }
  __syncthreads();

  // ================================ F3 ================================
  // Abf + Akk store first (they read Afull), then overlay Traw/Afull with
  // the bf16 kb*2^g and v*wg tiles built straight from gmem (kg/qg emitted
  // in the same fused pass, like the w_u kernel), then w/u on tensor cores.
  {
    for (int idx = threadIdx.x; idx < kBT * kBT; idx += kWarps * 32) {
      const int r = idx / kBT, c = idx % kBT;
      const bool lower = (c >> 4) <= (r >> 4);
      const float av = lower ? sm.Afull[idx] : 0.f;
      const bf16 ab = __float2bfloat16(av);
      sm.Abf[sw_off64(r, c)] = (r < rows_here) ? ab : __float2bfloat16(0.f);
      if (r < rows_here)
        Akk_o[((bos + t_lo + r) * H + i_h) * (long)kBT + c] = ab;
    }
    __syncthreads();   // Afull/Traw fully consumed; overlays may begin

    bf16* kbe_s = reinterpret_cast<bf16*>(sm.Traw);
    bf16* vwg_s = reinterpret_cast<bf16*>(sm.Afull);
    const int lastr = rows_here - 1;
    const long glast = g0 + (long)lastr * hk;
    for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kWarps * 32) {
      const int r = idx >> 4, c8 = (idx & 15) << 3;
      bf16 kbe8[8], vwg8[8];
      if (r < rows_here) {
        const long grow = g0 + (long)r * hk;
        const uint4 kk8 = __ldg(reinterpret_cast<const uint4*>(rowp(k, r, c8)));
        const uint4 bb8 = __ldg(reinterpret_cast<const uint4*>(rowp(b, r, c8)));
        const uint4 vv8 = __ldg(reinterpret_cast<const uint4*>(rowp(v, r, c8)));
        const uint4 ww8 = __ldg(reinterpret_cast<const uint4*>(rowp(wg, r, c8)));
        const bf16* kp = reinterpret_cast<const bf16*>(&kk8);
        const bf16* bp = reinterpret_cast<const bf16*>(&bb8);
        const bf16* vp = reinterpret_cast<const bf16*>(&vv8);
        const bf16* wp = reinterpret_cast<const bf16*>(&ww8);
        float gr[8], gl[8];
        {
          const float4 a0 = __ldg(reinterpret_cast<const float4*>(g + grow + c8));
          const float4 a1 = __ldg(reinterpret_cast<const float4*>(g + grow + c8 + 4));
          gr[0] = a0.x; gr[1] = a0.y; gr[2] = a0.z; gr[3] = a0.w;
          gr[4] = a1.x; gr[5] = a1.y; gr[6] = a1.z; gr[7] = a1.w;
          const float4 l0 = __ldg(reinterpret_cast<const float4*>(g + glast + c8));
          const float4 l1 = __ldg(reinterpret_cast<const float4*>(g + glast + c8 + 4));
          gl[0] = l0.x; gl[1] = l0.y; gl[2] = l0.z; gl[3] = l0.w;
          gl[4] = l1.x; gl[5] = l1.y; gl[6] = l1.z; gl[7] = l1.w;
        }
        bf16 kg8[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float kk = __bfloat162float(kp[e]);
          kbe8[e] = __float2bfloat16(kk * __bfloat162float(bp[e])
                                     * exp2f(gr[e]));
          vwg8[e] = __float2bfloat16(__bfloat162float(vp[e])
                                     * __bfloat162float(wp[e]));
          kg8[e] = __float2bfloat16(kk * exp2f(gl[e] - gr[e]));
        }
        *reinterpret_cast<uint4*>(
            &kg_o[((bos + t_lo + r) * H + i_h) * (long)kD + c8]) =
            *reinterpret_cast<uint4*>(kg8);
        if (qg_o != nullptr) {
          const uint4 qq8 = __ldg(reinterpret_cast<const uint4*>(rowp(q, r, c8)));
          const bf16* qp = reinterpret_cast<const bf16*>(&qq8);
          bf16 qg8[8];
          #pragma unroll
          for (int e = 0; e < 8; ++e)
            qg8[e] = __float2bfloat16(__bfloat162float(qp[e]) * exp2f(gr[e]));
          *reinterpret_cast<uint4*>(
              &qg_o[((bos + t_lo + r) * H + i_h) * (long)kD + c8]) =
              *reinterpret_cast<uint4*>(qg8);
        }
      } else {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
          kbe8[e] = __float2bfloat16(0.f);
          vwg8[e] = __float2bfloat16(0.f);
        }
      }
      *reinterpret_cast<uint4*>(&kbe_s[sw_off(r, c8)]) =
          *reinterpret_cast<uint4*>(kbe8);
      *reinterpret_cast<uint4*>(&vwg_s[sw_off(r, c8)]) =
          *reinterpret_cast<uint4*>(vwg8);
    }
    __syncthreads();

    // w = A @ kb_e, u = A @ vwg on tensor cores: warp (s, hf) owns
    // [16 rows x 64 cols], contraction over the 64 chunk rows.
    const int s = warp & 3, hf = warp >> 2;
    float w_fr[32], u_fr[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) { w_fr[i] = u_fr[i] = 0.f; }
    #pragma unroll
    for (int kc = 0; kc < kBT / 16; ++kc) {
      unsigned a_A[4];
      const int rr = s * 16 + (lane & 15);
      const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
      ldmatrix_x4(a_A, smem_u32(&sm.Abf[sw_off64(rr, cc)]));
      #pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int col = hf * 64 + n * 8;
        const int lrow = kc * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
        unsigned bw[2], bu[2];
        ldmatrix_x2_trans(bw, smem_u32(&kbe_s[sw_off(lrow, col)]));
        ldmatrix_x2_trans(bu, smem_u32(&vwg_s[sw_off(lrow, col)]));
        mma_16x8x16(a_A, bw, &w_fr[n * 4]);
        mma_16x8x16(a_A, bu, &u_fr[n * 4]);
      }
    }
    #pragma unroll
    for (int n = 0; n < 8; ++n)
      #pragma unroll
      for (int rr2 = 0; rr2 < 2; ++rr2)
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int i = n * 4 + rr2 * 2 + e;
          const int trow = s * 16 + (lane >> 2) + rr2 * 8;
          const int gc = hf * 64 + n * 8 + (lane & 3) * 2 + e;
          if (trow < rows_here) {
            const long off = ((bos + t_lo + trow) * H + i_h) * (long)kD + gc;
            w_o[off] = __float2bfloat16(w_fr[i]);
            u_o[off] = __float2bfloat16(u_fr[i]);
          }
        }
  }
}

// ============================================================================
// Standalone WY-auxiliary kernel (the backward-recompute counterpart of the
// fused forward's F3): given the SAVED solved A, produce w = A @ (b*2^g*k),
// u = A @ (wg*v), kg = k*2^(g_last-g), optional qg = q*2^g. Same bf16
// operand roundings and tensor-core dots as recompute_w_u_fwd_gdn2_kernel.
// ============================================================================

// 40KB total -> 2 CTAs/SM (16 warps): the transforms are fused into the
// load path so no raw operand tiles are ever staged.
struct WuShared {
  bf16 kbe_t[kBT * kD];  // 16KB k*b*2^g
  bf16 vwg_t[kBT * kD];  // 16KB v*wg
  bf16 Abf[kBT * kBT];   //  8KB A
};

__global__ void __launch_bounds__(kWarps * 32, 2)
gdn2_w_u_fused_kernel(
    const bf16* __restrict__ q,    // nullable (with qg_o)
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const float* __restrict__ g,
    const bf16* __restrict__ b,
    const bf16* __restrict__ wg,
    const bf16* __restrict__ A,
    bf16* __restrict__ w_o,
    bf16* __restrict__ u_o,
    bf16* __restrict__ kg_o,
    bf16* __restrict__ qg_o,       // nullable
    int T, int H) {
  extern __shared__ char smem_raw[];
  WuShared& sm = *reinterpret_cast<WuShared*>(smem_raw);

  const int i_t = blockIdx.x;
  const int i_bh = blockIdx.y;
  const int i_b = i_bh / H, i_h = i_bh % H;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long bos = (long)i_b * T;
  const int t_lo = i_t * kBT;
  const int rows_here = min(kBT, T - t_lo);
  if (rows_here <= 0) return;

  auto rowp = [&](const bf16* base, int t, int d) -> const bf16* {
    return base + ((bos + t_lo + t) * H + i_h) * (long)kD + d;
  };
  const long hk = (long)H * kD;
  const long g0 = ((bos + t_lo) * H + i_h) * (long)kD;
  auto g_at = [&](int t, int c) -> float {
    return __ldg(g + g0 + (long)t * hk + c);
  };

  for (int idx = threadIdx.x; idx < kBT * (kBT / 8); idx += kWarps * 32) {
    const int r = idx >> 3, c8 = (idx & 7) << 3;
    const bool ok = r < rows_here;
    const bf16* src = A + ((bos + t_lo + (ok ? r : 0)) * H + i_h) * (long)kBT + c8;
    cp_async_16(smem_u32(&sm.Abf[sw_off64(r, c8)]), src, ok);
  }
  cp_async_commit();

  // Fused load+transform pass: gmem k/b/v/wg (+q) -> smem kb*2^g and v*wg,
  // gmem kg (and qg). No raw operand tiles are staged.
  const int lastr = rows_here - 1;
  const long glast = g0 + (long)lastr * hk;
  for (int idx = threadIdx.x; idx < kBT * (kD / 8); idx += kWarps * 32) {
    const int r = idx >> 4, c8 = (idx & 15) << 3;
    bf16 kbe8[8], vwg8[8];
    if (r < rows_here) {
      const long grow = g0 + (long)r * hk;
      const uint4 kk8 = *reinterpret_cast<const uint4*>(rowp(k, r, c8));
      const uint4 bb8 = *reinterpret_cast<const uint4*>(rowp(b, r, c8));
      const uint4 vv8 = *reinterpret_cast<const uint4*>(rowp(v, r, c8));
      const uint4 ww8 = *reinterpret_cast<const uint4*>(rowp(wg, r, c8));
      const bf16* kp = reinterpret_cast<const bf16*>(&kk8);
      const bf16* bp = reinterpret_cast<const bf16*>(&bb8);
      const bf16* vp = reinterpret_cast<const bf16*>(&vv8);
      const bf16* wp = reinterpret_cast<const bf16*>(&ww8);
      float gr[8], gl[8];
      {
        const float4 a0 = __ldg(reinterpret_cast<const float4*>(g + grow + c8));
        const float4 a1 = __ldg(reinterpret_cast<const float4*>(g + grow + c8 + 4));
        gr[0] = a0.x; gr[1] = a0.y; gr[2] = a0.z; gr[3] = a0.w;
        gr[4] = a1.x; gr[5] = a1.y; gr[6] = a1.z; gr[7] = a1.w;
        const float4 l0 = __ldg(reinterpret_cast<const float4*>(g + glast + c8));
        const float4 l1 = __ldg(reinterpret_cast<const float4*>(g + glast + c8 + 4));
        gl[0] = l0.x; gl[1] = l0.y; gl[2] = l0.z; gl[3] = l0.w;
        gl[4] = l1.x; gl[5] = l1.y; gl[6] = l1.z; gl[7] = l1.w;
      }
      bf16 kg8[8];
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float kk = __bfloat162float(kp[e]);
        kbe8[e] = __float2bfloat16(kk * __bfloat162float(bp[e]) * exp2f(gr[e]));
        vwg8[e] = __float2bfloat16(__bfloat162float(vp[e])
                                   * __bfloat162float(wp[e]));
        kg8[e] = __float2bfloat16(kk * exp2f(gl[e] - gr[e]));
      }
      *reinterpret_cast<uint4*>(
          &kg_o[((bos + t_lo + r) * H + i_h) * (long)kD + c8]) =
          *reinterpret_cast<uint4*>(kg8);
      if (qg_o != nullptr) {
        const uint4 qq8 = *reinterpret_cast<const uint4*>(rowp(q, r, c8));
        const bf16* qp = reinterpret_cast<const bf16*>(&qq8);
        bf16 qg8[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e)
          qg8[e] = __float2bfloat16(__bfloat162float(qp[e]) * exp2f(gr[e]));
        *reinterpret_cast<uint4*>(
            &qg_o[((bos + t_lo + r) * H + i_h) * (long)kD + c8]) =
            *reinterpret_cast<uint4*>(qg8);
      }
    } else {
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        kbe8[e] = __float2bfloat16(0.f);
        vwg8[e] = __float2bfloat16(0.f);
      }
    }
    *reinterpret_cast<uint4*>(&sm.kbe_t[sw_off(r, c8)]) =
        *reinterpret_cast<uint4*>(kbe8);
    *reinterpret_cast<uint4*>(&sm.vwg_t[sw_off(r, c8)]) =
        *reinterpret_cast<uint4*>(vwg8);
  }
  cp_async_wait<0>();
  __syncthreads();

  const int s = warp & 3, hf = warp >> 2;
  float w_fr[32], u_fr[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) { w_fr[i] = u_fr[i] = 0.f; }
  #pragma unroll
  for (int kc = 0; kc < kBT / 16; ++kc) {
    unsigned a_A[4];
    const int rr = s * 16 + (lane & 15);
    const int cc = kc * 16 + ((lane & 16) ? 8 : 0);
    ldmatrix_x4(a_A, smem_u32(&sm.Abf[sw_off64(rr, cc)]));
    #pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int col = hf * 64 + n * 8;
      const int lrow = kc * 16 + (lane & 7) + ((lane & 8) ? 8 : 0);
      unsigned bw[2], bu[2];
      ldmatrix_x2_trans(bw, smem_u32(&sm.kbe_t[sw_off(lrow, col)]));
      ldmatrix_x2_trans(bu, smem_u32(&sm.vwg_t[sw_off(lrow, col)]));
      mma_16x8x16(a_A, bw, &w_fr[n * 4]);
      mma_16x8x16(a_A, bu, &u_fr[n * 4]);
    }
  }
  #pragma unroll
  for (int n = 0; n < 8; ++n)
    #pragma unroll
    for (int rr2 = 0; rr2 < 2; ++rr2)
      #pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int i = n * 4 + rr2 * 2 + e;
        const int trow = s * 16 + (lane >> 2) + rr2 * 8;
        const int gc = hf * 64 + n * 8 + (lane & 3) * 2 + e;
        if (trow < rows_here) {
          const long off = ((bos + t_lo + trow) * H + i_h) * (long)kD + gc;
          w_o[off] = __float2bfloat16(w_fr[i]);
          u_o[off] = __float2bfloat16(u_fr[i]);
        }
      }
}

}  // namespace

void gdn2_wy_dqkg(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor v_new,
    torch::Tensor g, torch::Tensor b, torch::Tensor wg, torch::Tensor A,
    torch::Tensor h, torch::Tensor dov, torch::Tensor dh, torch::Tensor dvin,
    torch::Tensor dq_o, torch::Tensor dk_o, torch::Tensor dv2_o,
    torch::Tensor dg_o, torch::Tensor db_o, torch::Tensor dw_o,
    torch::Tensor dA_o, double scale) {
  const at::cuda::CUDAGuard guard{q.device()};
  const int B = q.size(0), T = q.size(1), H = q.size(2);
  const int NT = (T + kBT - 1) / kBT;
  dim3 grid(NT, B * H);
  dim3 block(kWarps * 32);
  size_t smem = sizeof(SharedStorage);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        gdn2_wy_dqkg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(e == cudaSuccess, "smem opt-in failed: ", cudaGetErrorString(e));
  }
  gdn2_wy_dqkg_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()),
      reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()),
      reinterpret_cast<const bf16*>(v_new.data_ptr()),
      g.data_ptr<float>(),
      reinterpret_cast<const bf16*>(b.data_ptr()),
      reinterpret_cast<const bf16*>(wg.data_ptr()),
      reinterpret_cast<const bf16*>(A.data_ptr()),
      reinterpret_cast<const bf16*>(h.data_ptr()),
      reinterpret_cast<const bf16*>(dov.data_ptr()),
      reinterpret_cast<const bf16*>(dh.data_ptr()),
      reinterpret_cast<const bf16*>(dvin.data_ptr()),
      dq_o.data_ptr<float>(), dk_o.data_ptr<float>(),
      reinterpret_cast<bf16*>(dv2_o.data_ptr()),
      dg_o.data_ptr<float>(), db_o.data_ptr<float>(),
      reinterpret_cast<bf16*>(dw_o.data_ptr()),
      dA_o.data_ptr<float>(),
      T, H, NT, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn2_wy_intra_bwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor v_new,
    torch::Tensor g, torch::Tensor b, torch::Tensor wg, torch::Tensor A,
    torch::Tensor h, torch::Tensor dov, torch::Tensor dh, torch::Tensor dvin,
    torch::Tensor dAqk,
    torch::Tensor dq_ws, torch::Tensor dk_ws,
    torch::Tensor dq_o, torch::Tensor dk_o, torch::Tensor dv2_o,
    torch::Tensor dg_o, torch::Tensor db_o, torch::Tensor dw_o,
    double scale) {
  const at::cuda::CUDAGuard guard{q.device()};
  TORCH_CHECK(dAqk.scalar_type() == torch::kFloat32 && dAqk.is_contiguous(),
              "dAqk must be contiguous fp32");
  const int B = q.size(0), T = q.size(1), H = q.size(2);
  const int NT = (T + kBT - 1) / kBT;
  dim3 grid(NT, B * H);
  dim3 block(kWarps * 32);
  size_t smem = sizeof(SharedStorage);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        gdn2_wy_intra_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem);
    TORCH_CHECK(e == cudaSuccess, "smem opt-in failed: ", cudaGetErrorString(e));
  }
  gdn2_wy_intra_bwd_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()),
      reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()),
      reinterpret_cast<const bf16*>(v_new.data_ptr()),
      g.data_ptr<float>(),
      reinterpret_cast<const bf16*>(b.data_ptr()),
      reinterpret_cast<const bf16*>(wg.data_ptr()),
      reinterpret_cast<const bf16*>(A.data_ptr()),
      reinterpret_cast<const bf16*>(h.data_ptr()),
      reinterpret_cast<const bf16*>(dov.data_ptr()),
      reinterpret_cast<const bf16*>(dh.data_ptr()),
      reinterpret_cast<const bf16*>(dvin.data_ptr()),
      dAqk.data_ptr<float>(),
      dq_ws.data_ptr<float>(), dk_ws.data_ptr<float>(),
      reinterpret_cast<bf16*>(dq_o.data_ptr()),
      reinterpret_cast<bf16*>(dk_o.data_ptr()),
      reinterpret_cast<bf16*>(dv2_o.data_ptr()),
      dg_o.data_ptr<float>(), db_o.data_ptr<float>(),
      reinterpret_cast<bf16*>(dw_o.data_ptr()),
      T, H, NT, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn2_fwd_intra_fused(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor g,
    torch::Tensor b, torch::Tensor wg,
    torch::Tensor Aqk_o, torch::Tensor Akk_o, torch::Tensor w_o,
    torch::Tensor u_o, torch::Tensor kg_o,
    c10::optional<torch::Tensor> qg_o, double scale) {
  const at::cuda::CUDAGuard guard{q.device()};
  const int B = q.size(0), T = q.size(1), H = q.size(2);
  const int NT = (T + kBT - 1) / kBT;
  dim3 grid(NT, B * H);
  dim3 block(kWarps * 32);
  size_t smem = sizeof(FwdShared);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        gdn2_fwd_intra_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem);
    TORCH_CHECK(e == cudaSuccess, "smem opt-in failed: ", cudaGetErrorString(e));
  }
  gdn2_fwd_intra_fused_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()),
      reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()),
      g.data_ptr<float>(),
      reinterpret_cast<const bf16*>(b.data_ptr()),
      reinterpret_cast<const bf16*>(wg.data_ptr()),
      reinterpret_cast<bf16*>(Aqk_o.data_ptr()),
      reinterpret_cast<bf16*>(Akk_o.data_ptr()),
      reinterpret_cast<bf16*>(w_o.data_ptr()),
      reinterpret_cast<bf16*>(u_o.data_ptr()),
      reinterpret_cast<bf16*>(kg_o.data_ptr()),
      qg_o.has_value() ? reinterpret_cast<bf16*>(qg_o->data_ptr()) : nullptr,
      T, H, (float)scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn2_w_u_fused(
    c10::optional<torch::Tensor> q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor b, torch::Tensor wg, torch::Tensor A,
    torch::Tensor w_o, torch::Tensor u_o, torch::Tensor kg_o,
    c10::optional<torch::Tensor> qg_o) {
  const at::cuda::CUDAGuard guard{k.device()};
  const int B = k.size(0), T = k.size(1), H = k.size(2);
  const int NT = (T + kBT - 1) / kBT;
  dim3 grid(NT, B * H);
  dim3 block(kWarps * 32);
  size_t smem = sizeof(WuShared);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (smem > 48 * 1024) {
    cudaError_t e = cudaFuncSetAttribute(
        gdn2_w_u_fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem);
    TORCH_CHECK(e == cudaSuccess, "smem opt-in failed: ", cudaGetErrorString(e));
  }
  gdn2_w_u_fused_kernel<<<grid, block, smem, stream>>>(
      q.has_value() ? reinterpret_cast<const bf16*>(q->data_ptr()) : nullptr,
      reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()),
      g.data_ptr<float>(),
      reinterpret_cast<const bf16*>(b.data_ptr()),
      reinterpret_cast<const bf16*>(wg.data_ptr()),
      reinterpret_cast<const bf16*>(A.data_ptr()),
      reinterpret_cast<bf16*>(w_o.data_ptr()),
      reinterpret_cast<bf16*>(u_o.data_ptr()),
      reinterpret_cast<bf16*>(kg_o.data_ptr()),
      qg_o.has_value() ? reinterpret_cast<bf16*>(qg_o->data_ptr()) : nullptr,
      T, H);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn2_wy_dqkg", &gdn2_wy_dqkg, "GDN-2 wy_dqkg backward (SM120 CUDA)");
  m.def("gdn2_wy_intra_bwd", &gdn2_wy_intra_bwd,
        "GDN-2 merged wy_dqkg + intra backward (SM120 CUDA)");
  m.def("gdn2_fwd_intra_fused", &gdn2_fwd_intra_fused,
        "GDN-2 fused forward intra: scores + solve + WY auxiliaries");
  m.def("gdn2_w_u_fused", &gdn2_w_u_fused,
        "GDN-2 WY auxiliaries from saved A (backward recompute)");
}
