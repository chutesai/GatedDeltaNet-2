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
    // kg*b into t1-repurpose... build kgb tile [64, 64-half] in t3:
    __syncthreads();
    for (int idx = threadIdx.x; idx < kBT * kBV; idx += kWarps * 32) {
      const int r = idx / kBV, c = idx % kBV;      // c within half
      const int gc = hf * 64 + c;
      float kk = __bfloat162float(sm.h_t[sw_off(r, gc)]);
      float bb = __bfloat162float(sm.dh_t[sw_off(r, gc)]);
      float gg = (r < rows_here) ? exp2f(sm.u.gtile[r * kD + gc]) : 0.f;
      sm.t3[sw_off64(r, c)] = __float2bfloat16(kk * gg * bb);  // kg*b
    }
    __syncthreads();

    // dA += (-dwf) @ (kgb)^T : A-frags = dwf C-fragments (negated), B from t3.
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
        ldmatrix_x2(bkgb, smem_u32(&sm.t3[sw_off64(lrow, lcol)]));
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
    __syncthreads();
    for (int idx = threadIdx.x; idx < kBT * kBV; idx += kWarps * 32) {
      const int r = idx / kBV, c = idx % kBV;
      const int gc = hf * 64 + c;
      float kk = __bfloat162float(sm.h_t[sw_off(r, gc)]);
      float bb = __bfloat162float(sm.dh_t[sw_off(r, gc)]);
      float gg = (r < rows_here) ? exp2f(sm.u.gtile[r * kD + gc]) : 0.f;
      sm.t3[sw_off64(r, c)] = __float2bfloat16(kk * gg * bb);  // kg*b
    }
    __syncthreads();

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
        ldmatrix_x2(bkgb, smem_u32(&sm.t3[sw_off64(lrow, lcol)]));
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
      for (int j = 0; j < s * 16; ++j) {
        const float kE0 = __bfloat162float(k_s[sw_off(j, gc0 + 0)])
                          * exp2f(gn0 - g_at(j, 0));
        const float kE1 = __bfloat162float(k_s[sw_off(j, gc0 + 1)])
                          * exp2f(gn1 - g_at(j, 1));
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
          const float a_qk = dAqk_s[(s * 16 + t) * kBT + j];
          const float a_kk = dAkk_s[(s * 16 + t) * kBT + j];
          dq2[t * 2 + 0] += a_qk * kE0;
          dq2[t * 2 + 1] += a_qk * kE1;
          dk2[t * 2 + 0] += a_kk * kE0;
          dk2[t * 2 + 1] += a_kk * kE1;
        }
      }
      #pragma unroll
      for (int t = 0; t < 16; ++t) {
        const float s0 = exp2f(g_own[t * 2 + 0] - gn0);
        const float s1 = exp2f(g_own[t * 2 + 1] - gn1);
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
      for (int j = s * 16 + 16; j < rows_here; ++j) {
        const float E0 = exp2f(g_at(j, 0) - gl0);
        const float E1 = exp2f(g_at(j, 1) - gl1);
        const float qE0 = __bfloat162float(q_s[sw_off(j, gc0 + 0)]) * E0;
        const float qE1 = __bfloat162float(q_s[sw_off(j, gc0 + 1)]) * E1;
        const float kbE0 = __bfloat162float(k_s[sw_off(j, gc0 + 0)])
                           * __bfloat162float(b_s[sw_off(j, gc0 + 0)]) * E0;
        const float kbE1 = __bfloat162float(k_s[sw_off(j, gc0 + 1)])
                           * __bfloat162float(b_s[sw_off(j, gc0 + 1)]) * E1;
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
          const float a_qk = dAqk_s[j * kBT + (s * 16 + t)];
          const float a_kk = dAkk_s[j * kBT + (s * 16 + t)];
          dkt[t * 2 + 0] += a_qk * qE0 + a_kk * kbE0;
          dkt[t * 2 + 1] += a_qk * qE1 + a_kk * kbE1;
        }
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn2_wy_dqkg", &gdn2_wy_dqkg, "GDN-2 wy_dqkg backward (SM120 CUDA)");
  m.def("gdn2_wy_intra_bwd", &gdn2_wy_intra_bwd,
        "GDN-2 merged wy_dqkg + intra backward (SM120 CUDA)");
}
