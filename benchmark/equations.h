#ifndef SPTC_EQUATIONS_H
#define SPTC_EQUATIONS_H

#include <cblas.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "coo_loader.h"

using Clock = std::chrono::high_resolution_clock;

struct StageResult {
  std::string name;
  double seconds;
};

/// Dense tensor stored as a flat array with strides.
/// Supports up to rank 5.
struct DenseTensor {
  std::vector<double> data;
  std::vector<std::size_t> shape;
  std::vector<std::size_t> stride;

  DenseTensor() = default;

  explicit DenseTensor(const std::vector<std::size_t>& sh) : shape(sh) {
    std::size_t vol = 1;
    stride.resize(sh.size());
    for (int d = (int)sh.size() - 1; d >= 0; --d) {
      stride[d] = vol;
      vol *= sh[d];
    }
    data.assign(vol, 0.0);
  }

  std::size_t volume() const { return data.size(); }
  int rank() const { return (int)shape.size(); }

  double& operator()(std::size_t i0, std::size_t i1) {
    return data[i0 * stride[0] + i1 * stride[1]];
  }
  double& operator()(std::size_t i0, std::size_t i1, std::size_t i2) {
    return data[i0 * stride[0] + i1 * stride[1] + i2 * stride[2]];
  }
  double& operator()(std::size_t i0, std::size_t i1, std::size_t i2,
                     std::size_t i3) {
    return data[i0 * stride[0] + i1 * stride[1] + i2 * stride[2] +
                i3 * stride[3]];
  }
  double& operator()(std::size_t i0, std::size_t i1, std::size_t i2,
                     std::size_t i3, std::size_t i4) {
    return data[i0 * stride[0] + i1 * stride[1] + i2 * stride[2] +
                i3 * stride[3] + i4 * stride[4]];
  }
  const double& operator()(std::size_t i0, std::size_t i1) const {
    return data[i0 * stride[0] + i1 * stride[1]];
  }
  const double& operator()(std::size_t i0, std::size_t i1,
                           std::size_t i2) const {
    return data[i0 * stride[0] + i1 * stride[1] + i2 * stride[2]];
  }
};

/// Load a COO file into a dense tensor.
inline DenseTensor coo_to_dense(const COOTensor& coo) {
  std::vector<std::size_t> sh(coo.shape.begin(),
                              coo.shape.begin() + coo.rank);
  DenseTensor t(sh);
  for (std::size_t i = 0; i < coo.values.size(); ++i) {
    const auto& idx = coo.indices[i];
    std::size_t off = 0;
    for (int d = 0; d < coo.rank; ++d) off += idx[d] * t.stride[d];
    t.data[off] = coo.values[i];
  }
  return t;
}

//=============================================================================
// Eq2: R(i3,i2,a2,i5,a3) = g(i2,m1,k1) * g(i3,m2,k1) * c1(i2,m1,a2) *
//                           c2(i3,i5,m2,a3)
//
// Staged:
//   I0(i2,m1,i3,m2) = sum_k1 g(i2,m1,k1) * g(i3,m2,k1)
//   I1(i2,i3,m2,a2) = sum_m1 I0(i2,m1,i3,m2) * c1(i2,m1,a2)
//   R(i3,i2,a2,i5,a3) = sum_m2 I1(i2,i3,m2,a2) * c2(i3,i5,m2,a3)
//=============================================================================
inline std::vector<StageResult> run_eq2(const DenseTensor& g,
                                        const DenseTensor& c1,
                                        const DenseTensor& c2) {
  std::vector<StageResult> results;
  auto ni = g.shape[0], nm = g.shape[1], nk = g.shape[2];
  auto na_c1 = c1.shape[2];
  auto nj_c2 = c2.shape[1], na_c2 = c2.shape[3];

  std::cout << "  eq2 dims: ni=" << ni << " nm=" << nm << " nk=" << nk
            << " na_c1=" << na_c1 << " nj_c2=" << nj_c2 << " na_c2=" << na_c2
            << "\n";

  // Stage 1: I0(i2,m1,i3,m2) = sum_k g(i2,m1,k) * g(i3,m2,k)
  // Reshape: G as (ni*nm, nk), I0 = G * G^T => (ni*nm, ni*nm)
  auto t0 = Clock::now();
  DenseTensor I0({ni, nm, ni, nm});
  {
    int M = (int)(ni * nm);
    int K = (int)nk;
    // I0 = g_flat * g_flat^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, M, K, 1.0,
                g.data.data(), K, g.data.data(), K, 0.0, I0.data.data(), M);
  }
  auto t1 = Clock::now();
  results.push_back(
      {"eq2_stage1", std::chrono::duration<double>(t1 - t0).count()});
  std::cout << "  eq2_stage1: "
            << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  // Stage 2: I1(i2,i3,m2,a2) = sum_m1 I0(i2,m1,i3,m2) * c1(i2,m1,a2)
  // For each i2: I1[i2](i3,m2,a2) = sum_m1 I0[i2](m1,i3*m2) * c1[i2](m1,a2)
  // I0[i2] is (nm, ni*nm) with m1 as first dim, c1[i2] is (nm, na_c1)
  // I1[i2] = I0[i2]^T * c1[i2] => (ni*nm, na_c1) — reshaped as (i3,m2,a2)
  DenseTensor I1({ni, ni, nm, na_c1});
  {
    int M_out = (int)(ni * nm);  // rows of I0[i2]^T (= cols of I0[i2])
    int K = (int)nm;              // contracted dim m1
    int N = (int)na_c1;           // cols of c1[i2]
    for (std::size_t i2 = 0; i2 < ni; ++i2) {
      // I0[i2] at offset i2 * nm * ni * nm, shape (nm, ni*nm), row-major
      const double* A = I0.data.data() + i2 * nm * ni * nm;
      // c1[i2] at offset i2 * nm * na_c1, shape (nm, na_c1), row-major
      const double* B = c1.data.data() + i2 * nm * na_c1;
      // I1[i2] at offset i2 * ni * nm * na_c1, shape (ni*nm, na_c1)
      double* C = I1.data.data() + i2 * ni * nm * na_c1;
      // C = A^T * B
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M_out, N, K, 1.0,
                  A, M_out, B, N, 0.0, C, N);
    }
  }
  auto t2 = Clock::now();
  results.push_back(
      {"eq2_stage2", std::chrono::duration<double>(t2 - t1).count()});
  std::cout << "  eq2_stage2: "
            << std::chrono::duration<double>(t2 - t1).count() << " s\n";

  // Stage 3: R(i3,i2,a2,i5,a3) = sum_m2 I1(i2,i3,m2,a2) * c2(i3,i5,m2,a3)
  // For each (i2,i3): R[i2,i3](a2,i5,a3) = sum_m2 I1[i2,i3](m2,a2)^T *
  // c2[i3](i5*a3 | m2)
  // I1[i2,i3] is (nm, na_c1) with m2 as leading dim
  // c2[i3] is (nj_c2, nm, na_c2) — need c2[i3,:,:] reshaped as (nm, nj_c2*na_c2)
  // Wait, c2 is (ni, nj_c2, nm, na_c2). For index i3, fix i3:
  //   c2[i3] has shape (nj_c2, nm, na_c2), stride: (nm*na_c2, na_c2, 1)
  // We need sum_m2: I1[i2,i3](m2, a2) * c2[i3](m2, i5*a3)
  //   where c2[i3] reshaped as (nm, nj_c2*na_c2)
  //   but c2 is stored as (i3, i5, m2, a3) — stride is i5 fastest among non-m2
  //   c2[i3,i5,m2,a3] at offset i3*nj_c2*nm*na_c2 + i5*nm*na_c2 + m2*na_c2 + a3
  //   Reshaped as (nm, nj_c2*na_c2): need m2 stride = nj_c2*na_c2? No, actual
  //   stride is na_c2 per m2. Need transpose.
  //
  // Alternative: loop over i3, for each compute via explicit GEMM
  //   For fixed i3: c2_slice[i3] shape (nj_c2, nm, na_c2)
  //   Reshape to (nm, nj_c2*na_c2) — but layout: c2 is row-major as
  //   (i3, i5, m2, a3), so c2_slice is (nj_c2, nm, na_c2)
  //   For the GEMM: we want (nm, nj_c2*na_c2) which means m2 is leading.
  //   Current layout has i5 leading. Need to transpose (nj_c2, nm) dims.
  //
  // Simpler: loop over (i2, i3) and compute with explicit loops + BLAS.
  // R[i2,i3](a2, i5, a3) = sum_m2 I1[i2,i3](m2, a2)^T * c2[i3, i5](m2, a3)
  // For fixed (i2, i3, i5): R[i2,i3,i5](a2, a3) = I1[i2,i3](:,a2)^T * c2[i3,i5](:,a3)
  // This is a GEMM: (na_c1, nm)^T * (nm, na_c2) => (na_c1, na_c2)
  // where I1 slice is (nm, na_c1) with a2 as columns
  // and c2 slice is (nm, na_c2) with a3 as columns

  DenseTensor R({ni, ni, na_c1, nj_c2, na_c2});
  {
    int M = (int)na_c1;
    int N = (int)na_c2;
    int K = (int)nm;
    for (std::size_t i2 = 0; i2 < ni; ++i2) {
      for (std::size_t i3 = 0; i3 < ni; ++i3) {
        // I1[i2,i3] at (i2*ni + i3) * nm * na_c1, shape (nm, na_c1)
        const double* A = I1.data.data() + (i2 * ni + i3) * nm * na_c1;
        for (std::size_t i5 = 0; i5 < nj_c2; ++i5) {
          // c2[i3,i5] at i3*nj_c2*nm*na_c2 + i5*nm*na_c2, shape (nm, na_c2)
          const double* B =
              c2.data.data() + i3 * nj_c2 * nm * na_c2 + i5 * nm * na_c2;
          // R[i3,i2,a2,i5,a3] — stored as (i3,i2,a2,i5,a3)
          // But we produce R[i2,i3,a2,i5,a3] then need to transpose i2,i3.
          // Actually, let's store as R(i2,i3,a2,i5,a3) first.
          // R[i2,i3,:,i5,:] at
          // i2*ni*na_c1*nj_c2*na_c2 + i3*na_c1*nj_c2*na_c2 +
          // i5*na_c2... wait this needs careful stride computation.
          // Use flat index: for R(i2,i3,a2,i5,a3):
          //   offset = i2*(ni*na_c1*nj_c2*na_c2) + i3*(na_c1*nj_c2*na_c2) +
          //            a2*(nj_c2*na_c2) + i5*na_c2 + a3
          // For the GEMM: C[a2, a3] = A^T[a2, m2] * B[m2, a3]
          // C layout: (na_c1, na_c2), row-major
          // But C is stored non-contiguously in R (stride between a2 rows
          // includes i5 and a3). So we use a temp buffer.
          std::vector<double> temp(na_c1 * na_c2, 0.0);
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0,
                      A, M, B, N, 0.0, temp.data(), N);
          // Copy to R
          for (std::size_t a2 = 0; a2 < na_c1; ++a2) {
            for (std::size_t a3 = 0; a3 < na_c2; ++a3) {
              R(i2, i3, a2, i5, a3) = temp[a2 * na_c2 + a3];
            }
          }
        }
      }
    }
  }
  auto t3 = Clock::now();
  results.push_back(
      {"eq2_stage3", std::chrono::duration<double>(t3 - t2).count()});
  std::cout << "  eq2_stage3: "
            << std::chrono::duration<double>(t3 - t2).count() << " s\n";

  double total = std::chrono::duration<double>(t3 - t0).count();
  results.push_back({"eq2_total", total});
  return results;
}

//=============================================================================
// Eq0: R(i1,i2,a1,a2) = sum_{u1,u2,k1} g0(u1,u2,k1) * c(i1,u1,a1) *
//                        c(i2,u2,a2) * g1(i2,i1,k1)
//
// Staged:
//   I0(u2,k1,i1,a1) = sum_u1 g0(u1,u2,k1) * c(i1,u1,a1)
//   I1(k1,i1,a1,i2,a2) = sum_u2 I0(u2,k1,i1,a1) * c(i2,u2,a2)
//   R(i1,i2,a1,a2) = sum_k1 g1(i2,i1,k1) * I1(k1,i1,a1,i2,a2)
//=============================================================================
inline std::vector<StageResult> run_eq0(const DenseTensor& g0,
                                        const DenseTensor& g1,
                                        const DenseTensor& c) {
  std::vector<StageResult> results;
  auto nu = g0.shape[0];  // u1 and u2 dimensions (same)
  auto nk = g0.shape[2];
  auto ni = c.shape[0];
  auto na = c.shape[2];

  std::cout << "  eq0 dims: nu=" << nu << " nk=" << nk << " ni=" << ni
            << " na=" << na << "\n";

  // Stage 1: I0(u2,k1,i1,a1) = sum_u1 g0(u1,u2,k1) * c(i1,u1,a1)
  // g0 reshaped as (u1, u2*k1): M_g0 = u2*k1, K = u1
  // c reshaped as (i1, u1, a1) → (u1, i1*a1): K = u1, N = i1*a1
  // Need: (u2*k1, i1*a1) = g0^T(u2*k1, u1) * c_r(u1, i1*a1)
  // But c is (i1, u1, a1) row-major → stride: (u1*a1, a1, 1)
  // Reshaped as (u1, i1*a1): need u1 as leading dim with stride = i1*a1
  // Actual stride of u1 in c: a1. Not the same! Need transpose.
  // We need c permuted as (u1, i1, a1) → c_p(u1, i1*a1)
  auto t0 = Clock::now();
  // Permute c from (i1, u1, a1) → (u1, i1, a1) = c_p
  std::vector<double> c_p(ni * nu * na);
  for (std::size_t i1 = 0; i1 < ni; ++i1)
    for (std::size_t u1 = 0; u1 < nu; ++u1)
      for (std::size_t a1 = 0; a1 < na; ++a1)
        c_p[u1 * (ni * na) + i1 * na + a1] =
            c.data[i1 * (nu * na) + u1 * na + a1];

  // I0 as (u2*k1, i1*a1)
  int M = (int)(nu * nk);
  int K = (int)nu;
  int N = (int)(ni * na);
  DenseTensor I0({nu, nk, ni, na});
  // g0 is (u1, u2, k1) row-major = (u1, u2*k1) with stride (u2*k1, 1)
  // g0^T is (u2*k1, u1)
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0,
              g0.data.data(), M, c_p.data(), N, 0.0, I0.data.data(), N);
  auto t1 = Clock::now();
  results.push_back(
      {"eq0_stage1", std::chrono::duration<double>(t1 - t0).count()});
  std::cout << "  eq0_stage1: "
            << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  // Stage 2: I1(k1,i1,a1,i2,a2) = sum_u2 I0(u2,k1,i1,a1) * c(i2,u2,a2)
  // I0 is (u2, k1*i1*a1), c is (i1=i2, u1=u2, a1=a2)
  // For this contraction, u2 is shared: I0(u2, ...) and c(i2, u2, a2)
  // I0 reshaped: (u2, k1*ni*na) — leading dim u2 with stride = k1*ni*na
  // c permuted to (u2, i2, a2) → c_p again = (u2, ni*na) with stride ni*na
  // I1 = I0^T * c_p  where I0 is (u2, k1*ni*na), c_p is (u2, ni*na)
  // Result: (k1*ni*na, ni*na)
  // WARNING: this is HUGE. For C3H8: k1=483, ni=13, na=898, nu=202
  // I1 has 483*13*898 * 13*898 = 5.6M * 11.7K = 65 billion elements!
  // This is way too large. We need to avoid materializing I1.

  // Instead, fuse stages 2+3:
  // R(i1,i2,a1,a2) = sum_k1 g1(i2,i1,k1) * [sum_u2 I0(u2,k1,i1,a1)*c(i2,u2,a2)]
  // = sum_k1 g1(i2,i1,k1) * sum_u2 I0(u2,k1,i1,a1) * c(i2,u2,a2)
  //
  // For each (i1, i2):
  //   R[i1,i2](a1,a2) = sum_k1 g1(i2,i1,k1) *
  //                       sum_u2 I0(u2,k1,i1,a1) * c(i2,u2,a2)
  //
  // Inner sum for fixed (k1, i1, i2):
  //   T(a1,a2) = sum_u2 I0(u2,k1,i1,a1) * c(i2,u2,a2)
  // I0 slice: for fixed (k1,i1), I0[:,k1,i1,:] = (nu, na) with stride
  //   I0 layout: (u2, k1, i1, a1), I0[u2,k1,i1,a1] at u2*(nk*ni*na) + k1*(ni*na) + i1*na + a1
  //   For fixed (k1,i1): offset = k1*ni*na + i1*na, stride between u2's = nk*ni*na
  //   This is a column with nu entries, each row having na elements, but stride is nk*ni*na
  // c slice: for fixed i2, c[i2,:,:] = (nu, na), stride = na
  //
  // This is GEMM: T(na, na) = I0_slice^T(na, nu) * c_slice(nu, na)
  // But I0_slice has non-unit stride between u2, so we need to gather.

  // Simpler approach: precompute for each (i1, i2):
  //   R[i1,i2](a1,a2) = sum_k g1(i2,i1,k) * sum_u I0(u,k,i1,a1) * c(i2,u,a2)
  // Let T2[k](a1,a2) = sum_u I0(u,k,i1,a1) * c(i2,u,a2)
  //   = I0_ki1^T(a1, nu) * c_i2(nu, na)  where I0_ki1 is (nu, na) with stride
  // Then R[i1,i2](a1,a2) = sum_k g1(i2,i1,k) * T2[k](a1,a2)

  DenseTensor R({ni, ni, na, na});
  {
    // For each (i1, i2): compute R[i1,i2]
    std::vector<double> I0_slice(nu * na);   // (nu, na) contiguous
    std::vector<double> T2(na * na);         // (na, na)

    for (std::size_t i1 = 0; i1 < ni; ++i1) {
      for (std::size_t i2 = 0; i2 < ni; ++i2) {
        double* R_out = R.data.data() + (i1 * ni + i2) * na * na;
        std::memset(R_out, 0, na * na * sizeof(double));

        // c[i2] slice: (nu, na) at offset i2*nu*na, contiguous
        const double* c_i2 = c.data.data() + i2 * nu * na;

        for (std::size_t k = 0; k < nk; ++k) {
          double g1_val = g1.data[i2 * ni * nk + i1 * nk + k];
          // g1 is (i2, i1, k1) — g1(i2,i1,k) = g1.data[i2*ni*nk + i1*nk + k]
          // Note: g1 file is g_i_1_i_2_K_1.txt with shape (ni, ni, nk)
          // and in the eq: g1(i2, i1, k1), so first file dim=i2, second=i1

          if (g1_val == 0.0) continue;

          // Gather I0[:,k,i1,:] into I0_slice(nu, na)
          for (std::size_t u = 0; u < nu; ++u) {
            // I0(u, k, i1, a1) at u*(nk*ni*na) + k*(ni*na) + i1*na
            const double* src = I0.data.data() + u * nk * ni * na + k * ni * na + i1 * na;
            std::memcpy(I0_slice.data() + u * na, src, na * sizeof(double));
          }

          // T2(a1,a2) = I0_slice^T(a1, nu) * c_i2(nu, a2)
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)na, (int)na,
                      (int)nu, 1.0, I0_slice.data(), (int)na, c_i2, (int)na,
                      0.0, T2.data(), (int)na);

          // R[i1,i2] += g1_val * T2
          cblas_daxpy((int)(na * na), g1_val, T2.data(), 1, R_out, 1);
        }
      }
    }
  }
  auto t2 = Clock::now();
  results.push_back(
      {"eq0_stage2+3_fused",
       std::chrono::duration<double>(t2 - t1).count()});
  std::cout << "  eq0_stage2+3: "
            << std::chrono::duration<double>(t2 - t1).count() << " s\n";

  double total = std::chrono::duration<double>(t2 - t0).count();
  results.push_back({"eq0_total", total});
  return results;
}

//=============================================================================
// Eq1: R(i2,i3,i1,a2,a3) = sum_{u1,u2,k1} g0(u1,u2,k1) *
//        c2(i2,i1,u1,a2) * c2(i1,i3,u2,a3) * g1(i2,i3,k1)
//
// Fully-fused approach to avoid huge intermediates:
//   For each (i2, i3):
//     G(u1,u2) = sum_k g1(i2,i3,k) * g0(u1,u2,k)    [weighted sum of matrices]
//     For each i1:
//       P(a2,u2) = c2(i2,i1,:,a2)^T * G(u1,u2)       [GEMM: (na,nu)*(nu,nu)]
//       R(a2,a3) = P(a2,u2) * c2(i1,i3,u2,a3)         [GEMM: (na,nu)*(nu,na)]
//=============================================================================
inline std::vector<StageResult> run_eq1(const DenseTensor& g0,
                                        const DenseTensor& g1,
                                        const DenseTensor& c2) {
  std::vector<StageResult> results;
  auto nu = g0.shape[0];
  auto nk = g0.shape[2];
  auto ni = c2.shape[0];
  auto nm = c2.shape[2];  // u/m dim (should equal nu)
  auto na = c2.shape[3];

  std::cout << "  eq1 dims: nu=" << nu << " nk=" << nk << " ni=" << ni
            << " nm=" << nm << " na=" << na << "\n";

  auto t0 = Clock::now();

  // g0 layout: (u1, u2, k) row-major → g0[u1,u2,k] at u1*nu*nk + u2*nk + k
  // g1 layout: (i2, i3, k) row-major → g1[i2,i3,k] at i2*ni*nk + i3*nk + k
  // c2 layout: (i, j, u, a) row-major → c2[i,j,u,a] at i*ni*nm*na + j*nm*na + u*na + a

  DenseTensor R({ni, ni, ni, na, na});
  std::vector<double> G(nu * nu);       // G(u1, u2)
  std::vector<double> P(na * nu);       // P(a2, u2)

  for (std::size_t i2 = 0; i2 < ni; ++i2) {
    for (std::size_t i3 = 0; i3 < ni; ++i3) {
      // Compute G(u1,u2) = sum_k g1(i2,i3,k) * g0(u1,u2,k)
      std::memset(G.data(), 0, nu * nu * sizeof(double));
      const double* g1_slice = g1.data.data() + i2 * ni * nk + i3 * nk;

      for (std::size_t k = 0; k < nk; ++k) {
        double g1_val = g1_slice[k];
        if (g1_val == 0.0) continue;
        // g0[:,:,k] is at offsets u1*nu*nk + u2*nk + k, stride between
        // u1 = nu*nk, stride between u2 = nk
        // We need to accumulate g1_val * g0[:,:,k] into G
        for (std::size_t u1 = 0; u1 < nu; ++u1) {
          const double* g0_row = g0.data.data() + u1 * nu * nk + k;
          double* G_row = G.data() + u1 * nu;
          for (std::size_t u2 = 0; u2 < nu; ++u2) {
            G_row[u2] += g1_val * g0_row[u2 * nk];
          }
        }
      }

      for (std::size_t i1 = 0; i1 < ni; ++i1) {
        // c2(i2,i1,:,:) at offset i2*ni*nm*na + i1*nm*na, shape (nm, na)
        // = c2_A(u1, a2), contiguous, row-major
        const double* c2_A = c2.data.data() + i2 * ni * nm * na + i1 * nm * na;

        // P(a2, u2) = c2_A^T(a2, u1) * G(u1, u2)
        // c2_A is (nu, na), c2_A^T is (na, nu), G is (nu, nu)
        // P = c2_A^T * G => (na, nu)
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)na, (int)nu,
                    (int)nu, 1.0, c2_A, (int)na, G.data(), (int)nu, 0.0,
                    P.data(), (int)nu);

        // c2(i1,i3,:,:) at offset i1*ni*nm*na + i3*nm*na, shape (nm, na)
        const double* c2_B = c2.data.data() + i1 * ni * nm * na + i3 * nm * na;

        // R[i2,i3,i1](a2,a3) = P(a2,u2) * c2_B(u2, a3)
        double* R_out =
            R.data.data() + (i2 * ni * ni + i3 * ni + i1) * na * na;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)na,
                    (int)na, (int)nu, 1.0, P.data(), (int)nu, c2_B, (int)na,
                    0.0, R_out, (int)na);
      }
    }
  }

  auto t1 = Clock::now();
  double total = std::chrono::duration<double>(t1 - t0).count();
  results.push_back({"eq1_fused", total});
  results.push_back({"eq1_total", total});
  std::cout << "  eq1_fused: " << total << " s\n";
  return results;
}

#endif  // SPTC_EQUATIONS_H
