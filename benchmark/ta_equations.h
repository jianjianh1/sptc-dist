#ifndef SPTC_TA_EQUATIONS_H
#define SPTC_TA_EQUATIONS_H

#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// TiledArray Einstein notation uses single-character index names.
// Index mapping:
//   i,j,p,q = occupied orbital indices (small, ~13-25)
//   m,n     = unoccupied/auxiliary orbital indices (~202-376)
//   k       = RI auxiliary basis index (~483-906)
//   a,b     = virtual orbital indices (~585-2567)
//
// NOTE: When an index appears in BOTH operands but is NOT contracted
// (Hadamard product), we must use einsum() instead of operator*
// due to a TiledArray bug with mixed Hadamard+contraction in expressions.

struct TAStageResult {
  std::string equation;
  std::string stage;
  double wall_s;
  double compute_s;
  double sparsity;
  long peak_rss_kb;
};

inline long get_peak_rss_kb() {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      long val = 0;
      std::sscanf(line.c_str(), "VmHWM: %ld kB", &val);
      return val;
    }
  }
  return 0;
}

//=============================================================================
// Eq2: R(p,i,a,q,b) = g(i,m,k) * g(p,n,k) * c1(i,m,a) * c2(p,q,n,b)
//
// Stage 1: I0(i,m,p,n) = g(i,m,k) * g(p,n,k)              [contract k]
// Stage 2: I1(i,p,n,a) = sum_m I0(i,m,p,n) * c1(i,m,a)    [Hadamard i, contract m]
// Stage 3: R(p,i,a,q,b) = sum_n I1(i,p,n,a) * c2(p,q,n,b) [Hadamard p, contract n]
//=============================================================================
inline std::vector<TAStageResult> ta_run_eq2(TA::World& world,
                                             const TA::TSpArrayD& g,
                                             const TA::TSpArrayD& c1,
                                             const TA::TSpArrayD& c2) {
  std::vector<TAStageResult> results;
  using Clock = std::chrono::high_resolution_clock;

  // Stage 1: pure contraction (no Hadamard)
  world.gop.fence();
  auto t0 = Clock::now();
  TA::TSpArrayD I0;
  I0("i,m,p,n") = g("i,m,k") * g("p,n,k");
  auto tc = Clock::now();
  world.gop.fence();
  auto t1 = Clock::now();
  results.push_back({"eq2", "stage1",
                     std::chrono::duration<double>(t1 - t0).count(),
                     std::chrono::duration<double>(tc - t0).count(),
                     I0.shape().sparsity(), get_peak_rss_kb()});

  // Stage 2: Hadamard on i + contract m → use einsum
  world.gop.fence();
  auto t2s = Clock::now();
  auto I1 = TA::einsum(I0("i,m,p,n"), c1("i,m,a"), "i,p,n,a");
  auto t2c = Clock::now();
  world.gop.fence();
  auto t2e = Clock::now();
  results.push_back({"eq2", "stage2",
                     std::chrono::duration<double>(t2e - t2s).count(),
                     std::chrono::duration<double>(t2c - t2s).count(),
                     I1.shape().sparsity(), get_peak_rss_kb()});

  // Stage 3: Hadamard on p + contract n → use einsum
  world.gop.fence();
  auto t3s = Clock::now();
  auto R = TA::einsum(I1("i,p,n,a"), c2("p,q,n,b"), "p,i,a,q,b");
  auto t3c = Clock::now();
  world.gop.fence();
  auto t3e = Clock::now();
  results.push_back({"eq2", "stage3",
                     std::chrono::duration<double>(t3e - t3s).count(),
                     std::chrono::duration<double>(t3c - t3s).count(),
                     R.shape().sparsity(), get_peak_rss_kb()});

  double total_wall = std::chrono::duration<double>(t3e - t0).count();
  results.push_back({"eq2", "total", total_wall, total_wall, R.shape().sparsity(),
                     get_peak_rss_kb()});

  return results;
}

//=============================================================================
// Eq0: R(i,j,a,b) = g0(m,n,k) * c(i,m,a) * c(j,n,b) * g1(j,i,k)
//
// Stage 1: I0(n,k,i,a) = g0(m,n,k) * c(i,m,a)         [contract m]
// Stage 2: I1(k,i,a,j,b) = I0(n,k,i,a) * c(j,n,b)     [contract n]
// Stage 3: R(i,j,a,b) = g1(j,i,k) * I1(k,i,a,j,b)     [Hadamard i,j + contract k]
//=============================================================================
inline std::vector<TAStageResult> ta_run_eq0(TA::World& world,
                                             const TA::TSpArrayD& g0,
                                             const TA::TSpArrayD& g1,
                                             const TA::TSpArrayD& c) {
  std::vector<TAStageResult> results;
  using Clock = std::chrono::high_resolution_clock;

  // Stage 1: pure contraction
  world.gop.fence();
  auto t0 = Clock::now();
  TA::TSpArrayD I0;
  I0("n,k,i,a") = g0("m,n,k") * c("i,m,a");
  auto tc = Clock::now();
  world.gop.fence();
  auto t1 = Clock::now();
  results.push_back({"eq0", "stage1",
                     std::chrono::duration<double>(t1 - t0).count(),
                     std::chrono::duration<double>(tc - t0).count(),
                     I0.shape().sparsity(), get_peak_rss_kb()});
  if (world.rank() == 0)
    std::cerr << "  eq0 I0 sparsity: " << I0.shape().sparsity() * 100.0 << "%\n";

  // Stage 2: pure contraction (n shared, contracted; no Hadamard)
  world.gop.fence();
  auto t2s = Clock::now();
  TA::TSpArrayD I1;
  I1("k,i,a,j,b") = I0("n,k,i,a") * c("j,n,b");
  auto t2c = Clock::now();
  world.gop.fence();
  auto t2e = Clock::now();
  results.push_back({"eq0", "stage2",
                     std::chrono::duration<double>(t2e - t2s).count(),
                     std::chrono::duration<double>(t2c - t2s).count(),
                     I1.shape().sparsity(), get_peak_rss_kb()});
  if (world.rank() == 0)
    std::cerr << "  eq0 I1 sparsity: " << I1.shape().sparsity() * 100.0 << "%\n";

  // Stage 3: Hadamard on i,j + contract k → use einsum
  world.gop.fence();
  auto t3s = Clock::now();
  auto R = TA::einsum(g1("j,i,k"), I1("k,i,a,j,b"), "i,j,a,b");
  auto t3c = Clock::now();
  world.gop.fence();
  auto t3e = Clock::now();
  results.push_back({"eq0", "stage3",
                     std::chrono::duration<double>(t3e - t3s).count(),
                     std::chrono::duration<double>(t3c - t3s).count(),
                     R.shape().sparsity(), get_peak_rss_kb()});

  double total_wall = std::chrono::duration<double>(t3e - t0).count();
  results.push_back({"eq0", "total", total_wall, total_wall, R.shape().sparsity(),
                     get_peak_rss_kb()});
  return results;
}

//=============================================================================
// Eq1: R(i,p,j,a,b) = g0(m,n,k) * c2(i,j,m,a) * c2(j,p,n,b) * g1(i,p,k)
//
// Stage 1: I0(n,k,i,j,a) = g0(m,n,k) * c2(i,j,m,a)         [contract m]
// Stage 2: I1(j,k,i,a,p,b) = I0(n,k,i,j,a) * c2(j,p,n,b)   [Hadamard j + contract n]
// Stage 3: R(i,p,j,a,b) = I1(j,k,i,a,p,b) * g1(i,p,k)      [Hadamard i,p + contract k]
//=============================================================================
inline std::vector<TAStageResult> ta_run_eq1(TA::World& world,
                                             const TA::TSpArrayD& g0,
                                             const TA::TSpArrayD& g1,
                                             const TA::TSpArrayD& c2) {
  std::vector<TAStageResult> results;
  using Clock = std::chrono::high_resolution_clock;

  // Stage 1: pure contraction
  world.gop.fence();
  auto t0 = Clock::now();
  TA::TSpArrayD I0;
  I0("n,k,i,j,a") = g0("m,n,k") * c2("i,j,m,a");
  auto tc = Clock::now();
  world.gop.fence();
  auto t1 = Clock::now();
  results.push_back({"eq1", "stage1",
                     std::chrono::duration<double>(t1 - t0).count(),
                     std::chrono::duration<double>(tc - t0).count(),
                     I0.shape().sparsity(), get_peak_rss_kb()});
  if (world.rank() == 0)
    std::cerr << "  eq1 I0 sparsity: " << I0.shape().sparsity() * 100.0 << "%\n";

  // Stage 2: Hadamard on j + contract n → use einsum
  world.gop.fence();
  auto t2s = Clock::now();
  auto I1 = TA::einsum(I0("n,k,i,j,a"), c2("j,p,n,b"), "j,k,i,a,p,b");
  auto t2c = Clock::now();
  world.gop.fence();
  auto t2e = Clock::now();
  results.push_back({"eq1", "stage2",
                     std::chrono::duration<double>(t2e - t2s).count(),
                     std::chrono::duration<double>(t2c - t2s).count(),
                     I1.shape().sparsity(), get_peak_rss_kb()});
  if (world.rank() == 0)
    std::cerr << "  eq1 I1 sparsity: " << I1.shape().sparsity() * 100.0 << "%\n";

  // Stage 3: Hadamard on i,p + contract k → use einsum
  world.gop.fence();
  auto t3s = Clock::now();
  auto R = TA::einsum(I1("j,k,i,a,p,b"), g1("i,p,k"), "i,p,j,a,b");
  auto t3c = Clock::now();
  world.gop.fence();
  auto t3e = Clock::now();
  results.push_back({"eq1", "stage3",
                     std::chrono::duration<double>(t3e - t3s).count(),
                     std::chrono::duration<double>(t3c - t3s).count(),
                     R.shape().sparsity(), get_peak_rss_kb()});

  double total_wall = std::chrono::duration<double>(t3e - t0).count();
  results.push_back({"eq1", "total", total_wall, total_wall, R.shape().sparsity(),
                     get_peak_rss_kb()});
  return results;
}

#endif  // SPTC_TA_EQUATIONS_H
