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
// All contractions use einsum() which correctly handles both pure
// contractions and mixed Hadamard+contraction patterns. The standard
// operator* expression crashes on Hadamard+contraction due to a
// TiledArray bug in SparseShape::perm.

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
// Stage 1: I0(i,m,p,n) = einsum(g, g)              [contract k]
// Stage 2: I1(i,p,n,a) = einsum(I0, c1)            [Hadamard i, contract m]
// Stage 3: R(p,i,a,q,b) = einsum(I1, c2)           [Hadamard p, contract n]
//=============================================================================
inline std::vector<TAStageResult> ta_run_eq2(TA::World& world,
                                             const TA::TSpArrayD& g,
                                             const TA::TSpArrayD& c1,
                                             const TA::TSpArrayD& c2) {
  std::vector<TAStageResult> results;
  using Clock = std::chrono::high_resolution_clock;

  // Stage 1
  world.gop.fence();
  auto t0 = Clock::now();
  auto I0 = TA::einsum(g("i,m,k"), g("p,n,k"), "i,m,p,n");
  auto tc = Clock::now();
  world.gop.fence();
  auto t1 = Clock::now();
  results.push_back({"eq2", "stage1",
                     std::chrono::duration<double>(t1 - t0).count(),
                     std::chrono::duration<double>(tc - t0).count(),
                     I0.shape().sparsity(), get_peak_rss_kb()});

  // Stage 2
  world.gop.fence();
  auto t2s = Clock::now();
  auto I1 = TA::einsum(I0("i,m,p,n"), c1("i,m,a"), "i,p,n,a");
  auto t2c = Clock::now();
  world.gop.fence();
  auto t2e = Clock::now();
  I0 = TA::TSpArrayD();  // release stage-1 intermediate
  TA::TSpArrayD::wait_for_lazy_cleanup(world);
  results.push_back({"eq2", "stage2",
                     std::chrono::duration<double>(t2e - t2s).count(),
                     std::chrono::duration<double>(t2c - t2s).count(),
                     I1.shape().sparsity(), get_peak_rss_kb()});

  // Stage 3
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
// Stage 1: I0(n,k,i,a) = einsum(g0, c)             [contract m]
// Stage 2: I1(k,i,a,j,b) = einsum(I0, c)           [contract n]
// Stage 3: R(i,j,a,b) = einsum(g1, I1)             [Hadamard i,j + contract k]
//
// NOTE: I1 is very large even with sparsity. This equation needs
// multi-node distribution to fit I1 in aggregate memory.
//=============================================================================
inline std::vector<TAStageResult> ta_run_eq0(TA::World& world,
                                             const TA::TSpArrayD& g0,
                                             const TA::TSpArrayD& g1,
                                             const TA::TSpArrayD& c) {
  std::vector<TAStageResult> results;
  using Clock = std::chrono::high_resolution_clock;

  // Stage 1
  world.gop.fence();
  auto t0 = Clock::now();
  auto I0 = TA::einsum(g0("m,n,k"), c("i,m,a"), "n,k,i,a");
  auto tc = Clock::now();
  world.gop.fence();
  auto t1 = Clock::now();
  results.push_back({"eq0", "stage1",
                     std::chrono::duration<double>(t1 - t0).count(),
                     std::chrono::duration<double>(tc - t0).count(),
                     I0.shape().sparsity(), get_peak_rss_kb()});
  if (world.rank() == 0)
    std::cerr << "  eq0 I0 sparsity: " << I0.shape().sparsity() * 100.0 << "%\n";

  // Stage 2
  world.gop.fence();
  auto t2s = Clock::now();
  auto I1 = TA::einsum(I0("n,k,i,a"), c("j,n,b"), "k,i,a,j,b");
  auto t2c = Clock::now();
  world.gop.fence();
  auto t2e = Clock::now();
  I0 = TA::TSpArrayD();  // release stage-1 intermediate
  TA::TSpArrayD::wait_for_lazy_cleanup(world);
  results.push_back({"eq0", "stage2",
                     std::chrono::duration<double>(t2e - t2s).count(),
                     std::chrono::duration<double>(t2c - t2s).count(),
                     I1.shape().sparsity(), get_peak_rss_kb()});
  if (world.rank() == 0)
    std::cerr << "  eq0 I1 sparsity: " << I1.shape().sparsity() * 100.0 << "%\n";

  // Stage 3
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
// Stage 1: I0(n,k,i,j,a) = einsum(g0, c2)          [contract m]
// Stage 2: I1(j,k,i,a,p,b) = einsum(I0, c2)        [Hadamard j + contract n]
// Stage 3: R(i,p,j,a,b) = einsum(I1, g1)           [Hadamard i,p + contract k]
//
// NOTE: I1 is rank-6 and very large. Needs multi-node distribution.
//=============================================================================
inline std::vector<TAStageResult> ta_run_eq1(TA::World& world,
                                             const TA::TSpArrayD& g0,
                                             const TA::TSpArrayD& g1,
                                             const TA::TSpArrayD& c2) {
  std::vector<TAStageResult> results;
  using Clock = std::chrono::high_resolution_clock;

  // Stage 1
  world.gop.fence();
  auto t0 = Clock::now();
  auto I0 = TA::einsum(g0("m,n,k"), c2("i,j,m,a"), "n,k,i,j,a");
  auto tc = Clock::now();
  world.gop.fence();
  auto t1 = Clock::now();
  results.push_back({"eq1", "stage1",
                     std::chrono::duration<double>(t1 - t0).count(),
                     std::chrono::duration<double>(tc - t0).count(),
                     I0.shape().sparsity(), get_peak_rss_kb()});
  if (world.rank() == 0)
    std::cerr << "  eq1 I0 sparsity: " << I0.shape().sparsity() * 100.0 << "%\n";

  // Stage 2
  world.gop.fence();
  auto t2s = Clock::now();
  auto I1 = TA::einsum(I0("n,k,i,j,a"), c2("j,p,n,b"), "j,k,i,a,p,b");
  auto t2c = Clock::now();
  world.gop.fence();
  auto t2e = Clock::now();
  I0 = TA::TSpArrayD();  // release stage-1 intermediate
  TA::TSpArrayD::wait_for_lazy_cleanup(world);
  results.push_back({"eq1", "stage2",
                     std::chrono::duration<double>(t2e - t2s).count(),
                     std::chrono::duration<double>(t2c - t2s).count(),
                     I1.shape().sparsity(), get_peak_rss_kb()});
  if (world.rank() == 0)
    std::cerr << "  eq1 I1 sparsity: " << I1.shape().sparsity() * 100.0 << "%\n";

  // Stage 3
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
