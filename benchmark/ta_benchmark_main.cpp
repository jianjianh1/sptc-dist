#include <tiledarray.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// Fix MADNESS/OpenBLAS deadlock: TiledArray only throttles BLAS threading
// for Intel MKL. With OpenBLAS, workers spawn full thread pools causing
// oversubscription deadlock. Must call this BEFORE TA initialization.
extern "C" void openblas_set_num_threads(int);

#include "coo_loader.h"
#include "ta_builder.h"
#include "ta_equations.h"

namespace fs = std::filesystem;

static constexpr int NUM_TRIALS = 3;

// Tile size configuration
struct TileConfig {
  std::size_t occ  = 4;    // occupied orbital indices (i1, i2, i3, i5)
  std::size_t uocc = 50;   // unoccupied/auxiliary (m, u, a)
  std::size_t ri   = 50;   // RI auxiliary basis (k1/K1) — reduced from 200
};

/// Print CSV header (rank 0 only).
void print_csv_header(TA::World& world) {
  if (world.rank() == 0) {
    std::cout
        << "molecule,equation,stage,trial,nranks,wall_s,compute_s,sparsity,peak_rss_kb\n";
  }
}

/// Print a set of stage results as CSV rows.
void print_csv(TA::World& world, const std::string& mol,
               const std::vector<TAStageResult>& results, int trial) {
  if (world.rank() != 0) return;
  for (const auto& r : results) {
    std::cout << mol << "," << r.equation << "," << r.stage << "," << trial
              << "," << world.size() << "," << r.wall_s << "," << r.compute_s
              << "," << r.sparsity << "," << r.peak_rss_kb << "\n";
  }
  std::cout << std::flush;
}

/// Load all tensors for a molecule as TiledArray sparse arrays.
struct TATensors {
  TA::TSpArrayD g0;   // g_m_1_m_2_K_1 (u1, u2, k1)
  TA::TSpArrayD g1;   // g_i_1_i_2_K_1 (i1, i2, k1)
  TA::TSpArrayD g;    // g_i_1_m_1_K_1 (i, m, k1)
  TA::TSpArrayD c1;   // C_m_1_a_1_i_1 (i, m, a)
  TA::TSpArrayD c2;   // C_m_1_a_1_i_1_i_2 (i, j, m, a)
};

TATensors load_ta_tensors(TA::World& world, const std::string& dir,
                          const TileConfig& tc) {
  TATensors ts;
  auto t0 = std::chrono::high_resolution_clock::now();

  if (world.rank() == 0)
    std::cerr << "Loading tensors from " << dir << " ...\n";

  // g0(u1, u2, k1) — tile sizes: [uocc, uocc, ri]
  {
    auto coo = load_coo(dir + "/g_m_1_m_2_\xCE\x9A_1.txt");
    ts.g0 = build_sparse_array(world, coo, {tc.uocc, tc.uocc, tc.ri}, "g0");
  }
  // g1(i2, i1, k1) — tile sizes: [occ, occ, ri]
  {
    auto coo = load_coo(dir + "/g_i_1_i_2_\xCE\x9A_1.txt");
    ts.g1 = build_sparse_array(world, coo, {tc.occ, tc.occ, tc.ri}, "g1");
  }
  // g(i, m, k1) — tile sizes: [occ, uocc, ri]
  {
    auto coo = load_coo(dir + "/g_i_1_m_1_\xCE\x9A_1.txt");
    ts.g = build_sparse_array(world, coo, {tc.occ, tc.uocc, tc.ri}, "g");
  }
  // c1(i, m, a) — tile sizes: [occ, uocc, uocc]
  {
    auto coo = load_coo(dir + "/C_m_1_a_1_i_1.txt");
    ts.c1 = build_sparse_array(world, coo, {tc.occ, tc.uocc, tc.uocc}, "c1");
  }
  // c2(i, j, m, a) — tile sizes: [occ, occ, uocc, uocc]
  {
    auto coo = load_coo(dir + "/C_m_1_a_1_i_1_i_2.txt");
    ts.c2 =
        build_sparse_array(world, coo, {tc.occ, tc.occ, tc.uocc, tc.uocc}, "c2");
  }

  world.gop.fence();
  auto t1 = std::chrono::high_resolution_clock::now();
  if (world.rank() == 0)
    std::cerr << "  Load time: "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  return ts;
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " <data_dir> [data_dir2 ...]\n"
            << "  Options (via env vars):\n"
            << "    SPTC_TRIALS=N        number of timed trials (default: 3)\n"
            << "    SPTC_OCC_TILE=N      occupied tile size (default: 4)\n"
            << "    SPTC_UOCC_TILE=N     unoccupied tile size (default: 50)\n"
            << "    SPTC_RI_TILE=N        RI tile size (default: 50)\n"
            << "    SPTC_EQUATIONS=0,1,2  which equations to run (default: all)\n";
}

int main(int argc, char* argv[]) {
  // CRITICAL: Fix OpenBLAS thread oversubscription before TA init.
  openblas_set_num_threads(1);

  TiledArray::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  // Set sparse threshold to avoid pruning nearly-zero tiles
  TA::SparseShape<float>::threshold(1e-10f);

  if (argc < 2) {
    if (world.rank() == 0) print_usage(argv[0]);
    return 1;
  }

  // Parse configuration from environment
  TileConfig tc;
  int num_trials = NUM_TRIALS;
  bool run_eq0 = true, run_eq1 = true, run_eq2 = true;

  if (const char* v = std::getenv("SPTC_TRIALS"))
    num_trials = std::atoi(v);
  if (const char* v = std::getenv("SPTC_OCC_TILE"))
    tc.occ = std::stoull(v);
  if (const char* v = std::getenv("SPTC_UOCC_TILE"))
    tc.uocc = std::stoull(v);
  if (const char* v = std::getenv("SPTC_RI_TILE"))
    tc.ri = std::stoull(v);
  if (const char* v = std::getenv("SPTC_EQUATIONS")) {
    std::string eqs(v);
    run_eq0 = eqs.find('0') != std::string::npos;
    run_eq1 = eqs.find('1') != std::string::npos;
    run_eq2 = eqs.find('2') != std::string::npos;
  }

  if (world.rank() == 0) {
    std::cerr << "TiledArray distributed benchmark\n"
              << "  Ranks: " << world.size() << "\n"
              << "  Trials: " << num_trials << "\n"
              << "  Tile sizes: occ=" << tc.occ << " uocc=" << tc.uocc
              << " ri=" << tc.ri << "\n"
              << "  Equations: "
              << (run_eq0 ? "eq0 " : "") << (run_eq1 ? "eq1 " : "")
              << (run_eq2 ? "eq2 " : "") << "\n";
  }

  print_csv_header(world);

  for (int m = 1; m < argc; ++m) {
    std::string dir = argv[m];
    std::string mol = fs::path(dir).filename().string();

    if (world.rank() == 0)
      std::cerr << "\n========== " << mol << " ==========\n";

    auto ts = load_ta_tensors(world, dir, tc);

    for (int trial = 0; trial < num_trials; ++trial) {
      if (world.rank() == 0)
        std::cerr << "\n--- Trial " << (trial + 1) << "/" << num_trials
                  << " ---\n";

      if (run_eq2) {
        if (world.rank() == 0) std::cerr << "Running eq2...\n";
        auto r2 = ta_run_eq2(world, ts.g, ts.c1, ts.c2);
        print_csv(world, mol, r2, trial + 1);
      }

      if (run_eq0) {
        if (world.rank() == 0) std::cerr << "Running eq0...\n";
        auto r0 = ta_run_eq0(world, ts.g0, ts.g1, ts.c1);
        print_csv(world, mol, r0, trial + 1);
      }

      if (run_eq1) {
        if (world.rank() == 0) std::cerr << "Running eq1...\n";
        auto r1 = ta_run_eq1(world, ts.g0, ts.g1, ts.c2);
        print_csv(world, mol, r1, trial + 1);
      }
    }

    // Allow cleanup between molecules
    TA::TSpArrayD::wait_for_lazy_cleanup(world);
  }

  return 0;
}
