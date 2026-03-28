#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "coo_loader.h"
#include "equations.h"

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

static constexpr int NUM_TRIALS = 3;  // increase for more stable timings

void print_csv_header() {
  std::cout << "molecule,equation,stage,trial,time_s\n";
}

void print_csv(const std::string& mol, const std::vector<StageResult>& results,
               int trial) {
  for (const auto& r : results) {
    std::string eq = r.name.substr(0, 3);
    std::cout << mol << "," << eq << "," << r.name << "," << trial << ","
              << r.seconds << "\n";
  }
}

struct RawTensors {
  DenseTensor g0;   // g_m_1_m_2_K_1 (u1, u2, k1)
  DenseTensor g1;   // g_i_1_i_2_K_1 (i1, i2, k1)
  DenseTensor g;    // g_i_1_m_1_K_1 (i, m, k1)
  DenseTensor c1;   // C_m_1_a_1_i_1 (i, m, a)
  DenseTensor c2;   // C_m_1_a_1_i_1_i_2 (i, j, m, a)
};

RawTensors load_molecule(const std::string& dir) {
  RawTensors ts;
  auto t0 = Clock::now();

  std::cerr << "Loading tensors from " << dir << " ...\n";

  {
    auto coo = load_coo(dir + "/g_m_1_m_2_\xCE\x9A_1.txt");
    std::cerr << "  g0: " << coo.values.size() << " nnz, shape (" << coo.shape[0]
              << "," << coo.shape[1] << "," << coo.shape[2] << ")\n";
    ts.g0 = coo_to_dense(coo);
  }
  {
    auto coo = load_coo(dir + "/g_i_1_i_2_\xCE\x9A_1.txt");
    std::cerr << "  g1: " << coo.values.size() << " nnz, shape (" << coo.shape[0]
              << "," << coo.shape[1] << "," << coo.shape[2] << ")\n";
    ts.g1 = coo_to_dense(coo);
  }
  {
    auto coo = load_coo(dir + "/g_i_1_m_1_\xCE\x9A_1.txt");
    std::cerr << "  g:  " << coo.values.size() << " nnz, shape (" << coo.shape[0]
              << "," << coo.shape[1] << "," << coo.shape[2] << ")\n";
    ts.g = coo_to_dense(coo);
  }
  {
    auto coo = load_coo(dir + "/C_m_1_a_1_i_1.txt");
    std::cerr << "  c1: " << coo.values.size() << " nnz, shape (" << coo.shape[0]
              << "," << coo.shape[1] << "," << coo.shape[2] << ")\n";
    ts.c1 = coo_to_dense(coo);
  }
  {
    auto coo = load_coo(dir + "/C_m_1_a_1_i_1_i_2.txt");
    std::cerr << "  c2: " << coo.values.size() << " nnz, shape (" << coo.shape[0]
              << "," << coo.shape[1] << "," << coo.shape[2] << ","
              << coo.shape[3] << ")\n";
    ts.c2 = coo_to_dense(coo);
  }

  auto t1 = Clock::now();
  std::cerr << "  Load time: " << std::chrono::duration<double>(t1 - t0).count()
            << " s\n";
  return ts;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: benchmark_sptc <data_dir> [data_dir2 ...]\n"
              << "  data_dir: path to molecule folder\n";
    return 1;
  }

  print_csv_header();

  for (int m = 1; m < argc; ++m) {
    std::string dir = argv[m];
    std::string mol = fs::path(dir).filename().string();

    std::cerr << "\n========== " << mol << " ==========\n";
    auto ts = load_molecule(dir);

    // Memory usage summary
    std::size_t total_mem =
        (ts.g0.volume() + ts.g1.volume() + ts.g.volume() + ts.c1.volume() +
         ts.c2.volume()) *
        sizeof(double);
    std::cerr << "  Total tensor memory: " << total_mem / (1024 * 1024)
              << " MB\n";

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
      std::cerr << "\n--- Trial " << (trial + 1) << "/" << NUM_TRIALS
                << " ---\n";

      // Eq2 (smallest intermediates)
      std::cerr << "Running eq2...\n";
      auto r2 = run_eq2(ts.g, ts.c1, ts.c2);
      print_csv(mol, r2, trial + 1);

      // Eq0
      std::cerr << "Running eq0...\n";
      auto r0 = run_eq0(ts.g0, ts.g1, ts.c1);
      print_csv(mol, r0, trial + 1);

      // Eq1
      std::cerr << "Running eq1...\n";
      auto r1 = run_eq1(ts.g0, ts.g1, ts.c2);
      print_csv(mol, r1, trial + 1);
    }
  }

  return 0;
}
