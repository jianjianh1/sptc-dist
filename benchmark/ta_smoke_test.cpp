#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <iostream>

extern "C" void openblas_set_num_threads(int);

#include "coo_loader.h"
#include "ta_builder.h"

int main(int argc, char** argv) {
  openblas_set_num_threads(1);
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);
  TA::SparseShape<float>::threshold(1e-10f);

  if (argc < 2) {
    std::cerr << "Usage: ta_smoke_test <data_dir>\n";
    return 1;
  }
  std::string dir = argv[1];

  // Load all tensors
  std::cerr << "Loading tensors...\n";

  auto g = build_sparse_array(world,
      load_coo(dir + "/g_i_1_m_1_\xCE\x9A_1.txt"), {4, 50, 50}, "g");
  auto c1 = build_sparse_array(world,
      load_coo(dir + "/C_m_1_a_1_i_1.txt"), {4, 50, 50}, "c1");
  auto c2 = build_sparse_array(world,
      load_coo(dir + "/C_m_1_a_1_i_1_i_2.txt"), {4, 4, 50, 50}, "c2");
  auto g0 = build_sparse_array(world,
      load_coo(dir + "/g_m_1_m_2_\xCE\x9A_1.txt"), {50, 50, 50}, "g0");
  auto g1 = build_sparse_array(world,
      load_coo(dir + "/g_i_1_i_2_\xCE\x9A_1.txt"), {4, 4, 50}, "g1");
  world.gop.fence();
  std::cerr << "Tensors loaded.\n\n";

  using Clock = std::chrono::high_resolution_clock;

  // --- Eq2: single einsum ---
  // R(p,i,a,q,b) = g(i,m,k) * g(p,n,k) * c1(i,m,a) * c2(p,q,n,b)
  // Contracted: k, m, n. Free: i, p, a, q, b. Hadamard: i (in g & c1), p (in g & c2)
  //
  // Try as a chain of einsum: first contract two, then contract with third, etc.
  // einsum only takes 2 operands at a time.
  {
    std::cerr << "=== Eq2 single-chain einsum ===\n";
    auto t0 = Clock::now();

    // Step 1: contract g*g over k -> I0(i,m,p,n)
    auto I0 = TA::einsum(g("i,m,k"), g("p,n,k"), "i,m,p,n");
    world.gop.fence();
    auto t1 = Clock::now();
    std::cerr << "  I0 = g*g: " << std::chrono::duration<double>(t1-t0).count() << "s\n";

    // Step 2: contract I0*c1, Hadamard i, contract m -> I1(i,p,n,a)
    auto I1 = TA::einsum(I0("i,m,p,n"), c1("i,m,a"), "i,p,n,a");
    world.gop.fence();
    auto t2 = Clock::now();
    std::cerr << "  I1 = I0*c1: " << std::chrono::duration<double>(t2-t1).count() << "s\n";

    // Step 3: contract I1*c2, Hadamard p, contract n -> R(p,i,a,q,b)
    auto R = TA::einsum(I1("i,p,n,a"), c2("p,q,n,b"), "p,i,a,q,b");
    world.gop.fence();
    auto t3 = Clock::now();
    std::cerr << "  R = I1*c2: " << std::chrono::duration<double>(t3-t2).count() << "s\n";
    std::cerr << "  Eq2 total: " << std::chrono::duration<double>(t3-t0).count() << "s\n";
    std::cerr << "  R sparsity: " << R.shape().sparsity()*100 << "%\n\n";
  }

  // --- Eq0: single-chain einsum ---
  // R(i,j,a,b) = g0(m,n,k) * c(i,m,a) * c(j,n,b) * g1(j,i,k)
  // Try: contract g0*c over m, then *c over n, then *g1 over k (with Hadamard i,j)
  {
    std::cerr << "=== Eq0 single-chain einsum ===\n";
    auto t0 = Clock::now();

    // Step 1: contract g0*c over m -> I0(n,k,i,a) — pure contraction, use operator*
    TA::TSpArrayD I0;
    I0("n,k,i,a") = g0("m,n,k") * c1("i,m,a");
    world.gop.fence();
    auto t1 = Clock::now();
    std::cerr << "  I0 = g0*c: " << std::chrono::duration<double>(t1-t0).count()
              << "s, sparsity=" << I0.shape().sparsity()*100 << "%\n";

    // Step 2: contract I0*c over n -> I1(k,i,a,j,b) — pure contraction
    TA::TSpArrayD I1;
    I1("k,i,a,j,b") = I0("n,k,i,a") * c1("j,n,b");
    world.gop.fence();
    auto t2 = Clock::now();
    std::cerr << "  I1 = I0*c: " << std::chrono::duration<double>(t2-t1).count()
              << "s, sparsity=" << I1.shape().sparsity()*100 << "%\n";

    // Step 3: contract I1*g1 over k, Hadamard i,j -> R(i,j,a,b)
    auto R = TA::einsum(g1("j,i,k"), I1("k,i,a,j,b"), "i,j,a,b");
    world.gop.fence();
    auto t3 = Clock::now();
    std::cerr << "  R = I1*g1: " << std::chrono::duration<double>(t3-t2).count() << "s\n";
    std::cerr << "  Eq0 total: " << std::chrono::duration<double>(t3-t0).count() << "s\n\n";
  }

  // --- Eq1: single-chain einsum ---
  // R(i,p,j,a,b) = g0(m,n,k) * c2(i,j,m,a) * c2(j,p,n,b) * g1(i,p,k)
  {
    std::cerr << "=== Eq1 single-chain einsum ===\n";
    auto t0 = Clock::now();

    // Step 1: contract g0*c2 over m -> I0(n,k,i,j,a) — pure contraction
    TA::TSpArrayD I0;
    I0("n,k,i,j,a") = g0("m,n,k") * c2("i,j,m,a");
    world.gop.fence();
    auto t1 = Clock::now();
    std::cerr << "  I0 = g0*c2: " << std::chrono::duration<double>(t1-t0).count()
              << "s, sparsity=" << I0.shape().sparsity()*100 << "%\n";

    // Step 2: contract I0*c2 over n, Hadamard j -> I1(j,k,i,a,p,b)
    auto I1 = TA::einsum(I0("n,k,i,j,a"), c2("j,p,n,b"), "j,k,i,a,p,b");
    world.gop.fence();
    auto t2 = Clock::now();
    std::cerr << "  I1 = I0*c2: " << std::chrono::duration<double>(t2-t1).count()
              << "s, sparsity=" << I1.shape().sparsity()*100 << "%\n";

    // Step 3: contract I1*g1 over k, Hadamard i,p -> R(i,p,j,a,b)
    auto R = TA::einsum(I1("j,k,i,a,p,b"), g1("i,p,k"), "i,p,j,a,b");
    world.gop.fence();
    auto t3 = Clock::now();
    std::cerr << "  R = I1*g1: " << std::chrono::duration<double>(t3-t2).count() << "s\n";
    std::cerr << "  Eq1 total: " << std::chrono::duration<double>(t3-t0).count() << "s\n\n";
  }

  std::cerr << "All equations done!\n";
  return 0;
}
