#include <tiledarray.h>
#include <TiledArray/expressions/einsum.h>
#include <iostream>

extern "C" void openblas_set_num_threads(int);

int main(int argc, char** argv) {
  openblas_set_num_threads(1);
  TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);

  // Test: Hadamard + contraction via einsum
  {
    TA::TiledRange tr_I0({TA::TiledRange1{0,3,6}, TA::TiledRange1{0,4,8},
                          TA::TiledRange1{0,3,6}, TA::TiledRange1{0,4,8}});
    TA::TArrayD I0(world, tr_I0);
    I0.fill(1.0);
    world.gop.fence();

    TA::TiledRange tr_c1({TA::TiledRange1{0,3,6}, TA::TiledRange1{0,4,8},
                          TA::TiledRange1{0,5,10}});
    TA::TArrayD c1(world, tr_c1);
    c1.fill(1.0);
    world.gop.fence();

    std::cerr << "Testing einsum Hadamard+contract...\n";
    // einsum: contract m, Hadamard i, result = (i,p,n,a)
    auto I1 = TA::einsum(I0("i,m,p,n"), c1("i,m,a"), "i,p,n,a");
    world.gop.fence();
    std::cerr << "einsum PASS, norm=" << I1("i,p,n,a").norm(world).get() << "\n";
  }

  std::cerr << "All tests passed!\n";
  return 0;
}
