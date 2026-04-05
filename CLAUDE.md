# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for benchmarking **sparse tensor contraction** (SPTC) performance in quantum chemistry (DLPNO-CCSD context). It consists of:

1. **TiledArray library** (`tiledarray/`) — an unmodified clone of [ValeevGroup/TiledArray](https://github.com/ValeevGroup/tiledarray), a scalable block-sparse tensor framework built on MADNESS parallel runtime (MADWorld). Uses Einstein summation convention for tensor expressions in C++.
2. **Benchmark datasets** (`dataset/data_fusedsptc/`) — sparse tensor data for molecules C2H6 through C6H14, used to benchmark three tensor contraction equations (eq0, eq1, eq2).
3. **Equation documentation** (`dataset_context.md`) — defines the three contraction equations, their staged computation forms, implementation variants (fused, unfused, BLAS-based, SpGEMM), and tensor size inventories.

## Build Commands

TiledArray uses CMake with C++20. Two build directories exist:

```bash
# Release build (primary)
cd tiledarray/build
cmake --build . -j$(nproc)

# Debug build
cd tiledarray/build_debug
cmake --build . -j$(nproc)

# Run tests
cd tiledarray/build
cmake --build . --target check        # all tests (MPI, multi-process)
cmake --build . --target check_serial # serial tests only (np=1)

# Install
cmake --build build --target install  # installs to tiledarray/install/
```

Key CMake configuration used for the release build:
- Compiler: GCC/G++
- MPI: enabled
- BLAS/LAPACK: OpenBLAS (LP64, 32-bit integers)
- Build type: Release
- Install prefix: `tiledarray/install/`

To reconfigure from scratch:
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=install \
  -DENABLE_MPI=ON \
  .
```

## Benchmark Rules

- **No dense GEMM baselines.** Only benchmark with TiledArray's tile-sparse implementation (`ta_equations.h`, `ta_benchmark_main.cpp`). Dense OpenBLAS implementations have been removed.

## Architecture: The Three Contraction Equations

The benchmark targets three multi-tensor contractions from DLPNO-CCSD, each decomposed into 3 staged contractions:

- **Eq0**: `R(i1,i2,a1,a2) = g0(u1,u2,k1) * c(i1,u1,a1) * c(i2,u2,a2) * g1(i2,i1,k1)` — contracts u1, then u2, then k1
- **Eq1**: `R(i1,i3,i2,a2,a3) = g0(u1,u2,k1) * c(i2,i1,u1,a2) * c(i1,i3,u2,a3) * g1(i2,i3,k1)` — same contraction order
- **Eq2**: `R(i3,i2,a2,i5,a3) = g(i2,m1,k1) * g(i3,m2,k1) * c1(i2,m1,a2) * c2(i3,i5,m2,a3)` — contracts k1, then m1, then m2

Each equation has multiple implementation variants: `constfused`, `constblas`, `unfused`, `densedense`, `mklspgemm`, etc. These represent different strategies for handling sparse tile iteration and BLAS offloading.

## Dataset Layout

`dataset/data_fusedsptc/<molecule>/` contains sparse tensors as text files:
- `g_m_1_m_2_Κ_1.txt` — g0 tensor (3D, largest tensor, grows ~quadratically with molecule size)
- `g_i_1_i_2_Κ_1.txt` — g1 tensor (3D)
- `g_i_1_m_1_Κ_1.txt` — g tensor for eq2 (3D)
- `C_m_1_a_1_i_1.txt` — c1 coefficient tensor (3D)
- `C_m_1_a_1_i_1_i_2.txt` — c2 coefficient tensor (4D)

Download dataset: `./download_molecules.sh` (requires `gdown`; fetches from Google Drive).

## TiledArray Key Concepts

- **TiledRange / TiledRange1**: defines how tensor dimensions are partitioned into tiles
- **TArrayD / TSpArrayD**: dense and sparse tiled arrays (distributed across MPI ranks)
- **Einstein notation**: `C("i,j") = A("i,k") * B("k,j")` — repeated indices are contracted
- **TA::World**: parallel runtime context (wraps MADNESS World / MPI communicator)
- Source headers are in `tiledarray/src/TiledArray/`, public API via `#include <tiledarray.h>`
