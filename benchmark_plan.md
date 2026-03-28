# Benchmark Plan: Sparse Tensor Contractions

## Goal

Implement and benchmark the three DLPNO-CCSD tensor contraction equations (eq0, eq1, eq2) across all available molecules (C3H8 through C6H14).

## Implementation Note

TiledArray's MADNESS runtime has a task-scheduling deadlock on this system (all expressions hang after `fill()+fence()`). The benchmark uses **dense tensors + OpenBLAS GEMM** instead, with fused contraction stages to avoid materializing huge intermediates.

## C3H8 Results (16 cores, OpenBLAS)

| Equation | Stage | Time (s) | Notes |
|----------|-------|----------|-------|
| eq2 | stage1: I0 = g×g^T | 0.18 | GEMM (2626×483)×(483×2626) |
| eq2 | stage2: I1 = I0×c1 | 0.41 | per-i2 GEMM |
| eq2 | stage3: R = I1×c2 | 26.1 | per-(i2,i3,i5) GEMM |
| eq2 | **total** | **26.7** | |
| eq0 | stage1: I0 = g0^T×c_p | 7.8 | GEMM (97566×202)×(202×11674) |
| eq0 | stage2+3 fused | 344.5 | per-(i1,i2,k) gather+GEMM |
| eq0 | **total** | **352** | bottleneck: k-loop with gather |
| eq1 | fully fused | 51.4 | per-(i2,i3) weighted-sum + per-i1 double GEMM |
| eq1 | **total** | **51.4** | |

---

## Phase 1: COO Loader

Write a generic loader that reads the tab-separated sparse tensor files into memory.

**Input format** (space-separated): `idx0 idx1 ... idxN value`

**Data structures needed:**
```cpp
struct COOTensor {
    int rank;                                         // 3 or 4
    std::vector<std::array<std::size_t, 4>> indices;  // max rank 4
    std::vector<double> values;
    std::array<std::size_t, 4> shape;                 // max index + 1 per dim
};
```

**Steps:**
1. Parse each line, detect rank from column count
2. Track max index per dimension to infer shape
3. Return COO structure ready for TiledArray ingestion

---

## Phase 2: TiledArray Sparse Array Construction

Convert COO data into `TiledArray::TSpArrayD` (sparse tiled arrays).

**Tiling strategy:**
- Use the tile sizes from `input.json`: `occ_tile_size=4`, `uocc_tile_size=50`, `tile_size=200`
- Dimensions corresponding to occupied indices (i1, i2, i3, i5) use `occ_tile_size`
- Dimensions corresponding to auxiliary/virtual indices (m1, m2, u1, u2, a1, a2, a3) use `uocc_tile_size` or `tile_size`
- The K (auxiliary RI) dimension (k1) uses `tile_size`
- Construct `TiledRange1` per dimension from shape and tile size

**Array construction pattern** (following `examples/cc/input_data.h`):
1. Build `TiledRange` from per-dimension `TiledRange1`
2. Compute tile Frobenius norms from COO data → `SparseShape<float>`
3. Construct `TSpArrayD(world, trange, shape)`
4. Populate tiles element-by-element from COO data

---

## Phase 3: Implement the Three Equations

Each equation is a sequence of 3 binary contractions using Einstein notation.

### Eq0
```
R(i1, i2, a1, a2) = g0(u1,u2,k1) * c(i1,u1,a1) * c(i2,u2,a2) * g1(i2,i1,k1)
```
Tensors loaded:
- `g0` ← `g_m_1_m_2_Κ_1.txt` (indices: u1, u2, k1)
- `g1` ← `g_i_1_i_2_Κ_1.txt` (indices: i2, i1, k1)
- `c`  ← `C_m_1_a_1_i_1.txt`  (indices: i, u, a)

Staged contractions:
```cpp
// Step 1: I0(u2,k1,i1,a1) = g0(u1,u2,k1) * c(i1,u1,a1)    [contract u1]
I0("u2,k1,i1,a1") = g0("u1,u2,k1") * c("i1,u1,a1");

// Step 2: I1(k1,i1,a1,i2,a2) = I0(u2,k1,i1,a1) * c(i2,u2,a2)  [contract u2]
I1("k1,i1,a1,i2,a2") = I0("u2,k1,i1,a1") * c("i2,u2,a2");

// Step 3: R(i1,i2,a1,a2) = g1(i2,i1,k1) * I1(k1,i1,a1,i2,a2)  [contract k1]
R("i1,i2,a1,a2") = g1("i2,i1,k1") * I1("k1,i1,a1,i2,a2");
```

### Eq1
```
R(i1, i3, i2, a2, a3) = g0(u1,u2,k1) * c2(i2,i1,u1,a2) * c2(i1,i3,u2,a3) * g1(i2,i3,k1)
```
Tensors loaded:
- `g0` ← `g_m_1_m_2_Κ_1.txt` (indices: u1, u2, k1)
- `g1` ← `g_i_1_i_2_Κ_1.txt` (indices: i2, i3, k1)
- `c2` ← `C_m_1_a_1_i_1_i_2.txt` (indices: i, j, u, a)

Staged contractions:
```cpp
// Step 1: I0(u2,k1,i2,i1,a2) = g0(u1,u2,k1) * c2(i2,i1,u1,a2)   [contract u1]
I0("u2,k1,i2,i1,a2") = g0("u1,u2,k1") * c2("i2,i1,u1,a2");

// Step 2: I1(i1,k1,i2,a2,i3,a3) = I0(u2,k1,i2,i1,a2) * c2(i1,i3,u2,a3)  [contract u2]
I1("i1,k1,i2,a2,i3,a3") = I0("u2,k1,i2,i1,a2") * c2("i1,i3,u2,a3");

// Step 3: R(i2,i3,i1,a2,a3) = I1(i1,k1,i2,a2,i3,a3) * g1(i2,i3,k1)  [contract k1]
R("i2,i3,i1,a2,a3") = I1("i1,k1,i2,a2,i3,a3") * g1("i2,i3,k1");
```

### Eq2
```
R(i3,i2,a2,i5,a3) = g(i2,m1,k1) * g(i3,m2,k1) * c1(i2,m1,a2) * c2(i3,i5,m2,a3)
```
Tensors loaded:
- `g`  ← `g_i_1_m_1_Κ_1.txt` (indices: i, m, k1)
- `c1` ← `C_m_1_a_1_i_1.txt`  (indices: i, m, a)
- `c2` ← `C_m_1_a_1_i_1_i_2.txt` (indices: i, j, m, a)

Staged contractions:
```cpp
// Step 1: I0(i2,m1,i3,m2) = g(i2,m1,k1) * g(i3,m2,k1)     [contract k1]
I0("i2,m1,i3,m2") = g("i2,m1,k1") * g("i3,m2,k1");

// Step 2: I1(i2,i3,m2,a2) = I0(i2,m1,i3,m2) * c1(i2,m1,a2)  [contract m1]
I1("i2,i3,m2,a2") = I0("i2,m1,i3,m2") * c1("i2,m1,a2");

// Step 3: R(i3,i2,a2,i5,a3) = I1(i2,i3,m2,a2) * c2(i3,i5,m2,a3)  [contract m2]
R("i3,i2,a2,i5,a3") = I1("i2,i3,m2,a2") * c2("i3,i5,m2,a3");
```

---

## Phase 4: Timing and Benchmarking Harness

**What to measure:**
- Wall-clock time per stage (step 1, 2, 3) for each equation
- Total contraction time per equation
- Data loading time (file I/O + array construction)
- Memory high-water mark (via `getrusage` or `/proc/self/status`)

**Timing method:**
- Use `MPI_Wtime()` or `std::chrono::high_resolution_clock`
- Insert `world.gop.fence()` before each timing point to ensure all distributed work is complete
- Run each equation 3–5 times and report median to reduce variance

**Output format:**
- Print CSV-style rows: `molecule, equation, stage, time_seconds, nnz_result`
- Optionally verify correctness by printing norms: `R("i,j").norm()` or tile counts

---

## Phase 5: Multi-Molecule Sweep

Run the benchmark across all available molecules to show scaling behavior:

| Molecule | g0 nnz | c1 nnz | c2 nnz | Expected scaling driver |
|----------|--------|--------|--------|------------------------|
| C3H8     | 19.7M  | 159K   | 938K   | baseline                |
| C4H10    | 42.2M  | 239K   | 1.58M  | ~2× g0                  |
| C5H12    | 77.4M  | 313K   | 2.23M  | ~4× g0                  |
| C6H14    | 127.5M | 385K   | 2.88M  | ~6× g0                  |

---

## File Organization

```
sptc-dist/
├── benchmark/
│   ├── CMakeLists.txt          # build config, links against installed TiledArray
│   ├── coo_loader.h            # COO file parser
│   ├── ta_builder.h            # COO → TSpArrayD conversion with tiling
│   ├── eq0.h                   # eq0 staged contraction
│   ├── eq1.h                   # eq1 staged contraction
│   ├── eq2.h                   # eq2 staged contraction
│   ├── benchmark_main.cpp      # main: load data, run equations, print timings
│   └── run_benchmark.sh        # convenience script for MPI launch
├── dataset/
│   └── data_fusedsptc/         # (existing tensor data)
└── tiledarray/                 # (existing TA build/install)
```

**CMakeLists.txt sketch:**
```cmake
cmake_minimum_required(VERSION 3.21)
project(sptc_benchmark CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(tiledarray REQUIRED
    HINTS ${CMAKE_SOURCE_DIR}/../tiledarray/install/lib/cmake/tiledarray)

add_executable(benchmark_sptc benchmark_main.cpp)
target_link_libraries(benchmark_sptc tiledarray)
```

**Run script sketch:**
```bash
#!/usr/bin/env bash
BUILD_DIR=build
cmake -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release .
cmake --build $BUILD_DIR -j$(nproc)

for mol in C3H8 C4H10 C5H12 C6H14; do
    echo "=== $mol ==="
    mpirun -np 1 $BUILD_DIR/benchmark_sptc ../dataset/data_fusedsptc/$mol
done
```

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Intermediate tensors too large for memory (5D/6D in eq1 step 2) | Start with smallest molecule (C3H8); monitor RSS; consider tighter sparse threshold |
| TiledArray index label limitation (max rank or label conflicts) | Use short, distinct single-char labels; verify TA supports rank-5/6 expressions |
| Tiling mismatch causes silent correctness bugs | Validate result norms against reference values if available; cross-check with a small dense computation on C3H8 |
| File I/O dominates total time for large molecules | Report I/O and compute times separately; consider binary caching of loaded tensors |
| Non-contiguous index ranges (e.g., i-indices start at 3, not 0) | Use full shape [0, max+1) for TiledRange; zero tiles in the unused range will be automatically sparse |

---

## Implementation Order

1. **coo_loader.h** — file parser, testable standalone
2. **ta_builder.h** — COO → TSpArrayD, verify with print/norm on C3H8
3. **eq0.h** — simplest equation (3-index c tensor), validate correctness
4. **eq1.h** — 4-index c tensor, higher-rank intermediates
5. **eq2.h** — self-contraction of g tensor
6. **benchmark_main.cpp** — timing harness, CSV output
7. **run_benchmark.sh** — multi-molecule sweep
