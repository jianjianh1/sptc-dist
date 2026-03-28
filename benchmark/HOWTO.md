# SPTC Benchmark: What Was Done and How to Reproduce

## Overview

This document describes the implementation of benchmarks for three DLPNO-CCSD sparse tensor contraction equations (eq0, eq1, eq2) defined in `dataset_context.md`, using the molecule datasets in `dataset/data_fusedsptc/`.

---

## What was attempted

### Attempt 1: TiledArray expression engine (failed)

The original plan was to use TiledArray's built-in Einstein-notation expression engine, which allows writing contractions like:

```cpp
I0("u2,k1,i1,a1") = g0("u1,u2,k1") * c("i1,u1,a1");
```

**Steps taken:**

1. Wrote `ta_builder.h` to convert COO files into `TiledArray::TSpArrayD` sparse arrays.
2. Grouped COO elements by tile (3-pass: count, prefix-sum, scatter) to avoid the slow element-by-element `find().get()[]` pattern.
3. Worked around `SparseShape` threshold issues — the default `numeric_limits<float>::epsilon` combined with inverse-volume scaling pruned all tiles. Fixed by calling `SparseShape<float>::threshold(1e-10f)`.
4. Successfully loaded all 5 tensors with correct sparsity (c1: 72.5%, c2: 74.4%, g tensors: 0%).

**Why it failed:**

TiledArray's MADNESS runtime (Pthreads backend) has a task-scheduling issue on this system. After any `fill()` + `fence()` call, **all subsequent expression evaluations hang indefinitely** — including trivial operations like `B("i,j") = 2.0 * A("i,j")` on a 10x20 dense array. This was confirmed to also affect the official TiledArray examples (`ta_dense`, `ta_sparse`).

The problem is not specific to our code. The MADNESS thread pool creates 15+ worker threads (for 16 cores) and they appear active (high CPU usage), but tasks submitted by the expression engine never complete. Direct tile access via `fill()` works because it sets Futures directly without going through the task queue. Attempts to fix with `MAD_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `FI_PROVIDER`, and running without `mpirun` all failed.

### Attempt 2: Dense tensors + OpenBLAS GEMM (succeeded)

Bypassed TiledArray entirely. Loaded COO data into flat dense arrays and implemented each contraction as a sequence of `cblas_dgemm` calls.

**Key challenge: intermediate tensor sizes.** The naive staged approach from `dataset_context.md` produces intermediates too large to materialize:

- **Eq0 stage 2** intermediate `I1(k1,i1,a1,i2,a2)` would be 483×13×898×13×898 ≈ 65 billion elements (520 GB).
- **Eq1 stage 1** intermediate `I0(u2,k1,i2,i1,a2)` would be 202×483×13×13×876 ≈ 14 billion elements (115 GB).

**Solution: fused contraction stages** that avoid materializing huge intermediates by looping over small index sets (occupied indices, `ni` ≈ 13) and performing GEMM on slices.

---

## Final implementation

### Files

```
benchmark/
├── CMakeLists.txt        # CMake build, links OpenBLAS
├── coo_loader.h          # COO text file parser
├── equations.h           # Dense tensor type + all 3 equations
├── benchmark_main.cpp    # Main harness: load, run, CSV output
└── run_benchmark.sh      # Convenience script
```

### How each equation is computed

#### Eq2: `R(i3,i2,a2,i5,a3) = g(i2,m1,k1) * g(i3,m2,k1) * c1(i2,m1,a2) * c2(i3,i5,m2,a3)`

Three separate stages, all materializable:

1. **Stage 1:** `I0(i2,m1,i3,m2) = sum_k g(i2,m1,k) * g(i3,m2,k)`
   - Reshape g as (ni\*nm, nk), compute I0 = G \* G^T via single GEMM.
   - I0 size: (13\*202)^2 = 6.9M doubles = 55 MB.

2. **Stage 2:** `I1(i2,i3,m2,a2) = sum_m1 I0(i2,m1,i3,m2) * c1(i2,m1,a2)`
   - Per-i2 GEMM: I0[i2]^T \* c1[i2].

3. **Stage 3:** `R(i2,i3,a2,i5,a3) = sum_m2 I1(i2,i3,m2,a2) * c2(i3,i5,m2,a3)`
   - Per-(i2, i3, i5) GEMM on (nm, na) slices, using a temp buffer.

#### Eq0: `R(i1,i2,a1,a2) = g0(u1,u2,k1) * c(i1,u1,a1) * c(i2,u2,a2) * g1(i2,i1,k1)`

Stage 1 materialized, stages 2+3 fused:

1. **Stage 1:** `I0(u2,k1,i1,a1) = sum_u1 g0(u1,u2,k1) * c(i1,u1,a1)`
   - Permute c from (i1,u1,a1) to (u1,i1,a1), then GEMM: g0^T \* c_permuted.
   - I0 size: 202\*483\*13\*898 = 1.14B doubles = 9.1 GB. Large but fits in 64 GB RAM.

2. **Stages 2+3 fused:** For each (i1, i2, k):
   - Gather I0[:,k,i1,:] into contiguous (nu, na) slice.
   - GEMM: T(a1,a2) = I0_slice^T \* c[i2].
   - Accumulate: R[i1,i2] += g1(i2,i1,k) \* T.
   - Bottleneck: ni^2 \* nk iterations (13^2 \* 483 ≈ 82K) each with a gather + GEMM.

#### Eq1: `R(i2,i3,i1,a2,a3) = g0(u1,u2,k1) * c2(i2,i1,u1,a2) * c2(i1,i3,u2,a3) * g1(i2,i3,k1)`

Fully fused — no large intermediates at all:

For each (i2, i3):
1. **Weighted sum:** `G(u1,u2) = sum_k g1(i2,i3,k) * g0(u1,u2,k)`
   - Loop over k, accumulate g1\_val \* g0[:,:,k] into (nu, nu) matrix.

2. For each i1, two GEMMs:
   - `P(a2,u2) = c2[i2,i1]^T * G`  — GEMM (na, nu) \* (nu, nu)
   - `R[i2,i3,i1](a2,a3) = P * c2[i1,i3]` — GEMM (na, nu) \* (nu, na)

### Data loading

`coo_loader.h` reads space-separated text files line by line:
- Auto-detects tensor rank from the first line's column count.
- Tracks max index per dimension to infer shape.
- Returns a `COOTensor` struct with indices, values, and shape.

`coo_to_dense()` in `equations.h` converts COO to a flat row-major `DenseTensor`.

### Tensor file-to-variable mapping

| File | Variable | Shape (C3H8) | Equation role |
|------|----------|-------------|---------------|
| `g_m_1_m_2_Κ_1.txt` | g0 | (202, 202, 483) | eq0/eq1: g0(u1, u2, k1) |
| `g_i_1_i_2_Κ_1.txt` | g1 | (13, 13, 483) | eq0: g1(i2, i1, k1); eq1: g1(i2, i3, k1) |
| `g_i_1_m_1_Κ_1.txt` | g | (13, 202, 483) | eq2: g(i, m, k1) |
| `C_m_1_a_1_i_1.txt` | c1 | (13, 202, 898) | eq0: c(i, u, a); eq2: c1(i, m, a) |
| `C_m_1_a_1_i_1_i_2.txt` | c2 | (13, 13, 202, 876) | eq1: c2(i, j, u, a); eq2: c2(i, j, m, a) |

---

## How to reproduce

### Prerequisites

- Linux with GCC 11+ (C++20 support)
- CMake 3.21+
- OpenBLAS (or any CBLAS implementation)
- Dataset in `dataset/data_fusedsptc/` (run `./download_molecules.sh` if missing)

### Build

```bash
cd benchmark
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build -j$(nproc)
```

### Run a single molecule

```bash
OPENBLAS_NUM_THREADS=16 ./build/benchmark_sptc ../dataset/data_fusedsptc/C3H8
```

CSV output goes to stdout, progress/diagnostics to stderr. Redirect stdout to save results:

```bash
OPENBLAS_NUM_THREADS=16 ./build/benchmark_sptc ../dataset/data_fusedsptc/C3H8 > results_C3H8.csv 2>log.txt
```

### Run all molecules

```bash
bash run_benchmark.sh > results.csv 2>log.txt
```

### Output format

CSV with columns: `molecule,equation,stage,trial,time_s`

Example:
```
molecule,equation,stage,trial,time_s
C3H8,eq2,eq2_stage1,1,0.181445
C3H8,eq2,eq2_stage2,1,0.413834
C3H8,eq2,eq2_stage3,1,26.1325
C3H8,eq2,eq2_total,1,26.7278
```

### Adjusting parameters

- **Number of trials:** change `NUM_TRIALS` in `benchmark_main.cpp` (default: 3).
- **BLAS threads:** set `OPENBLAS_NUM_THREADS` environment variable.
- **Molecule selection:** pass specific directories as arguments.

---

## C3H8 benchmark results (16 cores, OpenBLAS)

Timings are medians of 3 trials (values very stable, <1% variance):

| Equation | Stage | Time (s) |
|----------|-------|----------|
| **Eq2** | stage1: I0 = g\*g^T | 0.18 |
| | stage2: I1 = I0\*c1 | 0.41 |
| | stage3: R = I1\*c2 | 26.1 |
| | **total** | **26.7** |
| **Eq0** | stage1: I0 = g0^T\*c_p | 7.8 |
| | stage2+3 fused | 344.5 |
| | **total** | **352** |
| **Eq1** | fully fused | 51.4 |
| | **total** | **51.4** |

### Memory footprint

- Input tensors: ~406 MB (dominated by g0 at 157 MB)
- Largest intermediate: I0 for eq0 at ~9.1 GB
- Peak RSS: ~10 GB for eq0

### Performance bottlenecks

- **Eq0** is dominated by the fused stage 2+3 loop: 13^2 \* 483 = 81,627 iterations, each performing a gather of (202, 898) strided data + a (898, 202) \* (202, 898) GEMM + a daxpy. The gather is bandwidth-bound.
- **Eq2 stage 3** has ni^2 \* nj = 2,197 GEMM calls of size (898, 202) \* (202, 876).
- **Eq1** is efficient thanks to precomputing `G(u1,u2) = sum_k g1*g0` per (i2,i3) pair, avoiding the k-loop inside the GEMM.

---

## Potential improvements

1. **Eq0 gather elimination:** Reshape I0 from (u2,k1,i1,a1) to (k1,i1,u2,a1) via a one-time permutation so the fused loop accesses contiguous memory.
2. **Sparse skipping:** The g1 tensor is very sparse for off-diagonal (i2,i1) pairs. Skipping zero g1 values (already done via `if (g1_val == 0.0) continue`) helps but could be improved with a compressed nonzero list.
3. **Blocking for cache:** The eq2 stage 3 temp buffer allocation inside the inner loop could be hoisted.
4. **Larger molecules:** C4H10+ will need more memory (g0 grows quadratically). May need out-of-core or tiled approaches for C6H14+.
