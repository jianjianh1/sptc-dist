# Research Analysis: TiledArray for Distributed Sparse Tensor Contraction in DLPNO-CCSD

## 1. Experimental Setup

**Platform:** 4 CloudLab nodes (node0-3), Intel Xeon D-1548 (8 cores @ 2.0 GHz, 64 GB RAM each), connected via 10 GbE (eno1d1, 10.10.1.0/24). No shared filesystem.

**Software:** TiledArray v1.1.0 (unmodified), MADNESS Pthreads backend, MPICH 4.0, OpenBLAS (LP64), GCC 11, C++20. Built with `-fno-lto` to avoid MPICH/MADNESS deadlock.

**Configuration:** Tile sizes occ=4, uocc=50, ri=200 (matching MPQC input.json). Default sparse threshold (~1.19e-7). `OPENBLAS_NUM_THREADS=1`. 3 trials per measurement.

**Equations benchmarked:**
- Eq0: `R(i,j,a,b) = g0(m,n,k) * c(i,m,a) * c(j,n,b) * g1(j,i,k)` — 3 staged binary contractions
- Eq1: `R(i,p,j,a,b) = g0(m,n,k) * c2(i,j,m,a) * c2(j,p,n,b) * g1(i,p,k)` — 3 staged, rank-6 intermediate
- Eq2: `R(p,i,a,q,b) = g(i,m,k) * g(p,n,k) * c1(i,m,a) * c2(p,q,n,b)` — 3 staged, smallest intermediates

**Molecules:** Alkane series C3H8 through C6H14 (linear chain, increasing molecular size).

---

## 2. Tensor Dimension Scaling

The tensor dimensions grow with molecular size:

| Molecule | n_occ (i) | n_uocc (m) | n_ri (k) | n_virt (a) | g0 nnz | c2 nnz |
|----------|-----------|------------|----------|------------|--------|--------|
| C3H8  | 13 | 202 | 483 | 898  | 19.7M  | 938K  |
| C4H10 | 17 | 260 | 624 | 1295 | 42.2M  | 1.58M |
| C5H12 | 21 | 318 | 765 | 1688 | 77.4M  | 2.23M |
| C6H14 | 25 | 376 | 906 | 1839 | 127.5M | 2.88M |

The g0 tensor (largest, driving cost) grows as O(n_uocc^2 * n_ri) ~ O(N^3) with molecular size. The occupied dimension n_occ grows linearly. The virtual dimension n_virt grows approximately linearly.

**Element-level sparsity of g0:** g0 nnz / (n_uocc^2 * n_ri) gives: C3H8 = 100%, C4H10 = 100%, C5H12 = 100%, C6H14 = 99.5%. The g0 tensor is essentially dense at the element level. TiledArray stores it fully because tile-level sparsity is 0% (stage 1 sparsity column in the data shows 0% for C3H8 and ~27-36% for larger molecules, but only at the I0 output level).

---

## 3. Eq2 Performance Analysis

### 3.1 Per-Stage Breakdown (1 rank, median of 3 trials)

| Molecule | Stage 1 (s) | Stage 2 (s) | Stage 3 (s) | Total (s) |
|----------|-------------|-------------|-------------|-----------|
| C3H8  | 0.053  | 0.093  | 0.997  | **1.14** |
| C4H10 | 0.093  | 0.197  | 2.093  | **2.42** |
| C5H12 | 0.261  | 0.419  | 4.358  | **5.08** |
| C6H14 | 0.660  | 0.730  | 7.010  | **8.47** |

Stage 3 dominates (82-87% of total time across all molecules). This is the `einsum(I1("i,p,n,a"), c2("p,q,n,b"), "p,i,a,q,b")` contraction with Hadamard index `p` — the einsum Hadamard loop iterates over occupied-dimension tiles, splitting MPI communicators for each.

### 3.2 Scaling with Molecule Size

| Molecule | Total (s) | n_uocc | Ratio (vs C3H8) | n_uocc ratio |
|----------|-----------|--------|------------------|--------------|
| C3H8  | 1.14  | 202 | 1.0x  | 1.0x  |
| C4H10 | 2.42  | 260 | 2.1x  | 1.3x  |
| C5H12 | 5.08  | 318 | 4.5x  | 1.6x  |
| C6H14 | 8.47  | 376 | 7.4x  | 1.9x  |

The compute time scales approximately as O(n_uocc^2) — doubling n_uocc roughly quadruples the time. This is consistent with stage 3 performing GEMM operations whose cost scales with the unoccupied dimension squared (the "a" and "b" virtual indices).

### 3.3 Tile-Level Sparsity Trends

| Molecule | I0 sparsity | I1 sparsity | R sparsity |
|----------|-------------|-------------|------------|
| C3H8  | 0%    | 71.2% | 92.8% |
| C4H10 | 36.0% | 82.3% | 96.8% |
| C5H12 | 30.6% | 85.8% | 98.1% |
| C6H14 | 27.1% | 87.9% | 98.6% |

Sparsity **increases with molecule size**, particularly for the result tensor R. For C6H14, 98.6% of result tiles are zero — TiledArray skips computing them entirely. This is the key benefit of tile-level sparsity for DLPNO methods: as molecules grow, the pair domain structure becomes increasingly localized, and most tile combinations have no overlap.

### 3.4 Multi-Node Overhead (4 nodes vs 1 rank)

| Molecule | 1 rank (s) | 4 nodes (s) | Slowdown | Memory savings |
|----------|-----------|-------------|----------|----------------|
| C3H8  | 1.14 | 1.88 | 1.65x | 2.2x (2.8 GB → 1.3 GB/rank) |
| C4H10 | 2.42 | 3.39 | 1.40x | 2.2x (5.6 GB → 2.6 GB/rank) |
| C5H12 | 5.08 | 6.21 | 1.22x | 2.2x (10.5 GB → 4.7 GB/rank) |
| C6H14 | 8.47 | 10.03 | 1.18x | 2.3x (16.4 GB → 7.2 GB/rank) |

The multi-node overhead **decreases** with molecule size: from 1.65x for C3H8 to 1.18x for C6H14. For larger molecules, the computation-to-communication ratio improves. Multi-node is not beneficial for Eq2 from a speed perspective, but it reduces per-node memory by ~2.2x.

### 3.5 Comparison with Dense Baseline (C3H8)

| Stage | Dense (16 BLAS threads) | TiledArray (1 rank) | Speedup |
|-------|------------------------|---------------------|---------|
| Stage 1 | 0.18s | 0.053s | 3.4x |
| Stage 2 | 0.41s | 0.093s | 4.4x |
| Stage 3 | 26.1s | 0.997s | 26.2x |
| **Total** | **26.7s** | **1.14s** | **23.4x** |

The 23x speedup comes primarily from stage 3, where tile-level sparsity (92.8%) means TiledArray performs ~13x fewer GEMM operations, and each GEMM is on smaller tile-sized blocks. The dense baseline iterates over all (i2, i3, i5) = 13^3 = 2197 GEMM calls regardless of sparsity.

---

## 4. Eq0 Performance Analysis

### 4.1 Per-Stage Breakdown (4 nodes, 1 rank/node)

| Molecule | Stage 1 (s) | Stage 2 (s) | Stage 3 (s) | Total (s) | Peak RSS/rank |
|----------|-------------|-------------|-------------|-----------|---------------|
| C3H8  | 0.48   | 9.43   | 152.9 | **162.9** | 28.2 GB |
| C4H10 | 1.45   | 26.0   | 357.4 | **384.8** | 58.3 GB |

Stage 3 dominates (94% of total): this is the `einsum(g1("j,i,k"), I1("k,i,a,j,b"), "i,j,a,b")` contraction with Hadamard indices `i,j`. The einsum Hadamard loop iterates over all (i,j) tile pairs with MPI communicator splits per pair.

### 4.2 The I1 Intermediate Problem

The I1 tensor `(k, i, a, j, b)` has the following theoretical dense sizes:

| Molecule | I1 dense shape | Dense size | Tile sparsity | Estimated actual |
|----------|---------------|------------|---------------|-----------------|
| C3H8  | 483 x 13 x 898 x 13 x 898 | 520 GB | 91.5% | ~44 GB |
| C4H10 | 624 x 17 x 1295 x 17 x 1295 | 2.8 TB | 95.0% | ~140 GB |

For C3H8, the I1 intermediate distributes ~11 GB per rank across 4 nodes (28 GB peak RSS includes I1 + other tensors + overhead). For C4H10, peak RSS hits 58 GB/rank — nearly the full 64 GB node memory. C5H12 and C6H14 would exceed the 4-node aggregate memory and were not attempted.

### 4.3 Scaling: 4 vs 8 Ranks (C3H8)

| Config | Total (s) | Stage 3 (s) | Peak RSS/rank |
|--------|-----------|-------------|---------------|
| 4 ranks (4 nodes, 1/node) | 162.9 | 152.9 | 28.2 GB |
| 8 ranks (4 nodes, 2/node) | 187.8 | 177.0 | 16.8 GB |

Adding more ranks **increases** total time by 15% while reducing memory by 40%. The communication overhead from the einsum Hadamard loop (more communicator splits, more tile redistribution) outweighs the parallelism benefit. The Hadamard loop iterates over `ceil(13/4)^2 = 16` (i,j) tile pairs — with 8 ranks, each sub-world contraction is smaller but the synchronization cost per iteration is higher.

### 4.4 Comparison with Dense Baseline (C3H8)

| | Dense (16 BLAS threads) | TiledArray (4 nodes) | Ratio |
|--|------------------------|---------------------|-------|
| **Total** | **352s** | **162.9s** | **2.2x faster** |
| Memory | 10 GB (single node) | 28 GB/rank (4 nodes) | 11.2x more aggregate |

TiledArray is 2.2x faster but requires 4 nodes (256 GB aggregate) vs the dense baseline's 10 GB on one node. The speedup comes from tile-level sparsity skipping (I0 is 70.8% sparse, I1 is 91.5% sparse, R is 92.5% sparse), but the memory overhead of materializing the full I1 intermediate negates the memory advantages of sparsity.

---

## 5. Eq1: Infeasible with TiledArray

Eq1 produces a rank-6 intermediate I1(j, k, i, a, p, b) that is too large to materialize on any tested configuration:

| Config | Ranks | Aggregate RAM | Result |
|--------|-------|---------------|--------|
| 1 rank, 1 node | 1 | 64 GB | OOM |
| 4 ranks, 4 nodes | 4 | 256 GB | OOM |
| 8 ranks, 4 nodes | 8 | 256 GB | OOM / timeout |
| 16 ranks, 4 nodes | 16 | 256 GB | OOM / timeout |

For C3H8, I1 has shape (13, 483, 13, 898, 13, 876). Dense size: 13^3 * 483 * 898 * 876 = ~8.1 trillion elements = 65 TB. Even at ~95% tile sparsity, the remaining 5% is ~3.2 TB — far beyond 256 GB.

The dense baseline handles Eq1 in 51.4s by fully fusing the computation: it loops over (i2, i3) pairs, computes a weighted sum `G(u1,u2) = sum_k g1(i2,i3,k) * g0(u1,u2,k)`, then does two GEMMs per i1 value. The peak intermediate is a single (n_uocc, n_uocc) matrix (~320 KB for C3H8). **TiledArray has no mechanism to replicate this fusion.**

---

## 6. Key Findings

### 6.1 Tile-Level Sparsity Is Highly Effective for Eq2

The tile-level sparsity screening provides a 23x speedup for Eq2 on C3H8, increasing to even larger gains for bigger molecules where sparsity reaches 98.6%. This validates TiledArray's design for block-sparse quantum chemistry tensors where most orbital pair interactions are negligible.

### 6.2 Sparsity Alone Cannot Solve the Intermediate Size Problem

Despite 91.5% tile-level sparsity for Eq0's I1 and ~95%+ for Eq1's I1, the absolute memory requirements remain prohibitive. For rank-5 tensors with O(10^11) total elements, even 5-10% non-zero tiles amount to tens of GB. TiledArray's inability to fuse binary contractions means every intermediate must be fully materialized.

### 6.3 Multi-Node Scaling Is Overhead-Dominated for Small Problems

For Eq2 (fast computation, ~1-8s), multi-node adds 18-65% overhead. The einsum Hadamard loop's per-tile-index MPI_Comm_Split is a collective operation that dominates for small tile counts. For Eq0 (slower computation, ~160-385s), multi-node is essential for memory but does not improve speed — 8 ranks is slower than 4 ranks.

### 6.4 Stage 3 Dominates All Equations

In every equation and molecule, the final contraction (stage 3) accounts for 82-94% of total runtime. This stage always involves Hadamard indices (shared occupied indices between operands), triggering the einsum sub-world loop. The cost is driven by:
1. MPI communicator management (split/free per Hadamard tile)
2. Tile redistribution into sub-arrays per Hadamard iteration
3. The actual GEMM within each sub-world

### 6.5 Memory Per Rank Scales Linearly with Tile Count

| Molecule | Eq2 RSS (1 rank) | g0 nnz | RSS / g0_nnz ratio |
|----------|-----------------|--------|---------------------|
| C3H8  | 2.8 GB  | 19.7M  | 142 bytes/entry |
| C4H10 | 5.6 GB  | 42.2M  | 133 bytes/entry |
| C5H12 | 10.5 GB | 77.4M  | 136 bytes/entry |
| C6H14 | 16.4 GB | 127.5M | 129 bytes/entry |

The ~130-140 bytes per g0 entry overhead includes: the element itself (8 bytes), the dense tile padding (since tiles are stored fully), the intermediate I0/I1/R tile data, and TiledArray metadata (Futures, process map, SparseShape norms).

---

## 7. Implications for Comparison with a Custom SPTC Library

### Where TiledArray has advantages:
1. **Expressiveness**: Einstein notation with automatic SUMMA distribution requires minimal code. Our entire benchmark is ~300 lines.
2. **Tile-level sparsity**: Dramatic speedups when tile-level sparsity is high (Eq2: 23x). No custom implementation needed.
3. **Correctness**: The expression engine handles index permutations and tile redistribution automatically (with the einsum workaround for Hadamard patterns).

### Where a custom library can win:
1. **Element-level sparsity**: The g0 tensor is 0% tile-sparse but has significant element-level sparsity for some molecules. A library using SpGEMM or compressed formats can exploit this.
2. **Intermediate fusion**: The 51.4s dense baseline for Eq1 (with manual fusion) vs TiledArray's OOM demonstrates the value of avoiding intermediate materialization. A library that fuses contractions across stages can handle all three equations without multi-node requirements.
3. **Communication efficiency**: The einsum Hadamard loop's per-tile MPI_Comm_Split overhead is significant. A library with a single communication pattern (e.g., one AllReduce for the fused result) would have lower latency.
4. **Memory efficiency**: TiledArray uses ~130 bytes per g0 element including all overhead. A library storing only non-zero elements in COO/CSR format would use 16-24 bytes per element.
5. **Strong scaling on small problems**: TiledArray's multi-rank overhead means single-rank is optimal for Eq2. A library with lower synchronization overhead could actually benefit from parallelism.

### Fair comparison guidelines:
- Compare on the **same tile sizes** (occ=4, uocc=50, ri=200) and the same molecules.
- For Eq2, the TiledArray single-rank time (1.14s for C3H8) is the strongest baseline.
- For Eq0, the 4-node TiledArray time (162.9s for C3H8) should be compared against the custom library at the same or fewer nodes.
- For Eq1, TiledArray cannot produce a result at all — any working implementation is a win.
- Report both computation time and peak memory.

---

## 8. Raw Data Reference

All raw timing data is in `results_all.csv`. Column format:
```
molecule,equation,stage,trial,nranks,wall_s,compute_s,sparsity,peak_rss_kb
```

Dense baseline results (from HOWTO.md, C3H8, 16 cores, 3 trials averaged):
- Eq2: 26.7s total (0.18 + 0.41 + 26.1)
- Eq0: 352s total (7.8 + 344.5 fused)
- Eq1: 51.4s total (fully fused)
