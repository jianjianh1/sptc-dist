# 32-Node Sparse Tensor Contraction Benchmark Analysis

**Date:** 2026-03-29
**Hardware:** 32x Intel Xeon D-1548 (8 cores / 16 threads @ 2.0 GHz), 64 GB RAM per node, 10 GbE interconnect
**Software:** TiledArray (MADWorld/MPI), GCC, OpenBLAS
**Dataset:** DLPNO-CCSD sparse tensors for alkane molecules C2H6 through C6H14

---

## Table of Contents

1. [Run Configurations](#1-run-configurations)
2. [Eq2 Results](#2-eq2-results)
3. [Eq0 Results](#3-eq0-results)
4. [Eq1 Results](#4-eq1-results)
5. [Memory Analysis](#5-memory-analysis)
6. [Key Findings](#6-key-findings)

---

## 1. Run Configurations

Three run configurations were tested:

| Config | Ranks per Node | Node Counts | Total Ranks |
|--------|---------------|-------------|-------------|
| Single-node | 1, 2, 4, 8 | 1 | 1, 2, 4, 8 |
| Multi-node 1ppn | 1 | 4, 8, 16, 32 | 4, 8, 16, 32 |
| Multi-node 2ppn | 2 | 4, 8, 16, 32 | 8, 16, 32, 64 |

Each configuration was run for 3 trials. Averages below use trials 2-3 (excluding cold-start trial 1) unless trial 1 was within 15% of the trial 2-3 mean, in which case all 3 trials are averaged.

**Duplicate disambiguation:** Where the same (equation, molecule, nranks) appears multiple times, runs are classified by wall time magnitude -- single-node multi-rank runs are slowest (high communication overhead on shared memory), multi-node 2ppn are intermediate, and multi-node 1ppn are fastest.

---

## 2. Eq2 Results

**Equation:** `R(i3,i2,a2,i5,a3) = g(i2,m1,k1) * g(i3,m2,k1) * c1(i2,m1,a2) * c2(i3,i5,m2,a3)`
Contracted indices: k1 -> m1 -> m2 (3 stages)

### 2.1 Single-Node Scaling (C3H8)

| Ranks | Avg Wall (s) | Speedup | Efficiency | Peak RSS/rank (MB) |
|------:|-------------:|--------:|-----------:|--------------------:|
| 1 | 1.117 | 1.00x | 100.0% | 2759 |
| 2 | 4.194 | 0.27x | 13.3% | 2077 |
| 4 | 5.562 | 0.20x | 5.0% | 1226 |
| 8 | 8.238 | 0.14x | 1.7% | 1174 |

**Observation:** Single-node multi-rank performance is catastrophically poor -- adding ranks on one node makes eq2 *slower*, not faster. Wall time increases nearly linearly with rank count. The MPI communication overhead within a single node completely dominates the relatively small compute workload of C3H8-eq2.

### 2.2 Multi-Node Scaling, 1 Process per Node (C3H8)

Baseline: single-node 1-rank = 1.117s

| Nodes | Ranks | Avg Wall (s) | Speedup vs 1-rank | Efficiency |
|------:|------:|-------------:|-------------------:|-----------:|
| 4 | 4 | 1.728 | 0.65x | 16.2% |
| 8 | 8 | 1.607 | 0.69x | 8.7% |
| 16 | 16 | 1.503 | 0.74x | 4.6% |
| 32 | 32 | 1.497 | 0.75x | 2.3% |

**Observation:** Multi-node 1ppn is much faster than single-node multi-rank at the same rank count (e.g., 1.607s vs 8.238s at 8 ranks), but still slower than a single rank. The problem is too small for C3H8 to benefit from distribution. Larger molecules tell a different story (see Section 2.4).

### 2.3 Multi-Node Scaling, 2 Processes per Node (C3H8)

Baseline: single-node 1-rank = 1.117s

| Nodes | Ranks | Avg Wall (s) | Speedup vs 1-rank | Efficiency |
|------:|------:|-------------:|-------------------:|-----------:|
| 4 | 8 | 6.542 | 0.17x | 2.1% |
| 8 | 16 | 7.333 | 0.15x | 1.0% |
| 16 | 32 | 8.673 | 0.13x | 0.4% |
| 32 | 64 | 9.917 | 0.11x | 0.2% |

**Observation:** 2ppn is dramatically worse than 1ppn. Two ranks per node reintroduces intra-node communication overhead, and performance degrades with scale. Adding more nodes with 2ppn actually makes things *slower*. This configuration should be avoided for eq2.

### 2.4 Multi-Node 1ppn Molecule Scaling (All Molecules)

Single-node baseline (1 rank):

| Molecule | Atoms | Wall (s) | Peak RSS (MB) | Sparsity |
|----------|------:|----------:|---------------:|---------:|
| C2H6 | 8 | 0.246 | 976 | 0.864 |
| C3H8 | 11 | 1.117 | 2759 | 0.928 |
| C4H10 | 14 | 2.329 | 5500 | 0.967 |
| C5H12 | 17 | 4.995 | 10179 | 0.982 |
| C6H14 | 20 | 8.323 | 15553 | 0.986 |

Multi-node 1ppn results:

| Molecule | 4 nodes (s) | 8 nodes (s) | 16 nodes (s) | 32 nodes (s) | Best Speedup |
|----------|------------:|------------:|--------------:|--------------:|-------------:|
| C2H6 | 0.763 | 0.627 | 0.703 | 0.795 | 0.39x (8n) |
| C3H8 | 1.728 | 1.607 | 1.503 | 1.497 | 0.75x (32n) |
| C4H10 | 3.023 | 2.646 | 2.053 | 1.778 | 1.31x (32n) |
| C5H12 | 5.223 | 4.206 | 3.331 | 3.219 | 1.55x (32n) |
| C6H14 | 9.125 | 7.680 | 5.188 | 4.244 | 1.96x (32n) |

**Key insight:** Distribution only pays off for larger molecules. C6H14 at 32 nodes achieves nearly 2x speedup, while C2H6 and C3H8 are always slower than a single rank. The crossover point where distribution breaks even is around C4H10 at 16 nodes.

---

## 3. Eq0 Results

**Equation:** `R(i1,i2,a1,a2) = g0(u1,u2,k1) * c(i1,u1,a1) * c(i2,u2,a2) * g1(i2,i1,k1)`
Contracted indices: u1 -> u2 -> k1 (3 stages)

Eq0 is far more compute-intensive than eq2 (100-1000x longer wall times) and memory-hungry.

### 3.1 Multi-Node 1ppn Scaling (C3H8)

Baseline: 4 nodes = 164.78s (no single-node data available -- eq0 OOMs at 1 node for all molecules)

| Nodes | Ranks | Avg Wall (s) | Speedup vs 4-node | Relative Efficiency |
|------:|------:|-------------:|-------------------:|--------------------:|
| 4 | 4 | 164.78 | 1.00x | 100% |
| 8 | 8 | 109.90 | 1.50x | 75.0% |
| 16 | 16 | 65.46 | 2.52x | 63.0% |
| 32 | 32 | 51.26 | 3.21x | 40.2% |

### 3.2 Multi-Node 2ppn Scaling (C3H8)

| Nodes | Ranks | Avg Wall (s) | Speedup vs 4n-1ppn | Relative Efficiency |
|------:|------:|-------------:|--------------------:|--------------------:|
| 4 | 8 | 188.10 | 0.88x | 43.8% |
| 8 | 16 | 120.02 | 1.37x | 34.3% |
| 16 | 32 | 94.91 | 1.74x | 21.7% |
| 32 | 64 | 67.43 | 2.44x | 15.3% |

**Observation:** Unlike eq2, eq0 shows meaningful scaling with both 1ppn and 2ppn configs because the compute workload is much larger. However, 1ppn is consistently better than 2ppn at the same rank count -- e.g., 8 ranks with 1ppn = 109.90s vs 8 ranks with 2ppn = 188.10s. The 1ppn configuration achieves a 3.2x speedup going from 4 to 32 nodes.

### 3.3 1ppn vs 2ppn Comparison at Same Rank Count (C3H8)

| Ranks | 1ppn Wall (s) | 2ppn Wall (s) | 1ppn Advantage |
|------:|--------------:|--------------:|---------------:|
| 8 | 109.90 | 188.10 | 1.71x faster |
| 16 | 65.46 | 120.02 | 1.83x faster |
| 32 | 51.26 | 94.91 | 1.85x faster |

The 1ppn advantage increases with rank count, confirming that intra-node MPI communication is a severe bottleneck.

### 3.4 Eq0 Molecule Scaling (Multi-Node 1ppn)

| Molecule | 8 nodes (s) | 16 nodes (s) | 32 nodes (s) | 8->32 Speedup |
|----------|------------:|--------------:|--------------:|---------------:|
| C3H8 | 109.90 | 65.46 | 51.26 | 2.14x |
| C4H10 | 348.22 | 209.47 | 95.99 | 3.63x |
| C5H12 | 782.03 | 480.56 | 270.60 | 2.89x |
| C6H14 | -- (OOM) | 753.27 | 477.03 | -- |

**Note:** C3H8 also runs at 4 nodes (164.78s). C6H14 requires at least 16 nodes (OOM at 8 nodes). C4H10 shows the best 8->32 node scaling at 3.63x.

---

## 4. Eq1 Results (All Failures)

**Equation:** `R(i1,i3,i2,a2,a3) = g0(u1,u2,k1) * c(i2,i1,u1,a2) * c(i1,i3,u2,a3) * g1(i2,i3,k1)`

| Molecule | Config | Result | Notes |
|----------|--------|--------|-------|
| C3H8 | All configs | OOM | Killed (exit code 9) |
| C4H10 | All configs | OOM | Killed (exit code 9) |
| C5H12 | All configs | OOM | Killed (exit code 9) |
| C6H14 | All configs | OOM | Killed (exit code 9) |

**All eq1 configurations failed with OOM**, even using all 32 nodes at 2ppn (64 ranks). Eq1 produces a 5-dimensional result tensor `R(i1,i3,i2,a2,a3)` with two extra orbital indices compared to eq0 and eq2, causing the intermediate and result tensors to far exceed available memory. The CSV shows multiple `BAD TERMINATION` entries with exit code 9 (SIGKILL, typically from OOM killer).

---

## 5. Memory Analysis

### 5.1 Eq2 Per-Rank Peak RSS (MB), Multi-Node 1ppn

| Molecule | 1 rank | 4 nodes | 8 nodes | 16 nodes | 32 nodes |
|----------|-------:|--------:|--------:|---------:|---------:|
| C2H6 | 976 | 525 | 388 | 353 | 328 |
| C3H8 | 2,759 | 1,226 | 1,174 | 1,174 | 1,174 |
| C4H10 | 5,500 | 2,511 | 2,326 | 2,326 | 2,326 |
| C5H12 | 10,179 | 4,629 | 4,630 | 4,629 | 4,630 |
| C6H14 | 15,553 | 7,032 | 5,950 | 5,455 | 5,196 |

**Observation:** Memory drops significantly going from 1 rank to 4 nodes (roughly halved), but then plateaus for most molecules. TiledArray replicates global metadata across all ranks, creating a per-rank memory floor. For C3H8, this floor is about 1.17 GB regardless of node count. Larger molecules (C6H14) show continued gradual reduction, suggesting their data distribution is more effective.

### 5.2 Eq0 Per-Rank Peak RSS (MB), Multi-Node 1ppn

| Molecule | 4 nodes | 8 nodes | 16 nodes | 32 nodes |
|----------|--------:|--------:|---------:|---------:|
| C3H8 | 27,685 | 16,039 | 8,439 | 5,349 |
| C4H10 | -- | 32,814 | 24,467 | 9,272 |
| C5H12 | -- | 59,751 | 33,409 | 17,561 |
| C6H14 | -- | -- (OOM) | 46,861 | 23,469 |

**Observation:** Eq0 memory scales much more effectively with node count compared to eq2. C3H8 drops from 27.7 GB at 4 nodes to 5.3 GB at 32 nodes (5.2x reduction for 8x more nodes). This is because eq0's intermediate tensors are large and genuinely distributed. C5H12 at 8 nodes uses nearly 60 GB per rank -- close to the 64 GB limit, explaining why fewer nodes fail with OOM.

### 5.3 Total Cluster Memory Usage Estimates (Eq0, 1ppn)

| Molecule | 8 nodes | 16 nodes | 32 nodes |
|----------|--------:|---------:|---------:|
| C3H8 | 125 GB | 132 GB | 167 GB |
| C4H10 | 256 GB | 383 GB | 290 GB |
| C5H12 | 466 GB | 522 GB | 549 GB |
| C6H14 | OOM | 732 GB | 733 GB |

Total cluster memory shows that the aggregate footprint grows with node count (data replication overhead), but each node stays within its 64 GB limit. C6H14 at 8 nodes would need roughly 90+ GB per node, exceeding the 64 GB limit.

---

## 6. Key Findings

### 6.1 Distribution is Counterproductive for Small Problems

For eq2 with small molecules (C2H6, C3H8), single-rank execution is fastest. The ~1s compute time is overwhelmed by MPI communication overhead. Even with 32 nodes, C3H8 eq2 cannot beat a single rank. **Recommendation:** Skip distribution for eq2 problems with wall time under ~3s.

### 6.2 Intra-Node Multi-Rank is Always Harmful

Running multiple MPI ranks on a single node causes severe slowdowns (up to 8x worse at 8 ranks). The 2ppn multi-node configuration is also consistently worse than 1ppn at the same rank count (1.7-1.9x slower for eq0). **Recommendation:** Always use 1 process per node.

### 6.3 Eq0 Scales Reasonably Well

Eq0 (the most compute-intensive equation) achieves meaningful speedups:
- C3H8: 3.2x speedup from 4 to 32 nodes (1ppn)
- C4H10: 3.6x speedup from 8 to 32 nodes (1ppn)
- Memory scales nearly linearly, enabling larger problems

The parallel efficiency is 40-75% depending on the scaling range, which is reasonable for sparse tensor operations over 10 GbE.

### 6.4 Eq2 Benefits Only Large Molecules

Eq2 distribution is only worthwhile for C5H12 (1.55x at 32 nodes) and C6H14 (1.96x at 32 nodes). The speedup trend suggests that even larger molecules (C7H16+) would show better scaling. **Recommendation:** For production use, only distribute eq2 for molecules with single-rank wall time exceeding ~5s.

### 6.5 Memory, Not Compute, Drives the Need for Distribution

The primary motivation for distributing eq0 is memory, not speed:
- C6H14 eq0 requires at least 16 nodes (46.9 GB/rank) to avoid OOM
- C5H12 eq0 requires at least 8 nodes (59.8 GB/rank, barely fitting in 64 GB)
- Eq1 exceeds 64 GB per rank even at 64 total ranks, making it infeasible on this hardware

### 6.6 Eq1 is Infeasible on Current Hardware

The 5-dimensional result tensor of eq1 creates memory requirements that exceed what 32 nodes x 64 GB can provide. Possible solutions:
- Out-of-core or checkpointing approaches
- Nodes with significantly more RAM (256+ GB)
- Algorithmic reformulation to avoid materializing the full 5D intermediate

### 6.7 Sparsity Increases with Molecule Size

| Molecule | Eq2 Sparsity | Eq0 Sparsity (stage 3) |
|----------|-------------:|-----------------------:|
| C2H6 | 86.4% | -- |
| C3H8 | 92.8% | 92.5% |
| C4H10 | 96.7% | 96.1% |
| C5H12 | 98.2% | 97.6% |
| C6H14 | 98.6% | 98.3% |

Higher sparsity means fewer non-zero tiles to compute, but also means communication-to-compute ratio worsens (less work per distributed tile). This explains why scaling efficiency does not improve proportionally with molecule size.

### 6.8 10 GbE is a Bottleneck

The consistent pattern of 1ppn outperforming 2ppn by 1.7-1.9x, combined with the poor multi-node scaling for small problems, strongly suggests the 10 GbE network is a bottleneck. Higher-bandwidth interconnects (25/100 GbE, InfiniBand) would likely improve multi-node scaling significantly, especially for eq2 where communication dominates.

---

## Appendix: Raw Averages

### Eq2 All Configurations (Wall Time in Seconds)

| Molecule | Config | Ranks | Nodes | Avg Wall (s) |
|----------|--------|------:|------:|--------------:|
| C2H6 | 1ppn | 4 | 4 | 0.763 |
| C2H6 | 1ppn | 8 | 8 | 0.627 |
| C2H6 | 1ppn | 16 | 16 | 0.703 |
| C2H6 | 1ppn | 32 | 32 | 0.795 |
| C3H8 | single | 1 | 1 | 1.117 |
| C3H8 | single | 2 | 1 | 4.194 |
| C3H8 | single | 4 | 1 | 5.562 |
| C3H8 | single | 8 | 1 | 8.238 |
| C3H8 | 1ppn | 4 | 4 | 1.728 |
| C3H8 | 1ppn | 8 | 8 | 1.607 |
| C3H8 | 1ppn | 16 | 16 | 1.503 |
| C3H8 | 1ppn | 32 | 32 | 1.497 |
| C3H8 | 2ppn | 8 | 4 | 6.542 |
| C3H8 | 2ppn | 16 | 8 | 7.333 |
| C3H8 | 2ppn | 32 | 16 | 8.673 |
| C3H8 | 2ppn | 64 | 32 | 9.917 |
| C4H10 | 1ppn | 4 | 4 | 3.023 |
| C4H10 | 1ppn | 8 | 8 | 2.646 |
| C4H10 | 1ppn | 16 | 16 | 2.053 |
| C4H10 | 1ppn | 32 | 32 | 1.778 |
| C5H12 | 1ppn | 4 | 4 | 5.223 |
| C5H12 | 1ppn | 8 | 8 | 4.206 |
| C5H12 | 1ppn | 16 | 16 | 3.331 |
| C5H12 | 1ppn | 32 | 32 | 3.219 |
| C6H14 | 1ppn | 4 | 4 | 9.125 |
| C6H14 | 1ppn | 8 | 8 | 7.680 |
| C6H14 | 1ppn | 16 | 16 | 5.188 |
| C6H14 | 1ppn | 32 | 32 | 4.244 |

### Eq0 All Configurations (Wall Time in Seconds)

| Molecule | Config | Ranks | Nodes | Avg Wall (s) |
|----------|--------|------:|------:|--------------:|
| C3H8 | 1ppn | 4 | 4 | 164.78 |
| C3H8 | 1ppn | 8 | 8 | 109.90 |
| C3H8 | 1ppn | 16 | 16 | 65.46 |
| C3H8 | 1ppn | 32 | 32 | 51.26 |
| C3H8 | 2ppn | 8 | 4 | 188.10 |
| C3H8 | 2ppn | 16 | 8 | 120.02 |
| C3H8 | 2ppn | 32 | 16 | 94.91 |
| C3H8 | 2ppn | 64 | 32 | 67.43 |
| C4H10 | 1ppn | 8 | 8 | 348.22 |
| C4H10 | 1ppn | 16 | 16 | 209.47 |
| C4H10 | 1ppn | 32 | 32 | 95.99 |
| C5H12 | 1ppn | 8 | 8 | 782.03 |
| C5H12 | 1ppn | 16 | 16 | 480.56 |
| C5H12 | 1ppn | 32 | 32 | 270.60 |
| C6H14 | 1ppn | 16 | 16 | 753.27 |
| C6H14 | 1ppn | 32 | 32 | 477.03 |
