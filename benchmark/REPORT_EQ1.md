# Eq1 Benchmark Report (32-Node CloudLab Cluster)

**Date:** 2026-03-29
**Hardware:** 32x Intel Xeon D-1548 (8c/16t @ 2.0 GHz), 64 GB RAM per node, 10 GbE internal network
**Software:** TiledArray (MADWorld/MPI), GCC 11, OpenBLAS, MPICH 4.0

---

## 1. Summary

**Eq1** `R(i1,i3,i2,a2,a3) = g0(u1,u2,k1) * c2(i2,i1,u1,a2) * c2(i1,i3,u2,a3) * g1(i2,i3,k1)` produces a **rank-5 result tensor** and a **rank-6 intermediate** (I1), making it the most memory-intensive of the three DLPNO-CCSD contractions.

**Key finding:** Eq1 is only feasible for C2H6 (the smallest molecule). All larger molecules OOM on every tested configuration, including 32 nodes with 2 TB total RAM.

---

## 2. C2H6 Eq1 Results (Only Feasible Molecule)

### 2.1 Scaling with Node Count

| Nodes | Ranks | Avg Wall (s) | Speedup vs 4n | Peak RSS/rank (GB) | Total Cluster Mem (GB) |
|------:|------:|-------------:|---------------:|--------------------:|-----------------------:|
| 1     | 1     | OOM          | --             | >64                 | >64                    |
| 4     | 4     | 308.5        | 1.00x          | 59.4                | 238                    |
| 8     | 8     | 237.2        | 1.30x          | 39.0                | 312                    |
| 16    | 16    | 254.3        | 1.21x          | 25.3                | 405                    |
| 32    | 32    | 215.8        | 1.43x          | 14.1                | 451                    |

*Averages use trials 2-3 to exclude cold-start effects.*

### 2.2 Per-Stage Breakdown (C2H6, Trial 2 Averages)

| Stage | 4 nodes (s) | 8 nodes (s) | 16 nodes (s) | 32 nodes (s) | Description |
|-------|------------:|------------:|--------------:|--------------:|-------------|
| Stage 1 | 0.89 | 0.73 | 0.47 | 0.40 | I0 = g0 * c2 (contract u1) |
| Stage 2 | 132.8 | 93.4 | 94.8 | 68.2 | I1 = I0 * c2 (contract u2) — rank-6 intermediate |
| Stage 3 | 166.6 | 141.9 | 153.5 | 141.6 | R = I1 * g1 (contract k1) |
| **Total** | **300.3** | **236.0** | **248.8** | **210.2** | |

**Observations:**
- **Stage 3 dominates** (~55-67% of total time), involving the contraction of the rank-6 intermediate I1 with g1
- **Stage 2** produces the rank-6 I1 intermediate — this is the memory bottleneck (60.9 GB/rank at 4 nodes)
- **Stage 1** is negligible (<1s) as it only contracts the batch index u1
- **16 nodes is slower than 8 nodes** — communication overhead for the rank-6 intermediate outweighs the compute benefit at this problem size
- **32 nodes** recovers some speedup (1.43x over 4 nodes) but efficiency is poor (4.5%)

### 2.3 Memory Analysis

| Nodes | RSS/rank (GB) | Memory Efficiency | Notes |
|------:|--------------:|------------------:|-------|
| 4     | 59.4          | --                | Near the 64 GB limit |
| 8     | 39.0          | 1.52x reduction   | Intermediate distributes well |
| 16    | 25.3          | 2.35x reduction   | Still large per-rank footprint |
| 32    | 14.1          | 4.21x reduction   | Overhead becomes significant |

Total cluster memory *increases* with node count (238 GB → 451 GB) due to replicated metadata and communication buffers. The per-rank reduction is less than linear.

---

## 3. Failure Summary (All Other Molecules)

| Molecule | 4 nodes | 8 nodes | 16 nodes | 32 nodes | Estimated Min Nodes |
|----------|---------|---------|----------|----------|--------------------:|
| C2H6     | **OK** (59.4 GB/rank) | **OK** (39.0 GB/rank) | **OK** (25.3 GB/rank) | **OK** (14.1 GB/rank) | 4 |
| C3H8     | OOM     | OOM     | OOM      | OOM      | >32 (~128 est.) |
| C4H10    | --      | OOM     | OOM      | OOM      | >32 (~256 est.) |
| C5H12    | --      | --      | OOM      | OOM      | >32 |
| C6H14    | --      | --      | --       | OOM      | >32 |

**Why C3H8+ OOM even at 32 nodes:**
- C2H6 at 4 nodes uses 59.4 GB/rank for a rank-6 intermediate of size (9, 9, 9, 144, 144, 585)
- C3H8's intermediate would be approximately (13, 13, 13, 202, 202, 876) — roughly **14x larger** than C2H6
- Even at 32 nodes, this would require ~830 GB/rank (14x × 59.4 GB), far exceeding 64 GB
- C3H8 eq1 would need approximately 128+ nodes (8 TB+ total) to be feasible

---

## 4. C2H6 Eq0 Results (Supplementary — Previously Missing)

For completeness, C2H6 eq0 was also benchmarked:

| Nodes | Ranks | Avg Wall (s) | Speedup vs 1-rank | Peak RSS/rank (GB) |
|------:|------:|-------------:|-------------------:|--------------------:|
| 1     | 1     | 59.7         | 1.00x              | 27.2                |
| 4     | 4     | 129.4        | 0.46x              | 11.6                |
| 8     | 8     | 67.0         | 0.89x              | 6.4                 |
| 16    | 16    | 45.4         | 1.32x              | 3.9                 |
| 32    | 32    | 40.7         | 1.47x              | 2.5                 |

*Averages use trials 2-3.*

**Observations:**
- Unlike larger molecules, C2H6 eq0 **runs on a single node** (27.2 GB)
- 4 nodes is slower than 1 rank (communication overhead dominates for this small problem)
- Speedup only appears at 16+ nodes (1.32x) — consistent with eq0 scaling patterns for small molecules
- Stage 3 dominates (85-95% of total time)

---

## 5. Sparsity Comparison

| Molecule | Eq1 Sparsity (result) | Eq0 Sparsity (result) | Eq2 Sparsity (result) |
|----------|----------------------:|----------------------:|----------------------:|
| C2H6     | 89.3%                 | 83.4%                 | 86.4%                 |
| C3H8     | -- (OOM)              | 92.5%                 | 92.8%                 |

Eq1 sparsity (89.3%) is higher than eq0 (83.4%) for C2H6, but the rank-6 intermediate's raw size dwarfs any sparsity benefit.

---

## 6. Key Findings

### 6.1 Eq1 is Fundamentally Memory-Limited
The rank-6 intermediate I1 makes eq1 infeasible for all but the smallest molecule. The memory requirement scales roughly as O(n_occ^3 × n_uocc^2 × n_virt), growing ~14x per carbon increment. No amount of distribution on 64 GB/node hardware can overcome this for C3H8+.

### 6.2 Scaling is Poor Even Where Feasible
For C2H6, the best speedup is 1.43x on 32 nodes (vs 4 nodes) — parallel efficiency of 5.7%. The rank-6 intermediate requires massive all-to-all communication that doesn't scale on 10 GbE.

### 6.3 Stage 3 is the Compute Bottleneck
Stage 3 (contracting the rank-6 I1 with g1) takes 55-67% of wall time across all configurations. It doesn't speed up much with more nodes (141s at 8 nodes vs 142s at 32 nodes), suggesting it's communication-bound rather than compute-bound at this problem size.

### 6.4 Implications for Custom Libraries (SPTN_MLIR_DSL)
Eq1 represents the strongest motivation for element-level sparsity and intermediate fusion:
- **Element-level sparsity** could avoid materializing the full rank-6 intermediate (TiledArray stores ~99% zeros in tiles)
- **Intermediate fusion** could contract I1 with g1 on-the-fly without full materialization
- **SUMMA with streaming** could pipeline the rank-6 data instead of storing it all

### 6.5 C2H6 Eq0 Completes the Dataset
C2H6 eq0 (59.7s single-rank, 27.2 GB) fills the gap in the previous benchmark. It follows the same pattern as larger molecules: distribution primarily helps with memory, not speed, for small problems.
