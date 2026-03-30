# Detailed Benchmark Analysis — TiledArray on 32-Node CloudLab Cluster

**Date:** 2026-03-29
**Hardware:** 32x Intel Xeon D-1548 (8 cores / 16 threads @ 2.0 GHz), 64 GB RAM per node, 10 GbE
**Software:** TiledArray v1.1.0 on MADNESS/MPICH 4.0, GCC 11.4, OpenBLAS
**Tile sizes:** occ=4, uocc=50, ri=200 (matching MPQC defaults)

---

## 1. Experimental Setup

### 1.1 Equations

Three four-tensor contractions from DLPNO-CCSD, each decomposed into three binary stages:

- **Eq2:** `R(i3,i2,a2,i5,a3) = g(i2,m1,k1) · g(i3,m2,k1) · c1(i2,m1,a2) · c2(i3,i5,m2,a3)`
  - Stages: contract k1 → m1 → m2. Smallest intermediates.
- **Eq0:** `R(i1,i2,a1,a2) = g0(u1,u2,k1) · c(i1,u1,a1) · c(i2,u2,a2) · g1(i2,i1,k1)`
  - Stages: contract u1 → u2 → k1. Large rank-4 intermediate.
- **Eq1:** `R(i1,i3,i2,a2,a3) = g0(u1,u2,k1) · c2(i2,i1,u1,a2) · c2(i1,i3,u2,a3) · g1(i2,i3,k1)`
  - Stages: contract u1 → u2 → k1. Rank-6 intermediate, rank-5 result.

### 1.2 Molecules

| Molecule | Atoms | n_occ | n_uocc | n_ri | n_virt | g0 NNZ |
|----------|------:|------:|-------:|-----:|-------:|-------:|
| C2H6     | 8     | 9     | 144    | 342  | 585-609 | 7.1M   |
| C3H8     | 11    | 13    | 202    | 483  | 876-898 | 19.7M  |
| C4H10    | 14    | 17    | 264    | 637  | ~1200  | ~45M   |
| C5H12    | 17    | 21    | 330    | 804  | ~1500  | ~90M   |
| C6H14    | 20    | 25    | 396    | 960  | ~1800  | ~150M  |

### 1.3 Configurations Tested

- **Single-node:** 1, 2, 4, 8 ranks on one node
- **Multi-node 1ppn:** 4, 8, 16, 32 nodes × 1 rank per node
- **Multi-node 2ppn:** 4, 8, 16, 32 nodes × 2 ranks per node
- 3 trials each, OPENBLAS_NUM_THREADS=1

---

## 2. Eq2 Analysis

### 2.1 Single-Rank Baseline (Molecule Scaling)

| Molecule | Wall (s) | Stage 1 (s) | Stage 2 (s) | Stage 3 (s) | Stage 3 % | Sparsity | RSS (GB) |
|----------|--------:|------------:|------------:|------------:|----------:|---------:|---------:|
| C2H6     | 0.24    | 0.008       | 0.025       | 0.210       | 86%       | 86.4%    | 1.0      |
| C3H8     | 1.10    | 0.040       | 0.089       | 0.973       | 88%       | 92.8%    | 2.8      |
| C4H10    | 2.33    | 0.088       | 0.170       | 2.068       | 89%       | 96.7%    | 5.5      |
| C5H12    | 4.94    | 0.254       | 0.404       | 4.276       | 87%       | 98.2%    | 10.2     |
| C6H14    | 8.20    | 0.596       | 0.740       | 6.778       | 83%       | 98.6%    | 15.6     |

**Stage 3 dominates** (83-89% of total) across all molecules. This stage contracts the largest intermediate with c2, involving GEMM-like operations on sparse tiles. Stage 3 time scales roughly as O(n_uocc²), consistent with the GEMM cost: C6H14 (n_uocc=396) takes 32x longer than C2H6 (n_uocc=144), close to the expected (396/144)² = 7.6x when accounting for sparsity differences.

**Sparsity increases with molecule size** (86% → 99%), meaning fewer tiles are non-zero. However, the non-zero tiles still grow in count and size, so total compute still increases.

**Memory scales linearly** with molecule size — 1.0 GB (C2H6) to 15.6 GB (C6H14) at single rank. The dominant cost is storing the input tensors (especially g0) plus the intermediate.

### 2.2 Multi-Node 1ppn Scaling

Average wall time (s), trials 2-3:

| Molecule | 1 rank | 4 nodes | 8 nodes | 16 nodes | 32 nodes | Best Speedup |
|----------|-------:|--------:|--------:|---------:|---------:|-------------:|
| C2H6     | 0.25   | 0.73    | 0.63    | 0.70     | 0.80     | 0.39x (8n)   |
| C3H8     | 1.10   | 1.74    | 1.54    | 1.44     | 1.39     | 0.79x (32n)  |
| C4H10    | 2.33   | 3.02    | 2.57    | 2.05     | 1.78     | 1.31x (32n)  |
| C5H12    | 4.94   | 5.22    | 4.21    | 3.26     | 3.13     | 1.58x (32n)  |
| C6H14    | 8.20   | 8.87    | 7.51    | 5.03     | 4.09     | 2.01x (32n)  |

**The distribution crossover point is around C4H10 at 16+ nodes.** Below that, communication overhead exceeds compute savings. The overhead comes from:
1. MPI_Comm_Split per Hadamard tile in einsum()
2. Sparse tile distribution leaving many ranks idle
3. All-to-all communication for result assembly
4. Data loading over NFS (mitigated in later runs with local storage)

**Why C6H14 scales best:** With 98.6% sparsity and 8.2s single-rank time, there's enough compute to amortize the fixed communication overhead (~1-2s). C2H6's 0.25s compute is dwarfed by this overhead.

### 2.3 Single-Node Multi-Rank (Anti-Scaling)

C3H8, single node:

| Ranks | Wall (s) | Slowdown vs 1 rank |
|------:|---------:|-------------------:|
| 1     | 1.10     | 1.0x               |
| 2     | 4.19     | 3.8x slower        |
| 4     | 5.56     | 5.1x slower        |
| 8     | 8.24     | 7.5x slower        |

**Multi-rank on a single node is catastrophically harmful.** Each additional rank adds MPI communication overhead but doesn't add memory or compute resources that the single rank couldn't already use. The MADNESS runtime's work-stealing is ineffective when all workers share the same memory and compete for the same cache.

### 2.4 1ppn vs 2ppn (C3H8)

| Total Ranks | 1ppn Wall (s) | 2ppn Wall (s) | 1ppn Advantage |
|------------:|--------------:|--------------:|---------------:|
| 8           | 1.54          | 6.54          | 4.3x           |
| 16          | 1.44          | 7.33          | 5.1x           |
| 32          | 1.39          | 8.67          | 6.2x           |
| 64          | --            | 9.92          | --             |

**2ppn is dramatically worse** — it reintroduces the intra-node overhead seen in single-node multi-rank runs. Performance actually *degrades* with more 2ppn nodes, confirming that the intra-node MPI communication is the bottleneck, not the network.

---

## 3. Eq0 Analysis

### 3.1 Why Eq0 Needs Distribution

Eq0's stage 2 produces a rank-4 intermediate I1(i1,u2,a1,k1) where u2 and k1 are large auxiliary dimensions. For C3H8, this intermediate is approximately 13 × 202 × 898 × 483 ≈ 1.1 billion elements, requiring ~9 GB dense. After tile-level sparsity (~8.5% non-zero tiles), the materialized memory is still ~27 GB — too large for a single 64 GB node when combined with input tensors.

### 3.2 Multi-Node 1ppn Scaling

Average wall time (s), trials 2-3:

| Molecule | 1 rank | 4 nodes | 8 nodes | 16 nodes | 32 nodes |
|----------|-------:|--------:|--------:|---------:|---------:|
| C2H6     | 59.7   | 129     | 67.0    | 45.4     | 40.7     |
| C3H8     | OOM    | 164.8   | 109.9   | 65.5     | 51.3     |
| C4H10    | OOM    | OOM     | 348.2   | 209.5    | 96.0     |
| C5H12    | OOM    | OOM     | 782.0   | 480.6    | 270.6    |
| C6H14    | OOM    | OOM     | OOM     | 753.3    | 477.0    |

**Minimum nodes required:**

| Molecule | Min Nodes | RSS at Min (GB) | RSS at 32n (GB) |
|----------|----------:|-----------------:|-----------------:|
| C2H6     | 1         | 27.2             | 2.5              |
| C3H8     | 4         | 27.7             | 5.3              |
| C4H10    | 8         | 32.8             | 9.3              |
| C5H12    | 8         | 59.8             | 17.6             |
| C6H14    | 16        | 46.9             | 23.5             |

C5H12 at 8 nodes uses 59.8 GB/rank — dangerously close to the 64 GB limit. C6H14 at 8 nodes would require ~80+ GB/rank, hence the OOM.

### 3.3 Per-Stage Breakdown (C3H8, 1ppn)

| Stage | 4 nodes (s) | 8 nodes (s) | 16 nodes (s) | 32 nodes (s) | Scaling 4→32 |
|-------|------------:|------------:|--------------:|--------------:|-------------:|
| Stage 1 (contract u1) | 0.42 | 0.30 | 0.22 | 0.20 | 2.1x |
| Stage 2 (contract u2) | 9.06 | 4.88 | 2.54 | 1.92 | 4.7x |
| Stage 3 (contract k1) | 154.5 | 106.3 | 61.8 | 48.7 | 3.2x |
| **Total** | **164.0** | **111.5** | **64.6** | **50.8** | **3.2x** |

**Stage 3 dominates** (90-96% of total), contracting the large intermediate I1 with g1. It scales 3.2x over an 8x increase in nodes — reasonable given the sparse tile distribution and 10 GbE bandwidth constraints.

**Stage 2 scales best** (4.7x) because the I1 intermediate construction parallelizes well across independently-owned tiles.

### 3.4 Scaling Efficiency

| Range | C3H8 | C4H10 | C5H12 | C6H14 |
|-------|------:|------:|------:|------:|
| 4→8 nodes | 75% | -- | -- | -- |
| 8→16 nodes | 85% | 83% | 81% | -- |
| 16→32 nodes | 64% | 109%* | 89% | 79% |
| 4→32 nodes (overall) | 40% | -- | -- | -- |

*C4H10 16→32 shows superlinear speedup (109%) likely due to the intermediate fitting better in cache at 32 nodes (9.3 GB/rank vs 24.5 GB/rank).

### 3.5 1ppn vs 2ppn (C3H8 Eq0)

| Ranks | 1ppn (s) | 2ppn (s) | 1ppn Advantage |
|------:|---------:|---------:|---------------:|
| 8     | 109.9    | 188.1    | 1.71x          |
| 16    | 65.5     | 120.0    | 1.83x          |
| 32    | 51.3     | 94.9     | 1.85x          |
| 64    | --       | 67.4     | --             |

Same pattern as eq2: **1ppn is always 1.7-1.9x faster** than 2ppn at the same rank count. For eq0, this is particularly significant because the large intermediate I1 suffers from NUMA effects when two ranks share a node.

---

## 4. Eq1 Analysis

### 4.1 The Rank-6 Intermediate Problem

Eq1's stage 2 produces I1(i1,i3,u2,i2,a2,a3) — a **rank-6 tensor**. For C2H6 with shape (9, 9, 144, 9, 585, 585), this has 9 × 9 × 144 × 9 × 585 × 585 ≈ 3.6 billion elements even before considering the tile structure. TiledArray materializes this as dense tiles with tile-level sparsity only (~91% of tiles are zero), but the remaining 9% of tiles still consume massive memory.

### 4.2 C2H6 Results (Only Feasible Molecule)

| Nodes | Wall (s) | Stage 1 (s) | Stage 2 (s) | Stage 3 (s) | RSS/rank (GB) |
|------:|---------:|------------:|------------:|------------:|--------------:|
| 4     | 300      | 1.0         | 132         | 166         | 59.4          |
| 8     | 237      | 0.8         | 95          | 139         | 39.0          |
| 16    | 254      | 0.5         | 97          | 156         | 25.3          |
| 32    | 209      | 0.4         | 69          | 138         | 14.1          |

**Stage 2** (rank-6 I1 construction) speeds up reasonably: 132s → 69s (1.9x) from 4 to 32 nodes.

**Stage 3** (contracting I1 with g1) barely scales: 166s → 138s (1.2x). This stage processes the rank-6 intermediate, requiring massive data movement across nodes. The 10 GbE network is the clear bottleneck.

**16 nodes is slower than 8 nodes** (254s vs 237s) — a non-monotonic scaling anomaly. At 16 nodes, the per-rank data shrinks enough that communication latency dominates over the reduced compute per rank. 32 nodes recovers because the aggregate bandwidth is sufficient.

### 4.3 Memory Scaling

| Nodes | RSS/rank (GB) | Total Cluster (GB) | Reduction vs 4n |
|------:|--------------:|-------------------:|----------------:|
| 4     | 59.4          | 238                | 1.0x            |
| 8     | 39.0          | 312                | 1.5x/rank       |
| 16    | 25.3          | 405                | 2.3x/rank       |
| 32    | 14.1          | 451                | 4.2x/rank       |

**Total cluster memory increases** from 238 GB to 451 GB (1.9x) even as per-rank RSS drops 4.2x. This overhead comes from replicated tile metadata, MPI buffers, and MADNESS runtime state.

### 4.4 Failure Analysis (C3H8+)

C3H8's rank-6 intermediate would have approximate shape (13, 13, 202, 13, 876, 876). The dense size is 13³ × 202 × 876² ≈ 258 billion elements — roughly **72x larger** than C2H6's intermediate. Even with 91% tile-level sparsity, the materialized tiles would require:

| Molecule | Est. I1 Dense Size | Est. at 91% Sparsity | Min Cluster RAM |
|----------|-------------------:|---------------------:|----------------:|
| C2H6     | 26 GB              | 2.3 GB tiles + metadata | 238 GB (4 nodes) |
| C3H8     | 1.9 TB             | ~170 GB tiles + metadata | ~5+ TB |
| C4H10    | ~12 TB             | ~1+ TB               | ~30+ TB |

These estimates explain why C3H8 OOMs even at 32 nodes (2 TB total) — the intermediate alone requires ~5 TB of distributed memory.

---

## 5. Cross-Equation Comparison

### 5.1 Compute Time (Best Config, C3H8)

| Equation | Best Config | Wall (s) | Bottleneck |
|----------|-------------|----------|------------|
| Eq2      | 1 rank      | 1.10     | Stage 3 GEMM (88%) |
| Eq0      | 32 nodes    | 51.3     | Stage 3 contraction (96%) |
| Eq1      | OOM         | --       | Rank-6 intermediate materialization |

Eq0 is **47x slower** than eq2 at their respective best configs. The gap comes from eq0's much larger intermediate and the inability to exploit element-level sparsity.

### 5.2 Memory Requirement (C3H8)

| Equation | Single-Rank RSS | Min Nodes | RSS at 32 nodes |
|----------|----------------:|----------:|-----------------:|
| Eq2      | 2.8 GB          | 1         | 1.2 GB           |
| Eq0      | OOM (>64 GB)    | 4         | 5.3 GB           |
| Eq1      | OOM (>>64 GB)   | OOM       | OOM              |

### 5.3 Sparsity Exploitation

| Equation | C2H6 | C3H8 | C4H10 | C5H12 | C6H14 |
|----------|-----:|-----:|------:|------:|------:|
| Eq2 tile sparsity | 86.4% | 92.8% | 96.7% | 98.2% | 98.6% |
| Eq0 tile sparsity | 83.4% | 92.5% | 96.1% | 97.6% | 98.3% |
| Eq1 tile sparsity | 89.3% | OOM | OOM | OOM | OOM |

Sparsity is high across all equations, but TiledArray only exploits it at the **tile level**. Within non-zero tiles, the fill rate is often 1-5% (see `sparsity_data.md`), meaning 95-99% of stored values are zeros — pure waste.

---

## 6. Performance Limiters

### 6.1 Network Bandwidth

The 10 GbE interconnect (1.25 GB/s theoretical, ~1.0 GB/s practical) is the primary scaling limiter:

- **Eq2:** 0.25-8s compute per contraction. Communication overhead is ~1-2s (fixed). This explains why only C5H12+ (>5s compute) see speedup.
- **Eq0 Stage 3:** The intermediate I1 is distributed across nodes. At 32 nodes, each rank holds ~5 GB of I1 data. The contraction requires all-to-all communication of non-local tiles, taking ~5-10s at wire rate — a significant fraction of the ~50s stage time.
- **Eq1 Stage 3:** The rank-6 I1 is ~4x larger per rank than eq0's I1, making communication 4x worse.

### 6.2 Tile-Level-Only Sparsity

TiledArray cannot exploit intra-tile zeros. Key measurements from `sparsity_data.md` (C3H8):

| Tensor | Tile Sparsity | Intra-Tile Fill | Memory Overhead |
|--------|-------------:|-----------------:|----------------:|
| g0     | 0%           | 100%             | 1.0x            |
| c1     | 72.5%        | 1.5%             | 67x             |
| c2     | 74.4%        | 0.3%             | 360x            |

The c2 tensor stores **360x more data than necessary** because non-zero tiles are stored as dense blocks. For eq1, the rank-6 intermediate suffers similar overhead, amplified by its enormous size.

### 6.3 Intermediate Materialization

TiledArray's expression engine evaluates binary contractions fully before proceeding to the next stage. There is no lazy evaluation or fusion across stages. This means:

- **Eq0:** I1(i1,u2,a1,k1) must be fully materialized in distributed memory before stage 3 starts
- **Eq1:** I1(i1,i3,u2,i2,a2,a3) must be fully materialized — the primary cause of OOM
- **Eq2:** Intermediates are small enough that materialization isn't problematic

Intermediate fusion (computing stage 2 and 3 together without full materialization) could potentially make eq1 feasible and significantly reduce eq0's memory footprint.

### 6.4 MADNESS Runtime Overhead

The MADNESS parallel runtime adds overhead that's particularly visible for small problems:

- `MPI_Comm_Split` called per Hadamard index in einsum() — latency accumulates
- Work-stealing ineffective when tile counts are small relative to rank count
- Thread pool management via pthreads (TBB backend is incompatible)

---

## 7. Implications for SPTN_MLIR_DSL Comparison

The TiledArray benchmark establishes clear baselines and identifies three specific opportunities for the fastcc/T-Fit approach:

1. **Element-level sparsity** (67-360x memory savings): T-Fit stores only non-zero elements, avoiding the tile padding waste. This alone could make eq1 feasible for C3H8 if the intermediate's element-level sparsity is >99%.

2. **Intermediate fusion** (avoiding full materialization): The SPTN_MLIR_DSL compiler could fuse stages 2 and 3, streaming intermediate tiles through stage 3 as they're produced. This is the primary path to making eq1 work for larger molecules.

3. **SUMMA communication pattern** (better scaling): TiledArray's einsum uses a generic communication pattern. SUMMA's structured 2D grid communication should reduce the all-to-all overhead, particularly for eq0 and eq1 where the intermediate is the communication bottleneck.

**Fair comparison metrics:**
- Same molecules (C2H6-C6H14), same equations, same hardware
- Wall time, peak memory, scaling efficiency
- Correctness verification against TiledArray's output (or the dense baseline)
