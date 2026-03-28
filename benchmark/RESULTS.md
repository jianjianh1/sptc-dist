# TiledArray Benchmark Results

All results from C3H8 through C6H14, 3 trials each. Times in seconds (median of 3 trials).
Peak RSS in MB per MPI rank. Tile sizes: occ=4, uocc=50, ri=200 (matching MPQC input.json).
Default sparse threshold (~1.19e-7).

Platform: CloudLab, 4 nodes (node0-3), Intel Xeon D-1548 (8 cores, 64 GB RAM each).
MPICH 4.0, OpenBLAS, GCC 11, TiledArray v1.1.0 (unmodified, Pthreads backend, -fno-lto).

---

## Eq2: R(p,i,a,q,b) = g(i,m,k) * g(p,n,k) * c1(i,m,a) * c2(p,q,n,b)

All intermediates fit in single-node memory. Three staged einsum contractions.

### 1 MPI rank (single node)

| Molecule | Trial 1 | Trial 2 | Trial 3 | Median | Sparsity | Peak RSS (MB) |
|----------|---------|---------|---------|--------|----------|---------------|
| C3H8     | 1.206   | 1.127   | 1.141   | **1.14** | 92.8%  | 2,811  |
| C4H10    | 2.460   | 2.404   | 2.420   | **2.42** | 96.7%  | 5,628  |
| C5H12    | 5.037   | 5.083   | 5.117   | **5.08** | 98.1%  | 10,467 |
| C6H14    | 8.772   | 8.466   | 8.386   | **8.47** | 98.6%  | 16,415 |

### 4 MPI ranks (4 nodes, 1 rank/node)

| Molecule | Trial 1 | Trial 2 | Trial 3 | Median | Sparsity | Peak RSS/rank (MB) |
|----------|---------|---------|---------|--------|----------|--------------------|
| C3H8     | 2.074   | 1.878   | 1.620   | **1.88** | 93.4%  | 1,255  |
| C4H10    | 3.550   | 3.389   | 3.107   | **3.39** | 96.8%  | 2,571  |
| C5H12    | 6.419   | 5.569   | 6.212   | **6.21** | 98.1%  | 4,740  |
| C6H14    | 10.193  | 10.034  | 8.565   | **10.03** | 98.5% | 7,200  |

---

## Eq0: R(i,j,a,b) = g0(m,n,k) * c(i,m,a) * c(j,n,b) * g1(j,i,k)

Large rank-5 intermediate I1(k,i,a,j,b). OOMs on single node. Requires multi-node.

### 4 MPI ranks (4 nodes, 1 rank/node)

| Molecule | Trial 1 | Trial 2 | Trial 3 | Median | I1 Sparsity | Peak RSS/rank (MB) |
|----------|---------|---------|---------|--------|-------------|---------------------|
| C3H8     | 161.8   | 163.8   | 162.9   | **162.9** | 92.5%    | 28,184 |
| C4H10    | 378.6   | 384.8   | 387.9   | **384.8** | 96.1%    | 58,310 |

### 8 MPI ranks (4 nodes, 2 ranks/node) — C3H8 only

| Molecule | Trial 1 | Trial 2 | Trial 3 | Median | Peak RSS/rank (MB) |
|----------|---------|---------|---------|--------|--------------------|
| C3H8     | 192.8   | 187.8   | 184.5   | **187.8** | 16,805 |

Note: 8 ranks is slower than 4 ranks (187.8s vs 162.9s) due to increased MPI communication overhead, but uses less memory per rank (16.8 GB vs 28.2 GB).

C5H12 and C6H14 eq0 were not attempted on 4 nodes — C4H10 already uses 58 GB/rank, so larger molecules would OOM.

---

## Eq1: R(i,p,j,a,b) = g0(m,n,k) * c2(i,j,m,a) * c2(j,p,n,b) * g1(i,p,k)

Rank-6 intermediate I1(j,k,i,a,p,b). OOMs on all tested configurations.

| Config | Result |
|--------|--------|
| 1 rank, 1 node | OOM |
| 4 ranks, 4 nodes | OOM |
| 8 ranks, 4 nodes (2/node) | OOM / timeout |
| 16 ranks, 4 nodes (4/node) | OOM / timeout |

Eq1's rank-6 intermediate is too large for 4 x 64 GB even with sparsity. Would need either significantly more nodes or a fused implementation that avoids materializing I1.

---

## Summary Table (median times, seconds)

| Equation | Molecule | 1 rank | 4 nodes | 8 ranks | Dense baseline |
|----------|----------|--------|---------|---------|----------------|
| **Eq2** | C3H8  | 1.14  | 1.88  | —     | 26.7  |
| **Eq2** | C4H10 | 2.42  | 3.39  | —     | —     |
| **Eq2** | C5H12 | 5.08  | 6.21  | —     | —     |
| **Eq2** | C6H14 | 8.47  | 10.03 | —     | —     |
| **Eq0** | C3H8  | OOM   | 162.9 | 187.8 | 352   |
| **Eq0** | C4H10 | OOM   | 384.8 | —     | —     |
| **Eq1** | C3H8  | OOM   | OOM   | OOM   | 51.4  |

---

## Key Observations

1. **Eq2 scales well with molecule size**: Sparsity increases from 92.8% (C3H8) to 98.6% (C6H14), keeping compute times modest even as tensor dimensions grow significantly.

2. **Multi-node adds overhead for Eq2**: 4-node runs are 1.2-1.6x slower than single-rank for all molecules. The computation is fast enough that MPI communication dominates.

3. **Eq0 requires multi-node for memory**: The rank-5 I1 intermediate uses 28 GB/rank on 4 nodes for C3H8 and 58 GB/rank for C4H10. Single-node OOMs. More ranks reduce per-rank memory but increase runtime.

4. **Eq1 cannot run with TiledArray**: The rank-6 intermediate is too large to materialize even with distribution. This is a fundamental TiledArray limitation (no intermediate fusion).

5. **Memory scales roughly with molecule size**: Eq2 peak RSS grows from 2.8 GB (C3H8) to 16.4 GB (C6H14) on 1 rank. Eq0 grows from 28 GB to 58 GB per rank on 4 nodes.
