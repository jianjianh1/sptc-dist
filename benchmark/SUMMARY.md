# TiledArray Benchmark Summary — 32-Node CloudLab Cluster

**Hardware:** 32x Intel Xeon D-1548 (8c/16t @ 2.0 GHz), 64 GB RAM, 10 GbE
**Molecules:** C2H6, C3H8, C4H10, C5H12, C6H14 (alkane series)
**Config:** 1 MPI rank per node, 3 trials each, OPENBLAS_NUM_THREADS=1

## Results at a Glance

### Eq2 — Best-case scenario (small intermediates)

| Molecule | 1 rank (s) | 4 nodes | 8 nodes | 16 nodes | 32 nodes |
|----------|----------:|--------:|--------:|---------:|---------:|
| C2H6     | 0.25     | 0.76    | 0.63    | 0.70     | 0.80     |
| C3H8     | 1.12     | 1.73    | 1.61    | 1.50     | 1.50     |
| C4H10    | 2.33     | 3.02    | 2.65    | 2.05     | 1.78     |
| C5H12    | 5.00     | 5.22    | 4.21    | 3.33     | 3.22     |
| C6H14    | 8.32     | 9.13    | 7.68    | 5.19     | 4.24     |

Distribution only helps for larger molecules. C6H14 achieves 2.0x speedup at 32 nodes. Smaller molecules are fastest on a single rank.

### Eq0 — Memory-driven distribution (large intermediates)

| Molecule | 1 rank (s) | 4 nodes | 8 nodes | 16 nodes | 32 nodes |
|----------|----------:|--------:|--------:|---------:|---------:|
| C2H6     | 59.7     | 129.4   | 67.0    | 45.4     | 40.7     |
| C3H8     | OOM      | 164.8   | 109.9   | 65.5     | 51.3     |
| C4H10    | OOM      | OOM     | 348.2   | 209.5    | 96.0     |
| C5H12    | OOM      | OOM     | 782.0   | 480.6    | 270.6    |
| C6H14    | OOM      | OOM     | OOM     | 753.3    | 477.0    |

Eq0 needs distribution primarily for memory. C3H8+ cannot fit on a single node. Scaling is reasonable: C4H10 gets 3.6x from 8→32 nodes.

### Eq1 — Infeasible for most molecules (rank-6 intermediate)

| Molecule | 4 nodes | 8 nodes | 16 nodes | 32 nodes |
|----------|--------:|--------:|---------:|---------:|
| C2H6     | 308.5   | 237.2   | 254.3    | 215.8    |
| C3H8     | OOM     | OOM     | OOM      | OOM      |
| C4H10+   | OOM     | OOM     | OOM      | OOM      |

Only C2H6 completes. The rank-6 intermediate grows ~14x per carbon increment, making C3H8 infeasible even with 2 TB total cluster RAM.

## Why Distribution Hurts Small Problems

Single-node 1-rank is fastest for small problems because:
- TiledArray's einsum triggers MPI_Comm_Split per Hadamard tile
- Sparse tile distribution leaves most ranks idle
- 10 GbE latency dominates sub-second compute

Multi-rank on a single node is always counterproductive (up to 8x slower).

## Memory Per Rank (GB) — 1 rank per node

| Molecule | Eq2 (1 rank) | Eq0 (min nodes) | Eq1 (4 nodes) |
|----------|-------------:|-----------------:|---------------:|
| C2H6     | 1.0          | 27.2 (1 node)    | 59.4           |
| C3H8     | 2.8          | 27.1 (4 nodes)   | OOM            |
| C4H10    | 5.5          | 31.6 (8 nodes)   | OOM            |
| C5H12    | 10.2         | 58.0 (8 nodes)   | OOM            |
| C6H14    | 15.6         | 46.9 (16 nodes)  | OOM            |

## Key Takeaways

1. **Eq2 works well** — tile-level sparsity (86-99%) makes it fast, but distribution only pays off for C5H12+ on this network
2. **Eq0 needs distribution for memory** — intermediates too large for single node; decent 2-4x scaling across 8→32 nodes
3. **Eq1 is fundamentally broken** — rank-6 intermediate defeats tile-level-only sparsity; only C2H6 fits in 2 TB cluster
4. **1 rank per node is always optimal** — intra-node MPI overhead is severe (1.7-1.9x worse with 2ppn)
5. **10 GbE is the scaling bottleneck** — communication dominates for small problems and limits strong scaling
6. **Element-level sparsity (fastcc/T-Fit) could help** — TiledArray wastes 67-360x memory on intra-tile zeros; avoiding this could make eq1 feasible and improve eq0 scaling
