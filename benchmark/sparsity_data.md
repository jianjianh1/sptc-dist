# Block-Sparsity Structure Analysis of DLPNO-CCSD Input Tensors

Analysis of all input tensors across molecules C3H8 through C6H14, using MPQC tile sizes:
**occ=4, uocc=50, ri=200**.

C2H6 is excluded (empty dataset directory).

## Tensor Definitions

| Label | File | Rank | Index Types (col order) | Tile Sizes |
|-------|------|------|-------------------------|------------|
| g0 | g\_m\_1\_m\_2\_K\_1.txt | 3 | uocc, uocc, ri | 50, 50, 200 |
| g1 | g\_i\_1\_i\_2\_K\_1.txt | 3 | occ, occ, ri | 4, 4, 200 |
| g  | g\_i\_1\_m\_1\_K\_1.txt | 3 | occ, uocc, ri | 4, 50, 200 |
| c1 | C\_m\_1\_a\_1\_i\_1.txt | 3 | uocc, uocc, occ | 50, 50, 4 |
| c2 | C\_m\_1\_a\_1\_i\_1\_i\_2.txt | 4 | uocc, uocc, occ, occ | 50, 50, 4, 4 |

## Summary Table

| Molecule | Tensor | NNZ | Shape | Tile Grid | NZ Tiles / Total | Tile Sparsity | Intra-Tile Fill | Element Sparsity | TA Mem (MB) | Ideal Mem (MB) | Overhead |
|----------|--------|-----|-------|-----------|-------------------|---------------|-----------------|------------------|-------------|----------------|----------|
| C3H8 | g0 | 19,708,332 | 202x202x483 | 5x5x3 | 75/75 | 0.00% | 52.56% | 0.00% | 286.10 | 150.36 | 1.90x |
| C3H8 | g1 | 48,300 | 13x13x483 | 4x4x3 | 48/48 | 0.00% | 31.45% | 40.83% | 1.17 | 0.37 | 3.18x |
| C3H8 | g | 975,660 | 13x202x483 | 4x5x3 | 60/60 | 0.00% | 40.65% | 23.08% | 18.31 | 7.44 | 2.46x |
| C3H8 | c1 | 159,352 | 13x202x898 | 1x5x225 | 1,060/1,125 | 5.78% | 1.50% | 93.24% | 80.87 | 1.22 | 66.52x |
| C3H8 | c2 | 938,488 | 13x13x202x876 | 1x1x51x219 | 8,454/11,169 | 24.31% | 0.28% | 96.86% | 2,579.96 | 7.16 | 360.32x |
| C4H10 | g0 | 42,182,400 | 260x260x624 | 6x6x4 | 144/144 | 0.00% | 58.59% | 0.00% | 549.32 | 321.83 | 1.71x |
| C4H10 | g1 | 105,456 | 17x17x624 | 5x5x4 | 64/100 | 36.00% | 51.49% | 41.52% | 1.56 | 0.80 | 1.94x |
| C4H10 | g | 2,109,120 | 17x260x624 | 5x6x4 | 96/120 | 20.00% | 54.93% | 23.53% | 29.30 | 16.09 | 1.82x |
| C4H10 | c1 | 239,128 | 17x260x1295 | 1x6x324 | 1,712/1,944 | 11.93% | 1.40% | 95.82% | 130.62 | 1.82 | 71.59x |
| C4H10 | c2 | 1,579,972 | 17x17x260x1262 | 1x1x65x316 | 13,750/20,540 | 33.06% | 0.29% | 98.33% | 4,196.17 | 12.05 | 348.11x |
| C5H12 | g0 | 77,359,860 | 318x318x765 | 7x7x4 | 196/196 | 0.00% | 78.94% | 0.00% | 747.68 | 590.21 | 1.27x |
| C5H12 | g1 | 195,840 | 21x21x765 | 6x6x4 | 100/144 | 30.56% | 61.20% | 41.95% | 2.44 | 1.49 | 1.63x |
| C5H12 | g | 3,892,320 | 21x318x765 | 6x7x4 | 140/168 | 16.67% | 69.51% | 23.81% | 42.72 | 29.70 | 1.44x |
| C5H12 | c1 | 313,486 | 21x318x1688 | 1x7x422 | 2,405/2,954 | 18.58% | 1.30% | 97.22% | 183.49 | 2.39 | 76.72x |
| C5H12 | c2 | 2,234,706 | 21x21x318x1655 | 1x1x80x414 | 19,410/33,120 | 41.39% | 0.29% | 99.04% | 5,923.46 | 17.05 | 347.43x |
| C6H14 | g0 | 127,481,856 | 376x376x906 | 8x8x5 | 320/320 | 0.00% | 79.68% | 0.47% | 1,220.70 | 972.61 | 1.26x |
| C6H14 | g1 | 327,066 | 25x25x906 | 7x7x5 | 180/245 | 26.53% | 56.78% | 42.24% | 4.39 | 2.50 | 1.76x |
| C6H14 | g | 6,472,464 | 25x376x906 | 7x8x5 | 240/280 | 14.29% | 67.42% | 24.00% | 73.24 | 49.38 | 1.48x |
| C6H14 | c1 | 385,430 | 25x376x1839 | 1x8x460 | 3,075/3,680 | 16.44% | 1.25% | 97.77% | 234.60 | 2.94 | 79.78x |
| C6H14 | c2 | 2,882,690 | 25x25x376x1806 | 1x1x94x452 | 25,320/42,488 | 40.41% | 0.28% | 99.32% | 7,727.05 | 21.99 | 351.34x |

## Key Findings

### 1. g0 (uocc x uocc x ri) -- Nearly Dense
- **Tile sparsity: 0% across all molecules** (every tile is nonzero)
- Element sparsity: 0-0.5% (essentially fully dense)
- Intra-tile fill: 53-80%, increasing with molecule size
- Overhead: 1.3-1.9x (modest; tiles are well-filled)
- This is the largest tensor (150-973 MB), dominating memory

### 2. g1 (occ x occ x ri) -- Moderate Tile Sparsity
- Tile sparsity: 0% (C3H8) to 36% (C4H10), averaging ~25-36%
- Element sparsity: ~41-42% (consistent across molecules)
- Intra-tile fill: 31-61%
- Overhead: 1.6-3.2x (small tensor, overhead is negligible in absolute terms)

### 3. g (occ x uocc x ri) -- Mild Tile Sparsity
- Tile sparsity: 0% (C3H8) to 20% (C4H10)
- Element sparsity: ~23-24% (consistent)
- Intra-tile fill: 41-69%
- Overhead: 1.4-2.5x

### 4. c1 (uocc x uocc x occ) -- Highly Sparse Elements, Poor Tile Capture
- Tile sparsity: 6-19% (tiles mostly populated despite extreme element sparsity)
- **Element sparsity: 93-98%** (only 2-7% of elements are nonzero)
- **Intra-tile fill: 1.3-1.5%** (tiles are almost empty inside)
- **Overhead: 67-80x** (TiledArray stores ~70x more data than needed)

### 5. c2 (uocc x uocc x occ x occ) -- Extremely Sparse, Catastrophic Overhead
- Tile sparsity: 24-41% (removing ~1/3 of tiles)
- **Element sparsity: 97-99.3%**
- **Intra-tile fill: 0.28-0.29%** (tiles are 99.7% zeros!)
- **Overhead: 347-360x** (TiledArray stores ~350x more data than needed)
- Absolute waste: 2.6-7.7 GB of TA memory vs 7-22 MB ideal

### Dense Storage Overhead Summary

| Tensor | Overhead Range | Root Cause |
|--------|---------------|------------|
| g0 | 1.3-1.9x | Nearly dense; tile boundaries cause minor padding |
| g1 | 1.6-3.2x | Small tensor; moderate sparsity within tiles |
| g | 1.4-2.5x | Moderate sparsity; reasonable tile fill |
| c1 | **67-80x** | Extremely sparse elements scattered across most tiles |
| c2 | **347-360x** | Ultra-sparse rank-4 tensor; tiny occ tiles (4x4) give huge tile count but each tile is 99.7% zero |

### Implications for TiledArray Performance

1. **g0 is the bottleneck tensor by size** but is well-suited to block-sparse storage (low overhead).
2. **c1 and c2 suffer catastrophically** from TiledArray's tile-level-only sparsity. The coefficient tensors have fine-grained sparsity that tile blocking cannot capture.
3. The occ dimension (size 4) creates degenerate tiles -- a tile size of 4 means every occupied orbital gets its own tile boundary, but the uocc dimensions (tile=50) are too coarse for the sparse structure.
4. **Total wasted memory** (TA minus ideal) grows from ~2.7 GB (C3H8) to ~8.0 GB (C6H14), dominated by c2.
5. A CSR/COO format or finer-grained blocking for c1/c2 would reduce memory by 1-2 orders of magnitude.
